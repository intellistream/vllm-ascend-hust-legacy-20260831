# ACLGraph-Compatible MoE Offload: Decision-Hoisting Feasibility Analysis

> Date: 2026-06-15 | Driven by skill: `model-infer-graph-mode` (+ `npu-arch` context)
> Status: design foundation for paper track 2. No NPU runs; conclusions are code-grounded.

## 0. Purpose

Establish whether the data-dependent MoE offload decision can be made compatible
with Ascend ACLGraph capture/replay, instead of being forced onto the eager path.
This is the linchpin of the paper's core contribution: reconciling dynamic MoE
expert offload with Ascend's static graph execution.

## 1. Verified Capture Boundary (from code, not assumption)

- vLLM-Ascend uses **piecewise ACLGraph** (`cudagraph_mode: PIECEWISE`), wrapper in
  `vllm_ascend/compilation/acl_graph.py` (`ACLGraphWrapper`).
- `splitting_ops` (from engine config) = **attention ops only**
  (`vllm::unified_attention_with_output`, MLA variants, mamba, etc.). MoE is **not**
  a splitting op.
- Therefore in decode the captured graph segments contain the **entire MoE layer**:
  `gate → select_experts (topk_ids) → fused_experts (dispatch → grouped_matmul →
  combine)`. Attention runs eager between captured segments.
- Confirming signal: the non-eager engine config lists `static_all_moe_layers =
  [model.layers.0.mlp.experts ... model.layers.47.mlp.experts]` (all 48); under
  `--enforce-eager` this list is empty. vLLM-Ascend **expects MoE experts to be
  static during capture** — directly at odds with dynamic offload.

## 2. The Exact Blocking Code Path

`vllm_ascend/ops/fused_moe/moe_comm_method.py:327` inside `_maybe_apply_moe_offload_plan`,
which runs inside the captured MoE region:

```python
active_experts = tuple(
    int(e) for e in torch.unique(fused_experts_input.topk_ids.detach().cpu()).tolist()
)                                   # (1) device->host sync, (2) value-dependent
decision = runtime.decide_layered_path(active_experts=...)   # (3) Python control flow
prepared_weights = runtime.prepare_fixed_slot_plan(active_experts=...)  # (4) conditional H2D
```

This trips three ACLGraph hard constraints simultaneously (per `model-infer-graph-mode`
"将动态变化的东西提取为模型输入,内部保持静态; Python 控制流 → Graph Break"):

| Line | Operation | ACLGraph violation |
|---|---|---|
| (1) | `.detach().cpu()` | mid-graph device→host sync — not recordable |
| (2) | `torch.unique(...).tolist()` | result is **data-dependent** (depends on this step's topk values), pulled to Python |
| (3) | `decide_layered_path` if/else | **Python control flow** branching on tensor values → graph break |
| (4) | `prepare_fixed_slot_plan` | **conditional, data-dependent H2D copy** (load miss experts) |

Note: the existing capture guards at `runtime.py:196,222` only protect the
**trace/classify (observability)** paths. The **execution** path above is unguarded —
it is the true reason offload needs `--enforce-eager`.

## 3. Existing Precedent: Attention Already Does Control/Data Decoupling

`acl_graph.py` already implements exactly the pattern we need, for attention:

- `update_full_graph_params` / `update_attn_params` (line 225+): the host computes
  attention metadata (seq lengths, block tables) and writes it into **fixed graph
  buffers** before replay.
- Explicit event ordering (`acl_graph.py:201-207`): a CPU `record_event` for
  iteration *i* is synchronized so the update only runs after replay of *i-1*
  completes. `GraphParams.attn_params` holds the per-graph param tuples.

So the machinery to feed a host-computed "decision" into a captured graph as **data**
(not control flow), with correct ordering, already exists. It is currently
attention-only; there is no MoE equivalent.

## 4. The Hard Problem: MoE Routing Timing

The attention precedent works because attn metadata is known **before** the forward
pass (from the scheduler). MoE routing is fundamentally different:

- `topk_ids` is produced by the **gate inside the forward**, depending on hidden
  states that flow through the captured graph.
- The offload decision for layer *N* cannot be known until layer *N*'s gate runs.
- Therefore we **cannot** simply hoist the decision to "before replay" the way attn
  params are hoisted. The decision input does not exist until mid-graph.

This is the crux that makes the problem non-trivial (and paper-worthy): the decision
is both **data-dependent** and **produced inside the captured region**.

## 5. Design Options for Decision Hoisting

The decision has two separable sub-problems. Treating them separately is the key:

- **D — decision compute**: which experts are active, expert→slot map (`log2phy`),
  physical `group_list`. Pure indexing math.
- **S — weight staging**: the actual host→HBM copy of miss experts into slots. A
  data-dependent DMA.

### Option 1 — On-device decision, masked/unconditional staging (fully in-graph)

Express D as **device tensor ops** (no `.cpu()`, no `unique`, no Python `if`):
`log2phy` via scatter, `group_list` via bincount/cumsum on `topk_ids`. Express S as
a **fixed-shape, unconditional** gather from a host-pinned expert buffer indexed by a
device tensor (always "copy N slots", mask handles hits). Then the whole MoE stays
captured.

- ✅ Maximum ACLGraph coverage; no split.
- ❌ Data-dependent H2D as a static device op is the hard part: Ascend has no clean
  "DMA expert E→slot S where E is a device-tensor value, only if miss" primitive.
  Masked-unconditional staging wastes bandwidth (copies even on hit) — and bandwidth
  is exactly our bottleneck. Likely a net loss.

### Option 2 — Split MoE at the routing point (piecewise extension) ⭐ recommended

Extend the existing piecewise model. Today attention is a splitting op; add the
**offload staging** as a splitting boundary so one MoE layer becomes:

```
[captured segment]  ... → gate → select_experts            (produces topk_ids)
[eager split]       offload staging: decision (D) + H2D into FIXED slots (S),
                    write log2phy / group_list into FIXED buffers
[captured segment]  grouped_matmul on FIXED slots → swiglu → combine → ...
```

- The big compute (grouped_matmul + combine) stays **captured** with **stable slot
  addresses** (this is what fixed-slot was always for) → keeps the bulk of ACLGraph
  launch savings.
- Only the **tiny staging op** runs eager. `topk_ids → group_list/log2phy` are fed as
  **runtime inputs into fixed buffers**, reusing the `update_*_params` precedent.
- Mirrors how attention is already handled — architecturally consistent, lowest risk.
- ✅ Dynamic decision allowed (eager split), bulk stays captured.
- ⚠️ Need to register the MoE split boundary as a custom op (none exists today) and
  ensure the captured grouped-matmul segment only ever sees fixed slot tensors +
  fixed-shape `group_list` buffer (pad to capacity).

### Option 3 — Decouple decision from execution across steps (predict-ahead)

Use step *i-1*'s routing to **prefetch** likely experts for step *i* before replay,
so staging is hoisted out like attn params. Decode temporal locality (logs: decode is
mostly top-8, layer-stable) may make this hit often.

- ✅ Fully reuses the attn-param hoisting precedent; staging truly before replay.
- ❌ Prediction misses must fall back (still need a safe in-step path); correctness
  requires a miss path that does not break the graph. Best as a **second-phase
  optimization on top of Option 2**, not the foundation.

## 6. Feasibility Verdict

**Decision hoisting is feasible, via Option 2 (split MoE at routing, like attention).**
The architectural precedent (attn param update + event-ordered fixed buffers) already
exists; the fixed-slot mechanism already gives stable addresses. The missing pieces are
concrete and bounded:

1. A registered MoE split boundary (custom op) so grouped-matmul/combine is a separate
   captured segment from gate/routing.
2. A fixed-shape `group_list` / `log2phy` buffer fed via the `update_*_params` path.
3. Moving the host decision + H2D into the eager split (out of the captured segment).
4. Async staging (MVP-E) layered on top so the eager split's H2D overlaps compute.

This converts the all-or-nothing "offload ⇒ enforce_eager" into "offload keeps ACLGraph
for the dominant compute, eager only for a tiny staging op" — which is the paper's
core systems contribution: **control/data-plane decoupling for graph-compatible MoE
offload on Ascend**.

## 7. Verification Plan (next, NPU-gated)

1. **Baselines (decisive 2×2)** to quantify the ACLGraph penalty offload currently pays:
   - no-offload + ACLGraph (upper bound) / no-offload + eager / offload + eager (current) /
     offload + ACLGraph-attempt (capture the exact break — the experiment we deferred).
   Each: token-id correctness + TTFT/TPOT, separate processes.
2. **Capture-boundary probe**: confirm via logs that the MoE region is captured and
   where the break is raised when offload runs non-eager.
3. **Option 2 prototype**: register MoE split boundary; verify grouped-matmul segment
   captures with fixed slots; token-id parity vs eager offload.
4. **Async staging (MVP-E)**: add load stream/event; measure exposed-stall reduction.

## 8. Risks / Threats to Validity

- **Recompile on dynamic `group_list`**: fixed-shape (pad-to-capacity) buffer needed,
  else per-step recompile (`model-infer-graph-mode` recompile section). No token drop —
  padding only.
- **Split overhead**: extra split point adds launch/sync cost; must show net win vs the
  ACLGraph launch savings recovered. If MoE compute per layer is large enough (it is —
  grouped_matmul dominates), the split cost amortizes.
- **Memory**: capturing grouped-matmul on fixed slots needs the slot bank present at
  capture; per-layer slot bank HBM cost (logs: slots=64≈42GB) tensions with offload
  goal — combine with quantization to halve slot bytes.
- **Correctness barrier**: event ordering between eager staging and captured replay must
  match the attn-param precedent (`acl_graph.py:201-207`) exactly, or stale slots.

<!-- EVIDENCE_PLACEHOLDER -->

## 9. Verified Evidence (NPU, 2026-06-15, Qwen3-30B-A3B, NPU 4)

The 2×2 capture matrix was run with `tools/sew_offload/run_fixed_slot_smoke.py`.
Artifacts under `benchmarks/results/aclgraph_2x2_20260615/`.

| | ACLGraph (`--no-enforce-eager`) | eager |
|---|---|---|
| **no-offload** | ✅ **captures + runs** — "Graph capturing finished", PIECEWISE, 48 layers, TTFT 298ms (1-tok; 8-tok OOM at 64GB edge, 56.9GB resident) | ✅ known-good (slower, loses launch savings) |
| **offload** | ❌ **fails during `capture_model()`** (see below) | ✅ only working path today (validated service) |

**The decisive break (offload + ACLGraph), root cause verbatim:**

```
Not allow to synchronize captured-stream, stream_id=1749.   [rt result 107027]
rtMemcpy execution failed, reason=the current capture mode does not support
  this operation.   [rt result 107030]
synchronized memcpy failed, kind = 2
→ during NPUModelRunner.capture_model() (model_runner_v1.py:3893)
```

`kind=2` is a device↔host memcpy — i.e. the `topk_ids.detach().cpu()` D2H sync in
`_maybe_apply_moe_offload_plan` (§2). Ascend **hard-forbids** synchronize and
synchronized memcpy on a captured stream. This is not a soft graph-break / recompile;
it is a runtime prohibition, proving `--enforce-eager` is a **hard architectural
constraint** of the current offload design, not a configuration choice.

This is exactly the §4 thesis confirmed empirically: the data-dependent offload decision
performs a host sync **inside** the captured region, which capture mode forbids. The §5
Option 2 fix (move the D2H decision + staging into an eager split, keep grouped-matmul on
fixed slots captured) directly removes this forbidden op from the captured stream.

**Still to measure (quantitative motivation table):** no-offload+eager and offload+eager
TPOT for the launch-savings delta; constrained by the no-offload+ACLGraph multi-token OOM
at 56.9GB resident (combine with quantization, or use the 122B stress model on multi-card,
to get clean multi-token decode numbers).

### 9.1 Quantitative 2×2 (NPU 4/5/7, 2026-06-15, inline 'Hello')

| | ACLGraph (`--no-enforce-eager`) | eager (`--enforce-eager`) |
|---|---|---|
| **no-offload** | ✅ TTFT **298 ms** (1-tok; 8-tok OOM) | ✅ TTFT **797 ms**, TPOT 200 ms (8-tok) |
| **offload** | ❌ capture hard-fails (§9) | ✅ only working path: TTFT 36 s, TPOT 5.4 s* (8-tok) |

- **ACLGraph launch-savings (no-offload, same prefill): TTFT 298 vs 797 ms = 2.67× faster.**
  This is the decode/launch benefit offload forfeits by being forced to eager.
- *offload+eager number is a worst case: this run set `--layered-runtime` without
  `--resident-layer-ids`, so all 48 layers took the synchronous slot-load path. The
  validated service (`--ascend-moe-offload-gb 14`) auto-sets resident layers and is
  faster; the exact figure is config-sensitive and not the core claim. The core claims
  (ACLGraph helps 2.67×; offload+ACLGraph is impossible) are config-independent.

## 10. Option 2 Prototype — Decoupling Primitives (implemented, CPU-tested)

First slice of Option 2 landed on `research` (default off, zero behavior change):

- `MoeOffloadConfig.graph_compatible_offload` + env
  `VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE` (default 0).
- `runtime.py`: persistent per-layer **fixed-address `log2phy` buffer** allocated at
  register time (`_log2phy_buffers`); `stage_fixed_slot_plan()` (eager pre-replay:
  decision + H2D + in-place buffer write, refuses to run during capture);
  `capture_safe_slot_weights()` (capture path: fixed slot tensors + fixed log2phy
  buffer, zero host sync / zero conditional H2D); `log2phy_buffer()` accessor.
- `moe_comm_method.py`: capture guard in `_maybe_apply_moe_offload_plan` routes to the
  capture-safe path when `graph_compatible_offload and _is_current_graph_capturing()`,
  bypassing the forbidden `torch.unique(...).cpu()`; shared `_with_prepared_slot_weights`.
- Tests: `tests/ut/moe_offload/test_graph_compatible_offload.py` (6 passed) — stable
  buffer address across staging calls, in-place update, capture-safe no-sync, stage
  refuses during capture. B-relevant suite: 47 passed.

**Not yet done (next):** model_runner lifecycle hook to call `stage_fixed_slot_plan`
before each replay (analogous to `update_attn_params`); register a MoE split boundary
so grouped-matmul is its own captured segment; pad-to-capacity fixed-shape `group_list`
to avoid recompile. **The NPU claim — offload captures under ACLGraph once the D2H
decision is hoisted — is therefore designed and unit-grounded but not yet hardware-verified.**

## 11. Experiment A — capture-pass VERIFIED on real NPU (2026-06-16)

Run on NPU 5, Qwen3-30B-A3B, offload 14GB → 8 slots / 128 experts, ACLGraph
PIECEWISE. Single controlled variable: `VLLM_ASCEND_MOE_OFFLOAD_GRAPH_COMPATIBLE`.

| Config | capture phase | evidence |
|---|---|---|
| offload + ACLGraph, **flag=0** | **CRASH** `107027 synchronize stream failed` + `107030 synchronized memcpy failed, kind=2` (`aclrtMemcpy`, D2H) inside `capture_model` (model_runner_v1.py:3893) | `logs/C_control_offloadON_flag0.log` |
| offload + ACLGraph, **flag=1** | **PASS** `Graph capturing finished in 10 secs, took 0.03 GiB`, no 107027/107030 | `logs/D_test_offloadON_flag1.log` |

Flipping only the flag turns offload+ACLGraph capture from hard crash to pass. The
core paper claim of §9 (control-plane D2H decision is what breaks capture, and hoisting
it via the capture-safe path unblocks ACLGraph) is now **empirically confirmed on
hardware**, not merely code-inferred.

**Milestone 2 boundary, exposed by the flag=1 run's generate phase:** after capture
passes and LOAD_OK, generation raises `Expected all tensors to be on the same device,
got weight is on cpu ... wrapper__npu_grouped_matmul`. Offload keeps expert weights on
the host; the capture-safe path points the captured graph at fixed slots, but with **no
staging hook** to load experts into NPU slots / write log2phy, replay's grouped-matmul
reads host weights. So **Milestone 1 (capture passes) = verified; Milestone 2 (replay
correctness) = needs the model_runner `stage_fixed_slot_plan` hook before replay**
(token-id parity is strict only in the full-residency num_slots≥128 degenerate config;
partial residency needs the full Option-2 split boundary).

Harness: `tools/sew_offload/run_graph_compat_capture_probe.py` (single-config probe) +
`tools/sew_offload/race_launch.sh` (claim-on-free retry for the contended shared host).

