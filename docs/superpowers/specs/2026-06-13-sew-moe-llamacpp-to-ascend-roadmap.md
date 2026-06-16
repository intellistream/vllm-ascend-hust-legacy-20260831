# SEW-MoE llama.cpp To Ascend Roadmap

## Purpose

This note maps the useful parts of llama.cpp's small-VRAM MoE execution into a concrete vLLM Ascend research path. It is not a request to copy llama.cpp's runtime. The goal is to translate the same sparse-expert principle into Ascend-friendly grouped execution, fixed HBM slots, and profile-driven kernel choices.

## What llama.cpp Actually Does

llama.cpp's MoE advantage is not one trick. It is a stack of small, compatible choices:

1. **Tensor placement override for MoE expert weights.**
   `--cpu-moe` keeps all MoE expert tensors on CPU. `--n-cpu-moe N` keeps the first N MoE layers' expert tensors on CPU by adding tensor buffer overrides for the FFN expert tensor regex. This is a placement policy, not a new model algorithm.

2. **MoE weights are represented as `MUL_MAT_ID` tensors.**
   Expert FFN tensors map to `GGML_OP_MUL_MAT_ID`. Qwen3MoE creates expert tensors shaped by expert dimension and builds graph nodes with `ggml_mul_mat_id(ctx, w, cur, ids)`, where `ids` contains selected experts.

3. **The backend scheduler copies only active expert slices.**
   In `ggml_backend_sched`, when a split backend needs host-resident MoE weights for a `MUL_MAT_ID` node, it reads the `ids` tensor, builds a bitset of used expert IDs, groups consecutive IDs, and copies only those expert slices into the backend copy tensor. Full expert tensors do not have to move every step.

4. **The CPU `MUL_MAT_ID` kernel groups rows by expert.**
   The CPU implementation builds per-expert row counts and row mappings from `ids`, skips experts with zero rows, and computes only selected expert groups.

5. **Some backends fuse MoE-adjacent ops.**
   Vulkan contains fusion checks for `MUL_MAT_ID + ADD_ID`, `MUL_MAT_ID + MUL`, `MUL_MAT_ID + ADD_ID + MUL`, and top-k MoE subgraphs. This matters because decode performance is often dominated by high-frequency small operations around the expert GEMM.

The important abstraction is therefore:

```text
selected expert IDs -> active expert working set -> staged/cached expert data -> grouped expert compute -> fused adjacent ops
```

## What Should Transfer To vLLM Ascend

### Transfer Directly

- Treat active experts as a first-class runtime signal.
- Preserve two views of the same MoE invocation:
  - logical selected experts from `topk_ids`;
  - physical grouped shape from `group_list`.
- Use observed active expert locality to decide slot budgets and residency.
- Avoid moving or preparing inactive expert weights in offload mode.
- Look for fusion immediately around grouped expert compute.

### Do Not Transfer Directly

- Do not copy ggml's graph scheduler architecture into vLLM.
- Do not convert vLLM Ascend's grouped MoE into one-expert CPU-style loops.
- Do not use generic dynamic expert cache semantics when Ascend benefits from fixed addresses and stable layouts.
- Do not chase host-to-HBM overlap before measuring whether transfer or compute dominates the current run.

## Current vLLM Ascend Evidence

The existing Qwen3-30B-A3B non-offload profile shows `GroupedMatmul` as the dominant cost:

- mixed: 62.6% of operator time;
- prefill: 59.0%;
- decode: 55.9%.

The same report shows routing/reorder and high-frequency small kernels as secondary decode costs. `GroupedMatmul` cube utilization is already high, so the likely non-offload win is not "make cube busier" in the abstract. It is:

- reduce repeated grouped matmul setup and shape handling;
- stabilize dominant grouped shapes;
- fuse routing/activation/combine work that currently surrounds GMM;
- specialize decode paths where the grouped signature repeats.

For offload, prior SEW-Offload evidence showed synchronous host-to-HBM transfer can dominate Stage T. That means hit-first phase overlap is only valuable after slot budget, resident layers, or async prefetch reduce exposed transfer enough for R/C/M to matter.

## Current P0 Foundation

P0 Active Expert Observability now makes the llama.cpp principle measurable in vLLM Ascend:

- logical events: `source="logical_topk"`;
- grouped events: `source="grouped_dispatch"`;
- shared `layer_id` and `step_id`;
- active expert set and per-expert token counts;
- grouped `group_list_signature`;
- analyzer summary in `benchmarks/scripts/analyze_ascend_moe_profile.py`.

This is the bridge from "llama.cpp seems fast" to "which Ascend optimization should be built next."

## Decision Gates For P1

Run Qwen3-30B-A3B with P0 trace and pipeline profiling, then choose one P1.

### P1-C: SEW-Compute GroupedMatmul Fast Path

Choose this if:

- Stage C dominates T/R/M in non-offload;
- a small number of `group_list_signature` values account for most decode invocations;
- grouped token counts are stable enough to bucket;
- routing/combine are secondary.

Implementation direction:

- add a decode-only grouped signature classifier;
- route dominant signatures to a stable-shape GMM path;
- reuse preallocated workspaces for the dominant bucket;
- keep fallback to existing grouped matmul for rare signatures.

### P1-RM: Routing/Dispatch/Combine Fusion

Choose this if:

- Stage R + M is a large fraction of decode;
- `MoeGatingTopK`, `MoeInitRoutingCustom`, `MoeTokenUnpermute`, `Sort`, or small vector ops dominate after GMM;
- `group_list_signature` is unstable, making GMM specialization premature.

Implementation direction:

- reduce sort-based routing where count/prefix-sum can generate expert offsets;
- produce grouped-matmul-ready metadata directly;
- fuse unpermute/combine or combine adjacent vector work;
- reuse decode workspaces across steps.

### P1-T: SEW-Offload Slot Simulation And Residency

Choose this if:

- Stage T dominates in offload runs;
- observed active expert fanout exceeds current slot budget often;
- repeated active expert locality suggests resident layers or larger slots reduce misses;
- R/C/M are too small to hide transfer.

Implementation direction:

- feed P0 trace into slot simulation;
- sweep `num_slots`, resident layer sets, and fanout thresholds;
- choose a fixed slot policy that minimizes miss bytes before adding overlap;
- only then add async host-to-HBM transfer.

### P1-H: Hit-First Phased Execution

Choose this only if:

- P1-T reduces exposed transfer to the same order as R/C/M;
- many records have mixed slot hits and misses;
- phase split overhead is lower than resident compute time.

Implementation direction:

- split active experts into ready and miss phases;
- compute ready experts first;
- wait for miss events and compute second phase;
- keep fail-closed semantics and one-phase fallback.

## Recommended Next Experiment

Use the current P0 trace path on the existing Qwen3-30B-A3B profile workload:

```bash
VLLM_ASCEND_MOE_OFFLOAD_ENABLED=1 \
VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY=1 \
VLLM_ASCEND_MOE_PIPELINE_PROFILING=1 \
VLLM_ASCEND_MOE_OFFLOAD_TRACE_PATH=/tmp/sew_moe_trace.jsonl \
VLLM_ASCEND_MOE_OFFLOAD_PROFILE_PATH=/tmp/sew_moe_profile.jsonl \
python benchmarks/scripts/run_ascend_moe_profile_suite.py ...
```

Then run the analyzer against the generated profiler directory and trace JSONL. The expected decision output is not "optimize everything." It should name one next target:

- `C` if grouped compute shape repeats and dominates;
- `R/M` if routing/combine overhead is large or shape instability blocks C;
- `T` if offload transfer dominates;
- `H` only after transfer is close enough to compute for overlap to matter.

## Engineering Rule

Do not implement P1 before a trace-backed decision. llama.cpp wins by exploiting sparse active experts. On Ascend, the equivalent win must be:

```text
active expert trace -> fixed shape/slot decision -> narrow optimized path -> measured throughput delta
```

Any P1 that does not consume P0 trace data risks optimizing the wrong bottleneck.
