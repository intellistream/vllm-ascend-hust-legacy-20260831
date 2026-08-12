# Fused Dispatch-FFN-Combine: Design and Acceptance Contract

## Status and purpose

This document is the reviewer entry point for the fused
`dispatch_ffn_combine` operator family. It records the motivation, shared A2/A3
design boundary, intended runtime contract, and the evidence required to accept
a change. It does not replace the tests or their immutable run artifacts.

The A2 enablement is under review in the following stacked changes:

- BF16 parent: [vLLM-HUST/vllm-ascend-hust#199](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/199)
- dynamic-W8A8 child: [vLLM-HUST/vllm-ascend-hust#203](https://github.com/vLLM-HUST/vllm-ascend-hust/pull/203)
- A2 capacity and mixed-workload validation: [#215](https://github.com/vLLM-HUST/vllm-ascend-hust/issues/215)
- A3 shared-kernel regression acceptance: [#224](https://github.com/vLLM-HUST/vllm-ascend-hust/issues/224)

The contract below is deliberately stricter than “the operator builds” or “a
model request completed.” A matrix row is accepted only when the implementation,
test oracle, rebuilt artifact, and tested Git revision agree.

## Motivation

In production inference work, overlapping communication with computation was
observed to reduce serialized waiting. Conventional MoE execution exposes
separate dispatch, expert-compute, and combine phases, and the resulting latency
made it natural to ask whether the same overlap could be applied to MoE.

The work did not begin with a preferred dtype. BF16 was the first practical
vehicle because the repository already contained an Ascend 910_93 implementation
that pipelines each expert's GMM2 result into the AIV combine path. Enabling that
path on Ascend 910B/910B2 exposed an A2/A3 HCCL-context difference: the A3 code
interpreted `GetHcclContext()` through an A3-specific structure, whereas A2
provides embedded communication windows through a different context layout.

The later dynamic-W8A8 work was motivated by deploying DeepSeek-V4-Flash on
Ascend. It mirrors the same communication/computation pipeline while retaining
its dtype-specific activation quantization, INT8 GMM, scale, and publication
semantics. It is therefore a child of the common A2 enablement, not an
independent claim that one dtype is universally faster than another.

## Why the implementation is shared

The A2 work extends existing operators rather than copying the complete kernels
into A2-only source files. Most of the pipeline is common: route interpretation,
expert scheduling, GMM/SwiGLU/GMM execution, output publication, combine, and
Torch binding semantics. Duplicating those kernels would create two large
implementations whose fixes and numerical behavior could drift.

Sharing source is acceptable only if the architecture boundary remains explicit:

- build and operator registration select the supported SoC variants;
- HCCL initialization and peer-window lookup use a public abstraction or an
  explicitly checked common layout rather than an unchecked private cast;
- SoC-specific resource limits are validated before launch;
- a shared kernel change is tested at runtime on every affected SoC; and
- a compile-only A3 result is not treated as an A3 regression receipt.

An A2 enablement patch must not silently redefine the existing A3 protocol. A
shared protocol optimization should either carry exact A2 and A3 reuse evidence
in the same change or be split from the enablement.

## Pipeline and ownership

For one local input token row, top-k routing creates routed rows that may target
experts on any EP rank. The fused implementation owns the following lifetime:

1. interpret the active mask and expert IDs without modifying caller-owned
   routing inputs;
2. dispatch the active routed rows into rank-owned HCCL window regions;
3. execute expert-local GMM1, SwiGLU, and GMM2 work;
4. publish completed expert output and token counts with an unambiguous
   generation/lifetime protocol;
5. combine returned rows with the supplied top-k probabilities; and
6. publish `out` and `expert_token_nums` before the call completes.

The caller owns `x`, `expert_idx`, weights, scales, biases, probabilities, and
the optional active mask. The schema marks only `out` and
`expert_token_nums` mutable. Scratch routing IDs, count-generation state, and
workspace aliases are implementation-owned and must not escape their declared
lifetime.

## Operator contract card

### Inputs and outputs

The Torch schema is:

```text
dispatch_ffn_combine(
    Tensor x,
    Tensor[] weight1,
    Tensor[] weight2,
    Tensor expert_idx,
    Tensor[] scale1,
    Tensor[] scale2,
    Tensor[] bias1,
    Tensor[] bias2,
    Tensor probs,
    str group,
    int max_output_size,
    Tensor! out,
    Tensor! expert_token_nums,
    Tensor? x_active_mask=None,
    float swiglu_limit=1000000.0,
) -> (Tensor out, Tensor expert_token_nums)
```

The accepted implementation must validate, at the earliest layer that owns the
information:

- `x` is rank 2 and its hidden dimension matches the first-layer weights;
- `expert_idx` and `probs` describe the same token and top-k domain;
- both weight lists describe the same local-expert set and compatible hidden/
  intermediate dimensions;
- scale and bias lists match the selected dtype/format contract;
- `x_active_mask`, when present, is one-dimensional, device-resident with the
  invocation, and has one entry per input row;
- `out` has the caller-visible token and hidden shape;
- `expert_token_nums` has one entry per local physical expert and the declared
  integer dtype; and
- all descriptor widths and derived byte sizes fit the types consumed by host
  tiling and device code.

### Supported variants

Acceptance is variant-specific. A passing cell for one row does not imply the
other rows.

| SoC | dtype/quantization | role |
| --- | --- | --- |
| Ascend 910_93 (A3) | BF16/FP16 variants already registered by the BF16 operator | existing path; runtime regression target |
| Ascend 910_93 (A3) | dynamic-W8A8 and any shared generic variant | existing path; runtime regression target |
| Ascend 910B/910B2 (A2) | BF16 | proposed explicit-opt-in capability in #199 |
| Ascend 910B/910B2 (A2) | dynamic-W8A8 | proposed explicit-opt-in capability in child #203 |

The A2 unquantized selector must test the model dtype it claims to support. If
the accepted capability is BF16-only, FP16 must not enter it accidentally. Draft
or speculative-model paths must also be excluded unless their graph, routing,
and lifetime contracts are independently demonstrated.

### Capacity and failure policy

Let:

- `M` be the live global token count for a forward;
- `C` be the scheduler token limit declared at startup;
- `TP` and `EP` be tensor- and expert-parallel world sizes;
- `K` be top-k; and
- `L = ceil(C / TP)` be the local token capacity before the documented minimum
  physical padding.

For an accepted A2 fused configuration:

- the communication family is selected at startup and remains identical on all
  participating ranks for every legal forward;
- all `1 <= M <= C` forwards remain in that family;
- a logical one-token batch may be physically padded, but synthetic rows must be
  inactive and invisible in counts and output;
- the routed receive capacity covers the worst legal skew, at least
  `L * EP * K` rows unless a tighter bound is proven from the routing contract;
- `M > C` fails with a clear contract error rather than switching communication
  family inside an already compiled model; and
- configurations outside the proven EP, top-k, expert-count, dtype, alignment,
  HCCL-window, workspace, and descriptor-width domain are rejected before the
  first fused launch.

The numeric bounds must be derived from the implementation and recorded in the
acceptance receipt. A scheduler setting is not safe merely because memory was
successfully allocated: every HCCL window subregion and device-side loop/index
must cover the derived domain without overlap or truncation.

### Current source-derived boundary audit

The following are properties of the code under review, not yet an accepted
support matrix. An implicit limit must either become an enforced startup/host
check or be removed by an implementation with boundary-complete evidence.

| Boundary | Current implementation | Acceptance consequence |
| --- | --- | --- |
| scalar widths | host tiling stores `M`, hidden/intermediate dimensions, local experts, top-k, EP, and `max_output_size` in `uint32_t`; device parameters narrow several of them to `int32_t` | validate every downcast and product before tiling serialization |
| route domain | `T = M * K`; route IDs use INT32; the inactive sentinel is `local_experts * EP` and routing is configured for one additional expert | validate positive dimensions, `T` arithmetic, every route ID, sentinel separation, and global physical-expert count |
| route expert bookkeeping | the imported routing helper caps an internal aligned expert buffer at 5,120 entries rather than rejecting a larger domain | declare and enforce a compatible global-expert bound or remove the truncating cap |
| BF16 geometry | GMM2 derives its intermediate input as `N / 2`; the expert pipeline also uses `local_experts - 2` | reject odd/incompatible `N` and too-small expert counts before device launch |
| copy descriptors | active-mask copies are safely chunked at 8,192 INT32 elements (32 KiB), while other expert-row copies cast byte counts/strides to `uint16_t` | derive explicit expert/stride bounds for every remaining descriptor cast |
| routed capacity | A2 Python currently derives `max_output_size = local_capacity * EP * K`; A3 retains a fixed 131,072-row value | prove the bound covers legal incoming skew and fits every local/peer region on each SoC |
| peer window | routed input starts at byte 0; the per-token-scale region starts at `align(segment_size / 3, 512)`; returned output starts 1 MiB later; count/control starts at `segment_size - 2 MiB` | check every region formula against the actual HCCL segment size before launch |
| count matrix | the final 2 MiB contains the aligned peer count matrix and control state; the count-matrix rows are aligned to 128 INT32 entries | enforce the derived `EP`, expert-count, alignment, and control-region fit rather than assuming it |
| publication marker | routed counts are published with high byte `0x4D`; consumers recognize mask `0xFF000000` and remove the marker | require every plain count to remain below `2^24` and prove stale generations cannot satisfy the same marker |
| polling/lifetime | readiness and cross-rank waits have no host-visible timeout; reset helpers exist but the proposed shared protocol does not invoke the full count reset | exact repeated-generation tests on A2 and A3 are mandatory; hangs are failures, not inconclusive passes |
| unpermute | its tiler divides work by top-k and per-core token rows | validate non-zero top-k and a domain in which per-core tiling cannot reach an invalid zero divisor |

The current A2 selector enforces only part of this domain (explicit opt-in,
expert parallelism, `2 <= EP <= 8`, and at least three local experts). It does
not yet establish a BF16-only dtype gate, a top-k/global-expert/window bound, or
the complete shape relationships above. Direct Torch/ACLNN callers bypass even
those framework checks. The host operator must therefore own the safety-critical
shape, width, route, and physical-resource validation; the Python selector may
add a narrower product policy but cannot be the only correctness boundary.

### HCCL window and publication invariants

The window layout is part of the operator ABI even when it is not exposed to
Python. Acceptance requires proof that, for every supported shape:

- input, per-token scale, returned output, count/control, and any scratch region
  have non-overlapping byte ranges;
- local and peer addresses are derived with architecture-correct semantics;
- all offsets, alignments, and sizes fit the physical `HCCL_BUFFSIZE`-derived
  window;
- a consumer cannot observe a count before its corresponding output is visible;
- a later invocation cannot accept a stale count or stale expert output from an
  earlier generation; and
- wraparound or tag-bit ambiguity is impossible in the declared token/count
  domain.

Removing a reset is a protocol change, not cleanup. It requires changing-route,
zero-count, buffer-poisoning, and repeated-window tests on A2 and A3.

The current marker is constant rather than a per-invocation generation value.
Consequently, the final acceptance argument must show why the completion and
cross-rank synchronization protocol prevents a count bearing the same marker
from a prior invocation from being accepted. If that argument cannot be made
from ordering and lifetime invariants, the protocol needs a real generation or
an explicit reset before its performance can be considered.

### Graph and process-lifetime invariants

- Dynamic token dimensions must remain symbolic through Meta/FakeTensor and
  graph capture.
- Ordinary integer attributes must not specialize a runtime-varying shape unless
  that specialization is an explicit graph-bucket contract.
- One captured graph must safely replay across multiple legal token counts and
  routing generations.
- Global capacity and reserved-mask state must be initialized before use and
  must reject incompatible runner configurations sharing one worker process.
- The live active-mask view must be derived from stable reserved storage; a
  short-lived profile view must not be retained as the runtime domain.
- DP participants must agree on the communication family and padded domain.

## Acceptance matrix

The table is a coverage index, not a substitute for commit-bound receipts.

| Layer | A2 BF16 | A2 dynamic-W8A8 | A3 shared regression | Required oracle |
| --- | --- | --- | --- | --- |
| build/package/registration | reported on #199 | reported on #203 | compile reported; exact installed artifact still required | tested SHA and package/library identity |
| schema and Meta | partial focused tests | inherited/partial | not yet runtime-bound | shape, dtype, device, mutability, symbolic dimensions |
| direct multi-rank output/counts | two-rank structured oracle reported | two-rank changing-route oracle reported | existing tests are primarily no-exception | exact counts plus exact/tolerance-bounded output |
| input immutability | BF16 test covers `expert_idx` | **known gap: generic mask path writes through `expert_idx`** | **required by #224** | clone-and-compare every caller-owned input |
| active mask | zero/one-active and repeated launch reported | zero/one-active reported | required by #224 | absent/all/partial/zero/one-active |
| window generation reuse | tagged/reuse coverage reported | tagged/reuse coverage reported | required by #224 | poisoned buffers, changing routes, zero wave between non-zero waves |
| capacity/workspace boundaries | mocked and selected A2 runs | `C=512` campaign reported | required by #224 for intersecting changes | `1`, `2`, `C`, `C+1`, skew, alignment, HCCL fit |
| graph replay | real-model campaigns reported | real-model campaigns reported | required by #224 | same graph, changing legal sizes/routes, no fallback/recapture |
| model quality | bounded Qwen receipt reported | bounded Qwen receipt reported | regression smoke only unless behavior changes | explicit numerical contract and retained negative results |
| performance | bounded model/topology-local results | bounded model/topology-local results | not an exit condition for compatibility | equivalent boundary, crossed repetitions, no universal claim |

Until the red and unbound cells are closed on the final reviewed heads, the
matrix does not justify merging a shared protocol change.

### Known blocking defects and gaps

- The generic dynamic-W8A8 active-mask implementation writes inactive sentinel
  IDs back into the caller's `expert_idx`. The Torch schema does not mark this
  tensor mutable. The intended resolution is implementation-owned scratch
  storage, followed by an A2/A3 input-immutability oracle; broadening the public
  mutation contract is not proposed.
- The A2 unquantized selector can currently admit an unquantized FP16 model even
  though the change is reviewed and evidenced as BF16 capability. It also lacks
  a demonstrated draft/MTP policy.
- Shape inference is effectively a no-op and Torch Meta checks only the optional
  mask rank/length. Weight-list consistency, output/count shape, route range,
  geometry, descriptor width, and window fit are not yet closed at the host
  boundary.
- Existing A3 direct tests mainly prove that the process does not raise. They do
  not currently establish exact route counts, output correctness, input
  ownership, changing-generation reuse, or graph-boundary behavior.
- The A3 selector and fixed routed-output capacity describe a wider runtime
  domain than the currently retained tests prove. #224 must either close that
  domain or motivate a narrower enforced contract.

## Evidence receipt format

Every hardware result used for acceptance must retain:

- repository, PR, exact Git SHA, and dirty-tree state;
- custom-op package and loaded-library identity or hash;
- SoC, card count/topology, CANN, compiler, PyTorch, vLLM, and vLLM Ascend
  versions;
- model/dtype/quantization, TP/EP/DP, expert count, top-k, hidden/intermediate
  dimensions, `C`, physical padding, `max_output_size`, and HCCL window size;
- exact command/test node, random seed, input/route generation, and oracle;
- pass, fail, or invalid status with logs/artifact paths and hashes; and
- explicit non-claims and residual gaps.

Historical performance and quality artifacts may explain a decision, but they
do not prove the correctness of a different kernel tree. If an earlier receipt
is reused, the relevant subtree identity and every intervening semantic change
must be demonstrated.

## Merge and review policy

The supported proposal is a default-off, explicit-opt-in capability. Evidence
for one model, topology, phase, or dtype does not justify universal enablement or
a model-independent profitability heuristic.

The BF16 parent should establish the common A2/A3 contract and land first. The
dynamic-W8A8 child should then be rebased into a reviewable dtype-specific delta.
Unrelated AllGather/EPLB changes and shared publication optimizations should be
separate unless they are necessary for the declared operator contract and carry
their own full acceptance evidence.

Review threads remain the source of truth for requested changes. Summary issues
and this document provide navigation and a durable contract; they do not mark a
review concern resolved without code and evidence.
