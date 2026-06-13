# SEW-MoE Dual Acceleration Design

## Summary

SEW-MoE is the shared design path for bringing the useful part of llama.cpp's MoE execution model into vLLM Ascend without copying llama.cpp's runtime architecture. The core idea is to make sparse expert activation a first-class runtime signal: every MoE layer invocation should expose which experts were active, how many tokens each expert received, how that became the grouped matmul shape, and where time was spent across the MoE pipeline.

The first approved scope is **P0: Active Expert Observability**. P0 is trace-only and changes no model execution semantics. It provides the evidence needed to choose between two later acceleration tracks:

- **SEW-Compute:** non-offload acceleration for all-HBM Qwen3-30B-A3B by optimizing routing, grouped matmul, and combine paths.
- **SEW-Offload:** HBM-limited acceleration by extending the existing fixed-slot offload runtime with measured active expert working sets, async transfer, and hit-first phased execution.

## Current Evidence

The Qwen3-30B-A3B non-offload Ascend profile report shows that `GroupedMatmul` dominates both prefill and decode. In the collected windows, it accounts for more than half of operator time, while routing, unpermute, and high-frequency small kernels also contribute meaningful decode overhead.

The repository already contains relevant foundations:

- `vllm_ascend/ops/fused_moe/moe_stage_contracts.py` defines typed boundaries for prepare, dispatch, MLP, and combine.
- `vllm_ascend/ops/fused_moe/moe_comm_method.py` already records the MoE pipeline as Stage T, R, C, and M when pipeline profiling is enabled.
- `vllm_ascend/moe_offload/` already contains fixed-slot concepts such as host store, slot bank, slot mapping, transfer engine, tiered residency, and phase split.
- `docs/sew-offload/00-charter.md` already frames Ascend offload as fixed expert-window scheduling rather than a generic dynamic cache.

P0 should therefore extend the existing MoE stage boundaries and offload observability instead of introducing a separate profiler stack.

## Goals

1. Record active expert working sets for every observed MoE layer invocation.
2. Connect logical routing output to physical grouped execution shape.
3. Preserve default behavior when observability is disabled.
4. Produce JSONL artifacts that can drive both non-offload and offload decisions.
5. Enhance profile analysis so the next optimization is chosen from evidence, not intuition.

## Non-Goals

P0 does not:

- change routing, top-k selection, gate weights, or token dispatch semantics;
- alter `GroupedMatmul`, routing, or combine kernels;
- add async host-to-HBM transfer;
- enable fixed-slot offload on new execution paths;
- modify scheduler, worker, or model runner behavior;
- attempt to improve throughput directly.

P0 is successful when it makes the next throughput-improving change obvious and measurable.

## Design

### ActiveExpertSet

Add a trace-oriented `ActiveExpertSet` payload with these fields:

- `layer_id`
- `step_id`
- `mode`: `prefill`, `decode`, or `unknown`
- `num_tokens`
- `top_k`
- `num_logical_experts`
- `active_experts`
- `expert_token_counts`
- `fanout`
- `source`: `logical_topk` or `grouped_dispatch`
- optional `group_list_type`
- optional `group_list_signature`
- optional `physical_expert_count`

There are two useful records per MoE invocation:

- **Logical record:** emitted after `select_experts`, based on `topk_ids`.
- **Grouped record:** emitted after token dispatch, based on `group_list` and `group_list_type`.

The logical record mirrors llama.cpp's key insight: only currently selected experts matter. The grouped record explains what Ascend actually sends into grouped matmul.

### Runtime Ownership

Extend the existing `MoeOffloadRuntime` and `TraceCollector` family rather than creating a second global runtime. The name can stay under `moe_offload` for P0 because the data is shared by offload and non-offload; later implementation may move shared observability into `vllm_ascend/ops/fused_moe/observability.py` if the boundary becomes cleaner.

The runtime should expose methods equivalent to:

```text
trace_logical_active_experts(layer_id, topk_ids, num_experts, mode)
trace_grouped_active_experts(layer_id, group_list, group_list_type, physical_expert_count, mode)
```

Both methods return their inputs unchanged or only return metadata. They must not allocate large device tensors or force synchronization in the default disabled path.

### Integration Points

The first integration point is `AscendUnquantizedFusedMoEMethod.apply()` after `select_experts`, where `topk_ids` and `topk_weights` are already available. This creates the logical active expert record.

The second integration point is `MoECommMethod.fused_experts()` after `token_dispatch`, where `token_dispatch_output.group_list` and `group_list_type` are available. This creates the grouped execution record and can be correlated with Stage R/C/M timings.

The existing Stage T/R/C/M profiler should remain the timing source:

- T: offload plan and transfer staging
- R: token dispatch and routing reorder
- C: MLP compute, including grouped matmul and activation
- M: token combine

P0 should add active expert metadata to the same JSONL profile stream or write a clearly named companion JSONL stream. A single stream is preferred if it keeps events easy to correlate by `layer_id` and `step_id`.

### Mode Detection

P0 should use a conservative mode label:

- `prefill` when the MoE invocation has more than one token and is clearly processing prompt tokens;
- `decode` when the invocation is in a single-step decode path or has one token per sequence;
- `unknown` when the local context does not prove either.

Mode labels are diagnostic only. Incorrect mode inference must not affect execution.

### Profile Analyzer Enhancements

Extend the existing profile analyzer output with a SEW-MoE section when active expert artifacts are present:

- fanout distribution by phase and layer;
- top layers by active expert count;
- top layers by grouped token count;
- common `group_list_signature` values;
- correlation hints between Stage C time and grouped shape;
- offload planning hints such as minimum observed slot budget and repeated active expert locality.

The analyzer should remain useful without active expert artifacts. Missing SEW-MoE data should produce a short note, not an error.

## Data Flow

1. `select_experts` produces `topk_ids` and `topk_weights`.
2. P0 records logical active experts from `topk_ids`.
3. Existing optional offload planning may run if configured.
4. Token dispatch produces sorted/permuted hidden states and `group_list`.
5. P0 records grouped active expert shape from `group_list`.
6. Existing MLP compute and combine continue unchanged.
7. Existing pipeline profiler records Stage T/R/C/M timings.
8. Analyzer joins active expert records and timing/profile data offline.

## Error Handling

P0 observability is fail-soft:

- Disabled observability must be a no-op.
- If a trace conversion fails, runtime should record a compact failure event when possible and continue execution.
- The trace path must create parent directories when writing JSONL.
- Invalid or unsupported `group_list_type` should be reported in the record as `unsupported`, not used to stop inference.
- Any future execution-changing offload path remains fail-closed; P0 must not weaken existing offload safety checks.

## Testing

Unit tests should cover:

- logical active expert extraction from small `topk_ids` tensors;
- grouped expert count extraction from count and cumsum `group_list` variants;
- JSONL serialization and deterministic field names;
- disabled mode preserving current behavior;
- analyzer behavior with and without active expert artifacts;
- no mutation of `topk_ids`, `topk_weights`, or `group_list`.

Existing tests under `tests/ut/moe_offload/` and `tests/ut/benchmarks/` are the natural homes for the first implementation plan.

## Acceptance Criteria

P0 is complete when:

- Active expert JSONL records are generated only when explicitly enabled.
- Non-offload and offload runs can both emit logical and grouped active expert records.
- Existing default tests for MoE and offload still pass.
- The profile analyzer can summarize fanout and grouped shape distributions from the new artifacts.
- A Qwen3-30B-A3B profiling run can identify whether the next optimization should target Stage R, C, M, or T.

## Future Work After P0

P1 for **SEW-Compute** should use P0 evidence to choose one narrow non-offload acceleration target, likely one of:

- decode grouped matmul fast path for dominant group signatures;
- routing and dispatch fusion;
- unpermute/combine fusion;
- high-frequency small-kernel fusion around RMSNorm, slice, RoPE, or cache write.

P1 for **SEW-Offload** should use P0 evidence to choose slot budgets and locality policies before adding async transfer. The intended order remains:

1. trace-backed slot simulation;
2. fixed-slot correctness for the narrow AllGather unquantized path;
3. async load metrics;
4. hit-first phased execution;
5. static window or graph replay eligibility.

## Review Notes

This design intentionally keeps P0 observational. The fastest path to real throughput improvement is to avoid premature kernel or offload work until active expert shape, stage timing, and fanout data show which part of the MoE path is limiting Qwen3-30B-A3B on Ascend.
