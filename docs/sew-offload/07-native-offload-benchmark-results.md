# Native vLLM Ascend Offloading Benchmark Pilot

## Purpose

This note records the first minimal benchmark run using
`docs/sew-offload/benchmark_config.yaml` and the current vLLM Ascend native
weight offloading path.

At this stage we only report:

- throughput
- TTFT
- TPOT

## Benchmark Runner

Runner:

```text
tools/sew_offload/run_minimal_offload_benchmark.py
```

Artifact root:

```text
artifacts/sew_offload/benchmarks/sew_bench_ascend_moe_30b/pilot_native_offload_20260529
```

The formal dataset in the benchmark config is `lmsys/lmsys-chat-1m`, but this
machine currently has no local `lmsys-chat-1m` and the active Python environment
does not include the `datasets` package. Therefore this pilot uses a
`synthetic_smoke` manifest with the same benchmark bucket schema:

```text
artifacts/sew_offload/benchmarks/sew_bench_ascend_moe_30b/requests.jsonl
```

The smoke request used here is one `short_chat` request with 128 prompt tokens
and 128 output tokens. This is not the final paper benchmark; it is a minimal
execution test for the native offloading path.

## Results

| Case | Offload Setting | Resident Weight | Status | Output Throughput | TTFT | TPOT |
| --- | --- | ---: | --- | ---: | ---: | ---: |
| native_prefetch_experts_short1 | `prefetch`, group4/num1/step1, `offload_params=experts` | 43.4001 GB | failed before generation | N/A | N/A | N/A |
| native_prefetch_all_short1 | `prefetch`, group4/num1/step1, all layer params | 42.9722 GB | failed before generation | N/A | N/A | N/A |
| no_offload_short1 | no weight offload | 56.9001 GB | ok | 7.6207 tok/s | 465.78 ms | 128.59 ms |

The no-offload row is only a sanity reference proving that the benchmark runner
and Qwen3-30B-A3B path can produce throughput, TTFT, and TPOT on this NPU. It
is not an offloading baseline.

## Failure Evidence

### Native expert prefetch

The expert-only native prefetch setting hits the intended 13.5GB offload
budget:

```text
Loading model weights took 43.4001 GB
```

It fails during profile forward before any request is generated:

```text
RuntimeError: Expected all tensors to be on the same device, but got weight is
on cpu, different from other tensors on npu:0 ... wrapper__npu_grouped_matmul
```

Interpretation: the native layer/parameter prefetch path leaves at least one
MoE expert weight visible to Ascend `npu_grouped_matmul` as a CPU tensor. This
means no throughput, TTFT, or TPOT can be reported for the current native
expert offloading path.

### Native all-parameter layer prefetch

The all-parameter variant also reduces resident weight:

```text
Loading model weights took 42.9722 GB
```

It fails earlier in a dense matmul path:

```text
RuntimeError: Expected all tensors to be on the same device, but got other is
on cpu, different from other tensors on npu:0 ... wrapper_NPU__matmul
```

The run also emits Ascend runtime errors:

```text
The vector core execution is abnormal.
The DDR address of the MTE instruction is out of range.
```

Interpretation: the problem is not limited to MoE grouped matmul. The current
native prefetch abstraction can expose CPU tensors or invalid device-side
addresses to NPU compute after parameter replacement/prefetch, which is unsafe
for Ascend execution.

## What This Means

The current vLLM native offloading path can reduce resident model weight to the
target budget, but it is not a valid Ascend MoE inference path yet. Its failure
boundary is before serving metrics:

```text
weight offload succeeds -> model profile forward fails -> no request generation
```

Therefore the next optimization should not start from prefetch policy tuning.
We first need a correct Ascend-specific weight residency abstraction.

## Optimization Directions Exposed By The Run

1. Fixed NPU-resident expert slots.

   The highest-priority direction is to allocate stable NPU expert slots and
   make `npu_grouped_matmul` always see NPU tensors. The dynamic object-level
   CPU/NPU parameter swap used by native prefetch is too fragile for Ascend MoE.

2. Layout-stable post-processed weight buffers.

   Expert weights should be offloaded after Ascend weight post-processing, and
   reloaded into buffers with the exact dtype, stride, shape, alignment, and
   layout expected by the Ascend MoE kernels. The MTE out-of-range error in the
   all-parameter run suggests that address/layout validity must be treated as a
   first-class invariant, not a side effect of `param.data` replacement.

3. MoE execution-boundary integration.

   The integration point should be the boundary before
   `moe_comm_method.fused_experts()`, not a generic decoder-layer forward hook.
   At that boundary we know the routed expert IDs, per-expert token counts, and
   grouped matmul inputs. That is where expert slot residency can be checked
   and repaired before NPU compute starts.

4. Native `torch.npu` transfer and synchronization.

   The native prefetch path is CUDA-shaped and only partially wrapped. SEW
   should use NPU streams/events directly and place explicit waits before the
   Ascend compute boundary. Otherwise a copy can be logically scheduled but not
   actually safe for AIC/MTE execution.

5. Prefetch/overlap only after correctness.

   Once fixed slots and layout-stable buffers make synchronous miss loading
   correct, the next measurable target is to hide load time with expert-aware
   prefetch and hit-first phased execution. The benchmark's main metric then
   becomes meaningful: exposed offloading stall per output token.

