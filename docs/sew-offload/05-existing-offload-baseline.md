# Existing Offload Baseline on Single Ascend 910B3

## Purpose

This note records the first real single-card baseline for Qwen3-30B-A3B on
the local vLLM-Hust plus vLLM-Ascend-Hust stack. The goal is to identify what
actually breaks or becomes tight before designing SEW-Offload further.

## Environment

- Model path used: `/data/shared-models/Qwen3-30B-A3B`
- User-provided path `/data/Qwen3-30B-A3B` was not present on this machine.
- Device: one Ascend 910B3, physical NPU 4, 64 GB HBM.
- Python: `/root/miniconda3/envs/vllm-hust-dev/bin/python`
- vLLM: `/root/vllm-hust`, `v0.17.2.post2.dev1186+g40dfe0e1f`
- vLLM Ascend: `/root/vllm-ascend-hust`
- Model config: Qwen3 MoE, 48 layers, 128 experts, top-8, bf16.
- Artifact directory:
  `/root/vllm-ascend-hust/artifacts/sew_offload/existing_offload_20260529T143705Z`

Small compatibility patches were needed to make the current local vLLM-Hust
and vLLM-Ascend-Hust revisions run together. They do not implement SEW-Offload;
they only bridge API drift in MoE runner fields and CUDA-to-NPU stream wrappers.

## Results

| Case | Result | Key Evidence |
| --- | --- | --- |
| no offload | Success | `LOAD_OK 49.062s`, `GENERATE_OK 17.158s` |
| UVA expert offload, 8 GB | Failed before weight loading | `get_accelerator_view_from_cpu_tensor` is unsupported on `npu` |
| Prefetch expert offload, group 4 / 1 / step 1 | Failed during profile forward | raw run first failed at unwrapped `torch.cuda.is_current_stream_capturing()` |
| Prefetch after CUDA wrapper compatibility patch | Failed during MoE GMM | `npu_grouped_matmul` saw expert weight on CPU and activations on NPU |

## Memory And Timing

| Case | Weight Log | Peak NPU 4 HBM | Load Time | End-to-End Init |
| --- | ---: | ---: | ---: | ---: |
| no offload | 56.9001 GB | 63,886 MB | 17.25 s | 49.062 s |
| prefetch experts | 43.4001 GB | 50,697 MB | 19.99 s | failed |
| prefetch experts after wrapper patch | 43.4001 GB | 50,696 MB | 20.75 s | failed |

The prefetch backend clearly reduces resident weight memory by about 13.5 GB
for this setting, so the HBM-saving direction is real. However, it is not a
working Ascend MoE offloading path yet.

## Diagnosis

1. Full residency barely fits.

   With a tiny 512-token context and only 0.5 GB KV cache, Qwen3-30B-A3B
   reaches about 63.9 GB HBM on a 64 GB card. This validates the research
   motivation: any longer context, larger batch, graph buffer, or extra runtime
   buffer can push the model over the edge.

2. UVA is the wrong abstraction for this Ascend stack.

   The UVA backend depends on creating an accelerator view over CPU pinned
   memory. The local implementation explicitly rejects `npu`, so it cannot be
   our baseline or design substrate.

3. Existing prefetch is layer/parameter oriented, not MoE expert-window oriented.

   It chooses layers by index and offloads matched parameter names such as
   `experts`. It does not know which experts are routed in the current request,
   which experts are already resident, or which expert miss is on the critical
   path.

4. Existing prefetch is CUDA-shaped.

   The raw run failed on `torch.cuda.is_current_stream_capturing()` in the
   NPU runtime. After adding the missing wrapper mapping, execution progressed
   further, which means this was a portability bug rather than the fundamental
   limitation.

5. The fundamental limitation appears at the Ascend MoE boundary.

   The patched prefetch run failed in `torch_npu.npu_grouped_matmul` because a
   weight tensor was still on CPU while hidden states were on `npu:0`. For
   Ascend grouped MoE, the actual compute boundary expects the weight objects
   passed into grouped matmul to already be NPU-resident with a compatible
   layout. A generic layer wrapper cannot guarantee that after Ascend-specific
   weight processing and MoE dispatch/finalize transformations.

## Implications For SEW-Offload

The next system should not be "vLLM prefetch with NPU aliases". It should make
offloaded experts visible at the Ascend MoE execution boundary:

- Fixed expert slots: expert weights used by `npu_grouped_matmul` should point
  to stable NPU slot tensors, not arbitrary CPU/device parameter objects.
- Expert-aware transfer: host-to-HBM copies should be scheduled for routed
  expert IDs and layer deadlines, not only for layer index groups.
- Layout-stable buffers: slot tensors must preserve the post-processed Ascend
  weight layout and dtype expected by grouped matmul.
- Overlap-aware execution: resident experts should execute first while missing
  expert slots are loaded, then miss experts should run in a small follow-up
  grouped phase.
- Native NPU streams/events: the implementation should use `torch.npu` streams
  and events directly, with graph-capture behavior made explicit instead of
  inherited from CUDA assumptions.

## Next Engineering Target

The immediate MVP should be a trace-and-measure mode plus a synchronous slot
prototype:

1. Record per-layer routed expert IDs and token counts for real prompts.
2. Build a CPU expert store after Ascend weight post-processing.
3. Allocate a small fixed NPU expert-slot pool for one layer.
4. On a slot miss, synchronously copy the expert into the slot before grouped
   matmul.
5. Measure exposed copy time, slot hit rate, and how much stall remains.

Only after this should we implement async deadline-aware prefetch and
hit-first phased execution.
