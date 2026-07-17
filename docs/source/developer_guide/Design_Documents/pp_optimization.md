# Pipeline-Parallel Optimization Integration

This repository supplies the Ascend runtime integration for the PP
optimization scheduler in the paired vLLM-HUST repository. Install both
repositories from sibling editable source trees; mixing this plugin with a
binary vLLM package is unsupported.

## Runtime contract

The scheduler keeps multiple decode microbatches in flight. Each
`SchedulerOutput` carries a microbatch ID, and a microbatch remains unavailable
until EngineCore retires its model output. The Ascend worker must therefore
preserve PP send-buffer lifetime across queued executions and expose rank-local
timing for calibration.

When `VLLM_PP_OPT_OVERLAP_SENDS=0`, the communication coordinator waits for the
previous tensor send immediately before issuing the next send. When overlap is
enabled, the worker clones intermediate tensors and retains references until
all asynchronous send handles complete. The benchmark's validated
configuration uses overlap disabled.

## Model runner changes

The NPU model runner records request count, aggregate context length, scheduled
tokens, and forward boundaries when `VLLM_PROFILE_PP_OPT_ENABLED=1`. Profiling
is disabled by default. Calibration mode synchronizes the NPU only around the
measured forward interval; normal serving does not add this synchronization.

`VLLM_CUSTOM_SCOPES_FOR_PROFILING=1` adds a CANN scope with PP rank,
microbatch ID, request count, context tokens, and scheduled tokens. This scope
supports device-side pipeline occupancy analysis through the built-in
`torch_npu` profiler.

The DecodeBench path uses CPU slot mapping when
`VLLM_ASCEND_FORCE_CPU_SLOT_MAPPING=1`. This avoids unsupported slot-mapping
kernel behavior for externally supplied KV blocks. KV filling accepts Ascend's
separate key/value cache tensors and batches fills by cache group.

## Model compatibility

The tested Qwen models use native rotary embedding when
`VLLM_ASCEND_FORCE_NATIVE_ROPE=1`. Qwen3-235B-A22B also requires the current
vLLM MoE runner API: shared experts are attached to the unified `FusedMoE`
runner, and the final tensor-parallel reduction happens after routed and shared
outputs are combined.

These compatibility changes affect baseline and optimized runs equally. The
optimization does not bypass transformer layers or replace model forward
execution.

## Editable setup

From the parent workspace:

```bash
python3.11 -m venv --system-site-packages .venv-pp-opt
source .venv-pp-opt/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
source /path/to/atb/set_env.sh --cxx_abi=1

VLLM_TARGET_DEVICE=empty \
  pip install --no-build-isolation -e ./vllm-hust
pip install --no-build-isolation -e ./vllm-ascend-hust
```

Verify the source paths before benchmarking:

```bash
python -c 'import vllm, vllm_ascend; print(vllm.__file__); print(vllm_ascend.__file__)'
```

The benchmark launcher in `vllm-hust/benchmarks/pp_opt` performs a stricter
editable-install check automatically.
