# Device KV Transfer Prototype

This branch carries the experimental mapped-host KV cache load path for the
CPU offload connector. It is intended as a collaboration branch for validating
device-driven H2D KV block gather on Ascend.

## What is included

The prototype has two parts:

1. `csrc/kv_cache_block_gather`: an ACLNN custom operator that gathers selected
   KV blocks from mapped host pages into the NPU KV cache.
2. `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload`: an optional
   connector load path that calls `torch.ops._C_ascend.kv_cache_block_gather`
   instead of issuing one host-side tensor copy per block.

The default behavior is unchanged. The mapped-host gather path is enabled only
when `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1`.

## Build

Build vLLM Ascend with custom kernels enabled. The custom operator is part of
the normal ACLNN custom-op list for `ascend910b` and `ascend910_93`, so a normal
source build installs it under `vllm_ascend/_cann_ops_custom`.

```bash
export COMPILE_CUSTOM_KERNELS=1
python -m pip install -e . --no-build-isolation
```

For focused custom-op iteration, build the operator directly:

```bash
bash csrc/build_aclnn.sh "$PWD" ascend910b
```

## Runtime

Enable the experimental path with:

```bash
export VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1
```

The in-tree torch binding first looks up `aclnnKvCacheBlockGather` through the
normal opapi resolution path. If needed while debugging an external operator
package, override it with:

```bash
export VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB=/path/to/libcust_opapi.so
```

If the torch op is unavailable or the tensors are not contiguous, dtype-matched
`float32`, `float16`, or `bfloat16` KV pages, the connector falls back to the
existing tensor-copy path.

## Smoke

Use a CPU offload run that creates non-empty `load_block_mapping`, then confirm:

1. `torch.ops._C_ascend.kv_cache_block_gather` exists after importing
   `vllm_ascend`.
2. `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1` does not log the unavailable-op
   fallback.
3. The request finishes with the same output as the tensor-copy path.

For operator-only validation, compile and run:

```bash
g++ -std=c++17 -O2 \
  csrc/kv_cache_block_gather/benchmarks/kv_cache_block_gather_benchmark.cpp \
  -I"$ASCEND_HOME_PATH/include" \
  -I"vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/include" \
  -L"$ASCEND_HOME_PATH/lib64" \
  -L"vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib" \
  -Wl,-rpath,"$ASCEND_HOME_PATH/lib64" \
  -Wl,-rpath,"$PWD/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib" \
  -lascendcl -lnnopbase -lcust_opapi \
  -o /tmp/kv_cache_block_gather_benchmark

/tmp/kv_cache_block_gather_benchmark 0 \
  --num-pages 4096 \
  --selected-blocks 4096 \
  --elems-per-block 16384 \
  --src-pattern random \
  --dst-pattern random \
  --warmup 3 \
  --iters 10
```
