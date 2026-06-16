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

Use a vLLM Ascend container that matches this branch's dependency line. The
current branch expects the `torch==2.9.0` / `torch-npu==2.9.0` line, so the
recommended development image is:

```text
quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler
```

A typical source-mounted development shell is:

```bash
docker run --rm -it \
  --privileged \
  --network host \
  --ipc host \
  -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
  -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro \
  -v /path/to/vllm-ascend-hust:/workspace/vllm-ascend-hust \
  -w /workspace/vllm-ascend-hust \
  quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler \
  bash
```

Inside the container, source CANN and mark the mounted checkout as a safe git
directory if the checkout is owned by the host user:

```bash
git config --global --add safe.directory /workspace/vllm-ascend-hust
source /usr/local/Ascend/cann-8.5.1/set_env.sh
```

Build vLLM Ascend with custom kernels enabled. The custom operator is part of
the normal ACLNN custom-op list for `ascend910b` and `ascend910_93`, so a normal
source build installs it under `vllm_ascend/_cann_ops_custom`.

```bash
export COMPILE_CUSTOM_KERNELS=1
export SOC_VERSION=ascend910b1
python -m pip install -e . --no-build-isolation
```

For focused custom-op iteration, build the operator directly:

```bash
bash csrc/build.sh -n kv_cache_block_gather -c ascend910b
```

The focused command writes `csrc/output/CANN-custom_ops--linux.aarch64.run`.

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

Current branch status:

1. The `quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler` container can see NPU
   devices through the host driver mount.
2. The focused `kv_cache_block_gather` ACLNN custom-op build completes and
   produces `csrc/output/CANN-custom_ops--linux.aarch64.run`.
3. A Python-only editable install works with `COMPILE_CUSTOM_KERNELS=0`,
   `SOC_VERSION=ascend910b1`, and `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1`; the
   plugin imports from the mounted source tree and the env flag resolves to
   `True`.

Full `COMPILE_CUSTOM_KERNELS=1 python -m pip install -e .` now reaches the
repository-wide C++ extension build after building the custom-op list. If it
fails in unrelated extension targets, validate this prototype first with the
focused custom-op build above and a Python-only editable install.

Python-only branch smoke:

```bash
export COMPILE_CUSTOM_KERNELS=0
export SOC_VERSION=ascend910b1
export ASCEND_RT_VISIBLE_DEVICES=0
export VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1

python -m pip install -e . --no-build-isolation --no-deps
python - <<'PY'
import importlib.metadata as metadata
import torch
import torch_npu
import vllm_ascend
from vllm_ascend import envs

print("dist", metadata.version("vllm-ascend-hust"))
print("torch", torch.__version__)
print("torch_npu", torch_npu.__version__)
print("npu_count", torch.npu.device_count())
print("vllm_ascend", vllm_ascend.__file__)
print("host_gather", envs.VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER)
PY
```

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
