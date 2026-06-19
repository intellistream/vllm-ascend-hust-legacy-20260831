# Worker-Local Mapped H2D Narrow Smoke

Generated: 2026-06-19 Asia/Shanghai

## Scope

This run validated the worker-local mapped-host H2D path through
`CpuNpuOffloadingHandler` after a branch-local narrow 910B build:

```bash
VLLM_ASCEND_ACLNN_OPS=kv_cache_block_gather \
VLLM_ASCEND_BUILD_ASCENDC_KERNELS=0 \
SOC_VERSION=ascend910b1 \
python3 -m pip install -e . --no-build-isolation --no-deps -v
```

The run used:

```text
git_sha: cce29d061dad2549f4b60098ac5ff600b4db5f38
image: quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler
device: /dev/davinci0
h2d_backend: mapped
block_bytes: 4096
selected_blocks: 8
direction: h2d
pattern: random
warmup: 1
iters: 3
```

## Result

Status: pass.

The narrow build registered both required torch ops:

```text
has swap_blocks_batch True
has kv_cache_block_gather True
```

The benchmark auto-configured the custom op runtime environment:

```text
ASCEND_CUSTOM_OPP_PATH=/tmp/vllm-ascend-hust/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend
VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB=/tmp/vllm-ascend-hust/vllm_ascend/_cann_ops_custom/vendors/vllm-ascend/op_api/lib/libcust_opapi.so
```

Measured row:

| direction | h2d_backend | block_bytes | selected_blocks | mean_ms | p95_ms | p99_ms | gbps | status |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| h2d | mapped | 4096 | 8 | 0.612 | 0.704 | 0.716 | 0.11 | pass |

## Interpretation

This closes the immediate engineering blocker for first-stage worker-local
mapped H2D validation:

- the branch can build the needed custom ACLNN op without `tmp/cann-stack`;
- `vllm_ascend_C` can import without compiling unrelated bundled AscendC
  kernels;
- `swap_blocks_batch` and `kv_cache_block_gather` can coexist in the narrow
  transfer build;
- mapped H2D can execute through `CpuNpuOffloadingHandler`.

The number above is only a tiny smoke payload, not a throughput threshold. The
next useful experiment is a mapped-vs-copy H2D matrix with larger block counts
and enough iterations to reduce Python runner noise.
