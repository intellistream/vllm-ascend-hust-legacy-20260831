# Worker-Local Mapped H2D Smoke Attempt

Generated: 2026-06-19 Asia/Shanghai

## Scope

This run attempted to validate the experimental worker-local mapped-host H2D
path in `CpuNpuOffloadingHandler` using:

- `--h2d-backend mapped`
- `--block-bytes 4096`
- `--selected-blocks 8`
- `--directions h2d`
- `--patterns random`
- `--warmup 1`
- `--iters 3`

## Result

The smoke did not reach a valid mapped transfer result.

The implementation bug found in the first attempt was fixed: the call order for
`torch.ops._C_ascend.kv_cache_block_gather` must be:

```text
src_block_ids, src_pages, dst_block_ids, out
```

After that fix, the remaining blocker was the reproducible registration/build
path for the required torch extension in the working container.

## Build Findings

- Single-op `kv_cache_block_gather` build succeeds.
- The custom opapi library exports:
  - `aclnnKvCacheBlockGather`
  - `aclnnKvCacheBlockGatherGetWorkspaceSize`
- A full `SOC_VERSION=ascend910b` root CMake build reaches kernel compilation
  but fails in unrelated MLA bf16 kernel code:
  `unknown type name 'bfloat16_t'`.
- A minimal `SOC_VERSION=ascend310p3` root CMake build can build
  `vllm_ascend_C`, but this path does not register:
  - `torch.ops._C_ascend.swap_blocks_batch`
  - `torch.ops._C_ascend.kv_cache_block_gather`

Because `swap_blocks_batch` is not registered in the minimal build, the
worker-local runner correctly refuses to proceed.

## Interpretation

This is now an engineering/build-path blocker, not an experimental feasibility
blocker:

- raw mapped-host gather already works in the C++ matrix;
- worker-local copy baseline already works through a manually built extension;
- worker-local mapped backend has a corrected call shape but still needs a
  910B extension build that registers both `swap_blocks_batch` and
  `kv_cache_block_gather`.

## Next Step

Use the previously successful manual CMake build environment, or repair the
current container's 910B root build by addressing the unrelated MLA bf16 compile
failure. Once both torch ops are registered, rerun the same smoke and then
expand to mapped-vs-copy H2D matrix cases.
