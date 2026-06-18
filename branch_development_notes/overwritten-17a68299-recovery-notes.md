# Overwritten 17a68299 Recovery Notes

Generated on 2026-06-18.

## What Was Preserved

The old `vLLM-HUST/experiment/device-kv-gather` branch head was:

```text
17a682993f5393b0ef5d14f4be62acadeb42bcff
Merge pull request #67 from CubeLander/feat/device-kv-transfer-prototype
2026-06-16 23:28:20 +0800
```

It has been preserved locally as:

```text
backup/vllm-hust-device-kv-gather-before-overwrite
```

Two Git bundle archives were also written under `branch_development_notes/archive/`:

```text
branch_development_notes/archive/17a68299-device-kv-transfer-prototype.bundle
branch_development_notes/archive/17a68299-full-overwritten-line.bundle
```

The first bundle contains only the 7 device KV transfer prototype commits:

```text
0001 Add experimental mapped-host KV cache gather
0002 Integrate KvCacheBlockGather custom op
0003 Fix KvCacheBlockGather tiling log include
0004 Avoid external tiling registry include for KvCacheBlockGather
0005 Add local tiling error helpers for KvCacheBlockGather
0006 Avoid KvCacheBlockGather tiling log format warnings
0007 Document Docker workflow for device KV transfer prototype
```

The second bundle contains the broader overwritten line from the old branch
base to `17a68299`, including issue-30/BidKV and worker device visibility work.

## Current Recovery Status

After the follow-up merge pass on 2026-06-18, the valuable parts of the old line
were folded back into the current `experiment/device-kv-gather` working tree:

```text
Recovered into current tree:
- standalone kv_cache_block_gather benchmark
- old op_graph/proto source layout files
- upstream-style device KV transfer prototype docs page
- issue-30 PR draft and acceptance report helper
- issue-30/BidKV victim selector, config, scheduler integration, and tests
- benchmark script portability fixes
- local install pybind11 bootstrap fix
- worker visible-device narrowing support
- worker device-index-aware CPU binding, HCCL init, and health check

Kept as archive:
- full Git bundle for the old line
- device-KV-only Git bundle
- backup branch pointing at the original overwritten merge commit
```

The current branch intentionally keeps the newer direct smoke path and
`branch_development_notes/device-kv-gather-reproduction.md` as the primary
prototype reproduction record.

## Commit Shape

`17a68299` is a merge commit:

```text
parent 1: f0ebba3bdab24b2b223613d8386ba166a37eea14
parent 2: 367b19f0b2c993841e46c6018330114da76df45f
```

The device KV transfer prototype lives on parent 2:

```text
54655797 Add experimental mapped-host KV cache gather
28316e90 Integrate KvCacheBlockGather custom op
4ab3cdf9 Fix KvCacheBlockGather tiling log include
0da5ba1d Avoid external tiling registry include for KvCacheBlockGather
44e09507 Add local tiling error helpers for KvCacheBlockGather
9466ab57 Avoid KvCacheBlockGather tiling log format warnings
367b19f0 Document Docker workflow for device KV transfer prototype
```

The broader overwritten line also includes:

```text
issue-30 / BidKV victim selector work
profiling config and profiling tests
worker visible-device isolation
CPU binding after warmup
benchmark script portability fixes
device KV transfer prototype
```

## Device KV Transfer Prototype Contents

The old line added a direct mapped-host KV gather path:

```text
csrc/torch_binding.cpp
vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py
vllm_ascend/envs.py
```

Core behavior:

- registers `torch.ops._C_ascend.kv_cache_block_gather`
- maps CPU host pages with `aclrtHostRegister(... ACL_HOST_REGISTER_MAPPED ...)`
- resolves `aclnnKvCacheBlockGatherGetWorkspaceSize` and
  `aclnnKvCacheBlockGather` dynamically
- enables the connector path with
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1`
- falls back to the existing tensor-copy path when the op or tensor layout is
  unavailable

This behavior overlaps with the current branch's prototype.

## Old Custom Op Implementation

The old line had a more source-tree-complete custom op layout than the current
minimal extraction:

```text
csrc/kv_cache_block_gather/CMakeLists.txt
csrc/kv_cache_block_gather/benchmarks/kv_cache_block_gather_benchmark.cpp
csrc/kv_cache_block_gather/op_graph/CMakeLists.txt
csrc/kv_cache_block_gather/op_graph/kv_cache_block_gather_graph_plugin.cpp
csrc/kv_cache_block_gather/op_graph/kv_cache_block_gather_proto.h
csrc/kv_cache_block_gather/op_host/CMakeLists.txt
csrc/kv_cache_block_gather/op_host/error_log.h
csrc/kv_cache_block_gather/op_host/kv_cache_block_gather_def.cpp
csrc/kv_cache_block_gather/op_host/kv_cache_block_gather_infershape.cpp
csrc/kv_cache_block_gather/op_host/kv_cache_block_gather_tiling.cpp
csrc/kv_cache_block_gather/op_kernel/kv_cache_block_gather.cpp
csrc/kv_cache_block_gather/op_kernel/kv_cache_block_gather.h
csrc/kv_cache_block_gather/op_kernel/kv_cache_block_gather_tiling_data.h
csrc/kv_cache_block_gather/op_kernel/kv_cache_block_gather_tiling_key.h
```

Pieces worth considering for re-import:

1. `benchmarks/kv_cache_block_gather_benchmark.cpp`
   - standalone ACL benchmark for the custom op
   - supports `num-pages`, `selected-blocks`, `elems-per-block`,
     source/destination access patterns, warmup, and iteration count
   - explicitly exercises page-aligned host allocation, host registration,
     mapped device address, ACL tensor creation, and op execution

2. `op_graph/`
   - keeps graph plugin/proto material separated from `op_host`
   - current branch moved the graph plugin into `op_host` for a simpler build
   - old layout may be closer to the standalone CANN stack convention

3. `op_host/error_log.h`
   - local tiling/log compatibility helpers
   - current branch uses `kv_cache_block_gather_compat.h` instead

4. The CANN warning fixes
   - log include fixes
   - removing external tiling registry dependency
   - local tiling error helper
   - log format warning cleanup

Many of these fixes have equivalent intent in the current branch, but the
benchmark and old docs are not currently present.

## Old Prototype Documentation

The old line added:

```text
docs/source/developer_guide/feature_guide/device_kv_transfer_prototype.md
```

It documents:

- purpose of the prototype
- Docker image:
  `quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler`
- build flow with `COMPILE_CUSTOM_KERNELS=1` and `SOC_VERSION=ascend910b1`
- focused custom-op build:
  `bash csrc/build.sh -n kv_cache_block_gather -c ascend910b`
- runtime flag:
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1`
- optional opapi override:
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB=/path/to/libcust_opapi.so`
- Python-only branch smoke
- standalone benchmark compile/run command

This overlaps with the current branch note:

```text
branch_development_notes/device-kv-gather-reproduction.md
```

but the old doc has a more upstream-docs-facing format and includes benchmark
instructions that the current reproduction note does not fully preserve.

## Worker / Multiproc Device Visibility Work

The old broader line also contains worker process device visibility handling:

```text
vllm_ascend/patch/platform/patch_multiproc_executor.py
vllm_ascend/worker/worker.py
tests/ut/worker/test_worker_v1.py
```

Notable concepts:

- `VLLM_ASCEND_WORKER_NARROW_VISIBLE_DEVICES`
- `VLLM_ASCEND_WORKER_DEVICE_INDEX`
- `VLLM_ASCEND_WORKER_PHYSICAL_DEVICE`
- per-worker narrowing of `ASCEND_RT_VISIBLE_DEVICES`,
  `ASCEND_VISIBLE_DEVICES`, and `NPU_VISIBLE_DEVICES`
- preserving original visible-device env values under
  `VLLM_ASCEND_ORIGINAL_*`
- starting each worker process under a temporary narrowed environment
- worker-side re-application in `worker_main`
- worker health check using the physical device id

This is not directly part of `kv_cache_block_gather`, but it matters for
reliable multi-process NPU isolation and avoids some false device selection
problems in profiling or worker startup.

## Issue-30 / BidKV Work In The Same Old Line

The full overwritten line also carried issue-30/BidKV related work:

```text
vllm_ascend/core/victim_selector.py
vllm_ascend/core/recompute_scheduler.py
vllm_ascend/core/scheduler_dynamic_batch.py
vllm_ascend/core/scheduler_profiling_chunk.py
vllm_ascend/ascend_config.py
vllm_ascend/profiling_config.py
tests/ut/core/test_recompute_victim_selector.py
tests/ut/core/test_utility_victim_config.py
tests/ut/core/test_victim_selector.py
tests/ut/test_profiling_config.py
docs/pr_drafts/issue-30-pr-content.md
scripts/issue30_acceptance_report.py
```

This work is mostly orthogonal to device KV gather, but it is real engineering
content and is preserved in the full Git bundle archive.

## Recovery Options

To inspect the preserved old branch:

```bash
git checkout backup/vllm-hust-device-kv-gather-before-overwrite
```

To return to the current branch:

```bash
git checkout experiment/device-kv-gather
```

To inspect or recover only the device KV transfer prototype bundle on a scratch
branch:

```bash
git clone . /tmp/recover-device-kv-transfer
cd /tmp/recover-device-kv-transfer
git fetch /home/jingyuan/workspace/vllm-ascend-hust/branch_development_notes/archive/17a68299-device-kv-transfer-prototype.bundle \
  backup/device-kv-transfer-prototype:backup/device-kv-transfer-prototype
git checkout backup/device-kv-transfer-prototype
```

To generate patches from the recovered bundle branch:

```bash
git format-patch 546557970b376dce10df5978075a512ef100771a^..backup/device-kv-transfer-prototype
```

Expect conflicts if applying those patches directly to the current branch
because the current branch already contains a different integration of the same
op. The most useful conflict-resolution targets are:

```text
keep current Docker PASS smoke path
consider importing the old standalone benchmark
consider importing the upstream-style docs page
compare old op_graph layout with current simplified op_host layout
preserve current branch_development_notes
```

To inspect or recover the full overwritten line bundle:

```bash
git clone . /tmp/recover-full-17a68299
cd /tmp/recover-full-17a68299
git fetch /home/jingyuan/workspace/vllm-ascend-hust/branch_development_notes/archive/17a68299-full-overwritten-line.bundle \
  backup/vllm-hust-device-kv-gather-before-overwrite:backup/vllm-hust-device-kv-gather-before-overwrite
git checkout backup/vllm-hust-device-kv-gather-before-overwrite
```

This will likely conflict more heavily because the full line includes scheduler,
profiling, worker, and CI changes outside the KV gather prototype.

## Recommendation

Do not blindly merge the full old line back into the current branch.

Suggested salvage order:

1. Keep the local backup branches and Git bundle archives.
2. Cherry-pick or manually import the old standalone
   `kv_cache_block_gather_benchmark.cpp`.
3. Fold the useful parts of
   `docs/source/developer_guide/feature_guide/device_kv_transfer_prototype.md`
   into the current reproduction note or an upstream-facing docs page.
4. Compare old `op_graph/` layout against the current minimal custom-op layout
   only if the current layout causes packaging or review friction.
5. Treat worker visible-device isolation and issue-30/BidKV work as separate
   recovery topics.
