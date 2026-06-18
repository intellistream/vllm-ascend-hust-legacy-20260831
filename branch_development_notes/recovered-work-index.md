# Recovered Work Index

Generated on 2026-06-18.

This is the entry point for work recovered from the overwritten
`17a682993f5393b0ef5d14f4be62acadeb42bcff` branch head.

## Preserved Anchors

Original overwritten branch head:

```text
17a682993f5393b0ef5d14f4be62acadeb42bcff
Merge pull request #67 from CubeLander/feat/device-kv-transfer-prototype
```

Local backup branch:

```text
backup/vllm-hust-device-kv-gather-before-overwrite
```

Git bundle archives:

```text
branch_development_notes/archive/17a68299-device-kv-transfer-prototype.bundle
branch_development_notes/archive/17a68299-full-overwritten-line.bundle
```

Detailed investigation:

```text
branch_development_notes/overwritten-17a68299-recovery-notes.md
```

## What Was Merged Back

### Device KV Gather Prototype

Recovered or preserved in the active tree:

```text
csrc/kv_cache_block_gather/benchmarks/kv_cache_block_gather_benchmark.cpp
csrc/kv_cache_block_gather/op_graph/
docs/source/developer_guide/feature_guide/device_kv_transfer_prototype.md
docs/source/developer_guide/feature_guide/index.md
```

The current branch keeps the newer direct smoke and reproduction notes:

```text
tools/smoke_device_kv_gather.py
branch_development_notes/device-kv-gather-reproduction.md
branch_development_notes/engineering/device-kv-gather-smoke-status.md
```

### Worker Device Isolation

Recovered into the active code path:

```text
vllm_ascend/patch/platform/patch_multiproc_executor.py
vllm_ascend/worker/worker.py
vllm_ascend/envs.py
tests/ut/worker/test_worker_v1.py
```

Purpose:

```text
Optionally narrow each multiprocessing worker's Ascend visible-device env vars
to its selected physical device, then let the worker initialize using local
device index npu:0. CPU binding, HCCL init, and health checks use the same
worker-local device mapping.
```

Related env vars:

```text
VLLM_ASCEND_WORKER_NARROW_VISIBLE_DEVICES
VLLM_ASCEND_WORKER_DEVICE_INDEX
VLLM_ASCEND_WORKER_PHYSICAL_DEVICE
```

### Issue-30 / BidKV Selector Work

Recovered into the active code path:

```text
vllm_ascend/core/victim_selector.py
vllm_ascend/ascend_config.py
vllm_ascend/profiling_config.py
vllm_ascend/platform.py
vllm_ascend/core/recompute_scheduler.py
vllm_ascend/core/scheduler_dynamic_batch.py
vllm_ascend/core/scheduler_profiling_chunk.py
tests/ut/core/test_recompute_victim_selector.py
tests/ut/core/test_utility_victim_config.py
tests/ut/core/test_victim_selector.py
tests/ut/test_ascend_config.py
tests/ut/test_profiling_config.py
docs/pr_drafts/issue-30-pr-content.md
scripts/issue30_acceptance_report.py
```

Purpose:

```text
Restore utility/BidKV-style victim selection and the corresponding config,
scheduler integration, tests, and review/acceptance materials.
```

### Miscellaneous Support Fixes

Recovered into the active tree:

```text
benchmarks/scripts/run-performance-benchmarks.sh
scripts/install_local_ascend_plugin.sh
.github/workflows/_unit_test.yaml
.github/workflows/scripts/ut_blacklist.yaml
```

Purpose:

```text
Benchmark script portability, local editable-install bootstrap, and UT
blacklist cleanup from the overwritten line.
```

## What Remains Archived Rather Than Directly Adopted

Some old branch implementation details are intentionally kept as archive
material rather than used as the active path:

```text
old csrc/kv_cache_block_gather/op_host/error_log.h
old minimal op_host/CMakeLists.txt
old CANN-stack-style graph/plugin split as the only implementation layout
old Docker smoke prose that predates the newer direct PASS reproduction note
```

Reason:

```text
The current branch already has a Docker-validated direct smoke path. We keep
the current op_host build integration and compatibility shim, while preserving
old files and commits in the Git bundle archive for future review.
```

## How To Recover More Later

Inspect the old branch:

```bash
git checkout backup/vllm-hust-device-kv-gather-before-overwrite
```

Return to current work:

```bash
git checkout experiment/device-kv-gather
```

Inspect or recover only the device KV prototype series in a scratch clone:

```bash
git clone . /tmp/recover-device-kv-transfer
cd /tmp/recover-device-kv-transfer
git fetch /home/jingyuan/workspace/vllm-ascend-hust/branch_development_notes/archive/17a68299-device-kv-transfer-prototype.bundle \
  backup/device-kv-transfer-prototype:backup/device-kv-transfer-prototype
git checkout backup/device-kv-transfer-prototype
```

Inspect or recover the full overwritten line in a scratch clone:

```bash
git clone . /tmp/recover-full-17a68299
cd /tmp/recover-full-17a68299
git fetch /home/jingyuan/workspace/vllm-ascend-hust/branch_development_notes/archive/17a68299-full-overwritten-line.bundle \
  backup/vllm-hust-device-kv-gather-before-overwrite:backup/vllm-hust-device-kv-gather-before-overwrite
git checkout backup/vllm-hust-device-kv-gather-before-overwrite
```

Expected conflict areas:

```text
csrc/kv_cache_block_gather/op_host/
csrc/build_aclnn.sh
worker device initialization
scheduler/victim-selector internals
docs that duplicate branch_development_notes
```

## Verification On This Recovery Commit

Performed in the host Python environment:

```text
python3 -m py_compile selected recovered Python modules and tests: pass
git diff --check: pass
```

Not performed on the host:

```text
pytest, because this host Python does not have pytest installed.
Docker custom-op smoke after this merge pass, because this pass mainly restored
archived support material and non-smoke-path scheduler/worker work.
```
