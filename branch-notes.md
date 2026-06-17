# Branch Notes: experiment/device-kv-gather

Generated on 2026-06-17.

## Comparison Baseline

- Branch: `experiment/device-kv-gather`
- Branch HEAD: `1f863a1756be7d0c8fdcc287e5284620cfe6c526`
- Local `main`: `209ccbe7e1884540eb809fe677a604c7a4133b04`
- Merge base: `eedae0414f7d0f67c5d21c51a4e5ed0e517515d5`
- Compared with: `git diff main...HEAD`

This branch diverged from an older `main`. Current `main` has 5 commits that are
not in this branch, touching worker device visibility and CPU binding behavior.

## Commit Range

Commits present on this branch but not on local `main`:

- `1f863a17` Add experimental mapped-host KV cache gather
- `8c93d2c8` Fix Ascend device type check ordering
- `85927fef` Merge pull request #38 from vLLM-HUST/ci/pr-smart-ut-feedback
- `d9d233bb` Merge pull request #42 from vLLM-HUST/ws/fix-actions-26145186169-26145186140
- `165d9708` docs(changelog): note runtime visibility normalization
- `52835ae5` Merge pull request #36 from vLLM-HUST/perf/issue-22-eplb-control-plane-overhead
- `7b79a18e` fix(scripts): normalize Ascend runtime visible devices
- `34550cee` ci: add PR smart unit test workflow
- `9f544743` test(eplb): cover moe load manager dict visibility
- `46e5edf0` perf(eplb): reduce control-plane overhead

## File Summary

`git diff --stat main...HEAD` reports 14 changed files:

- `.github/workflows/README.md`
- `.github/workflows/pr_smart_ut.yaml`
- `CHANGELOG.md`
- `csrc/torch_binding.cpp`
- `scripts/use_single_ascend_env.sh`
- `tests/ut/eplb/core/test_eplb_worker.py`
- `tests/ut/eplb/test_eplb_updator.py`
- `tests/ut/worker/test_worker_multi_instance.py`
- `tests/ut/worker/test_worker_v1.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`
- `vllm_ascend/envs.py`
- `vllm_ascend/eplb/core/eplb_worker.py`
- `vllm_ascend/eplb/eplb_updator.py`
- `vllm_ascend/worker/worker.py`

Net diff size: 741 insertions, 25 deletions.

## Main Theme

The branch adds an experimental CPU-offload load path that gathers KV-cache
blocks directly from mapped host memory into NPU cache tensors. It also carries
several earlier support changes around EPLB control-plane overhead, CI unit-test
routing, Ascend runtime device visibility, and worker device-type validation.

## Experimental Mapped-Host KV Cache Gather

The core branch-specific change is in `csrc/torch_binding.cpp` and
`vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`.

What changed:

- Adds a new custom op binding:
  `torch.ops._C_ascend.kv_cache_block_gather`.
- Adds host-memory registration through `aclrtHostRegister` with
  `ACL_HOST_REGISTER_MAPPED`, then creates an ACL tensor whose data pointer is
  the mapped device-visible host pointer.
- Caches mapped host ranges in a static process-local vector so later calls can
  reuse an existing registration when the requested host range is contained in
  a previously registered range.
- Dynamically resolves `aclnnKvCacheBlockGatherGetWorkspaceSize` and
  `aclnnKvCacheBlockGather`, optionally from
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB`.
- Supports `float32`, `float16`, and `bfloat16` KV-cache tensors.
- Requires block-id tensors and the output tensor to be on NPU, with source
  pages on CPU.
- Wires CPU offload loading so, when enabled, each layer can gather selected CPU
  blocks into destination NPU blocks instead of issuing per-block tensor copies.
- Keeps the old copy loop as fallback when the op is unavailable, disabled, or
  the tensor layout/dtype checks fail.

New environment variables:

- `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER`: enables the mapped-host gather path.
  Default is off.
- `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_LIB`: optionally loads a torch extension
  that provides `_C_ascend.kv_cache_block_gather`.
- `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB`: optionally loads a custom
  opapi library used by the in-tree binding.

Notes and risks:

- This is explicitly experimental and disabled by default.
- The branch does not add a focused unit or system test for
  `kv_cache_block_gather`.
- Host mappings are cached but not unregistered in this patch. If CPU KV-cache
  backing allocations churn, the mapping list can grow for the lifetime of the
  process.
- The fast path creates NPU index tensors for each load call from
  `load_block_mapping`; benchmark impact should be measured on real workloads.
- The env var comments do not yet spell out sensitivity and valid values in the
  stricter style requested by `AGENTS.md`.
- Runtime behavior depends on CANN/opapi availability for the gather symbols.

## EPLB Control-Plane Changes

The branch includes earlier EPLB performance and correctness work.

What changed:

- `EplbUpdator.compute_and_set_moe_load` keeps the gathered tensor on device
  through the all-gather and optional `multi_stage` permute, then copies into a
  reusable CPU buffer.
- The reusable CPU buffer avoids allocating a fresh host tensor every time the
  MoE load is published into the shared dict.
- `EplbWorker.compose_expert_update_info_greedy` now skips additional planning
  work for unchanged layers after yielding an empty send/recv update.
- Expert source lookup now precomputes expert-to-source maps via `tolist()`
  rather than repeatedly constructing tensors and calling `torch.isin`.
- `pack_update_info` materializes update info once, batches tensor-to-list
  conversion with `torch.stack(...).tolist()` when tensor metadata matches, and
  falls back to per-tensor conversion otherwise.

Test coverage added:

- `tests/ut/eplb/test_eplb_updator.py` covers CPU buffer reuse and validates
  that Manager-backed shared dict readers in another process can see updates.
- `tests/ut/eplb/core/test_eplb_worker.py` covers precomputed expert sources,
  unchanged-layer behavior, and batched packing.

## Worker Device-Type Check Ordering

`vllm_ascend/worker/worker.py` moves `check_ascend_device_type()` out of
`NPUWorker.__init__` and into `_init_device()` after `torch.npu.set_device`.

Why it matters:

- The check now runs after the worker has selected and set its actual NPU.
- Unit tests were updated to assert that initialization alone does not call the
  check, while `_init_device()` does call it for explicit, auto-selected, and
  fallback device paths.

Potential merge note:

- Current `main` has later worker changes for visible-device isolation and CPU
  binding after warmup. This area is likely to need conflict-aware review when
  rebasing or merging the branch.

## Runtime Device Visibility Script

`scripts/use_single_ascend_env.sh` now normalizes Ascend visible-device
environment values.

What changed:

- Adds `normalize_visible_devices`, which trims whitespace and removes empty
  device entries from comma-separated values.
- Derives `ASCEND_RT_VISIBLE_DEVICES` from `ASCEND_VISIBLE_DEVICES` when the
  runtime-specific variable is absent.
- Normalizes a non-empty parent `ASCEND_RT_VISIBLE_DEVICES`.
- Unsets an empty inherited `ASCEND_RT_VISIBLE_DEVICES` and emits a warning.

The changelog records this as preventing local shells and benchmark wrappers
from inheriting invalid empty runtime masks.

## PR Smart UT Workflow

The branch adds `.github/workflows/pr_smart_ut.yaml` and documents the PR CI
flow in `.github/workflows/README.md`.

What changed:

- Runs on PRs to `main`, `*-dev`, and `releases/v*`.
- Triggers only for source, unit-test, dependency, and Smart UT workflow/config
  path changes.
- Uses `determine_smart_e2e_scope.py` to map changed files to scoped unit-test
  groups.
- Reuses `_optional_smart_e2e.yaml` when matching test targets exist.
- Posts a GitHub step summary with matched modules and selected test targets.

## Testing Status

I did not run the unit tests while writing this note. The branch itself adds or
updates tests for EPLB and worker device-check behavior, but the mapped-host
gather path still needs NPU/CANN validation.

Suggested targeted checks before depending on the branch:

- `pytest -sv tests/ut/eplb/test_eplb_updator.py`
- `pytest -sv tests/ut/eplb/core/test_eplb_worker.py`
- `pytest -sv tests/ut/worker/test_worker_v1.py`
- `pytest -sv tests/ut/worker/test_worker_multi_instance.py`
- A real NPU CPU-offload load test with
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=1`
- A fallback-path test with the env var enabled but the custom op unavailable

## Follow-Up Checklist

- Rebase or merge current `main` before further work; worker initialization code
  has diverged.
- Add focused coverage for the mapped-host gather path, including fallback
  behavior and dtype/layout rejection.
- Decide whether host mappings need explicit unregister/lifetime management.
- Confirm the three new environment variables satisfy the repository env-var
  review rules for defaults, valid values, and sensitivity.
- Benchmark CPU-offload load latency and memory behavior on Ascend 910B/C with
  realistic KV-cache block mappings.
