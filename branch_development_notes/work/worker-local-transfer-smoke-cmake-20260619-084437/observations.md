# Worker-Local Transfer Smoke Observations

Generated: 2026-06-19 Asia/Shanghai

Run directory:
`branch_development_notes/work/worker-local-transfer-smoke-cmake-20260619-084437`

## Run Scope

| Item | Value |
| --- | --- |
| Branch | `experiment/device-kv-gather` |
| Git SHA | `dadc39ae2e09cce504a9786281fb5498ea3ffe55` |
| Device | NPU 0 |
| Build method | Manual CMake build of `vllm_ascend_C` in Docker |
| Custom ACLNN full build | Skipped for this smoke |
| Block bytes | 4096 |
| Selected blocks | 8 |
| Warmup / iters | 1 / 3 |
| Directions | H2D, D2H, bidirectional |
| Status | pass |

## Build Notes

The normal editable install path failed before measurement because
`csrc/third_party/catlass` is empty and the git index still contains a
`tmp/cann-stack` gitlink that has no `.gitmodules` entry. That causes
`git submodule update --init --recursive` to fail during `build_aclnn.sh`.

For this worker-local copy smoke, the full ACLNN custom-op build is not needed.
The smoke therefore used a manual CMake build of `vllm_ascend_C`, which
successfully registered:

- `torch.ops._C_ascend.swap_blocks_batch`
- `torch.ops._C_ascend.kv_cache_block_gather`

## Result Summary

| Direction | Bytes per transfer | Mean ms | p95 ms | p99 ms | GB/s | Status |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| h2d | 65536 | 0.334 | 0.371 | 0.376 | 0.20 | pass |
| d2h | 65536 | 0.399 | 0.593 | 0.619 | 0.16 | pass |
| bidirectional_d2h | 65536 | 0.358 | 0.469 | 0.484 | 0.18 | pass |
| bidirectional_h2d | 65536 | 0.381 | 0.488 | 0.499 | 0.17 | pass |
| bidirectional_combined_wall | 131072 | 1.093 | 1.186 | 1.196 | 0.12 | pass |

This is a smoke result, not a throughput conclusion. The payload is only
64 KiB per direction, so fixed launch and synchronization overhead dominates.

## Interpretation

1. The worker-local copy path can now be exercised outside full serving through
   `branch_development_notes/tools/bench_cpu_npu_offload_transfer.py`.
2. H2D, D2H, and bidirectional scheduling all complete with `TransferResult`
   timing populated.
3. The next useful run should increase payload size and selected block count
   before comparing against raw mapped-host gather.

## Next Actions

1. Add a small worker-local matrix configuration under
   `branch_development_notes/benchmarks/device_kv_gather`.
2. Run H2D/D2H/bidirectional copy baseline over larger payloads, for example
   4KB/16KB/64KB fragments and 32/128/512 selected blocks.
3. Fix or quarantine the stale `tmp/cann-stack` gitlink before relying on the
   normal full editable install path for repeatable Docker validation.
