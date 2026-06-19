# Worker-Local Transfer Quick Baseline Observations

Generated: 2026-06-19 Asia/Shanghai

Run directory:
`branch_development_notes/work/worker-local-transfer-quick-20260619-084951`

## Run Scope

| Item | Value |
| --- | --- |
| Branch | `experiment/device-kv-gather` |
| Git SHA | `dadc39ae2e09cce504a9786281fb5498ea3ffe55` |
| Device | NPU 0 |
| Build method | Manual CMake build of `vllm_ascend_C` in Docker |
| Directions | H2D, D2H, bidirectional |
| Block bytes | 4096, 16384, 65536 |
| Selected blocks | 32, 128 |
| Pattern | random |
| Warmup / iters | 2 / 5 |
| Cases | 18 |
| Result rows | 30 |
| Status | 30 pass, 0 fail |

## Result Highlights

Single-direction H2D/D2H:

| Block bytes | Blocks | H2D mean ms | H2D GB/s | D2H mean ms | D2H GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 32 | 1.421 | 0.18 | 1.690 | 0.16 |
| 4096 | 128 | 6.150 | 0.17 | 6.013 | 0.17 |
| 16384 | 32 | 2.973 | 0.35 | 3.438 | 0.31 |
| 16384 | 128 | 14.782 | 0.28 | 14.565 | 0.29 |
| 65536 | 32 | 9.001 | 0.47 | 5.887 | 0.71 |
| 65536 | 128 | 20.601 | 0.81 | 19.883 | 0.84 |

Bidirectional combined wall-time:

| Block bytes | Blocks | Mean ms | p95 ms | GB/s | Status |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4096 | 32 | 3.022 | 3.351 | 0.17 | pass |
| 4096 | 128 | 12.560 | 13.030 | 0.17 | pass |
| 16384 | 32 | 7.604 | 9.967 | 0.28 | pass |
| 16384 | 128 | 24.608 | 26.079 | 0.34 | pass |
| 65536 | 32 | 15.563 | 19.136 | 0.54 | pass |
| 65536 | 128 | 39.254 | 42.009 | 0.85 | pass |

## Interpretation

1. The worker-local copy baseline runner is now usable for matrix work: all H2D,
   D2H, and bidirectional cases completed and emitted timing rows.
2. Throughput improves as payload grows, reaching about 0.81 GB/s H2D and
   0.84 GB/s D2H at 64KB x 128 blocks in this quick run.
3. These numbers are not directly comparable to the raw C++ copy numbers yet:
   this path includes Python runner overhead, handler allocation per case,
   block-id expansion, pointer-array creation, stream/event bookkeeping, and
   `swap_blocks_batch` execution.
4. The next comparison should keep this worker-local runner but add a mapped
   H2D backend inside `CpuNpuOffloadingHandler`, so copy and mapped paths share
   the same Python/handler overhead.

## Next Actions

1. Avoid reallocating `CpuNpuOffloadingHandler` per case when measuring stable
   throughput; reuse a handler per shape or add an inner loop mode.
2. Add mapped H2D backend selection to `CpuNpuOffloadingHandler` for the same
   synthetic cases.
3. Repeat with larger selected-block counts, especially 512 and 2048, after the
   runner overhead is controlled.
