# Device KV Gather Quick Raw Matrix Observations

Generated: 2026-06-19 Asia/Shanghai

Run directory:
`branch_development_notes/work/raw-matrix-quick-20260619-073434`

## Run Scope

| Item | Value |
| --- | --- |
| Branch | `experiment/device-kv-gather` |
| Git SHA | `dadc39ae2e09cce504a9786281fb5498ea3ffe55` |
| Device | NPU 0 |
| Binary | `/tmp/kv_cache_block_gather_benchmark` |
| Warmup / iters | 1 / 3 |
| Cases | 52 |
| Result rows | 208 |
| Status | 208 pass, 0 fail |
| Backends | mapped-host gather op, page copy, contiguous copy, HBM gather |

This is a quick directional run. Because each point used only 3 measured
iterations, noisy copy-path outliers should be rechecked before making hard
threshold decisions.

## Main Observations

1. The raw mapped-host gather path is usable and stable enough to continue the
   experiment plan. All size, count, and locality cases passed.
2. For random sparse loads, mapped-host gather is dramatically faster than
   page-wise `aclrtMemcpyAsync/page`. At 4KB x 1024 blocks, mapped gather took
   0.394 ms while page copy took 47.180 ms in this quick run.
3. Mapped gather reaches roughly 10-11 GB/s once the transfer is large enough.
   In the fragment-size sweep, it rose from 5.01 GB/s at 128B fragments to
   10.64-11.00 GB/s from 4KB through 1MB fragments.
4. Count sweep shows launch/index overhead dominates tiny transfers, then
   throughput saturates. For 4KB fragments, mapped gather improved from
   0.25 GB/s at 1 block to 10.34 GB/s at 512 blocks and 10.83 GB/s at 2048
   blocks.
5. Large contiguous copies can still beat mapped gather. In the 1MB x 1024
   point, contiguous copy reached 13.62 GB/s versus 11.00 GB/s for mapped
   gather, which supports a backend policy that keeps bulk contiguous reloads
   on the copy path.
6. HBM gather is much faster than host gather and should be treated only as an
   upper/reference backend for kernel/index overhead, not as a host-transfer
   replacement.
7. Locality pattern changes had little effect on mapped gather for the tested
   points. For 4KB x 1024 blocks, mapped gather stayed in 9.45-9.59 GB/s across
   sequential, stride, and random source/destination patterns.
8. Some page/contiguous copy measurements are noisy with `iters=3`; repeat with
   higher iteration counts before deriving exact crossover thresholds.

## Fragment Size Sweep

All rows use 1024 selected blocks with random source and destination patterns.

| Fragment bytes | Mapped GB/s | Mapped mean ms | Page mean ms | Contig GB/s | Contig mean ms | HBM GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 5.01 | 0.026 | 28.999 | 1.00 | 0.131 | 5.96 |
| 512 | 8.95 | 0.059 | 31.731 | 6.94 | 0.076 | 29.09 |
| 1024 | 9.57 | 0.110 | 23.090 | 0.11 | 9.722 | 58.73 |
| 4096 | 10.64 | 0.394 | 47.180 | 0.46 | 9.034 | 204.47 |
| 16384 | 10.84 | 1.547 | 142.375 | 5.79 | 2.899 | 588.26 |
| 65536 | 10.97 | 6.117 | 75.593 | 7.02 | 9.560 | 972.87 |
| 1048576 | 11.00 | 97.609 | 483.470 | 13.62 | 78.821 | 450.06 |

## Fragment Count Sweep

Mapped gather throughput by fragment size and selected block count:

| Fragment bytes | Blocks | Mapped GB/s | Mapped mean ms | Page mean ms | Contig mean ms | HBM GB/s |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 1 | 0.25 | 0.016 | 0.053 | 0.031 | 0.25 |
| 4096 | 8 | 1.07 | 0.031 | 0.640 | 0.038 | 2.06 |
| 4096 | 32 | 5.62 | 0.023 | 1.026 | 0.060 | 7.11 |
| 4096 | 128 | 8.73 | 0.060 | 3.636 | 0.139 | 33.24 |
| 4096 | 512 | 10.34 | 0.203 | 20.790 | 3.505 | 114.14 |
| 4096 | 2048 | 10.83 | 0.774 | 62.312 | 3.279 | 320.26 |
| 16384 | 1 | 0.74 | 0.022 | 0.038 | 7.764 | 1.12 |
| 16384 | 8 | 4.33 | 0.030 | 3.412 | 0.107 | 15.12 |
| 16384 | 32 | 8.92 | 0.059 | 0.925 | 2.171 | 34.06 |
| 16384 | 128 | 10.39 | 0.202 | 21.913 | 3.892 | 137.19 |
| 16384 | 512 | 10.78 | 0.778 | 124.146 | 10.309 | 381.88 |
| 16384 | 2048 | 10.94 | 3.068 | 98.868 | 33.309 | 795.38 |
| 65536 | 1 | 2.49 | 0.026 | 0.253 | 0.040 | 4.56 |
| 65536 | 8 | 7.49 | 0.070 | 6.380 | 0.107 | 40.29 |
| 65536 | 32 | 9.45 | 0.222 | 3.661 | 2.815 | 122.40 |
| 65536 | 128 | 11.25 | 0.746 | 73.744 | 21.874 | 539.81 |
| 65536 | 512 | 11.34 | 2.959 | 261.908 | 19.877 | 882.70 |
| 65536 | 2048 | 9.86 | 13.618 | 904.099 | 34.134 | 555.86 |

## Locality Sweep Summary

Each row covers all tested source/destination pattern combinations at
1024 selected blocks.

| Fragment bytes | Backend | GB/s range | Mean ms range |
| ---: | --- | ---: | ---: |
| 4096 | mapped-host gather op | 9.45-9.59 | 0.437-0.444 |
| 4096 | aclrtMemcpyAsync/page | 0.11-0.16 | 26.863-37.956 |
| 4096 | aclrtMemcpyAsync/contig | 1.32-1.93 | 2.168-3.172 |
| 16384 | mapped-host gather op | 9.29-9.82 | 1.708-1.806 |
| 16384 | aclrtMemcpyAsync/page | 0.08-0.46 | 36.416-211.350 |
| 16384 | aclrtMemcpyAsync/contig | 0.63-3.52 | 4.769-26.552 |
| 65536 | mapped-host gather op | 9.38-9.41 | 7.128-7.154 |
| 65536 | aclrtMemcpyAsync/page | 0.46-1.74 | 38.664-146.609 |
| 65536 | aclrtMemcpyAsync/contig | 1.66-2.18 | 30.766-40.471 |

## Next Actions

1. Repeat the raw matrix with higher iteration counts, preferably
   `--iters 10` or `--iters 30`, to stabilize copy-path percentiles.
2. Add contiguous-run-length cases so the policy can distinguish scattered
   sparse reloads from large coalesced reloads.
3. Add the worker-local transfer runner around `CpuNpuOffloadingHandler` for
   H2D, D2H, and bidirectional copy baseline.
4. Move mapped H2D selection into the worker-local offload path after the copy
   baseline is measured.
