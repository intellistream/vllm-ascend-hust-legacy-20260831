# CPU/NPU Offload Transfer Results

- git_branch: `experiment/device-kv-gather`
- git_sha: `dadc39ae2e09cce504a9786281fb5498ea3ffe55`
- dry_run: `False`

| direction | block_bytes | selected_blocks | pattern | mean_ms | p95_ms | p99_ms | gbps | status |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| h2d | 4096 | 32 | random | 1.421 | 1.927 | 1.933 | 0.18 | pass |
| d2h | 4096 | 32 | random | 1.690 | 1.914 | 1.918 | 0.16 | pass |
| bidirectional_d2h | 4096 | 32 | random | 2.180 | 2.533 | 2.603 | 0.12 | pass |
| bidirectional_h2d | 4096 | 32 | random | 2.449 | 2.683 | 2.696 | 0.11 | pass |
| bidirectional_combined_wall | 4096 | 32 | random | 3.022 | 3.351 | 3.414 | 0.17 | pass |
| h2d | 4096 | 128 | random | 6.150 | 6.813 | 6.820 | 0.17 | pass |
| d2h | 4096 | 128 | random | 6.013 | 6.774 | 6.866 | 0.17 | pass |
| bidirectional_d2h | 4096 | 128 | random | 10.842 | 11.770 | 11.848 | 0.10 | pass |
| bidirectional_h2d | 4096 | 128 | random | 12.050 | 12.403 | 12.438 | 0.09 | pass |
| bidirectional_combined_wall | 4096 | 128 | random | 12.560 | 13.030 | 13.067 | 0.17 | pass |
| h2d | 16384 | 32 | random | 2.973 | 6.340 | 6.840 | 0.35 | pass |
| d2h | 16384 | 32 | random | 3.438 | 4.490 | 4.535 | 0.31 | pass |
| bidirectional_d2h | 16384 | 32 | random | 6.733 | 9.102 | 9.593 | 0.16 | pass |
| bidirectional_h2d | 16384 | 32 | random | 5.447 | 6.460 | 6.681 | 0.19 | pass |
| bidirectional_combined_wall | 16384 | 32 | random | 7.604 | 9.967 | 10.449 | 0.28 | pass |
| h2d | 16384 | 128 | random | 14.782 | 15.221 | 15.292 | 0.28 | pass |
| d2h | 16384 | 128 | random | 14.565 | 15.199 | 15.297 | 0.29 | pass |
| bidirectional_d2h | 16384 | 128 | random | 20.244 | 24.841 | 24.936 | 0.21 | pass |
| bidirectional_h2d | 16384 | 128 | random | 23.186 | 24.725 | 25.016 | 0.18 | pass |
| bidirectional_combined_wall | 16384 | 128 | random | 24.608 | 26.079 | 26.177 | 0.34 | pass |
| h2d | 65536 | 32 | random | 9.001 | 12.199 | 12.640 | 0.47 | pass |
| d2h | 65536 | 32 | random | 5.887 | 8.392 | 8.585 | 0.71 | pass |
| bidirectional_d2h | 65536 | 32 | random | 14.656 | 18.209 | 18.942 | 0.29 | pass |
| bidirectional_h2d | 65536 | 32 | random | 12.643 | 18.576 | 19.361 | 0.33 | pass |
| bidirectional_combined_wall | 65536 | 32 | random | 15.563 | 19.136 | 19.873 | 0.54 | pass |
| h2d | 65536 | 128 | random | 20.601 | 22.532 | 22.754 | 0.81 | pass |
| d2h | 65536 | 128 | random | 19.883 | 21.176 | 21.490 | 0.84 | pass |
| bidirectional_d2h | 65536 | 128 | random | 38.046 | 40.771 | 41.082 | 0.44 | pass |
| bidirectional_h2d | 65536 | 128 | random | 37.761 | 40.708 | 40.857 | 0.44 | pass |
| bidirectional_combined_wall | 65536 | 128 | random | 39.254 | 42.009 | 42.325 | 0.85 | pass |
