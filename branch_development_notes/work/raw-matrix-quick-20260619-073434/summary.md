# Device KV Gather Matrix Results

- git_branch: `experiment/device-kv-gather`
- git_sha: `dadc39ae2e09cce504a9786281fb5498ea3ffe55`
- binary: `/tmp/kv_cache_block_gather_benchmark`
- dry_run: `False`

| case_name | backend | fragment_bytes | selected_blocks | src_pattern | dst_pattern | mean_ms | p50_ms | p95_ms | p99_ms | gbps | status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| fragment_size_sweep | mapped-host gather op | 128 | 1024 | random | random | 0.026 | 0.022 | 0.034 | 0.035 | 5.01 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/page | 128 | 1024 | random | random | 28.999 | 29.067 | 29.476 | 29.512 | 0.0 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/contig | 128 | 1024 | random | random | 0.131 | 0.056 | 0.264 | 0.282 | 1.0 | pass |
| fragment_size_sweep | HBM gather op | 128 | 1024 | random | random | 0.022 | 0.017 | 0.03 | 0.031 | 5.96 | pass |
| fragment_size_sweep | mapped-host gather op | 512 | 1024 | random | random | 0.059 | 0.056 | 0.065 | 0.066 | 8.95 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/page | 512 | 1024 | random | random | 31.731 | 31.353 | 33.166 | 33.327 | 0.02 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/contig | 512 | 1024 | random | random | 0.076 | 0.072 | 0.082 | 0.083 | 6.94 | pass |
| fragment_size_sweep | HBM gather op | 512 | 1024 | random | random | 0.018 | 0.017 | 0.02 | 0.02 | 29.09 | pass |
| fragment_size_sweep | mapped-host gather op | 1024 | 1024 | random | random | 0.11 | 0.113 | 0.113 | 0.113 | 9.57 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/page | 1024 | 1024 | random | random | 23.09 | 19.051 | 30.062 | 31.041 | 0.05 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/contig | 1024 | 1024 | random | random | 9.722 | 11.109 | 16.425 | 16.897 | 0.11 | pass |
| fragment_size_sweep | HBM gather op | 1024 | 1024 | random | random | 0.018 | 0.017 | 0.02 | 0.02 | 58.73 | pass |
| fragment_size_sweep | mapped-host gather op | 4096 | 1024 | random | random | 0.394 | 0.39 | 0.402 | 0.403 | 10.64 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | random | random | 47.18 | 47.155 | 47.467 | 47.495 | 0.09 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | random | random | 9.034 | 8.15 | 11.178 | 11.447 | 0.46 | pass |
| fragment_size_sweep | HBM gather op | 4096 | 1024 | random | random | 0.021 | 0.018 | 0.025 | 0.026 | 204.47 | pass |
| fragment_size_sweep | mapped-host gather op | 16384 | 1024 | random | random | 1.547 | 1.538 | 1.566 | 1.569 | 10.84 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | random | random | 142.375 | 141.954 | 143.491 | 143.628 | 0.12 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | random | random | 2.899 | 3.194 | 3.233 | 3.236 | 5.79 | pass |
| fragment_size_sweep | HBM gather op | 16384 | 1024 | random | random | 0.029 | 0.028 | 0.029 | 0.029 | 588.26 | pass |
| fragment_size_sweep | mapped-host gather op | 65536 | 1024 | random | random | 6.117 | 6.114 | 6.124 | 6.125 | 10.97 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | random | random | 75.593 | 31.334 | 150.915 | 161.544 | 0.89 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | random | random | 9.56 | 10.687 | 10.73 | 10.733 | 7.02 | pass |
| fragment_size_sweep | HBM gather op | 65536 | 1024 | random | random | 0.069 | 0.069 | 0.07 | 0.07 | 972.87 | pass |
| fragment_size_sweep | mapped-host gather op | 1048576 | 1024 | random | random | 97.609 | 97.59 | 97.658 | 97.664 | 11.0 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/page | 1048576 | 1024 | random | random | 483.47 | 421.024 | 890.62 | 932.362 | 2.22 | pass |
| fragment_size_sweep | aclrtMemcpyAsync/contig | 1048576 | 1024 | random | random | 78.821 | 79.711 | 80.762 | 80.856 | 13.62 | pass |
| fragment_size_sweep | HBM gather op | 1048576 | 1024 | random | random | 2.386 | 2.388 | 2.394 | 2.395 | 450.06 | pass |
| fragment_count_sweep | mapped-host gather op | 4096 | 1 | random | random | 0.016 | 0.009 | 0.031 | 0.033 | 0.25 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 4096 | 1 | random | random | 0.053 | 0.049 | 0.064 | 0.066 | 0.08 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 4096 | 1 | random | random | 0.031 | 0.03 | 0.033 | 0.034 | 0.13 | pass |
| fragment_count_sweep | HBM gather op | 4096 | 1 | random | random | 0.017 | 0.017 | 0.025 | 0.025 | 0.25 | pass |
| fragment_count_sweep | mapped-host gather op | 4096 | 8 | random | random | 0.031 | 0.019 | 0.052 | 0.055 | 1.07 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 4096 | 8 | random | random | 0.64 | 0.255 | 1.319 | 1.414 | 0.05 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 4096 | 8 | random | random | 0.038 | 0.038 | 0.04 | 0.041 | 0.87 | pass |
| fragment_count_sweep | HBM gather op | 4096 | 8 | random | random | 0.016 | 0.015 | 0.017 | 0.017 | 2.06 | pass |
| fragment_count_sweep | mapped-host gather op | 4096 | 32 | random | random | 0.023 | 0.019 | 0.032 | 0.033 | 5.62 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 4096 | 32 | random | random | 1.026 | 1.005 | 1.068 | 1.073 | 0.13 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 4096 | 32 | random | random | 0.06 | 0.062 | 0.063 | 0.063 | 2.2 | pass |
| fragment_count_sweep | HBM gather op | 4096 | 32 | random | random | 0.018 | 0.019 | 0.02 | 0.02 | 7.11 | pass |
| fragment_count_sweep | mapped-host gather op | 4096 | 128 | random | random | 0.06 | 0.054 | 0.072 | 0.073 | 8.73 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 4096 | 128 | random | random | 3.636 | 3.654 | 3.709 | 3.714 | 0.14 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 4096 | 128 | random | random | 0.139 | 0.137 | 0.145 | 0.146 | 3.78 | pass |
| fragment_count_sweep | HBM gather op | 4096 | 128 | random | random | 0.016 | 0.016 | 0.021 | 0.021 | 33.24 | pass |
| fragment_count_sweep | mapped-host gather op | 4096 | 512 | random | random | 0.203 | 0.199 | 0.212 | 0.213 | 10.34 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 4096 | 512 | random | random | 20.79 | 20.747 | 21.145 | 21.18 | 0.1 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 4096 | 512 | random | random | 3.505 | 3.08 | 4.419 | 4.538 | 0.6 | pass |
| fragment_count_sweep | HBM gather op | 4096 | 512 | random | random | 0.018 | 0.014 | 0.026 | 0.028 | 114.14 | pass |
| fragment_count_sweep | mapped-host gather op | 4096 | 2048 | random | random | 0.774 | 0.77 | 0.784 | 0.785 | 10.83 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 4096 | 2048 | random | random | 62.312 | 62.342 | 62.946 | 63.0 | 0.13 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 4096 | 2048 | random | random | 3.279 | 3.138 | 3.883 | 3.949 | 2.56 | pass |
| fragment_count_sweep | HBM gather op | 4096 | 2048 | random | random | 0.026 | 0.023 | 0.031 | 0.032 | 320.26 | pass |
| fragment_count_sweep | mapped-host gather op | 16384 | 1 | random | random | 0.022 | 0.021 | 0.032 | 0.033 | 0.74 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 16384 | 1 | random | random | 0.038 | 0.036 | 0.042 | 0.043 | 0.43 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 16384 | 1 | random | random | 7.764 | 0.055 | 20.882 | 22.733 | 0.0 | pass |
| fragment_count_sweep | HBM gather op | 16384 | 1 | random | random | 0.015 | 0.016 | 0.019 | 0.02 | 1.12 | pass |
| fragment_count_sweep | mapped-host gather op | 16384 | 8 | random | random | 0.03 | 0.022 | 0.044 | 0.046 | 4.33 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 16384 | 8 | random | random | 3.412 | 0.513 | 8.477 | 9.185 | 0.04 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 16384 | 8 | random | random | 0.107 | 0.112 | 0.129 | 0.13 | 1.22 | pass |
| fragment_count_sweep | HBM gather op | 16384 | 8 | random | random | 0.009 | 0.006 | 0.014 | 0.015 | 15.12 | pass |
| fragment_count_sweep | mapped-host gather op | 16384 | 32 | random | random | 0.059 | 0.054 | 0.068 | 0.069 | 8.92 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 16384 | 32 | random | random | 0.925 | 0.885 | 1.032 | 1.045 | 0.57 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 16384 | 32 | random | random | 2.171 | 0.682 | 5.134 | 5.53 | 0.24 | pass |
| fragment_count_sweep | HBM gather op | 16384 | 32 | random | random | 0.015 | 0.016 | 0.023 | 0.023 | 34.06 | pass |
| fragment_count_sweep | mapped-host gather op | 16384 | 128 | random | random | 0.202 | 0.199 | 0.209 | 0.21 | 10.39 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 16384 | 128 | random | random | 21.913 | 21.823 | 22.176 | 22.208 | 0.1 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 16384 | 128 | random | random | 3.892 | 5.152 | 5.261 | 5.271 | 0.54 | pass |
| fragment_count_sweep | HBM gather op | 16384 | 128 | random | random | 0.015 | 0.011 | 0.023 | 0.024 | 137.19 | pass |
| fragment_count_sweep | mapped-host gather op | 16384 | 512 | random | random | 0.778 | 0.774 | 0.788 | 0.789 | 10.78 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 16384 | 512 | random | random | 124.146 | 124.164 | 124.299 | 124.311 | 0.07 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 16384 | 512 | random | random | 10.309 | 10.295 | 10.495 | 10.513 | 0.81 | pass |
| fragment_count_sweep | HBM gather op | 16384 | 512 | random | random | 0.022 | 0.018 | 0.029 | 0.03 | 381.88 | pass |
| fragment_count_sweep | mapped-host gather op | 16384 | 2048 | random | random | 3.068 | 3.061 | 3.082 | 3.084 | 10.94 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 16384 | 2048 | random | random | 98.868 | 99.967 | 100.03 | 100.036 | 0.34 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 16384 | 2048 | random | random | 33.309 | 30.238 | 54.712 | 56.887 | 1.01 | pass |
| fragment_count_sweep | HBM gather op | 16384 | 2048 | random | random | 0.042 | 0.042 | 0.043 | 0.043 | 795.38 | pass |
| fragment_count_sweep | mapped-host gather op | 65536 | 1 | random | random | 0.026 | 0.025 | 0.036 | 0.037 | 2.49 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 65536 | 1 | random | random | 0.253 | 0.059 | 0.597 | 0.644 | 0.26 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 65536 | 1 | random | random | 0.04 | 0.04 | 0.043 | 0.043 | 1.63 | pass |
| fragment_count_sweep | HBM gather op | 65536 | 1 | random | random | 0.014 | 0.015 | 0.019 | 0.02 | 4.56 | pass |
| fragment_count_sweep | mapped-host gather op | 65536 | 8 | random | random | 0.07 | 0.062 | 0.084 | 0.086 | 7.49 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 65536 | 8 | random | random | 6.38 | 7.953 | 9.108 | 9.21 | 0.08 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 65536 | 8 | random | random | 0.107 | 0.111 | 0.119 | 0.12 | 4.92 | pass |
| fragment_count_sweep | HBM gather op | 65536 | 8 | random | random | 0.013 | 0.007 | 0.023 | 0.025 | 40.29 | pass |
| fragment_count_sweep | mapped-host gather op | 65536 | 32 | random | random | 0.222 | 0.218 | 0.229 | 0.23 | 9.45 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 65536 | 32 | random | random | 3.661 | 4.439 | 4.993 | 5.043 | 0.57 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 65536 | 32 | random | random | 2.815 | 2.437 | 5.207 | 5.453 | 0.75 | pass |
| fragment_count_sweep | HBM gather op | 65536 | 32 | random | random | 0.017 | 0.016 | 0.025 | 0.026 | 122.4 | pass |
| fragment_count_sweep | mapped-host gather op | 65536 | 128 | random | random | 0.746 | 0.741 | 0.756 | 0.758 | 11.25 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 65536 | 128 | random | random | 73.744 | 70.2 | 84.19 | 85.434 | 0.11 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 65536 | 128 | random | random | 21.874 | 26.993 | 28.352 | 28.472 | 0.38 | pass |
| fragment_count_sweep | HBM gather op | 65536 | 128 | random | random | 0.016 | 0.015 | 0.016 | 0.016 | 539.81 | pass |
| fragment_count_sweep | mapped-host gather op | 65536 | 512 | random | random | 2.959 | 2.962 | 2.964 | 2.964 | 11.34 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 65536 | 512 | random | random | 261.908 | 241.267 | 311.563 | 317.811 | 0.13 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 65536 | 512 | random | random | 19.877 | 21.563 | 28.961 | 29.619 | 1.69 | pass |
| fragment_count_sweep | HBM gather op | 65536 | 512 | random | random | 0.038 | 0.038 | 0.038 | 0.039 | 882.7 | pass |
| fragment_count_sweep | mapped-host gather op | 65536 | 2048 | random | random | 13.618 | 13.624 | 13.625 | 13.625 | 9.86 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/page | 65536 | 2048 | random | random | 904.099 | 891.467 | 928.903 | 932.231 | 0.15 | pass |
| fragment_count_sweep | aclrtMemcpyAsync/contig | 65536 | 2048 | random | random | 34.134 | 22.137 | 54.626 | 57.514 | 3.93 | pass |
| fragment_count_sweep | HBM gather op | 65536 | 2048 | random | random | 0.241 | 0.241 | 0.243 | 0.243 | 555.86 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | sequential | sequential | 0.439 | 0.431 | 0.454 | 0.456 | 9.56 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | sequential | sequential | 26.919 | 26.98 | 27.357 | 27.391 | 0.16 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | sequential | sequential | 2.168 | 2.027 | 2.453 | 2.491 | 1.93 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | sequential | sequential | 0.022 | 0.017 | 0.029 | 0.031 | 194.9 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | sequential | stride | 0.438 | 0.432 | 0.452 | 0.454 | 9.57 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | sequential | stride | 27.085 | 25.872 | 29.217 | 29.514 | 0.15 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | sequential | stride | 2.319 | 2.186 | 2.612 | 2.649 | 1.81 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | sequential | stride | 0.021 | 0.017 | 0.027 | 0.028 | 199.29 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | sequential | random | 0.442 | 0.443 | 0.452 | 0.453 | 9.49 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | sequential | random | 37.956 | 37.858 | 38.28 | 38.318 | 0.11 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | sequential | random | 2.883 | 2.741 | 3.127 | 3.161 | 1.46 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | sequential | random | 0.021 | 0.018 | 0.028 | 0.029 | 195.81 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | stride | sequential | 0.439 | 0.431 | 0.454 | 0.456 | 9.55 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | stride | sequential | 26.863 | 26.981 | 27.136 | 27.15 | 0.16 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | stride | sequential | 2.332 | 2.183 | 2.667 | 2.71 | 1.8 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | stride | sequential | 0.02 | 0.017 | 0.025 | 0.026 | 210.49 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | stride | stride | 0.437 | 0.43 | 0.451 | 0.453 | 9.59 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | stride | stride | 30.363 | 30.821 | 31.567 | 31.633 | 0.14 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | stride | stride | 3.172 | 3.001 | 3.769 | 3.837 | 1.32 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | stride | stride | 0.017 | 0.017 | 0.017 | 0.017 | 248.58 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | stride | random | 0.444 | 0.445 | 0.456 | 0.457 | 9.45 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | stride | random | 28.033 | 27.884 | 28.811 | 28.894 | 0.15 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | stride | random | 2.733 | 2.628 | 3.146 | 3.192 | 1.53 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | stride | random | 0.017 | 0.017 | 0.017 | 0.018 | 247.79 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | random | sequential | 0.442 | 0.432 | 0.459 | 0.462 | 9.5 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | random | sequential | 31.297 | 31.374 | 32.234 | 32.311 | 0.13 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | random | sequential | 2.995 | 2.733 | 3.514 | 3.583 | 1.4 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | random | sequential | 0.017 | 0.017 | 0.018 | 0.018 | 249.36 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | random | stride | 0.441 | 0.445 | 0.449 | 0.449 | 9.5 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | random | stride | 27.513 | 27.743 | 27.801 | 27.806 | 0.15 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | random | stride | 2.803 | 2.686 | 3.269 | 3.321 | 1.5 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | random | stride | 0.019 | 0.017 | 0.021 | 0.022 | 226.07 | pass |
| locality_pattern_sweep | mapped-host gather op | 4096 | 1024 | random | random | 0.438 | 0.43 | 0.452 | 0.454 | 9.58 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 4096 | 1024 | random | random | 28.118 | 28.321 | 28.899 | 28.951 | 0.15 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 4096 | 1024 | random | random | 2.763 | 2.638 | 3.219 | 3.27 | 1.52 | pass |
| locality_pattern_sweep | HBM gather op | 4096 | 1024 | random | random | 0.019 | 0.017 | 0.021 | 0.022 | 222.39 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | sequential | sequential | 1.708 | 1.701 | 1.72 | 1.721 | 9.82 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | sequential | sequential | 38.727 | 38.08 | 42.799 | 43.218 | 0.43 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | sequential | sequential | 4.769 | 4.494 | 5.253 | 5.321 | 3.52 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | sequential | sequential | 0.032 | 0.028 | 0.039 | 0.04 | 525.49 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | sequential | stride | 1.708 | 1.702 | 1.72 | 1.721 | 9.82 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | sequential | stride | 146.127 | 142.674 | 172.157 | 174.777 | 0.11 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | sequential | stride | 12.5 | 12.636 | 12.763 | 12.774 | 1.34 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | sequential | stride | 0.033 | 0.031 | 0.039 | 0.04 | 509.74 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | sequential | random | 1.792 | 1.785 | 1.804 | 1.806 | 9.36 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | sequential | random | 36.416 | 36.495 | 36.591 | 36.6 | 0.46 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | sequential | random | 25.276 | 25.137 | 25.76 | 25.815 | 0.66 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | sequential | random | 0.029 | 0.029 | 0.03 | 0.031 | 575.09 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | stride | sequential | 1.796 | 1.794 | 1.803 | 1.804 | 9.34 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | stride | sequential | 110.676 | 110.619 | 110.921 | 110.948 | 0.15 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | stride | sequential | 21.173 | 20.496 | 22.561 | 22.745 | 0.79 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | stride | sequential | 0.034 | 0.034 | 0.039 | 0.039 | 494.9 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | stride | stride | 1.797 | 1.793 | 1.808 | 1.81 | 9.34 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | stride | stride | 37.514 | 37.544 | 37.693 | 37.706 | 0.45 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | stride | stride | 25.627 | 24.875 | 27.136 | 27.337 | 0.65 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | stride | stride | 0.032 | 0.029 | 0.037 | 0.038 | 530.14 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | stride | random | 1.799 | 1.796 | 1.808 | 1.809 | 9.33 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | stride | random | 40.716 | 40.086 | 42.229 | 42.419 | 0.41 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | stride | random | 25.336 | 25.093 | 26.616 | 26.751 | 0.66 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | stride | random | 0.028 | 0.028 | 0.029 | 0.029 | 599.04 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | random | sequential | 1.806 | 1.806 | 1.81 | 1.81 | 9.29 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | random | sequential | 174.698 | 171.338 | 183.638 | 184.732 | 0.1 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | random | sequential | 25.773 | 25.378 | 26.63 | 26.741 | 0.65 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | random | sequential | 0.028 | 0.028 | 0.028 | 0.028 | 605.68 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | random | stride | 1.798 | 1.791 | 1.814 | 1.816 | 9.33 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | random | stride | 37.241 | 37.374 | 37.385 | 37.386 | 0.45 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | random | stride | 26.552 | 26.474 | 27.414 | 27.497 | 0.63 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | random | stride | 0.03 | 0.028 | 0.034 | 0.035 | 559.36 | pass |
| locality_pattern_sweep | mapped-host gather op | 16384 | 1024 | random | random | 1.797 | 1.792 | 1.807 | 1.808 | 9.34 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 16384 | 1024 | random | random | 211.35 | 210.01 | 214.898 | 215.333 | 0.08 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 16384 | 1024 | random | random | 25.113 | 24.98 | 25.724 | 25.79 | 0.67 | pass |
| locality_pattern_sweep | HBM gather op | 16384 | 1024 | random | random | 0.028 | 0.029 | 0.029 | 0.029 | 589.09 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | sequential | sequential | 7.135 | 7.138 | 7.141 | 7.141 | 9.41 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | sequential | sequential | 146.609 | 45.696 | 319.943 | 344.321 | 0.46 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | sequential | sequential | 30.766 | 22.616 | 48.951 | 51.292 | 2.18 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | sequential | sequential | 0.07 | 0.07 | 0.071 | 0.071 | 954.61 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | sequential | stride | 7.128 | 7.126 | 7.133 | 7.134 | 9.41 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | sequential | stride | 46.188 | 45.31 | 48.155 | 48.408 | 1.45 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | sequential | stride | 32.467 | 24.222 | 50.605 | 52.95 | 2.07 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | sequential | stride | 0.07 | 0.069 | 0.07 | 0.07 | 965.32 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | sequential | random | 7.133 | 7.132 | 7.134 | 7.135 | 9.41 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | sequential | random | 47.889 | 47.48 | 49.068 | 49.209 | 1.4 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | sequential | random | 33.078 | 25.435 | 50.681 | 52.926 | 2.03 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | sequential | random | 0.07 | 0.07 | 0.07 | 0.07 | 961.81 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | stride | sequential | 7.15 | 7.145 | 7.159 | 7.16 | 9.39 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | stride | sequential | 38.989 | 38.919 | 39.498 | 39.55 | 1.72 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | stride | sequential | 37.454 | 25.284 | 64.159 | 67.615 | 1.79 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | stride | sequential | 0.069 | 0.07 | 0.07 | 0.07 | 967.17 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | stride | stride | 7.153 | 7.15 | 7.165 | 7.166 | 9.38 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | stride | stride | 43.619 | 42.874 | 44.981 | 45.168 | 1.54 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | stride | stride | 40.471 | 28.454 | 66.73 | 70.132 | 1.66 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | stride | stride | 0.07 | 0.07 | 0.07 | 0.07 | 962.36 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | stride | random | 7.154 | 7.148 | 7.164 | 7.166 | 9.38 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | stride | random | 43.913 | 43.359 | 45.136 | 45.294 | 1.53 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | stride | random | 37.757 | 28.999 | 55.45 | 57.801 | 1.78 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | stride | random | 0.07 | 0.07 | 0.071 | 0.071 | 955.51 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | random | sequential | 7.145 | 7.143 | 7.148 | 7.148 | 9.39 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | random | sequential | 38.664 | 38.64 | 39.869 | 39.979 | 1.74 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | random | sequential | 37.162 | 28.189 | 55.525 | 57.955 | 1.81 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | random | sequential | 0.07 | 0.07 | 0.071 | 0.071 | 957.42 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | random | stride | 7.15 | 7.146 | 7.158 | 7.159 | 9.39 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | random | stride | 41.483 | 40.617 | 43.313 | 43.552 | 1.62 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | random | stride | 37.467 | 34.01 | 48.951 | 50.279 | 1.79 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | random | stride | 0.069 | 0.07 | 0.07 | 0.07 | 966.62 | pass |
| locality_pattern_sweep | mapped-host gather op | 65536 | 1024 | random | random | 7.154 | 7.149 | 7.167 | 7.168 | 9.38 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/page | 65536 | 1024 | random | random | 39.51 | 38.771 | 40.785 | 40.964 | 1.7 | pass |
| locality_pattern_sweep | aclrtMemcpyAsync/contig | 65536 | 1024 | random | random | 36.288 | 27.937 | 53.343 | 55.602 | 1.85 | pass |
| locality_pattern_sweep | HBM gather op | 65536 | 1024 | random | random | 0.069 | 0.069 | 0.07 | 0.07 | 967.54 | pass |
