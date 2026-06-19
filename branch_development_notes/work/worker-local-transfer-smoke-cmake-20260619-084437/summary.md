# CPU/NPU Offload Transfer Results

- git_branch: `experiment/device-kv-gather`
- git_sha: `dadc39ae2e09cce504a9786281fb5498ea3ffe55`
- dry_run: `False`

| direction | block_bytes | selected_blocks | pattern | mean_ms | p95_ms | p99_ms | gbps | status |
| --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| h2d | 4096 | 8 | random | 0.334 | 0.371 | 0.376 | 0.20 | pass |
| d2h | 4096 | 8 | random | 0.399 | 0.593 | 0.619 | 0.16 | pass |
| bidirectional_d2h | 4096 | 8 | random | 0.358 | 0.469 | 0.484 | 0.18 | pass |
| bidirectional_h2d | 4096 | 8 | random | 0.381 | 0.488 | 0.499 | 0.17 | pass |
| bidirectional_combined_wall | 4096 | 8 | random | 1.093 | 1.186 | 1.196 | 0.12 | pass |
