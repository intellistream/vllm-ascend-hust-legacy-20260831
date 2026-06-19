# CPU/NPU Offload Transfer Results

- git_branch: `experiment/device-kv-gather`
- git_sha: `80da4b338dd400c1c9bd3e3dab7024dbd3c781a6`
- dry_run: `False`

| direction | h2d_backend | block_bytes | selected_blocks | pattern | mean_ms | p95_ms | p99_ms | gbps | status |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| h2d | copy | 4096 | 8 | random | 0.451 | 0.528 | 0.551 | 0.15 | pass |
| h2d | copy | 4096 | 32 | random | 1.744 | 2.256 | 2.517 | 0.15 | pass |
| h2d | copy | 4096 | 128 | random | 6.405 | 7.478 | 8.095 | 0.16 | pass |
| h2d | copy | 4096 | 512 | random | 26.074 | 29.387 | 30.261 | 0.16 | pass |
| h2d | copy | 16384 | 8 | random | 0.519 | 0.707 | 0.748 | 0.51 | pass |
| h2d | copy | 16384 | 32 | random | 1.889 | 1.938 | 1.939 | 0.56 | pass |
| h2d | copy | 16384 | 128 | random | 7.199 | 7.322 | 7.330 | 0.58 | pass |
| h2d | copy | 16384 | 512 | random | 28.348 | 29.523 | 29.566 | 0.59 | pass |
| h2d | copy | 65536 | 8 | random | 1.634 | 3.647 | 4.361 | 0.64 | pass |
| h2d | copy | 65536 | 32 | random | 5.251 | 5.933 | 5.964 | 0.80 | pass |
| h2d | copy | 65536 | 128 | random | 21.846 | 28.497 | 29.937 | 0.77 | pass |
| h2d | copy | 65536 | 512 | random | 86.524 | 96.648 | 100.998 | 0.78 | pass |
