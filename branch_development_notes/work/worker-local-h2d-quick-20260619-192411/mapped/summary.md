# CPU/NPU Offload Transfer Results

- git_branch: `experiment/device-kv-gather`
- git_sha: `80da4b338dd400c1c9bd3e3dab7024dbd3c781a6`
- dry_run: `False`

| direction | h2d_backend | block_bytes | selected_blocks | pattern | mean_ms | p95_ms | p99_ms | gbps | status |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| h2d | mapped | 4096 | 8 | random | 0.591 | 0.607 | 0.609 | 0.11 | pass |
| h2d | mapped | 4096 | 32 | random | 0.596 | 0.608 | 0.608 | 0.44 | pass |
| h2d | mapped | 4096 | 128 | random | 0.600 | 0.618 | 0.620 | 1.75 | pass |
| h2d | mapped | 4096 | 512 | random | 0.626 | 0.655 | 0.657 | 6.70 | pass |
| h2d | mapped | 16384 | 8 | random | 0.612 | 0.643 | 0.651 | 0.43 | pass |
| h2d | mapped | 16384 | 32 | random | 0.615 | 0.643 | 0.651 | 1.71 | pass |
| h2d | mapped | 16384 | 128 | random | 0.608 | 0.627 | 0.629 | 6.90 | pass |
| h2d | mapped | 16384 | 512 | random | 0.609 | 0.629 | 0.632 | 27.55 | pass |
| h2d | mapped | 65536 | 8 | random | 0.600 | 0.621 | 0.621 | 1.75 | pass |
| h2d | mapped | 65536 | 32 | random | 0.592 | 0.601 | 0.602 | 7.08 | pass |
| h2d | mapped | 65536 | 128 | random | 0.615 | 0.769 | 0.850 | 27.29 | pass |
| h2d | mapped | 65536 | 512 | random | 0.637 | 0.769 | 0.770 | 105.39 | pass |
