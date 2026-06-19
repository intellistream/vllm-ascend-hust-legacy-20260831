# CPU/NPU Offload Transfer Results

- git_branch: `experiment/device-kv-gather`
- git_sha: `cce29d061dad2549f4b60098ac5ff600b4db5f38`
- dry_run: `False`

| direction | h2d_backend | block_bytes | selected_blocks | pattern | mean_ms | p95_ms | p99_ms | gbps | status |
| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- |
| h2d | mapped | 4096 | 8 | random | 0.612 | 0.704 | 0.716 | 0.11 | pass |
