# Worker-Local H2D Copy vs Mapped Quick Matrix

Generated: 2026-06-19 Asia/Shanghai

## Scope

This run used the branch-local narrow 910B transfer build at:

```text
git_sha: 80da4b338dd400c1c9bd3e3dab7024dbd3c781a6
```

Build command:

```bash
VLLM_ASCEND_ACLNN_OPS=kv_cache_block_gather \
VLLM_ASCEND_BUILD_ASCENDC_KERNELS=0 \
SOC_VERSION=ascend910b1 \
python3 -m pip install -e . --no-build-isolation --no-deps
```

Both transfer ops and the experimental cleanup op were registered:

```text
has swap_blocks_batch True
has kv_cache_block_gather True
has clear True
```

Matrix:

```text
direction: h2d
pattern: random
block_bytes: 4096, 16384, 65536
selected_blocks: 8, 32, 128, 512
warmup: 3
iters: 10
```

## Result

Both backends completed all 12 cases.

The mapped backend initially failed in an earlier partial run when stale
process-global host mappings overlapped newly allocated CPU tensors. The
`clear_kv_cache_block_gather_host_mappings` op fixed the multi-case run: before
cases 2-12 the runner reported `cleared 2 mapped host ranges`.

Comparison:

| block_bytes | selected_blocks | copy_ms | mapped_ms | copy/mapped | copy_gbps | mapped_gbps |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 8 | 0.451 | 0.591 | 0.76 | 0.15 | 0.11 |
| 4096 | 32 | 1.744 | 0.596 | 2.93 | 0.15 | 0.44 |
| 4096 | 128 | 6.405 | 0.600 | 10.67 | 0.16 | 1.75 |
| 4096 | 512 | 26.074 | 0.626 | 41.64 | 0.16 | 6.70 |
| 16384 | 8 | 0.519 | 0.612 | 0.85 | 0.51 | 0.43 |
| 16384 | 32 | 1.889 | 0.615 | 3.07 | 0.56 | 1.71 |
| 16384 | 128 | 7.199 | 0.608 | 11.84 | 0.58 | 6.90 |
| 16384 | 512 | 28.348 | 0.609 | 46.55 | 0.59 | 27.55 |
| 65536 | 8 | 1.634 | 0.600 | 2.72 | 0.64 | 1.75 |
| 65536 | 32 | 5.251 | 0.592 | 8.87 | 0.80 | 7.08 |
| 65536 | 128 | 21.846 | 0.615 | 35.54 | 0.77 | 27.29 |
| 65536 | 512 | 86.524 | 0.637 | 135.89 | 0.78 | 105.39 |

## Interpretation

The worker-local mapped path now has a real multi-case signal, not just a
single smoke:

- copy H2D scales roughly with the number of random selected blocks;
- mapped H2D is nearly flat in this quick matrix;
- mapped is slower only for the smallest 4KB/16KB x 8 cases;
- mapped wins clearly once the random selected block count grows.

The very flat mapped timing should be treated as an experimental signal, not a
policy threshold yet. The next validation step should add output correctness
checks and an msprof/TraceLoom pair for matched copy and mapped cases, because
the current benchmark measures transfer timing through the handler but does not
compare the destination KV contents after each transfer.
