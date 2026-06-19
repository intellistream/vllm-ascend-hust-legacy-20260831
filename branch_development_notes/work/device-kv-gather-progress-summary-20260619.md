# Device KV Gather Progress Summary

Generated: 2026-06-19 Asia/Shanghai

## Current Status

The experiment is now past the initial feasibility question.

| Area | Status | Evidence |
| --- | --- | --- |
| Custom op source | Ready for prototype work | `csrc/kv_cache_block_gather` exists and builds via single-op flow. |
| Raw mapped-host gather benchmark | First matrix completed | `branch_development_notes/work/raw-matrix-quick-20260619-073434` has 52 cases / 208 rows, all pass. |
| Raw matrix runner | Assetized | `branch_development_notes/tools/bench_device_kv_gather_matrix.py`. |
| Matrix config | Assetized | `branch_development_notes/benchmarks/device_kv_gather/matrix.json`. |
| Worker-local copy runner | Assetized and quick-baselined | `branch_development_notes/tools/bench_cpu_npu_offload_transfer.py`; smoke run under `worker-local-transfer-smoke-cmake-20260619-084437`; quick baseline under `worker-local-transfer-quick-20260619-084951`. |
| Worker-local mapped H2D backend | Quick matrix passed | 910B narrow build registers both transfer ops; mapped H2D smoke and 12-case H2D quick matrix pass. |
| Reproduction docs | Assetized | `branch_development_notes/reproduction/device-kv-gather-experiment-assets.md`. |

## Main Experimental Signal

Raw mapped-host gather is promising for sparse/random CPU-to-NPU reloads:

- 52-case quick raw matrix: all pass.
- 4KB x 1024 random blocks: mapped gather 0.394 ms vs page copy 47.180 ms.
- 4KB-1MB fragment-size sweep: mapped gather roughly 10.64-11.00 GB/s.
- 1MB x 1024 contiguous copy can still win: contiguous copy 13.62 GB/s vs
  mapped gather 11.00 GB/s.

This points toward a policy shape:

- use mapped-host gather for sparse/non-contiguous H2D reloads;
- keep contiguous/bulk copy for large coalesced reloads;
- keep D2H on copy path until a mapped scatter/write backend exists.

## Worker-Local Status

Worker-local copy transfer is now directly testable outside serving:

- manual CMake build of `vllm_ascend_C`: pass;
- `swap_blocks_batch` registration: pass;
- H2D 4KB x 8 blocks: pass;
- D2H 4KB x 8 blocks: pass;
- bidirectional H2D+D2H 4KB x 8 blocks: pass.
- quick baseline over 18 H2D/D2H/bidirectional cases: pass.

The smoke payload is only 64 KiB per direction, so it proves path viability, not
throughput. The quick baseline reaches 0.81 GB/s H2D and 0.84 GB/s D2H at
64KB x 128 blocks, but still includes Python runner and handler setup overhead.

Worker-local mapped H2D selection has been added behind the existing
`VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER` switch. The first real smoke exposed and
fixed an implementation bug: `kv_cache_block_gather` takes
`src_block_ids, src_pages, dst_block_ids, out`.

The mapped H2D smoke now passes with the narrow 910B transfer build. That build
removes the stale `tmp/cann-stack` dependency and avoids unrelated bundled
AscendC kernel compilation:

```bash
VLLM_ASCEND_ACLNN_OPS=kv_cache_block_gather \
VLLM_ASCEND_BUILD_ASCENDC_KERNELS=0 \
SOC_VERSION=ascend910b1 \
python3 -m pip install -e . --no-build-isolation --no-deps -v
```

This builds only the required ACLNN custom op and skips unrelated bundled
AscendC kernels while preserving `vllm_ascend_C` transfer op registration.

The first passing worker-local mapped H2D row is:

```text
run: branch_development_notes/work/worker-local-mapped-h2d-narrow-20260619-191240
case: h2d, mapped, 4096-byte blocks, 8 random blocks, warmup=1, iters=3
mean_ms: 0.612
p95_ms: 0.704
p99_ms: 0.716
gbps: 0.11
status: pass
```

The first worker-local H2D copy-vs-mapped quick matrix is:

```text
run: branch_development_notes/work/worker-local-h2d-quick-20260619-192411
cases: 12 copy H2D + 12 mapped H2D
block_bytes: 4096, 16384, 65536
selected_blocks: 8, 32, 128, 512
pattern: random
warmup: 3
iters: 10
status: all pass
```

Headline signal: mapped H2D is slower for the smallest 4KB/16KB x 8 cases, but
wins clearly once random selected block count grows. The 64KB x 512 case was
86.524 ms on copy and 0.637 ms on mapped in this quick run.

## Known Blockers

1. Raw matrix quick run used only 3 measured iterations, so exact copy/gather
   crossover thresholds need a higher-iteration repeat.
2. Worker-local mapped-vs-copy H2D now has a quick matrix, but still needs
   destination KV correctness checks and profiler-backed synchronization
   validation before policy thresholds are trusted.
3. The full production-style 910B build may still require proper `catlass`
   submodule initialization for unrelated fused/MLA kernels, but transfer-path
   validation no longer depends on those kernels.

## Recommended Next Step

Add destination-content validation to `bench_cpu_npu_offload_transfer.py`, then
capture matched copy and mapped msprof traces for representative small and
large random H2D cases.

The concrete git sync and next-phase plan is recorded in:

```text
branch_development_notes/work/git-sync-next-phase-plan-20260619.md
```
