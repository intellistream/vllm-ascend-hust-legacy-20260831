# Git Sync And Next Phase Plan

Date: 2026-06-19 Asia/Shanghai

## Current Git Hygiene

The working tree now separates durable experiment assets from transient run
artifacts.

Ignored local-only artifacts:

- `branch_development_notes/external/vllm-hust-perf-analyzer/**`
- `branch_development_notes/work/**/docker.log`
- `branch_development_notes/work/raw-matrix-*/run_*.stdout`
- `branch_development_notes/work/raw-matrix-*/run_*.stderr`
- `branch_development_notes/work/traceloom-*/db*.traceloom_augmented.db`
- `branch_development_notes/work/traceloom-kickstart-20260619/`

The ignored files remain on disk for local debugging, but they should not be
part of the branch history.

## Suggested Commit Split

### Commit 1: Experiment Assets And Reproduction Notes

Scope:

- `branch_development_notes/benchmarks/device_kv_gather/matrix.json`
- `branch_development_notes/tools/bench_device_kv_gather_matrix.py`
- `branch_development_notes/tools/bench_cpu_npu_offload_transfer.py`
- `branch_development_notes/reproduction/device-kv-gather-experiment-assets.md`
- `branch_development_notes/external/.gitignore`
- `branch_development_notes/external/README.md`
- `branch_development_notes/work/.gitignore`
- structured work outputs:
  - `raw-matrix-quick-20260619-073434/{manifest,results,summary,observations}`
  - `worker-local-transfer-smoke-cmake-20260619-084437/{manifest,results,summary,observations}`
  - `worker-local-transfer-quick-20260619-084951/{manifest,results,summary,observations}`
  - `worker-local-mapped-h2d-smoke-20260619-095730/observations.md`
  - `traceloom-kickstart-full-20260619/{README,meta,summary,tree-map,queries,observations}`
  - `device-kv-gather-progress-summary-20260619.md`
  - `experiment-matrix-gap-report.md`

Purpose:

- assetize raw custom-op benchmarking;
- assetize worker-local copy/mapped transfer benchmarking;
- record the TraceLoom workflow without vendoring the analyzer source;
- keep benchmark data reviewable without committing logs or large generated DBs.

### Commit 2: Experimental Worker-Local Mapped H2D Prototype

Scope:

- `vllm_ascend/kv_offload/cpu_npu.py`
- `csrc/kv_cache_block_gather/benchmarks/kv_cache_block_gather_benchmark.cpp`

Purpose:

- add the worker-local mapped-host H2D selector behind
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER`;
- keep D2H on `swap_blocks_batch`;
- add fallback checks for dtype/layout/op availability;
- make the raw benchmark parser richer with p95/p99 output.

Status:

- Python syntax check passes for the new runners and modified worker-local
  offload file.
- Raw custom-op matrix completed successfully.
- Worker-local copy baseline completed successfully.
- Worker-local mapped H2D smoke is still blocked on a 910B extension build that
  registers both `swap_blocks_batch` and `kv_cache_block_gather`.

## Verification Already Done

- `python3 -m py_compile branch_development_notes/tools/bench_device_kv_gather_matrix.py branch_development_notes/tools/bench_cpu_npu_offload_transfer.py vllm_ascend/kv_offload/cpu_npu.py`
- Raw matrix quick run: 52 cases / 208 rows, all pass.
- Worker-local copy smoke: H2D, D2H, and bidirectional pass.
- Worker-local copy quick matrix: 18 cases / 30 rows, all pass.
- TraceLoom kickstart full run: two-device `msprof` bundle analyzed with
  `--max-main-events-per-device 0`.

## Next Phase

### P0: Rerun Mapped H2D With Narrow 910B Transfer Build

Goal:

- validate the reproducible 910B transfer-only build that registers both
  `torch.ops._C_ascend.swap_blocks_batch` and
  `torch.ops._C_ascend.kv_cache_block_gather`.

Why:

- `tmp/cann-stack` has been removed from the repository index and must not be
  used as a resource;
- unrelated bundled AscendC kernels should not block transfer-path validation.

Work items:

- build with
  `VLLM_ASCEND_ACLNN_OPS=kv_cache_block_gather`,
  `VLLM_ASCEND_BUILD_ASCENDC_KERNELS=0`, and `SOC_VERSION=ascend910b1`;
- verify both transfer ops register;
- rerun `worker-local-mapped-h2d-smoke`;
- record the run under `branch_development_notes/work`.

### P1: Add Inner-Loop Worker-Local Benchmark Mode

Goal:

- reduce Python setup overhead in `bench_cpu_npu_offload_transfer.py` by reusing
  the handler and running transfer iterations inside one process/session.

Why:

- the current quick baseline proves path viability, but small payload numbers
  include runner and handler setup overhead.

Work items:

- keep the current matrix interface;
- add a mode that reuses `CpuNpuOffloadingHandler` per shape/backend;
- report warmup, measured iterations, p50/p90/p95/p99, and GB/s for copy and
  mapped H2D with the same row schema.

### P2: Run Matched Copy Vs Mapped H2D Matrix

Goal:

- identify the sparse/random H2D crossover where mapped gather beats copy in
  the worker-local path.

Initial matrix:

- `block_bytes`: 4 KiB, 16 KiB, 64 KiB, 1 MiB
- `selected_blocks`: 1, 8, 32, 128, 512, 2048
- `patterns`: sequential, stride, random
- `directions`: H2D first; D2H remains copy baseline

Outputs:

- structured `results.csv` and `results.jsonl`;
- `summary.md` with threshold notes;
- `observations.md` with build SHA, custom-op SHA/path, and environment.

### P3: TraceLoom End-To-End Profile Pair

Goal:

- capture matched `msprof` traces for copy-backend and mapped-gather runs and
  compare execution structure.

Why:

- raw microbenchmarks show capability, but TraceLoom can tell whether mapped
  gather changes the actual prefill/decode loop, idle attribution, comm balance,
  or rank skew.

Work items:

- define a small stable serving workload;
- capture copy and mapped runs with identical model/request shape;
- run TraceLoom full analysis with `--max-main-events-per-device 0`;
- add a branch-local SQL/report note for:
  - loop shape alignment;
  - gather/copy event placement;
  - comm/idle percentage shifts;
  - cross-rank imbalance.

### P4: Decide Prototype Policy

Goal:

- convert measured results into an experimental backend policy.

Likely first policy:

- mapped gather for sparse/non-contiguous H2D reloads;
- copy path for large contiguous/coalesced H2D reloads;
- copy path for D2H until a mapped scatter/write path exists;
- fallback to copy on unsupported dtype/layout/op availability.

This phase should remain explicitly experimental until the TraceLoom profile
pair and worker-local matrix both support the policy.
