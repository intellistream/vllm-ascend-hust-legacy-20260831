# Device KV Gather Production Gate Gap Report

Generated: 2026-06-19 Asia/Shanghai

Scope: assess whether the current `experiment/device-kv-gather` branch can run
the benchmark matrix proposed in
`branch_development_notes/notes/strata_on_ascend_comment.md:352`, and list the
missing capabilities plus concrete ways to add them.

## Executive Summary

Current status: **capable of pushing the experimental matrix forward item by
item, but not yet packaged as a full production benchmark gate**.

The branch has enough source to validate the prototype path:

- `csrc/kv_cache_block_gather` exists and is wired into the custom-op build.
- `tools/smoke_device_kv_gather.py` exists for direct op correctness smoke.
- `csrc/kv_cache_block_gather/benchmarks/kv_cache_block_gather_benchmark.cpp`
  can compare mapped-host gather against page-wise copy, contiguous copy, and
  HBM gather for selected microbenchmark shapes.

But the production gate needs much more:

- automated sweeps and structured result output;
- D2H and bidirectional transfer cases;
- end-to-end TTFT/ITL workloads under CPU-hit and partial-prefix-hit patterns;
- prefill/decode interference tests;
- TP/MLA layout validation;
- fallback/error/registration/mapped-pointer metrics;
- bounded host registration lifecycle;
- integration of mapped gather into the newer worker-local offload path.

The current prototype should be treated as **custom-op execution proven +
partial microbenchmark coverage**. For the first experimental phase, most gaps
below are engineering gaps: matrix runners, extra cases, metrics plumbing, and
integration shape. Production-safe lifecycle requirements are still listed, but
they do not need to block the first round of experiments.

## Environment Check

Host:

| Item | Observation | Impact |
| --- | --- | --- |
| Branch | `experiment/device-kv-gather` | Correct branch for this investigation. |
| Host Python | `torch`, `torch_npu`, and `vllm` are not importable from host `python3`; `vllm_ascend` imports as local source. | Host cannot directly run NPU/vLLM checks. Use Docker. |
| NPU | 8 x Ascend 910B2 visible via `npu-smi`; NPU 1 has a large `VLLMEngineCor` allocation, other cards are mostly free. | Enough hardware for one-card smoke and multi-card follow-up, avoiding NPU 1. |
| Docker | `quay.io/ascend/vllm-ascend:v0.16.0rc1-openeuler` exists locally. | Recommended runtime image is available. |
| Mounted branch in Docker | With `/home/jingyuan/workspace/vllm-ascend-hust` mounted read-only, Docker sees `torch 2.9.0`, `torch_npu 2.9.0`, `vllm 0.16.0`, `torch.npu.is_available() == True`, `torch.npu.device_count() == 8`. | Runtime prerequisites are available in Docker. |
| Host build artifacts | No host-side `vllm_ascend_C*.so`, `libcust_opapi.so`, or benchmark binary found under the current checkout. | This is not a capability blocker: follow `branch_development_notes/notes/reproduction.md` to build inside Docker and generate `libcust_opapi.so`. |
| Existing long-running containers | Existing `/workspace/vllm-ascend-hust` in one container is on `main`; it has `swap_blocks_batch` but not `kv_cache_block_gather`. | Existing built container cannot validate this branch's gather op without rebuilding/mounting this branch. |

## Source Capability Inventory

| Capability | Current state | Evidence | Readiness |
| --- | --- | --- | --- |
| Custom ACLNN op source | Present under `csrc/kv_cache_block_gather`. | `op_host`, `op_kernel`, and benchmark source exist. | Prototype-ready. |
| Custom-op build wiring | Present. | `csrc/build_aclnn.sh` includes `kv_cache_block_gather` for custom-op build. | Build-ready, but not built in host checkout. |
| Python torch binding | Present. | `torch.ops._C_ascend.kv_cache_block_gather` binding is declared in `csrc/torch_binding.cpp`. | Build-dependent. |
| Direct correctness smoke | Present. | `tools/smoke_device_kv_gather.py`. | Good for one-op smoke only. |
| Microbenchmark | Present but manual. | `kv_cache_block_gather_benchmark.cpp` accepts block count, block bytes, source/destination patterns, warmup, and iterations. | Partial matrix coverage. |
| Legacy CPU offload integration | Present. | `CPUOffloadingConnectorWorker` optionally calls `kv_cache_block_gather`. | Experimental, deprecated path. |
| Worker-local offload copy path | Present. | `vllm_ascend/kv_offload/cpu_npu.py` has H2D/D2H streams, events, and `swap_blocks_batch`. | Production-better baseline, but no mapped gather backend. |
| End-to-end CPU offload test | Present but skipped. | `tests/e2e/singlecard/test_cpu_offloading.py` is marked skipped because the old CPU offload connector is deprecated. | Needs rework. |
| Prefix TTFT benchmark examples | Present. | Prefix-cache nightly YAMLs and `tools/aisbench.py::get_TTFT`. | Reusable, not connected to CPU-offload gather. |
| Metrics for fallback/registration | Absent. | Fallback is only logged once; host mappings are static C++ state. | Must be added. |

## Experiment Matrix Gap

| Matrix item | Required question | Current ability | Missing capability | Supplement method | Target files / components | Acceptance check |
| --- | --- | --- | --- | --- | --- | --- |
| Fragment size sweep: 128B, 512B, 1KB, 4KB, 16KB, 64KB, 1MB | Find custom-vs-copy crossover by fragment size. | C++ benchmark can vary `--elems-per-block`; effective bytes are `elems_per_block * sizeof(float)`. | No automated sweep; no JSON/CSV output; float32-only benchmark does not directly express dtype-specific byte sizes. | Add a Python or shell runner that maps target bytes to `--elems-per-block`, invokes the C++ binary, parses stats, and writes JSONL/CSV. Extend benchmark to print machine-readable output. | Add `tools/bench_device_kv_gather_matrix.py`; update `csrc/kv_cache_block_gather/benchmarks/kv_cache_block_gather_benchmark.cpp`. | One command emits rows for all fragment sizes with mapped gather GB/s, page-copy GB/s, contig-copy GB/s, p50/p95/p99 latency. |
| Fragment count sweep: 1, 8, 32, 128, 512, 2048 | Measure device-side parallelism and launch/index overhead. | C++ benchmark can vary `--selected-blocks`. | No sweep runner; current output has mean/p50/p90 only; no p95/p99. | Add selected-block sweep and percentile expansion. Use fixed fragment sizes near expected crossover. | Same as above. | JSON rows exist for every count and include latency percentiles. |
| Contiguous run length sweep | Decide when bulk copy is better than gather. | Benchmark has source/destination patterns but no contiguous-run concept. | Cannot generate block ids with controlled run lengths like 1, 2, 4, 8, 16 contiguous pages per run. | Add `--src-run-length` / `--dst-run-length` or a `run_length` pattern that creates grouped contiguous block ids. Compare gather vs one-run or coalesced-copy implementation. | C++ benchmark block-id generator; optional Python runner. | Results show crossover between many small runs and large contiguous runs. |
| Random vs sorted block ids | Measure index locality and host DRAM locality. | C++ benchmark supports `random`, `sequential`, `reverse`, `stride`; sequential is a proxy for sorted. | No explicit "same random set sorted by src" comparison; no host locality labels in output. | Add `sorted_random` pattern: generate random ids, then sort by source id while preserving destination mapping variants. | C++ benchmark block-id generator. | Report contains random and sorted-random cases with identical selected set. |
| HtoD only | CPU to NPU load policy. | Covered by mapped-host gather and H2D copy in C++ benchmark; worker-local copy path supports CPU->NPU. | C++ path is op-level, not vLLM integration; legacy connector only. | Keep C++ microbench for raw backend; add worker-local H2D benchmark through `CpuNpuOffloadingHandler.transfer_async`. | New tests/tool around `vllm_ascend/kv_offload/cpu_npu.py`. | Raw op and worker-local H2D numbers are both reported. |
| DtoH only | NPU to CPU save policy. | Worker-local copy path supports D2H via `swap_blocks_batch`. | No mapped-host write/scatter backend; C++ gather benchmark has no D2H case; legacy gather path only loads. | Add baseline D2H copy benchmark first. Treat mapped D2H as future custom op (`kv_cache_block_scatter` or read/write variant) if needed. | `CpuNpuOffloadingHandler` benchmark harness; optional new custom op later. | D2H copy latency/GB/s reported; mapped D2H explicitly marked unsupported until op exists. |
| Bidirectional | Save/load default strategy and interference. | Worker-local copy path has separate H2D/D2H streams. | No bidirectional benchmark; no overlapping H2D+D2H runner; mapped gather participates only in H2D legacy path. | Add a worker-local stress tool that schedules simultaneous save and load jobs and records both queues. Later include mapped H2D plus copy D2H. | New `tools/bench_cpu_npu_offload_transfer.py`; `CpuNpuOffloadingHandler` instrumentation. | Output includes H2D, D2H, combined throughput, queue delay, and stream wait. |
| With prefill running | Detect prefill regression from device-driven gather using NPU cores. | Existing benchmark is transfer-only; repo has model benchmark harnesses. | No concurrent prefill + transfer test. | Add an e2e scenario that issues long-prefill requests while background CPU-hit loads are triggered. Use two modes: copy backend and mapped backend. | New e2e benchmark config or script; reuse `vllm bench` / `aisbench`; add backend env toggles. | p50/p95/p99 TTFT and prefill latency regression emitted for copy vs mapped. |
| With decode running | Detect ITL regression from device-driven gather. | Existing benchmark is transfer-only; `aisbench` can collect serving metrics. | No decode-heavy workload paired with host gather; no ITL extraction helper in current local helper analogous to `get_TTFT`. | Add decode-heavy dataset/config and `get_ITL` parser. Run copy vs mapped with matched seeds and traffic. | `tools/aisbench.py`; nightly/e2e benchmark YAML or standalone runner. | p50/p95/p99 ITL regression <= configured threshold, or mapped backend is limited to prefill window. |
| Long-context CPU-hit workload | Measure TTFT/throughput benefit in real CPU-hit path. | Skipped CPU offload test has a latency pattern with cold/GPU-hit/CPU-hit; prefix benchmark YAMLs exist. | Test is skipped and not matrix-grade; no mapped-vs-copy toggle; no percentiles. | Unskip/rewrite for worker-local offload. Add requests with controlled CPU cache residency and compare cold, copy CPU-hit, mapped CPU-hit. | `tests/e2e/singlecard/test_cpu_offloading.py` replacement; standalone benchmark runner for longer runs. | CPU-hit workload emits TTFT/throughput percentiles and correctness check. |
| Partial prefix hit workload | Measure true benefit zone for sparse/partial onload. | Prefix-cache YAMLs compare prefix0 vs prefix75, but not CPU offload gather. | No way to force partial CPU-resident prefix blocks and mapped gather selection. | Build synthetic prompt sets with controlled overlap and CPU block residency; record hit ratio and selected block count. Add scheduler/connector telemetry for partial hits. | New benchmark config; scheduler/kv-offload telemetry. | Report includes prefix-hit ratio, selected bytes/full bytes, TTFT benefit, fallback count. |
| TP layout | Prevent shard/layout transform overhead surprises. | Worker-local copy path handles flattened layer/key/value tensors; repo has TP tests. | Mapped gather path is in legacy connector and not validated for TP worker-local layout. | Integrate mapped gather into `CpuNpuOffloadingHandler` after validating per-rank block ids, tensor strides, and local shard semantics. Add TP=2/4/8 correctness and perf cases. | `vllm_ascend/kv_offload/cpu_npu.py`; e2e TP benchmark YAML. | Copy and mapped outputs match; transfer stats are per rank; no fallback from layout mismatch. |
| MLA layout | Avoid layout transform cliff. | Legacy connector has `use_mla` metadata path; model code has MLA attention. | No mapped gather validation for MLA shapes; no matrix case. | Add MLA-specific shape/stride checks to backend selection and benchmark a DeepSeek-style MLA model or synthetic KV cache tensors. | `CPUOffloadingConnectorWorker` short-term; worker-local mapped backend long-term; MLA e2e config. | MLA case either uses mapped backend safely or records deterministic fallback reason. |

## Required Metrics Gap

| Metric | Current ability | Gap | Supplement method | Acceptance check |
| --- | --- | --- | --- | --- |
| Custom kernel effective GB/s | C++ benchmark prints GB/s for mapped-host gather. | Not structured; no dtype/layout labels. | Machine-readable output and matrix runner. | JSON row per run includes `backend=mapped_gather`, bytes, duration, GB/s. |
| Copy path effective GB/s | C++ benchmark prints page-copy and contig-copy GB/s; worker-local copy has transfer size/time. | No unified reporting across raw and worker-local paths. | Normalize result schema across C++ benchmark and Python worker-local benchmark. | Same report compares raw copy, worker copy, and mapped gather. |
| p50/p95/p99 transfer latency | C++ currently prints p50 and p90; worker-local events expose elapsed time per transfer. | Missing p95/p99; no per-transfer sample export. | Add percentile calculation and sample dump. | p50/p95/p99 present for every case. |
| p50/p95/p99 TTFT | Aisbench and prefix configs can report TTFT; helper currently extracts average TTFT. | Need percentiles and copy-vs-mapped pairing. | Extend result parser for p50/p95/p99; add paired benchmark runner. | TTFT percentile table for cold, copy CPU-hit, mapped CPU-hit. |
| p50/p95/p99 ITL | Benchmark framework likely records ITL in result tables. | No helper function and no decode-heavy matrix case. | Add `get_ITL` parser and decode workload configs. | ITL percentile table and regression calculation. |
| Prefill latency regression | Some scheduler profiling code exists. | Not tied to transfer backend. | Add prefill-only benchmark with backend toggle and profiler marks. | Regression reported as percentage vs copy baseline. |
| Decode latency regression | Not available for mapped gather. | Need decode-heavy traffic and ITL parser. | Same as ITL work. | Regression reported; gate can enforce <= 3%. |
| NPU core occupancy | `npu-smi` can show coarse AICore%; profiling configs exist. | No automated occupancy capture synchronized with benchmark windows. | Add sampler around `npu-smi info` or use Ascend profiling/torch profiler where available. | Result contains average/max AICore occupancy for transfer window and e2e window. |
| Stream wait time | Worker-local copy path uses events and waits; legacy gather synchronizes layer load stream. | No wait-time measurement or reporting. | Instrument stream waits in worker-local offload handler; record queue delay, wait-on-previous, wait-on-model-stream. | Transfer result includes wait breakdown. |
| Registration failure/fallback count | Legacy connector logs fallback once. C++ registration errors throw. | No counters, no reason histogram, no fallback-to-copy after registration failure in C++ wrapper. | Add backend stats object in Python and C++ error-to-status path; expose counters via logs/metrics. | Result includes `fallback_count`, `fallback_reason_counts`, `registration_failure_count`. |
| Mapped pointer validation failure | Not present. | No generation/context/lifetime validation. | Add host mapping registry with generation ids, registered range validation, device/context association, and shutdown checks. | Invalid pointer use fails closed to copy and increments validation counter. |
| Long-running churn leak | Prototype C++ mapping cache is static and never unregisters. | Cannot pass leak gate. | Move registration lifecycle to worker-owned registry; unregister at shutdown after stream sync; add churn test. | Repeated allocate/free cycles keep registered range count and bytes bounded. |

## Main Architectural Gaps

### 1. Mapped gather lives in the wrong production path

The mapped gather integration is currently in the legacy distributed CPU
offload connector. The more production-shaped path is
`vllm_ascend/kv_offload/cpu_npu.py`, which already owns CPU tensors, transfer
streams, events, in-flight transfer queues, and H2D/D2H copy execution.

Supplement:

1. Keep `swap_blocks_batch` as the default backend.
2. Add a backend selector inside `CpuNpuOffloadingHandler`:
   `copy`, `mapped_gather`, and `fallback_reason`.
3. Enable mapped gather only for CPU->NPU loads, contiguous supported dtypes,
   supported device/product, and validated block layout.
4. Continue using copy for NPU->CPU until a mapped write/scatter backend exists.

### 2. Host registration lifecycle is not production-safe

`csrc/torch_binding.cpp` caches host mappings in process-static state and does
not unregister them. This is acceptable for a direct smoke but cannot satisfy
the long-running churn gate.

Supplement:

1. Add a worker-local `HostMappingRegistry`.
2. Register full worker-owned CPU swap arenas first, not arbitrary temporary
   tensor spans.
3. Store host base, size, mapped pointer, device id/context, generation, ref
   count, and state.
4. Synchronize transfer streams before unregister.
5. Call `aclrtHostUnregister` deterministically at worker shutdown.
6. Add fault injection for registration and unregister failures.

### 3. The benchmark is not yet a matrix runner

The C++ benchmark is useful, but it is a manual primitive. It should become one
input to a repeatable gate.

Supplement:

1. Add `tools/bench_device_kv_gather_matrix.py`.
2. Build or locate the benchmark binary.
3. Run parameter sweeps from a YAML/JSON matrix file.
4. Emit JSONL and Markdown summaries.
5. Keep raw command, env, git SHA, CANN version, NPU model, and busy-device
   state in the result manifest.

### 4. End-to-end workload coverage is missing

The matrix requires production behavior: TTFT, ITL, prefill/decode regression,
and real prefix-hit scenarios. Current source has benchmark infrastructure, but
not a CPU-offload-gather-specific suite.

Supplement:

1. Add copy-vs-mapped backend toggles to benchmark jobs.
2. Add CPU-hit and partial-prefix-hit workloads with controlled prompt overlap.
3. Add prefill-heavy and decode-heavy traffic profiles.
4. Parse TTFT and ITL percentiles from the benchmark output.
5. Pair every mapped run with a copy baseline run on the same device class.

## Suggested Implementation Order

| Phase | Goal | Work items | Exit criteria |
| --- | --- | --- | --- |
| 0 | Reproducible smoke | Build current branch in Docker, set custom opapi path, run `tools/smoke_device_kv_gather.py` for fp16/bf16/fp32. | Direct op smoke passes on one free 910B2. |
| 1 | Raw microbenchmark matrix | Add matrix runner around C++ benchmark; add JSON output and p95/p99. | Size/count/pattern sweeps produce structured results. |
| 2 | Worker-local copy baseline | Add worker-local H2D/D2H/bidirectional benchmark using `CpuNpuOffloadingHandler`. | Copy baseline covers H2D, D2H, and bidirectional with stream wait metrics. |
| 3 | Worker-local mapped H2D backend | Move mapped gather backend selection into worker-local offload path with fallback stats. | Copy and mapped CPU->NPU produce identical KV contents; fallback counters work. |
| 4 | Registration lifecycle | Add host mapping registry, deterministic unregister, churn tests, failure injection. | Long-running churn has bounded mapped bytes/ranges and no stale pointer use. |
| 5 | End-to-end benchmark gate | Add CPU-hit, partial-prefix-hit, prefill-heavy, and decode-heavy workloads. | Report includes TTFT/ITL percentiles and regression vs copy. |
| 6 | TP/MLA coverage | Add TP and MLA shape/layout cases. | TP/MLA either use mapped backend safely or fallback with explicit reason. |

## Minimal New Artifacts To Add

| Artifact | Purpose |
| --- | --- |
| `tools/bench_device_kv_gather_matrix.py` | Drives raw C++ microbenchmark sweeps and writes JSONL/CSV/Markdown. |
| `benchmarks/device_kv_gather/matrix.yaml` | Declarative fragment size/count/pattern/run-length matrix. |
| `tools/bench_cpu_npu_offload_transfer.py` | Exercises worker-local H2D/D2H/bidirectional transfers without full serving. |
| `vllm_ascend/kv_offload/host_mapping.py` or C++ equivalent | Worker-local registration lifecycle and mapped pointer registry. |
| `vllm_ascend/kv_offload/backend_stats.py` | Shared counters for backend selected, fallback reasons, registration failures, validation failures. |
| `tests/e2e/singlecard/test_cpu_offloading_worker_local.py` | Correctness and basic latency regression for worker-local CPU-hit path. |
| `benchmarks/device_kv_gather/e2e_cpu_hit.yaml` | Long-context CPU-hit serving workload. |
| `benchmarks/device_kv_gather/e2e_partial_prefix.yaml` | Partial prefix hit workload. |
| `benchmarks/device_kv_gather/e2e_decode_interference.yaml` | Decode-heavy ITL regression workload. |

## Current Gate Readiness By Category

| Category | Readiness | Rationale |
| --- | --- | --- |
| Direct op correctness | High for experiment | `reproduction.md` records that the custom ACLNN op can be built, registered, loaded, and executed in Docker on one NPU. |
| Raw H2D microbenchmark | Medium | C++ benchmark exists but needs runner and richer output. |
| Raw D2H/bidirectional benchmark | Low | Copy D2H exists in worker-local path, but no raw mapped D2H op or benchmark. |
| Worker-local integration | Low | Production-shaped copy path exists, mapped gather not integrated there. |
| Host registration lifecycle | Low for production, non-blocking for phase-1 experiments | Static mapping cache has no unregister or generation safety, but phase-1 can run bounded experiments without solving this first. |
| End-to-end TTFT/ITL gate | Low | Benchmark infrastructure exists, CPU-offload-gather-specific workloads do not. |
| TP/MLA layout gate | Low | No dedicated mapped gather validation. |
| Metrics and observability | Low | Need structured transfer, fallback, registration, validation, occupancy, and wait metrics. |

## Practical Verdict

The current branch can support a **Phase 0/1 experimental validation
campaign**:

```text
build custom op -> run direct smoke -> run manual/raw H2D microbenchmarks
```

It cannot yet support the fully packaged production gate from
`strata_on_ascend_comment.md:352`:

```text
microbench matrix + end-to-end CPU-hit/prefix workloads + interference
+ TP/MLA layout + production lifecycle/metrics
```

The fastest route is to run the experimental matrix incrementally: first raw
custom-op sweeps, then copy-vs-mapped worker-local transfer tests, then
end-to-end CPU-hit/prefix-hit workloads. The production route should still move
mapped gather behind the worker-local offload backend, keep copy as fallback,
and add lifecycle/metrics hardening after the first experimental evidence is
collected.
