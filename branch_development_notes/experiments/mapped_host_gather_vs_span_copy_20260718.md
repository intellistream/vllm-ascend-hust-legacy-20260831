# Mapped-host Gather vs Main Span-copy Milestone

Date: 2026-07-18

Branch: `wangjie/main-device-kv-gather-staging-port-20260709`

Commit: `e2260f1722288fd2663abfaf323404538acfc7b1`

## Status

**GO. Keep developing the mapped-host gather path.**

The experiment compared the branch's mapped-host `kv_cache_block_gather`
operator with the actual `origin/main` baseline: adjacent CPU/NPU block pairs
coalesced into contiguous span copies. Across two complete 24-case runs, with
the backend order reversed in the second run, all 48 comparisons exceeded the
pre-agreed 10% advantage gate.

- Smallest mapped-host advantage: **17.95%**
- Largest mapped-host advantage: **99.03%**
- Result: the direction clears its microbenchmark performance gate, including
  the best possible one-span baseline.

This is a milestone, not an end-to-end production verdict. The next gate is a
real connector A/B using observed serving mappings and workloads.

## Why This Experiment Was Needed

The old prototype benchmarks compared mapped-host gather with page-by-page
copies. That was no longer a valid baseline after `origin/main` learned to
coalesce adjacent `(cpu_block_id, gpu_block_id)` pairs into spans. If realistic
block mappings usually collapse into a few long spans, the operator's random
access capability might not justify its additional custom-op, mapping, and
workspace complexity.

The question was therefore deliberately narrow:

> At the same logical transfer size, how does mapped-host gather compare with
> main's span-copy path as block-pair fragmentation varies from one contiguous
> span to one span per block?

The decision rule agreed before measuring was:

- continue only if mapped-host gather has a stable advantage of at least 10%
  for heavily fragmented realistic mappings;
- otherwise close the direction rather than expanding its engineering surface.

## Compared Operations

### Main span-copy baseline

The benchmark mirrors the load loop in
`cpu_offload_connector.py`:

```python
for cpu_start, npu_start, span_len in spans:
    for part in range(parts):
        out[part, npu_start : npu_start + span_len].copy_(
            source[part, cpu_start : cpu_start + span_len],
            non_blocking=True,
        )
```

Both CPU and NPU IDs must increase by one for a pair to join the current span.
The benchmark's pure-Python coalescer intentionally mirrors this contract.

### Mapped-host gather

For each K/V part, the benchmark invokes:

```python
torch.ops._C_ascend.kv_cache_block_gather(
    src_block_ids,
    source[part],
    dst_block_ids,
    out[part],
)
```

The CPU source is registered with `aclrtHostRegister(...,
ACL_HOST_REGISTER_MAPPED, ...)`, and the AIV kernel reads the mapped host range
directly using the device-visible pointer.

The branch's existing workspace behavior was left unchanged. Tiling requests a
16 MiB workspace, and the torch binding allocates it when invoking the ACLNN
operator. This experiment makes **no** `workspace=0` assumption.

## Benchmark Design

The reusable benchmark is
[`tools/benchmark_kv_gather_vs_span.py`](../../tools/benchmark_kv_gather_vs_span.py).

### Matrix

- dtype: fp16
- tensor parts: 2, representing K and V
- selected blocks: 512
- CPU block capacity: 4096
- NPU block capacity: 4096
- bytes per block per part: 4 KiB, 16 KiB, 64 KiB
- total logical bytes per layer: 4 MiB, 16 MiB, 64 MiB
- requested span lengths: 1, 2, 4, 8, 16, 32, 64, 512 blocks
- resulting span counts: 512, 256, 128, 64, 32, 16, 8, 1
- warmups: 5 per backend and case
- measured iterations: 30 per backend and case
- decision margin: 10%

### Mapping construction

For each requested span length, the generator creates exact contiguous runs and
places at least one unused block between runs. CPU and NPU run starts are
shuffled independently. This guarantees that the coalescer observes the
requested span distribution instead of accidentally joining neighboring runs.

The mapping unit tests cover exact spans, a partial final span, uniqueness, and
insufficient address-space rejection:
[`tests/ut/tools/test_benchmark_kv_gather_vs_span.py`](../../tests/ut/tools/test_benchmark_kv_gather_vs_span.py).

### Fairness controls

- Both backends use the same CPU sources, NPU destinations, block IDs, logical
  bytes, dtype, and current NPU stream.
- Host registration is performed before either backend is timed. The span-copy
  baseline therefore receives the same registered/pinned source memory; this is
  conservative in favor of span-copy.
- Registration latency is reported separately and excluded from steady-state
  samples.
- The primary metric is wall time around the operation plus a device-wide
  synchronization. NPU event time is retained as a secondary diagnostic.
- Every source element is non-zero. The first element also encodes its source
  block ID, so validation catches wrong block selection as well as truncated
  copies.
- The complete selected output is checked for exact equality after each backend
  and case.
- One run measures span first; the repeat measures mapped gather first to expose
  ordering or cache bias.

The reported gain is:

```text
mapped_gain = (span_wall_ms - mapped_wall_ms) / span_wall_ms
```

## Environment and Build

- Hardware: Ascend 910B2, device 0
- Container: `quay.io/ascend/vllm-ascend:v0.21.0rc1-openeuler`
- CANN: 9.0.0
- PyTorch: 2.10.0
- torch-npu: 2.10.0
- Build selection: `VLLM_ASCEND_ACLNN_CUSTOM_OPS=kv_cache_block_gather`

The branch was copied to a detached, self-contained test clone at the commit
above so the existing working tree and earlier artifacts were not overwritten.
The custom operator and vLLM Ascend extension were then built in a disposable
container.

The CANN 9.0 installer emitted the vendor directory as `custom_transformer`,
while the initial container harness assumed `vendors/vllm-ascend`. The run
therefore corrected the harness by explicitly setting:

```bash
export VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB="$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"
```

This setup correction did not change the operator implementation or timing
path.

Before performance measurement, the direct smoke test passed for fp16 mapped
host pages on NPU 0.

## Measurement Correction During the Experiment

The first exploratory matrix reported an impossible, nearly size-independent
mapped time and up to roughly 250 GB/s. It was rejected rather than interpreted
as a result.

Root cause: the initial harness created a Python-side custom stream and waited
only on that stream's end event. The ACLNN `OpCommand` custom handler could
submit work without retaining that Python stream context, allowing the event to
measure submission rather than completion.

The retained benchmark corrected this in three ways:

1. use PyTorch's current stream, matching the connector path;
2. use device-wide synchronization for the primary wall-time boundary;
3. fill every source element with non-zero data so a partial copy cannot pass
   correctness validation merely because untouched elements are zero.

After the correction, mapped time scaled with bytes, full correctness still
passed, and event time closely tracked the completed device work. The invalid
run is explicitly labeled and excluded from every result and decision below.

## Results

Each cell below is the range across the valid span-first and mapped-first runs.

| block bytes | span len | spans | span wall ms | mapped wall ms | mapped gain |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 1 | 512 | 37.244–41.900 | 0.372–0.405 | +99.0–99.0% |
| 4 KiB | 2 | 256 | 19.027–21.026 | 0.373–0.389 | +98.0–98.2% |
| 4 KiB | 4 | 128 | 9.572–10.786 | 0.370–0.392 | +96.1–96.4% |
| 4 KiB | 8 | 64 | 4.943–5.466 | 0.368–0.378 | +92.6–93.1% |
| 4 KiB | 16 | 32 | 2.560–2.870 | 0.362–0.385 | +85.9–86.6% |
| 4 KiB | 32 | 16 | 1.389–1.651 | 0.361–0.384 | +74.0–76.7% |
| 4 KiB | 64 | 8 | 0.840–0.912 | 0.367–0.383 | +56.4–57.9% |
| 4 KiB | 512 | 1 | 0.509–0.513 | 0.370–0.385 | +25.0–27.4% |
| 16 KiB | 1 | 512 | 37.914–41.878 | 0.901–0.949 | +97.6–97.7% |
| 16 KiB | 2 | 256 | 18.657–21.088 | 0.895–0.951 | +95.2–95.5% |
| 16 KiB | 4 | 128 | 9.469–10.616 | 0.896–0.925 | +90.5–91.3% |
| 16 KiB | 8 | 64 | 4.936–5.480 | 0.902–0.921 | +81.7–83.2% |
| 16 KiB | 16 | 32 | 2.573–2.868 | 0.901–0.920 | +65.0–67.9% |
| 16 KiB | 32 | 16 | 1.501–1.582 | 0.904–0.919 | +39.8–41.9% |
| 16 KiB | 64 | 8 | 1.123–1.352 | 0.907–0.922 | +18.0–32.9% |
| 16 KiB | 512 | 1 | 1.284–1.322 | 0.902–0.919 | +29.8–30.5% |
| 64 KiB | 1 | 512 | 37.864–42.157 | 3.053–3.163 | +91.9–92.5% |
| 64 KiB | 2 | 256 | 18.768–21.336 | 3.047–3.166 | +83.8–85.2% |
| 64 KiB | 4 | 128 | 13.278–13.494 | 3.063–3.134 | +76.4–77.3% |
| 64 KiB | 8 | 64 | 8.491–10.474 | 3.049–3.080 | +63.7–70.9% |
| 64 KiB | 16 | 32 | 10.385–17.058 | 3.055–3.079 | +70.6–81.9% |
| 64 KiB | 32 | 16 | 8.924–10.476 | 3.048–3.075 | +65.8–70.7% |
| 64 KiB | 64 | 8 | 9.343–9.895 | 3.040–3.067 | +67.5–69.0% |
| 64 KiB | 512 | 1 | 7.876–12.914 | 3.039–3.075 | +61.0–76.5% |

### Throughput summary

| block bytes | logical bytes | mapped throughput across runs | one-span gain | fully fragmented gain |
| ---: | ---: | ---: | ---: | ---: |
| 4 KiB | 4 MiB | 10.36–11.61 GB/s | +25.0–27.4% | +99.0% |
| 16 KiB | 16 MiB | 17.64–18.75 GB/s | +29.8–30.5% | +97.6–97.7% |
| 64 KiB | 64 MiB | 21.19–22.08 GB/s | +61.0–76.5% | +91.9–92.5% |

### Registration cost

Registration is a setup cost, not part of the steady-state rows:

| one K or V region | registration wall time across runs |
| ---: | ---: |
| 16 MiB | 6.17–6.90 ms |
| 64 MiB | 25.88–28.37 ms |
| 256 MiB | 108.67–122.67 ms |

The mapping must therefore be registered once and reused for a long-lived CPU
KV slab. Registering in a per-layer or per-request hot path would erase much of
the steady-state benefit.

## Interpretation

1. **Fragmentation is where mapped gather is overwhelmingly stronger.** Span
   copy pays host/API scheduling cost per span and per K/V part. With 512 spans,
   the operator reduces 1024 copy submissions to two gather submissions.
2. **Perfect coalescing does not eliminate the advantage on this platform.**
   Even at one span, mapped gather remained 25% to 76.5% faster in the tested
   sizes. This was the experiment's most important and least obvious result.
3. **Mapped timing is stable and primarily byte-dependent.** Reversing backend
   order moved mapped mean time by roughly 1% to 9%. Span-copy was noisier,
   especially for 64 MiB transfers, but its most favorable observed comparison
   still left mapped gather 17.95% ahead.
4. **The current workspace does not invalidate the direction.** The 16 MiB
   workspace contract and wrapper allocation were included in the measured
   steady-state invocation. Workspace removal may still be useful later, but it
   is not required to pass this gate.
5. **Registration lifetime is now an explicit architectural constraint.** The
   steady-state result is attractive only when the mapped CPU allocation and
   its registration cache are persistent.

## Limitations

- This is a direct operator microbenchmark, not a full vLLM serving result.
- It measures one layer-shaped transfer at a time without model compute,
  multi-stream contention, or concurrent requests.
- The mappings have controlled fragmentation but are synthetic; production
  prefix-cache traces may have different span distributions and ordering.
- The CPU source is an ordinary contiguous torch allocation registered through
  the same runtime API. Production multiprocessing shared-memory slabs may
  differ in NUMA placement, alignment, and lifetime behavior.
- Results currently cover Ascend 910B2 with CANN 9.0.0 and fp16 only.
- Device-wide synchronization makes completion time trustworthy but removes
  any overlap that a full connector pipeline might achieve.
- The span-copy baseline showed meaningful run-to-run noise at 64 MiB. The
  reversed-order repeat protects the decision, but a serving-level A/B remains
  necessary before choosing a default policy.

## Decision and Next Evidence Gate

The mapped-host gather direction should continue. It should not yet become an
unconditional default based on this microbenchmark alone.

The next experiment should:

1. instrument real connector block mappings and retain their span-count and
   span-length distributions;
2. run the same workload with main span-copy and mapped-host gather under the
   real load stream and layer synchronization behavior;
3. measure request latency, scheduler stalls, transfer overlap, throughput, and
   correctness on long-context prefix-cache hits;
4. verify long-lived registration, teardown, and multiprocessing shared-memory
   behavior;
5. keep the current workspace contract unchanged during the A/B so workspace
   optimization remains a separate variable.

## Artifacts

All retained raw results are stored locally under
`branch_development_notes/work/kv-gather-vs-span-20260718-115632/`. The
`work/` tree is intentionally excluded from Git because it contains large raw
build and profiling artifacts. Key local files are:

- `corrected/summary.md`: valid span-first run
- `reverse-order/summary.md`: valid mapped-first repeat
- `conclusion.md`: compact decision record
- `invalid-stream-only-timing/INVALID.md`: excluded exploratory measurement
  and reason
- `run.log`: clean-container custom-op and extension build log

## Reproduction Outline

Inside the same NPU container environment, the essential sequence is:

```bash
source /usr/local/Ascend/cann-9.0.0/set_env.sh
export VLLM_ASCEND_ACLNN_CUSTOM_OPS=kv_cache_block_gather
pip install -e . --no-build-isolation -v

source vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
export VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB="$PWD/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/op_api/lib/libcust_opapi.so"

python tools/smoke_device_kv_gather.py
python tools/benchmark_kv_gather_vs_span.py \
    --backend-order span-first \
    --output-dir branch_development_notes/work/kv-gather-vs-span/span-first
python tools/benchmark_kv_gather_vs_span.py \
    --backend-order mapped-first \
    --output-dir branch_development_notes/work/kv-gather-vs-span/mapped-first
```

The actual run used a disposable detached clone and mounted device 0 plus the
host Ascend driver into the container. See `run.log` for the complete build
record and each result manifest for the exact benchmark argv.

## Validation

- custom operator build: passed
- direct `kv_cache_block_gather` smoke: passed
- complete output validation in every retained benchmark case: passed
- benchmark mapping unit tests: `3 passed`
- Ruff lint and format checks for the benchmark and tests: passed
- `git diff --check`: passed
