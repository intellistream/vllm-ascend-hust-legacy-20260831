# Mapped-host KV Gather Production Debt

Date: 2026-07-18

Branch: `wangjie/main-device-kv-gather-staging-port-20260709`

## Purpose

This note records the engineering debt that remains between the successful
`kv_cache_block_gather` prototype and a production-safe CPU-offload restore
backend. It separates work that is already proven from work that must be
completed before mapped-host gather can be enabled by default.

The device kernel itself is no longer the main uncertainty. The remaining work
is primarily around packaging, ownership and lifetime of mapped host memory,
stream ordering, fallback semantics, compatibility, and serving-level evidence.

## Current Proven State

The following pieces exist and have passed their current gates:

- The AscendC custom operator is in-tree under
  [`csrc/kv_cache_block_gather`](../../csrc/kv_cache_block_gather/).
- The custom-op build produces the ACLNN entry points
  `aclnnKvCacheBlockGatherGetWorkspaceSize` and
  `aclnnKvCacheBlockGather`.
- The Torch binding exposes
  `torch.ops._C_ascend.kv_cache_block_gather` and mapped-host registration,
  inspection, statistics, and explicit clear helpers.
- The direct operator smoke test passes on Ascend 910B2 with CANN 9.0.
- The corrected mapped-host gather versus production span-copy microbenchmark
  passed the agreed 10% continuation gate in all 48 comparisons.
- The current CPU-offload connector contains an env-gated mapped-host backend and
  retains main's coalesced span-copy implementation as fallback.
- The current 16 MiB workspace request and wrapper-side allocation were included
  in the valid measurements. No `workspace=0` assumption is required for the
  direction to remain viable.

The performance result and its limitations are recorded in
[`mapped_host_gather_vs_span_copy_20260718.md`](../experiments/mapped_host_gather_vs_span_copy_20260718.md).

## Production Invariants

The integration must preserve these rules:

1. **Span-copy remains the baseline and safe fallback.** Fallback must return to
   the coalesced main path, never to page-by-page copies.
2. **Mapped gather is initially CPU-to-NPU only.** NPU-to-CPU save continues to
   use the existing copy path until a separately designed scatter/write backend
   exists.
3. **The production integration point is the current CPU-offload connector.** Do
   not revive a deprecated connector or maintain a parallel cache manager.
4. **Registration is control-plane work.** Long-lived worker-owned memory may be
   registered once and reused; per-layer or per-request registration is not an
   acceptable steady-state design.
5. **Mapped memory must outlive all enqueued device reads.** CPU allocation
   release, shared-memory unlink, host unregister, stream completion, and worker
   teardown need one explicit ordering contract.
6. **No global synchronization in the hot path.** Correctness must be expressed
   through the connector's stream and event dependencies.
7. **Keep workspace optimization separate.** First integrate and validate the
   measured workspace contract; change it only in an isolated follow-up.

## P0: Safety And Packaging Debt

### 1. Replace the process-global mapping cache with owned registrations

[`csrc/torch_binding.cpp`](../../csrc/torch_binding.cpp) currently stores
registrations in a process-static `std::vector<HostMapping>`. The entries record
only host base, size, and mapped device base. A mutex protects lookup and
mutation, and `clear_kv_cache_block_gather_host_mappings()` can unregister every
entry.

The explicit clear primitive is useful for tests, but it is not yet a production
lifecycle model:

- entries have no worker/allocator owner;
- entries have no device or context identity;
- entries have no allocation generation or stale-address protection;
- entries have no reference count or in-flight use count;
- only global clear is available, not unregister-by-handle/owner;
- direct mapped registrations are not cleared by connector shutdown;
- staging-pool close waits for its slot events, but does not unregister the
  corresponding C++ mapping entries;
- a global clear can be unsafe if another connector or stream still uses an
  entry.

Required production shape:

- Introduce a worker-local `HostMappingRegistry`, or an equivalent C++ registry
  with explicit worker ownership.
- Return an opaque registration handle rather than exposing only a process-wide
  cache side effect.
- Record at least: host base, registered size, mapped base, device/context,
  allocation generation, owner, state, references, and in-flight uses.
- Prefer registering complete worker-owned CPU KV arenas or complete staging
  slabs. Do not lazily accumulate arbitrary temporary tensor spans.
- Reject overlapping registrations with incompatible ownership instead of
  silently accumulating them.
- Provide idempotent unregister-by-handle and unregister-by-owner operations.
- Synchronize all streams that may read a mapping before unregistering it.
- Unregister mappings before the underlying CPU tensor/shared-memory allocation
  is freed or unlinked.
- Define fork and multiprocessing behavior explicitly. A child process must not
  inherit a cache entry that refers to an invalid runtime registration/context.

Exit criteria:

- repeated allocate/register/use/unregister cycles leave mapped bytes and entry
  count bounded;
- address reuse cannot resolve to a stale mapped pointer;
- shutdown deterministically unregisters all mappings owned by that worker;
- unregister during in-flight device access is prevented by construction and
  covered by a failure test.

### 2. Define one teardown order across connector, streams, and shared memory

[`CPUOffloadingConnectorWorker.shutdown()`](../../vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py)
currently closes the optional staging pool, but direct mapped registrations are
not part of the shutdown sequence. Shared-memory cleanup is owned elsewhere by
the metadata process.

Required shutdown order:

1. stop accepting new restore work;
2. drain or cancel pending connector work according to a documented policy;
3. wait on the connector load stream and all staging-slot events;
4. unregister worker-owned mapped regions;
5. release CPU tensors and staging slots;
6. close/unlink shared-memory objects only after all workers have released them;
7. make repeated shutdown calls harmless.

The normal shutdown path, exception path, worker restart path, and interpreter
teardown path must converge on the same idempotent ownership logic. `__del__`
may remain a last-resort guard, but cannot be the primary correctness mechanism.

### 3. Make custom-op packaging self-contained

The CANN 9.0 build currently emits the vendor directory as
`vendors/custom_transformer`, while earlier harnesses assumed
`vendors/vllm-ascend`. The experiment works by setting
`VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB` to the generated
`libcust_opapi.so` explicitly.

Production packaging must not depend on a manually discovered development-tree
path:

- select and document the canonical vendor name;
- install the op implementation, op metadata, ACLNN header, and
  `libcust_opapi.so` into the wheel/package consistently;
- activate the custom OPP path as part of the supported package initialization;
- resolve the ACLNN symbols from the packaged library without requiring a user
  override;
- retain the explicit opapi path only as a diagnostic/development escape hatch;
- fail early with a precise capability reason if the packaged symbols are
  absent or incompatible;
- test wheel installation in a clean container, not only editable installation
  from the source tree.

Exit criteria:

- a clean wheel install can import the extension, locate the packaged custom
  operator, resolve both ACLNN symbols, and pass the direct smoke without custom
  path environment variables.

### 4. Prove current-stream and completion semantics

The connector calls mapped gather inside `torch.npu.stream(self.load_stream)`,
and the C++ wrapper submits ACLNN work on the current NPU stream. An early
benchmark incorrectly used a Python custom-stream event that did not reliably
bound the ACLNN completion, producing impossible throughput. The corrected
benchmark used the current stream and device-wide synchronization for trusted
measurement.

Production must establish the asynchronous contract without device-wide
synchronization:

- verify that ACLNN submission observes `self.load_stream` on every supported
  torch-npu/CANN combination;
- record a completion event after the last K/V gather for the layer;
- make the model-compute stream wait on that event before consuming restored KV;
- ensure staging slots are not reused until their gather completion event fires;
- ensure shutdown waits for the same completion objects before unregister;
- cover exceptions between the K and V submissions and between layers;
- avoid timing instrumentation that changes stream selection or synchronization.

Exit criteria:

- a stream-ordering test fails when the dependency is removed and passes with
  the production event contract;
- no device-wide synchronize is used in steady-state serving;
- race-focused stress tests show no partial or stale KV consumption.

### 5. Specify failure and fallback semantics before partial enqueue

Capability rejection can safely return to span-copy before the first gather is
submitted. Runtime failure is harder: if K succeeds and V submission fails, a
naive fallback can race with partially enqueued work or overwrite only part of
the layer.

Required behavior:

- preflight the complete layer before enqueue: op availability, dtype, layout,
  shapes, index tensors, registered mapping ownership, and destination bounds;
- distinguish capability fallback from runtime execution failure;
- allow span-copy fallback only before mapped gather has modified/enqueued any
  destination part, unless an explicit synchronization and full-layer retry
  protocol is implemented;
- treat failures after partial enqueue as a controlled worker/request failure,
  not as an invisible fallback;
- surface structured fallback/error reasons and counters;
- test registration failure, workspace allocation failure, ACLNN lookup failure,
  K/V partial failure, and shutdown during failure handling.

## P1: Hot-path And Compatibility Debt

### 6. Remove repeated index-tensor construction from the layer hot path

The direct mapped path currently derives `src_min/src_max` and creates NPU
`src_block_ids` and `dst_block_ids` in `_make_host_gather_indices()` for each
layer load. This control-plane cost was visible enough to receive dedicated
instrumentation, but has not yet been optimized for serving.

Required work:

- measure index construction/copy cost in real prefix-hit workloads;
- build the request/batch restore plan once when connector metadata is bound;
- reuse device index buffers across layers when the mapping is identical;
- use bounded reusable buffers for changing mappings rather than allocating new
  tensors per layer;
- preserve correct lifetime until the load stream has consumed the indices;
- keep the span-count/span-length profile in the same prepared plan so backend
  selection does not repeat mapping analysis.

### 7. Pre-register the correct CPU allocation, not incidental slices

Direct gather currently slices each CPU part from `src_min` through `src_max` and
the C++ wrapper lazily registers that span. For sparse IDs, this may map unused
blocks between the minimum and maximum. Across changing requests it can also
produce overlapping or differently sized registration attempts.

Required work:

- identify the true long-lived CPU KV backing arena for each layer/part;
- register those arenas during worker/cache initialization;
- store registration handles beside the owning cache allocation;
- pass only views contained in a known registered arena to mapped gather;
- reject temporary tensors and unknown allocations from the production fast
  path;
- measure registration time, mapped bytes, NUMA placement, and page alignment at
  initialization;
- determine whether one registration per layer/part, one per shared-memory slab,
  or another bounded layout is best for the actual allocator.

### 8. Keep workspace behavior stable, then optimize independently

The op-host tiling currently requests 16 MiB and the Torch wrapper allocates an
NPU byte tensor of that size per invocation. The device kernel does not currently
dereference workspace, but zero-workspace behavior has not been accepted as a
contract for this port.

Production sequence:

1. retain the measured workspace behavior through the first serving A/B;
2. measure allocation/cache behavior and peak memory under concurrency;
3. separately evaluate executor/workspace reuse or a verified tiling change;
4. repeat correctness and performance gates after any workspace contract change.

Workspace removal is an optimization opportunity, not a prerequisite for the
first production-shaped integration.

### 9. Harden backend selection and compatibility checks

Mapped gather must remain opt-in until the compatibility matrix is known. The
selector should make a single explainable decision per restore plan and retain
span-copy for unsupported cases.

The decision must account for:

- transfer direction;
- SoC and CANN version;
- opapi symbol/version availability;
- CPU and NPU dtype equality (`float16`, `bfloat16`, or `float32` today);
- contiguous layout and equal per-block payload shape;
- standard attention versus MLA layouts;
- TP/PP rank-local cache layout;
- zero blocks, duplicate IDs, non-monotonic IDs, and bounds;
- registration ownership and device/context match;
- request size and any evidence-based policy threshold.

Every rejection needs a stable reason code, not only a once-per-process log
message.

### 10. Make observability cheap, bounded, and operationally useful

The current JSONL instrumentation is suitable for an experiment but opens and
appends to a file for individual events. Production metrics must not turn the
transfer hot path into synchronous file I/O.

Required counters/histograms include:

- selected backend and fallback reason;
- blocks, spans, span-length distribution, and logical bytes;
- index preparation, registration, enqueue, and completion latency;
- mapping count, current/peak mapped bytes, hits/misses, and unregister results;
- workspace bytes and allocation failures;
- load-stream wait and staging-slot wait;
- validation failures and partial-enqueue failures;
- request latency, TTFT, ITL, scheduler stalls, and throughput deltas.

Use the project's metrics/logging path or buffered sampling. Keep raw JSONL
tracing explicitly diagnostic and disabled by default.

## P2: Production Evidence Debt

### 11. Complete correctness and lifecycle coverage

Required focused tests:

- direct op smoke for fp16, bf16, and fp32;
- randomized source/destination mappings, including one-span and fully fragmented
  cases;
- duplicates, reverse order, empty mapping, out-of-range IDs, and mismatched
  shapes/dtypes;
- direct mapped backend versus span-copy exact output comparison;
- MLA and non-MLA KV layouts;
- TP/PP rank-local layouts;
- mapping cache containment, overlap, generation, and per-owner unregister;
- repeated worker construction/shutdown and engine restart;
- request cancellation and exception teardown;
- multiprocessing shared-memory allocation, registration, close, and unlink;
- registration/unregister/workspace/opapi fault injection;
- long-running allocation churn with bounded registration state;
- asynchronous stream ordering under concurrent model work.

### 12. Run serving-level A/B gates

The microbenchmark proves that the direction is worth integrating; it does not
prove a default production policy.

The serving gate must compare the same workload and device allocation with:

1. main span-copy;
2. direct mapped-host gather;
3. optionally, the worker-local staging-pool alternative.

Workloads must include:

- real CPU prefix-cache hits and partial-prefix hits;
- prefill-heavy and decode-heavy traffic;
- long contexts and realistic restored block counts;
- observed production block mapping/span distributions;
- concurrent requests and model-compute contention;
- TP and MLA configurations used in deployment;
- sustained runs long enough to expose mapping or shared-memory lifecycle bugs.

Report at least correctness, TTFT/ITL percentiles, request throughput, scheduler
stalls, transfer overlap, NPU memory, mapped host bytes, CPU utilization, and
fallback frequency.

Mapped gather may become the default only if the serving result remains positive
and the P0 lifecycle/packaging gates are complete. Otherwise it remains an
explicit experimental backend with span-copy as default.

## Proposed Implementation Order

| Phase | Work | Exit criterion |
| --- | --- | --- |
| 0 | Preserve current direct smoke and microbenchmark | Existing evidence remains reproducible from a clean build |
| 1 | Self-contained wheel/custom-op packaging | No manual opapi/vendor path is required |
| 2 | Owned mapping registry and explicit handles | No stale mappings; bounded churn; per-worker unregister works |
| 3 | Connector teardown and stream/event contract | Shutdown safely drains, unregisters, and releases memory |
| 4 | Preflight and failure/fallback semantics | No unsafe fallback after partial enqueue |
| 5 | Pre-register real CPU KV arenas | No lazy registration in the layer/request hot path |
| 6 | Reusable restore plans and index buffers | Per-layer metadata allocation is removed or shown negligible |
| 7 | Compatibility and lifecycle test matrix | dtype/layout/TP/MLA/restart/fault tests pass |
| 8 | Real serving A/B | TTFT/ITL/throughput and correctness gates pass |
| 9 | Optional workspace optimization | Isolated change passes all earlier gates again |

## Decisions Still Needed

These are architectural decisions, not implementation details to choose
silently:

1. **Mapping owner:** connector worker, CPU KV allocation manager, or a shared
   process service with explicit owner handles.
2. **Registered unit:** full shared-memory slab, per-layer/part arena, or another
   bounded allocator-owned region.
3. **Primary backend:** direct mapping of the production CPU cache versus the
   worker-local staging pool. Both may remain for A/B, but production ownership
   should not be duplicated indefinitely.
4. **Failure after partial enqueue:** fatal request/worker error versus an
   explicitly synchronized full-layer retry.
5. **Canonical package vendor name and loader contract.**
6. **Default-enablement gate:** exact supported SoC/CANN/layout set and the
   serving-level performance margin required to turn the backend on by default.

## Explicitly Out Of Scope For The First Production Gate

- a virtual-memory page-table redesign;
- a mapped-host NPU-to-CPU scatter operator;
- assuming or forcing zero workspace without a separate verified change;
- replacing the existing CPU prefix-cache allocator;
- removing span-copy fallback;
- enabling mapped gather unconditionally from direct-op microbenchmark results
  alone.

## Related Notes

- [`main_port_scope_20260709.md`](../porting/main_port_scope_20260709.md)
- [`mapped_host_gather_vs_span_copy_20260718.md`](../experiments/mapped_host_gather_vs_span_copy_20260718.md)
- Historical detailed smoke/integration note:
  `vllm-hust/experiment/device-kv-gather:branch_development_notes/notes/engineering/device-kv-gather-smoke-status.md`
- Historical production gap report:
  `vllm-hust/experiment/device-kv-gather:branch_development_notes/work/experiment-matrix-gap-report.md`
