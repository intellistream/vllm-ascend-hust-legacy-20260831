# Global Shared CPU KV Pool Design Review

Generated on 2026-06-17.

## Position

I am not convinced that a global shared-memory CPU KV pool is the right
abstraction for production mapped-host KV gather.

The default design should be:

- worker-owned CPU swap pools
- prefix-aware routing
- recomputation when a worker-local prefix cache is lost
- optional shared metadata for routing hints, not shared ownership of DRAM

The burden of proof should be on the global shared-memory pool design, because
it introduces distributed shared state into a path that can otherwise be local.

## Main Concern

The global pool seems to optimize cross-worker prefix-cache reuse, but it pays
for that with a much larger systems problem:

- ownership tracking
- lifecycle management
- synchronization
- fault recovery
- metadata consistency
- debugging complexity
- host registration ownership
- cleanup ordering

Prefix cache is a performance optimization, not persistent correctness state.
That changes the design calculus. Losing cached KV should usually degrade to
recompute, not require a distributed recovery protocol.

## TP Case

For TP, especially MLA + TP, KV ownership is already rank-local during serving.
Each rank consumes its own KV shard.

The legacy connector has a special MLA behavior where `tp_rank` is collapsed to
0 for CPU KV shared memory, and the save path stripes work across TP ranks. But
this does not prove that a global shared-memory pool is the right abstraction.

For mapped-host gather, each rank ultimately needs device-readable access to
the KV it will consume. Worker/rank-local CPU swap is the natural owner:

- local NPU context
- local host registration
- local mapped pointer cache
- local stream ordering
- local cleanup

If TP ranks need coordinated CPU prefix cache behavior, that coordination can be
represented as metadata or deterministic placement. It does not require Python
SharedMemory as a global backing store.

## DP Case

DP reuse is the strongest argument for sharing, but prefix-aware routing can
capture much of the value with less complexity.

Instead of moving KV blocks across workers or making all workers share one host
pool, route a request to a worker that already owns the prefix. This follows the
usual "move compute to data" principle.

The router can score candidate workers using:

- prefix hit length
- queue depth
- NPU memory pressure
- active request count
- utilization
- expected recompute cost

In this model, prefix hit rate is one scheduling signal among many. It does not
force a global memory pool into the worker runtime.

## Worker Failure

If a worker dies, its worker-local CPU prefix cache dies with it.

That is acceptable unless measurements show recomputation is too expensive for
the target workload. The fallback behavior is simple:

- remove the worker from routing
- discard its prefix index entries
- route future requests elsewhere
- recompute missing prefixes when needed

This is much simpler than recovering a shared pool whose data, metadata,
reference counts, and in-flight transfers may now disagree.

## Load Balancing

Load balancing should live at the routing layer, not in global DRAM ownership.

Device memory pressure, queue depth, utilization, and prefix hit rate can all be
inputs to routing. A router can intentionally choose a lower prefix hit if the
owning worker is overloaded.

That keeps the core worker memory model simple:

- worker owns HBM KV cache
- worker owns CPU swap cache
- worker owns host registration
- worker reports capacity and prefix availability

Global coordination stays advisory rather than becoming part of correctness.

## Why Global SharedMemory Is Especially Awkward For Mapped-Host Gather

Mapped-host gather tightens the lifetime rules around host memory.

The owner of host memory should also own:

- `aclrtHostRegister`
- mapped device pointer cache
- device/context association
- stream synchronization
- `aclrtHostUnregister`

Python SharedMemory plus a central metadata server splits these responsibilities
across processes. That makes the failure modes larger than the performance
problem being solved.

For production mapped-host gather, a worker-local host allocation is a better
fit:

- pinned host tensor
- CANN-managed host allocation
- or a custom worker-local allocator that can be registered once and
  unregistered deterministically

## When A Global Pool Might Be Justified

A global shared CPU KV pool should require evidence, not just intuition.

It might be justified if measurements show all of the following:

- high cross-worker prefix reuse that routing cannot capture
- large recompute cost relative to routing or transfer cost
- low locality/stickiness in real traffic
- acceptable complexity for fault handling and cleanup
- bounded mapped-host registration lifetime under churn
- clear benefit over worker-local CPUOffload plus prefix-aware routing

Without that evidence, global shared memory is likely premature architecture.

## Recommended Default

Use worker-local CPU swap pools first.

Implementation direction:

1. Give each worker a CPU swap quota.
2. Allocate worker-owned host KV buffers.
3. Register host buffers for mapped-device access in the same worker.
4. Add mapped-host gather to the worker-local offload path.
5. Expose prefix availability and pressure metrics to the router.
6. Implement prefix-aware routing.
7. Recompute on cache loss.

Only add global shared storage if data proves routing is insufficient.

## Measurement Plan

To decide objectively, compare three designs:

1. Worker-local swap with no prefix-aware routing.
2. Worker-local swap with prefix-aware routing.
3. Global shared CPU KV pool.

Measure:

- end-to-end latency
- TTFT for repeated prefixes
- throughput under mixed reuse/load
- prefix hit rate
- recompute rate
- tail latency under worker failure
- host memory usage
- mapped registration count and total registered bytes
- operational/debug complexity observed during stress runs

The global pool should only win if it provides a substantial improvement that
cannot be recovered by routing.

## Bottom Line

We should avoid solving a relatively small problem, cross-worker cache reuse, by
introducing a much larger one, distributed shared state management.

The robust baseline is:

- worker-owned swap pools
- prefix-aware routing
- recomputation when cache is lost

That baseline aligns better with mapped-host gather and should be the production
default unless benchmarks clearly disprove it.
