2# CPU KV Cache SharedMemory Rationale Notes

Generated on 2026-06-17.

## Question

Why does the legacy `CPUOffloadingConnector` allocate CPU KV cache through
Python `multiprocessing.shared_memory.SharedMemory` instead of letting each
worker allocate normal host memory or pinned host memory locally?

Short answer: the design is not necessary for generic device swap. It appears
to be an artifact of the legacy connector architecture, whose goal was a shared
CPU prefix-cache pool coordinated by a single metadata server. SharedMemory was
used because it is easy to pass through ZeroMQ/pickle as a re-openable memory
handle, and because some ranks intentionally share the same CPU KV backing
storage.

## Historical Origin

`git blame` traces this design back to:

- `0f3939e5 [Feature]cpu offload connector (#1659)`
- Commit date: 2025-09-23
- Commit message: "This PR implements cpu offload connector to enable NPU kv
  cache offload to host DRAM."

The original implementation already used:

- a `MetadataServer`
- ZeroMQ RPC
- `SharedMemory`
- a server-side `CPUKVCacheManager`
- worker-side `torch.frombuffer(shm.buf, dtype=...).reshape(layer_size)`

So this is not caused by the mapped-host gather prototype. The prototype is
layered on top of an older host-DRAM offload design.

## What SharedMemory Buys This Connector

### 1. A ZeroMQ/pickle-friendly large-buffer handle

The code comment says:

```python
# only this format can share during ZeroMQ+pickle
```

The server returns `SharedMemory` objects from `init_cpu_kv_caches`. The worker
client receives them through the RPC response and reconstructs tensors with
`torch.frombuffer`.

A normal `torch.Tensor` would be a bad RPC payload for this purpose because the
large backing storage would be copied or serialized in an unsuitable way.
SharedMemory gives the RPC layer a small, re-openable object that refers to a
large host buffer.

This is a practical reason, not a fundamental device-memory reason.

### 2. One metadata server owns CPU KV allocation and cleanup

Only one worker starts the metadata server:

```python
if data_parallel_rank == 0 and tp_rank == 0 and pp_rank == 0:
    self.init_metadata_server(config)
```

The metadata server:

- computes available CPU swap space
- creates one shared-memory object per `(pp_rank, tp_rank, layer_name)` logical
  CPU KV cache
- stores those objects in `self.shared_memory`
- unlinks them during `shutdown`
- owns the `CPUKVCacheManager` that decides block allocation, cache hits, touch,
  and free behavior

This centralizes the block-id namespace and host-buffer lifecycle. It also
means all schedulers/workers talk to the same authority for CPU prefix-cache
state.

### 3. DP replicas appear to share one CPU prefix-cache pool

The server startup comment says all DP ranks share the same metadata server.
The shared-memory key is only `(pp_rank, tp_rank)`, not `(dp_rank, pp_rank,
tp_rank)`.

That implies data-parallel replicas can map the same CPU KV cache for a given
pipeline/tensor-parallel shard. Because DP replicas run the same model, a cached
prefix KV value is reusable across replicas in principle.

This explains why the design is not just "each worker swaps its own KV to its
own DRAM." It is closer to a node-local CPU prefix cache shared across DP
replicas.

### 4. MLA explicitly shares CPU KV cache across TP ranks

In `MetadataServer.init_cpu_kv_caches`:

```python
if use_mla:
    tp_rank = 0
```

The comment says MLA shares the same KV cache among different TP ranks. That
means all TP ranks for the same PP rank can receive the same shared-memory
object.

The save path also has MLA-specific rank splitting:

```python
if self.use_mla:
    start, step = self.tp_rank, self.tp_world_size
```

So SharedMemory has a real data-sharing purpose for MLA: multiple TP worker
processes can cooperate on and/or read the same CPU KV backing store.

## Why This Still Feels Wrong For Device KV Gather

The mapped-host gather direction is about letting device-side code access host
KV pages efficiently. That does not inherently require Python SharedMemory.

For a local device swap path, better host backing could be:

- normal worker-owned pinned host tensors
- `aclrtMallocHost`-backed buffers
- a custom allocator that is both host-register-safe and shareable only if
  sharing is truly needed
- per-device CPU KV buffers owned by the worker/offload handler

The newer `vllm_ascend/kv_offload/cpu_npu.py` path already allocates CPU tensors
inside the worker with:

```python
torch.zeros(..., device="cpu", pin_memory=pin_memory)
```

That newer design does not use Python SharedMemory. It precomputes CPU/NPU base
pointers and uses batched copies. This is a much more natural foundation for a
per-device swap path, and probably a better place to integrate mapped-host
device gather if the goal is production offload rather than legacy connector
compatibility.

## When SharedMemory Is Actually Justified

SharedMemory is justified if the product requirement is:

- one CPU prefix-cache pool shared across DP replicas
- one MLA CPU KV cache shared across TP ranks
- one central metadata server that owns block allocation and cache state
- ability to hand large host buffers to worker processes over ZeroMQ without
  copying them

In that architecture, SharedMemory is a reasonable Python-level building block.
It solves cross-process access and cleanup in a straightforward way.

## When SharedMemory Is Not Necessary

SharedMemory is not necessary if the requirement is only:

- "swap my device KV blocks to host and later read them back on the same worker"
- "accelerate H2D load with mapped-host gather"
- "avoid DRAM copies for device-facing movement"
- "benchmark a device-side random access path"

For those cases, SharedMemory adds friction:

- it is not known-safe pinned/locked host memory from CANN's perspective
- it complicates `aclrtHostRegister` ownership
- it couples the fast path to a deprecated connector
- it makes per-device/context mapping lifetime harder to reason about

## Production Interpretation

The current design has two separate ideas tangled together:

1. Legacy CPU prefix-cache pool:
   - global metadata server
   - shared block manager
   - CPU cache reuse across DP and MLA TP ranks
   - Python SharedMemory as cross-process backing storage

2. New mapped-host device gather:
   - device-side random access to host KV pages
   - host registration and mapped device pointers
   - stream and long-running resource safety

For production, these should be separated.

If the target feature is a shared CPU prefix-cache pool, keep a shared storage
abstraction but replace raw Python SharedMemory with a memory owner that can
also own host registration and unregister cleanly.

If the target feature is fast local CPU/NPU offload, avoid the old
`CPUOffloadingConnector` and prototype the mapped-host gather inside the newer
worker-local CPU offload path.

## Bottom Line

There are reasons this code used SharedMemory, but they are architectural
reasons from the legacy shared CPU prefix-cache connector, not requirements of
device KV gather itself.

For mapped-host gather, Python SharedMemory is probably the wrong default host
allocation. It is useful only if we explicitly want cross-process shared CPU KV
storage. Otherwise, worker-owned pinned or CANN-managed host memory should be a
cleaner and safer production direction.
