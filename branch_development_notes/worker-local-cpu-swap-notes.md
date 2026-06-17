# Worker-Local CPU Swap Notes

Generated on 2026-06-17.

## Thesis

CPU KV swap space should probably be worker-local, not centrally owned by the
scheduler or a node-global metadata server.

The scheduler needs to know scheduling facts: which request has a prefix hit,
how many logical blocks are needed, and which transfer jobs are pending. It does
not necessarily need to own the host DRAM backing store or decide concrete host
addresses/block ids for every worker.

This matters more for mapped-host gather, because host memory ownership,
registration, unregister, stream synchronization, and device/context affinity
are all local resource-lifetime problems.

## What The Legacy Connector Does Today

The legacy `CPUOffloadingConnector` splits responsibilities like this:

- Scheduler-side `CPUOffloadingConnectorScheduler` asks a metadata server for
  CPU prefix hits through `get_matched_num_and_touch`.
- It asks the same server for CPU block ids through `allocate_slots`.
- It embeds `cpu_block_ids` and `gpu_block_ids` into connector metadata.
- Worker-side `CPUOffloadingConnectorWorker` only receives those ids and copies
  between `self.cpu_kv_caches` and `self.gpu_kv_caches`.
- The metadata server owns the `CPUKVCacheManager`, the shared-memory segments,
  and the CPU block namespace.

That means CPU swap allocation is not local to the worker. The worker writes
into CPU memory chosen by a remote manager.

## Why That Feels Over-Centralized

Host DRAM is generally abundant compared with NPU HBM. For a single server,
giving each worker a CPU swap quota is simpler and more robust than globally
allocating every CPU KV block.

The centralized design increases the failure surface:

- If the metadata server hangs, worker CPU swap allocation hangs.
- If the server's block manager state diverges from worker transfer state,
  correctness becomes hard to recover.
- Cleanup is split across worker-side tensor views, server-side SharedMemory,
  request lifecycle metadata, and RPC state.
- Host registration for mapped device access cannot be naturally owned by the
  process that owns the underlying memory.

For mapped-host gather specifically, worker-local ownership is cleaner because
the same worker process can own:

- the CPU swap tensor/storage
- the `aclrtHostRegister` mapping
- the mapped device pointer cache
- stream synchronization for readers/writers
- unregister and cleanup

## What Centralization Was Trying To Buy

The old design has one legitimate goal: a node-local shared CPU prefix cache.

Evidence:

- Only one metadata server is started on
  `data_parallel_rank == 0 && tp_rank == 0 && pp_rank == 0`.
- Comments say all DP ranks share the same metadata server.
- Shared-memory keys are `(pp_rank, tp_rank, layer_name)`, not DP-specific.
- For MLA, `tp_rank` is forced to `0`, explicitly sharing CPU KV across TP
  ranks.
- The central `CPUKVCacheManager` uses vLLM block-pool logic for prefix-cache
  hits, touches, caching, and freeing.

So the central design is not completely arbitrary. It is trying to make CPU
DRAM act like a shared prefix-cache tier.

The question is whether that goal is worth the reliability and memory-lifetime
cost.

## Local Autonomy Design

A worker-local CPU swap design would look like this:

1. Each worker receives a CPU swap quota:
   - `num_cpu_blocks`
   - or `cpu_swap_space_gb / local_worker_count`
   - possibly adjusted by PP/TP/DP topology

2. Each worker allocates its own CPU KV backing store:
   - pinned host tensor
   - CANN host allocation
   - or another host-register-safe allocator

3. Each worker owns its CPU block allocator:
   - local free list
   - local request-to-CPU-block table
   - local LRU or prefix-cache metadata if needed

4. Scheduler emits logical transfer intent:
   - request id
   - source GPU block ids
   - destination GPU block ids
   - number of cached/computed tokens

5. Worker resolves local CPU placement:
   - choose CPU block ids
   - copy/gather locally
   - report transfer completion and optional cache events

6. Worker owns mapped-host lifecycle:
   - register CPU swap memory at initialization or by bounded windows
   - reuse mapped pointers safely
   - unregister at shutdown

This keeps the reliability boundary local. The scheduler can retry or drop a
worker without carrying a global CPU block heap that must remain consistent with
that worker's private memory.

## Prefix Cache Question

The hard part is prefix-cache hit discovery.

For local autonomy, there are two options:

### Option A: Per-worker CPU prefix cache

Each worker only serves CPU prefix hits from its own local CPU swap pool.

Pros:

- simplest ownership model
- no SharedMemory
- no cross-worker coherency
- best fit for mapped-host gather
- failure is isolated to one worker

Cons:

- DP replicas do not share CPU prefix cache entries
- lower hit rate if requests are load-balanced randomly
- scheduler may need affinity/stickiness to route similar prefixes to the same
  worker

This is probably the right starting point for productionizing mapped-host
gather.

### Option B: Shared metadata, local storage

Keep a lightweight shared prefix index, but let each worker own storage.

The shared index maps:

```text
prefix hash -> worker id -> local CPU block ids / availability
```

The scheduler can prefer a worker that already has a CPU prefix hit. Actual host
memory remains worker-local.

Pros:

- preserves some DP-level prefix reuse
- avoids shared host backing storage
- still keeps host registration local

Cons:

- scheduler/load-balancer integration is more complex
- needs eviction notifications
- remote worker hits are useful only if the scheduler can route the request to
  that worker

This is a better architecture than central SharedMemory if cross-worker CPU
prefix reuse is important.

### Option C: Central shared CPU pool

This is roughly the legacy connector design.

Pros:

- maximizes reuse across local DP workers
- one block manager owns global CPU prefix cache state

Cons:

- shared-memory lifecycle complexity
- central metadata server reliability risk
- mapped-host registration ownership is awkward
- hard to reason about long-running cleanup
- deprecated connector path

This option looks least attractive for device KV gather unless shared CPU
prefix reuse is the primary product requirement.

## Relationship To New vLLM CPUOffload Path

The newer `vllm_ascend/kv_offload/cpu_npu.py` path already behaves closer to
local autonomy:

- `NPUOffloadingSpec` accepts a configured `num_cpu_blocks`.
- Its worker-side handler allocates CPU tensors in the worker process.
- It uses `pin_memory=is_pin_memory_available()`.
- It precomputes local CPU/NPU base pointers and block sizes.
- It manages transfer streams, events, and in-flight transfer queues locally.

It still has a scheduler-side `CPUOffloadingManager`, but that manager does not
own Python SharedMemory segments. This split is much healthier:

- scheduler-side: offload/load planning and logical block tracking
- worker-side: concrete memory and transfer execution

Mapped-host gather should probably target this path first.

## Practical Recommendation

Treat the legacy `CPUOffloadingConnector` as a prototype/reference for prefix
cache semantics, not as the production memory owner.

For production mapped-host gather:

1. Start with worker-local CPU swap.
2. Allocate CPU KV backing in the worker with a host-register-safe allocator.
3. Register full CPU swap tensors once per worker/device.
4. Add mapped-host gather as an alternative H2D load implementation in the
   worker-local offload handler.
5. Measure against `swap_blocks_batch`.
6. Only add cross-worker/shared prefix index later if local hit rate is not
   enough.

This gives the simplest reliability story:

- one worker owns one CPU swap region
- one worker owns its mapped host registrations
- one worker unregisters its resources
- scheduler failure does not imply leaked host mappings
- metadata failure does not corrupt host allocation state

## Bottom Line

The "地方自治" model is not just cleaner philosophically; it lines up with the
hard resource-lifetime constraints of mapped-host memory.

Centralized CPU swap only makes sense if the main objective is shared CPU prefix
cache reuse across ranks. For device KV gather, local ownership should be the
default, and shared/global metadata should be optional scheduling intelligence
rather than the owner of DRAM.
