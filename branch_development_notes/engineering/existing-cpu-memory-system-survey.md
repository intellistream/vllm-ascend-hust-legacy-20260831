# Existing CPU-Side KV Memory System Survey

Generated on 2026-06-17.

## Scope

This note surveys the existing CPU-side KV memory paths in this branch as input
for a worker-owned DRAM swap subsystem.

The focus is:

- where the current CPU memory code lives
- which components own memory, metadata, and transfer execution
- what scheduler/worker protocols already exist
- what constraints a production worker-local DRAM subsystem must preserve or
  deliberately replace

One caveat: the current Python environment does not have upstream `vllm`
installed, so upstream classes such as `CPUOffloadingManager`,
`OffloadingSpec`, `OffloadingHandler`, `CPULoadStoreSpec`, and
`GPULoadStoreSpec` could not be inspected by import. Their behavior below is
therefore inferred from vLLM Ascend usage and should be rechecked in the vLLM
workspace before implementation kickoff.

## Source Distribution

### Newer vLLM KV offload integration

- `vllm_ascend/kv_offload/npu.py`
  - Defines `NPUOffloadingSpec`.
  - Bridges upstream vLLM `OffloadingConnector` to Ascend-specific
    CPU/NPU transfer code.
  - Requires `num_cpu_blocks` in `kv_connector_extra_config`.
- `vllm_ascend/kv_offload/cpu_npu.py`
  - Defines `CpuNpuOffloadingHandler`.
  - Allocates worker-local CPU tensors.
  - Owns NPU transfer streams, events, in-flight queues, and the
    `swap_blocks_batch` copy path.
- `tests/e2e/singlecard/test_cpu_offloading.py`
  - Exercises `OffloadingConnector` with `NPUOffloadingSpec`.
  - Configures `num_cpu_blocks`, connector `block_size`, `spec_name`, and
    `spec_module_path`.

This path is the closest existing match for a worker-local DRAM subsystem.

### Legacy SharedMemory CPU offload connector

- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`
  - Implements `CPUOffloadingConnector`, scheduler side, and worker side.
  - Uses a central metadata server for CPU block allocation and prefix lookup.
  - Worker side reconstructs CPU tensors from shared-memory handles.
  - Contains the experimental mapped-host gather load path.
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/metadata.py`
  - Owns `MetadataServer`, ZMQ RPC, and Python `SharedMemory` allocation.
  - Creates shared CPU KV tensors and exposes them to workers.
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py`
  - Wraps vLLM `BlockPool` and `single_type_manager`.
  - Owns prefix-cache block lookup, touch, allocation, cache, and free
    metadata.

This path is useful as a compatibility reference, but its physical ownership
model is not a good production target for mapped-host or worker-local DRAM.

### External KV pool / AscendStore path

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
  - Worker-side bridge to backends such as Mooncake, Memcache, and Yuanrong.
  - Registers NPU KV cache address ranges with the backend.
  - Starts send/receive worker threads.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
  - Scheduler-side lookup and metadata construction.
  - Tracks request token progress, load specs, preemption, and delayed frees.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`
  - Defines key namespace data, token chunking, request metadata, load specs,
    and block address calculations.
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`
  - Implements sending/receiving threads, key existence checks, and backend
    get/put calls.

This path is not a local CPU swap allocator. It is still relevant because it
has a mature-ish request metadata protocol, key namespace construction, async
threading, and completion handling.

### Native transfer and mapped-host prototype

- `csrc/torch_binding.cpp`
  - `swap_blocks_batch`: CPU pointer-array driven batched H2D/D2H/D2D copy.
  - `kv_cache_block_gather`: experimental mapped-host gather op.
  - `get_mapped_host_device_ptr`: page-aligns host memory, calls
    `aclrtHostRegister(... ACL_HOST_REGISTER_MAPPED ...)`, and caches mapped
    ranges in process-global static state.
- `vllm_ascend/envs.py`
  - Defines relevant environment variables:
    - `VLLM_ASCEND_ENABLE_BATCH_MEMCPY`
    - `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER`
    - `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_LIB`
    - `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB`

The native copy path is important for Phase 1 copy-path parity. The mapped-host
prototype is important for later backend integration, but its lifecycle model
must not be reused as-is.

## Path 1: Newer Worker-Local CPU/NPU Offload

### High-level shape

`NPUOffloadingSpec` is an Ascend implementation of upstream vLLM's generic KV
offload abstraction.

Scheduler side:

- `NPUOffloadingSpec.get_manager()` constructs upstream
  `CPUOffloadingManager`.
- It passes `block_size = gpu_block_size * block_size_factor`.
- It passes `num_blocks = num_cpu_blocks`.
- Optional KV cache events are enabled from `kv_events_config`.

Worker side:

- `NPUOffloadingSpec.get_handlers()` constructs one
  `CpuNpuOffloadingHandler`.
- It yields bidirectional handlers:
  - `GPULoadStoreSpec -> CPULoadStoreSpec`
  - `CPULoadStoreSpec -> GPULoadStoreSpec`
- The same handler owns both save and load directions.

### CPU arena layout

`CpuNpuOffloadingHandler` receives:

- `gpu_block_size`
- `cpu_block_size`
- `num_cpu_blocks`
- `gpu_caches`
- `attn_backends`

It requires:

```text
cpu_block_size % gpu_block_size == 0
```

Then:

```text
block_size_factor = cpu_block_size // gpu_block_size
```

For every GPU KV cache tensor:

- take `gpu_tensor[0].shape`
- replace dimension 0 with `num_cpu_blocks * block_size_factor`
- allocate two CPU tensors with the same dtype:
  - one for key
  - one for value
- use `pin_memory=is_pin_memory_available()`

The public CPU block id in this path is not a raw row in the CPU tensor. A CPU
block may correspond to `block_size_factor` rows of the allocated CPU tensor.
This is why `expand_block_ids()` maps one logical block id into multiple
sub-block ids.

### Transfer protocol

The upstream transfer spec is inferred from usage:

```text
TransferSpec = (src_spec, dst_spec)

src/dst can be:
- CPULoadStoreSpec(block_ids=np.ndarray)
- GPULoadStoreSpec(block_ids=np.ndarray)
```

The handler decides direction by source/destination spec types:

- CPU -> NPU:
  - stream: `h2d_stream`
  - direction passed to native op: `0`
- NPU -> CPU:
  - stream: `d2h_stream`
  - direction passed to native op: `1`

For each transfer:

1. Validate source and destination block id arrays are 1D.
2. Expand logical block ids into sub-block ids according to source and
   destination block-size factors.
3. Build flat pointer arrays for every `(layer, kv_part, block_pair)`.
4. Convert pointer arrays to CPU int64 tensors.
5. Record start/end NPU events on the transfer stream.
6. Submit `torch.ops._C_ascend.swap_blocks_batch`.
7. Append a `Transfer` record to the in-flight queue.

The code serializes transfers per direction:

- D2H waits for the current model stream before reading NPU KV.
- Each new transfer waits for the previous transfer's end event in the same
  direction.
- H2D and D2H have separate streams and queues.

Completion is polled by `get_finished()`:

- query the oldest end event in each direction
- pop completed transfers
- report `TransferResult(job_id, success=True, transfer_size, transfer_time,
  transfer_type)`
- recycle events through an event pool

Blocking wait is available through `wait(job_ids)`, which synchronizes matching
end events.

### Properties worth preserving

- Worker-local CPU tensor ownership already exists.
- Transfer streams and events are worker-owned.
- The baseline copy backend already supports batched per-layer/per-part copy.
- Copy-path execution is independent of mapped-host registration.
- Transfer completion already has an async polling model.
- `block_size_factor` gives a useful precedent for decoupling scheduler block
  size from CPU transfer granularity.

### Constraints and gaps

The current handler is a transfer engine, not a full DRAM memory subsystem.

Missing or under-specified:

- no explicit CPU block state machine
- no active swap vs prefix cache distinction
- no generation-safe CPU block references
- no GPU/NPU block pinning contract
- no admission control or quota policy inside the worker
- no eviction policy inside the worker
- no request epoch or cancellation model visible here
- no worker shutdown cleanup beyond Python object lifetime
- no per-transfer ownership of source/destination block refs
- no explicit validation that upstream scheduler block ids still refer to the
  same logical request state

Potential lifetime concern:

- `transfer_async()` creates CPU tensors for pointer arrays locally and passes
  them to `swap_blocks_batch`.
- The non-batch native path reads those arrays synchronously while submitting
  individual `aclrtMemcpyAsync` calls.
- The `aclrtMemcpyBatchAsync` path passes array pointers into CANN. It is not
  clear from this code whether CANN copies descriptor arrays before returning.
  A production path should either confirm this contract or retain descriptor
  tensors until the transfer end event completes.

## Path 2: Legacy SharedMemory CPUOffloadingConnector

### Activation and roles

`CPUOffloadingConnector` is a vLLM KV connector.

It only initializes scheduler/worker components when prefix caching is enabled:

```text
if not enable_prefix_caching:
    connector_scheduler = None
    connector_worker = None
```

Otherwise:

- scheduler role creates `CPUOffloadingConnectorScheduler`
- worker role creates `CPUOffloadingConnectorWorker`

This is already an important semantic signal: the legacy connector is tied to
prefix-cache behavior, not just generic active CPU swap.

### Scheduler-side protocol

Scheduler state:

- `num_gpu_computed_tokens[request_id]`
- `num_cpu_computed_tokens[request_id]`
- `allocated_req_ids`
- `finished_req_ids`
- ZMQ RPC client to metadata server
- optional `swap_in_threshold`

Protocol:

1. `get_num_new_matched_tokens(request, num_computed_tokens)`
   - deep-copies the request
   - disables `get_hash_new_full_blocks` before pickling/RPC
   - calls `metadata.get_matched_num_and_touch(request)`
   - records GPU and CPU computed token counts
   - returns extra CPU-hit tokens only if they exceed `swap_in_threshold`

2. `update_state_after_alloc(request)`
   - records that the scheduler allocated blocks for this request

3. `build_connector_meta(scheduler_output)`
   - computes target token counts for scheduled new and cached requests
   - identifies unallocated requests and asks metadata server to free them
   - calls `metadata.allocate_slots(num_tokens, unallocated_req_ids)`
   - builds `CPUOffloadingConnectorMetadata`
   - per request metadata includes:
     - GPU block ids
     - CPU block ids
     - scheduled token counts
     - GPU-computed token count
     - CPU-computed token count

4. `request_finished(request)`
   - records request as finished
   - calls `metadata.record_request_cache_and_free_slots(request)`
   - returns `(True, None)`, meaning the request can be considered delayed by
     connector completion semantics

### Worker-side protocol

Worker state:

- `requests: dict[str, ReqMeta]`
- `load_stream`
- `save_stream`
- ZMQ RPC client
- `load_block_mapping: list[(cpu_block_id, gpu_block_id)]`
- background save thread
- queues for save input/output
- TP coordination counters
- shared-memory CPU KV tensors returned by metadata server

`register_kv_caches(kv_caches)`:

- stores worker GPU KV cache tensors
- computes KV cache spec from model layers
- sends `(pp_rank, tp_rank, kv_cache_spec, mla_config)` to metadata server
- receives shared-memory tensors
- reconstructs CPU tensors via `torch.frombuffer(shm.buf, dtype).reshape(...)`
- for MLA, splits the last dimension into `(nope, rope)` parts

`bind_connector_metadata(metadata)`:

- merges incoming `ReqMeta` into worker request state
- constructs load mappings for blocks in:

```text
range(num_gpu_computed_tokens / block_size,
      num_computed_tokens / block_size)
```

This range represents blocks that the CPU cache has but the GPU cache does not.

`start_load_kv()` and `wait_for_layer_load()`:

- load is layer-by-layer
- `start_load_kv()` loads layer 0
- `wait_for_layer_load()` synchronizes `load_stream`, increments layer index,
  then submits the next layer
- this keeps connector load aligned with model layer execution

Load implementation:

- default: for every `(cpu_block_id, gpu_block_id)`, run
  `gpu_layer_part[gpu_block_id].copy_(cpu_layer_part[cpu_block_id],
  non_blocking=True)`
- optional mapped-host gather:
  - build NPU int32 source/destination block id tensors
  - slice the CPU tensor from `src_min` to `src_max`
  - require CPU span and GPU output to be contiguous
  - require dtype match and dtype in fp32/fp16/bf16
  - call `torch.ops._C_ascend.kv_cache_block_gather`
  - fall back to copy on layout/dtype/capability mismatch

Save implementation:

- background `_save_listener()` consumes finished requests
- builds save mappings for blocks in:

```text
range(num_cpu_computed_tokens / block_size,
      min((num_computed_tokens + num_scheduled_tokens) / block_size,
          len(cpu_block_ids)))
```

- copies NPU KV blocks to CPU KV blocks over all layers and KV parts
- for MLA, distributes save mappings across TP ranks using:

```text
start = tp_rank
step = tp_world_size
```

- synchronizes `save_stream`
- puts request id into `save_output_queue`

`get_finished()`:

- worker collects save completion ids from the background queue
- deletes local request state for completed ids
- if TP world size is 1, returns done ids directly
- otherwise:
  - rank 0 receives completion ids from other TP ranks
  - waits until all TP ranks report the request
  - asynchronously calls `metadata.cache_and_free_slots(req_id)`
  - returns all-rank-complete request ids
  - nonzero ranks send done ids to rank 0 and return local done ids

### Metadata server and SharedMemory ownership

`MetadataServer` owns the physical shared CPU memory.

Transport:

- ZMQ ROUTER/DEALER over `ipc://$VLLM_RPC_BASE_PATH/metadata.ipc`
- messages are Python pickle payloads
- RPC functions are registered by string name

Memory sizing:

- config key: `cpu_swap_space_gb`
- default: 800 GB
- for non-MLA:
  - divide by `world_size`
  - divide by number of KV layers
  - layer shape:

```text
(2, num_blocks, block_size, num_kv_heads, head_size)
```

- for MLA:
  - force `tp_rank = 0`, so TP ranks share the same CPU memory namespace
  - divide by `pipeline_parallel_size`
  - divide by number of KV layers
  - layer shape:

```text
(num_blocks, block_size, num_kv_heads, head_size)
```

Shared memory:

- one Python `SharedMemory` object per layer
- name format includes `(pp_rank, tp_rank, layer_name)`
- existing same-name shared memory is unlinked before recreation
- worker RPC clients keep returned SharedMemory objects so they can close them
  in `__del__`
- server shutdown closes and unlinks all shared memory

Block manager:

- `post_init()` creates one `CPUKVCacheManager`
- the manager is initialized with the minimum `num_cpu_blocks` observed during
  shared-memory creation
- `post_init()` then registers RPC handlers:
  - `get_matched_num_and_touch`
  - `allocate_slots`
  - `record_request_cache_and_free_slots`
  - `cache_and_free_slots`

### CPUKVCacheManager behavior

The manager reuses upstream vLLM block-pool machinery:

- `BlockPool(num_cpu_blocks, enable_caching=True, block_size, events)`
- `get_manager_for_kv_cache_spec(...)`

Request state:

- request id -> block hashes
- request id -> computed prefix blocks touched during lookup
- request id -> allocation failed flag
- request id -> current token count
- request id -> request waiting for cache/free

Lookup:

- prompt logprobs disable prefix caching
- block hashes come from `request.block_hashes`
- `find_longest_cache_hit(...)` returns computed blocks
- touched blocks are recorded and `block_pool.touch()` is called
- hit stats are logged

Allocation:

- unallocated request ids are freed first
- compute number of CPU blocks needed
- if insufficient free blocks:
  - release previously touched computed blocks
  - mark request as failed to allocate
  - do not allocate partial blocks
- otherwise:
  - save new computed blocks into request state
  - allocate new blocks
  - return `block.block_id` for computed plus newly allocated blocks

Finish:

- `record_request_cache_and_free_slots(request)` delays caching/free until
  worker save completion
- `cache_and_free_slots(request_id)`:
  - caches request blocks unless allocation failed
  - frees request blocks
  - clears request metadata

### Legacy path constraints

Things worth preserving logically:

- longest-prefix hit behavior
- all-or-none allocation on insufficient CPU blocks
- delayed free until save completion
- TP all-rank completion before caching/free for MLA-related paths
- layer-by-layer load integration with model execution
- copy fallback when mapped gather cannot be used

Things to replace physically:

- scheduler/metadata-server ownership of physical CPU block ids
- Python SharedMemory as the production memory arena
- central metadata server as the authoritative owner of block allocation
- cross-worker reconstruction of CPU tensors from shared-memory handles
- process-global mapped-host registration cache
- implicit coupling between prefix cache and active swap state

Sharp edges:

- metadata RPC uses pickle over local IPC
- metadata server signal handlers are commented out
- `MetadataServerProc.run_metadata_server()` runs in a daemon thread, not a
  separate durable service process
- there is no explicit block generation or ABA protection
- no explicit in-flight pin count for CPU or NPU blocks
- save thread is infinite and has no shutdown protocol
- request cancellation and worker failure semantics are mostly implicit
- `save_block_mapping` is local to the save thread but accumulates per request
  then clears after synchronized save
- mapped-host registration has no unregister path in the prototype

## Path 3: AscendStore / External KV Pool

This path stores and retrieves KV cache through external backends. It is not a
CPU DRAM swap pool, but it offers useful protocol examples.

### Key namespace

`KeyMetadata` includes:

- model name
- head or TP rank
- prefill context parallel rank
- decode context parallel rank
- pipeline parallel rank

`PoolKey.to_string()` combines that metadata with a chunk hash. Layerwise mode
uses `LayerPoolKey`, adding a layer id.

This is a useful precedent: a safe cache key is not just token hash. It includes
layout and rank ownership metadata.

### Address calculation

`ChunkedTokenDatabase` stores:

- KV cache base addresses
- per-block byte lengths
- chunk block size
- optional pipeline partition information

For a token span and local GPU block ids, it computes backend get/put address
lists:

```text
addr = base_addr + block_id * block_len
size = block_len / block_size * (end - start)
```

This is similar to what a worker-local CPU swap transfer planner will need,
except the new subsystem should produce generation-safe CPU and NPU block refs
rather than raw addresses directly from scheduler metadata.

### Scheduler protocol

`KVPoolScheduler`:

- looks up external prefix hit length through a ZMQ lookup client
- tracks request token progress and allocated block ids
- creates `LoadSpec(vllm_cached_tokens, kvpool_cached_tokens, can_load)`
- builds `AscendConnectorMetadata`
- handles preempted and finished request ids
- may delay free of request blocks if async store still needs them

Important semantics:

- external hit length is advisory until blocks are allocated and metadata is
  built
- partial chunks can be discarded by config
- load only proceeds when `can_load` is set after allocation
- request finish may delay block free while store is still in flight

### Worker protocol

`KVPoolWorker`:

- registers NPU KV cache ranges with backend via `register_buffer(ptrs, lengths)`
- computes block lengths differently for MLA/sparse vs non-MLA
- starts sending/receiving threads
- supports non-layerwise and layerwise transfers
- supports async receive through background threads

Threading model:

- request metadata is pushed to worker threads
- `m_store.exists()` checks which keys already exist
- `m_store.put(keys, addrs, sizes)` stores missing blocks
- `m_store.get(keys, addrs, sizes)` retrieves blocks into local NPU KV cache
- completion is tracked by request id sets guarded by locks

Useful lessons for local DRAM design:

- completion state must be separated from request metadata construction
- block free may need to be delayed until transfer completion
- key namespace must encode model/rank/layout dimensions
- layerwise transfer can reduce latency but complicates completion semantics

## Native Copy Path: `swap_blocks_batch`

`swap_blocks_batch(src_ptrs, dst_ptrs, sizes, direction)` is registered on the
CPU backend because pointer arrays are CPU int64 tensors.

Requirements:

- `src_ptrs`, `dst_ptrs`, `sizes` must be CPU tensors
- all three must be int64
- all arrays must have equal length
- direction:
  - `0`: host to device
  - `1`: device to host
  - `2`: device to device

Execution:

- uses current NPU stream
- if compiled with `CANN_MEMCPY_BATCH_ASYNC` and direction is H2D/D2H, uses
  `aclrtMemcpyBatchAsync`
- otherwise loops over entries and submits `aclrtMemcpyAsync`

Constraints for production use:

- caller must ensure pointer validity and range correctness
- caller must ensure source/destination block lifetimes cover stream execution
- descriptor tensor lifetime must be verified for the batch async path
- native op has no knowledge of request ownership, block generation, pin count,
  or cancellation

This op should remain the Phase 1 baseline transfer backend, but it should be
wrapped by a transfer planner that owns refs, pins, and events.

## Mapped-Host Gather Prototype

The prototype path is:

```text
CPUOffloadingConnectorWorker.load_kv_layer()
  -> torch.ops._C_ascend.kv_cache_block_gather(...)
  -> get_mapped_host_device_ptr(...)
  -> aclrtHostRegister(... ACL_HOST_REGISTER_MAPPED ...)
  -> aclnnKvCacheBlockGather
```

Current behavior:

- page-aligns the CPU tensor span
- registers the host range on first use
- caches mappings in a process-global static vector
- reuses a mapping when the requested range is fully contained
- never unregisters mapped host memory
- uses an optional env var to load opapi library:
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER_OPAPI_LIB`

Python-side checks before using gather:

- host gather feature flag enabled
- custom op exists
- CPU span and GPU output are contiguous
- dtype matches
- dtype is fp32/fp16/bf16

Important production implications:

- mapped host registration must move out of static global state into a
  worker-owned registry
- mapping keys need device/context/generation identity
- unregister must be deterministic and stream-safe
- source CPU blocks and destination NPU blocks must be pinned during gather
- mapped gather must remain an optional backend after copy-path correctness
  exists

## Cross-Cutting Constraints For Worker-Owned DRAM

### Shape and layout constraints

The existing paths assume or encode:

- KV cache is organized by layer.
- Non-MLA usually has key/value parts.
- MLA can have different local layout and split parts.
- CPU block size can differ from GPU block size by an integer factor.
- Context parallelism and decode context parallelism may multiply effective
  chunk size in external KV pool paths.
- Some paths assume all KV layers share `page_size_bytes`.

The new subsystem should make `kv_layout_id` explicit and derive arena layout
from registered worker KV caches, not from scheduler guesses.

### Ownership constraints

Current systems split ownership in different ways:

- new `CpuNpuOffloadingHandler`: worker owns CPU tensors and transfer streams
- legacy `CPUOffloadingConnector`: metadata server owns CPU blocks and shared
  memory; worker owns copy execution
- AscendStore: backend owns external cache; worker owns NPU buffers and backend
  transfer threads

The target design should choose the first model for local DRAM:

```text
worker owns CPU memory, CPU block ids, block state, transfer streams,
host registration, and cleanup
```

### Scheduler contract constraints

The scheduler currently expects to ask:

- how many prefix tokens can be loaded
- which request blocks were allocated
- when load/save has finished
- whether request block free must be delayed

The worker-local design should keep scheduler intent logical:

- request id and request epoch
- active reload vs prefix cache load
- token/block ranges
- NPU block refs reserved by scheduler
- idempotency key

It should not expose raw CPU addresses or global CPU block ids as scheduler
authority.

### Transfer ordering constraints

Existing code has several ordering contracts that must remain explicit:

- D2H save must wait for model compute stream before reading NPU KV.
- H2D load must complete before the model layer consumes loaded KV.
- Legacy layerwise load synchronizes before moving to the next layer.
- Save completion controls when scheduler may free or cache request blocks.
- TP/MLA paths need all-rank completion before a prefix is considered saved.

The new subsystem should encode these through transfer events and block pins,
not through ad hoc queue timing.

### Memory safety constraints

Required for production:

- CPU block id plus generation, not raw id alone
- NPU block id plus generation/reservation, not raw id alone
- all-or-none admission before save submission
- no eviction while CPU block is pinned
- no NPU block reuse while transfer is in flight
- transfer descriptor buffers retained until stream completion if required by
  CANN API semantics
- deterministic shutdown:
  - stop new transfers
  - drain/cancel in-flight transfers
  - unregister mapped host ranges
  - release host memory

### Prefix cache constraints

Legacy code treats CPU memory primarily as prefix cache:

- lookup uses request block hashes
- touched blocks are LRU-like cache state
- `cache_and_free_slots()` turns completed request blocks into prefix cache

The worker-local DRAM design must split:

- active swap: correctness-critical request state
- prefix cache: disposable performance state

It can preserve prefix lookup semantics, but active swap must not depend on
prefix hash identity or disposable cache behavior.

## Engineering Takeaways

1. The best starting point is `vllm_ascend/kv_offload/cpu_npu.py`, not the
   legacy SharedMemory connector.

2. `CpuNpuOffloadingHandler` already owns the right low-level pieces:
   CPU tensors, streams, events, in-flight queues, and baseline copy backend.
   It lacks allocator semantics, block lifecycle, admission, and generation
   safety.

3. The legacy connector provides useful logical behavior to preserve:
   prefix-hit query, all-or-none allocation, delayed free until save completion,
   TP all-rank save completion, and layerwise load ordering.

4. The legacy connector's physical memory model should not be preserved:
   central SharedMemory ownership, scheduler-visible CPU block ids, and
   metadata-server-authoritative physical placement are the wrong abstraction
   for worker-owned DRAM.

5. `swap_blocks_batch` should be the first backend target for worker-local DRAM
   because it does not require host registration. A worker-local planner can
   wrap it with block refs, pins, and events.

6. Mapped-host gather should be added only after the worker-local copy path has
   parity. Its current static mapping cache and missing unregister path are
   prototype-only.

7. `ascend_store` is not the local DRAM design, but its key namespace and
   request metadata model are useful references for prefix availability,
   rank/layout scoping, and delayed free behavior.

8. The immediate design object should be a worker-local `HostSwapAllocator` plus
   `KVSwapTransferEngine`, sitting below the scheduler contract and above
   `swap_blocks_batch`.

## Suggested Next Investigation

- Inspect upstream vLLM `CPUOffloadingManager`, `OffloadingConnector`, and
  `OffloadingHandler` in the actual vLLM source tree.
- Trace how `TransferSpec` jobs are created, cancelled, and completed upstream.
- Identify where request preemption/cancellation reaches the offloading manager.
- Determine whether `aclrtMemcpyBatchAsync` copies pointer descriptor arrays
  before returning.
- Check whether torch pinned CPU tensors are always valid inputs for
  `aclrtHostRegister` on target CANN versions.
- Build a small design sketch for `HostSwapAllocator` state and the adapter
  needed to replace upstream `CPULoadStoreSpec` raw block ids with
  generation-safe worker-local refs.
