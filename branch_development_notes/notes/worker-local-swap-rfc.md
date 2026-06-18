# RFC: Worker-Local Host Swap Memory Management for KV Offload

Generated on 2026-06-17.

## Status

Draft for design review.

## Summary

This RFC proposes replacing the legacy global SharedMemory-backed CPU KV pool
with a device-affined, worker-local host swap memory subsystem.

The first production milestone is not mapped-host gather. The first milestone is
a correct worker-owned host swap pool that preserves existing CPU offload
behavior through the baseline copy path. Mapped-host device gather/read-write
ops are introduced later as an optional transfer backend after memory ownership,
block lifecycle, compatibility, and copy-path behavior are stable.

The core principle is:

```text
Scheduler owns logical residency and routing intent.
Worker owns physical residency and transfer execution.
Router consumes advisory locality signals only.
```

Device-affined means each worker/device/rank owns its local host swap arena, CPU
block allocator, block metadata, transfer streams, and memory lifecycle. The
scheduler may request logical offload/load, but the worker resolves that intent
to concrete local CPU blocks.

The scheduler may decide that a request should load or save KV blocks, and it
may use prefix-aware routing to improve locality. However, the physical CPU swap
region, host registration, mapped device pointers, transfer streams, and cleanup
should be owned by the worker process that uses the memory.

This avoids making cross-worker prefix-cache reuse depend on a distributed
shared-memory pool. Prefix cache remains a performance optimization: if local
cache state is lost, the system should recompute.

Any mapped-gather failure must degrade to a correctness-preserving fallback:
copy path, worker-local miss, recompute, request retry, or explicit failure. It
must never produce silent partial reuse or silent data corruption.

## Motivation

The current experimental mapped-host gather branch builds on the legacy
`CPUOffloadingConnector`. That connector allocates CPU KV cache in a centralized
metadata server using Python `multiprocessing.shared_memory.SharedMemory`.
Workers reconstruct CPU tensors from shared-memory handles received over
ZeroMQ/pickle, and scheduler-side code asks the metadata server for CPU prefix
hits and CPU block ids.

That design has a plausible historical motivation: it tries to provide a
node-local shared CPU prefix-cache pool across DP replicas and, for MLA, across
TP ranks. But mapped-host gather changes the memory-lifetime requirements:

- host memory registration should be owned by the process that owns the host
  memory
- mapped device pointers should be associated with a concrete NPU device/context
- stream synchronization should be local to the worker transfer engine
- unregister and cleanup must happen deterministically

Python SharedMemory plus a central metadata server splits those responsibilities
across processes. That is a poor fit for a production mapped-host memory path.

The implementation should therefore separate two workstreams:

1. Host swap memory architecture migration:
   - worker-owned host arenas
   - worker-local CPU block allocator
   - active swap vs prefix cache lifecycle
   - baseline copy-path correctness
   - logical compatibility with existing CPU offload behavior

2. Device-side mapped access optimization:
   - host registration lifecycle
   - mapped pointer registry
   - mapped-host gather/read-write ops
   - runtime fallback to copy path

The first workstream must be correct without the second.

## Goals

- Introduce a worker-local CPU KV swap subsystem.
- Keep physical host memory ownership inside the worker process.
- Preserve existing CPU offload behavior through a baseline copy path first.
- Provide a compatibility layer for legacy unified-pool logical behavior.
- Support mapped-host gather/read-write later as optimized transfer backends.
- Provide deterministic host registration and unregister lifecycle.
- Preserve a clean scheduler/worker contract for KV offload/load.
- Support prefix-aware routing without requiring globally shared host memory.
- Keep failure handling simple: lost worker-local cache degrades to recompute.
- Build on the newer `vllm_ascend/kv_offload/cpu_npu.py` path where practical.
- Separate active request swap state from disposable prefix-cache state.
- Define in-flight transfer pinning so eviction/unregister cannot race with
  device work.
- Make router prefix metadata advisory and worker-local lookup authoritative.

## Non-Goals

- This RFC does not propose a globally shared CPU KV memory pool.
- This RFC does not require cross-worker KV block transfer.
- This RFC does not make CPU prefix cache persistent state.
- This RFC does not attempt to recover worker-local CPU cache after worker
  failure.
- This RFC does not depend on Python `multiprocessing.shared_memory`.
- This RFC does not make the deprecated `CPUOffloadingConnector` the production
  target.
- This RFC does not make prefix-aware routing a correctness dependency.
- This RFC does not require mapped-host gather to be implemented before the
  worker-local host swap pool is behaviorally compatible with the existing copy
  path.

## Background

Relevant current code:

- Legacy connector:
  - `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`
  - `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/metadata.py`
  - `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py`
- Newer CPU/NPU offload path:
  - `vllm_ascend/kv_offload/npu.py`
  - `vllm_ascend/kv_offload/cpu_npu.py`
- Experimental mapped-host gather:
  - `csrc/torch_binding.cpp`

The newer offload path already has a healthier split:

- scheduler-side `CPUOffloadingManager` tracks logical offload state
- worker-side `CpuNpuOffloadingHandler` allocates CPU tensors locally
- the worker handler owns transfer streams, events, and in-flight transfer queues

This RFC proposes extending that worker-local path rather than hardening the
legacy SharedMemory connector as the primary production route.

## Core Invariants

The implementation must preserve these invariants:

1. The scheduler never owns host memory, host registration, or mapped device
   pointers.
2. The worker is the sole authority for physical CPU block ids and mapped-host
   lifecycle.
3. Router prefix availability is advisory; worker-local lookup is authoritative.
4. Prefix-cache blocks are disposable, but active swap blocks are
   correctness-critical.
5. A CPU block cannot be evicted, overwritten, or unregistered while referenced
   by an in-flight transfer.
6. A GPU/NPU KV block referenced by an in-flight save/load transfer must remain
   reserved until the transfer reaches `completed`, `failed`, or `cancelled`
   state and the corresponding stream event has been observed.
7. A prefix cannot be advertised until all required local KV parts and layers
   are in `READY` state.
8. Worker restart invalidates all previously advertised prefix availability via
   `worker_generation`.
9. Mapped gather must always have a correctness-preserving fallback path.
10. Any local CPU cache failure must degrade to worker miss, recompute, request
   retry, or explicit failure, never silent partial reuse.
11. The scheduler may reclaim NPU blocks after offload only when the worker has
    acknowledged durable local CPU save completion.
12. Save/offload must reserve all required CPU blocks before submitting any
    NPU-to-CPU transfer. If reservation fails, no partial save is submitted.

## State Model

Worker-local CPU swap stores two different classes of data:

```python
class CPUBlockKind(Enum):
    ACTIVE_SWAP = "active_swap"
    PREFIX_CACHE = "prefix_cache"
```

`ACTIVE_SWAP` blocks belong to a currently executing request. They are
correctness-critical. They must not be evicted or overwritten until the owning
request no longer needs them, or until the request is explicitly aborted and
will be recomputed or retried from a higher-level boundary.

`PREFIX_CACHE` blocks are reusable performance cache. They may be evicted when
not pinned by in-flight transfer or active use.

Every local CPU block has an explicit state:

```python
class CPUBlockState(Enum):
    FREE = "free"
    RESERVED = "reserved"
    SAVING = "saving"
    READY = "ready"
    LOADING = "loading"
    EVICTING = "evicting"
    ERROR = "error"
```

State meanings:

- `FREE`: available for allocation.
- `RESERVED`: allocated for a request or prefix, but no valid KV data yet.
- `SAVING`: NPU-to-CPU transfer is in flight.
- `READY`: CPU block contains complete valid local KV data.
- `LOADING`: CPU-to-NPU transfer is in flight.
- `EVICTING`: no new readers allowed; block is waiting for pins to drain.
- `ERROR`: transfer or validation failed; block must not be advertised or reused
  until reset to `FREE`.

Minimum transition sketch:

```text
FREE -> RESERVED -> SAVING -> READY
READY -> LOADING -> READY
READY -> EVICTING -> FREE
RESERVED/SAVING/LOADING -> ERROR -> FREE
```

The allocator must track block kind, owner, state, generation, and pin count.

### CPU Block Granularity

A local CPU block id denotes one logical KV block index across all local layers
and KV parts in the worker-owned arena.

A block is `READY` only if every required local layer and KV part for that block
contains valid data. Implementations may internally store per-layer/per-part
physical offsets, but the public allocator state is tracked at the logical block
level.

### Admission Control

Admission must be explicit and all-or-none for save/offload:

```python
@dataclass
class AdmissionResult:
    admitted: bool
    reserved_blocks: list[LocalCPUBlockRef]
    rejected_reason: Literal[
        "insufficient_cpu_swap_capacity",
        "active_swap_quota_exceeded",
        "prefix_cache_quota_exceeded",
        "eviction_in_progress",
        "namespace_mismatch",
    ] | None
```

Rules:

- `ACTIVE_SWAP` allocation has priority over `PREFIX_CACHE` admission.
- `PREFIX_CACHE` blocks may be evicted to satisfy `ACTIVE_SWAP` allocation.
- If `ACTIVE_SWAP` allocation still fails after eligible prefix-cache eviction,
  the worker returns `insufficient_cpu_swap_capacity`.
- The scheduler must not reclaim source NPU blocks when active-save admission
  fails.
- Save/offload reserves all required CPU blocks before submitting any transfer.
  If reservation fails, no partial save is submitted.

### Promotion And Eviction

Phase 1-3 do not implicitly promote `ACTIVE_SWAP` blocks into `PREFIX_CACHE`.
Prefix cache population is handled by an explicit save/materialization path.

Future promotion is allowed only when:

- the request reaches a safe completion or prefix materialization point
- all blocks are `READY`
- namespace and token/position hash are computed and validated
- no active request owner remains
- prefix-cache admission accepts the blocks

`READY -> EVICTING` must atomically:

1. prevent new pins
2. remove the block from authoritative local prefix lookup
3. enqueue prefix availability invalidation
4. wait for existing pins to drain
5. release the block

## In-Flight Transfer Model

Transfers must pin all CPU blocks and host mappings they may access.

```python
@dataclass
class InFlightTransfer:
    transfer_id: str
    request_id: str
    request_epoch: int
    source_cpu_blocks: list[LocalCPUBlockRef]
    dest_cpu_blocks: list[LocalCPUBlockRef]
    source_gpu_blocks: list[GPUBlockRef]
    dest_gpu_blocks: list[GPUBlockRef]
    host_mapping_refs: list[MappingRef]
    stream_event: TransferEvent
    state: Literal["submitted", "completed", "failed", "cancelled"]
```

Required behavior:

- A CPU block with a transfer pin cannot be evicted or overwritten.
- A GPU/NPU block with a transfer pin cannot be reused, evicted, or overwritten.
- A mapped host range with a transfer reference cannot be unregistered.
- Worker shutdown first stops new transfers, then drains or cancels in-flight
  transfers, then unregisters host ranges, then releases host memory.
- C++ op wrappers must keep source tensors, destination tensors, index buffers,
  workspace buffers, and ACL handles alive until device work is complete.
- Async device errors must mark the transfer and affected blocks as failed; they
  must not be treated as cache misses without explicit state transition.

## Save/Offload Correctness

The save/offload path is as important as the load/gather path.

If the scheduler asks a worker to offload NPU KV blocks to CPU, it must not
reclaim or overwrite the source NPU blocks until the worker reports durable local
CPU save completion.

In this RFC, durable means:

- every required local layer is saved
- every required KV part is saved
- transfer stream work is complete and visible
- affected CPU blocks transitioned from `SAVING` to `READY`
- failed or cancelled saves are reported explicitly

A prefix hash must not be advertised until every required local CPU block for
that prefix is in `READY` state for all local layers and KV parts.

Request cancellation policy must be explicit:

- cancellation before save submission releases `RESERVED` CPU blocks
- cancellation during active-swap save either drains to `READY` for the owning
  request, or cancels and moves partial blocks to `ERROR`
- cancelled active-swap blocks must not be published as prefix cache unless an
  explicit prefix materialization or promotion policy accepts them
- cancellation during load unpins source blocks after stream completion or
  cancellation acknowledgement
- partial save/load results must not be advertised

## Proposed Architecture

### Components

```text
Router / Scheduler
  |
  | logical load/save/offload decisions
  v
WorkerLocalSwapManager
  |
  +-- HostSwapAllocator
  +-- HostMappingRegistry
  +-- LocalCPUBlockAllocator
  +-- KVSwapTransferEngine
  +-- PrefixCacheIndexLocal
```

### Delivery Principle

The memory subsystem must be useful before mapped-host gather exists.

The first implementation target is:

```text
worker-local host swap pool + baseline copy path = existing behavior parity
```

Mapped-host gather/read-write is a later transfer backend. It must not be a
precondition for validating CPU arena layout, block allocation, request
lifecycle, eviction, cancellation, save/load ordering, or worker shutdown.

This sequencing keeps failures attributable:

- copy path correct, mapped path disabled: memory pool and lifecycle can be
  validated independently
- host registration enabled, mapped path disabled: registration lifecycle can be
  validated independently
- mapped path enabled: remaining failures are isolated to mapped access,
  indexing, stream, or op behavior

### Scheduler Responsibilities

The scheduler may:

- decide whether a request should use CPU KV cache
- decide how many logical tokens/blocks are eligible
- emit logical load/save/offload intent
- receive transfer completion status
- publish prefix availability to a router or prefix index
- use prefix-aware routing signals

The scheduler should not:

- allocate concrete host memory
- own CPU block physical addresses
- own mapped host registrations
- own shared-memory lifecycle
- decide CANN host registration/unregister timing

### Worker Responsibilities

The worker should:

- allocate its CPU swap quota
- manage local CPU block ids
- map logical transfer requests to local CPU blocks
- perform CPU-to-NPU and NPU-to-CPU transfers
- register host memory for mapped-device access
- own mapped device pointers
- synchronize transfer streams
- unregister host memory during shutdown
- report cache availability, capacity pressure, and transfer status

## Migration Compatibility Layer

Before enabling mapped-host gather, the implementation must provide a
compatibility layer that maps legacy unified CPU pool behavior onto the
worker-local host swap pool.

The compatibility layer preserves logical behavior:

- request offload/load semantics
- prefix lookup semantics
- block allocation/free semantics
- active swap block lifecycle
- prefix cache block lifecycle
- eviction semantics
- cancellation behavior
- scheduler-facing completion behavior

The compatibility layer must not preserve legacy physical ownership:

- no production global SharedMemory-backed host pool
- no scheduler-owned physical CPU blocks
- no cross-worker host tensor reconstruction
- no central process owning host registration lifecycle

Legacy concepts map to the new system as follows:

| Legacy concept | Worker-local mapping |
| --- | --- |
| global CPU block id | worker-local CPU block id plus worker/device owner |
| metadata server lookup | worker-local authoritative lookup |
| shared memory handle | local host tensor or host arena reference |
| global prefix hit | advisory local or routed prefix hit |
| cross-worker block reuse | not guaranteed initially; later addressed by routing |
| block transfer | baseline copy path first |

Compatibility scope:

- Must preserve single-worker/single-device offload/load correctness.
- Must preserve request lifecycle and scheduler completion contracts.
- Must preserve local prefix hit/miss and recompute fallback for disposable
  prefix cache.
- May temporarily drop direct cross-worker physical CPU KV reuse.
- May drop stable global CPU block ids.
- May replace metadata-server-authoritative prefix hit with
  worker-authoritative local lookup.

The compatibility layer is a migration bridge. It must not grow back into a
global physical memory owner.

Compatibility acceptance boundary:

| Behavior | Required in Phase 1-3? | Notes |
| --- | --- | --- |
| Single-worker CPU offload/save/load correctness | Yes | KV contents must match baseline copy behavior. |
| Scheduler completion ordering | Yes | Acknowledgement semantics must be equivalent or stricter. |
| Request cancellation | Yes | Cancellation must not cause silent corruption. |
| Local prefix hit/miss | Yes | Local behavior must match logical prefix-cache semantics. |
| Direct cross-worker CPU memory reuse | No | Record as performance delta only. |
| Stable global CPU block ids | No | New system uses worker-local generation-safe refs. |
| Metadata-server-authoritative prefix hit | No | Worker-local lookup is authoritative. |
| Prefix hit rate identical to legacy | No | Prefix-aware routing may recover locality later. |

## KVHostSwapPool Interface

Phase 0 should define the stable worker-local interface before implementing
mapped access optimizations.

Block identity must not assume every CPU block is a prefix hash. Active swap and
prefix cache have different keys:

```python
class KVLoadIntentType(Enum):
    ACTIVE_SWAP_RELOAD = "active_swap_reload"
    PREFIX_CACHE_LOAD = "prefix_cache_load"


@dataclass(frozen=True)
class ActiveSwapBlockKey:
    request_id: str
    request_epoch: int
    block_index: int
    token_range: tuple[int, int]
    position_range: tuple[int, int]


@dataclass(frozen=True)
class PrefixCacheBlockKey:
    cache_namespace: str
    prefix_hash: BlockHash
    block_index: int
    token_range: tuple[int, int]
    position_range: tuple[int, int]


KVBlockKey = ActiveSwapBlockKey | PrefixCacheBlockKey
```

Physical block references must be generation-safe:

```python
@dataclass(frozen=True)
class LocalCPUBlockRef:
    arena_id: int
    block_id: int
    block_generation: int
    kind: CPUBlockKind
    state: CPUBlockState


@dataclass(frozen=True)
class GPUBlockRef:
    block_id: int
    block_generation: int
    device_id: int
```

Before submitting any transfer, the worker must validate:

```text
current(cpu_block.block_id).generation == ref.block_generation
current(cpu_block.block_id).state is compatible with transfer direction
pin increment succeeded for every CPU and GPU block ref
```

Conceptual interface:

```python
class KVHostSwapPool:
    def allocate(
        self,
        request_id: str,
        request_epoch: int,
        block_keys: list[KVBlockKey],
        kind: CPUBlockKind,
    ) -> LocalAllocation:
        ...

    def save_from_device(
        self,
        request_id: str,
        request_epoch: int,
        gpu_blocks: list[GPUBlockRef],
        block_keys: list[KVBlockKey],
        idempotency_key: str,
    ) -> KVSaveResult:
        ...

    def load_to_device(
        self,
        request_id: str,
        request_epoch: int,
        intent_type: KVLoadIntentType,
        block_keys: list[KVBlockKey],
        gpu_blocks: list[GPUBlockRef],
        idempotency_key: str,
    ) -> KVLoadResult:
        ...

    def free_request(self, request_id: str, request_epoch: int | None = None) -> None:
        ...

    def lookup_prefix(
        self,
        cache_namespace: str,
        prefix_hashes: list[BlockHash],
    ) -> LocalPrefixLookupResult:
        ...

    def evict(self, policy: EvictionPolicy) -> EvictionResult:
        ...
```

The interface must cover:

- block allocate/free
- request ownership
- active swap block lifecycle
- prefix cache block lifecycle
- prefix hash lookup
- eviction
- load/save acknowledgement
- cancellation
- worker shutdown

## Memory Ownership Model

Each worker owns one or more CPU swap arenas.

An arena is a set of host buffers sized for the worker's KV cache layout:

```text
layer -> kv part -> [num_cpu_blocks, block_size, num_kv_heads, head_size]
```

For non-MLA models, KV parts are typically key and value.

For MLA models, the arena should follow the actual local KV layout consumed by
that worker/rank. If MLA requires special TP-aware layout, it should be encoded
in the worker-local arena layout, not implemented by collapsing all TP ranks
onto a global SharedMemory object.

### Allocation Backends

The worker-local allocator should support pluggable host backends:

1. Pinned torch CPU tensors:
   - easiest integration with current `CpuNpuOffloadingHandler`
   - compatible with existing `torch.zeros(..., pin_memory=True)` path

2. CANN host allocation:
   - stronger ownership contract for mapped-host access
   - potentially better match for `aclrtHostRegister` or related APIs

3. Custom host-register-safe allocator:
   - future option if torch pinned memory and CANN host allocation do not meet
     all layout or sharing requirements

The first production target should be the simplest backend that passes
correctness, long-running lifecycle, and copy-path performance tests on target
hardware.

Phase 1-3 copy-path backend requirements:

- stable tensor/storage lifetime
- pinned or otherwise acceptable copy performance on target hardware
- compatibility with `swap_blocks_batch` or the selected baseline copy path
- bounded allocation/free overhead under churn
- deterministic release on normal shutdown
- acceptable behavior under process crash

Phase 5+ mapped-host backend requirements:

- compatibility with `aclrtHostRegister` or equivalent mapped-host registration
- stable mapped pointer behavior for the registered range lifetime
- bounded registered bytes and mapping count
- deterministic unregister on normal shutdown
- no silent pageable-memory fallback when pinned or registered memory is required
- clear NUMA and device-affinity behavior where relevant

Host registration compatibility must not block Phase 1-3 copy-path parity.

## Host Registration Model

The worker owns a `HostMappingRegistry`.

The registry maps local host ranges to mapped device-visible pointers:

```text
(device_id, device_context_id, worker_generation, host_base, size, storage_id)
    -> mapped_device_ptr
```

Requirements:

- include NPU device/context identity in the key
- register full arenas or fixed-size windows, not arbitrary per-request spans by
  default
- avoid overlapping duplicate registrations
- expose debug counters:
  - mapping count
  - registered bytes
  - register failures
  - unregister failures
  - fallback count
- synchronize relevant streams before unregister
- call `aclrtHostUnregister` deterministically at shutdown

The current prototype's process-global static vector is not sufficient for
production. It should become an explicit worker-owned lifecycle object.

### Registration Mode

The production default should be arena registration first:

- register full worker-local CPU swap arenas during initialization
- avoid per-request registration churn
- keep mapping count stable
- fall back to copy path if registration fails

Required guardrails:

- `max_registered_bytes_per_worker`
- `max_mapping_count_per_worker`
- `registration_failure_fallback=copy`

Window registration is useful for very large CPU swap quotas or runtime
registration limits, but it is more complex. It requires window cache eviction,
overlap handling, per-window references, and transfer-aware unregister.

Therefore:

```text
The host-mapping phase first implements arena mode.
Window mode remains experimental until stress tests prove bounded mapping count
and no registration-churn regression.
```

## Transfer Engine

The worker owns a `KVSwapTransferEngine` with two load implementations:

1. Baseline copy path:
   - current `swap_blocks_batch` or equivalent batched copy implementation
   - always available fallback

2. Mapped-host gather path:
   - uses mapped host pointers
   - gathers selected CPU blocks directly into NPU KV cache
   - enabled only after runtime capability checks

The transfer engine should decide at runtime:

```text
if mapped_gather_enabled and tensors/layout/dtype/capability are valid:
    use mapped-host gather
else:
    use baseline copy
```

Fallback should be visible through metrics.

## Scheduler/Worker Contract

The scheduler should pass logical transfer intent, not concrete host memory
ownership.

Router metadata is advisory. The selected worker performs the authoritative
local lookup and may return a partial hit or miss.

Example load intent:

```python
@dataclass
class KVLoadIntent:
    request_id: str
    request_epoch: int
    intent_type: KVLoadIntentType
    worker_generation: int
    model_id: str
    cache_namespace: str
    requested_blocks: list[KVBlockKey]
    requested_token_count: int
    gpu_blocks: list[GPUBlockRef]
    allow_partial_hit: bool
    idempotency_key: str
```

Rules:

- `ACTIVE_SWAP_RELOAD` must not allow partial hit unless the scheduler
  explicitly splits the request at a safe recompute boundary.
- `PREFIX_CACHE_LOAD` may allow partial hit and recompute the missing suffix.
- Active reload miss must return explicit failure/retry/recompute status, not a
  silent prefix miss.

Worker-local planning result:

```python
@dataclass
class LocalKVLoadPlan:
    cpu_blocks: list[LocalCPUBlockRef]
    gpu_blocks: list[GPUBlockRef]
    layer_range: range
    transfer_backend: Literal["copy", "mapped_gather"]
```

Worker response:

```python
@dataclass
class KVLoadResult:
    request_id: str
    request_epoch: int
    intent_type: KVLoadIntentType
    worker_id: str
    worker_generation: int
    loaded_token_count: int
    loaded_block_count: int
    missed_blocks: list[KVBlockKey]
    gpu_blocks_loaded: list[GPUBlockRef]
    backend: Literal["copy", "mapped_gather"]
    fallback_reason: str | None
    status: Literal["ok", "partial_hit", "miss", "failed"]
    error_code: str | None
```

The scheduler should not need to know `cpu_block_ids` unless those ids are part
of a worker-local reporting/debug API.

Idempotency:

- `idempotency_key` is scoped by `request_id`, `request_epoch`, and
  `intent_type`.
- The worker may return the previous result for an already completed key.
- Duplicate in-flight requests should attach to the existing transfer or be
  rejected with a retryable status.
- Cancellation closes or increments the request epoch.
- Completed idempotency records have bounded TTL and are removed on request
  finalization.

## Prefix-Aware Routing

Worker-local CPU swap reduces cross-worker reuse unless routing is prefix-aware.

The proposed routing model:

```text
worker reports prefix availability and pressure
router scores workers
request is routed to the best worker
worker uses local CPU prefix cache when available
```

Candidate score inputs:

- longest prefix hit length
- queue depth
- NPU memory pressure
- CPU swap pressure
- active request count
- expected recompute cost
- recent failure or eviction signals

This keeps global metadata advisory. The router can prefer locality without
owning DRAM.

### Prefix Availability Metadata

Prefix availability reports should include enough information to invalidate
stale entries and avoid unsafe reuse:

```python
@dataclass
class PrefixAvailability:
    worker_id: str
    worker_generation: int
    model_id: str
    kv_layout_id: str
    cache_namespace: str
    prefix_hash: BlockHash
    num_tokens: int
    num_blocks: int
    ready_block_count: int
    state: Literal["ready", "evicting", "stale"]
    cpu_pressure: float
    updated_at_ms: int
    ttl_ms: int
```

Rules:

- `(worker_id, worker_generation)` is the validity domain for local prefix
  metadata.
- Worker restart increments `worker_generation` and invalidates all old
  advertisements.
- Prefix advertisements must expire through TTL or heartbeat loss.
- Router hit selection is not authoritative; the worker must confirm
  availability locally before loading.
- Stale or missing local entries degrade to worker miss and recompute.
- Prefix availability reporting must be bounded. Workers may report top-K hot
  prefixes, compressed ranges, sampled entries, or router-requested probes.
  Raw unbounded prefix-hash streams are not allowed.

### Cache Namespace

Prefix hash reuse must be scoped by a cache namespace.

The namespace should include at least:

- model id
- model revision
- tokenizer revision
- KV cache dtype
- KV layout id
- block size
- rope or position-encoding configuration
- attention backend flags that affect KV interpretation
- LoRA or adapter id when applicable
- tenant or security namespace when multi-tenant isolation is required

Conceptually:

```text
BlockHash = hash(cache_namespace, token_ids[start:end], position_range)
```

Prefix cache reuse must never cross incompatible namespaces.

### TP And MLA Hit Semantics

For TP serving, a request is served by a group of ranks. A routable prefix hit
must be defined at group level, not by a single rank.

```text
A prefix is routable as a TP-group hit only if all required ranks in that group
report READY availability for their local shard/layout of that prefix.
```

Suggested group view:

```python
@dataclass
class PrefixAvailabilityGroupView:
    dp_replica_id: str
    tp_group_id: str
    worker_generations: dict[int, int]
    prefix_hash: BlockHash
    min_ready_tokens_across_ranks: int
    all_ranks_ready: bool
    group_pressure_score: float
```

For MLA, the worker-local arena layout must produce a stable `kv_layout_id`.
That layout id participates in `cache_namespace`, and the transfer engine must
validate shapes, strides, dtypes, and local shard semantics before using mapped
gather.

## Failure Model

Worker-local CPU prefix cache is disposable.

Worker-local active swap is not disposable. If active swap state is lost, the
owning request must be aborted, retried, or recomputed from a higher-level
boundary. It must not be treated as an ordinary prefix-cache miss.

If a worker fails:

- remove it from routing
- discard its prefix availability entries
- cancel or fail requests whose active swap state was owned by that worker
- release its process-owned memory through normal process cleanup
- recompute prefixes on other workers as needed

No distributed cache recovery is required for correctness.

This is intentionally simpler than recovering a global shared pool whose
metadata, reference counts, memory mappings, and in-flight transfers may be
partially inconsistent.

## Runtime Configuration

Suggested configuration surface:

```text
VLLM_ASCEND_CPU_SWAP_BACKEND=torch_pinned|cann_host|auto
VLLM_ASCEND_CPU_SWAP_LOCAL_GB=<float>
VLLM_ASCEND_CPU_SWAP_NUM_BLOCKS=<int>
VLLM_ASCEND_CPU_SWAP_ENABLE_MAPPED_GATHER=0|1
VLLM_ASCEND_CPU_SWAP_REGISTER_MODE=arena|window
VLLM_ASCEND_CPU_SWAP_WINDOW_MB=<int>
```

Suggested defaults:

```text
VLLM_ASCEND_CPU_SWAP_BACKEND=auto
VLLM_ASCEND_CPU_SWAP_LOCAL_GB=0
VLLM_ASCEND_CPU_SWAP_NUM_BLOCKS unset
VLLM_ASCEND_CPU_SWAP_ENABLE_MAPPED_GATHER=0
VLLM_ASCEND_CPU_SWAP_REGISTER_MODE=arena
VLLM_ASCEND_CPU_SWAP_WINDOW_MB unset
```

Conflict handling:

- CPU swap is disabled if both `LOCAL_GB` and `NUM_BLOCKS` are unset or zero.
- If both `LOCAL_GB` and `NUM_BLOCKS` are set, they must resolve to the same
  block count after layout calculation; otherwise configuration is invalid.
- `WINDOW_MB` is valid only when `REGISTER_MODE=window`.
- mapped gather requires CPU swap to be enabled.
- mapped gather registration failure falls back to copy unless configured to
  fail fast for testing.

Environment variables should follow the repository's env-var review rules:

- `VLLM_ASCEND_*` naming
- documented default
- valid values/range
- sensitive or non-sensitive classification

If this integrates through vLLM config rather than env vars, the same
information should be documented in the config schema.

## Observability

Expose metrics/logs for:

- CPU swap total blocks
- CPU swap free blocks
- CPU swap used bytes
- CPU swap admission count
- CPU swap admission reject count
- CPU swap active blocks
- CPU swap prefix-cache blocks
- CPU swap eviction reason count
- in-flight load count
- in-flight save count
- local prefix hit length
- local prefix hit/miss count
- prefix advertisement lag
- prefix advertisement stale hit count
- recompute count due to missing CPU cache
- transfer backend selected: copy vs mapped gather
- transfer latency and bytes
- transfer queue wait
- transfer execution latency
- mapped registration count
- total registered bytes
- host registration latency
- host unregister latency
- register/unregister failures
- fallback reasons
- worker-local cache evictions
- mapped gather validation failure count
- mapped gather async error count

These metrics are necessary to decide whether worker-local routing is enough or
whether any global pool design is justified.

Metrics must avoid high-cardinality labels such as raw `request_id` or
`prefix_hash`.

Fallback reasons should use a bounded enum:

```text
disabled
capability_missing
dtype_unsupported
layout_unsupported
non_contiguous
invalid_block_id
registration_failed
mapping_missing
stream_error
index_buffer_error
inflight_conflict
unknown
```

## Implementation Plan

### Phase 0: Define interface and correctness guardrails

- Define `KVHostSwapPool` interface.
- Define active-swap and prefix-cache block keys.
- Define generation-safe `LocalCPUBlockRef` and `GPUBlockRef`.
- Define `KVLoadIntentType` for active reload vs prefix-cache load.
- Define CPU block kind and state machine.
- Separate `ACTIVE_SWAP` and `PREFIX_CACHE`.
- Add transfer pin/refcount semantics.
- Add NPU/GPU KV block reservation semantics for in-flight save/load.
- Define admission control and reserve-all-or-none behavior.
- Add `worker_generation`.
- Add worker-local authoritative lookup result.
- Define save/load acknowledgement semantics.
- Define cancellation behavior.
- Define worker shutdown ordering.
- Define legacy compatibility acceptance boundaries.
- Keep mapped-host gather disabled by default.
- Add Python-side bounds validation for block ids.

### Phase 1: Device-affined worker-local host swap pool

- Introduce `HostSwapAllocator`.
- Allocate per-worker CPU KV arenas.
- Support torch pinned host tensors first.
- Compute arena layout from registered NPU KV cache shapes.
- Introduce `LocalCPUBlockAllocator`.
- Track local CPU block ownership by active-swap or prefix-cache block key.
- Implement admission control.
- Implement local eviction policy.
- Enforce `READY`/`SAVING`/`LOADING`/`EVICTING` states.
- Enforce CPU block generation validation before transfer submission.
- Enforce NPU/GPU block reservation for transfer destinations and sources.
- Keep baseline `swap_blocks_batch` transfer path working.
- Keep mapped gather disabled.
- Prove worker-local copy-path semantics first.

### Phase 2: Legacy unified-pool compatibility adapter

- Introduce a compatibility layer such as `LegacyCPUOffloadCompatLayer`.
- Map legacy logical API calls to `KVHostSwapPool`.
- Preserve request offload/load semantics.
- Preserve prefix lookup semantics where locally available.
- Preserve scheduler-facing completion behavior.
- Do not reintroduce global physical host memory ownership.
- Do not reintroduce scheduler-owned physical CPU block ids.

### Phase 3: Copy-path behavior parity

- Run the legacy path and worker-local pool path on the same cases.
- Compare KV cache contents or block-level checksums.
- Compare block allocation/free behavior.
- Compare prefix hit/miss behavior for local hits.
- Compare eviction behavior.
- Compare request cancellation behavior.
- Compare scheduler completion ordering.
- Verify copy-path latency has no unacceptable regression.

### Phase 4: Local prefix index and eviction hardening

- Support partial hit.
- Keep metadata purely local in the worker.
- Report prefix availability to scheduler/router as advisory data.
- Bound prefix availability reporting volume.
- Add namespace and layout validation.
- Ensure active swap blocks cannot be evicted as prefix cache.
- Implement `READY -> EVICTING` lookup removal and invalidation ordering.
- Add stale generation rejection.

### Phase 5: Host mapping registry lifecycle

- Introduce `HostMappingRegistry`.
- Implement arena registration first.
- Include device/context identity in mapping keys.
- Add deterministic unregister on worker shutdown.
- Add stress tests for mapping count and registered bytes.
- Keep window registration experimental.
- Continue using copy path for data transfer.

### Phase 6: Mapped-host device gather/read-write backend

- Move/extend `kv_cache_block_gather` into the worker-local offload path.
- Add device-side host read/write ops behind feature flags.
- Add RAII cleanup for ACL tensor handles in the C++ op.
- Add explicit unregister support in the C++ helper layer.
- Reuse index buffers where practical.
- Add fallback to `swap_blocks_batch`.
- Compare correctness against copy path.
- Measure latency and allocator churn.

### Phase 7: Prefix-aware routing

- Expose local prefix availability from workers.
- Add `worker_generation`, TTL, namespace, and layout id to reports.
- Add routing score that includes prefix hit length and worker pressure.
- Ensure worker can reject stale or missing local hits.
- Make disposable cache loss degrade to recompute.
- Compare worker-local routing against the legacy shared pool.

### Phase 8: Deprecation and cleanup

- Treat legacy `CPUOffloadingConnector` as a reference/prototype.
- Do not build production mapped-host gather around Python SharedMemory.
- Remove or quarantine experimental env vars once the new config surface exists.

## Validation Plan

### Behavior Parity

Before enabling mapped-host gather, the worker-local host swap pool must pass
copy-path behavior parity against existing CPU offload behavior.

Required checks:

- offload/load KV contents match
- block-level checksum matches when full tensor comparison is too expensive
- block allocation and free behavior is equivalent or stricter
- admission reject behavior is explicit and all-or-none
- local prefix hit/miss behavior is equivalent
- eviction behavior is equivalent or safer
- request cancellation behavior is equivalent or safer
- scheduler-facing save/load completion ordering is preserved
- worker failure does not cause silent corruption
- copy-path latency has no unacceptable regression

Parity compatibility is logical, not physical:

- direct cross-worker reuse of the same CPU memory is not required
- stable global CPU block ids are not required
- metadata-server-authoritative prefix hits are not required

### Correctness

- Copy path and mapped-gather path produce identical KV cache contents.
- Test fp32, fp16, and bf16.
- Test single block, many blocks, non-contiguous block order, and repeated
  prefix requests.
- Test partial prefix hit.
- Test stale router metadata followed by worker miss.
- Test simultaneous load and eviction attempts for overlapping prefixes.
- Test cancellation during save.
- Test cancellation during load.
- Test repeated prefix reused by concurrent requests.
- Test CPU block id reuse after in-flight transfer completion.
- Test stale CPU block generation rejection before transfer submission.
- Test GPU/NPU block reuse attempts while save/load is in flight.
- Test reserve-all-or-none admission failure for large active swaps.
- Test active reload exact-hit failure path.
- Test prefix-cache partial-hit path.
- Test invalid or stale `worker_generation` rejection.
- Test prefix hash namespace mismatch rejection.
- Test MLA and non-MLA layouts.
- Test TP and DP configurations where available.
- Test quantized KV or fp8 if supported by the platform.

### Lifecycle

- Run long request churn tests.
- Verify mapped registration count is bounded.
- Verify total registered bytes is bounded.
- Verify unregister runs during normal shutdown.
- Verify worker failure does not require shared cache recovery.
- Inject `aclrtHostRegister` failure.
- Inject `aclrtHostUnregister` failure.
- Inject mapped gather C++ op failure.
- Inject async device error after op submission.
- Send worker `SIGTERM` during in-flight transfer.
- Send worker `SIGKILL` during registered memory lifetime.
- Trigger Python GC while C++ op has device work in flight.
- Keep stale router prefix metadata after worker restart and verify rejection.

### Performance

- Compare:
  - baseline no CPU offload
  - worker-local copy path
  - worker-local mapped gather
  - legacy SharedMemory connector if available
- Measure:
  - TTFT p50/p90/p95/p99
  - ITL p50/p90/p95/p99
  - throughput
  - CPU prefix hit latency
  - transfer bandwidth
  - transfer queue wait p95
  - transfer execution latency p95
  - fallback rate
  - prefix hit rate
  - stale advertised hit rate
  - scheduler overhead
  - NPU allocator fragmentation
  - host memory pressure
  - pinned host memory pressure

Use realistic prefix distributions such as Zipfian reuse, not only synthetic
uniform traffic.

### Routing

- Compare worker-local swap with and without prefix-aware routing.
- Measure hit rate and tail latency under realistic load-balancing scenarios.
- Simulate worker failure and cache loss.

## Open Questions

- Which host allocation backend is safest on target CANN/kernel versions:
  torch pinned memory, CANN host allocation, or custom allocator?
- Should registration be full-arena by default, or fixed windows to reduce
  registered bytes?
- Which bounded prefix availability reporting strategy should the router use:
  top-K hot prefixes, compressed ranges, sampled entries, or requested probes?
- How should MLA TP layouts be represented in worker-local arenas?
- Can mapped-host gather share index/workspace buffers across transfers without
  unsafe stream lifetime assumptions?
- What is the minimum measurable benefit required to justify any future global
  shared CPU pool?

## Alternatives Considered

### Keep global SharedMemory pool

Pros:

- maximal local cross-worker CPU prefix reuse
- one central block manager
- existing prototype path

Cons:

- distributed shared state
- difficult cleanup and failure recovery
- awkward host registration ownership
- deprecated connector path
- not necessary for local device KV swap

### Worker-local swap with no routing

Pros:

- simplest worker memory model
- best reliability
- easiest mapped-host lifecycle

Cons:

- may lose DP cross-worker prefix reuse
- load balancer may send repeated prefixes to different workers

### Worker-local swap with prefix-aware routing

Pros:

- preserves local ownership
- captures much of cross-worker reuse through routing
- failure degrades to recompute
- aligns with mapped-host gather lifecycle

Cons:

- requires routing integration
- needs prefix availability reporting
- may still miss some reuse that a global pool could capture

This RFC recommends worker-local swap with prefix-aware routing.

## Decision Criteria

Adopt the global shared-memory pool only if benchmarks show a substantial,
production-relevant improvement over worker-local swap with prefix-aware routing.

Otherwise, choose worker-local swap as the default because it has a much simpler
correctness and reliability model.

## Conclusion

Mapped-host KV gather should be productionized around worker-local CPU swap
ownership.

The global SharedMemory pool solves cross-worker reuse by introducing
distributed shared state. That is the wrong default for a performance cache.
Worker-local swap, prefix-aware routing, and recomputation on cache loss provide
a cleaner and more reliable baseline.
