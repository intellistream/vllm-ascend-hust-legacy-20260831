# RFC: Reuse Plan for Worker-Local DRAM KV Swap

Generated on 2026-06-17.

## Status

Draft for design review.

## Summary

vLLM Ascend currently contains several KV transfer/cache subsystems that serve
different purposes:

1. local CPU/NPU KV offload
2. legacy SharedMemory CPU prefix-cache pool
3. AscendStore external KV pool
4. Mooncake P2P disaggregated KV transfer
5. UCM connector
6. LMCache connector shim
7. MultiConnector orchestration

They look confusing because they all implement some part of the vLLM
`KVConnector` or offloading surface, but they solve different problems.

For the worker-local DRAM memory system, we should reuse code selectively:

- reuse the newer `kv_offload` copy-path transfer engine as the starting point
- reuse vLLM KVConnector lifecycle methods as the compatibility boundary
- reuse AscendStore/Mooncake metadata patterns for request tracking, delayed
  free, key namespace, events, and metrics
- reuse native `swap_blocks_batch` for Phase 1 copy-path parity
- do not reuse legacy SharedMemory physical ownership
- do not reuse process-global mapped-host registration state

The target design is a worker-owned local DRAM subsystem:

```text
Scheduler owns logical intent.
Worker owns DRAM arenas, CPU block ids, block states, transfer execution,
host registration, and cleanup.
```

## KV Subsystem Taxonomy

### 1. `kv_offload`: local CPU/NPU offload

Code:

- `vllm_ascend/kv_offload/npu.py`
- `vllm_ascend/kv_offload/cpu_npu.py`
- `tests/e2e/singlecard/test_cpu_offloading.py`

Purpose:

- Implements Ascend support for upstream vLLM's generic offloading connector.
- Moves KV blocks between NPU KV cache and worker-local CPU tensors.
- Uses `swap_blocks_batch` as the copy backend.

What it owns today:

- worker-local CPU tensors
- H2D and D2H NPU streams
- NPU events
- in-flight transfer queues
- pointer-array construction for batched copy

What it does not own yet:

- block state machine
- active swap vs prefix cache distinction
- admission control
- eviction
- block generations
- NPU block pinning
- cancellation policy
- mapped-host registration

Verdict:

This is the best implementation seed for worker-local DRAM.

### 2. Legacy `CPUOffloadingConnector`: SharedMemory CPU prefix pool

Code:

- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py`

Purpose:

- Provides CPU prefix-cache reuse through a central metadata server.
- Allocates Python `multiprocessing.shared_memory.SharedMemory`.
- Workers reconstruct CPU tensors from shared-memory handles.
- Scheduler queries metadata server for prefix hit length and CPU block ids.

Useful logical behavior:

- longest prefix hit lookup through vLLM block hashes
- all-or-none CPU block allocation
- delayed free until save completion
- request finish turns blocks into prefix cache
- TP all-rank completion before cache/free
- layer-by-layer load hook shape

Dangerous physical behavior:

- central metadata server owns physical CPU memory
- scheduler sees physical CPU block ids
- workers depend on cross-process SharedMemory reconstruction
- request and prefix-cache semantics are mixed
- no block generation or in-flight pinning
- save thread has no explicit shutdown protocol

Verdict:

Use as a behavior reference, not as a production implementation base.

### 3. AscendStore: external KV pool

Code:

- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/ascend_store_connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_scheduler.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/pool_worker.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/config_data.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/ascend_store/kv_transfer.py`

Purpose:

- Stores/retrieves KV blocks through external backends:
  - Mooncake
  - Memcache
  - Yuanrong
- Uses string keys derived from model/rank/chunk hash metadata.
- Registers NPU KV buffer ranges with the backend.

Useful code/patterns:

- `KeyMetadata` and `PoolKey` namespace construction
- `RequestTracker` and `LoadSpec`
- chunked token processing
- delayed free through `request_finished()`
- layerwise and non-layerwise worker transfer modes
- KV cache event aggregation via `AscendStoreKVEvents`
- lookup server/client pattern for prefix hit length

Not directly reusable:

- backend get/put semantics
- remote key-value store ownership
- direct use of NPU KV addresses as external storage payloads

Verdict:

Borrow metadata and lifecycle patterns. Do not borrow storage backend shape for
local DRAM.

### 4. Mooncake P2P connectors

Code:

- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake_layerwise_connector.py`
- `vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py`
- `vllm_ascend/distributed/kv_transfer/utils/utils.py`

Purpose:

- P/D disaggregated KV transfer through Mooncake.
- Transfers KV between producer and consumer workers.
- Supports non-layerwise and layerwise transfer forms.
- Uses handshake metadata and remote transfer parameters.

Useful code/patterns:

- worker/scheduler split under `KVConnectorBase_V1`
- `request_finished()` delayed-free semantics
- `request_finished_all_groups()` for grouped block ownership
- layerwise load/save hook integration
- transfer threads and finished request sets
- memory registration wrapper in `GlobalTE.register_buffer`
- TP/CP/DCP mapping helpers in `utils.py`

Not directly reusable:

- remote host/port/block mapping
- Mooncake transfer engine dependency
- P/D handshake protocol

Verdict:

Borrow interface and lifecycle patterns. Avoid coupling local DRAM to Mooncake.

### 5. UCM connector

Code:

- `vllm_ascend/distributed/kv_transfer/kv_pool/ucm_connector.py`

Purpose:

- Thin wrapper around `ucm.integration.vllm.ucm_connector.UCMConnector`.
- Exposes vLLM `KVConnectorBase_V1` methods and metrics hooks.

Useful patterns:

- complete connector method forwarding
- `get_block_ids_with_load_errors()`
- stats and Prometheus metrics factory methods

Not directly reusable:

- UCM engine internals are external.

Verdict:

Borrow connector interface completeness and metrics shape.

### 6. LMCache connector shim

Code:

- `vllm_ascend/distributed/kv_transfer/kv_pool/lmcache_ascend_connector.py`

Purpose:

- Imports `lmcache_ascend`.
- Re-exports upstream `LMCacheConnectorV1`.

Verdict:

No local DRAM implementation value, except as evidence that connector shims
should stay thin when ownership belongs elsewhere.

### 7. AscendMultiConnector

Code:

- `vllm_ascend/distributed/kv_transfer/ascend_multi_connector.py`

Purpose:

- Overrides upstream `MultiConnector`.
- Routes `update_state_after_alloc()` only to the chosen connector, except
  `MooncakeLayerwiseConnector`, which receives full block information.

Useful pattern:

- multiple KV systems can coexist, but block ownership must be explicit.
- non-selected connectors should get empty block sets to avoid accidental
  physical ownership.

Verdict:

Useful for migration and compatibility if worker-local DRAM needs to coexist
with external KV connectors.

## Existing Interfaces We Should Reuse

### vLLM `KVConnectorBase_V1` lifecycle

Methods visible across connectors:

Scheduler side:

- `get_num_new_matched_tokens(request, num_computed_tokens)`
- `update_state_after_alloc(request, blocks, num_external_tokens)`
- `build_connector_meta(scheduler_output)`
- `request_finished(request, block_ids)`

Worker side:

- `register_kv_caches(kv_caches)`
- `bind_connector_metadata(connector_metadata)`
- `clear_connector_metadata()`
- `start_load_kv(forward_context, **kwargs)`
- `wait_for_layer_load(layer_name)`
- `save_kv_layer(layer_name, kv_layer, attn_metadata, **kwargs)`
- `wait_for_save()`
- `get_finished(finished_req_ids)`

These hooks already match where worker-local DRAM must integrate with the model
runner. Reusing the lifecycle avoids inventing a parallel integration surface.

### Upstream offloading spec shape

`NPUOffloadingSpec` currently plugs into upstream `OffloadingConnector` through:

- `get_manager()`
- `get_handlers(kv_caches, attn_backends)`
- `CPULoadStoreSpec`
- `GPULoadStoreSpec`
- `OffloadingHandler.transfer_async()`
- `OffloadingHandler.get_finished()`
- `OffloadingHandler.wait()`

This shape is attractive for Phase 1 copy-path parity, but the raw block-id
specs are not enough for production worker-local DRAM. We need an internal
adapter that maps upstream specs to generation-safe local refs.

### Native transfer backend

`swap_blocks_batch` is reusable as-is for Phase 1:

- no host registration dependency
- supports H2D, D2H, and D2D directions
- uses the current NPU stream
- accepts flat CPU pointer arrays

But the caller must add:

- descriptor lifetime protection
- CPU/NPU block pinning
- range validation
- generation validation
- transfer completion ownership

### Key namespace and token chunking

Borrow from AscendStore:

- `KeyMetadata`
- `PoolKey`
- rank/layout dimensions in the key
- token chunking through block hashes
- layerwise key splitting

Do not copy the exact string format blindly. The worker-local DRAM key should
include:

- model id/revision
- tokenizer revision
- KV layout id
- dtype
- block size
- position encoding config
- adapter/LoRA id
- tenant/security namespace when relevant
- TP/PP/CP rank identity as needed

## What We Should Not Reuse

Do not reuse these as production foundations:

- Python `SharedMemory` as the physical DRAM arena
- central `MetadataServer` as the CPU block owner
- scheduler-authoritative physical CPU block ids
- raw `list[int]` CPU block ids across async transfer boundaries
- process-global mapped host registration vector
- save threads without explicit shutdown/drain protocol
- prefix hash as the identity of active swap state
- implicit request cancellation behavior

These are exactly the areas where the worker-local design is supposed to be
more robust.

## Proposed Architecture

Introduce a new worker-local package:

```text
vllm_ascend/kv_offload/local_dram/
  __init__.py
  block.py
  allocator.py
  arena.py
  transfer.py
  prefix_index.py
  connector.py
  metrics.py
```

### `block.py`

Defines stable state and refs:

```python
class CPUBlockKind(Enum):
    ACTIVE_SWAP = "active_swap"
    PREFIX_CACHE = "prefix_cache"


class CPUBlockState(Enum):
    FREE = "free"
    RESERVED = "reserved"
    SAVING = "saving"
    READY = "ready"
    LOADING = "loading"
    EVICTING = "evicting"
    ERROR = "error"


@dataclass(frozen=True)
class LocalCPUBlockRef:
    arena_id: int
    block_id: int
    block_generation: int
    kind: CPUBlockKind


@dataclass(frozen=True)
class NPUBlockRef:
    block_id: int
    block_generation: int
    device_id: int
```

### `arena.py`

Owns worker-local CPU tensors:

- allocates per-layer/per-part CPU arenas
- starts with torch pinned CPU tensors
- later supports CANN host allocation
- derives layout from registered NPU KV caches
- exposes stable base pointers and block sizes to the transfer planner

Initial implementation should reuse the allocation logic in
`CpuNpuOffloadingHandler`:

- one CPU tensor per layer and KV part
- same dtype as NPU KV cache
- dimension 0 sized by `num_cpu_blocks * block_size_factor`
- pinned memory when available

### `allocator.py`

Owns physical CPU block state:

- admission control
- all-or-none reservation
- active swap priority over prefix cache
- prefix-cache eviction
- pin/unpin
- generation increments on reuse
- request ownership
- cancellation cleanup
- shutdown cleanup

This is the main missing piece in current code.

### `transfer.py`

Wraps transfer execution:

- initially wraps `swap_blocks_batch`
- later can select mapped-host gather/read-write
- owns transfer queues and events
- validates CPU/NPU block refs before submission
- pins refs until stream completion
- retains pointer descriptor tensors if required

Initial implementation should reuse from `CpuNpuOffloadingHandler`:

- stream creation
- event pool
- in-flight transfer queues
- pointer-array vectorization
- `expand_block_ids`
- transfer stats

But the transfer API should accept local refs, not raw block ids.

### `prefix_index.py`

Owns worker-local prefix metadata:

- maps prefix cache keys to `LocalCPUBlockRef`
- supports local authoritative lookup
- supports partial hit
- removes entries before `READY -> EVICTING`
- emits advisory availability reports

Borrow from legacy `CPUKVCacheManager`:

- longest prefix hit semantics
- block hash handling
- cache stats shape

Borrow from AscendStore:

- key namespace discipline
- rank/layout-aware cache keys

### `connector.py`

Provides the vLLM integration surface.

There are two possible integration layers:

1. OffloadingSpec-compatible path for Phase 1 copy-path parity.
2. KVConnector-compatible path for prefix-cache routing and compatibility with
   existing connector lifecycle hooks.

Recommended staging:

- Phase 1: keep using upstream `OffloadingConnector` with a new handler adapter
  under `NPUOffloadingSpec`.
- Phase 2: add a local DRAM connector facade only where prefix-cache semantics
  need vLLM `KVConnectorBase_V1` hooks.
- Phase 3: support coexistence with external connectors through
  `AscendMultiConnector`.

### `metrics.py`

Expose bounded metrics:

- CPU total/free/active/prefix blocks
- admission accepted/rejected
- eviction reason
- in-flight H2D/D2H count
- transfer latency
- transfer queue wait
- copy vs mapped backend choice
- prefix local hit/miss
- stale advisory hit
- generation validation failure
- NPU block pin conflict
- cancellation cleanup count

Borrow UCM's pattern for connector stats/prom metrics if integrating as a
`KVConnectorBase_V1`.

## Reuse Matrix

`kv_offload/cpu_npu.py`:

- Reuse: CPU tensor allocation, streams, events, pointer arrays, copy backend.
- Do not reuse: raw block ids as long-term ownership.

`kv_offload/npu.py`:

- Reuse: `OffloadingSpec` integration.
- Do not reuse: current minimal `num_cpu_blocks` config as the final config
  surface.

Legacy CPU offload:

- Reuse: prefix-hit semantics, all-or-none allocation, delayed free.
- Do not reuse: SharedMemory and metadata server physical ownership.

`CPUKVCacheManager`:

- Reuse: vLLM `BlockPool` compatibility ideas and cache stats.
- Do not reuse: global CPU block id authority.

AscendStore:

- Reuse: key namespace, request tracker, load spec, events.
- Do not reuse: external backend storage model.

Mooncake:

- Reuse: delayed free, grouped finish, layerwise hooks, memory registration
  wrapper pattern.
- Do not reuse: remote transfer protocol.

UCM:

- Reuse: full connector forwarding, metrics hooks, load error reporting.
- Do not reuse: external UCM engine.

LMCache shim:

- Reuse: thin integration style.
- Do not reuse: no local memory logic exists here.

`swap_blocks_batch`:

- Reuse: Phase 1 transfer backend.
- Do not reuse: unmanaged pointer lifetime.

Mapped gather prototype:

- Reuse: capability checks, op shape, fallback idea.
- Do not reuse: static mappings and missing unregister.

## Implementation Plan

### Phase 0: Interface extraction

- Define `LocalCPUBlockRef`, `NPUBlockRef`, block keys, and state machine.
- Define `HostDRAMArena` and `HostDRAMAllocator` interfaces.
- Define transfer plan API using refs, not raw ids.
- Define adapter from upstream `CPULoadStoreSpec`/`GPULoadStoreSpec` to local
  refs.
- Add unit tests for state transitions and generation validation.

### Phase 1: Copy-path local DRAM allocator

- Fork or refactor `CpuNpuOffloadingHandler` into a reusable transfer engine.
- Keep torch pinned CPU tensors as the first arena backend.
- Add allocator state on top of current CPU tensors.
- Implement all-or-none admission.
- Implement active swap priority.
- Use `swap_blocks_batch` only.
- Keep mapped gather disabled.
- Match current `test_cpu_offloading.py` behavior.

### Phase 2: Lifecycle hardening

- Add request epoch and cancellation handling.
- Add CPU and NPU block pinning.
- Add delayed free until transfer completion.
- Add explicit shutdown drain.
- Retain transfer descriptor tensors until completion if needed.
- Add stress tests for long-running churn.

### Phase 3: Prefix cache local index

- Add local prefix index.
- Implement local longest-hit lookup.
- Keep worker lookup authoritative.
- Add prefix namespace and layout id.
- Add advisory prefix availability reports.
- Do not add cross-worker physical sharing.

### Phase 4: KVConnector compatibility facade

- Implement vLLM `KVConnectorBase_V1` facade only for prefix-cache workflows
  that require connector hooks.
- Preserve scheduler lifecycle:
  - matched-token query
  - update after allocation
  - connector metadata build
  - request finished delayed free
  - get finished
- Support coexistence through `AscendMultiConnector`.

### Phase 5: Host registration and mapped backend

- Add worker-owned `HostMappingRegistry`.
- Register arenas first, not per request.
- Add deterministic unregister.
- Add mapped-host gather/read-write as optional backend.
- Fallback to copy on any capability or validation failure.

## Compatibility Strategy

The new system should preserve logical behavior, not physical behavior.

Must preserve:

- CPU offload save/load correctness
- scheduler completion ordering
- delayed free when transfer is in flight
- prefix hit/miss behavior for local prefix cache
- copy-path fallback
- no silent corruption on cancellation or failure

Not required:

- global SharedMemory pool
- stable global CPU block ids
- direct cross-worker CPU memory reuse
- metadata-server-authoritative physical placement
- identical legacy prefix hit rate before prefix-aware routing

## Open Questions

- Should Phase 1 live entirely under `NPUOffloadingSpec`, or should we create a
  new connector name immediately?
- Can upstream `CPULoadStoreSpec` be extended with opaque local refs, or should
  we keep an internal mapping layer?
- Does `aclrtMemcpyBatchAsync` copy descriptor arrays before returning?
- What is the exact request cancellation signal path in upstream vLLM
  offloading?
- How should TP-group local prefix availability be aggregated for routing?
- Should prefix-cache promotion from active swap be deferred until after local
  prefix index is stable?

## Verdict

Start from the newer local CPU/NPU offload path and turn it into a real memory
subsystem.

Use existing connector systems as reference libraries:

- `kv_offload`: implementation seed
- legacy CPU offload: behavior compatibility reference
- AscendStore/Mooncake: protocol and lifecycle reference
- UCM: metrics/interface completeness reference
- native ops: transfer backend

Do not start from the legacy SharedMemory pool. It solves a different problem
with the wrong physical ownership boundary for worker-local DRAM.
