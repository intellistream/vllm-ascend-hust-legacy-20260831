# Device KV Gather: Host/Device Memory Notes

Generated on 2026-06-17.

## Scope

This note investigates the experimental `kv_cache_block_gather` path from the
branch `experiment/device-kv-gather`, with a focus on host memory, device memory,
long-running scheduler safety, and resource lifetime.

Related local files:

- `csrc/torch_binding.cpp`
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_offload_connector.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/metadata.py`
- `vllm_ascend/distributed/kv_transfer/kv_pool/cpu_offload/cpu_kv_cache_manager.py`
- `vllm_ascend/kv_offload/cpu_npu.py`

External API references used:

- CANN `aclrtHostRegister` documentation:
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/83RC1alpha002/API/appdevgapi/aclcppdevg_03_1804.html>
- CANN 9.1 runtime API index:
  <https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/910beta1/API/runtimeapi/aclcppdevg_03_0022.html>

## Current Memory Model

There are three distinct memory pools in this path.

1. NPU KV cache:
   - Owned by vLLM/vllm-ascend worker model execution.
   - Passed into `CPUOffloadingConnectorWorker.register_kv_caches`.
   - `load_kv_layer` writes selected blocks into these NPU tensors.

2. CPU KV cache:
   - Created by `MetadataServer.init_cpu_kv_caches`.
   - Backed by Python `multiprocessing.shared_memory.SharedMemory`.
   - Worker-side `ZMQRPCClient.call("init_cpu_kv_caches")` reconstructs CPU
     tensors with `torch.frombuffer(shm.buf, dtype=...).reshape(layer_size)`.
   - The server keeps the owning `SharedMemory` objects in
     `MetadataServer.shared_memory`; the worker client keeps handles and closes
     them in `ZMQRPCClient.__del__`.

3. Mapped host registrations:
   - Created by `get_mapped_host_device_ptr` in `csrc/torch_binding.cpp`.
   - The function page-aligns the CPU tensor pointer, calls
     `aclrtHostRegister(..., ACL_HOST_REGISTER_MAPPED, &mapped_base)`, and
     stores `(host_base, size, device_base)` in a static process-local vector.
   - The mapped device-visible pointer is then used as the data pointer of an
     ACL tensor passed to `aclnnKvCacheBlockGather`.

The fast path is therefore:

```text
SharedMemory host buffer
  -> torch.frombuffer CPU tensor
  -> aclrtHostRegister mapped host range
  -> ACL tensor with mapped device-visible pointer
  -> aclnnKvCacheBlockGather
  -> preallocated NPU KV cache tensor
```

## Important CANN Constraints

The CANN `aclrtHostRegister` documentation says the host pointer must be 4K page
aligned. The current C++ code handles this by aligning the start address down to
the OS page size and rounding the registered size up.

The same documentation says the mapped address is a device-accessible address
for registered host memory and that `aclrtHostRegister` must be paired with
`aclrtHostUnregister`. The branch currently registers but never unregisters.

The documentation also states that the mapped device address must not be used as
a memcpy address. The branch does not use it in `aclrtMemcpyAsync`; it passes the
mapped pointer into an ACLNN custom op. The prototype has already shown that the
mapped device address can be randomly read/written by device-side logic. Treat
the documentation as a conservative public contract, not as proof that random
access is impossible.

The documentation warns that on OS kernels 5.10 or lower, non-locked host memory
can be problematic and `aclrtMallocHost` is required. This branch uses Python
`SharedMemory`, not `aclrtMallocHost`, so kernel/CANN/platform compatibility is
not guaranteed by construction.

## Current Safety Properties

The path has some useful safety properties already:

- The feature is disabled by default through
  `VLLM_ASCEND_CPU_OFFLOAD_HOST_GATHER=0`.
- `CPUOffloadingConnectorWorker` falls back to the old tensor-copy path when the
  custom op is unavailable.
- Python checks that CPU and NPU KV tensors are contiguous and have matching
  supported dtypes before using host gather.
- `load_stream.synchronize()` is called before advancing each loaded layer, so
  the current implementation is conservative about cross-layer visibility.
- Save-side CPU writes are synchronized on `save_stream` before a request is
  reported as finished and before the metadata server is told to cache/free CPU
  slots.

## Production Blockers

### 1. Registered host mappings are never unregistered

`get_mapped_host_device_ptr` pushes mappings into a static vector after
`aclrtHostRegister`, but there is no call to `aclrtHostUnregister`.

Why this matters:

- CANN documents the API as a register/unregister pair.
- Long-running servers can accumulate mapped ranges for the whole process
  lifetime.
- If CPU KV memory is recreated, resized, or replaced, the stale mapping cache
  can hold device-visible mappings for memory that no longer belongs to the same
  tensor/storage.

The present CPU KV shared-memory allocation is mostly one-shot, so the leak may
look bounded in simple runs. It is not a production-safe lifecycle contract.

### 2. Mapping cache is keyed only by host range

The static `HostMapping` key does not include NPU device id, context, process
role, dtype, storage identity, or owner.

Risk:

- If one process ever touches multiple NPU devices or contexts, a mapping made
  while device A is current may be reused while device B is current.
- `OptionalNPUGuard(out.device())` sets a device before registration, but the
  cache lookup can return an old mapping without checking that it belongs to the
  same device/context.

Even if current deployment is process-per-device, production code should encode
that assumption or make the mapping table device-aware.

### 3. Partial-span registration can grow or overlap

`load_kv_layer` slices CPU KV by the current request span:

```python
cpu_layer_span = cpu_layer_part[src_min : src_max + 1]
```

The C++ op registers exactly that span, page-aligned. If future scheduler steps
touch different windows of the same CPU KV tensor, the mapping cache can collect
many ranges. A later larger span may not be contained in earlier smaller spans,
so it can trigger another registration over an overlapping area.

Production-friendly alternatives:

- Register each full CPU KV tensor once during `register_kv_caches`.
- Or register fixed-size page/block windows with an LRU and explicit
  `aclrtHostUnregister`.
- Or keep the current partial registration only behind a debug/experimental flag
  with telemetry proving bounded growth.

### 4. CPU SharedMemory is not guaranteed pinned/locked host memory

`MetadataServer` allocates CPU KV cache through Python `SharedMemory`. That is
good for inter-process sharing, but it is not the same ownership model as
`aclrtMallocHost`.

Risk:

- CANN documents stricter behavior for non-locked host memory on older kernels.
- The current code does not check kernel version, CANN version, product support,
  or whether the shared-memory pages meet the runtime's expectations.
- If registration pins pages internally, repeatedly registering large or
  overlapping spans can put pressure on system memory and IOMMU-like resources.

Production options:

- Add startup capability checks and refuse host gather when unsupported.
- Prefer an allocation strategy that is both shareable and known-safe for
  `aclrtHostRegister` on target kernels.
- If shared memory stays, add a small runtime self-test that registers,
  gathers, unregisters, and verifies data before enabling the fast path.

### 5. ACL tensor/executor cleanup is not RAII-safe

`kv_cache_block_gather` creates four `aclTensor*` values. They are released in
the custom handler after `op_api` returns, but several earlier failure paths can
throw before those releases happen:

- `aclnnKvCacheBlockGatherGetWorkspaceSize` failure.
- NPU workspace tensor allocation failure.
- Any exception between `ConvertType`/`aclCreateTensor` and handler execution.

Production code should wrap ACL tensor handles in a small RAII helper or
`unique_ptr`-like guard so every path releases them.

### 6. Workspace lifetime needs explicit validation

When `workspace_size != 0`, the code allocates a temporary NPU byte tensor and
passes its raw data pointer to the ACLNN op. The tensor is local to
`kv_cache_block_gather`; after `cmd.Run()` returns, its storage can be released
to the NPU caching allocator.

This may be safe if the torch-npu allocator records stream use correctly for the
current custom handler. It should not be assumed without validation. A
production version should either rely on a known torch-npu pattern for custom op
workspace lifetime or keep an owning tensor associated with the stream until the
op is complete.

### 7. Index validation happens mostly by construction

The op receives NPU int32 `src_block_ids` and `dst_block_ids`. C++ cannot inspect
their values without a device-to-host sync. Python constructs them from
`load_block_mapping`, so normal operation should be valid, but the fast path
does not locally validate:

- `src_min >= 0`
- `src_max < cpu_layer_part.shape[0]`
- every destination block is within `gpu_layer_part.shape[0]`
- number of block ids is acceptable for int32 and op limits

These checks can be done in Python before creating the NPU tensors because the
mapping list is already on CPU.

### 8. Per-step NPU allocations can fragment or add scheduler overhead

Every `load_kv_layer` creates new NPU tensors for source and destination block
ids. Every custom op may allocate a workspace tensor. Under long scheduling,
this can create allocator churn even if the host mapping table is fixed.

Possible hardening:

- Reuse index buffers sized to the largest observed or configured batch.
- Keep per-layer/per-device workspace buffers when the op API permits.
- Record allocator metrics during long-running CPU-prefix-hit workloads.

### 9. Connector lifecycle itself is old/deprecated

Local release notes say `CPUOffloadingConnector` is deprecated and expected to
be replaced by vLLM CPUOffload. A production decision should happen before
heavy hardening:

- If this branch is only a prototype, harden enough to benchmark and prove the
  concept.
- If this becomes product code, move the gather mechanism into the newer
  offload path (`vllm_ascend/kv_offload/cpu_npu.py`) or whichever upstream vLLM
  CPUOffload interface is current.

## Host Memory Lifecycle Recommendation

For a production path, the owner of host memory should also own host
registration.

Recommended design:

1. During CPU KV cache creation, build explicit registration objects for each
   full CPU KV tensor or fixed window.
2. Store mapping entries with:
   - host base address
   - registered size
   - mapped device pointer
   - owning NPU device id/context
   - Python storage/shared-memory owner identity
   - reference count or explicit lifecycle state
3. On connector shutdown, synchronize streams that may read mapped host memory.
4. Call `aclrtHostUnregister` for every registered host base.
5. Only after unregistering, close worker-side shared-memory handles and unlink
   server-side shared-memory segments.

This lifecycle avoids a raw process-global mapping cache that outlives its real
memory owner.

## Device Memory Lifecycle Recommendation

The destination NPU KV cache should remain owned by the existing vLLM KV cache
allocator. The gather op should be treated as a writer into those already-owned
blocks, not as a device allocator.

Production expectations:

- No persistent device memory should be allocated by the gather path after
  initialization except bounded/reused index and workspace buffers.
- The op should run on the load stream and publish completion through the same
  synchronization contract used by `wait_for_layer_load`.
- Workspace buffers should either be stack-like per call with verified
  stream-safe allocator behavior, or reused per device/stream with explicit
  lifetime.
- The gather path must not hide any `item()`-style device sync in hot loops.

## Suggested Production Plan

Phase 1: Make the prototype measurable.

- Add debug counters for number of host mappings, total registered bytes,
  registration failures, fallback count, and gather calls.
- Add Python-side bounds validation before creating NPU index tensors.
- Add a feature-gated stress test that repeatedly loads varied CPU block spans
  and reports mapping growth.

Phase 2: Fix lifecycle.

- Add C++ RAII for ACL tensor handles.
- Add `aclrtHostUnregister` support and expose a cleanup op if Python owns the
  registration lifecycle.
- Include device id/context in the mapping key.
- Decide whether to register full CPU KV tensors at initialization or implement
  bounded window registration.

Phase 3: Validate platform support.

- At startup, run a small host-register/gather/unregister self-test.
- Record CANN version, product type, OS kernel version, and whether host gather
  was enabled or disabled.
- Disable fast path automatically on unsupported products or kernel/runtime
  combinations.

Phase 4: Move to the right connector path.

- Decide whether to continue with deprecated `CPUOffloadingConnector` or port
  the gather op into the newer CPU offload handler.
- If porting, reuse the newer handler's event queues and CPU tensor allocation
  shape, then compare against `swap_blocks_batch`.

## Validation Matrix

Minimum tests before considering this production-ready:

- Correctness:
  - gather one block, many blocks, duplicate destination blocks if allowed, and
    non-contiguous block id order.
  - compare against the fallback copy path for fp32/fp16/bf16.
  - test MLA and non-MLA KV cache shapes.

- Lifecycle:
  - repeated request churn for hours with varied prefix lengths.
  - assert registered mapping count and total bytes are bounded.
  - force connector shutdown and verify all mappings are unregistered before
    shared memory is closed/unlinked.

- Compatibility:
  - CANN versions used by the project, especially 8.5.x and newer.
  - target products such as Ascend 910B/C and any A2/A3 variants in scope.
  - OS kernels at or below/above 5.10 if those deployments matter.

- Performance:
  - CPU-prefix-hit latency vs old per-block copy path.
  - H2D bandwidth under random and contiguous block mappings.
  - NPU allocator fragmentation and workspace/index allocation overhead.
  - scheduler latency impact from Python index construction.

## Current Bottom Line

The branch is a useful performance experiment, but it is not yet production
safe. The main missing contract is resource ownership:

- CPU KV memory is owned by `SharedMemory`.
- Mapped host registrations are owned by a static C++ vector.
- NPU writes are launched through an ACLNN custom op.

Those three lifetimes need to be unified or explicitly ordered. Until that is
done, the safest default remains disabled, with the fast path enabled only for
controlled benchmarking.
