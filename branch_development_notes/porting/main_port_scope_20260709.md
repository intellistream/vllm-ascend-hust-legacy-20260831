# Main Port Scope: Device KV Gather + Staging Pool

This branch ports the device KV gather prototype onto `origin/main` in small,
reviewable pieces. It is not a merge of the old experiment branch.

## Ported

- `kv_cache_block_gather` Ascend custom op sources and build hooks.
- Torch binding for:
  - `torch.ops._C_ascend.kv_cache_block_gather`
  - host mapping registration/query/stat helpers
  - host mapping clear helper for explicit test cleanup
- Env-gated CPU offload load paths:
  - direct mapped-host gather
  - worker-local staging pool with persistent CPU staging slabs
  - C++ CPU pack backend
- Experimental `VLLM_ASCEND_KV_GATHER_MAX_AIV_CORES` tiling knob from the
  prototype. It is disabled unless explicitly set and should only be used for
  resource-contention experiments.
- Minimal JSONL observability through `ASCEND_HOST_GATHER_STATS_PATH`.
- Direct smoke script for custom op correctness.

## Intentionally Not Ported

- Old `always` / `auto` mapped-gather policy and block-threshold logic.
- Runtime preregistration of original CPU KV cache slabs.
- Allocator/refcount diagnosis logs from the old branch.
- Metadata shutdown RPC / runner-specific shared-memory cleanup changes.
- Old benchmark cases whose baseline was page-by-page copy instead of main's
  coalesced span-copy path.

## Baseline Rule

`origin/main` already coalesces adjacent CPU/GPU block pairs into span-copy
loads. That span-copy path must remain the baseline for performance A/B.
Fallback from direct mapped gather or staging pool must return to that path,
not to page-by-page copy.

## Validation Order

1. Build custom op and verify `aclnnKvCacheBlockGather` symbols.
2. Run `tools/smoke_device_kv_gather.py`.
3. Run Qwen2.5-14B 1024/4096 serving smoke on this main-based branch.
4. Re-run bounded allocator and teardown/shm checks on main before migrating any
   old cleanup fixes.
5. Compare main span-copy, direct mapped gather, and staging pool on long-context
   serving workloads.
