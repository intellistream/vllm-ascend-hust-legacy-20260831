# Split-Batch (Dual-Stream) Execution

## Overview

Split-batch (internally: `DUAL_INPLACE`) is an **experimental, opt-in** decode
optimization for Ascend NPU. When a decode batch meets the admission conditions,
the engine splits it into two halves and executes them back-to-back instead of
running the padded batch as one unit:

- `inplace_serial` — replays two exact-size graphs sequentially, avoiding the
  padding overhead of one large graph.
- `inplace_parallel` — additionally runs the two halves on two NPU streams
  (a main stream and a parallel stream) so their compute overlaps.
- `dual_pad` — pads both halves to capture sizes and dispatches dual-padded
  graphs, trading some padding for graph reuse.

The feature is fully gated: without `additional_config["split_batch_config"]`
nothing changes relative to the default single-stream path.

## Getting Started

Enable it through `additional_config`:

```python
from vllm import LLM

llm = LLM(
    model="Qwen/Qwen3-0.6B",
    additional_config={
        "split_batch_config": {
            "enabled": True,
            "mode": "inplace_parallel",
            "num_splits": 2,
            "enable_parallel_streams": True,
        },
    },
    compilation_config={
        "cudagraph_mode": "FULL",
        "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
    },
)
```

A runnable example with graph capture sizes is available at
`examples/dual_inplace_parallel.py`.

## Admission Conditions

A batch is only split when **all** of the following hold (otherwise the engine
silently falls back to the normal path):

- `enabled` is `True` and `mode` is one of the split modes.
- The batch is a uniform decode batch (`inplace_parallel`/`inplace_serial`);
  non-uniform or speculative-decoding batches additionally require
  `enable_inplace_spec_decode`.
- `cudagraph_mode` is `FULL` or `PIECEWISE`.
- No LoRA adapters and no MLA models are active.
- MRoPE models require `enable_inplace_mrope`.
- The number of scheduled requests is at least `min_batch_size_for_split`.
- `inplace_parallel` additionally requires `enable_parallel_streams`.

## Configuration Reference

`additional_config["split_batch_config"]` accepts the following keys:

| Key | Type / Default | Description |
| --- | --- | --- |
| `enabled` | `bool`, `False` | Master switch. |
| `mode` | `str`, `"parallel_buffer"` | `parallel_buffer` (inert default, never splits), `inplace_serial`, `inplace_parallel`, `dual_pad`. |
| `num_splits` | `int`, `2` | Number of sub-batches; must be `2` for the inplace/dual-pad modes. |
| `enable_parallel_streams` | `bool`, `False` | Run the two halves on two NPU streams (required by `inplace_parallel`). |
| `min_batch_size_for_split` | `int`, `4` | Smallest batch size eligible for splitting. |
| `force_split` | `bool`, `False` | Bypass some planner heuristics for testing. |
| `cudagraph_split_pad_threshold` | `int`, `0` | Minimum saved padding required before `dual_pad` actually splits. |
| `enable_inplace_lazy_capture` | `bool`, `True` | Allow offset graphs to be captured lazily at first replay. |
| `inplace_parallel_replay_policy` | `str`, `"full_graph_parallel"` | `full_graph_parallel` or `piecewise_attention_parallel` (requires `PIECEWISE` cudagraph mode). |
| `inplace_split_planner_policy` | `str`, `"largest_lower"` | How the planner assigns requests to halves: `largest_lower` or `balanced`. |
| `inplace_offset_match_policy` | `str`, `"exact"` | Offset-graph key matching: `exact` or `bucket`. |
| `inplace_offset_capture_sizes` | `list[int]`, `None` | Explicit size list for offset graphs (auto-derived when unset). |
| `parallel_capture_sizes` | `list[int]`, `None` | Explicit size list for parallel-pool graphs (auto-derived when unset). |
| `inplace_offset_min_graph_tokens` | `int`, `1` | Lower bound for offset graph sizes. |
| `inplace_offset_max_padding_tokens` | `int`, `None` | Cap on padding tokens accepted for an offset graph. |
| `inplace_offset_max_padding_ratio` | `float`, `None` | Cap on padding ratio accepted for an offset graph. |
| `inplace_offset_max_graph_tokens_by_start` | `dict[int, int]`, `None` | Per-start-position graph size caps. |
| `inplace_offset_allowed_graph_tokens_by_start` | `dict[int, list[int]]`, `None` | Per-start-position allowed size lists. |
| `inplace_max_remainder_tokens` | `int`, `None` | Cap on remainder tokens after splitting. |
| `enable_inplace_spec_decode` | `bool`, `False` | Opt-in for speculative-decoding / mixed batches. |
| `enable_inplace_mrope` | `bool`, `False` | Opt-in for MRoPE models. |
| `inplace_validate_metadata_ptrs` | `bool`, `False` | Debug validation of metadata pointers per step. |
| `unified_capture_sizes` | `list[int]`, `[]` | Size list for the unified exact-size row graph (P11); also requires `VLLM_ASCEND_DUAL_UNIFIED_GEMM=1`. |

## Environment Variables

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_ASCEND_INPLACE_PARALLEL_MERGE_SYNC_POLICY` | `event_wait` | How split outputs are merged: `event_wait` (lightweight NPU events) or `host_sync` (full device sync). |
| `VLLM_ASCEND_INPLACE_PARALLEL_SPLIT_OUTPUT_MODE` | `auto` | Whether split outputs are cloned before merging: `auto`, `clone`, or `direct`. |
| `VLLM_ASCEND_INPLACE_PARALLEL_REUSE_SPLIT0_COS_SIN` | `1` | Reuse split-0 cos/sin buffers in split-1 for rotary embedding. |
| `VLLM_ASCEND_INPLACE_PARALLEL_REPLAY_STREAM_LIMITS` | `None` | Replay stream cube/vector limits, format `"main_cube,main_vector:parallel_cube,parallel_vector"`. |
| `VLLM_ASCEND_INPLACE_PARALLEL_UPDATE_STREAM_LIMITS` | `None` | Same format, for the update stream. |
| `VLLM_ASCEND_DUAL_UNIFIED_GEMM` | `0` | Set to `1` to enable the unified exact-size row graph (together with `unified_capture_sizes`). |
| `VLLM_ASCEND_SPLIT_INPLACE_DEBUG` | `0` | Set to `1` to emit the JSONL split-batch event log (see Debugging). |
| `VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE` | None | Output path for the JSONL event log. |

## Debugging

Set `VLLM_ASCEND_SPLIT_INPLACE_DEBUG=1` (and optionally
`VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE=<path>`) to record a per-step JSONL
event log with split decisions, planner plans, and replay outcomes. The
engine also logs one startup line
(`DUAL_INPLACE check: enabled=..., mode=..., cudagraph_mode=...`) confirming
whether the feature is active.

## Limitations

- Experimental: the default path is unchanged, but enabling the feature changes
  scheduling and graph-capture behavior; validate outputs and performance for
  your model and workload before production use.
- Rotary-embedding cos/sin buffers are double-allocated at startup to support
  two streams (one extra `max_num_batched_tokens x rope_dim` buffer pair).
- LoRA and MLA are not supported; MRoPE and speculative decoding require their
  explicit opt-in keys.
