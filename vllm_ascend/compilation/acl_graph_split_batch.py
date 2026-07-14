# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
from typing import Any, TYPE_CHECKING

import torch
import torch_npu
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor
from vllm.logger import logger

from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend.compilation.acl_graph_diagnostics import SPLIT_INPLACE_DEBUG

if TYPE_CHECKING:
    from vllm_ascend.compilation.acl_graph import ACLGraphWrapper, ACLGraphEntry, GraphParams

_ACLGRAPH_REPLAY_GLOBAL_SYNC = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_REPLAY_GLOBAL_SYNC", "0")
    in ("1", "true", "True")
)


def is_allowed_inplace_lazy_capture(
    forward_context: Any,
    batch_descriptor: BatchDescriptor,
    aclgraph_runtime_mode: CUDAGraphMode,
) -> bool:
    start = int(getattr(batch_descriptor, "start_num_tokens", 0) or 0)
    if start <= 0:
        return False
    if aclgraph_runtime_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
        return False
    if not bool(getattr(forward_context, "allow_inplace_lazy_capture", False)):
        return False
    if getattr(forward_context, "split_inplace_mode", None) not in (
        "inplace_serial",
        "inplace_parallel",
    ):
        return False
    if getattr(batch_descriptor, "graph_variant", "") not in (
        "inplace_serial",
        "inplace_parallel",
    ):
        return False
    if getattr(batch_descriptor, "attention_backend", "") not in ("fia", "pa"):
        return False
    return True


def extract_block_table_from_metadata(metadata: Any):
    metadata_block_table = None
    metadata_block_source = None
    for attr in ("block_table", "block_tables", "block_table_tensor"):
        candidate = getattr(metadata, attr, None)
        if isinstance(candidate, torch.Tensor):
            metadata_block_table = candidate
            metadata_block_source = attr
            break

    if metadata_block_table is None:
        decode_metadata = getattr(metadata, "decode", None)
        for attr in ("block_table", "block_tables", "block_table_tensor"):
            candidate = getattr(decode_metadata, attr, None)
            if isinstance(candidate, torch.Tensor):
                metadata_block_table = candidate
                metadata_block_source = f"decode.{attr}"
                break

    return metadata_block_table, metadata_block_source


def refresh_block_table_in_place(
    dst_block_table: Any,
    src_block_table: Any,
) -> bool:
    if (not isinstance(dst_block_table, torch.Tensor)
            or not isinstance(src_block_table, torch.Tensor)):
        return False

    if dst_block_table.data_ptr() == src_block_table.data_ptr():
        return False

    try:
        if dst_block_table.ndim == 2 and src_block_table.ndim == 2:
            rows = min(dst_block_table.shape[0], src_block_table.shape[0])
            cols = min(dst_block_table.shape[1], src_block_table.shape[1])
            if rows <= 0 or cols <= 0:
                return False
            dst_block_table[:rows, :cols].copy_(
                src_block_table[:rows, :cols], non_blocking=False)
            return True

        count = min(dst_block_table.numel(), src_block_table.numel())
        if count <= 0:
            return False
        dst_block_table.view(-1)[:count].copy_(
            src_block_table.view(-1)[:count], non_blocking=False)
        return True
    except Exception:
        return False


def should_template_fia_seq_lens(forward_context: Any) -> bool:
    batch_descriptor = getattr(forward_context, "batch_descriptor", None)
    return (getattr(batch_descriptor, "capture_metadata_mode", "") == "template"
            and getattr(batch_descriptor, "attention_backend", "") == "fia")


def _get_fia_key_t(key_tensor: Any, fallback: int) -> int:
    if isinstance(key_tensor, torch.Tensor):
        if key_tensor.ndim > 1:
            return int(key_tensor.shape[1])
        if key_tensor.ndim > 0 and int(fallback) <= 0:
            return int(key_tensor.shape[0])
    return int(fallback)


def maybe_template_fia_seq_lens(
    forward_context: Any,
    seq_lens: Any,
    target_t: int,
    *,
    source: str = "",
) -> Any:
    if not should_template_fia_seq_lens(forward_context):
        return seq_lens
    if not isinstance(seq_lens, (list, tuple)):
        return seq_lens
    if len(seq_lens) == 0:
        return seq_lens
    result = list(seq_lens)
    result[-1] = target_t
    return type(seq_lens)(result)


def _iter_attn_metadata_objects(attn_metadata: Any):
    if isinstance(attn_metadata, dict):
        for value in attn_metadata.values():
            yield from _iter_attn_metadata_objects(value)
        return
    if isinstance(attn_metadata, list):
        for value in attn_metadata:
            yield from _iter_attn_metadata_objects(value)
        return
    if attn_metadata is not None:
        yield attn_metadata


def template_fia_seq_lens_list(attn_metadata: Any, target_t: int) -> int:
    updated = 0
    for metadata_obj in _iter_attn_metadata_objects(attn_metadata):
        seq_lens_list = getattr(metadata_obj, "seq_lens_list", None)
        if not isinstance(seq_lens_list, list) or not seq_lens_list:
            continue
        templated = list(seq_lens_list)
        templated[-1] = int(target_t)
        setattr(metadata_obj, "seq_lens_list", templated)
        updated += 1
    return updated


GraphParamKey = int | BatchDescriptor


def get_graph_param_key(
    forward_context: Any,
    runtime_shape: int,
) -> GraphParamKey:
    desc = getattr(forward_context, "batch_descriptor", None)
    if not isinstance(desc, BatchDescriptor):
        return runtime_shape

    start = desc.start_num_tokens
    has_descriptor_variant = (
        start > 0
        or bool(desc.graph_variant)
        or bool(desc.attention_backend)
        or bool(desc.capture_metadata_mode)
    )
    if has_descriptor_variant:
        return desc
    return runtime_shape


def resolve_graph_param_key(graph_params, forward_context, runtime_shape):
    param_key = get_graph_param_key(forward_context, runtime_shape)
    if param_key in graph_params.attn_params:
        return param_key
    if isinstance(param_key, BatchDescriptor):
        int_key = int(param_key.num_tokens)
        if int_key in graph_params.attn_params:
            return int_key
    return param_key


def ensure_graph_param_key(graph_params, key):
    if graph_params is None:
        return
    graph_params.events.setdefault(key, [])
    graph_params.handles.setdefault(key, [])
    graph_params.attn_params.setdefault(key, [])
    graph_params.workspaces.setdefault(key, None)
    if hasattr(graph_params, 'conv1d_params'):
        graph_params.conv1d_params.setdefault(key, [])
        graph_params.conv1d_handles.setdefault(key, [])
        graph_params.conv1d_events.setdefault(key, [])


def require_graph_param_key(graph_params, key, *, op):
    if graph_params is None:
        raise KeyError(f"Missing GraphParams for {op}: {key!r}")
    if key not in graph_params.attn_params:
        raise KeyError(f"Missing GraphParams key for {op}: {key!r}")
    if key not in graph_params.handles or key not in graph_params.events:
        raise KeyError(f"Incomplete GraphParams key for {op}: {key!r}")


def has_dual_stream_attention_metadata(forward_context) -> bool:
    dual_metadata = getattr(forward_context,
                            "dual_stream_attention_metadata", None)
    return isinstance(dual_metadata, list) and len(dual_metadata) == 2


def get_dual_attention_update_metadata(
    forward_context: Any, dual_metadata: Any, split_idx: int,
    key: Any) -> Any:
    runtime_metadata = getattr(
        forward_context, "macro_graph_dual_attention_update_metadata", None)
    if runtime_metadata is not None:
        try:
            return runtime_metadata[split_idx][key]
        except (KeyError, IndexError, TypeError) as exc:
            raise KeyError(
                "macro_graph_dual_attention_update_metadata is missing "
                f"split={split_idx}, layer={key!r}") from exc
    return dual_metadata[split_idx][key]


def _update_attn_fia_params(update_stream, forward_context, runtime_shape,
                            refresh_block_table: bool = False,
                            in_parallel_streams: bool = False):
    from vllm_ascend.compilation.acl_graph import get_graph_params
    graph_params = get_graph_params(in_parallel_streams)
    if graph_params is None:
        return
    param_key = resolve_graph_param_key(graph_params, forward_context, runtime_shape)
    if param_key not in graph_params.attn_params or not graph_params.attn_params[param_key]:
        return
    if param_key not in graph_params.handles or not graph_params.handles[param_key]:
        return
    if param_key not in graph_params.events or not graph_params.events[param_key]:
        return

    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ):
            (query, key_cache, value, block_tables, attn_mask, block_size,
             seq_lens, actual_seq_lengths_q_capture, num_kv_heads, num_heads, scale,
             attn_output, softmax_lse, *rest) = param

            metadata = forward_context.attn_metadata[key]
            seq_lens = maybe_template_fia_seq_lens(
                forward_context, metadata.seq_lens_list,
                _get_fia_key_t(key_cache, block_size),
                source=f"acl_graph_update:{key}")
            actual_seq_lengths_q = metadata.actual_seq_lengths_q

            metadata_block_table, metadata_block_source = extract_block_table_from_metadata(
                metadata)
            if refresh_block_table:
                block_table_refreshed = refresh_block_table_in_place(
                    block_tables, metadata_block_table)
                if SPLIT_INPLACE_DEBUG:
                    from vllm_ascend.inplace_split_debug import log_event, tensor_info
                    log_event(
                        "fia_block_table_refresh",
                        {
                            "key": key,
                            "refreshed": block_table_refreshed,
                            "in_parallel_streams": in_parallel_streams,
                            "graph_block_table": tensor_info(block_tables),
                            "meta_block_table": tensor_info(metadata_block_table),
                            "meta_source": metadata_block_source,
                        },
                    )

            if SPLIT_INPLACE_DEBUG:
                from vllm_ascend.inplace_split_debug import log_event, tensor_info
                log_event(
                    "fia_update_before_call",
                    {
                        "key": key,
                        "in_parallel_streams": in_parallel_streams,
                        "query": tensor_info(query),
                        "key_cache": tensor_info(key_cache),
                        "value": tensor_info(value),
                        "block_tables": tensor_info(block_tables),
                        "attn_mask": tensor_info(attn_mask),
                        "attn_output": tensor_info(attn_output),
                        "softmax_lse": tensor_info(softmax_lse),
                        "block_size": block_size,
                        "actual_seq_lengths_q": actual_seq_lengths_q,
                        "seq_lens": seq_lens,
                        "seq_lens_list": getattr(metadata, "seq_lens_list", None),
                        "num_kv_heads": num_kv_heads,
                        "num_heads": num_heads,
                        "should_template": should_template_fia_seq_lens(forward_context),
                    },
                )
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu.npu_fused_infer_attention_score.out(
                query=query,
                key=key_cache,
                value=value,
                block_table=block_tables,
                atten_mask=attn_mask,
                input_layout="TND",
                block_size=block_size,
                actual_seq_lengths=actual_seq_lengths_q,
                actual_seq_lengths_kv=seq_lens,
                num_key_value_heads=num_kv_heads,
                num_heads=num_heads,
                scale=scale,
                sparse_mode=3,
                workspace=graph_params.workspaces.get(param_key),
                out=[attn_output, softmax_lse],
            )
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def _update_attn_pa_params(update_stream, forward_context, runtime_shape,
                           refresh_block_table: bool = False,
                           in_parallel_streams: bool = False):
    from vllm_ascend.compilation.acl_graph import get_graph_params
    graph_params = get_graph_params(in_parallel_streams)
    if graph_params is None:
        return
    param_key = resolve_graph_param_key(graph_params, forward_context, runtime_shape)
    if param_key not in graph_params.attn_params or not graph_params.attn_params[param_key]:
        return
    if param_key not in graph_params.handles or not graph_params.handles[param_key]:
        return
    if param_key not in graph_params.events or not graph_params.events[param_key]:
        return

    with torch.npu.stream(update_stream):
        for key, param, handle, event in zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ):
            (
                query,
                key_cache,
                value_cache,
                num_kv_heads,
                num_heads,
                scale,
                block_table,
                seq_lens,
                output,
            ) = param
            seq_lens = forward_context.attn_metadata[key].seq_lens

            metadata = forward_context.attn_metadata[key]
            metadata_block_table, metadata_block_source = extract_block_table_from_metadata(
                metadata)
            if refresh_block_table:
                refresh_block_table_in_place(
                    block_table, metadata_block_table)

            workspace = torch_npu._npu_paged_attention_get_workspace(
                query=query,
                key_cache=key_cache,
                value_cache=value_cache,
                num_kv_heads=num_kv_heads,
                num_heads=num_heads,
                scale_value=scale,
                block_table=block_table,
                context_lens=seq_lens,
                out=output)
            torch.npu.graph_task_update_begin(update_stream, handle)
            torch_npu._npu_paged_attention(query=query,
                                           key_cache=key_cache,
                                           value_cache=value_cache,
                                           num_kv_heads=num_kv_heads,
                                           num_heads=num_heads,
                                           scale_value=scale,
                                           block_table=block_table,
                                           context_lens=seq_lens,
                                           out=output,
                                           workspace=workspace)
            torch.npu.graph_task_update_end(update_stream)

            event.record(update_stream)


def update_attn_params(update_stream, forward_context, runtime_shape,
                       vllm_config, in_parallel_streams: bool = False):
    if has_dual_stream_attention_metadata(forward_context):
        _update_attn_dual_fia_params(update_stream, forward_context,
                                     runtime_shape,
                                     in_parallel_streams=in_parallel_streams)
    elif using_paged_attention(runtime_shape, vllm_config):
        _update_attn_pa_params(update_stream, forward_context, runtime_shape,
                               in_parallel_streams=in_parallel_streams)
    else:
        _update_attn_fia_params(update_stream, forward_context, runtime_shape,
                                in_parallel_streams=in_parallel_streams)


def update_attn_params_split(update_stream, forward_context,
                             runtime_shape, vllm_config,
                             in_parallel_streams: bool = False):
    if has_dual_stream_attention_metadata(forward_context):
        _update_attn_dual_fia_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
            in_parallel_streams=in_parallel_streams,
        )
    elif using_paged_attention(runtime_shape, vllm_config):
        _update_attn_pa_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
            in_parallel_streams=in_parallel_streams,
        )
    else:
        _update_attn_fia_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
            in_parallel_streams=in_parallel_streams,
        )


def _update_attn_dual_fia_params(update_stream, forward_context,
                                 runtime_shape,
                                 refresh_block_table: bool = True,
                                 in_parallel_streams: bool = False):
    from vllm_ascend.compilation.acl_graph import get_graph_params
    graph_params = get_graph_params(in_parallel_streams)
    if graph_params is None:
        return
    param_key = get_graph_param_key(forward_context, runtime_shape)
    if param_key not in graph_params.attn_params:
        return
    require_graph_param_key(graph_params, param_key,
                            op="_update_attn_dual_fia_params")
    dual_metadata = getattr(forward_context,
                            "dual_stream_attention_metadata", None)
    if not isinstance(dual_metadata, list) or len(dual_metadata) != 2:
        return

    with torch.npu.stream(update_stream):
        attn_items = list(zip(
                forward_context.attn_metadata,
                graph_params.attn_params[param_key],
                graph_params.handles[param_key],
                graph_params.events[param_key],
        ))
        for layer_idx, (key, param, handles, events) in enumerate(attn_items):
            if not isinstance(param, tuple) or len(param) < 2:
                continue

            if param[0] == "dual_stream_fia_pta":
                if len(param) != 5:
                    continue
                if not hasattr(torch.npu, "dual_fused_infer_attention_score_update"):
                    continue

                _, query, attn_output, split_params, split_ranges = param
                if not isinstance(split_params, list) or len(split_params) != 2:
                    continue
                if not isinstance(split_ranges, list) or len(split_ranges) != 2:
                    continue
                if isinstance(events, (list, tuple)):
                    update_events = tuple(events)
                elif events is None:
                    update_events = ()
                else:
                    update_events = (events,)

                split_update_params = []
                for split_idx, split_param in enumerate(split_params):
                    (query_view, key_cache, value, block_tables, attn_mask,
                     block_size, seq_lens, query_start_loc, num_kv_heads,
                     num_heads, scale, attn_output_view, softmax_lse,
                     workspace_key) = split_param

                    metadata = dual_metadata[split_idx][key]
                    runtime_metadata = get_dual_attention_update_metadata(
                        forward_context, dual_metadata, split_idx, key)
                    seq_lens = maybe_template_fia_seq_lens(
                        forward_context,
                        getattr(runtime_metadata, "seq_lens_list",
                                metadata.seq_lens_list),
                        _get_fia_key_t(key_cache, block_size),
                        source=f"acl_graph_update_dual:{key}:{split_idx}")
                    actual_seq_lengths_q = getattr(
                        runtime_metadata, "actual_seq_lengths_q",
                        metadata.actual_seq_lengths_q)
                    metadata_block_table, _ = (
                        extract_block_table_from_metadata(runtime_metadata))
                    if refresh_block_table:
                        refresh_block_table_in_place(
                            block_tables, metadata_block_table)

                    split_update_params.append({
                        "key_cache": key_cache,
                        "value": value,
                        "block_tables": block_tables,
                        "attn_mask": attn_mask,
                        "block_size": block_size,
                        "seq_lens": seq_lens,
                        "actual_seq_lengths_q": actual_seq_lengths_q,
                        "num_kv_heads": num_kv_heads,
                        "num_heads": num_heads,
                        "scale": scale,
                        "softmax_lse": softmax_lse,
                        "workspace":
                        graph_params.workspaces.get(workspace_key),
                    })

                split0, split1 = split_update_params
                split_start_0, split_graph_tokens_0 = split_ranges[0]
                split_start_1, split_graph_tokens_1 = split_ranges[1]
                torch.npu.dual_fused_infer_attention_score_update(
                    update_stream,
                    handles,
                    query,
                    split0["key_cache"],
                    split0["value"],
                    attn_output,
                    block_table_0=split0["block_tables"],
                    block_table_1=split1["block_tables"],
                    actual_seq_lengths_0=split0["actual_seq_lengths_q"],
                    actual_seq_lengths_1=split1["actual_seq_lengths_q"],
                    actual_seq_lengths_kv_0=split0["seq_lens"],
                    actual_seq_lengths_kv_1=split1["seq_lens"],
                    split_start_0=int(split_start_0),
                    split_graph_tokens_0=int(split_graph_tokens_0),
                    split_start_1=int(split_start_1),
                    split_graph_tokens_1=int(split_graph_tokens_1),
                    atten_mask=split0["attn_mask"],
                    workspace_0=split0["workspace"],
                    workspace_1=split1["workspace"],
                    softmax_lse_0=split0["softmax_lse"],
                    softmax_lse_1=split1["softmax_lse"],
                    num_heads=split0["num_heads"],
                    scale=split0["scale"],
                    block_size=split0["block_size"],
                    num_key_value_heads=split0["num_kv_heads"],
                    sparse_mode=3,
                    input_layout="TND",
                    softmax_lse_flag=False,
                )
                for event in update_events:
                    event.record(update_stream)
                continue

            if param[0] == "dual_stream_fia":
                split_params = param[1]
                if not isinstance(split_params, list) or len(split_params) != 2:
                    continue
                if not isinstance(handles, list) or len(handles) != 2:
                    continue
                if not isinstance(events, list) or len(events) != 2:
                    continue

                for split_idx, split_param in enumerate(split_params):
                    (query, key_cache, value, block_tables, attn_mask, block_size,
                     seq_lens, query_start_loc, num_kv_heads, num_heads, scale,
                     attn_output, softmax_lse, workspace_key) = split_param

                    metadata = dual_metadata[split_idx][key]
                    seq_lens = maybe_template_fia_seq_lens(
                        forward_context,
                        metadata.seq_lens_list,
                        _get_fia_key_t(key_cache, block_size),
                        source=f"acl_graph_update_dual:{key}:{split_idx}")
                    actual_seq_lengths_q = metadata.actual_seq_lengths_q

                    metadata_block_table, _ = extract_block_table_from_metadata(metadata)
                    if refresh_block_table:
                        refresh_block_table_in_place(
                            block_tables, metadata_block_table)

                    torch.npu.graph_task_update_begin(update_stream,
                                                      handles[split_idx])
                    torch_npu.npu_fused_infer_attention_score.out(
                        query=query,
                        key=key_cache,
                        value=value,
                        block_table=block_tables,
                        atten_mask=attn_mask,
                        input_layout="TND",
                        block_size=block_size,
                        actual_seq_lengths=actual_seq_lengths_q,
                        actual_seq_lengths_kv=seq_lens,
                        num_key_value_heads=num_kv_heads,
                        num_heads=num_heads,
                        scale=scale,
                        sparse_mode=3,
                        workspace=graph_params.workspaces.get(workspace_key),
                        out=[attn_output, softmax_lse],
                    )
                    torch.npu.graph_task_update_end(update_stream)
                    events[split_idx].record(update_stream)


def dual_stream_attention_plan_to_slices(
    plan: Any,
    query_len: int) -> tuple[list[Any], list[Any]]:
    from vllm.v1.worker.ubatch_utils import UBatchSlice
    from vllm_ascend.worker.inplace_split_utils import SplitBatchSlice
    query_len = int(query_len)
    if query_len <= 0:
        raise RuntimeError(
            "dual_stream_attention_config requires positive query_len")
    split_batch_slices: list[Any] = []
    for split_idx in range(2):
        token_start = int(plan.split_start_tokens[split_idx])
        actual_tokens = int(plan.split_actual_tokens[split_idx])
        graph_tokens = int(plan.split_graph_tokens[split_idx])
        if token_start % query_len != 0:
            raise RuntimeError(
                "dual_stream_attention_config split_start_tokens must be "
                "request-aligned")
        if actual_tokens % query_len != 0 or graph_tokens % query_len != 0:
            raise RuntimeError(
                "dual_stream_attention_config split tokens must be "
                "request-aligned")
        request_start = token_start // query_len
        request_stop = request_start + actual_tokens // query_len
        split_batch_slices.append(
            SplitBatchSlice(
                request_slice=slice(request_start, request_stop),
                token_slice=slice(token_start, token_start + actual_tokens),
                padded_num_tokens=graph_tokens,
                start_num_tokens=token_start,
            ))
    ubatch_slices = [
        UBatchSlice(s.request_slice, s.token_slice)
        for s in split_batch_slices
    ]
    return split_batch_slices, ubatch_slices