#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import copy
from dataclasses import dataclass
from typing import Any

import torch
from vllm.distributed import get_tensor_model_parallel_world_size
from vllm.sequence import IntermediateTensors

from vllm_ascend.attention.utils import AscendCommonAttentionMetadata
from vllm_ascend.utils import enable_sp


def clone_attn_metadata_block_tables(attn_metadata: Any) -> Any:
    import dataclasses

    def _clone_single(meta: Any) -> Any:
        if meta is None or not dataclasses.is_dataclass(meta):
            return meta
        kwargs: dict = {}
        if getattr(meta, "block_tables", None) is not None:
            kwargs["block_tables"] = meta.block_tables.clone()
        for sub_field in ("prefill", "decode_meta"):
            sub = getattr(meta, sub_field, None)
            if sub is not None and dataclasses.is_dataclass(sub):
                sub_kwargs: dict = {}
                if getattr(sub, "block_tables", None) is not None:
                    sub_kwargs["block_tables"] = sub.block_tables.clone()
                if sub_kwargs:
                    kwargs[sub_field] = dataclasses.replace(sub, **sub_kwargs)
        return dataclasses.replace(meta, **kwargs) if kwargs else meta

    if isinstance(attn_metadata, dict):
        return {k: _clone_single(v) for k, v in attn_metadata.items()}
    if isinstance(attn_metadata, list):
        return [
            {k: _clone_single(v) for k, v in d.items()}
            if isinstance(d, dict) else _clone_single(d)
            for d in attn_metadata
        ]
    return _clone_single(attn_metadata)


@dataclass
class AscendUbatchMetadata:
    context: Any
    input_ids: torch.Tensor | None
    positions: torch.Tensor
    inputs_embeds: torch.Tensor | None
    intermediate_tensors: Any
    num_tokens: int


def dual_stream_attention_config(split_cfg: Any) -> Any:
    if split_cfg is None:
        return None
    return getattr(split_cfg, "dual_stream_attention_config", None)



def slice_split_batch_inputs(
    tokens_slice: slice,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    inputs_embeds: torch.Tensor | None,
    intermediate_tensors: Any,
    *,
    vllm_config: Any = None,
) -> tuple[torch.Tensor | None, torch.Tensor, torch.Tensor | None, Any]:
    sliced_input_ids = input_ids[tokens_slice] if input_ids is not None else None
    sliced_positions = positions[tokens_slice] if positions.ndim == 1 else positions[:, tokens_slice]
    sliced_inputs_embeds = inputs_embeds[tokens_slice] if inputs_embeds is not None else None
    if intermediate_tensors is not None:
        tp_size = get_tensor_model_parallel_world_size() if enable_sp(vllm_config) else 1
        if tp_size > 1:
            start = (tokens_slice.start + tp_size - 1) // tp_size
            stop = start + (tokens_slice.stop - tokens_slice.start + tp_size - 1) // tp_size
            it_slice = slice(start, stop)
        else:
            it_slice = tokens_slice
        sliced_intermediate_tensors = intermediate_tensors[it_slice]
    else:
        sliced_intermediate_tensors = None
    return sliced_input_ids, sliced_positions, sliced_inputs_embeds, sliced_intermediate_tensors



def expand_tensor_view_for_graph_slice(
    tensor: Any,
    graph_size: int,
    *,
    name: str,
    dim: int = 0,
    backing_tensor: torch.Tensor | None = None,
    backing_start: int = 0,
    allow_copy: bool = False,
) -> Any:
    if not isinstance(tensor, torch.Tensor):
        return tensor
    graph_size = int(graph_size)
    dim = int(dim)
    if graph_size <= int(tensor.shape[dim]):
        if dim == 0:
            return tensor[:graph_size]
        if dim == 1:
            return tensor[:, :graph_size]
        raise ValueError(f"Unsupported dim={dim} for {name}")
    shape = list(tensor.shape)
    shape[dim] = graph_size
    try:
        return tensor.as_strided(
            tuple(shape),
            tensor.stride(),
            storage_offset=tensor.storage_offset(),
        )
    except RuntimeError:
        if (isinstance(backing_tensor, torch.Tensor)
                and int(backing_tensor.shape[dim])
                >= int(backing_start) + graph_size):
            stop = int(backing_start) + graph_size
            if dim == 0:
                return backing_tensor[int(backing_start):stop]
            if dim == 1:
                return backing_tensor[:, int(backing_start):stop]
            raise ValueError(f"Unsupported dim={dim} for {name}")
        if allow_copy:
            expanded = tensor.new_zeros(tuple(shape))
            copy_slice = [slice(None)] * tensor.ndim
            copy_slice[dim] = slice(0, int(tensor.shape[dim]))
            expanded[tuple(copy_slice)].copy_(tensor)
            return expanded
        raise RuntimeError(
            "Inplace offset graph requires zero-copy backing view for "
            f"{name}: graph_size={graph_size}, dim={dim}, "
            f"tensor_shape={tuple(tensor.shape)}, "
            f"tensor_storage_offset={int(tensor.storage_offset())}, "
            f"backing_start={int(backing_start)}, "
            f"backing_shape={(None if backing_tensor is None else tuple(backing_tensor.shape))}")


def fill_tensor_local_tail(
    tensor: Any,
    start: int,
    stop: int,
    *,
    name: str,
    dim: int = 0,
) -> bool:
    if not isinstance(tensor, torch.Tensor) or stop <= start:
        return False
    start = int(start)
    stop = int(stop)
    dim = int(dim)
    if stop > int(tensor.shape[dim]):
        raise RuntimeError(
            f"Inplace offset padded local tail for {name} exceeds tensor "
            f"shape: tail_stop={stop}, shape={tuple(tensor.shape)}, "
            f"dim={dim}")
    if dim == 0:
        tensor[start:stop].fill_(0)
    elif dim == 1:
        tensor[:, start:stop].fill_(0)
    else:
        raise ValueError(f"Unsupported dim={dim} for {name}")
    return True


def pad_query_start_loc_for_graph(
    tensor: torch.Tensor,
    pad_reqs: int,
    query_len: int,
) -> torch.Tensor:
    if not isinstance(tensor, torch.Tensor) or pad_reqs <= 0:
        return tensor
    increments = torch.arange(
        1, pad_reqs + 1, dtype=tensor.dtype, device=tensor.device,
    ) * int(query_len)
    return torch.cat([tensor, tensor[-1] + increments], dim=0)


def metadata_positions_for_graph_slice(
    positions: Any,
    split_slice: Any,
    *,
    mrope_positions_gpu: Any = None,
    positions_gpu: Any = None,
) -> Any:
    if not isinstance(positions, torch.Tensor):
        return positions
    graph_tokens = int(split_slice.graph_num_tokens)
    token_dim = 1 if positions.ndim == 2 else 0
    token_start = int(split_slice.token_slice.start)
    backing = None
    if token_dim == 1:
        backing = mrope_positions_gpu
    if backing is None:
        backing = positions_gpu
    return expand_tensor_view_for_graph_slice(
        positions,
        graph_tokens,
        name="positions",
        dim=token_dim,
        backing_tensor=backing,
        backing_start=token_start,
    )


def stabilize_inplace_common_attn_metadata(
    common: Any,
    *,
    split_idx: int,
    split_slice: Any | None = None,
    fill_padding: bool = True,
    uniform_decode_query_len: int | None = None,
    mrope_positions_gpu: Any = None,
    positions_gpu: Any = None,
) -> Any:
    if split_slice is not None and split_slice.start_num_tokens > 0:
        actual_tokens = int(split_slice.num_tokens)
        graph_tokens = int(split_slice.graph_num_tokens)
        query_len = int(getattr(common, "decode_token_per_req", 0)
                       or uniform_decode_query_len or 1)
        if query_len <= 0:
            query_len = 1
        pad_tokens = graph_tokens - actual_tokens
        actual_reqs = int(common.num_reqs)
        graph_reqs = graph_tokens // query_len
        pad_reqs = graph_reqs - actual_reqs

        padded_common = copy(common)
        padded_common.query_start_loc = pad_query_start_loc_for_graph(
            common.query_start_loc, pad_reqs, query_len)
        padded_common.query_start_loc_cpu = pad_query_start_loc_for_graph(
            common.query_start_loc_cpu, pad_reqs, query_len)

        padded_common.seq_lens = expand_tensor_view_for_graph_slice(
            common.seq_lens, graph_reqs, name="seq_lens")
        padded_common.seq_lens_cpu = expand_tensor_view_for_graph_slice(
            common.seq_lens_cpu, graph_reqs, name="seq_lens_cpu")
        _seq_lens_cpu = getattr(common, "_seq_lens_cpu", None)
        if _seq_lens_cpu is not None:
            padded_common._seq_lens_cpu = expand_tensor_view_for_graph_slice(
                _seq_lens_cpu, graph_reqs, name="_seq_lens_cpu")
        padded_common.num_computed_tokens_cpu = expand_tensor_view_for_graph_slice(
            common.num_computed_tokens_cpu, graph_reqs,
            name="num_computed_tokens_cpu")
        padded_common.block_table_tensor = expand_tensor_view_for_graph_slice(
            common.block_table_tensor, graph_reqs,
            name="block_table_tensor")
        padded_common.slot_mapping = expand_tensor_view_for_graph_slice(
            common.slot_mapping, graph_tokens, name="slot_mapping")
        padded_common.positions = metadata_positions_for_graph_slice(
            common.positions, split_slice,
            mrope_positions_gpu=mrope_positions_gpu,
            positions_gpu=positions_gpu)

        if fill_padding:
            fill_tensor_local_tail(
                padded_common.seq_lens, actual_reqs, graph_reqs,
                name="seq_lens")
            fill_tensor_local_tail(
                padded_common.seq_lens_cpu, actual_reqs, graph_reqs,
                name="seq_lens_cpu")
            if getattr(padded_common, "_seq_lens_cpu", None) is not None:
                fill_tensor_local_tail(
                    padded_common._seq_lens_cpu, actual_reqs, graph_reqs,
                    name="_seq_lens_cpu")
            fill_tensor_local_tail(
                padded_common.num_computed_tokens_cpu, actual_reqs,
                graph_reqs, name="num_computed_tokens_cpu")
            fill_tensor_local_tail(
                padded_common.block_table_tensor, actual_reqs,
                graph_reqs, name="block_table_tensor")
            fill_tensor_local_tail(
                padded_common.slot_mapping, actual_tokens, graph_tokens,
                name="slot_mapping")
            positions_dim = 1 if (
                isinstance(padded_common.positions, torch.Tensor)
                and padded_common.positions.ndim == 2) else 0
            fill_tensor_local_tail(
                padded_common.positions, actual_tokens, graph_tokens,
                name="positions", dim=positions_dim)

        padded_common.num_reqs = graph_reqs
        padded_common.num_actual_tokens = graph_tokens
        padded_common.num_input_tokens = graph_tokens
        padded_common.max_query_len = max(int(common.max_query_len), query_len)
        padded_common.actual_seq_lengths_q = list(
            range(query_len, graph_tokens + 1, query_len))
        padded_common.graph_pad_size = graph_reqs
        return padded_common
    return common


def stabilize_inplace_common_attn_metadata_list(
    common_attn_metadata_list: list[Any],
    *,
    split_mode: str,
    inplace_split_plan: Any | None,
    uniform_decode_query_len: int | None = None,
    mrope_positions_gpu: Any = None,
    positions_gpu: Any = None,
) -> list[Any]:
    if split_mode not in ("inplace_serial", "inplace_parallel") or inplace_split_plan is None:
        return common_attn_metadata_list
    stabilized: list[Any] = []
    for split_idx, common_attn_metadata in enumerate(common_attn_metadata_list):
        split_slice = inplace_split_plan.split_slices[split_idx]
        stabilized.append(
            stabilize_inplace_common_attn_metadata(
                common_attn_metadata,
                split_idx=split_idx,
                split_slice=split_slice,
                uniform_decode_query_len=uniform_decode_query_len,
                mrope_positions_gpu=mrope_positions_gpu,
                positions_gpu=positions_gpu))
    return stabilized


def trim_split_output(output: Any, num_tokens: int) -> Any:
    if isinstance(output, torch.Tensor):
        return output[:num_tokens]
    if isinstance(output, list):
        return [trim_split_output(item, num_tokens) for item in output]
    if isinstance(output, tuple):
        return tuple(trim_split_output(item, num_tokens) for item in output)
    if isinstance(output, IntermediateTensors):
        return IntermediateTensors({k: v[:num_tokens] for k, v in output.tensors.items()})
    return output


def clone_split_output(output: Any) -> Any:
    if isinstance(output, torch.Tensor):
        return output.clone()
    if isinstance(output, list):
        return [clone_split_output(item) for item in output]
    if isinstance(output, tuple):
        return tuple(clone_split_output(item) for item in output)
    if isinstance(output, IntermediateTensors):
        return IntermediateTensors({k: v.clone() for k, v in output.tensors.items()})
    return output


def merge_split_outputs(outputs: list[Any]) -> Any:
    if not outputs:
        return None
    first = outputs[0]
    if isinstance(first, torch.Tensor):
        return torch.cat(outputs, dim=0)
    if isinstance(first, IntermediateTensors):
        return merge_intermediate_tensors(outputs)
    if isinstance(first, (list, tuple)):
        if any(len(output) != len(first) for output in outputs):
            raise RuntimeError("Cannot merge split outputs with different container lengths")
        merged = [
            merge_split_outputs([output[idx] for output in outputs])
            for idx in range(len(first))
        ]
        return type(first)(merged)
    return first


def merge_intermediate_tensors(tensor_list: list[IntermediateTensors]) -> IntermediateTensors:
    result = {}
    for key in tensor_list[0].tensors:
        result[key] = torch.cat([t.tensors[key] for t in tensor_list], dim=0)
    return IntermediateTensors(result)


def graph_token_slice_for_split(split_slice: Any) -> slice:
    start = int(split_slice.token_slice.start)
    return slice(start, start + int(split_slice.graph_num_tokens))


def padding_tail_slice_for_split(split_slice: Any) -> slice | None:
    actual_stop = int(split_slice.token_slice.stop)
    graph_stop = int(split_slice.token_slice.start) + int(split_slice.graph_num_tokens)
    if graph_stop <= actual_stop:
        return None
    return slice(actual_stop, graph_stop)


def fill_tensor_token_tail(
    tensor: torch.Tensor | None,
    tail_slice: slice,
    *,
    name: str,
    token_dim: int = 0,
) -> bool:
    if tensor is None:
        return False
    if tail_slice.stop > int(tensor.shape[token_dim]):
        raise RuntimeError(
            f"Inplace offset padded tail for {name} exceeds tensor "
            f"shape: tail_stop={tail_slice.stop}, "
            f"shape={tuple(tensor.shape)}, token_dim={token_dim}")
    if token_dim == 0:
        tensor[tail_slice].fill_(0)
    elif token_dim == 1:
        tensor[:, tail_slice].fill_(0)
    else:
        raise ValueError(f"Unsupported token_dim={token_dim} for {name}")
    return True


def maybe_expand_tensor_for_graph_slice(
    tensor: torch.Tensor | None,
    backing_tensor: torch.Tensor | None,
    graph_stop: int,
    *,
    name: str,
    token_dim: int = 0,
) -> torch.Tensor | None:
    if tensor is None:
        return None
    if graph_stop <= int(tensor.shape[token_dim]):
        return tensor
    if backing_tensor is None:
        raise RuntimeError(
            f"Inplace offset graph requires input backing buffer: "
            f"name={name}, graph_stop={int(graph_stop)}, "
            f"tensor_shape={tuple(tensor.shape)}, token_dim={token_dim}, "
            f"backing_shape=None")
    if graph_stop > int(backing_tensor.shape[token_dim]):
        raise RuntimeError(
            f"Inplace offset graph input backing buffer is too small: "
            f"name={name}, graph_stop={int(graph_stop)}, "
            f"tensor_shape={tuple(tensor.shape)}, token_dim={token_dim}, "
            f"backing_shape={tuple(backing_tensor.shape)}")
    if token_dim == 0:
        return backing_tensor[:graph_stop]
    if token_dim == 1:
        return backing_tensor[:, :graph_stop]
    raise ValueError(f"Unsupported token_dim={token_dim} for {name}")


def tokens_slice_for_inplace_execution(split_slice: Any) -> slice:
    if split_slice.start_num_tokens > 0:
        return graph_token_slice_for_split(split_slice)
    return split_slice.token_slice


def context_ubatch_slices_for_inplace(split_batch_slices) -> list:
    from vllm.v1.worker.ubatch_utils import UBatchSlice
    return [
        UBatchSlice(s.request_slice, tokens_slice_for_inplace_execution(s))
        for s in split_batch_slices
    ]