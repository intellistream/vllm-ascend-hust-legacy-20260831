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

import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import BatchDescriptor, get_forward_context, override_forward_context
from vllm.logger import logger
from vllm.sequence import IntermediateTensors

from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.attention.utils import AscendCommonAttentionMetadata, using_paged_attention
from vllm_ascend.worker.inplace_split_ops import (
    AscendUbatchMetadata,
    clone_split_output,
    context_ubatch_slices_for_inplace,
    dual_stream_attention_config,
    fill_tensor_token_tail,
    maybe_expand_tensor_for_graph_slice,
    merge_split_outputs,
    padding_tail_slice_for_split,
    slice_split_batch_inputs,
    stabilize_inplace_common_attn_metadata_list,
    tokens_slice_for_inplace_execution,
    trim_split_output,
)
from vllm_ascend.worker.inplace_split_utils import SplitBatchSlice

if TYPE_CHECKING:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class InplaceSplitRunner:
    """Orchestrates inplace split-batch execution.

    Holds a reference to NPUModelRunner via composition rather than inheritance.
    """

    def __init__(self, runner: "NPUModelRunner"):
        self._runner = runner

    @property
    def device(self) -> torch.device:
        return self._runner.device

    @property
    def vllm_config(self):
        return self._runner.vllm_config

    @property
    def uniform_decode_query_len(self) -> int | None:
        return self._runner.uniform_decode_query_len

    @property
    def stream_main(self) -> torch.npu.Stream | None:
        return self._runner.stream_main

    @property
    def stream_parallel(self) -> torch.npu.Stream | None:
        return self._runner.stream_parallel

    @property
    def ascend_config(self):
        return self._runner.ascend_config

    @property
    def model_config(self):
        return self._runner.model_config

    @property
    def pcp_size(self):
        return self._runner.pcp_size

    @property
    def dcp_size(self):
        return self._runner.dcp_size

    @property
    def block_size(self):
        return self._runner.block_size

    @property
    def cudagraph_dispatcher(self):
        return self._runner.cudagraph_dispatcher

    @property
    def compilation_config(self):
        return self._runner.compilation_config

    @property
    def model(self):
        return self._runner.model


    def _stabilize_inplace_common_attn_metadata_list(
            self,
            common_attn_metadata_list: list[Any],
            *,
            split_mode: str,
            inplace_split_plan: Any | None,
    ) -> list[Any]:
        mrope_positions_gpu = getattr(
            getattr(self._runner, "mrope_positions", None), "gpu", None)
        positions_gpu = getattr(
            getattr(self._runner, "positions", None), "gpu", None)
        return stabilize_inplace_common_attn_metadata_list(
            common_attn_metadata_list,
            split_mode=split_mode,
            inplace_split_plan=inplace_split_plan,
            uniform_decode_query_len=self.uniform_decode_query_len,
            mrope_positions_gpu=mrope_positions_gpu,
            positions_gpu=positions_gpu)

    def inplace_split_precheck_reason(
        self,
        *,
        split_batch_config: Any,
        cudagraph_mode: CUDAGraphMode,
        num_scheduled_tokens_np: np.ndarray,
        num_reqs: int,
        has_lora: bool,
        is_mla: bool,
        is_mrope: bool,
        spec_decode_enabled: bool,
        uniform_decode: bool = True,
    ) -> str | None:
        from vllm_ascend.worker.inplace_split_utils import (
            _INPLACE_SPLIT_MODES,
            NO_SPLIT_BATCH_TOO_SMALL,
            NO_SPLIT_CUDAGRAPH_MODE_NOT_FULL,
            NO_SPLIT_LORA_CONFLICT,
            NO_SPLIT_MLA_CONFLICT,
            NO_SPLIT_MODE_NOT_INPLACE,
            NO_SPLIT_MROPE_CONFLICT,
            NO_SPLIT_NON_UNIFORM_DECODE,
            NO_SPLIT_PARALLEL_STREAMS_DISABLED,
            NO_SPLIT_SPEC_DECODE_CONFLICT,
        )
        mode = split_batch_config.mode
        if mode not in _INPLACE_SPLIT_MODES:
            return NO_SPLIT_MODE_NOT_INPLACE
        if not split_batch_config.enabled:
            return NO_SPLIT_MODE_NOT_INPLACE
        if mode == "inplace_parallel" and not split_batch_config.enable_parallel_streams:
            return NO_SPLIT_PARALLEL_STREAMS_DISABLED
        if not uniform_decode and not getattr(split_batch_config, "enable_mixed_request_split", False) and not getattr(split_batch_config, "enable_inplace_spec_decode", False):
            return NO_SPLIT_NON_UNIFORM_DECODE
        if uniform_decode and cudagraph_mode not in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE):
            return NO_SPLIT_CUDAGRAPH_MODE_NOT_FULL
        if not uniform_decode and cudagraph_mode != CUDAGraphMode.PIECEWISE:
            return NO_SPLIT_CUDAGRAPH_MODE_NOT_FULL
        if spec_decode_enabled and not split_batch_config.enable_inplace_spec_decode:
            return NO_SPLIT_SPEC_DECODE_CONFLICT
        if has_lora:
            return NO_SPLIT_LORA_CONFLICT
        if is_mla:
            return NO_SPLIT_MLA_CONFLICT
        if is_mrope and not split_batch_config.enable_inplace_mrope:
            return NO_SPLIT_MROPE_CONFLICT
        if num_reqs < split_batch_config.min_batch_size_for_split:
            return NO_SPLIT_BATCH_TOO_SMALL
        return None

    def should_split(
        self,
        split_batch_config: Any,
        cudagraph_mode: CUDAGraphMode,
        num_scheduled_tokens_np: np.ndarray,
        num_reqs: int,
        total_num_tokens: int,
        has_lora: bool,
        is_mla: bool,
        is_mrope: bool,
        spec_decode_enabled: bool,
        cudagraph_capture_sizes: list[int],
    ) -> tuple[Any, str]:
        from vllm_ascend.inplace_split_debug import (
            is_enabled as _debug_enabled,
        )
        from vllm_ascend.inplace_split_debug import (
            log_event,
            next_step_id,
            set_current_step_id,
        )
        from vllm_ascend.worker.inplace_split_utils import (
            INPLACE_SPLIT_DRY_RUN,
            NO_SPLIT_ATTENTION_BACKEND_MISMATCH,
            create_inplace_split_batch_slices,
            inplace_split_first_graph_matches_attention_backend,
            select_inplace_attention_backend,
        )
        precheck_reason = self.inplace_split_precheck_reason(
            split_batch_config=split_batch_config,
            cudagraph_mode=cudagraph_mode,
            num_scheduled_tokens_np=num_scheduled_tokens_np,
            num_reqs=num_reqs,
            has_lora=has_lora,
            is_mla=is_mla,
            is_mrope=is_mrope,
            spec_decode_enabled=spec_decode_enabled,
        )
        if precheck_reason is not None:
            if _debug_enabled():
                log_event("inplace_split_precheck_failed", {"reason": precheck_reason})
            return None, precheck_reason

        uniform_decode_query_len = self.uniform_decode_query_len
        if uniform_decode_query_len < 1:
            return None, "no_split_non_uniform_decode"

        offset_capture_sizes = split_batch_config.inplace_offset_capture_sizes
        if offset_capture_sizes is None:
            offset_capture_sizes = cudagraph_capture_sizes

        if _debug_enabled():
            step_id = next_step_id()
            set_current_step_id(step_id)
            log_event("inplace_split_planning_start", {
                "total_num_tokens": total_num_tokens,
                "num_reqs": num_reqs,
                "uniform_decode_query_len": uniform_decode_query_len,
            })

        split_plan, reason = create_inplace_split_batch_slices(
            num_scheduled_tokens_per_request=num_scheduled_tokens_np,
            total_num_tokens=total_num_tokens,
            uniform_decode_query_len=uniform_decode_query_len,
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            inplace_max_remainder_tokens=split_batch_config.inplace_max_remainder_tokens,
            offset_match_policy=split_batch_config.inplace_offset_match_policy,
            offset_capture_sizes=offset_capture_sizes,
            offset_min_graph_tokens=split_batch_config.inplace_offset_min_graph_tokens,
            offset_max_padding_tokens=split_batch_config.inplace_offset_max_padding_tokens,
            offset_max_padding_ratio=split_batch_config.inplace_offset_max_padding_ratio,
            offset_max_graph_tokens_by_start=split_batch_config.inplace_offset_max_graph_tokens_by_start,
            offset_allowed_graph_tokens_by_start=split_batch_config.inplace_offset_allowed_graph_tokens_by_start,
            first_tokens_policy=split_batch_config.inplace_split_planner_policy,
        )

        if (split_plan is not None
                and not inplace_split_first_graph_matches_attention_backend(
                    split_plan,
                    lambda shape: using_paged_attention(shape, self.vllm_config),
                )):
            reason = NO_SPLIT_ATTENTION_BACKEND_MISMATCH
            split_plan = None

        if _debug_enabled():
            log_event("inplace_split_planning_result", {
                "reason": reason,
                "has_plan": split_plan is not None,
            })

        logger.info(
            "DUAL_INPLACE split decision: has_plan=%s, reason=%s, total_tokens=%d",
            split_plan is not None, reason,
            split_plan.total_num_tokens if split_plan else 0,
        )

        return split_plan, reason

    def _has_aclgraph_for_context(self, forward_context: Any) -> bool:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
        batch_descriptor = getattr(forward_context, "batch_descriptor", None)
        if batch_descriptor is None:
            return False
        in_parallel_streams = forward_context.in_parallel_streams
        if isinstance(self.model, ACLGraphWrapper):
            return self.model.has_graph(batch_descriptor, in_parallel_streams)
        return False

    def _expand_inplace_inputs_for_graph_slice(
        self,
        graph_stop: int,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        input_ids_backing = getattr(getattr(self._runner, "input_ids", None), "gpu", None)
        positions_backing = getattr(getattr(self._runner, "positions", None), "gpu", None)
        inputs_embeds_backing = getattr(getattr(self._runner, "inputs_embeds", None), "gpu", None)

        input_ids = maybe_expand_tensor_for_graph_slice(
            input_ids, input_ids_backing, graph_stop, name="input_ids")

        if positions is not None:
            positions_token_dim = 1 if positions.ndim == 2 else 0
            if positions_token_dim == 1:
                mrope_backing = getattr(getattr(self._runner, "mrope_positions", None), "gpu", None)
                if mrope_backing is not None:
                    positions_backing = mrope_backing
            positions = maybe_expand_tensor_for_graph_slice(
                positions, positions_backing, graph_stop,
                name="positions", token_dim=positions_token_dim)

        inputs_embeds = maybe_expand_tensor_for_graph_slice(
            inputs_embeds, inputs_embeds_backing, graph_stop, name="inputs_embeds")
        return input_ids, positions, inputs_embeds

    def _fill_inplace_padding_tail(
        self,
        split_slice: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: Any | None,
        *,
        stream: Any | None = None,
    ) -> dict[str, Any]:
        tail_slice = padding_tail_slice_for_split(split_slice)
        if tail_slice is None:
            return {"tail_filled": False}

        def _fill() -> dict[str, Any]:
            fill_tensor_token_tail(input_ids, tail_slice, name="input_ids")
            if positions is not None:
                positions_token_dim = 1 if positions.ndim == 2 else 0
                fill_tensor_token_tail(
                    positions, tail_slice, name="positions", token_dim=positions_token_dim)
            fill_tensor_token_tail(inputs_embeds, tail_slice, name="inputs_embeds")
            if intermediate_tensors is not None:
                for name, tensor in intermediate_tensors.tensors.items():
                    fill_tensor_token_tail(tensor, tail_slice, name=name)
            return {"tail_filled": True}

        if stream is not None:
            with torch.npu.stream(stream):
                return _fill()
        return _fill()

    def _context_ubatch_slices_for_inplace(self, split_batch_slices) -> list:
        return context_ubatch_slices_for_inplace(split_batch_slices)


    def _prepare_inplace_split_inputs_for_execution(
        self,
        split_batch_slices: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: Any,

        stream_for_split: Any | None = None,
    ) -> list[dict]:
        prepared = []
        for idx, split_slice in enumerate(split_batch_slices):
            ts = tokens_slice_for_inplace_execution(split_slice)
            graph_stop = int(ts.stop)
            split_input_ids, split_positions, split_inputs_embeds = (
                self._expand_inplace_inputs_for_graph_slice(
                    graph_stop, input_ids, positions, inputs_embeds))
            stream = stream_for_split(idx) if stream_for_split else None
            padding_tail_payload = self._fill_inplace_padding_tail(
                split_slice, split_input_ids, split_positions,
                split_inputs_embeds, intermediate_tensors,
                stream=stream)
            prepared.append({
                "tokens_slice": ts,
                "input_ids": split_input_ids,
                "positions": split_positions,
                "inputs_embeds": split_inputs_embeds,
                "padding_tail_payload": padding_tail_payload,
            })
        return prepared

    def needs_inplace_serial_offset_capture(
        self, metadata: AscendUbatchMetadata
    ) -> bool:
        context = metadata.context
        batch_descriptor = getattr(context, "batch_descriptor", None)
        if batch_descriptor is None:
            return False
        cg_mode = getattr(context, "cudagraph_runtime_mode", CUDAGraphMode.NONE)
        start_nt = int(getattr(batch_descriptor, "start_num_tokens", 0) or 0)
        allow_lazy = bool(getattr(context, "allow_inplace_lazy_capture", False))
        has_graph = self._has_aclgraph_for_context(context)
        result = (cg_mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE)
                  and start_nt > 0 and allow_lazy and not has_graph)

        from vllm_ascend.inplace_split_debug import is_enabled as _is_enabled
        from vllm_ascend.inplace_split_debug import log_event as _log_event
        if _is_enabled():
            in_parallel_streams = bool(getattr(context, "in_parallel_streams", False))
            _log_event(
                "needs_offset_capture_check",
                {
                    "result": result,
                    "cg_mode": cg_mode.name if isinstance(cg_mode, CUDAGraphMode) else str(cg_mode),
                    "start_nt": start_nt,
                    "allow_lazy": allow_lazy,
                    "has_graph": has_graph,
                    "in_parallel_streams": in_parallel_streams,
                },
            )

        return result

    @contextmanager
    def _bind_inplace_parallel_rope_capture_slot(
        self, context: Any, *, parallel_streams: bool
    ):
        if (not parallel_streams
                or getattr(context, "split_inplace_mode", "")
                != "inplace_parallel"):
            yield
            return

        slot_id = int(getattr(context, "cos_sin_slot_id", 0) or 0)
        if slot_id <= 0:
            yield
            return

        import vllm_ascend.ops.rotary_embedding as rotary_embedding

        cos, sin = rotary_embedding.get_cos_and_sin_slice(slot_id=slot_id)
        if cos is None or sin is None:
            raise RuntimeError(
                "Missing rotary cos/sin buffers for inplace parallel "
                f"capture slot {slot_id}")

        previous_cos = rotary_embedding._cos_slice_slots[0]
        previous_sin = rotary_embedding._sin_slice_slots[0]
        rotary_embedding._cos_slice_slots[0] = cos
        rotary_embedding._sin_slice_slots[0] = sin
        from vllm_ascend.inplace_split_debug import is_enabled as _is_enabled
        from vllm_ascend.inplace_split_debug import log_event as _log_event
        from vllm_ascend.inplace_split_debug import tensor_info as _tensor_info
        if _is_enabled():
            _log_event(
                "inplace_parallel_rope_slot_binding",
                {
                    "phase": "bind",
                    "compiled_slot_id": 0,
                    "capture_slot_id": slot_id,
                    "cos": _tensor_info(cos),
                    "sin": _tensor_info(sin),
                },
            )
        try:
            yield
        finally:
            rotary_embedding._cos_slice_slots[0] = previous_cos
            rotary_embedding._sin_slice_slots[0] = previous_sin
            if _is_enabled():
                _log_event(
                    "inplace_parallel_rope_slot_binding",
                    {
                        "phase": "restore",
                        "compiled_slot_id": 0,
                        "capture_slot_id": slot_id,
                    },
                )

    def _run_inplace_serial_offset_capture(
        self,
        metadata: AscendUbatchMetadata,
        split_slice: Any,
        model_kwargs: dict[str, Any],
        *,
        parallel_streams: bool = False,
    ) -> Any:
        from vllm.compilation.monitor import set_cudagraph_capturing_enabled

        from vllm_ascend.inplace_split_debug import (
            is_enabled as _is_enabled,
            log_event as _log_event,
            batch_descriptor_info as _bd_info,
            tensor_info as _tensor_info,
            metadata_tensor_info as _mti,
        )

        split_cfg = self.ascend_config.split_batch_config
        context = metadata.context
        sliced_input_ids = metadata.input_ids
        sliced_positions = metadata.positions
        sliced_intermediate_tensors = metadata.intermediate_tensors
        sliced_inputs_embeds = metadata.inputs_embeds

        batch_descriptor = getattr(context, "batch_descriptor", None)

        warmups = int(getattr(self.compilation_config, 'cudagraph_num_of_warmups', 0) or 0)

        previous_mode = getattr(context, 'cudagraph_runtime_mode', CUDAGraphMode.NONE)
        previous_capturing = bool(getattr(context, 'capturing', False))
        capture_stream = (self.stream_parallel if parallel_streams
                          else self.stream_main)
        is_offset = split_slice.start_num_tokens > 0
        if is_offset:
            forced_backend = getattr(context, 'forced_attention_backend', '')
            if split_cfg.inplace_force_pa_for_offset:
                forced_backend = "pa"
            context.forced_attention_backend = forced_backend
            logger.debug("[inplace_serial] lazy capture: forced_attention_backend=%s, "
                         "inplace_force_pa_for_offset=%s, start_num_tokens=%d",
                          forced_backend, split_cfg.inplace_force_pa_for_offset,
                          split_slice.start_num_tokens)

        if _is_enabled():
            _log_event(
                "inplace_capture_entry",
                {
                    "parallel_streams": parallel_streams,
                    "input_ids": _tensor_info(sliced_input_ids),
                    "positions": _tensor_info(sliced_positions),
                    "num_tokens": int(split_slice.num_tokens),
                    "graph_num_tokens": int(split_slice.graph_num_tokens),
                    "start_num_tokens": int(split_slice.start_num_tokens),
                },
            )
            _log_event(
                "inplace_capture_attn_diag",
                {
                    "parallel_streams": parallel_streams,
                    "metadata": _mti(context.attn_metadata),
                },
            )

        previous_capturing_enabled = True
        set_cudagraph_capturing_enabled(True)

        try:
            with (
                self._bind_inplace_parallel_rope_capture_slot(
                    context, parallel_streams=parallel_streams),
                torch.npu.stream(capture_stream),
            ):

                if _is_enabled():
                    _log_event(
                        "inplace_capture_before_warmup",
                        {
                            "parallel_streams": parallel_streams,
                            "warmups": warmups,
                        },
                    )

                for warmup_idx in range(warmups):
                    context.cudagraph_runtime_mode = CUDAGraphMode.NONE
                    context.capturing = False
                    with override_forward_context(context):
                        _ = self.model(
                            input_ids=sliced_input_ids,
                            positions=sliced_positions,
                            intermediate_tensors=sliced_intermediate_tensors,
                            inputs_embeds=sliced_inputs_embeds,
                            **model_kwargs,
                        )
                    capture_stream.synchronize()
                    if _is_enabled():
                        _log_event(
                            "inplace_capture_warmup_done",
                            {
                                "warmup_idx": warmup_idx,
                                "parallel_streams": parallel_streams,
                            },
                        )

                if _is_enabled():
                    _log_event(
                        "inplace_capture_before_capture",
                        {
                            "parallel_streams": parallel_streams,
                        },
                    )
                context.cudagraph_runtime_mode = CUDAGraphMode.FULL
                context.capturing = False
                with override_forward_context(context):
                    _ = self.model(
                        input_ids=sliced_input_ids,
                        positions=sliced_positions,
                        intermediate_tensors=sliced_intermediate_tensors,
                        inputs_embeds=sliced_inputs_embeds,
                        **model_kwargs,
                    )
                capture_stream.synchronize()
                if _is_enabled():
                    _log_event(
                        "inplace_capture_after_capture",
                        {
                            "parallel_streams": parallel_streams,
                        },
                    )

        finally:
            context.cudagraph_runtime_mode = previous_mode
            context.capturing = previous_capturing
            set_cudagraph_capturing_enabled(previous_capturing_enabled)

        _has_graph = self._has_aclgraph_for_context(context)

        if not _has_graph:
            raise RuntimeError(
                f"Inplace offset graph capture did not create an ACL "
                f"graph entry for {getattr(context, 'batch_descriptor', None)!r}"
            )

        if _is_enabled():
            _log_event(
                "inplace_capture_has_graph_check",
                {
                    "parallel_streams": parallel_streams,
                    "has_graph": True,
                    "in_parallel_streams": parallel_streams,
                    "batch_descriptor": _bd_info(batch_descriptor),
                },
            )

        with torch.npu.stream(capture_stream):
            context.cudagraph_runtime_mode = previous_mode
            context.capturing = False
            with override_forward_context(context):
                replay_result = self.model(
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    intermediate_tensors=sliced_intermediate_tensors,
                    inputs_embeds=sliced_inputs_embeds,
                    **model_kwargs,
                )
                if context.cudagraph_runtime_mode == CUDAGraphMode.FULL:
                    if split_slice.start_num_tokens > 0:
                        self._update_attn_params_for_split_ubatch(
                            context,
                            split_slice.graph_num_tokens,
                            parallel_streams=parallel_streams)
                    else:
                        self._update_attn_params_for_wrapper(
                            context,
                            split_slice.graph_num_tokens)
            capture_stream.synchronize()

        if _is_enabled():
            _log_event(
                "inplace_lazy_capture_complete",
                {
                    "batch_descriptor": _bd_info(batch_descriptor),
                    "num_tokens": int(split_slice.num_tokens),
                    "graph_num_tokens": int(split_slice.graph_num_tokens),
                    "returned": "replay_after_capture",
                    "in_parallel_streams": parallel_streams,
                    "stream": "parallel" if parallel_streams else "main",
                },
            )

        return replay_result

    def _update_attn_params_for_wrapper(
        self,
        forward_context,
        num_tokens,
    ):
        from vllm.forward_context import get_forward_context

        from vllm_ascend.compilation.acl_graph_split_batch import update_attn_params

        forward_context = get_forward_context()
        if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return
        if getattr(forward_context, 'capturing', False):
            return

        use_mla = getattr(self.vllm_config.model_config, 'use_mla', False)
        if use_mla:
            logger.warning_once(
                "[inplace_split] MLA attention backend is not yet supported "
                "for inplace split attn param update; skipping.")
            return

        update_stream = getattr(self._runner, 'update_stream_main', None) or getattr(self._runner, 'update_stream', None)
        if update_stream is None:
            update_stream = torch.npu.current_stream()
        update_attn_params(
            update_stream,
            forward_context,
            num_tokens,
            self.vllm_config,
        )

    def _update_attn_params_for_split_ubatch(
        self,
        forward_context,
        num_tokens,
        parallel_streams: bool = False,
    ):
        from vllm_ascend.compilation.acl_graph_split_batch import update_attn_params_split

        if forward_context.cudagraph_runtime_mode != CUDAGraphMode.FULL:
            return
        if getattr(forward_context, 'capturing', False):
            return

        use_mla = getattr(self.vllm_config.model_config, 'use_mla', False)
        if use_mla:
            logger.warning_once(
                "[inplace_split] MLA attention backend is not yet supported "
                "for inplace split attn param update; skipping.")
            return

        if parallel_streams:
            update_stream = getattr(self._runner, 'update_stream_parallel', None)
            if update_stream is None:
                update_stream = getattr(self._runner, 'stream_parallel', None)
            if update_stream is None:
                update_stream = torch.npu.current_stream()
        else:
            update_stream = getattr(self._runner, 'update_stream_main', None) or getattr(self._runner, 'update_stream', None)
            if update_stream is None:
                update_stream = torch.npu.current_stream()
        update_attn_params_split(
            update_stream,
            forward_context,
            num_tokens,
            self.vllm_config,
            in_parallel_streams=parallel_streams,
        )

    def _make_split_batch_metadata_inplace_parallel(
        self,
        split_ubatch_slices,
        split_batch_slices,
        attn_metadata,
        input_ids,
        positions,
        inputs_embeds,
        intermediate_tensors,
        batch_descriptor,
        aclgraph_runtime_mode,
        inplace_attention_backend,
    ):
        from vllm.forward_context import get_forward_context

        from vllm_ascend.ascend_forward_context import create_ascend_forward_context

        cur_forward_context = get_forward_context()
        dp_metadata = getattr(cur_forward_context, 'dp_metadata', None)
        context_ubatch_slices = self._context_ubatch_slices_for_inplace(split_batch_slices)
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        allow_lazy = bool(split_cfg is not None and getattr(split_cfg, "enable_inplace_lazy_capture", True))
        force_pa_for_offset = bool(split_cfg is not None and getattr(split_cfg, "inplace_force_pa_for_offset", False))

        from vllm_ascend.inplace_split_debug import (
            is_enabled as _split_debug_enabled,
            log_event as _log_event,
            batch_descriptor_info as _bd_info,
            tensor_view_info as _tvi,
            metadata_tensor_info as _mti,
        )

        def _stream_for_split(split_idx: int):
            return self.stream_parallel if split_idx > 0 else self.stream_main

        prepared_split_inputs = self._prepare_inplace_split_inputs_for_execution(
            split_batch_slices,
            input_ids,
            positions,
            inputs_embeds,
            intermediate_tensors,
            stream_for_split=_stream_for_split,
        )

        _cached_dual_stream_metadata = getattr(
            self._runner, "_dual_stream_attention_metadata", None)
        _cached_dual_stream_slices = getattr(
            self._runner, "_dual_stream_attention_slices", None)
        _cached_dual_stream_plan = getattr(
            self._runner, "_dual_stream_attention_plan", None)
        _cached_dual_stream_secondary_mode = None
        if _cached_dual_stream_metadata is not None:
            _dsa_cfg = dual_stream_attention_config(
                getattr(self.ascend_config, "split_batch_config", None))
            _cached_dual_stream_secondary_mode = getattr(
                _dsa_cfg, "secondary_stream_mode", "dedicated_pair")

        ubatch_metadata = []
        for i, split_slice in enumerate(split_batch_slices):
            in_parallel_streams = i > 0
            split_attention_backend = inplace_attention_backend
            if force_pa_for_offset and split_slice.start_num_tokens > 0:
                split_attention_backend = "pa"
            capture_metadata_mode = (
                "template"
                if split_slice.start_num_tokens > 0
                and split_attention_backend == "fia" else ""
            )
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            ubatch_cudagraph_mode, ubatch_batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    num_tokens=split_slice.graph_num_tokens,
                    uniform_decode=True,
                    has_lora=batch_descriptor.has_lora,
                    start_num_tokens=split_slice.start_num_tokens,
                    allow_inplace_lazy_key=(allow_lazy and split_slice.start_num_tokens > 0),
                    graph_variant=("inplace_parallel" if split_slice.start_num_tokens > 0 else ""),
                    attention_backend=(split_attention_backend if split_slice.start_num_tokens > 0 else ""),
                    capture_metadata_mode=capture_metadata_mode,
                ))

            allow_inplace_lazy_capture = bool(
                allow_lazy and split_slice.start_num_tokens > 0
                and ubatch_cudagraph_mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE)
                and getattr(ubatch_batch_descriptor, "graph_variant", "")
                == "inplace_parallel"
                and getattr(ubatch_batch_descriptor, "attention_backend", "")
                in ("fia", "pa"))

            validate_inplace_ptrs = bool(
                split_cfg is not None
                and getattr(split_cfg, "inplace_validate_metadata_ptrs", False)
                and split_slice.start_num_tokens > 0)

            if _split_debug_enabled():
                _log_event(
                    "split_descriptor",
                    {
                        "idx": i,
                        "execution": "inplace_parallel",
                        "stream": ("parallel" if in_parallel_streams
                                   else "main"),
                        "dispatch_num_tokens":
                        int(split_slice.graph_num_tokens),
                        "actual_num_tokens": int(split_slice.num_tokens),
                        "runtime_mode": (
                            ubatch_cudagraph_mode.name
                            if isinstance(ubatch_cudagraph_mode,
                                          CUDAGraphMode) else
                            str(ubatch_cudagraph_mode)),
                        "batch_descriptor": _bd_info(ubatch_batch_descriptor),
                        "in_parallel_streams": in_parallel_streams,
                        "graph_params_pool": ("parallel"
                                              if in_parallel_streams
                                              else "main"),
                        "graph_entry_pool": ("parallel"
                                             if in_parallel_streams
                                             else "main"),
                        "allow_inplace_lazy_capture":
                        allow_inplace_lazy_capture,
                        "validate_inplace_ptrs": validate_inplace_ptrs,
                        "forced_attention_backend":
                        split_attention_backend,
                        "force_pa_for_offset":
                        force_pa_for_offset,
                    },
                )

            if (split_slice.start_num_tokens > 0
                    and split_attention_backend == "fia"
                    and ubatch_attn_metadata is not None
                    and ubatch_batch_descriptor.capture_metadata_mode == "template"):
                from vllm_ascend.compilation.acl_graph_split_batch import template_fia_seq_lens_list
                template_fia_seq_lens_list(ubatch_attn_metadata, self.block_size)

            ctx_stream = self.stream_parallel if in_parallel_streams else self.stream_main
            with torch.npu.stream(ctx_stream):
                split_forward_context = create_ascend_forward_context(
                    cur_forward_context,
                    attn_metadata=ubatch_attn_metadata,
                    vllm_config=self.vllm_config,
                    dp_metadata=dp_metadata,
                    ubatch_slices=context_ubatch_slices,
                    batch_descriptor=ubatch_batch_descriptor,
                    cudagraph_runtime_mode=ubatch_cudagraph_mode,
                    ubatch_num=i,
                    positions=prepared_split_inputs[i]["positions"],
                    in_parallel_streams=in_parallel_streams,
                    cos_sin_slot_id=i,
                )
            setattr(split_forward_context, "split_inplace_mode",
                    "inplace_parallel")
            setattr(split_forward_context, "forced_attention_backend",
                    split_attention_backend)
            setattr(split_forward_context, "allow_inplace_lazy_capture",
                    allow_inplace_lazy_capture)
            setattr(split_forward_context, "split_actual_num_tokens",
                    int(split_slice.num_tokens))
            setattr(split_forward_context, "split_graph_num_tokens",
                    int(split_slice.graph_num_tokens))
            setattr(split_forward_context, "validate_inplace_metadata_ptrs",
                    validate_inplace_ptrs)
            setattr(split_forward_context, "validate_inplace_input_ptrs",
                    validate_inplace_ptrs)

            if _cached_dual_stream_metadata is not None:
                split_forward_context.dual_stream_attention_metadata = (
                    _cached_dual_stream_metadata)
                split_forward_context.dual_stream_attention_slices = (
                    _cached_dual_stream_slices)
                split_forward_context.dual_stream_attention_plan = (
                    _cached_dual_stream_plan)
                split_forward_context.dual_stream_attention_secondary_stream_mode = (
                    _cached_dual_stream_secondary_mode)

            prepared_inputs = prepared_split_inputs[i]
            tokens_slice = prepared_inputs["tokens_slice"]
            sliced_input_ids, sliced_positions, sliced_inputs_embeds, sliced_intermediate_tensors = (
                slice_split_batch_inputs(
                    tokens_slice, prepared_inputs["input_ids"],
                    prepared_inputs["positions"],
                    prepared_inputs["inputs_embeds"],
                    intermediate_tensors,
                    vllm_config=self.vllm_config,
                )
            )

            if _split_debug_enabled():
                _log_event(
                    "inplace_parallel_execution",
                    {
                        "idx": i,
                        "stream": ("parallel" if in_parallel_streams
                                   else "main"),
                        "buffer_source": "original_offset_view",
                        "graph_params_pool": ("parallel"
                                              if in_parallel_streams
                                              else "main"),
                        "graph_entry_pool": ("parallel"
                                             if in_parallel_streams
                                             else "main"),
                        "token_start": int(split_slice.token_slice.start),
                        "token_stop": int(split_slice.token_slice.stop),
                        "graph_token_stop": int(tokens_slice.stop),
                        "start_num_tokens":
                        int(split_slice.start_num_tokens),
                        "num_tokens": int(split_slice.num_tokens),
                        "graph_num_tokens": int(split_slice.graph_num_tokens),
                        "input_ids": _tvi(sliced_input_ids),
                        "positions": _tvi(sliced_positions),
                        "inputs_embeds": _tvi(sliced_inputs_embeds),
                        "metadata": _mti(
                            split_forward_context.attn_metadata),
                        "batch_descriptor": _bd_info(
                            split_forward_context.batch_descriptor),
                    },
                )

            ubatch_metadata.append(
                AscendUbatchMetadata(
                    context=split_forward_context,
                    input_ids=sliced_input_ids,
                    positions=sliced_positions,
                    inputs_embeds=sliced_inputs_embeds,
                    intermediate_tensors=sliced_intermediate_tensors,
                    num_tokens=split_slice.graph_num_tokens,
                )
            )
        return ubatch_metadata

    def run_inplace_parallel(
        self,
        split_plan: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any,
        inputs_embeds: torch.Tensor | None,
        attn_metadata: Any,
        cudagraph_mode: CUDAGraphMode,
        batch_descriptor: BatchDescriptor,
        inplace_attention_backend: str | None = None,
        **model_kwargs,
    ) -> torch.Tensor:

        from vllm_ascend.inplace_split_debug import is_enabled as _split_debug_enabled
        from vllm_ascend.inplace_split_debug import log_event as _split_debug_log_event

        split_cfg = getattr(self.ascend_config, "split_batch_config", None)

        split_slices = split_plan.split_slices
        original_forward_context = get_forward_context()
        num_splits = len(split_slices)

        if inplace_attention_backend is None:
            from vllm_ascend.worker.inplace_split_utils import (
                inplace_split_preserves_attention_backend,
                select_inplace_attention_backend,
            )
            inplace_attention_backend = select_inplace_attention_backend(
                split_plan, lambda shape: using_paged_attention(shape, self.vllm_config))
            if not inplace_split_preserves_attention_backend(
                split_plan, lambda shape: using_paged_attention(shape, self.vllm_config)):
                logger.debug(
                    "Inplace parallel split changes attention backend; using %s for both splits.",
                    inplace_attention_backend)

        if self.stream_main is None:
            self._runner.stream_main = torch.npu.current_stream()
        if self.stream_parallel is None:
            self._runner.stream_parallel = torch.npu.Stream(device=self.device)

        from vllm.v1.worker.ubatch_utils import UBatchSlice
        split_ubatch_slices = [
            UBatchSlice(s.request_slice, s.token_slice)
            for s in split_slices
        ]

        ubatch_metadata = self._make_split_batch_metadata_inplace_parallel(
            split_ubatch_slices=split_ubatch_slices,
            split_batch_slices=split_slices,
            attn_metadata=attn_metadata,
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            intermediate_tensors=intermediate_tensors,
            batch_descriptor=batch_descriptor,
            aclgraph_runtime_mode=cudagraph_mode,
            inplace_attention_backend=inplace_attention_backend,
        )

        results: list[Any | None] = [None] * num_splits
        split_errors: list[tuple[int, Exception]] = []
        split_error_lock = threading.Lock()

        split_debug_active = _split_debug_enabled()

        def _run_inplace_parallel_worker(slice_idx: int) -> None:
            try:
                split_slice = split_slices[slice_idx]
                metadata = ubatch_metadata[slice_idx]
                parallel_streams = slice_idx > 0
                target_stream = self.stream_parallel if parallel_streams else self.stream_main

                needs_offset = self.needs_inplace_serial_offset_capture(metadata)
                if split_debug_active:
                    _split_debug_log_event(
                        "parallel_worker_start",
                        {
                            "slice_idx": slice_idx,
                            "path": "offset_capture" if needs_offset
                            else "normal_replay",
                            "start_num_tokens":
                            int(split_slice.start_num_tokens),
                            "graph_num_tokens":
                            int(split_slice.graph_num_tokens),
                            "parallel_streams": parallel_streams,
                            **({"has_aclgraph": True} if not needs_offset
                               else {}),
                        },
                    )

                with torch.inference_mode():
                    if needs_offset:
                        split_result = self._run_inplace_serial_offset_capture(
                            metadata,
                            split_slice,
                            model_kwargs,
                            parallel_streams=parallel_streams,
                        )
                    else:

                        if (int(getattr(
                                metadata.context.batch_descriptor,
                                "start_num_tokens", 0) or 0) > 0
                                and metadata.context.cudagraph_runtime_mode
                                == CUDAGraphMode.FULL
                                and not self._has_aclgraph_for_context(
                                    metadata.context)):
                            raise RuntimeError(
                                "Missing inplace parallel offset ACL graph "
                                "before normal replay path: "
                                f"{metadata.context.batch_descriptor!r}")
                        with torch.npu.stream(target_stream):

                            with override_forward_context(metadata.context):
                                if (metadata.context.cudagraph_runtime_mode
                                        == CUDAGraphMode.FULL):
                                    if split_slice.start_num_tokens > 0:
                                        self._update_attn_params_for_split_ubatch(
                                            metadata.context,
                                            split_slice.graph_num_tokens,
                                            parallel_streams=parallel_streams)
                                    else:
                                        self._update_attn_params_for_wrapper(
                                            metadata.context,
                                            split_slice.graph_num_tokens)
                                split_result = self.model(
                                    input_ids=metadata.input_ids,
                                    positions=metadata.positions,
                                    inputs_embeds=metadata.inputs_embeds,
                                    intermediate_tensors=metadata.intermediate_tensors,
                                    **model_kwargs,
                                )

                    with torch.npu.stream(target_stream):
                        results[slice_idx] = clone_split_output(
                            trim_split_output(split_result,
                                              split_slice.num_tokens))
            except Exception as e:
                with split_error_lock:
                    split_errors.append((slice_idx, e))

        split_workers: list[threading.Thread] = []
        for slice_idx in range(num_splits):
            worker = threading.Thread(
                target=_run_inplace_parallel_worker,
                args=(slice_idx,),
                name=f"inplace-parallel-replay-{slice_idx}",
            )
            split_workers.append(worker)
            worker.start()

        for worker in split_workers:
            worker.join()

        if split_errors:
            split_errors.sort(key=lambda item: item[0])
            failed_slice_idx, first_error = split_errors[0]
            raise RuntimeError(
                "inplace parallel replay worker failed at "
                f"slice_idx={failed_slice_idx}: {first_error}") from first_error

        self.stream_main.synchronize()
        if num_splits > 1:
            self.stream_parallel.synchronize()

        with override_forward_context(original_forward_context):
            return merge_split_outputs([r for r in results if r is not None])

    def capture_parallel_stream_graphs(self) -> None:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        runner = self._runner
        if not isinstance(runner.model, ACLGraphWrapper):
            return
        if not hasattr(runner, 'cudagraph_batch_sizes_parallel') or runner.cudagraph_batch_sizes_parallel is None:
            return
        for batch_size in runner.cudagraph_batch_sizes_parallel:
            runner._dummy_run(
                batch_size,
                in_parallel_streams=True,
            )