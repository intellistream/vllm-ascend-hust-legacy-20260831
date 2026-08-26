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

import os
import threading
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import (
    BatchDescriptor,
    CUDAGraphRuntimeMetadata,
    get_forward_context,
    override_forward_context,
)
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
from vllm_ascend.worker.inplace_split_worker_pool import SplitReplayWorkerPool
from vllm_ascend.worker.inplace_split_utils import SplitBatchSlice

if TYPE_CHECKING:
    from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


_INPLACE_PARALLEL_MERGE_SYNC_POLICY = os.environ.get(
    "VLLM_ASCEND_INPLACE_PARALLEL_MERGE_SYNC_POLICY", "event_wait").strip(
    ).lower()
if _INPLACE_PARALLEL_MERGE_SYNC_POLICY not in ("event_wait", "host_sync"):
    logger.warning(
        "Unknown VLLM_ASCEND_INPLACE_PARALLEL_MERGE_SYNC_POLICY=%r; "
        "falling back to event_wait",
        _INPLACE_PARALLEL_MERGE_SYNC_POLICY,
    )
    _INPLACE_PARALLEL_MERGE_SYNC_POLICY = "event_wait"


def _parse_inplace_parallel_split_output_mode() -> str:
    mode = os.environ.get("VLLM_ASCEND_INPLACE_PARALLEL_SPLIT_OUTPUT_MODE")
    if mode is None:
        legacy_clone = os.environ.get(
            "VLLM_ASCEND_INPLACE_PARALLEL_CLONE_SPLIT_OUTPUTS")
        if legacy_clone is None:
            return "auto"
        return ("direct" if legacy_clone in ("0", "false", "False") else
                "clone")
    mode = mode.strip().lower()
    if mode in ("auto", "clone", "direct"):
        return mode
    if mode in ("0", "false"):
        return "direct"
    if mode in ("1", "true"):
        return "clone"
    logger.warning(
        "Unknown VLLM_ASCEND_INPLACE_PARALLEL_SPLIT_OUTPUT_MODE=%r; "
        "falling back to auto",
        mode,
    )
    return "auto"


_INPLACE_PARALLEL_SPLIT_OUTPUT_MODE = (
    _parse_inplace_parallel_split_output_mode())


def _parse_stream_limit_pair(value: str, env_name: str) -> tuple[int, int] | None:
    parts = [part.strip() for part in value.split(",")]
    if len(parts) != 2:
        logger.warning(
            "Ignoring %s=%r: expected cube,vector",
            env_name,
            value,
        )
        return None
    try:
        return int(parts[0]), int(parts[1])
    except ValueError:
        logger.warning(
            "Ignoring %s=%r: cube/vector must be integers",
            env_name,
            value,
        )
        return None


def _parse_stream_limit_spec(
        env_name: str) -> tuple[tuple[int, int], tuple[int, int]] | None:
    raw = os.environ.get(env_name, "").strip()
    if not raw:
        return None
    parts = [part.strip() for part in raw.split(";")]
    if len(parts) == 1:
        limit = _parse_stream_limit_pair(parts[0], env_name)
        if limit is None:
            return None
        return limit, limit
    if len(parts) == 2:
        main_limit = _parse_stream_limit_pair(parts[0], env_name)
        parallel_limit = _parse_stream_limit_pair(parts[1], env_name)
        if main_limit is None or parallel_limit is None:
            return None
        return main_limit, parallel_limit
    logger.warning(
        "Ignoring %s=%r: expected cube,vector or "
        "main_cube,main_vector;parallel_cube,parallel_vector",
        env_name,
        raw,
    )
    return None


_INPLACE_PARALLEL_REPLAY_STREAM_LIMITS = _parse_stream_limit_spec(
    "VLLM_ASCEND_INPLACE_PARALLEL_REPLAY_STREAM_LIMITS")
_INPLACE_PARALLEL_UPDATE_STREAM_LIMITS = _parse_stream_limit_spec(
    "VLLM_ASCEND_INPLACE_PARALLEL_UPDATE_STREAM_LIMITS")

_INPLACE_PARALLEL_REUSE_SPLIT0_COS_SIN = os.environ.get(
    "VLLM_ASCEND_INPLACE_PARALLEL_REUSE_SPLIT0_COS_SIN", "1") not in (
        "0", "false", "False")


def _inplace_parallel_clone_split_outputs(
        *,
        allow_auto_direct_outputs: bool,
        split_cfg: Any,
        split_batch_slices: list[Any],
        aclgraph_runtime_mode: CUDAGraphMode,
        merge_sync_policy: str) -> bool:
    if _INPLACE_PARALLEL_SPLIT_OUTPUT_MODE == "clone":
        return True
    if _INPLACE_PARALLEL_SPLIT_OUTPUT_MODE == "direct":
        return False
    if not allow_auto_direct_outputs:
        return True
    if aclgraph_runtime_mode != CUDAGraphMode.FULL:
        return True
    if merge_sync_policy != "event_wait":
        return True
    if split_cfg is None:
        return True
    if getattr(split_cfg, "mode", "") != "inplace_parallel":
        return True
    if not bool(getattr(split_cfg, "enable_parallel_streams", False)):
        return True
    return len(split_batch_slices) != 2


def _record_stream_tree_for_npugraph_ex(value: Any, stream: Any) -> None:
    if isinstance(value, torch.Tensor):
        try:
            if value.device.type == "npu":
                value.record_stream(stream)
        except Exception:
            return
        return
    if isinstance(value, IntermediateTensors):
        for tensor in value.tensors.values():
            _record_stream_tree_for_npugraph_ex(tensor, stream)
        return
    if isinstance(value, dict):
        for child in value.values():
            _record_stream_tree_for_npugraph_ex(child, stream)
        return
    if isinstance(value, (list, tuple)):
        for child in value:
            _record_stream_tree_for_npugraph_ex(child, stream)


class InplaceSplitRunner:
    """Orchestrates inplace split-batch execution.

    Holds a reference to NPUModelRunner via composition rather than inheritance.
    """

    def __init__(self, runner: "NPUModelRunner"):
        self._runner = runner
        self._inplace_parallel_output_release_events: list[
            torch.npu.Event | None] = []
        self._replay_worker_pool: SplitReplayWorkerPool | None = None

    def _get_replay_worker_pool(self) -> SplitReplayWorkerPool:
        """Lazy persistent replay workers (created on the engine device)."""
        if self._replay_worker_pool is None:
            self._replay_worker_pool = SplitReplayWorkerPool(num_workers=2)
        return self._replay_worker_pool

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

    def _record_replay_stream_event(
            self, stream: torch.npu.Stream) -> torch.npu.Event:
        event = torch.npu.Event()
        event.record(stream)
        return event

    def _wait_replay_stream_events(
            self, stream: torch.npu.Stream,
            events: list[torch.npu.Event | None]) -> None:
        for event in events:
            if event is not None:
                stream.wait_event(event)

    def _wait_inplace_parallel_output_release_events(
            self, streams: list[torch.npu.Stream]) -> int:
        events = getattr(self, "_inplace_parallel_output_release_events", [])
        if not events:
            return 0
        self._inplace_parallel_output_release_events = []
        for stream in streams:
            self._wait_replay_stream_events(stream, events)
        return len(events)

    def _record_inplace_parallel_output_release_event(
            self) -> torch.npu.Event:
        event = self._record_replay_stream_event(self.stream_main)
        self._inplace_parallel_output_release_events = [event]
        return event

    def _set_stream_limit(self, stream: torch.npu.Stream,
                          limit: tuple[int, int], label: str) -> None:
        cube_num, vector_num = limit
        try:
            torch.npu.set_stream_limit(stream,
                                       cube_num=cube_num,
                                       vector_num=vector_num)
            logger.info_once(
                "Applied inplace_parallel stream limit for %s: "
                "cube_num=%s vector_num=%s",
                label,
                cube_num,
                vector_num,
            )
        except Exception as exc:
            logger.warning(
                "Failed to apply inplace_parallel stream limit for %s: %s",
                label,
                exc,
            )

    def _apply_inplace_parallel_replay_stream_limits(self) -> None:
        limits = _INPLACE_PARALLEL_REPLAY_STREAM_LIMITS
        if limits is None:
            return
        self._set_stream_limit(self.stream_main, limits[0], "replay_main")
        self._set_stream_limit(self.stream_parallel, limits[1],
                               "replay_parallel")


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
        if split_batch_config.mode == "dual_pad":
            return self._should_split_dual_pad(
                split_batch_config=split_batch_config,
                cudagraph_mode=cudagraph_mode,
                num_scheduled_tokens_np=num_scheduled_tokens_np,
                num_reqs=num_reqs,
                total_num_tokens=total_num_tokens,
                has_lora=has_lora,
                is_mla=is_mla,
                is_mrope=is_mrope,
                spec_decode_enabled=spec_decode_enabled,
                cudagraph_capture_sizes=cudagraph_capture_sizes,
            )
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
            force_split=split_batch_config.force_split,
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

    def _should_split_dual_pad(
        self,
        *,
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
        """DUAL_PAD split decision: largest main graph hit + padded remainder.

        Uses the reference implementation's decision logic (no offset graphs,
        no lazy capture): split only when the padding saved exceeds
        ``cudagraph_split_pad_threshold`` (unless ``force_split``).
        """
        from vllm_ascend.inplace_split_debug import (
            is_enabled as _debug_enabled,
            log_event,
        )
        from vllm_ascend.worker.dual_pad_utils import (
            create_dual_pad_split_batch_slices,
            dual_pad_precheck_reason,
        )

        precheck_reason = dual_pad_precheck_reason(
            split_batch_config=split_batch_config,
            cudagraph_mode=cudagraph_mode,
            num_reqs=num_reqs,
            has_lora=has_lora,
            is_mla=is_mla,
            is_mrope=is_mrope,
            spec_decode_enabled=spec_decode_enabled,
        )
        if precheck_reason is not None:
            if _debug_enabled():
                log_event("inplace_split_precheck_failed",
                          {"reason": precheck_reason})
            return None, precheck_reason

        split_plan, reason = create_dual_pad_split_batch_slices(
            num_scheduled_tokens_per_request=num_scheduled_tokens_np,
            total_num_tokens=total_num_tokens,
            cudagraph_capture_sizes=cudagraph_capture_sizes,
            parallel_capture_sizes=getattr(
                split_batch_config, "parallel_capture_sizes", None),
            cudagraph_split_pad_threshold=getattr(
                split_batch_config, "cudagraph_split_pad_threshold", 0),
            force_split=bool(getattr(split_batch_config, "force_split", False)),
        )

        if _debug_enabled():
            log_event("inplace_split_planning_result", {
                "reason": reason,
                "has_plan": split_plan is not None,
            })

        logger.info(
            "DUAL_PAD split decision: has_plan=%s, reason=%s, total_tokens=%d",
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
        _rm = getattr(batch_descriptor, "runtime_metadata", None)
        start_nt = int(_rm.token_offset) if _rm is not None else 0
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

            ubatch_runtime_metadata = None
            if split_slice.start_num_tokens > 0:
                ubatch_runtime_metadata = CUDAGraphRuntimeMetadata(
                    token_offset=split_slice.start_num_tokens,
                    variant="inplace_parallel",
                    backend_tag=split_attention_backend,
                    metadata_mode=capture_metadata_mode,
                )
            ubatch_cudagraph_mode, ubatch_batch_descriptor = (
                self.cudagraph_dispatcher.dispatch(
                    num_tokens=split_slice.graph_num_tokens,
                    uniform_decode=True,
                    has_lora=batch_descriptor.has_lora,
                    runtime_metadata=ubatch_runtime_metadata,
                    allow_runtime_key_registration=(
                        allow_lazy and split_slice.start_num_tokens > 0),
                ))

            ubatch_rm = getattr(ubatch_batch_descriptor, "runtime_metadata", None)
            allow_inplace_lazy_capture = bool(
                allow_lazy and split_slice.start_num_tokens > 0
                and ubatch_cudagraph_mode in (CUDAGraphMode.FULL, CUDAGraphMode.PIECEWISE)
                and ubatch_rm is not None
                and ubatch_rm.variant == "inplace_parallel"
                and ubatch_rm.backend_tag in ("fia", "pa"))

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
                    and getattr(ubatch_batch_descriptor, "runtime_metadata", None) is not None
                    and getattr(ubatch_batch_descriptor, "runtime_metadata", None).metadata_mode == "template"
                    and split_slice.graph_num_tokens > split_slice.num_tokens):
                # Template the dummy padding request's KV length only. For a
                # zero-padding split (graph_num_tokens == num_tokens) the last
                # seq_lens entry belongs to a real request; templating it
                # would pin that request's KV length to the block-size
                # template value and corrupt its attention output.
                from vllm_ascend.compilation.acl_graph_split_batch import template_fia_seq_lens_list
                template_fia_seq_lens_list(ubatch_attn_metadata, self.block_size)

            ctx_stream = self.stream_parallel if in_parallel_streams else self.stream_main
            clone_cos_sin = not (
                _INPLACE_PARALLEL_REUSE_SPLIT0_COS_SIN
                and aclgraph_runtime_mode == CUDAGraphMode.FULL
                and split_cfg is not None
                and getattr(split_cfg, "mode", "") == "inplace_parallel"
                and bool(getattr(split_cfg, "enable_parallel_streams", False))
                and len(split_batch_slices) == 2
            )
            reuse_cos_sin = (
                _INPLACE_PARALLEL_REUSE_SPLIT0_COS_SIN
                and i == 0
                and aclgraph_runtime_mode == CUDAGraphMode.FULL
                and split_cfg is not None
                and getattr(split_cfg, "mode", "") == "inplace_parallel"
                and bool(getattr(split_cfg, "enable_parallel_streams", False))
                and len(split_batch_slices) == 2
            )
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
                    reuse_existing_cos_sin=reuse_cos_sin,
                    clone_cos_sin=clone_cos_sin,
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

        self._apply_inplace_parallel_replay_stream_limits()

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

        clone_split_outputs = _inplace_parallel_clone_split_outputs(
            allow_auto_direct_outputs=True,
            split_cfg=split_cfg,
            split_batch_slices=split_slices,
            aclgraph_runtime_mode=cudagraph_mode,
            merge_sync_policy=_INPLACE_PARALLEL_MERGE_SYNC_POLICY,
        )

        results: list[Any | None] = [None] * num_splits
        split_done_events: list[torch.npu.Event | None] = [None] * num_splits
        split_errors: list[tuple[int, Exception]] = []
        split_error_lock = threading.Lock()

        split_debug_active = _split_debug_enabled()

        self._wait_inplace_parallel_output_release_events(
            (self.stream_main, self.stream_parallel))

        def _finish_inplace_parallel_split_result(
                slice_idx: int,
                split_result: Any,
                *,
                parallel_streams: bool,
                target_stream: torch.npu.Stream) -> None:
            split_slice = split_slices[slice_idx]
            with torch.npu.stream(target_stream):
                trimmed_result = trim_split_output(
                    split_result, split_slice.num_tokens)
                if clone_split_outputs:
                    trimmed_result = clone_split_output(trimmed_result)
                results[slice_idx] = trimmed_result
                if parallel_streams:
                    split_done_events[slice_idx] = (
                        self._record_replay_stream_event(target_stream))

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
                                split_result = self.model(
                                    input_ids=metadata.input_ids,
                                    positions=metadata.positions,
                                    inputs_embeds=metadata.inputs_embeds,
                                    intermediate_tensors=metadata.intermediate_tensors,
                                    **model_kwargs,
                                )
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

                    _finish_inplace_parallel_split_result(
                        slice_idx,
                        split_result,
                        parallel_streams=parallel_streams,
                        target_stream=target_stream)
            except Exception as e:
                with split_error_lock:
                    split_errors.append((slice_idx, e))

        self._get_replay_worker_pool().dispatch([
            (lambda i=slice_idx: _run_inplace_parallel_worker(i))
            for slice_idx in range(num_splits)
        ])

        if split_errors:
            split_errors.sort(key=lambda item: item[0])
            failed_slice_idx, first_error = split_errors[0]
            raise RuntimeError(
                "inplace parallel replay worker failed at "
                f"slice_idx={failed_slice_idx}: {first_error}") from first_error

        merged_results: list[Any] = [
            result for result in results if result is not None
        ]
        if len(merged_results) != num_splits:
            raise RuntimeError(
                "Missing inplace parallel split result: "
                f"expected={num_splits}, got={len(merged_results)}")

        if _INPLACE_PARALLEL_MERGE_SYNC_POLICY == "host_sync":
            self.stream_main.synchronize()
            if num_splits > 1:
                self.stream_parallel.synchronize()
            with override_forward_context(original_forward_context):
                result = merge_split_outputs(merged_results)
                if not clone_split_outputs:
                    self._record_inplace_parallel_output_release_event()
            return result

        with torch.npu.stream(self.stream_main):
            self._wait_replay_stream_events(self.stream_main,
                                            split_done_events)
            for merged_result in merged_results:
                _record_stream_tree_for_npugraph_ex(merged_result,
                                                    self.stream_main)
            with override_forward_context(original_forward_context):
                result = merge_split_outputs(merged_results)
                if not clone_split_outputs:
                    self._record_inplace_parallel_output_release_event()
        return result

    def _extract_second_split_slot_mapping(
        self,
        split_batch_slices: Any,
        attn_metadata: Any,
    ) -> torch.Tensor | None:
        """Return split-1's slot_mapping tensor from its attention metadata.

        All layers share the common-metadata slot_mapping view, so any layer
        object provides it.  Returns None when unavailable (the caller then
        skips the parallel slot_mapping copy).
        """
        if len(split_batch_slices) < 2 or attn_metadata is None:
            return None
        if not (isinstance(attn_metadata, list) and len(attn_metadata) > 1):
            return None
        second_meta = attn_metadata[1]
        if isinstance(second_meta, dict):
            second_meta = next(iter(second_meta.values()), None)
        for candidate in (second_meta,
                          getattr(second_meta, "decode", None),
                          getattr(second_meta, "decode_meta", None)):
            slot_mapping = getattr(candidate, "slot_mapping", None)
            if isinstance(slot_mapping, torch.Tensor):
                return slot_mapping
        return None

    def _prepare_dual_pad_split_inputs(
        self,
        split_batch_slices: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        second_slot_mapping: torch.Tensor | None = None,
    ) -> None:
        """Dual-pad input pre-write.

        Copies the second split's input_ids / positions / inputs_embeds into
        the dedicated parallel-stream buffers and zero-pads each split tail to
        its own padded graph size (mirrors the reference implementation's
        pre-write in model_runner_v3.py L1174-1220).

        Also copies the second split's slot_mapping into the dedicated
        parallel slot_mapping buffer that the parallel-pool graphs' KV-write
        op (npu_scatter_pa_kv_cache) was captured against.  Without this copy
        split-1 would scatter its K/V into split-0's cache slots, corrupting
        split-0's KV cache.
        """
        if len(split_batch_slices) < 2:
            return
        runner = self._runner
        second = split_batch_slices[1]
        second_start = int(second.token_slice.start)
        second_stop = int(second.token_slice.stop)
        second_num_tokens = second_stop - second_start
        second_padded = int(second.graph_num_tokens)

        # The source buffers (input_ids / positions / slot_mapping) were
        # produced on stream_main.  Order the parallel-stream copies after
        # them so split-1 never reads a stale value (the reference pre-write
        # runs on the default stream and so avoids this hazard entirely).
        self.stream_parallel.wait_stream(self.stream_main)

        with torch.npu.stream(self.stream_parallel):
            parallel_slot_mapping = getattr(
                runner, "slot_mapping_parallel_streams", None)
            if (parallel_slot_mapping is not None
                    and second_slot_mapping is not None):
                parallel_slot_mapping[:second_num_tokens].copy_(
                    second_slot_mapping[:second_num_tokens])
                if second_padded > second_num_tokens:
                    # PADDING_SLOT_ID (-1): the graph's KV-write op processes
                    # the full padded size; padded lanes must not write KV.
                    parallel_slot_mapping[
                        second_num_tokens:second_padded].fill_(-1)

            parallel_positions = getattr(
                runner, "positions_parallel_streams", None)
            if parallel_positions is not None and positions is not None:
                if positions.ndim == 2:
                    parallel_positions[:, :second_num_tokens].copy_(
                        positions[:, second_start:second_stop])
                    if second_padded > second_num_tokens:
                        parallel_positions[:, second_num_tokens:second_padded].fill_(0)
                else:
                    parallel_positions[:second_num_tokens].copy_(
                        positions[second_start:second_stop])
                    if second_padded > second_num_tokens:
                        parallel_positions[second_num_tokens:second_padded].fill_(0)

            parallel_input_ids = getattr(
                runner, "input_ids_parallel_streams", None)
            if parallel_input_ids is not None and input_ids is not None:
                parallel_input_ids[:second_num_tokens].copy_(
                    input_ids[second_start:second_stop])
                if second_padded > second_num_tokens:
                    parallel_input_ids[second_num_tokens:second_padded].fill_(0)

            parallel_inputs_embeds = getattr(
                runner, "inputs_embeds_parallel_streams", None)
            if (parallel_inputs_embeds is not None
                    and inputs_embeds is not None):
                parallel_inputs_embeds[:second_num_tokens].copy_(
                    inputs_embeds[second_start:second_stop], non_blocking=True)
                if second_padded > second_num_tokens:
                    parallel_inputs_embeds[
                        second_num_tokens:second_padded].fill_(0)

        # Zero-pad the main-stream (split 0) tail in the original buffers.
        first = split_batch_slices[0]
        first_num_tokens = int(first.num_tokens)
        first_padded = int(first.graph_num_tokens)
        if first_padded > first_num_tokens and input_ids is not None:
            input_ids[first_num_tokens:first_padded].fill_(0)
            if positions is not None:
                if positions.ndim == 2:
                    positions[:, first_num_tokens:first_padded].fill_(0)
                else:
                    positions[first_num_tokens:first_padded].fill_(0)

    def _make_split_batch_metadata_dual_pad(
        self,
        split_ubatch_slices: Any,
        split_batch_slices: Any,
        attn_metadata: Any,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor | None,
        inputs_embeds: torch.Tensor | None,
        intermediate_tensors: Any,
        batch_descriptor: BatchDescriptor,
        aclgraph_runtime_mode: CUDAGraphMode,
        inplace_attention_backend: str | None = None,
    ):
        """Build per-split forward contexts + sliced inputs for dual-pad.

        Both splits dispatch through the standard (``start_num_tokens=0``)
        dispatcher path so they reuse the graphs captured at startup (main
        pool for split 0, parallel pool for split 1 via
        ``in_parallel_streams``).  Split 1's execution inputs are rebound to
        the dedicated parallel-stream buffers (values pre-copied by
        ``_prepare_dual_pad_split_inputs``).
        """
        from vllm.forward_context import get_forward_context

        from vllm_ascend.ascend_forward_context import create_ascend_forward_context

        cur_forward_context = get_forward_context()
        dp_metadata = getattr(cur_forward_context, 'dp_metadata', None)
        runner = self._runner

        # Contexts use local (per-buffer) token coordinates so cos/sin covers
        # the full graph size, matching what graph capture bound.  The
        # attention metadata, by contrast, was already split with the original
        # (main-buffer) token coordinates by split_attn_metadata + stabilize.
        from vllm.v1.worker.ubatch_utils import UBatchSlice
        context_ubatch_slices = [
            UBatchSlice(s.request_slice, slice(0, int(s.graph_num_tokens)))
            for s in split_batch_slices
        ]

        ubatch_metadata = []
        for i, split_slice in enumerate(split_batch_slices):
            in_parallel_streams = i > 0
            ubatch_attn_metadata = None
            if attn_metadata is not None:
                if isinstance(attn_metadata, list) and i < len(attn_metadata):
                    ubatch_attn_metadata = attn_metadata[i]
                else:
                    ubatch_attn_metadata = attn_metadata

            if in_parallel_streams:
                # Parallel-pool graphs are keyed by the exact padded parallel
                # size; the main dispatcher would re-pad the size to the
                # nearest main capture size and miss the captured graph.
                from vllm_ascend.worker.dual_pad_utils import (
                    make_dual_pad_parallel_batch_descriptor)
                ubatch_cudagraph_mode = CUDAGraphMode.FULL
                ubatch_batch_descriptor = (
                    make_dual_pad_parallel_batch_descriptor(
                        split_slice.graph_num_tokens,
                        has_lora=batch_descriptor.has_lora,
                        num_active_loras=getattr(
                            batch_descriptor, "num_active_loras", 0),
                        uniform_decode_query_len=(
                            self.cudagraph_dispatcher
                            .uniform_decode_query_len),
                        max_num_seqs=(
                            self.vllm_config.scheduler_config.max_num_seqs),
                    ))
            else:
                ubatch_cudagraph_mode, ubatch_batch_descriptor = (
                    self.cudagraph_dispatcher.dispatch(
                        num_tokens=split_slice.graph_num_tokens,
                        uniform_decode=True,
                        has_lora=batch_descriptor.has_lora,
                    ))

            ctx_stream = (self.stream_parallel if in_parallel_streams
                          else self.stream_main)

            if in_parallel_streams:
                ctx_positions = getattr(
                    runner, "positions_parallel_streams", None)
            else:
                ctx_positions = positions
            if ctx_positions is None:
                ctx_positions = positions
            if ctx_positions is not None:
                ctx_positions = ctx_positions[:split_slice.graph_num_tokens]

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
                    positions=ctx_positions,
                    in_parallel_streams=in_parallel_streams,
                    cos_sin_slot_id=i,
                )
            setattr(split_forward_context, "split_inplace_mode", "dual_pad")
            setattr(split_forward_context, "forced_attention_backend",
                    inplace_attention_backend)
            setattr(split_forward_context, "allow_inplace_lazy_capture",
                    False)
            setattr(split_forward_context, "split_actual_num_tokens",
                    int(split_slice.num_tokens))
            setattr(split_forward_context, "split_graph_num_tokens",
                    int(split_slice.graph_num_tokens))

            sliced_input_ids, sliced_positions, sliced_inputs_embeds, \
                sliced_intermediate_tensors = slice_split_batch_inputs(
                    split_slice.token_slice, input_ids, positions,
                    inputs_embeds, intermediate_tensors,
                    vllm_config=self.vllm_config,
                )

            if in_parallel_streams:
                num_tokens = int(split_slice.num_tokens)
                padded_tokens = int(split_slice.graph_num_tokens)
                if (sliced_input_ids is not None
                        and getattr(runner, "input_ids_parallel_streams", None) is not None):
                    sliced_input_ids = (
                        runner.input_ids_parallel_streams[:padded_tokens])
                if (sliced_positions is not None
                        and getattr(runner, "positions_parallel_streams", None) is not None):
                    sliced_positions = (
                        runner.positions_parallel_streams[:padded_tokens])
                if (sliced_inputs_embeds is not None
                        and getattr(runner, "inputs_embeds_parallel_streams", None) is not None):
                    sliced_inputs_embeds = (
                        runner.inputs_embeds_parallel_streams[:padded_tokens])

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

    def run_dual_pad(
        self,
        split_plan: Any,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        intermediate_tensors: Any,
        inputs_embeds: torch.Tensor | None,
        attn_metadata: Any,
        cudagraph_mode: CUDAGraphMode,
        batch_descriptor: BatchDescriptor,
        **model_kwargs,
    ) -> torch.Tensor:
        """Execute a dual-pad split (two independent padded buffers).

        Reuses the dual-inplace parallel skeleton (two worker threads on
        stream_main / stream_parallel, per-split attn param updates, event
        based merge) but with dual-pad input semantics: split 1 reads from the
        dedicated parallel-stream buffers and both splits dispatch through the
        standard (non-offset) graph path.
        """
        split_cfg = getattr(self.ascend_config, "split_batch_config", None)
        split_slices = split_plan.split_slices
        original_forward_context = get_forward_context()
        num_splits = len(split_slices)

        from vllm_ascend.worker.inplace_split_utils import (
            select_inplace_attention_backend,
        )
        inplace_attention_backend = select_inplace_attention_backend(
            split_plan,
            lambda shape: using_paged_attention(shape, self.vllm_config))

        if self.stream_main is None:
            self._runner.stream_main = torch.npu.current_stream()
        if self.stream_parallel is None:
            self._runner.stream_parallel = torch.npu.Stream(device=self.device)

        self._apply_inplace_parallel_replay_stream_limits()

        second_slot_mapping = self._extract_second_split_slot_mapping(
            split_slices, attn_metadata)
        self._prepare_dual_pad_split_inputs(
            split_slices, input_ids, positions, inputs_embeds,
            second_slot_mapping=second_slot_mapping)

        from vllm.v1.worker.ubatch_utils import UBatchSlice
        split_ubatch_slices = [
            UBatchSlice(s.request_slice, s.token_slice)
            for s in split_slices
        ]

        ubatch_metadata = self._make_split_batch_metadata_dual_pad(
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

        clone_split_outputs = _inplace_parallel_clone_split_outputs(
            allow_auto_direct_outputs=True,
            split_cfg=split_cfg,
            split_batch_slices=split_slices,
            aclgraph_runtime_mode=cudagraph_mode,
            merge_sync_policy=_INPLACE_PARALLEL_MERGE_SYNC_POLICY,
        )

        results: list[Any | None] = [None] * num_splits
        split_done_events: list[torch.npu.Event | None] = [None] * num_splits
        split_errors: list[tuple[int, Exception]] = []
        split_error_lock = threading.Lock()

        self._wait_inplace_parallel_output_release_events(
            (self.stream_main, self.stream_parallel))

        def _finish_dual_pad_split_result(
                slice_idx: int,
                split_result: Any,
                *,
                parallel_streams: bool,
                target_stream: torch.npu.Stream) -> None:
            split_slice = split_slices[slice_idx]
            with torch.npu.stream(target_stream):
                trimmed_result = trim_split_output(
                    split_result, split_slice.num_tokens)
                if clone_split_outputs:
                    trimmed_result = clone_split_output(trimmed_result)
                results[slice_idx] = trimmed_result
                if parallel_streams:
                    split_done_events[slice_idx] = (
                        self._record_replay_stream_event(target_stream))

        def _run_dual_pad_worker(slice_idx: int) -> None:
            try:
                split_slice = split_slices[slice_idx]
                metadata = ubatch_metadata[slice_idx]
                parallel_streams = slice_idx > 0
                target_stream = (self.stream_parallel if parallel_streams
                                 else self.stream_main)

                with torch.inference_mode():
                    with torch.npu.stream(target_stream):
                        with override_forward_context(metadata.context):
                            split_result = self.model(
                                input_ids=metadata.input_ids,
                                positions=metadata.positions,
                                inputs_embeds=metadata.inputs_embeds,
                                intermediate_tensors=metadata.intermediate_tensors,
                                **model_kwargs,
                            )
                            if (metadata.context.cudagraph_runtime_mode
                                    == CUDAGraphMode.FULL):
                                if parallel_streams:
                                    self._update_attn_params_for_split_ubatch(
                                        metadata.context,
                                        split_slice.graph_num_tokens,
                                        parallel_streams=True)
                                else:
                                    self._update_attn_params_for_wrapper(
                                        metadata.context,
                                        split_slice.graph_num_tokens)

                _finish_dual_pad_split_result(
                    slice_idx,
                    split_result,
                    parallel_streams=parallel_streams,
                    target_stream=target_stream)
            except Exception as e:
                with split_error_lock:
                    split_errors.append((slice_idx, e))

        self._get_replay_worker_pool().dispatch([
            (lambda i=slice_idx: _run_dual_pad_worker(i))
            for slice_idx in range(num_splits)
        ])

        if split_errors:
            split_errors.sort(key=lambda item: item[0])
            failed_slice_idx, first_error = split_errors[0]
            raise RuntimeError(
                "dual pad replay worker failed at "
                f"slice_idx={failed_slice_idx}: {first_error}") from first_error

        merged_results: list[Any] = [
            result for result in results if result is not None
        ]
        if len(merged_results) != num_splits:
            raise RuntimeError(
                "Missing dual pad split result: "
                f"expected={num_splits}, got={len(merged_results)}")

        if _INPLACE_PARALLEL_MERGE_SYNC_POLICY == "host_sync":
            self.stream_main.synchronize()
            if num_splits > 1:
                self.stream_parallel.synchronize()
            with override_forward_context(original_forward_context):
                result = merge_split_outputs(merged_results)
                if not clone_split_outputs:
                    self._record_inplace_parallel_output_release_event()
            return result

        with torch.npu.stream(self.stream_main):
            self._wait_replay_stream_events(self.stream_main,
                                            split_done_events)
            for merged_result in merged_results:
                _record_stream_tree_for_npugraph_ex(merged_result,
                                                    self.stream_main)
            with override_forward_context(original_forward_context):
                result = merge_split_outputs(merged_results)
                if not clone_split_outputs:
                    self._record_inplace_parallel_output_release_event()
        return result

    def capture_parallel_stream_graphs(self) -> None:
        from vllm_ascend.compilation.acl_graph import ACLGraphWrapper

        runner = self._runner
        if not isinstance(runner.model, ACLGraphWrapper):
            return
        if not hasattr(runner, 'cudagraph_batch_sizes_parallel') or runner.cudagraph_batch_sizes_parallel is None:
            return
        for batch_size in runner.cudagraph_batch_sizes_parallel:
            # Capture uniform-decode graphs so the BatchDescriptor (uniform
            # flag included) matches the runtime dispatch used by dual-pad.
            runner._dummy_run(
                batch_size,
                in_parallel_streams=True,
                uniform_decode=True,
            )