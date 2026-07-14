# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, ClassVar, Optional
from unittest.mock import patch

import torch
import torch_npu
import vllm.envs as envs
from vllm.compilation.counter import compilation_counter
from vllm.compilation.cuda_graph import CUDAGraphOptions
from vllm.compilation import monitor as compilation_monitor
from vllm.config import CUDAGraphMode, VllmConfig
from vllm.forward_context import BatchDescriptor, get_forward_context
from vllm.logger import logger
from vllm.platforms import current_platform

from vllm_ascend.ascend_forward_context import _EXTRA_CTX


from ..utils import weak_ref_tensors
from vllm.distributed.device_communicators.pynccl_allocator import \
    set_graph_pool_id

from vllm_ascend.compilation.acl_graph_diagnostics import (
    collect_attn_metadata_tensor_infos as _collect_attn_metadata_tensor_infos,
    collect_tensor_arg_infos as _collect_tensor_arg_infos,
    resolve_callable_arg_names as _resolve_callable_arg_names,
    resolve_callable_name as _resolve_callable_name,
    should_validate_inplace_metadata_ptrs as _should_validate_inplace_metadata_ptrs,
    validate_input_addresses,
)

from vllm_ascend.compilation.acl_graph_split_batch import (
    is_allowed_inplace_lazy_capture,
)
_ACLGRAPH_REPLAY_GLOBAL_SYNC = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_REPLAY_GLOBAL_SYNC", "0")
    in ("1", "true", "True")
)

_acl_graph_wrappers: weakref.WeakSet[Any] = weakref.WeakSet()
_STREAM_RESOURCE_ERROR_CODE = "207008"
_STREAM_RESOURCE_ERROR_MARKERS = (
    "insufficient_stream_resources",
    "stream resources are insufficient",
)
_STREAM_RESOURCE_GUIDANCE = (
    "ACL graph capture failed with a known stream-resource exhaustion "
    "signature. Consider upgrading to a newer HDK/CANN stack, reducing "
    "cudagraph_capture_sizes, lowering max_cudagraph_capture_size, preferring "
    "FULL or FULL_DECODE_ONLY for mostly uniform decode workloads, or "
    "temporarily disabling graph mode to confirm the failure is capture-related."
)


def _is_stream_resource_capture_error(exc: RuntimeError) -> bool:
    message = str(exc)
    lowered_message = message.lower()
    has_error_code = _STREAM_RESOURCE_ERROR_CODE in message
    has_stream_resource_marker = any(marker in lowered_message for marker in _STREAM_RESOURCE_ERROR_MARKERS)
    return has_stream_resource_marker or (has_error_code and "stream resource" in lowered_message)


def _raise_stream_resource_capture_error(exc: RuntimeError) -> None:
    raise RuntimeError(f"{_STREAM_RESOURCE_GUIDANCE}\nOriginal error:\n{exc}") from exc


@dataclass
class ACLGraphEntry:
    batch_descriptor: BatchDescriptor
    aclgraph: torch.npu.NPUGraph | None = None
    output: Any | None = None
    capture_count: int = 0
    replay_count: int = 0
    fallback_eager_count: int = 0

    input_addresses: list[int] | None = None
    input_tensor_infos: list[dict[str, Any]] | None = None
    attn_metadata_addresses: list[int] | None = None
    attn_metadata_tensor_infos: list[dict[str, Any]] | None = None


class ACLGraphWrapper:
    """Wraps a runnable to add acl graph capturing and replaying ability. And
    provide attribute access to the underlying `runnable` via `__getattr__`.

    The workflow of this wrapper in the aclgraph dispatching is as follows:
    1. At initialization, a runtime mode is assigned to the wrapper (FULL or
    PIECEWISE).
    2. At runtime, the wrapper receives a runtime_mode and a
    batch_descriptor(key) from the forward context and blindly trust them
    for aclgraph dispatching.
    3. If runtime_mode is NONE or runtime_mode does not match the mode of the
    wrapper, just call the runnable directly.
    4. Otherwise, i.e., the runtime_mode matches the mode of the wrapper,
    the wrapper will perform aclgraph capture(if key does not exist, create
    a new entry and cache it) or replay (if key exists in the cache).

    Note: ACLGraphWrapper does not store persistent buffers or copy any
    runtime inputs into that buffers for replay. We assume implementing them
    is done outside of the wrapper. That is because we do not make any
    assumption on the dynamic shape (batch size) of the runtime inputs, as a
    trade-off for staying orthogonal to compilation logic. Nevertheless,
    tracing and checking the input addresses to be consistent during replay is
    guaranteed when VLLM_LOGGING_LEVEL == "DEBUG".
    """

    _all_instances: ClassVar[weakref.WeakSet["ACLGraphWrapper"]] = weakref.WeakSet()

    @classmethod
    def clear_all_graphs(cls) -> None:
        for instance in list(cls._all_instances):
            instance.clear_graphs()

    def __init__(
        self,
        runnable: Callable,
        vllm_config: VllmConfig,
        runtime_mode: CUDAGraphMode,
        cudagraph_options: CUDAGraphOptions | None = None,
        *,
        use_eagle: bool = False,
        enable_enpu: bool = False,
    ):
        self.runnable = runnable
        self.vllm_config = vllm_config
        self.runtime_mode = runtime_mode
        self.compilation_config = vllm_config.compilation_config

        self.first_run_finished = False
        self.is_debugging_mode = envs.VLLM_LOGGING_LEVEL == "DEBUG"
        self._runnable_str = str(runnable) if self.is_debugging_mode else None
        self.runnable_arg_names = _resolve_callable_arg_names(runnable)
        self.runnable_name = _resolve_callable_name(runnable)

        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()
        self.graph_pool_parallel_streams = torch.npu.graph_pool_handle()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}
        self.concrete_aclgraph_entries2: dict[BatchDescriptor, ACLGraphEntry] = {}
        self.enable_enpu = enable_enpu
        self.use_eagle = use_eagle
        _acl_graph_wrappers.add(self)

        ACLGraphWrapper._all_instances.add(self)

    def __getattr__(self, key: str):
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of aclgraph wrapper: {self._runnable_str}"
            )
        raise AttributeError(f"Attribute {key} not found. Set VLLM_LOGGING_LEVEL=DEBUG for more details.")

    def unwrap(self) -> Callable:
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "ACLGraphWrapper":
        return self

    def clear_graphs(self) -> None:
        self.concrete_aclgraph_entries.clear()

    def has_graph(
        self,
        batch_descriptor: BatchDescriptor,
        in_parallel_streams: bool = False,
    ) -> bool:
        entries = (
            self.concrete_aclgraph_entries2
            if in_parallel_streams
            else self.concrete_aclgraph_entries
        )
        entry = entries.get(batch_descriptor)
        return entry is not None and entry.aclgraph is not None

    def __call__(self, *args, **kwargs):
        forward_context = get_forward_context()
        batch_descriptor = forward_context.batch_descriptor
        aclgraph_runtime_mode = forward_context.cudagraph_runtime_mode

        if aclgraph_runtime_mode == CUDAGraphMode.NONE or aclgraph_runtime_mode != self.runtime_mode:
            return self.runnable(*args, **kwargs)

        in_parallel_streams = bool(
            getattr(forward_context, "in_parallel_streams", False))
        entries = (
            self.concrete_aclgraph_entries2
            if in_parallel_streams
            else self.concrete_aclgraph_entries
        )
        graph_pool = (
            self.graph_pool_parallel_streams
            if in_parallel_streams
            else self.graph_pool
        )

        if batch_descriptor not in entries:
            entries[batch_descriptor] = ACLGraphEntry(batch_descriptor=batch_descriptor)

        entry = entries[batch_descriptor]

        if entry.aclgraph is None:
            start_num_tokens = batch_descriptor.start_num_tokens
            is_inplace_lazy_capture = is_allowed_inplace_lazy_capture(
                forward_context, batch_descriptor, aclgraph_runtime_mode
            )

            if start_num_tokens > 0 and not is_inplace_lazy_capture:
                raise RuntimeError(
                    f"Offset ACL Graph (start_num_tokens={start_num_tokens}) not found "
                    f"and lazy capture is not allowed. "
                    f"BatchDescriptor: {batch_descriptor}"
                )


            previous_capture_enabled = compilation_monitor.cudagraph_capturing_enabled
            if is_inplace_lazy_capture:
                compilation_monitor.set_cudagraph_capturing_enabled(True)
            try:
                compilation_monitor.validate_cudagraph_capturing_enabled()
            except Exception:
                if is_inplace_lazy_capture:
                    compilation_monitor.set_cudagraph_capturing_enabled(previous_capture_enabled)
                raise

            input_addresses, input_tensor_infos = _collect_tensor_arg_infos(
                args,
                self.runnable_arg_names,
            )
            entry.input_addresses = input_addresses
            entry.input_tensor_infos = input_tensor_infos

            if _should_validate_inplace_metadata_ptrs(forward_context):
                (entry.attn_metadata_addresses,
                 entry.attn_metadata_tensor_infos) = (
                     _collect_attn_metadata_tensor_infos(
                         getattr(forward_context, "attn_metadata", None)))
            aclgraph = torch.npu.NPUGraph()

            with ExitStack() as stack:
                if self.aclgraph_options.gc_disable:
                    stack.enter_context(patch("gc.collect", lambda: None))
                    stack.enter_context(patch("torch.npu.empty_cache", lambda: None))

                from vllm.model_executor.offloader.base import get_offloader

                get_offloader().sync_prev_onload()
                previous_capturing = bool(getattr(forward_context, 'capturing', False))
                forward_context.capturing = True
                set_graph_pool_id(graph_pool)
                try:
                    with torch.npu.graph(aclgraph, pool=graph_pool):
                        output = self.runnable(*args, **kwargs)
                        get_offloader().join_after_forward()
                        if self.aclgraph_options.weak_ref_output:
                            output = weak_ref_tensors(output)
                except RuntimeError as exc:
                    if _is_stream_resource_capture_error(exc):
                        _raise_stream_resource_capture_error(exc)
                    raise
                finally:
                    forward_context.capturing = previous_capturing
                    if is_inplace_lazy_capture:
                        compilation_monitor.set_cudagraph_capturing_enabled(previous_capture_enabled)

            global _graph_params, _draft_graph_params, _draft_graph_prefill_params
            global _graph_params_parallel
            weak_ref_workspaces(_graph_params)
            weak_ref_workspaces(_draft_graph_params)
            weak_ref_workspaces(_draft_graph_prefill_params)
            weak_ref_workspaces(_graph_params_parallel)

            entry.output = weak_ref_tensors(output)
            entry.aclgraph = aclgraph
            entry.capture_count += 1

            compilation_counter.num_cudagraph_captured += 1

            return output

        if self.is_debugging_mode or _should_validate_inplace_metadata_ptrs(forward_context):
            validate_input_addresses(
                entry.input_addresses, args, self.runnable_name)

        logger.info_once("Replaying aclgraph")
        is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        if not in_parallel_streams:
            need_sync = self.runtime_mode == CUDAGraphMode.FULL and not is_draft_eagle
            if not self.enable_enpu and need_sync:
                torch.npu.current_stream().synchronize()
        elif _ACLGRAPH_REPLAY_GLOBAL_SYNC:
            torch.npu.synchronize()
        set_graph_pool_id(graph_pool)
        entry.aclgraph.replay()
        entry.replay_count += 1
        return entry.output


def weak_ref_workspaces(params):
    if params is None:
        return
    for num_tokens in params.workspaces:
        if params.workspaces[num_tokens] is None:
            continue
        params.workspaces[num_tokens] = weak_ref_tensors(params.workspaces[num_tokens])


def update_full_graph_params(
    attn_backend,
    update_stream,
    forward_context,
    num_tokens,
    vllm_config,
    speculative_config=None,
    num_dcp_pcp_tokens=None,
    draft_attn_metadatas=None,

):
    impl_cls = attn_backend.get_impl_cls()
    impl_cls.update_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        speculative_config,
        num_dcp_pcp_tokens,
        draft_attn_metadatas,
    )

    from vllm_ascend.ops.gdn import update_conv1d_graph_params

    update_conv1d_graph_params(
        update_stream,
        forward_context,
        num_tokens,
        vllm_config,
        _EXTRA_CTX.is_draft_model,
        draft_attn_metadatas,
    )


@dataclass
class GraphParams:
    events: dict[int, list[torch.npu.ExternalEvent]]
    workspaces: dict[int, torch.Tensor]
    handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    attn_params: dict[int, list[tuple]]
    conv1d_params: dict[int, list[tuple]]
    conv1d_handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]
    conv1d_events: dict[int, list[torch.npu.ExternalEvent]]


_graph_params: GraphParams | None = None


def reset_graph_params():
    global _graph_params, _draft_graph_params, _draft_graph_prefill_params, _graph_params_parallel
    _graph_params = None
    _draft_graph_params = None
    _draft_graph_prefill_params = None
    _graph_params_parallel = None


def set_graph_params(aclgraph_capture_sizes: list[int]):
    global _graph_params
    if _graph_params is not None:
        raise ValueError("Graph parameters have already been set!")
    _graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_graph_params_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params
    if _graph_params is not None:
        _graph_params.workspaces[num_tokens] = workspace


def get_graph_params(in_parallel_streams: bool = False) -> GraphParams | None:
    if in_parallel_streams and _graph_params_parallel is not None:
        return _graph_params_parallel
    return _graph_params


_draft_graph_params: GraphParams | None = None


def set_draft_graph_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_params
    if _draft_graph_params is not None:
        raise ValueError("DraftGraph parameters have already been set!")
    _draft_graph_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_draft_graph_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_params
    if _draft_graph_params is not None:
        _draft_graph_params.workspaces[num_tokens] = workspace


def get_draft_graph_params():
    return _draft_graph_params


_draft_graph_prefill_params: GraphParams | None = None


def set_draft_graph_prefill_params(aclgraph_capture_sizes: list[int]):
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        raise ValueError("DraftGraph preill parameters have already been set!")
    _draft_graph_prefill_params = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_draft_graph_prefill_params_workspaces(num_tokens: int, workspace: Any):
    global _draft_graph_prefill_params
    if _draft_graph_prefill_params is not None:
        _draft_graph_prefill_params.workspaces[num_tokens] = workspace


def get_draft_graph_prefill_params():
    return _draft_graph_prefill_params


_graph_params_parallel: GraphParams | None = None


def set_graph_params_parallel(aclgraph_capture_sizes: list[int]):
    global _graph_params_parallel
    if _graph_params_parallel is not None:
        raise ValueError("Parallel graph parameters have already been set!")
    _graph_params_parallel = GraphParams(
        {size: [] for size in aclgraph_capture_sizes},
        {size: None for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
        {size: [] for size in aclgraph_capture_sizes},
    )


def update_graph_params_parallel_workspaces(num_tokens: int, workspace: torch.Tensor):
    global _graph_params_parallel
    if _graph_params_parallel is not None:
        _graph_params_parallel.workspaces[num_tokens] = workspace


def get_graph_params_parallel():
    return _graph_params_parallel
