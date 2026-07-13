# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import inspect
import os
import time
import weakref
from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, ClassVar, Optional
from unittest.mock import patch

_ACL_DIAG_LOG_PATH = os.environ.get("VLLM_ASCEND_SPLIT_DIAG_LOG", "")
_acl_diag_log_file = None
_ACLGRAPH_REPLAY_GLOBAL_SYNC = (
    os.environ.get("VLLM_ASCEND_ACLGRAPH_REPLAY_GLOBAL_SYNC", "0")
    in ("1", "true", "True")
)


def _acl_diag_log(msg: str):
    global _acl_diag_log_file
    if not _ACL_DIAG_LOG_PATH:
        return
    if _acl_diag_log_file is None:
        try:
            _acl_diag_log_file = open(_ACL_DIAG_LOG_PATH, "a")
        except Exception:
            return
    _acl_diag_log_file.write(msg + "\n")
    _acl_diag_log_file.flush()


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
from vllm_ascend.attention.utils import using_paged_attention
from vllm_ascend.worker.inplace_split_utils import SplitBatchSlice

from ..utils import weak_ref_tensors
from vllm.distributed.device_communicators.pynccl_allocator import \
    set_graph_pool_id

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


def _safe_tensor_ptr(t: torch.Tensor) -> Optional[int]:
    try:
        return t.data_ptr()
    except Exception:
        return None


def _safe_tensor_shape(t: torch.Tensor) -> Optional[list[int]]:
    try:
        return list(t.shape)
    except Exception:
        return None


def _resolve_callable_arg_names(runnable: Callable) -> Optional[list[str]]:
    try:
        return list(inspect.signature(runnable).parameters.keys())
    except Exception:
        return None


def _resolve_callable_name(runnable: Callable) -> str:
    if hasattr(runnable, "__qualname__"):
        return str(getattr(runnable, "__qualname__"))
    if hasattr(runnable, "__name__"):
        return str(getattr(runnable, "__name__"))
    return type(runnable).__name__


def _collect_tensor_arg_infos(
    args: tuple[Any, ...],
    arg_names: Optional[list[str]] = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    addresses: list[int] = []
    tensor_infos: list[dict[str, Any]] = []
    for arg_index, arg in enumerate(args):
        if not isinstance(arg, torch.Tensor):
            continue
        ptr = _safe_tensor_ptr(arg)
        if ptr is None:
            ptr = -1
        addresses.append(ptr)
        tensor_infos.append({
            "tensor_index": len(addresses) - 1,
            "arg_index": arg_index,
            "arg_name": (
                arg_names[arg_index]
                if arg_names is not None and arg_index < len(arg_names)
                else None
            ),
            "shape": _safe_tensor_shape(arg),
            "dtype": str(arg.dtype),
            "device": str(arg.device),
            "stride": list(arg.stride()),
            "is_contiguous": bool(arg.is_contiguous()),
        })
    return addresses, tensor_infos


def _collect_attn_metadata_tensor_infos(
    attn_metadata: Any,
    *,
    max_tensors: int = 200,
    max_depth: int = 8,
) -> tuple[list[int], list[dict[str, Any]]]:
    addresses: list[int] = []
    tensor_infos: list[dict[str, Any]] = []
    visited: set[int] = set()

    def visit(value: Any, path: str, depth: int) -> None:
        if len(addresses) >= max_tensors or depth > max_depth:
            return
        if isinstance(value, torch.Tensor):
            ptr = _safe_tensor_ptr(value)
            if ptr is None:
                ptr = -1
            addresses.append(ptr)
            tensor_infos.append({
                "tensor_index": len(addresses) - 1,
                "path": path,
                "shape": _safe_tensor_shape(value),
                "dtype": str(value.dtype),
                "device": str(value.device),
                "stride": list(value.stride()),
                "is_contiguous": bool(value.is_contiguous()),
                "storage_offset": int(value.storage_offset()),
            })
            return

        if value is None or isinstance(value, (str, bytes, int, float, bool)):
            return
        value_id = id(value)
        if value_id in visited:
            return
        visited.add(value_id)

        if isinstance(value, dict):
            for key, child in value.items():
                visit(child, f"{path}.{key}", depth + 1)
            return
        if isinstance(value, (list, tuple)):
            for idx, child in enumerate(value):
                visit(child, f"{path}[{idx}]", depth + 1)
            return
        if dataclasses.is_dataclass(value):
            for fld in dataclasses.fields(value):
                visit(getattr(value, fld.name, None),
                      f"{path}.{fld.name}", depth + 1)
            return

        attrs = getattr(value, "__dict__", None)
        if isinstance(attrs, dict):
            for name, child in attrs.items():
                if name.startswith("__"):
                    continue
                visit(child, f"{path}.{name}", depth + 1)

    visit(attn_metadata, "attn_metadata", 0)
    return addresses, tensor_infos


def _should_validate_inplace_metadata_ptrs(forward_context: Any) -> bool:
    return bool(getattr(forward_context, "validate_inplace_metadata_ptrs",
                        False))


def _is_allowed_inplace_lazy_capture(
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


def _extract_block_table_from_metadata(metadata: Any):
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


def _refresh_block_table_in_place(
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


def _resolve_graph_param_key(graph_params, forward_context, runtime_shape):
    if hasattr(graph_params.attn_params, 'keys'):
        for k in graph_params.attn_params.keys():
            if k == runtime_shape:
                return k
    return runtime_shape


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


def require_graph_param_key(graph_params, param_key, *, op: str = ""):
    if param_key not in graph_params.attn_params:
        raise KeyError(
            f"{op}: graph_param_key {param_key!r} not found in "
            f"graph_params.attn_params (keys={list(graph_params.attn_params.keys())})")


def _has_dual_stream_attention_metadata(forward_context) -> bool:
    dual_metadata = getattr(forward_context,
                            "dual_stream_attention_metadata", None)
    return isinstance(dual_metadata, list) and len(dual_metadata) == 2


_SPLIT_INPLACE_DEBUG = os.environ.get("VLLM_ASCEND_SPLIT_INPLACE_DEBUG", "0") == "1"


@dataclasses.dataclass
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

        # assert runtime_mode is not NONE(no aclgraph), otherwise, we don't
        # need to initialize a ACLGraphWrapper.
        assert self.runtime_mode != CUDAGraphMode.NONE
        self.graph_pool = current_platform.get_global_graph_pool()
        self.graph_pool_parallel_streams = torch.npu.graph_pool_handle()

        if cudagraph_options is None:
            cudagraph_options = CUDAGraphOptions()
        self.aclgraph_options = cudagraph_options
        # the entries for different batch descriptors that we need to capture
        # aclgraphs for.
        self.concrete_aclgraph_entries: dict[BatchDescriptor, ACLGraphEntry] = {}
        self.concrete_aclgraph_entries2: dict[BatchDescriptor, ACLGraphEntry] = {}
        self.enable_enpu = enable_enpu
        self.use_eagle = use_eagle
        _acl_graph_wrappers.add(self)

        ACLGraphWrapper._all_instances.add(self)

    def __getattr__(self, key: str):
        # allow accessing the attributes of the runnable.
        if hasattr(self.runnable, key):
            return getattr(self.runnable, key)
        if self.is_debugging_mode:
            raise AttributeError(
                f"Attribute {key} not exists in the runnable of aclgraph wrapper: {self._runnable_str}"
            )
        raise AttributeError(f"Attribute {key} not found. Set VLLM_LOGGING_LEVEL=DEBUG for more details.")

    def unwrap(self) -> Callable:
        # in case we need to access the original runnable.
        return self.runnable

    @property
    def cudagraph_wrapper(self) -> "ACLGraphWrapper":
        return self

    def clear_graphs(self) -> None:
        self.concrete_aclgraph_entries.clear()

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
            is_inplace_lazy_capture = _is_allowed_inplace_lazy_capture(
                forward_context, batch_descriptor, aclgraph_runtime_mode
            )
            logger.info(
                "ACLGRAPH_MISS: desc=%s, start=%d, variant=%s, attn=%s, "
                "parallel=%s, lazy_allowed=%s, entries_count=%d, "
                "in_parallel_streams=%s, runtime_mode=%s",
                batch_descriptor.num_tokens, start_num_tokens,
                batch_descriptor.graph_variant, batch_descriptor.attention_backend,
                in_parallel_streams, is_inplace_lazy_capture,
                len(entries), in_parallel_streams, self.runtime_mode.name,
            )

            if start_num_tokens > 0 and not is_inplace_lazy_capture:
                raise RuntimeError(
                    f"Offset ACL Graph (start_num_tokens={start_num_tokens}) not found "
                    f"and lazy capture is not allowed. "
                    f"BatchDescriptor: {batch_descriptor}"
                )

            if self.aclgraph_options.debug_log_enable:
                logger.debug("Capturing a aclgraph on (%s,%s)", self.runtime_mode.name, entry.batch_descriptor)
            logger.info(
                "CAPTURE aclgraph: desc=%s, start=%d, variant=%s, attn=%s, parallel=%s",
                batch_descriptor.num_tokens, start_num_tokens,
                batch_descriptor.graph_variant, batch_descriptor.attention_backend,
                in_parallel_streams,
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
            new_input_addresses = [x.data_ptr() for x in args if isinstance(x, torch.Tensor)]
            if self.is_debugging_mode and new_input_addresses != entry.input_addresses:
                logger.warning(
                    "Input addresses for aclgraphs are different during replay. "
                    "Expected %d addrs, got %d addrs",
                    len(entry.input_addresses) if entry.input_addresses else 0,
                    len(new_input_addresses),
                )

        logger.info_once("Replaying aclgraph")
        is_draft_eagle = _EXTRA_CTX.is_draft_model and self.use_eagle
        if _ACLGRAPH_REPLAY_GLOBAL_SYNC and not in_parallel_streams:
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

    # For GDN Attention: AscendC operate(conv1d update) update graph params
    # No patch can be loaded, update method call is temporarily placed here
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
    conv1d_params: dict[int, list[tuple]]  # for causal conv1d params
    conv1d_handles: dict[int, list[torch_npu._C._NPUTaskGroupHandle]]  # for causal conv1d params handles
    conv1d_events: dict[int, list[torch.npu.ExternalEvent]]  # for causal conv1d params events


_graph_params: GraphParams | None = None


def reset_graph_params():
    global _graph_params, _draft_graph_params, _draft_graph_prefill_params
    _graph_params = None
    _draft_graph_params = None
    _draft_graph_prefill_params = None


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


def _update_attn_fia_params(update_stream, forward_context, runtime_shape,
                            refresh_block_table: bool = False,
                            in_parallel_streams: bool = False):
    graph_params = get_graph_params(in_parallel_streams)
    if graph_params is None:
        return
    param_key = _resolve_graph_param_key(graph_params, forward_context, runtime_shape)
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

            metadata_block_table, metadata_block_source = _extract_block_table_from_metadata(
                metadata)
            if refresh_block_table:
                _refresh_block_table_in_place(
                    block_tables, metadata_block_table)

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
    graph_params = get_graph_params(in_parallel_streams)
    if graph_params is None:
        return
    param_key = _resolve_graph_param_key(graph_params, forward_context, runtime_shape)
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
            metadata_block_table, metadata_block_source = _extract_block_table_from_metadata(
                metadata)
            if refresh_block_table:
                _refresh_block_table_in_place(
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
    if _has_dual_stream_attention_metadata(forward_context):
        _update_attn_dual_fia_params(update_stream, forward_context,
                                     runtime_shape,
                                     in_parallel_streams=in_parallel_streams)
    elif using_paged_attention(runtime_shape, vllm_config, forward_context):
        _update_attn_pa_params(update_stream, forward_context, runtime_shape,
                               in_parallel_streams=in_parallel_streams)
    else:
        _update_attn_fia_params(update_stream, forward_context, runtime_shape,
                                in_parallel_streams=in_parallel_streams)


def update_attn_params_split(update_stream, forward_context,
                             runtime_shape, vllm_config,
                             in_parallel_streams: bool = False):
    if _has_dual_stream_attention_metadata(forward_context):
        _update_attn_dual_fia_params(
            update_stream,
            forward_context,
            runtime_shape,
            refresh_block_table=True,
            in_parallel_streams=in_parallel_streams,
        )
    elif using_paged_attention(runtime_shape, vllm_config, forward_context):
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

                    metadata_block_table, _ = _extract_block_table_from_metadata(metadata)
                    if refresh_block_table:
                        _refresh_block_table_in_place(
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
