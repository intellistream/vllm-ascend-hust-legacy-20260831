# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import dataclasses
import inspect
import os
from typing import Any, Optional

import torch

_DIAG_LOG_PATH = os.environ.get("VLLM_ASCEND_SPLIT_DIAG_LOG", "")
_diag_log_file = None
SPLIT_INPLACE_DEBUG = os.environ.get("VLLM_ASCEND_SPLIT_INPLACE_DEBUG", "0") == "1"


def diag_log(msg: str):
    global _diag_log_file
    if not _DIAG_LOG_PATH:
        return
    if _diag_log_file is None:
        try:
            _diag_log_file = open(_DIAG_LOG_PATH, "a")
        except Exception:
            return
    _diag_log_file.write(msg + "\n")
    _diag_log_file.flush()


def safe_tensor_ptr(t: torch.Tensor) -> Optional[int]:
    try:
        return t.data_ptr()
    except Exception:
        return None


def safe_tensor_shape(t: torch.Tensor) -> Optional[list[int]]:
    try:
        return list(t.shape)
    except Exception:
        return None


def resolve_callable_arg_names(runnable) -> Optional[list[str]]:
    try:
        return list(inspect.signature(runnable).parameters.keys())
    except Exception:
        return None


def resolve_callable_name(runnable) -> str:
    if hasattr(runnable, "__qualname__"):
        return str(getattr(runnable, "__qualname__"))
    if hasattr(runnable, "__name__"):
        return str(getattr(runnable, "__name__"))
    return type(runnable).__name__


def collect_tensor_arg_infos(
    args: tuple[Any, ...],
    arg_names: Optional[list[str]] = None,
) -> tuple[list[int], list[dict[str, Any]]]:
    addresses: list[int] = []
    tensor_infos: list[dict[str, Any]] = []
    for arg_index, arg in enumerate(args):
        if not isinstance(arg, torch.Tensor):
            continue
        ptr = safe_tensor_ptr(arg)
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
            "shape": safe_tensor_shape(arg),
            "dtype": str(arg.dtype),
            "device": str(arg.device),
            "stride": list(arg.stride()),
            "is_contiguous": bool(arg.is_contiguous()),
        })
    return addresses, tensor_infos


def collect_attn_metadata_tensor_infos(
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
            ptr = safe_tensor_ptr(value)
            if ptr is None:
                ptr = -1
            addresses.append(ptr)
            tensor_infos.append({
                "tensor_index": len(addresses) - 1,
                "path": path,
                "shape": safe_tensor_shape(value),
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


def should_validate_inplace_metadata_ptrs(forward_context: Any) -> bool:
    return bool(getattr(forward_context, "validate_inplace_metadata_ptrs",
                        False))


def validate_input_addresses(
    entry_input_addresses: list[int] | None,
    current_args: tuple[Any, ...],
    runnable_name: str,
) -> None:
    if entry_input_addresses is None:
        return
    new_input_addresses = [x.data_ptr() for x in current_args if isinstance(x, torch.Tensor)]
    if new_input_addresses != entry_input_addresses:
        assert entry_input_addresses is not None, (
            "input_addresses not recorded during capture")
        assert len(new_input_addresses) == len(entry_input_addresses), (
            f"Input address count mismatch: expected "
            f"{len(entry_input_addresses)}, got {len(new_input_addresses)}")
        for i, (new_addr, cached_addr) in enumerate(
                zip(new_input_addresses, entry_input_addresses)):
            if new_addr != cached_addr:
                raise AssertionError(
                    f"Input address mismatch at index {i}: "
                    f"expected {cached_addr:#x}, got {new_addr:#x}")