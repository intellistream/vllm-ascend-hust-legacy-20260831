from __future__ import annotations

import json
import os
import time
from itertools import count
from typing import Any

_DEBUG_ENV = "VLLM_ASCEND_SPLIT_INPLACE_DEBUG"
_DEBUG_FILE_ENV = "VLLM_ASCEND_SPLIT_INPLACE_DEBUG_FILE"
_DEFAULT_DEBUG_FILE = "/tmp/vllm_ascend_inplace_split.jsonl"

_step_counter = count(1)
_current_step_id: int | None = None


def _env_truthy(name: str) -> bool:
    return bool(int(os.getenv(name, "0")))


def is_enabled() -> bool:
    return _env_truthy(_DEBUG_ENV)


def next_step_id() -> int:
    return next(_step_counter)


def set_current_step_id(step_id: int) -> None:
    global _current_step_id
    _current_step_id = step_id


def get_current_step_id() -> int | None:
    return _current_step_id


def log_event(
    event: str,
    payload: dict[str, Any] | None = None,
    *,
    step_id: int | None = None,
) -> None:
    if not is_enabled():
        return
    if step_id is None:
        step_id = get_current_step_id()
    debug_file = os.getenv(_DEBUG_FILE_ENV, _DEFAULT_DEBUG_FILE)
    record: dict[str, Any] = {
        "event": event,
        "ts_ns": time.time_ns(),
        "pid": os.getpid(),
    }
    if step_id is not None:
        record["step_id"] = step_id
    if payload is not None:
        record["payload"] = payload
    try:
        with open(debug_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")

    except OSError:
        pass


def tensor_info(tensor: Any) -> dict[str, Any] | None:
    import torch
    if not isinstance(tensor, torch.Tensor):
        return None
    return {
        "ptr": tensor.data_ptr(),
        "shape": list(tensor.shape),
        "ndim": tensor.ndim,
        "dtype": str(tensor.dtype),
        "stride": list(tensor.stride()),
        "device": str(tensor.device),
        "is_contiguous": tensor.is_contiguous(),
    }


def batch_descriptor_info(batch_descriptor: Any) -> Any:
    if batch_descriptor is None:
        return None
    info: dict[str, Any] = {
        "num_tokens": getattr(batch_descriptor, "num_tokens", None),
        "num_reqs": getattr(batch_descriptor, "num_reqs", None),
        "uniform": getattr(batch_descriptor, "uniform", None),
        "has_lora": getattr(batch_descriptor, "has_lora", None),
    }
    for attr in ("start_num_tokens", "graph_variant", "attention_backend", "capture_metadata_mode"):
        val = getattr(batch_descriptor, attr, None)
        if val is not None and val != "" and val != 0:
            info[attr] = val
    return info


def metadata_tensor_info(attn_metadata: Any) -> dict[str, Any]:
    if attn_metadata is None:
        return {}
    info: dict[str, Any] = {}
    for attr in ("query_start_loc", "seq_lens", "block_table_tensor"):
        val = getattr(attn_metadata, attr, None)
        if val is not None:
            ti = tensor_info(val)
            if ti is not None:
                info[attr] = ti
    return info


def tensor_view_info(tensor: Any) -> dict[str, Any] | None:
    info = tensor_info(tensor)
    if info is None:
        return None
    info["data_ptr"] = info.pop("ptr")
    return info


def slice_info(value: slice) -> list:
    return [value.start, value.stop, value.step]