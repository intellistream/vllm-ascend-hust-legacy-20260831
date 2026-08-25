# SPDX-License-Identifier: Apache-2.0
"""Best-effort structured tracing for Ascend graph dispatch decisions."""

from __future__ import annotations

import json
import os
import time
from typing import Any

TRACE_PATH_ENV = "VLLM_COMPILATION_TRACE_PATH"
EVENT = "ascend_aclgraph_dispatch"


def emit_aclgraph_dispatch(
    *,
    action: str,
    batch_descriptor: Any,
    runtime_mode: Any,
    wrapper_mode: Any,
    is_draft_model: bool,
    use_eagle: bool,
) -> None:
    """Append one dispatch event without changing serving on trace failure."""
    path = os.getenv(TRACE_PATH_ENV, "").strip()
    if not path:
        return

    record = {
        "schema_version": 1,
        "event": EVENT,
        "timestamp_ns": time.time_ns(),
        "monotonic_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "rank": os.getenv("RANK"),
        "local_rank": os.getenv("LOCAL_RANK"),
        "action": action,
        "runtime_mode": getattr(runtime_mode, "name", str(runtime_mode)),
        "wrapper_mode": getattr(wrapper_mode, "name", str(wrapper_mode)),
        "is_draft_model": is_draft_model,
        "use_eagle": use_eagle,
        "batch": {
            "num_tokens": getattr(batch_descriptor, "num_tokens", None),
            "num_reqs": getattr(batch_descriptor, "num_reqs", None),
            "uniform": getattr(batch_descriptor, "uniform", None),
            "has_lora": getattr(batch_descriptor, "has_lora", None),
            "num_active_loras": getattr(
                batch_descriptor, "num_active_loras", None
            ),
        },
    }
    try:
        payload = (json.dumps(record, sort_keys=True) + "\n").encode()
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
        try:
            os.write(fd, payload)
        finally:
            os.close(fd)
    except (OSError, TypeError, ValueError):
        return
