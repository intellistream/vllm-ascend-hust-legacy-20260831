import json
import os
import time
from pathlib import Path
from typing import Any


STATS_PATH_ENV = "ASCEND_HOST_GATHER_STATS_PATH"


def record_host_gather_event(component: str, event: str, **payload: Any) -> None:
    stats_path = os.getenv(STATS_PATH_ENV)
    if not stats_path:
        return

    row = {
        "component": component,
        "event": event,
        "pid": os.getpid(),
        "time_s": time.time(),
        **payload,
    }
    try:
        path = Path(stats_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, sort_keys=True, ensure_ascii=True) + "\n")
    except Exception:
        # Statistics must never change the transfer path behavior.
        return
