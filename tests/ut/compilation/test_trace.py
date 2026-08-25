import importlib.util
import json
import stat
from dataclasses import dataclass
from enum import Enum
from pathlib import Path


TRACE_MODULE = (
    Path(__file__).parents[3] / "vllm_ascend" / "compilation" / "trace.py"
)
SPEC = importlib.util.spec_from_file_location("ascend_compilation_trace", TRACE_MODULE)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
emit_aclgraph_dispatch = MODULE.emit_aclgraph_dispatch


class Mode(Enum):
    NONE = 0
    FULL = 1


@dataclass
class Descriptor:
    num_tokens: int
    num_reqs: int | None
    uniform: bool
    has_lora: bool = False
    num_active_loras: int = 0


def test_trace_is_opt_in(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.delenv("VLLM_COMPILATION_TRACE_PATH", raising=False)

    emit_aclgraph_dispatch(
        action="replay",
        batch_descriptor=Descriptor(16, 8, True),
        runtime_mode=Mode.FULL,
        wrapper_mode=Mode.FULL,
        is_draft_model=True,
        use_eagle=True,
    )

    assert not path.exists()


def test_trace_preserves_dispatch_identity(tmp_path, monkeypatch):
    path = tmp_path / "trace.jsonl"
    monkeypatch.setenv("VLLM_COMPILATION_TRACE_PATH", str(path))

    emit_aclgraph_dispatch(
        action="replay",
        batch_descriptor=Descriptor(16, 8, True),
        runtime_mode=Mode.FULL,
        wrapper_mode=Mode.FULL,
        is_draft_model=True,
        use_eagle=True,
    )

    row = json.loads(path.read_text())
    assert row["event"] == "ascend_aclgraph_dispatch"
    assert row["action"] == "replay"
    assert row["runtime_mode"] == "FULL"
    assert row["wrapper_mode"] == "FULL"
    assert row["is_draft_model"] is True
    assert row["use_eagle"] is True
    assert row["batch"] == {
        "num_tokens": 16,
        "num_reqs": 8,
        "uniform": True,
        "has_lora": False,
        "num_active_loras": 0,
    }
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_trace_failure_is_non_fatal(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_COMPILATION_TRACE_PATH", str(tmp_path))

    emit_aclgraph_dispatch(
        action="bypass",
        batch_descriptor=Descriptor(3, None, False),
        runtime_mode=Mode.NONE,
        wrapper_mode=Mode.FULL,
        is_draft_model=False,
        use_eagle=False,
    )
