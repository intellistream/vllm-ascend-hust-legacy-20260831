import ast
import json
from dataclasses import replace
from pathlib import Path

import pytest

from vllm_ascend.attention.dispatch_audit import (
    FUSED_INFER_ATTENTION_ACTION,
    PAGED_ATTENTION_ACTION,
    AttentionDispatchAudit,
    AttentionDispatchIdentity,
    build_attention_dispatch_action,
)

IDENTITY = AttentionDispatchIdentity(num_heads=40, num_kv_heads=8, head_size=128)
ATTENTION_V1 = Path(__file__).parents[3] / "vllm_ascend" / "attention" / "attention_v1.py"


def _config(**overrides):
    audit = {
        "enabled": True,
        "run_id": "m0-action-gate",
        "expected_identity": {
            "num_heads": 40,
            "num_kv_heads": 8,
            "head_size": 128,
        },
    }
    audit.update(overrides)
    return {"attention_dispatch_audit": audit}


def test_dispatch_action_covers_paged_fused_and_graph_capture() -> None:
    paged = build_attention_dispatch_action(
        attention_state="DecodeOnly",
        graph_capture=False,
        num_tokens=8,
        identity=IDENTITY,
        paged_attention_eligible=True,
        sliding_window_enabled=False,
    )
    fused = build_attention_dispatch_action(
        attention_state="PrefillNoCache",
        graph_capture=False,
        num_tokens=512,
        identity=IDENTITY,
        paged_attention_eligible=False,
        sliding_window_enabled=False,
    )
    graph = replace(paged, graph_capture=True)

    assert paged.selected_action == PAGED_ATTENTION_ACTION
    assert fused.selected_action == FUSED_INFER_ATTENTION_ACTION
    assert graph.graph_capture is True


def test_disabled_or_absent_audit_preserves_native_default() -> None:
    assert AttentionDispatchAudit.from_config({}, identity=IDENTITY) is None
    assert (
        AttentionDispatchAudit.from_config({"attention_dispatch_audit": {"enabled": False}}, identity=IDENTITY) is None
    )


def test_enabled_audit_fails_closed_on_missing_config_or_identity_drift() -> None:
    with pytest.raises(ValueError, match="run_id"):
        AttentionDispatchAudit.from_config(
            {"attention_dispatch_audit": {"enabled": True}},
            identity=IDENTITY,
            sink=lambda _payload: None,
        )
    with pytest.raises(ValueError, match="identity drift"):
        AttentionDispatchAudit.from_config(
            _config(expected_identity={"num_heads": 32, "num_kv_heads": 8, "head_size": 128}),
            identity=IDENTITY,
            sink=lambda _payload: None,
        )


def test_audit_records_schema_complete_unique_actions_and_rejects_late_drift() -> None:
    records = []
    audit = AttentionDispatchAudit.from_config(_config(), identity=IDENTITY, sink=records.append)
    assert audit is not None
    action = build_attention_dispatch_action(
        attention_state="DecodeOnly",
        graph_capture=True,
        num_tokens=8,
        identity=IDENTITY,
        paged_attention_eligible=True,
        sliding_window_enabled=False,
    )
    audit.record(action)
    audit.record(action)

    assert len(records) == 1
    assert records[0]["schema_version"] == 1
    assert records[0]["run_id"] == "m0-action-gate"
    assert records[0]["identity"] == {
        "num_heads": 40,
        "num_kv_heads": 8,
        "head_size": 128,
    }
    assert records[0]["selected_action"] == PAGED_ATTENTION_ACTION
    json.dumps(records[0])

    with pytest.raises(ValueError, match="identity changed"):
        audit.record(replace(action, identity=replace(IDENTITY, head_size=64)))


def test_file_sink_requires_absolute_existing_parent(tmp_path) -> None:
    with pytest.raises(ValueError, match="must be absolute"):
        AttentionDispatchAudit.from_config(_config(output_path="relative/actions.jsonl"), identity=IDENTITY)
    output = tmp_path / "actions.jsonl"
    audit = AttentionDispatchAudit.from_config(_config(output_path=str(output)), identity=IDENTITY)
    assert audit is not None
    action = build_attention_dispatch_action(
        attention_state="PrefillNoCache",
        graph_capture=False,
        num_tokens=64,
        identity=IDENTITY,
        paged_attention_eligible=False,
        sliding_window_enabled=False,
    )
    audit.record(action)
    assert json.loads(output.read_text()) == {
        "attention_state": "PrefillNoCache",
        "decision_reason": "attention_state_requires_fused_infer_attention",
        "graph_capture": False,
        "identity": {"head_size": 128, "num_heads": 40, "num_kv_heads": 8},
        "num_tokens": 64,
        "process_id": records_process_id(output),
        "run_id": "m0-action-gate",
        "schema_version": 1,
        "selected_action": FUSED_INFER_ATTENTION_ACTION,
    }


def test_actual_backend_forward_impl_keeps_signature_and_records_before_action() -> None:
    module = ast.parse(ATTENTION_V1.read_text())
    backend = next(
        node for node in module.body if isinstance(node, ast.ClassDef) and node.name == "AscendAttentionBackendImpl"
    )
    forward_impl = next(
        node for node in backend.body if isinstance(node, ast.FunctionDef) and node.name == "forward_impl"
    )
    assert [argument.arg for argument in forward_impl.args.args] == [
        "self",
        "query",
        "key",
        "value",
        "kv_cache",
        "attn_metadata",
        "output",
    ]

    calls = [node for node in ast.walk(forward_impl) if isinstance(node, ast.Call)]

    def call_line(name):
        return min(
            node.lineno
            for node in calls
            if (isinstance(node.func, ast.Name) and node.func.id == name)
            or (isinstance(node.func, ast.Attribute) and node.func.attr == name)
        )

    record_line = call_line("record")
    assert call_line("build_attention_dispatch_action") < record_line
    assert record_line < call_line("forward_paged_attention")
    assert record_line < call_line("forward_fused_infer_attention")

    audit_branch = next(
        statement
        for statement in forward_impl.body
        if isinstance(statement, ast.If)
        and any(
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "build_attention_dispatch_action"
            for node in ast.walk(statement)
        )
    )
    assert "self.dispatch_audit is not None" in ast.unparse(audit_branch.test)
    native_dispatch = next(
        statement
        for statement in forward_impl.body
        if isinstance(statement, ast.If) and ast.unparse(statement.test) == "use_paged_attention"
    )
    assert any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "forward_paged_attention"
        for node in ast.walk(native_dispatch.body[0])
    )


def records_process_id(path):
    return json.loads(path.read_text())["process_id"]
