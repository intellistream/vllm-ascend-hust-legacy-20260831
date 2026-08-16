# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from __future__ import annotations

import json
import os
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ATTENTION_DISPATCH_AUDIT_CONFIG_KEY = "attention_dispatch_audit"
ATTENTION_DISPATCH_AUDIT_SCHEMA_VERSION = 1
PAGED_ATTENTION_ACTION = "paged_attention"
FUSED_INFER_ATTENTION_ACTION = "fused_infer_attention"


@dataclass(frozen=True)
class AttentionDispatchIdentity:
    num_heads: int
    num_kv_heads: int
    head_size: int


@dataclass(frozen=True)
class AttentionDispatchAction:
    attention_state: str
    graph_capture: bool
    num_tokens: int
    identity: AttentionDispatchIdentity
    selected_action: str
    decision_reason: str
    schema_version: int = ATTENTION_DISPATCH_AUDIT_SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_attention_dispatch_action(
    *,
    attention_state: str,
    graph_capture: bool,
    num_tokens: int,
    identity: AttentionDispatchIdentity,
    paged_attention_eligible: bool,
    sliding_window_enabled: bool,
) -> AttentionDispatchAction:
    use_paged_attention = attention_state == "DecodeOnly" and paged_attention_eligible and not sliding_window_enabled
    if use_paged_attention:
        selected_action = PAGED_ATTENTION_ACTION
        reason = "decode_paged_attention_eligible"
    else:
        selected_action = FUSED_INFER_ATTENTION_ACTION
        if attention_state != "DecodeOnly":
            reason = "attention_state_requires_fused_infer_attention"
        elif sliding_window_enabled:
            reason = "sliding_window_requires_fused_infer_attention"
        else:
            reason = "paged_attention_not_eligible"
    return AttentionDispatchAction(
        attention_state=attention_state,
        graph_capture=graph_capture,
        num_tokens=num_tokens,
        identity=identity,
        selected_action=selected_action,
        decision_reason=reason,
    )


def _atomic_append_json_line(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
    descriptor = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
    try:
        written = os.write(descriptor, encoded)
        if written != len(encoded):
            raise OSError(f"partial attention dispatch audit write: {written}/{len(encoded)}")
    finally:
        os.close(descriptor)


class AttentionDispatchAudit:
    """Fail-closed, deduplicated action recorder for the real attention seam."""

    def __init__(
        self,
        *,
        run_id: str,
        identity: AttentionDispatchIdentity,
        sink: Callable[[Mapping[str, Any]], None],
    ) -> None:
        self._run_id = run_id
        self._identity = identity
        self._sink = sink
        self._seen_actions: set[str] = set()

    @classmethod
    def from_config(
        cls,
        additional_config: Any,
        *,
        identity: AttentionDispatchIdentity,
        sink: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> AttentionDispatchAudit | None:
        if not isinstance(additional_config, Mapping):
            return None
        config = additional_config.get(ATTENTION_DISPATCH_AUDIT_CONFIG_KEY)
        if config is None:
            return None
        if not isinstance(config, Mapping):
            raise ValueError("attention_dispatch_audit config must be a mapping")
        if config.get("enabled") is not True:
            return None

        run_id = config.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            raise ValueError("enabled attention dispatch audit requires a non-empty run_id")
        expected = config.get("expected_identity")
        if not isinstance(expected, Mapping):
            raise ValueError("enabled attention dispatch audit requires expected_identity")
        actual_identity = asdict(identity)
        missing = sorted(set(actual_identity) - set(expected))
        if missing:
            raise ValueError(f"attention dispatch expected_identity missing fields: {missing}")
        drift = {
            field: {"expected": expected[field], "actual": actual_value}
            for field, actual_value in actual_identity.items()
            if expected[field] != actual_value
        }
        if drift:
            raise ValueError(f"attention dispatch identity drift: {drift}")

        if sink is None:
            raw_output_path = config.get("output_path")
            if not isinstance(raw_output_path, str) or not raw_output_path:
                raise ValueError("enabled attention dispatch audit requires output_path")
            output_path = Path(raw_output_path)
            if not output_path.is_absolute():
                raise ValueError("attention dispatch audit output_path must be absolute")
            if not output_path.parent.is_dir():
                raise ValueError("attention dispatch audit output parent must already exist")
            sink = lambda payload: _atomic_append_json_line(output_path, payload)
        return cls(run_id=run_id, identity=identity, sink=sink)

    def record(self, action: AttentionDispatchAction) -> None:
        if action.identity != self._identity:
            raise ValueError(
                "attention dispatch identity changed after audit initialization: "
                f"expected={asdict(self._identity)}, actual={asdict(action.identity)}"
            )
        action_payload = action.to_dict()
        action_key = json.dumps(action_payload, sort_keys=True, separators=(",", ":"))
        if action_key in self._seen_actions:
            return
        payload = {
            "run_id": self._run_id,
            "process_id": os.getpid(),
            **action_payload,
        }
        self._sink(payload)
        self._seen_actions.add(action_key)
