# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the runnable PEARL custom-proposer bridge."""

from __future__ import annotations

import socket
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from vllm_ascend.spec_decode import get_spec_decode_method
from vllm_ascend.spec_decode.pearl import draft_server, launcher, remote_proposer
from vllm_ascend.spec_decode.pearl.remote_proposer import PearlRemoteProposer
from vllm_ascend.spec_decode.pearl.transport import (
    PearlTransportError,
    exchange_unix_message,
    receive_message,
    send_message,
)


class _FakeTPGroup:
    def __init__(self, rank_in_group: int, received_value=None) -> None:
        self.rank_in_group = rank_in_group
        self.received_value = received_value
        self.broadcast_calls: list[tuple[object, int]] = []

    def broadcast_object(self, value, src: int = 0):
        self.broadcast_calls.append((value, src))
        return value if self.rank_in_group == 0 else self.received_value


def _config(socket_path: str = "/tmp/pearl-test.sock"):
    return SimpleNamespace(
        additional_config={"pearl": {"draft_socket_path": socket_path}},
        speculative_config=SimpleNamespace(
            model="vllm_ascend.spec_decode.pearl.remote_proposer.PearlRemoteProposer",
            num_speculative_tokens=3,
        ),
    )


def test_custom_class_factory_loads_pearl_remote_proposer():
    proposer = get_spec_decode_method("custom_class", _config(), "npu", None)

    assert isinstance(proposer, PearlRemoteProposer)
    assert not hasattr(proposer, "load_model")
    assert not hasattr(proposer, "dummy_run")


def test_remote_proposer_sends_only_active_contexts_and_restores_batch_order(monkeypatch):
    proposer = PearlRemoteProposer(_config())
    tp_group = _FakeTPGroup(rank_in_group=0)
    captured_prompts: list[list[int]] = []
    monkeypatch.setattr(remote_proposer, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        proposer,
        "_request_candidates",
        lambda prompts: captured_prompts.extend(prompts) or [[91, 92]],
    )

    candidates = proposer.propose(
        sampled_token_ids=[[7], []],
        num_tokens_no_spec=np.array([3, 2], dtype=np.int32),
        token_ids_cpu=np.array([[10, 11, 12, -1], [20, 21, -1, -1]], dtype=np.int32),
    )

    assert captured_prompts == [[10, 11, 12]]
    assert candidates == [[91, 92], []]
    assert tp_group.broadcast_calls == [([[91, 92]], 0)]


def test_remote_proposer_nonleader_uses_target_tp_broadcast(monkeypatch):
    proposer = PearlRemoteProposer(_config())
    tp_group = _FakeTPGroup(rank_in_group=1, received_value=[[63]])
    monkeypatch.setattr(remote_proposer, "get_tp_group", lambda: tp_group)
    monkeypatch.setattr(
        proposer,
        "_request_candidates",
        lambda _prompts: pytest.fail("nonleader must not call the draft socket"),
    )

    candidates = proposer.propose(
        sampled_token_ids=[[7]],
        num_tokens_no_spec=np.array([2], dtype=np.int32),
        token_ids_cpu=np.array([[10, 11, -1]], dtype=np.int32),
    )

    assert candidates == [[63]]
    assert tp_group.broadcast_calls == [(None, 0)]


def test_remote_proposer_rejects_invalid_service_candidate():
    proposer = PearlRemoteProposer(_config())

    with pytest.raises(PearlTransportError, match="candidate-window length"):
        proposer._validate_candidate_row([1, 2, 3, 4])


def test_remote_proposer_uses_launcher_vocab_projection():
    config = _config()
    config.additional_config["pearl"].update(
        draft_vocab_size=3,
        target_vocab_size=5,
    )

    proposer = PearlRemoteProposer(config)

    assert proposer.pearl_vocab_projection is not None
    assert proposer.pearl_vocab_projection.draft_vocab_size == 3


def test_draft_server_handles_greedy_proposal_and_shutdown():
    response, should_shutdown = draft_server.handle_request(
        {
            "op": "propose",
            "prompt_token_ids": [[1, 2], [3]],
            "num_speculative_tokens": 2,
        },
        lambda prompts, _gamma: [[prompt[-1], 9] for prompt in prompts],
    )

    assert response == {"ok": True, "candidate_token_ids": [[2, 9], [3, 9]]}
    assert not should_shutdown
    assert draft_server.handle_request({"op": "shutdown"}, lambda _p, _g: []) == ({"ok": True}, True)


def test_draft_server_allows_an_empty_candidate_row_for_immediate_eos():
    response, should_shutdown = draft_server.handle_request(
        {
            "op": "propose",
            "prompt_token_ids": [[1, 2]],
            "num_speculative_tokens": 2,
        },
        lambda _prompts, _gamma: [[]],
    )

    assert response == {"ok": True, "candidate_token_ids": [[]]}
    assert not should_shutdown


def test_transport_round_trip_and_incomplete_message_error():
    sender, receiver = socket.socketpair()
    try:
        send_message(sender, {"op": "health"})
        assert receive_message(receiver) == {"op": "health"}
        sender.close()
        with pytest.raises(PearlTransportError, match="mid-message"):
            receive_message(receiver)
    finally:
        receiver.close()


def test_draft_server_serves_a_socket_request_and_cleans_up(tmp_path):
    socket_path = str(tmp_path / "draft.sock")
    server_thread = threading.Thread(
        target=draft_server._serve,
        args=(socket_path, lambda prompts, _gamma: [[prompt[-1]] for prompt in prompts]),
    )
    server_thread.start()
    try:
        response = _wait_for_socket_response(
            socket_path,
            {
                "op": "propose",
                "prompt_token_ids": [[1, 2]],
                "num_speculative_tokens": 2,
            },
        )
        assert response == {"ok": True, "candidate_token_ids": [[2]]}
        assert exchange_unix_message(socket_path, {"op": "shutdown"}, 1.0) == {"ok": True}
    finally:
        server_thread.join(timeout=3)

    assert not server_thread.is_alive()
    assert not (tmp_path / "draft.sock").exists()


def test_launcher_rejects_managed_target_options():
    with pytest.raises(ValueError, match="managed by the PEARL launcher"):
        launcher._reject_conflicting_target_args(["target", "--speculative-config", "{}"])


def test_launcher_device_environment_isolated_from_parent(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")

    environment = launcher._environment_with_devices("2,3")

    assert environment["ASCEND_RT_VISIBLE_DEVICES"] == "2,3"
    assert environment["HCCL_NPU_SOCKET_PORT_RANGE"] == "auto"


def test_launcher_starts_target_through_the_console_script():
    args = SimpleNamespace(
        draft_model="/models/draft",
        num_speculative_tokens=4,
        request_timeout_seconds=30.0,
        target_devices="1",
    )

    with (
        patch("vllm_ascend.spec_decode.pearl.launcher.subprocess.Popen") as popen,
        patch(
            "vllm_ascend.spec_decode.pearl.launcher.validate_model_pair",
            return_value=SimpleNamespace(draft_vocab_size=3, target_vocab_size=5),
        ),
    ):
        launcher._start_target_process(args, "/tmp/pearl.sock", ["/models/target"])

    command = popen.call_args.args[0]
    assert command[:2] == [str(Path(sys.executable).with_name("vllm")), "serve"]
    assert popen.call_args.kwargs["start_new_session"] is True


@pytest.mark.parametrize(
    ("target_args", "expected_model"),
    [
        (["/models/target"], "/models/target"),
        (["--model", "/models/target"], "/models/target"),
        (["--model=/models/target"], "/models/target"),
    ],
)
def test_launcher_resolves_target_model(target_args, expected_model):
    assert launcher._target_model_from_args(target_args) == expected_model


def test_launcher_requires_an_explicit_target_model():
    with pytest.raises(ValueError, match="target arguments must include a model path"):
        launcher._target_model_from_args(["--tensor-parallel-size", "2"])


def _wait_for_socket_response(socket_path: str, request: dict) -> dict:
    deadline = time.monotonic() + 3
    last_error: PearlTransportError | None = None
    while time.monotonic() < deadline:
        try:
            return exchange_unix_message(socket_path, request, 0.2)
        except PearlTransportError as error:
            last_error = error
            time.sleep(0.01)
    assert last_error is not None
    raise last_error
