# PEARL_STAGE5_BATCH_GT1_V1
#!/usr/bin/env python3
"""Target-side vLLM custom proposer for the Stage-5 nano-PEARL bridge."""

from __future__ import annotations

import json
import os
import socket
import threading
import time
from typing import Any


class PearlExternalProposer:
    """Ask the resident Draft process for token IDs over a Unix socket.

    The HUST custom proposer API does not pass request IDs.  This first
    integration therefore requires ``max_num_seqs=1`` and uses one persistent
    Draft sequence.  The prefix is reconstructed from ``token_ids_cpu`` plus
    the token sampled in the current Target step.
    """

    def __init__(self, vllm_config: Any) -> None:
        spec_config = vllm_config.speculative_config
        self.gamma = int(spec_config.num_speculative_tokens)
        self.socket_path = os.environ.get("PEARL_DRAFT_SOCKET")
        if not self.socket_path:
            raise RuntimeError("PEARL_DRAFT_SOCKET is not set")
        self.timeout = float(os.environ.get("PEARL_DRAFT_SOCKET_TIMEOUT", "120"))
        self._sock: socket.socket | None = None
        self._reader = None
        self._lock = threading.RLock()

    def load_model(self, model: Any = None) -> None:
        """Compatibility hook for vLLM-HUST's ModelRunner.

        This proposer is model-free: the Draft model lives in the separate
        persistent Draft worker and is queried through AF_UNIX.  The Target
        ModelRunner nevertheless calls ``load_model(self.model)`` for every
        proposer implementation, so this method intentionally does nothing.
        """
        return None

    def dummy_run(self, *args: Any, **kwargs: Any) -> None:
        """Compatibility hook for vLLM-HUST's startup memory profiling.

        The Draft model is resident in the separate Draft worker.  The
        external proposer therefore has no model forward or KV-cache work to
        perform during Target's dummy/profile run.
        """
        return None

    def _connect(self) -> None:
        deadline = time.monotonic() + self.timeout
        last_error: Exception | None = None
        while time.monotonic() < deadline:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            try:
                sock.settimeout(max(1.0, deadline - time.monotonic()))
                sock.connect(self.socket_path)
                self._sock = sock
                self._reader = sock.makefile("r", encoding="utf-8")
                return
            except OSError as exc:
                last_error = exc
                sock.close()
                time.sleep(0.1)
        raise TimeoutError(
            f"timed out connecting to Draft socket {self.socket_path}; "
            f"last_error={last_error!r}"
        )

    def _close(self) -> None:
        if self._reader is not None:
            try:
                self._reader.close()
            except Exception:
                pass
            self._reader = None
        if self._sock is not None:
            try:
                self._sock.close()
            except Exception:
                pass
            self._sock = None

    def _request(self, message: dict[str, Any]) -> dict[str, Any]:
        if self._sock is None or self._reader is None:
            self._connect()
        assert self._sock is not None and self._reader is not None
        try:
            self._sock.sendall(
                (json.dumps(message, ensure_ascii=False) + "\n").encode("utf-8")
            )
            line = self._reader.readline()
            if not line:
                raise ConnectionError("Draft socket closed")
            response = json.loads(line)
        except Exception:
            self._close()
            raise
        if response.get("status") == "error":
            raise RuntimeError(response.get("error", "Draft proposer error"))
        return response

    @staticmethod
    def _row_ids(token_ids_cpu: Any, row: int, count: int) -> list[int]:
        values = token_ids_cpu[row, :count]
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [int(token_id) for token_id in values]

    def propose(
        self,
        sampled_token_ids: list[list[int]],
        num_tokens_no_spec: Any,
        token_ids_cpu: Any,
        slot_mappings: Any = None,
        request_ids: Any = None,
    ) -> list[list[int]]:
        """Return one Draft proposal list for every Target batch row.

        The custom proposer API does not carry request IDs by default, so
        the Ascend model runner passes ``input_batch.req_ids`` explicitly.
        The external Draft worker keeps one persistent state per ID.
        """
        batch_size = len(sampled_token_ids)
        if batch_size == 0:
            return []

        if request_ids is None:
            row_request_ids = [f"row-{index}" for index in range(batch_size)]
        else:
            raw_ids = request_ids.tolist() if hasattr(request_ids, "tolist") else list(request_ids)
            row_request_ids = [str(value) for value in raw_ids[:batch_size]]
            if len(row_request_ids) < batch_size:
                row_request_ids.extend(
                    f"row-{index}" for index in range(len(row_request_ids), batch_size)
                )

        def scalar_at(values: Any, row: int) -> int:
            value = values[row] if hasattr(values, "__getitem__") else values
            if hasattr(value, "item"):
                value = value.item()
            return int(value)

        def row_ids(row: int, count: int) -> list[int]:
            values = token_ids_cpu[row, :count]
            if hasattr(values, "tolist"):
                values = values.tolist()
            return [int(token_id) for token_id in values]

        proposals = [[] for _ in range(batch_size)]
        requests: list[dict[str, Any]] = []
        for row, sampled_row in enumerate(sampled_token_ids):
            sampled = [
                int(token_id)
                for token_id in sampled_row
                if int(token_id) >= 0
            ]
            # During initial prefill there is no just-sampled token for
            # this row.  It must still occupy a result slot, but does not
            # need a Draft request yet.
            if not sampled:
                continue

            count = scalar_at(num_tokens_no_spec, row)
            prefix = row_ids(row, count)
            if prefix[-len(sampled) :] != sampled:
                prefix.extend(sampled)
            requests.append(
                {
                    "request_id": row_request_ids[row],
                    "prefix_token_ids": prefix,
                    "gamma": self.gamma,
                }
            )

        if not requests:
            return proposals

        print(
            "[target proposer] "
            f"batch_size={batch_size} request_count={len(requests)} "
            f"request_ids={[item['request_id'] for item in requests]} "
            f"gamma={self.gamma}",
            flush=True,
        )
        with self._lock:
            response = self._request(
                {
                    "cmd": "draft_batch",
                    "requests": requests,
                }
            )

        results = response.get("results")
        if not isinstance(results, list):
            raise RuntimeError(
                "Draft batch response is missing a list-valued 'results'"
            )
        result_by_id = {
            str(item.get("request_id")): item
            for item in results
            if isinstance(item, dict) and item.get("request_id") is not None
        }
        for row, request in enumerate(requests):
            item = result_by_id.get(str(request["request_id"]))
            if item is None:
                raise RuntimeError(
                    "Draft batch response omitted request "
                    f"{request['request_id']!r}"
                )
            draft_ids = item.get("draft_token_ids", [])
            if not isinstance(draft_ids, list):
                raise RuntimeError(
                    "Draft batch response has non-list draft_token_ids for "
                    f"{request['request_id']!r}"
                )
            target_row = row_request_ids.index(str(request["request_id"]))
            proposals[target_row] = [
                int(token_id) for token_id in draft_ids[: self.gamma]
            ]
        return proposals
