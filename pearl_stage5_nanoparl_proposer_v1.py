# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_PROPOSER_V1
#!/usr/bin/env python3
"""Opt-in nano-PEARL proposer wrapper for the existing Stage-5 proposer.

The wrapper leaves the user's existing proposer and KV fixes untouched.  It
adds only the PRE_VERIFY/POST_VERIFY prefetch controller and uses the existing
Unix/HCCL bridge commands (``draft`` for one row and ``draft_batch`` for a
batch).  The Target ModelRunner remains the source of truth for verification.
"""

from __future__ import annotations

import os
import threading
from typing import Any

from pearl_stage5_nanoparl_runtime_v1 import (
    DraftRequest,
    NanoPearlPrefetchController,
    VerifyResult,
    trace_from_env,
)
from pearl_stage5_proposer import PearlExternalProposer


class PearlNanoPearlProposer(PearlExternalProposer):
    """Existing Stage-5 proposer plus optimistic POST-VERIFY prefetch."""

    def __init__(self, vllm_config: Any) -> None:
        super().__init__(vllm_config)
        self._nano_io_lock = threading.RLock()
        self._nano_controller = NanoPearlPrefetchController(
            self._request_batch_for_nano,
            trace=trace_from_env(),
            commit_batch=self._commit_batch_for_nano,
        )

    def _commit_batch_for_nano(
        self, updates: list[dict[str, Any]]
    ) -> None:
        """Send one ordered accepted/valid-length commit to Draft."""
        if not updates:
            return
        with self._nano_io_lock:
            response = self._request(
                {"cmd": "commit_batch", "updates": updates}
            )
        if response.get("status") != "result":
            raise RuntimeError(
                "Draft commit_batch returned an invalid response: "
                f"{response!r}"
            )

    def _request_batch_for_nano(self, requests: list[dict[str, Any]]) -> list[list[int]]:
        # Base _request owns socket setup and JSON framing.  Serialize the
        # complete transaction because the stable-slot HCCL bridge is ordered.
        with self._nano_io_lock:
            if len(requests) == 1:
                request = requests[0]
                response = self._request({"cmd": "draft", **request})
                return [[int(x) for x in response.get("draft_token_ids", [])]]

            response = self._request(
                {"cmd": "draft_batch", "requests": requests}
            )
            raw_results = response.get("results")
            if not isinstance(raw_results, list):
                raise RuntimeError("Draft batch response has no results list")
            by_id = {
                str(item.get("request_id")): item
                for item in raw_results
                if isinstance(item, dict)
            }
            rows: list[list[int]] = []
            for request in requests:
                item = by_id.get(str(request["request_id"]))
                if item is None:
                    raise RuntimeError(
                        f"Draft batch response omitted {request['request_id']!r}"
                    )
                rows.append([int(x) for x in item.get("draft_token_ids", [])])
            return rows

    @staticmethod
    def _row_ids_fallback(token_ids_cpu: Any, row: int, count: int) -> list[int]:
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
    ) -> list[list[int]]:
        del slot_mappings
        output: list[list[int]] = [[] for _ in sampled_token_ids]
        requests: list[DraftRequest] = []
        rows: list[int] = []
        row_ids = getattr(self, "_row_ids", self._row_ids_fallback)

        for row, raw_sampled in enumerate(sampled_token_ids):
            sampled = [int(x) for x in raw_sampled if int(x) >= 0]
            if not sampled:
                continue
            count = int(num_tokens_no_spec[row])
            prefix = row_ids(token_ids_cpu, row, count)
            if prefix[-len(sampled) :] != sampled:
                prefix.extend(sampled)
            requests.append(
                DraftRequest(
                    request_id=f"target-row-{row}",
                    prefix_token_ids=tuple(prefix),
                    gamma=self.gamma,
                )
            )
            rows.append(row)

        if not requests:
            return output

        results = self._nano_controller.get_or_request(requests)
        for row, result in zip(rows, results):
            output[row] = [int(x) for x in result.draft_token_ids[: self.gamma]]
        return output

    def notify_verify(self, results: list[VerifyResult]) -> None:
        self._nano_controller.notify_verify(results)

    def shutdown(self) -> None:
        self._nano_controller.close()
        close = getattr(self, "_close", None)
        if callable(close):
            close()

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

