#!/usr/bin/env python3
"""Synchronous batch proposer used as the Stage-5 AC control."""

from __future__ import annotations

from typing import Any

from pearl_stage5_proposer import PearlExternalProposer


class PearlSerialBatchProposer(PearlExternalProposer):
    """Keep stable request IDs but issue exactly one synchronous batch call."""

    @staticmethod
    def _row_ids(token_ids_cpu: Any, row: int, count: int) -> list[int]:
        values = token_ids_cpu[row, :count]
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [int(token_id) for token_id in values]

    @staticmethod
    def _request_id_list(request_ids: Any) -> list[str] | None:
        if request_ids is None:
            return None
        if hasattr(request_ids, "tolist"):
            request_ids = request_ids.tolist()
        if isinstance(request_ids, (str, bytes)):
            request_ids = [request_ids]
        return [str(value) for value in request_ids]

    def _request_batch(
        self,
        requests: list[dict[str, Any]],
    ) -> list[list[int]]:
        if len(requests) == 1:
            response = self._request({"cmd": "draft", **requests[0]})
            return [
                [int(x) for x in response.get("draft_token_ids", [])]
            ]

        response = self._request(
            {"cmd": "draft_batch", "requests": requests}
        )
        raw_results = response.get("results")
        if not isinstance(raw_results, list):
            raise RuntimeError(
                "synchronous Draft batch response has no results list"
            )
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
                    "synchronous Draft batch response omitted "
                    f"{request['request_id']!r}"
                )
            rows.append(
                [int(x) for x in item.get("draft_token_ids", [])]
            )
        return rows

    def propose(
        self,
        *args: Any,
        request_ids: Any = None,
        **kwargs: Any,
    ) -> list[list[int]]:
        sampled_token_ids = kwargs.pop("sampled_token_ids", None)
        num_tokens_no_spec = kwargs.pop("num_tokens_no_spec", None)
        token_ids_cpu = kwargs.pop("token_ids_cpu", None)
        kwargs.pop("slot_mappings", None)
        positional = list(args)
        if sampled_token_ids is None and positional:
            sampled_token_ids = positional.pop(0)
        if num_tokens_no_spec is None and positional:
            num_tokens_no_spec = positional.pop(0)
        if token_ids_cpu is None and positional:
            token_ids_cpu = positional.pop(0)
        if positional or kwargs:
            raise TypeError(
                "unsupported serial proposer arguments: "
                f"kwargs={sorted(kwargs)}, positional={len(positional)}"
            )
        if sampled_token_ids is None:
            raise TypeError("propose() is missing sampled_token_ids")
        if num_tokens_no_spec is None:
            raise TypeError("propose() is missing num_tokens_no_spec")
        if token_ids_cpu is None:
            raise TypeError("propose() is missing token_ids_cpu")

        stable_ids = self._request_id_list(request_ids)
        requests: list[dict[str, Any]] = []
        active_rows: list[int] = []
        output: list[list[int]] = [[] for _ in sampled_token_ids]
        for row, raw_sampled in enumerate(sampled_token_ids):
            sampled = [int(x) for x in raw_sampled if int(x) >= 0]
            if not sampled:
                continue
            count = int(num_tokens_no_spec[row])
            prefix = self._row_ids(token_ids_cpu, row, count)
            if prefix[-len(sampled):] != sampled:
                prefix.extend(sampled)
            request_id = (
                stable_ids[row]
                if stable_ids is not None and row < len(stable_ids)
                else f"target-row-{row}"
            )
            requests.append(
                {
                    "request_id": request_id,
                    "prefix_token_ids": prefix,
                    "gamma": self.gamma,
                }
            )
            active_rows.append(row)

        if not requests:
            return output
        rows = self._request_batch(requests)
        for row, draft_ids in zip(active_rows, rows):
            output[row] = [int(x) for x in draft_ids[: self.gamma]]
        return output


__all__ = ["PearlSerialBatchProposer"]
