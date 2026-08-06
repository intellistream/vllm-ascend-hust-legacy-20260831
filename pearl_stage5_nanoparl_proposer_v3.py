# PEARL_STAGE5_NANOPEARL_STRICT_PROPOSER_V1
# PEARL_STAGE5_NANOPEARL_COMMIT_STATE_PROPOSER_V3
# PEARL_STAGE5_NANOPEARL_EXPLICIT_VERIFY_PROPOSER_V1
# PEARL_STAGE5_NANOPEARL_DISCARD_REBASE_PROPOSER_V1
#!/usr/bin/env python3
"""nano-PEARL proposer v3 with real ModelRunner request IDs."""

from __future__ import annotations

import os
from typing import Any

from pearl_stage5_nanoparl_proposer_v1 import (
    PearlNanoPearlProposer as _PearlNanoPearlProposerV1,
)
from pearl_stage5_nanoparl_runtime_v1 import (
    DraftRequest,
    NanoPearlPrefetchController,
    trace_from_env,
)


class PearlNanoPearlProposer(_PearlNanoPearlProposerV1):
    """Use request_ids for stable HCCL slots while retaining v1 transport."""

    def __init__(self, vllm_config: Any) -> None:
        super().__init__(vllm_config)
        # Replace v1's ordinary controller with the rebase-aware controller.
        self._nano_controller.close()
        self._nano_controller = NanoPearlPrefetchController(
            self._request_batch_for_nano,
            trace=trace_from_env(),
            commit_batch=self._commit_batch_for_nano,
            rebase_batch=self._rebase_batch_for_nano,
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

    def _rebase_batch_for_nano(
        self, requests: list[dict[str, Any]]
    ) -> None:
        if os.environ.get("PEARL_STAGE5_NANOPEARL_STRICT", "0") == "1":
            raise RuntimeError(
                "nano-PEARL strict mode forbids rebase_batch commands"
            )
        with self._nano_io_lock:
            response = self._request(
                {"cmd": "rebase_batch", "requests": requests}
            )
        if response.get("status") != "result":
            raise RuntimeError(
                "Draft rebase_batch returned an invalid response: "
                f"{response!r}"
            )

    @staticmethod
    def _row_ids_fallback(token_ids_cpu: Any, row: int, count: int) -> list[int]:
        values = token_ids_cpu[row, :count]
        if hasattr(values, "tolist"):
            values = values.tolist()
        return [int(token_id) for token_id in values]

    @staticmethod
    def _as_request_id_list(request_ids: Any) -> list[str] | None:
        if request_ids is None:
            return None
        if hasattr(request_ids, "tolist"):
            request_ids = request_ids.tolist()
        if isinstance(request_ids, (str, bytes)):
            request_ids = [request_ids]
        return [str(value) for value in request_ids]

    def propose(
        self,
        *args: Any,
        request_ids: Any = None,
        **kwargs: Any,
    ) -> list[list[int]]:
        sampled_token_ids = kwargs.pop("sampled_token_ids", None)
        verify_results = kwargs.pop("verify_results", None)
        num_tokens_no_spec = kwargs.pop("num_tokens_no_spec", None)
        token_ids_cpu = kwargs.pop("token_ids_cpu", None)
        slot_mappings = kwargs.pop("slot_mappings", None)
        positional = list(args)
        if sampled_token_ids is None and positional:
            sampled_token_ids = positional.pop(0)
        if num_tokens_no_spec is None and positional:
            num_tokens_no_spec = positional.pop(0)
        if token_ids_cpu is None and positional:
            token_ids_cpu = positional.pop(0)
        if slot_mappings is None and positional:
            slot_mappings = positional.pop(0)
        del slot_mappings
        if kwargs or positional:
            raise TypeError(
                "unsupported nano-PEARL proposer arguments: "
                f"kwargs={sorted(kwargs)}, positional_count={len(positional)}"
            )
        if sampled_token_ids is None:
            raise TypeError("propose() is missing sampled_token_ids")
        if num_tokens_no_spec is None:
            raise TypeError("propose() is missing num_tokens_no_spec")
        if token_ids_cpu is None:
            raise TypeError("propose() is missing token_ids_cpu")

        stable_ids = self._as_request_id_list(request_ids)
        row_ids = getattr(self, "_row_ids", self._row_ids_fallback)
        output: list[list[int]] = [[] for _ in sampled_token_ids]
        requests: list[DraftRequest] = []
        active_rows: list[int] = []
        for row, raw_sampled in enumerate(sampled_token_ids):
            sampled = [int(x) for x in raw_sampled if int(x) >= 0]
            if not sampled:
                continue
            count = int(num_tokens_no_spec[row])
            prefix = row_ids(token_ids_cpu, row, count)
            if prefix[-len(sampled) :] != sampled:
                prefix.extend(sampled)
            if stable_ids is not None and row < len(stable_ids):
                request_id = stable_ids[row]
            else:
                request_id = f"target-row-{row}"
            requests.append(
                DraftRequest(
                    request_id=request_id,
                    prefix_token_ids=tuple(prefix),
                    gamma=self.gamma,
                )
            )
            active_rows.append(row)

        if not requests:
            return output

        normalized_verify_results = None
        if verify_results is not None:
            normalized_verify_results = []
            for raw in verify_results:
                item = dict(raw) if isinstance(raw, dict) else raw
                if isinstance(item, dict):
                    row_index = item.get("row_index")
                    if row_index is not None and stable_ids is not None:
                        row_index = int(row_index)
                        if 0 <= row_index < len(stable_ids):
                            item["request_id"] = stable_ids[row_index]
                    normalized_verify_results.append(item)
                else:
                    normalized_verify_results.append(item)
        results = self._nano_controller.get_or_request(
            requests, verify_results=normalized_verify_results
        )
        for row, result in zip(active_rows, results):
            output[row] = [int(x) for x in result.draft_token_ids[: self.gamma]]
        return output
