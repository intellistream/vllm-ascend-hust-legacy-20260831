#!/usr/bin/env python3
"""Compatibility wrapper for the vLLM-HUST proposer API with request_ids."""

from __future__ import annotations

from typing import Any

from pearl_stage5_nanoparl_proposer_v1 import (
    PearlNanoPearlProposer as _PearlNanoPearlProposerV1,
)


class PearlNanoPearlProposer(_PearlNanoPearlProposerV1):
    """Accept both the older and newer HUST proposer call signatures.

    The current vLLM-HUST ModelRunner passes ``request_ids`` as a keyword.
    The v1 wrapper predated that API addition and consequently failed before
    reaching the HCCL bridge.  Extra future-compatible keyword arguments are
    accepted and intentionally ignored here; prefix/token state remains the
    source of truth until the ModelRunner verify-result hook is connected.
    """

    def propose(
        self,
        *args: Any,
        request_ids: Any = None,
        **kwargs: Any,
    ) -> list[list[int]]:
        if request_ids is not None:
            if hasattr(request_ids, "tolist"):
                request_ids = request_ids.tolist()
            if isinstance(request_ids, (str, bytes)):
                request_ids = [request_ids]
            self._active_request_ids = [str(x) for x in request_ids]
        else:
            self._active_request_ids = None

        sampled_token_ids = kwargs.pop("sampled_token_ids", None)
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

        if kwargs or positional:
            unexpected = sorted(kwargs)
            raise TypeError(
                "unsupported nano-PEARL proposer arguments: "
                f"kwargs={unexpected}, positional_count={len(positional)}"
            )
        if sampled_token_ids is None:
            raise TypeError("propose() is missing sampled_token_ids")
        if num_tokens_no_spec is None:
            raise TypeError("propose() is missing num_tokens_no_spec")
        if token_ids_cpu is None:
            raise TypeError("propose() is missing token_ids_cpu")

        try:
            return super().propose(
                sampled_token_ids,
                num_tokens_no_spec,
                token_ids_cpu,
                slot_mappings,
            )
        finally:
            self._active_request_ids = None
