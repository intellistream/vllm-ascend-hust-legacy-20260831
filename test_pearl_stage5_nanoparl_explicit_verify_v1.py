#!/usr/bin/env python3
"""CPU-only contract test for the explicit nano-PEARL verify path."""

from __future__ import annotations

from pearl_stage5_nanoparl_runtime_v1 import (
    DraftRequest,
    NanoPearlPrefetchController,
    VerifyResult,
)


def main() -> None:
    calls: list[list[str]] = []

    def request_batch(rows: list[dict]) -> list[list[int]]:
        calls.append([str(row["request_id"]) for row in rows])
        return [[100 + len(row["prefix_token_ids"]), 200 + len(row["prefix_token_ids"])] for row in rows]

    trace: list[str] = []
    controller = NanoPearlPrefetchController(request_batch, trace=trace.append)
    try:
        first = controller.get_or_request([DraftRequest("a", (1, 2), 2)])[0]
        assert first.draft_token_ids == (102, 202)

        # Target explicitly accepted both draft tokens.  Its bonus token 104
        # is the first token of the pending continuation, so consume only the
        # remaining suffix instead of issuing a new Draft request for row a.
        consumed = controller.get_or_request(
            [DraftRequest("a", (1, 2, 102, 202, 104), 2)],
            verify_results=[
                VerifyResult(
                    "a", accepted_len=2, draft_len=2,
                    replacement_token_id=104,
                )
            ],
        )[0]
        assert consumed.draft_token_ids == (204,)
        assert any("verify_result_received" in line for line in trace)
        assert any("consume_rows=1" in line for line in trace)

        # A partial rejection is authoritative even if a prefix heuristic
        # could accidentally look compatible.  The row must be refreshed from
        # the corrected Target prefix.
        refreshed = controller.get_or_request(
            [DraftRequest("a", (1, 2, 102, 202, 999), 2)],
            verify_results=[
                VerifyResult(
                    "a", accepted_len=0, draft_len=2,
                    replacement_token_id=999,
                )
            ],
        )[0]
        assert refreshed.prefix_token_ids == (1, 2, 102, 202, 999)
        assert any("explicit_rebase_rows=1" in line for line in trace)
    finally:
        controller.close()

    assert calls == [["a"], ["a"], ["a"], ["a"], ["a"]], calls
    print("explicit nano-PEARL verify contract: PASS")
    for line in trace:
        print(line)


if __name__ == "__main__":
    main()
