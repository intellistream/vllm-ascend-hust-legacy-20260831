from __future__ import annotations

import time

from pearl_stage5_nanoparl_runtime_v1 import (
    DraftRequest,
    NanoPearlPrefetchController,
    PearlMode,
    VerifyResult,
)


def main() -> None:
    calls: list[tuple[tuple[int, ...], ...]] = []
    trace: list[str] = []

    def request_batch(rows: list[dict]) -> list[list[int]]:
        prefixes = tuple(tuple(row["prefix_token_ids"]) for row in rows)
        calls.append(prefixes)
        # Make the background request observable without depending on timing.
        time.sleep(0.005)
        return [[100 + len(prefix), 200 + len(prefix)] for prefix in prefixes]

    controller = NanoPearlPrefetchController(request_batch, trace=trace.append)
    try:
        first = controller.get_or_request(
            [DraftRequest("row-0", (1, 2, 3), 2)]
        )
        assert controller.mode is PearlMode.POST_VERIFY
        assert first[0].draft_token_ids == (103, 203)

        # Extra token 999 models Target's bonus token after all draft tokens
        # were accepted.  It remains compatible with the optimistic prefix.
        second = controller.get_or_request(
            [DraftRequest("row-0", (1, 2, 3, 103, 203, 999), 2)]
        )
        assert second[0].prefix_token_ids == (1, 2, 3, 103, 203)
        assert "post_verify consume_prefetch" in " ".join(trace)

        # Divergence before the optimistic prefix models a rejection.  The
        # pending post-verify result is discarded and a fresh PRE request is
        # issued from the corrected prefix.
        controller.get_or_request(
            [DraftRequest("row-0", (1, 2, 3, 103, 777), 2)]
        )
        assert "pre_verify discard_prefetch" in " ".join(trace)

        controller.notify_verify(
            [VerifyResult("row-0", accepted_len=1, draft_len=2)]
        )
        assert controller.mode is PearlMode.PRE_VERIFY
    finally:
        controller.close()

    assert len(calls) >= 3

    # The same controller is batch-shaped; each row is matched independently
    # by stable request_id and optimistic prefix.
    batch_calls: list[int] = []

    def batch_request(rows: list[dict]) -> list[list[int]]:
        batch_calls.append(len(rows))
        return [[10 + i, 20 + i] for i, _ in enumerate(rows)]

    batch_controller = NanoPearlPrefetchController(batch_request)
    try:
        batch_controller.get_or_request(
            [
                DraftRequest("row-0", (1, 2), 2),
                DraftRequest("row-1", (3, 4), 2),
            ]
        )
        batch_result = batch_controller.get_or_request(
            [
                DraftRequest("row-0", (1, 2, 10, 20, 30), 2),
                DraftRequest("row-1", (3, 4, 11, 21, 31), 2),
            ]
        )
        assert [row.draft_token_ids for row in batch_result] == [(10, 20), (11, 21)]
        assert batch_controller.mode is PearlMode.POST_VERIFY
        assert batch_calls[0] == 2
    finally:
        batch_controller.close()

    print("nano-pearl runtime test: PASS")
    print("calls:", calls)
    print("trace:")
    for item in trace:
        print("  ", item)


if __name__ == "__main__":
    main()
