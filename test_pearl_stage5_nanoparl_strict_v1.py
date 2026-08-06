#!/usr/bin/env python3
"""CPU-only contract test for nano-PEARL strict-mode guards."""

from __future__ import annotations

import os
from pathlib import Path

from pearl_stage5_nanoparl_runtime_v1 import (
    DraftRequest,
    NanoPearlPrefetchController,
    VerifyResult,
)
from pearl_stage5_draft import PersistentDraftEngine


class scoped_env:
    def __init__(self, **values: str) -> None:
        self.values = values
        self.old: dict[str, str | None] = {}

    def __enter__(self) -> None:
        for key, value in self.values.items():
            self.old[key] = os.environ.get(key)
            os.environ[key] = value

    def __exit__(self, *_exc: object) -> None:
        for key, value in self.old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def assert_raises_contains(fn, needle: str) -> None:
    try:
        fn()
    except RuntimeError as exc:
        if needle not in str(exc):
            raise AssertionError(f"{needle!r} not in {exc!r}") from exc
    else:
        raise AssertionError(f"expected RuntimeError containing {needle!r}")


def request_batch(rows: list[dict]) -> list[list[int]]:
    return [[100 + len(row["prefix_token_ids"])] for row in rows]


def test_strict_requires_commit_batch() -> None:
    with scoped_env(PEARL_STAGE5_NANOPEARL_STRICT="1"):
        controller = NanoPearlPrefetchController(request_batch)
        try:
            controller.get_or_request([DraftRequest("row-0", (1, 2), 1)])
            assert_raises_contains(
                lambda: controller.get_or_request(
                    [DraftRequest("row-0", (1, 2, 102), 1)],
                    verify_results=[
                        VerifyResult("row-0", accepted_len=0, draft_len=1)
                    ],
                ),
                "strict mode requires commit_batch",
            )
        finally:
            controller.close()


def test_strict_forbids_discard_rebase() -> None:
    rebases: list[list[dict]] = []
    with scoped_env(PEARL_STAGE5_NANOPEARL_STRICT="1"):
        controller = NanoPearlPrefetchController(
            request_batch,
            rebase_batch=lambda rows: rebases.append(rows),
        )
        try:
            controller.get_or_request([DraftRequest("row-0", (1, 2), 1)])
            assert_raises_contains(
                lambda: controller.get_or_request(
                    [DraftRequest("row-0", (1, 2, 777), 1)]
                ),
                "strict mode forbids discard rebase scheduling",
            )
        finally:
            controller.close()
    assert rebases == [], rebases


def test_strict_allows_new_rows_without_rebase() -> None:
    commits: list[list[dict]] = []
    rebases: list[list[dict]] = []
    trace: list[str] = []
    with scoped_env(PEARL_STAGE5_NANOPEARL_STRICT="1"):
        controller = NanoPearlPrefetchController(
            request_batch,
            trace=trace.append,
            commit_batch=lambda rows: commits.append(rows),
            rebase_batch=lambda rows: rebases.append(rows),
        )
        try:
            controller.get_or_request([DraftRequest("row-0", (1, 2), 1)])
            result = controller.get_or_request(
                [
                    DraftRequest("row-0", (1, 2, 102), 1),
                    DraftRequest("row-1", (7, 8), 1),
                ],
                verify_results=[
                    VerifyResult("row-0", accepted_len=1, draft_len=1)
                ],
            )
        finally:
            controller.close()

    assert commits and commits[0][0]["request_id"] == "row-0", commits
    assert rebases == [], rebases
    assert [row.request_id for row in result] == ["row-0", "row-1"]
    assert any("discard_rebase_skip_new_rows" in line for line in trace), trace


def test_explicit_valid_len_controls_commit_boundary() -> None:
    commits: list[list[dict]] = []
    controller = NanoPearlPrefetchController(
        request_batch,
        commit_batch=lambda rows: commits.append(rows),
    )
    try:
        controller.get_or_request([DraftRequest("row-0", (1, 2), 1)])
        controller.get_or_request(
            [DraftRequest("row-0", (1, 2, 102, 999), 1)],
            verify_results=[
                VerifyResult(
                    "row-0",
                    accepted_len=1,
                    draft_len=1,
                    valid_len=2,
                    replacement_token_id=999,
                )
            ],
        )
    finally:
        controller.close()

    assert len(commits) == 1, commits
    row = commits[0][0]
    assert row["request_id"] == "row-0", row
    assert row["valid_len"] == 2, row
    assert row["target_prefix_len"] == 4, row
    assert not row["length_only"], row


def test_full_commit_adopts_pipeline_candidate_boundary() -> None:
    engine = object.__new__(PersistentDraftEngine)
    engine._active_key = "row-0"
    engine._states = {
        "row-0": {
            "request_id": "draft-0",
            "prompt_token_ids": [1, 2],
            "committed_token_ids": [1, 2, 3],
        }
    }
    engine._pipeline_candidates = {
        "row-0": {
            "internal_id": "draft-0",
            "base_prefix": [1, 2],
            "generated": [1, 2, 3, 4, 5],
        }
    }

    committed, adopted = engine._adopt_resident_tokens_for_full_commit(
        prefix=[1, 2, 3, 4, 9],
        valid_len=4,
        accepted_len=1,
    )

    assert committed == [1, 2, 3, 4], committed
    assert adopted == 1, adopted


def test_draft_side_pipeline_is_ac_safe_by_default() -> None:
    engine = object.__new__(PersistentDraftEngine)
    old_explicit = os.environ.pop(
        "PEARL_STAGE5_DRAFT_LOOKAHEAD_PIPELINE", None
    )
    try:
        with scoped_env(
            PEARL_STAGE5_PIPELINE="1",
            PEARL_STAGE5_NANOPEARL_COMMIT_STATE="1",
        ):
            assert engine._pipeline_enabled() is False

        with scoped_env(
            PEARL_STAGE5_PIPELINE="1",
            PEARL_STAGE5_NANOPEARL_COMMIT_STATE="1",
            PEARL_STAGE5_DRAFT_LOOKAHEAD_PIPELINE="1",
        ):
            assert engine._pipeline_enabled() is True

        with scoped_env(
            PEARL_STAGE5_PIPELINE="1",
            PEARL_STAGE5_NANOPEARL_COMMIT_STATE="1",
            PEARL_STAGE5_DRAFT_LOOKAHEAD_PIPELINE="0",
        ):
            assert engine._pipeline_enabled() is False
    finally:
        if old_explicit is not None:
            os.environ["PEARL_STAGE5_DRAFT_LOOKAHEAD_PIPELINE"] = old_explicit


def test_no_bonus_source_contract() -> None:
    runner = Path("vllm_ascend/worker/model_runner_v1.py").read_text(
        encoding="utf-8"
    )
    scheduler = Path("../vllm-hust/vllm/v1/core/sched/scheduler.py").read_text(
        encoding="utf-8"
    )

    assert "PEARL_STAGE5_NANOPEARL_NO_BONUS" in runner
    assert "valid_len = optimistic_len + accepted_len" in runner
    assert "raw[\"replacement_token_id\"] = None" in runner
    assert "self._trim_nanoparl_bonus_tokens(valid_sampled_token_ids)" in runner

    assert "PEARL_STAGE5_NANOPEARL_NO_BONUS" in scheduler
    assert "all_accepted_no_bonus" in scheduler
    assert "expected_valid_len -= num_sampled" in scheduler


def test_target_async_benchmark_optin_contract() -> None:
    benchmark = Path("pearl_stage5_gsm8k_ac_benchmark_v1.py").read_text(
        encoding="utf-8"
    )
    runner = Path("vllm_ascend/worker/model_runner_v1.py").read_text(
        encoding="utf-8"
    )
    worker = Path("pearl_stage5_worker.py").read_text(encoding="utf-8")
    config = Path("../vllm-hust/vllm/config/vllm.py").read_text(
        encoding="utf-8"
    )

    assert "PEARL_STAGE5_TARGET_ASYNC_BENCH_OPTIN_V1" in benchmark
    assert (
        'env.setdefault("PEARL_STAGE5_TARGET_ASYNC_SCHEDULING", "0")'
        in benchmark
    )
    assert 'env.pop("PEARL_STAGE5_TARGET_ASYNC_SCHEDULING", None)' not in benchmark
    assert "PEARL_STAGE5_TARGET_ASYNC_CUSTOM_CLASS_V1" in config
    assert "NANO_PEARL_CUSTOM_PROPOSERS" in config
    assert "nano-PEARL custom-class speculative decoding" in config
    assert "PEARL_STAGE5_TARGET_ASYNC_CUSTOM_CLASS_RUNNER_V1" in runner
    assert "_nanoparl_async_custom_class_enabled" in runner
    assert "_stage_nanoparl_async_sample_state" in runner
    assert "_tensorize_nanoparl_async_draft_tokens" in runner
    assert "PEARL_STAGE5_EXPERIMENTAL_TARGET_ASYNC_CUSTOM_CLASS" in runner
    assert "PEARL_STAGE5_EXPERIMENTAL_TARGET_ASYNC_CUSTOM_CLASS" in config
    assert "PEARL_STAGE5_TARGET_ASYNC_CUSTOM_CLASS_DISABLED_V1" in worker
    assert "custom_class_requires_ordered_commit" in worker


def test_strict_guard_sources() -> None:
    draft = Path("pearl_stage5_draft.py").read_text(encoding="utf-8")
    worker = Path("pearl_stage5_worker.py").read_text(encoding="utf-8")
    proposer = Path("pearl_stage5_nanoparl_proposer_v3.py").read_text(
        encoding="utf-8"
    )

    required = (
        "strict mode forbids Draft request reset/remove-add",
        "strict mode requires in-place rollback",
        "strict mode forbids removing the Draft",
        "strict mode forbids rebase_batch",
    )
    for needle in required:
        assert needle in draft, needle
    assert "PEARL_STAGE5_PIPELINE_AC_SAFE_V1" in draft
    assert "PEARL_STAGE5_DRAFT_LOOKAHEAD_PIPELINE" in draft
    assert "strict mode forbids " in worker
    assert "rebase_batch commands" in worker
    assert "strict mode forbids rebase_batch commands" in proposer


def main() -> None:
    test_strict_requires_commit_batch()
    test_strict_forbids_discard_rebase()
    test_strict_allows_new_rows_without_rebase()
    test_explicit_valid_len_controls_commit_boundary()
    test_full_commit_adopts_pipeline_candidate_boundary()
    test_draft_side_pipeline_is_ac_safe_by_default()
    test_no_bonus_source_contract()
    test_target_async_benchmark_optin_contract()
    test_strict_guard_sources()
    print("nano-pearl strict-mode contract: PASS")


if __name__ == "__main__":
    main()
