#!/usr/bin/env python3
"""Offline contract test for the batch-level nano-PEARL commit patch."""

from __future__ import annotations

import importlib.util
from pathlib import Path


def load_patch():
    path = Path(__file__).with_name("pearl_stage5_nanoparl_commit_state_v1.py")
    spec = importlib.util.spec_from_file_location("pearl_commit_patch", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    patch = load_patch()
    import pearl_stage5_nanoparl_explicit_verify_v1 as explicit

    runtime_source = Path("pearl_stage5_nanoparl_runtime_v1.py").read_text()
    runtime_source = explicit.transform_runtime(runtime_source)
    runtime_source = patch.transform_runtime(runtime_source)
    compile(runtime_source, "pearl_stage5_nanoparl_runtime_v1.py", "exec")

    for relative, transform in (
        ("pearl_stage5_nanoparl_proposer_v1.py", patch.transform_proposer_v1),
        ("pearl_stage5_nanoparl_proposer_v3.py", patch.transform_proposer_v3),
        ("pearl_stage5_worker.py", patch.transform_worker),
    ):
        compile(
            transform(Path(relative).read_text()),
            relative,
            "exec",
        )

    draft_fixture = """from __future__ import annotations
import os
from typing import Any

class PersistentDraftEngine:
    def _replace_tokens(self, prefix): pass
    def _sync_model_runner_state(self, prefix, request): pass
    def rebase_batch(self, updates): return []
    def propose_batch(self, updates): return []
"""
    compile(
        patch.transform_draft(draft_fixture),
        "pearl_stage5_draft.py",
        "exec",
    )

    namespace: dict[str, object] = {}
    exec(
        compile(
            runtime_source,
            "pearl_stage5_nanoparl_runtime_v1.py",
            "exec",
        ),
        namespace,
    )
    DraftRequest = namespace["DraftRequest"]
    VerifyResult = namespace["VerifyResult"]
    Controller = namespace["NanoPearlPrefetchController"]

    commit_batches: list[list[dict]] = []

    def request_batch(rows: list[dict]) -> list[list[int]]:
        result: list[list[int]] = []
        for row in rows:
            request_id = str(row["request_id"])
            result.append([101, 102] if request_id == "r0" else [201, 202])
        return result

    controller = Controller(
        request_batch,
        commit_batch=lambda rows: commit_batches.append(rows),
    )
    try:
        controller.get_or_request(
            [
                DraftRequest("r0", (1, 2, 3), 2),
                DraftRequest("r1", (4, 5, 6), 2),
            ]
        )
        controller.get_or_request(
            [
                DraftRequest("r0", (1, 2, 3, 101, 999), 2),
                DraftRequest("r1", (4, 5, 6, 201, 888), 2),
            ],
            verify_results=[
                VerifyResult(
                    "r0",
                    accepted_len=1,
                    draft_len=2,
                    replacement_token_id=999,
                ),
                VerifyResult(
                    "r1",
                    accepted_len=1,
                    draft_len=2,
                    replacement_token_id=888,
                ),
            ],
        )
    finally:
        controller.close()

    assert len(commit_batches) == 1, commit_batches
    assert len(commit_batches[0]) == 2, commit_batches
    assert [row["request_id"] for row in commit_batches[0]] == ["r0", "r1"]
    assert [row["accepted_len"] for row in commit_batches[0]] == [1, 1]
    assert [row["valid_len"] for row in commit_batches[0]] == [4, 4]
    assert all(row["target_prefix_len"] == 5 for row in commit_batches[0])

    print("batch_commit_transform=PASS")
    print("batch_commit_runtime=PASS")
    print("commit_batch_rows=2")
    print("accepted_len=[1, 1] valid_len=[4, 4]")


if __name__ == "__main__":
    main()

