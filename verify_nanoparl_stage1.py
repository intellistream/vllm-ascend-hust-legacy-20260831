#!/usr/bin/env python3
"""Verify the Stage-1 native Scheduler/KV-Manager nano-PEARL bridge.

This is deliberately a static/runtime-import boundary check. It does not claim
that a model run, Target rollback, HCCL commit, or edge-case matrix has passed.
Run it after ``apply_nanoparl_core_lifecycle_v2.py --apply`` on the real
vLLM and vLLM-Ascend repositories.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path


def _require(path: Path, fragments: tuple[str, ...]) -> str:
    if not path.is_file():
        raise AssertionError(f"missing target file: {path}")
    text = path.read_text(encoding="utf-8")
    missing = [fragment for fragment in fragments if fragment not in text]
    if missing:
        raise AssertionError(
            f"{path}: missing required anchor(s): {missing!r}"
        )
    return text


def _method_body(text: str, signature: str) -> str:
    start = text.find(signature)
    if start < 0:
        raise AssertionError(f"method signature not found: {signature!r}")
    next_method = re.search(r"\n    (?:async )?def ", text[start + len(signature) :])
    end = start + len(signature) + next_method.start() if next_method else len(text)
    return text[start:end]


def _check_order(text: str, before: str, after: str, path: Path) -> None:
    before_at = text.find(before)
    after_at = text.find(after)
    if before_at < 0 or after_at < 0 or before_at >= after_at:
        raise AssertionError(
            f"{path}: expected {before!r} before {after!r}"
        )


def verify(vllm_repo: Path, ascend_repo: Path, run_py_compile: bool) -> None:
    vllm = vllm_repo / "vllm"
    ascend = ascend_repo / "vllm_ascend"

    outputs = _require(
        vllm / "v1/outputs.py",
        (
            "from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias",
            "nanoparl_verify_results: list[dict[str, Any]] | None = None",
        ),
    )
    block_pool = _require(
        vllm / "v1/core/block_pool.py",
        ("def invalidate_cached_block(self, block: KVCacheBlock) -> None:",),
    )
    manager = _require(
        vllm / "v1/core/single_type_kv_cache_manager.py",
        (
            "def retain_request_boundary(self, request_id: str, valid_len: int) -> None:",
            "[PEARL_STAGE5_NATIVE_SCHED_KV_V1] manager_boundary ",
        ),
    )
    _require(
        vllm / "v1/core/kv_cache_coordinator.py",
        ("def retain_request_boundary(self, request_id: str, valid_len: int) -> None:",),
    )
    _require(
        vllm / "v1/core/kv_cache_manager.py",
        ("def retain_request_boundary(self, request: Request, valid_len: int) -> None:",),
    )
    _require(
        vllm / "v1/worker/block_table.py",
        ("def set_nanoparl_logical_boundary(",),
    )
    _require(
        vllm / "v1/worker/gpu_input_batch.py",
        ("self.block_table.set_nanoparl_logical_boundary(",),
    )
    _require(
        vllm / "v1/worker/gpu_model_runner.py",
        ("self.input_batch.block_table.set_nanoparl_logical_boundary(",),
    )
    scheduler = _require(
        vllm / "v1/core/sched/scheduler.py",
        (
            "nanoparl_results_by_req_id: dict[str, Any] = {}",
            "self.kv_cache_manager.retain_request_boundary(",
            "if not nanoparl_boundary_applied and request.num_computed_tokens > 0:",
            "[PEARL_STAGE5_NATIVE_SCHED_KV_V1] scheduler_boundary ",
        ),
    )
    _require(
        ascend / "worker/block_table.py",
        ("def set_nanoparl_logical_boundary(",),
    )
    ascend_runner = _require(
        ascend / "worker/model_runner_v1.py",
        (
            "def _normalize_nanoparl_verify_results(",
            "nanoparl_verify_results=nanoparl_verify_results,",
        ),
    )

    # The explicit result must be normalized before it is put into the output,
    # and the scheduler must inspect it before the compatibility decrement.
    _check_order(
        ascend_runner,
        "nanoparl_verify_results = self._normalize_nanoparl_verify_results(",
        "model_runner_output = ModelRunnerOutput(",
        ascend / "worker/model_runner_v1.py",
    )
    _check_order(
        scheduler,
        "nanoparl_results_by_req_id: dict[str, Any] = {}",
        "self.kv_cache_manager.retain_request_boundary(",
        vllm / "v1/core/sched/scheduler.py",
    )

    # Stage 1 must retain the request-owned blocks. Physical free/pop belongs
    # to the normal finish/eviction path and must not be hidden in this method.
    manager_body = _method_body(
        manager,
        "def retain_request_boundary(self, request_id: str, valid_len: int) -> None:",
    )
    forbidden = ("pop_blocks_for_free(", "self.free(", "free_blocks(")
    if any(token in manager_body for token in forbidden):
        raise AssertionError(
            "retain_request_boundary must not release physical blocks in Stage 1"
        )

    if run_py_compile:
        compile_targets = [
            vllm / "v1/outputs.py",
            vllm / "v1/core/block_pool.py",
            vllm / "v1/core/single_type_kv_cache_manager.py",
            vllm / "v1/core/kv_cache_coordinator.py",
            vllm / "v1/core/kv_cache_manager.py",
            vllm / "v1/worker/block_table.py",
            vllm / "v1/worker/gpu_input_batch.py",
            vllm / "v1/worker/gpu_model_runner.py",
            vllm / "v1/core/sched/scheduler.py",
            ascend / "worker/block_table.py",
            ascend / "worker/model_runner_v1.py",
        ]
        subprocess.run(
            [sys.executable, "-m", "py_compile", *map(str, compile_targets)],
            check=True,
        )

    # Keep the local variable used above meaningful without adding a runtime
    # dependency on vLLM's logger in this standalone checker.
    if "PEARL_STAGE5_NATIVE_SCHED_KV_V1" in manager:
        trace_status = "trace-enabled"
    else:
        trace_status = "trace-marker-not-required"
    print(
        "Stage-1 native Scheduler/KV Manager bridge: PASS "
        f"({trace_status}; py_compile={'on' if run_py_compile else 'off'})"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-repo", type=Path, required=True)
    parser.add_argument("--ascend-repo", type=Path, required=True)
    parser.add_argument(
        "--skip-py-compile",
        action="store_true",
        help="only check anchors and ordering",
    )
    args = parser.parse_args()
    verify(args.vllm_repo, args.ascend_repo, not args.skip_py_compile)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
