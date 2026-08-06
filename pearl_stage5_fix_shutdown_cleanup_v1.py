#!/usr/bin/env python3
"""Patch the Stage-5 Target worker's V1 shutdown compatibility path.

The vLLM-HUST LLMEngine used by the Target worker can raise an
AttributeError for the legacy model_executor cleanup attribute.  The
inference has already completed at that point, but the worker prints a full
traceback and relies on process teardown to stop EngineCore.  This patch
handles only that known shutdown mismatch and explicitly shuts down
EngineCore when possible.
"""

from __future__ import annotations

import argparse
import datetime as dt
import shutil
from pathlib import Path


MARKER = "# PEARL_STAGE5_TARGET_SHUTDOWN_COMPAT_V1"
TARGET_REL = Path("pearl_stage5_worker.py")

RUN_TARGET_ANCHOR = "\n\ndef _run_target(args: argparse.Namespace, control_server: socket.socket) -> None:\n"

SHUTDOWN_OLD = """        if llm is not None:
            shutdown = getattr(llm.llm_engine, "shutdown", None)
            if callable(shutdown):
                try:
                    shutdown()
                except Exception:
                    traceback.print_exc()
        control_conn.close()
"""

SHUTDOWN_NEW = """        if llm is not None:
            _shutdown_target_engine(llm)
        control_conn.close()
"""

HELPER = f'''\n\n{MARKER}
def _shutdown_target_engine(llm: Any) -> None:
    """Run Target cleanup without exposing the known V1 LLMEngine mismatch."""
    engine = getattr(llm, "llm_engine", None)
    if engine is None:
        return

    shutdown = getattr(engine, "shutdown", None)
    if not callable(shutdown):
        return

    try:
        shutdown()
        return
    except AttributeError as exc:
        # vLLM-HUST's V1 LLMEngine currently reaches a legacy cleanup helper
        # that references self.model_executor, which this engine does not own.
        # This is a shutdown-only compatibility issue; inference is complete.
        if "model_executor" not in str(exc):
            raise

        engine_core = getattr(engine, "engine_core", None)
        core_shutdown = getattr(engine_core, "shutdown", None)
        if callable(core_shutdown):
            try:
                core_shutdown()
            except Exception as core_exc:
                print(
                    "[target] EngineCore shutdown fallback warning: "
                    f"{{core_exc!r}}",
                    flush=True,
                )
        print(
            "[target] shutdown compatibility fallback: ignored missing "
            "LLMEngine.model_executor after EngineCore cleanup",
            flush=True,
        )
    except Exception:
        # Preserve the old diagnostic for unrelated cleanup failures.
        traceback.print_exc()
'''


def inspect_state(source: str) -> str:
    marker_count = source.count(MARKER)
    old_count = source.count(SHUTDOWN_OLD)
    new_count = source.count(SHUTDOWN_NEW)
    helper_count = source.count(f"{MARKER}\ndef _shutdown_target_engine")

    if marker_count == 0:
        if old_count != 1:
            raise RuntimeError(
                "pearl_stage5_worker.py: expected one exact Target shutdown "
                f"anchor, found {{'shutdown anchor': {old_count}}}; "
                "no files were changed"
            )
        if source.count(RUN_TARGET_ANCHOR) != 1:
            raise RuntimeError(
                "pearl_stage5_worker.py: expected one exact _run_target "
                "anchor; no files were changed"
            )
        return "pre"

    if (
        marker_count == 1
        and helper_count == 1
        and old_count == 0
        and new_count == 1
    ):
        return "post"

    raise RuntimeError(
        "pearl_stage5_worker.py: shutdown compatibility patch is in an "
        f"unexpected partial state: marker={marker_count}, "
        f"helper={helper_count}, old={old_count}, new={new_count}; "
        "no files were changed"
    )


def apply_patch(repo: Path, backup_dir: Path | None, dry_run: bool) -> None:
    target = repo / TARGET_REL
    if not target.is_file():
        raise FileNotFoundError(target)

    original = target.read_text(encoding="utf-8")
    state = inspect_state(original)
    print(f"target: {target}")
    print(f"state: {state}")
    print("change: Target shutdown compatibility fallback")

    if dry_run or state == "post":
        if dry_run:
            print("dry-run: no files were changed and no backup was created")
        else:
            print("already applied: no files were changed and no backup was created")
        return

    if backup_dir is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        backup_dir = repo.parent / (
            f"{repo.name}.pearl_stage5_fix_shutdown_cleanup_v1.{stamp}"
        )
    backup_dir = backup_dir.expanduser().resolve()
    if backup_dir.exists():
        raise FileExistsError(
            f"backup path already exists; refusing to overwrite: {backup_dir}"
        )

    patched = original.replace(RUN_TARGET_ANCHOR, HELPER + RUN_TARGET_ANCHOR, 1)
    patched = patched.replace(SHUTDOWN_OLD, SHUTDOWN_NEW, 1)
    if patched == original:
        raise RuntimeError("patch produced no change; no files were changed")
    if inspect_state(patched) != "post":
        raise RuntimeError("patched source failed post-state validation; no files were changed")

    shutil.copytree(repo, backup_dir)
    target.write_text(patched, encoding="utf-8")
    print(f"backup: {backup_dir}")
    print(f"patched: {target}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    apply_patch(args.repo.expanduser().resolve(), args.backup_dir, args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
