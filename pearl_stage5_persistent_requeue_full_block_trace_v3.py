#!/usr/bin/env python3
"""Apply the full-block trace using a single-line enqueue anchor.

v1/v2 stopped safely during source inspection because the persistent-requeue
code after ``scheduler.waiting.prepend_request(request)`` differs from the
assumed surrounding text.  This v3 matches only that unique statement and
keeps the same dry-run, compile, full-backup, and atomic-write safeguards.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_v1():
    source = Path(__file__).with_name(
        "pearl_stage5_persistent_requeue_full_block_trace_v1.py"
    )
    if not source.is_file():
        raise FileNotFoundError(f"required helper patch is missing: {source}")
    spec = importlib.util.spec_from_file_location(
        "pearl_stage5_full_block_trace_v1_impl_v3", source
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load helper patch: {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> int:
    base = load_v1()

    # Match only the unique enqueue statement; do not assume the following
    # persistent-requeue trace condition's formatting.
    base.AFTER_ENQUEUE_OLD = (
        "        scheduler.waiting.prepend_request(request)\n"
    )
    base.AFTER_ENQUEUE_NEW = (
        "        scheduler.waiting.prepend_request(request)\n"
        "\n"
        "        if os.environ.get(\"PEARL_STAGE5_FULL_BLOCK_TRACE\", \"0\") == \"1\":\n"
        "            self._pearl_full_block_trace_pending = True\n"
        "            self._trace_full_block_state(\n"
        "                \"full_requeue.after_enqueue\", request=request\n"
        "            )\n"
    )

    def make_backup_dir(repo, explicit):
        backup_dir = (
            Path(explicit).expanduser().resolve()
            if explicit is not None
            else repo.parent
            / f"{repo.name}.pearl_stage5_persistent_requeue_full_block_trace_v3."
            f"{base.timestamp()}"
        )
        if backup_dir.exists():
            raise FileExistsError(f"backup directory already exists: {backup_dir}")
        backup_dir.mkdir(parents=True, exist_ok=False)
        return backup_dir

    base.make_backup_dir = make_backup_dir
    base.__file__ = __file__
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
