#!/usr/bin/env python3
"""Read-only diagnosis for the Stage-5 speculative-decoding runtime path.

This script does not modify the repository.  It reports:

* the Python executable and import locations used by the current environment;
* whether the acceptance-debug marker is present in the imported module;
* source locations containing the likely speculative-decoding acceptance
  symbols in vllm and vllm-ascend.

Run it in the same activated environment and from the same machine as the
Stage-5 coordinator:

    python pearl_stage5_runtime_trace_v1.py \
        --repo /root/data/vllm-hust
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys
from pathlib import Path


MODULES = (
    "vllm",
    "vllm.v1.sample.rejection_sampler",
    "vllm.v1.worker.model_runner_v1",
    "vllm.v1.worker.gpu_model_runner",
    "vllm.v1.spec_decode.custom_class_proposer",
)

SYMBOLS = (
    "PEARL_ACCEPTANCE_DEBUG_V1",
    "rejection_sample(",
    "RejectionSampler",
    "num_accepted_tokens",
    "num_draft_tokens",
    "sampled_token_ids",
    "spec_decode_metadata",
    "custom_class",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo",
        type=Path,
        default=Path("/root/data/vllm-hust"),
        help="vllm core repository root",
    )
    parser.add_argument(
        "--max-matches",
        type=int,
        default=240,
        help="maximum source matches to print",
    )
    return parser.parse_args()


def print_environment() -> None:
    print(f"python: {sys.executable}")
    print(f"python_version: {sys.version.split()[0]}")
    print(f"cwd: {Path.cwd()}")
    print(f"PEARL_ACCEPTANCE_DEBUG: {os.environ.get('PEARL_ACCEPTANCE_DEBUG')!r}")
    print(f"VLLM_USE_V1: {os.environ.get('VLLM_USE_V1')!r}")
    print("sys.path:")
    for item in sys.path:
        print(f"  {item or '<cwd>'}")


def print_import_locations() -> None:
    print("import locations:")
    for name in MODULES:
        try:
            module = importlib.import_module(name)
        except Exception as exc:
            print(f"  {name}: IMPORT_ERROR {type(exc).__name__}: {exc}")
            continue

        module_file = getattr(module, "__file__", None)
        print(f"  {name}: {module_file}")
        if module_file is None:
            continue

        source = Path(module_file)
        try:
            text = source.read_text(encoding="utf-8")
        except (OSError, UnicodeError) as exc:
            print(f"    read_error: {type(exc).__name__}: {exc}")
            continue

        present = [symbol for symbol in SYMBOLS if symbol in text]
        print(f"    symbols: {present}")
        if name == "vllm.v1.sample.rejection_sampler":
            print(
                "    acceptance_debug_marker: "
                f"{'PEARL_ACCEPTANCE_DEBUG_V1' in text}"
            )


def scan_sources(repo: Path, max_matches: int) -> None:
    roots = [repo / "vllm", repo / "vllm_ascend"]
    print("source scan:")
    matches = 0
    for root in roots:
        if not root.is_dir():
            print(f"  missing: {root}")
            continue
        for source in sorted(root.rglob("*.py")):
            try:
                lines = source.read_text(encoding="utf-8").splitlines()
            except (OSError, UnicodeError):
                continue
            for line_number, line in enumerate(lines, start=1):
                if not any(symbol in line for symbol in SYMBOLS):
                    continue
                print(f"  {source}:{line_number}: {line.strip()}")
                matches += 1
                if matches >= max_matches:
                    print(f"  ... truncated at {max_matches} matches")
                    return
    print(f"  total_matches: {matches}")


def main() -> int:
    args = parse_args()
    repo = args.repo.resolve()
    print_environment()
    print_import_locations()
    scan_sources(repo, args.max_matches)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
