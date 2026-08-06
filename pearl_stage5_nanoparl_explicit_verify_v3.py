#!/usr/bin/env python3
"""Standalone v3 patch for explicit nano-PEARL verification wiring.

This file is intentionally a new patch-script name.  It does not overwrite
the v1 or v2 patch scripts.  The ModelRunner transform is tolerant of HUST
formatting and selects the custom_class proposer call specifically, because
the same method also contains a draft-model proposer call with a similar
argument prefix.
"""

from __future__ import annotations

import argparse
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

import pearl_stage5_nanoparl_explicit_verify_v1 as base


def _matching_close(source: str, open_pos: int) -> int:
    depth = 0
    for index in range(open_pos, len(source)):
        char = source[index]
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return index
    return -1


def _next_line_indent(source: str, position: int) -> str:
    line_end = source.find("\n", position)
    if line_end < 0:
        return ""
    next_start = line_end + 1
    match = re.match(r"[ \t]*", source[next_start:])
    return match.group(0) if match else ""


def _find_call(source: str, pattern: re.Pattern[str], predicate):
    selected = []
    for match in pattern.finditer(source):
        open_pos = source.find("(", match.start(), match.end())
        close_pos = _matching_close(source, open_pos)
        if close_pos < 0:
            continue
        body = source[match.start(): close_pos + 1]
        context = source[max(0, match.start() - 1200): match.start()]
        if predicate(body, context):
            selected.append((match.start(), close_pos, match.group("indent")))
    if len(selected) != 1:
        raise RuntimeError(
            f"expected one matching call, found {len(selected)}; "
            "no files were changed"
        )
    return selected[0]


def transform_runner(source: str) -> str:
    marker = base.MARKERS["vllm_ascend/worker/model_runner_v1.py"]
    if marker in source:
        return source
    source = marker + "\n" + source

    ordinary = re.compile(
        r"^(?P<indent>[ \t]*)return self\.sampler\(", re.MULTILINE
    )
    ordinary_matches = list(ordinary.finditer(source))
    if ordinary_matches:
        match = ordinary_matches[0]
        source = (
            source[: match.start()]
            + match.group("indent")
            + "self._pearl_verify_results = None\n"
            + source[match.start():]
        )

    rejection_pattern = re.compile(
        r"^(?P<indent>[ \t]*)sampler_output\s*=\s*"
        r"self\.rejection_sampler\(",
        re.MULTILINE,
    )
    _, rejection_close, rejection_indent = _find_call(
        source,
        rejection_pattern,
        lambda body, context: "self.rejection_sampler" in body,
    )
    rejection_close_end = rejection_close + 1
    source = (
        source[:rejection_close_end]
        + "\n"
        + rejection_indent
        + "self._pearl_verify_results = getattr(\n"
        + rejection_indent
        + "    self.rejection_sampler, \"last_verify_results\", None\n"
        + rejection_indent
        + ")"
        + source[rejection_close_end:]
    )

    proposer_pattern = re.compile(
        r"^(?P<indent>[ \t]*)draft_token_ids\s*=\s*"
        r"self\.drafter\.propose\(",
        re.MULTILINE,
    )
    proposer_start, proposer_close, proposer_indent = _find_call(
        source,
        proposer_pattern,
        lambda body, context: (
            "self.input_batch.num_tokens_no_spec" in body
            and "self.input_batch.token_ids_cpu" in body
            and "sampled_token_ids should be a python list for custom_class"
            in context
        ),
    )

    open_pos = source.find("(", proposer_start, proposer_close + 1)
    arg_indent = _next_line_indent(source, open_pos)
    if not arg_indent:
        arg_indent = proposer_indent + "    "

    close_line_start = source.rfind("\n", 0, proposer_close) + 1
    before_close = source[:close_line_start].rstrip()
    comma_added = 0
    if before_close and not before_close.endswith(","):
        source = source[: len(before_close)] + "," + source[len(before_close):]
        comma_added = 1
        proposer_close += comma_added
        close_line_start += comma_added

    addition = (
        arg_indent
        + "request_ids=getattr(self.input_batch, \"req_ids\", None),\n"
        + arg_indent
        + "verify_results=getattr(self, \"_pearl_verify_results\", None),\n"
    )
    source = source[:close_line_start] + addition + source[close_line_start:]
    new_close = proposer_close + len(addition)
    source = (
        source[: new_close + 1]
        + "\n"
        + proposer_indent
        + "self._pearl_verify_results = None"
        + source[new_close + 1:]
    )
    compile(source, "vllm_ascend/worker/model_runner_v1.py", "exec")
    return source


base.TRANSFORMS[Path("vllm_ascend/worker/model_runner_v1.py")] = transform_runner


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--backup-dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    repo = args.repo.expanduser().resolve()

    transformed: dict[Path, str] = {}
    changed: list[Path] = []
    for relative, transform in base.TRANSFORMS.items():
        target = repo / relative
        if not target.is_file():
            raise FileNotFoundError(target)
        original = target.read_text(encoding="utf-8")
        updated = transform(original)
        transformed[relative] = updated
        state = "post" if updated == original else "pre"
        print(f"target: {target}")
        print(f"state: {state}")
        if updated != original:
            changed.append(relative)

    print("change: explicit per-request Target verification for nano-PEARL v3")
    if not changed:
        print("already patched: no files changed")
        return 0
    if args.dry_run:
        print("dry-run: no files were changed and no backup was created")
        return 0

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
    backup_dir = (
        args.backup_dir.expanduser().resolve()
        if args.backup_dir is not None
        else repo.parent / f"{repo.name}.pearl_stage5_nanoparl_explicit_verify_v3.{stamp}"
    )
    if backup_dir.exists():
        raise FileExistsError(f"backup directory already exists: {backup_dir}")
    backup_dir.mkdir(parents=True)
    for relative in changed:
        saved = backup_dir / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(repo / relative, saved)
    for relative in changed:
        target = repo / relative
        temporary = target.with_name(target.name + ".pearl_stage5_v3_tmp")
        temporary.write_text(transformed[relative], encoding="utf-8")
        temporary.replace(target)

    print(f"backup: {backup_dir}")
    for relative in changed:
        print(f"patched: {repo / relative}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
