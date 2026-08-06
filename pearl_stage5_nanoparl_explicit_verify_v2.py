#!/usr/bin/env python3
"""v2 of the explicit nano-PEARL verify wiring.

This wrapper reuses the guarded v1 patch for runtime/proposer/sampler and
replaces only the ModelRunner transform with a formatting-tolerant version.
The v1 script expected one exact ``None,  # draft_probs`` block; HUST source
variants may format that call differently.
"""

from __future__ import annotations

import re
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


def _line_indent(source: str, position: int) -> str:
    line_start = source.rfind("\n", 0, position) + 1
    match = re.match(r"[ \t]*", source[line_start:])
    return match.group(0) if match else ""


def _next_line_indent(source: str, position: int) -> str:
    line_end = source.find("\n", position)
    if line_end < 0:
        return ""
    next_start = line_end + 1
    match = re.match(r"[ \t]*", source[next_start:])
    return match.group(0) if match else ""


def _find_call(source: str, pattern: re.Pattern[str], predicate) -> tuple[int, int, str]:
    candidates = list(pattern.finditer(source))
    selected: list[tuple[int, int, str]] = []
    for match in candidates:
        open_pos = source.find("(", match.start(), match.end())
        close_pos = _matching_close(source, open_pos)
        if close_pos < 0:
            continue
        body = source[match.start(): close_pos + 1]
        if predicate(body):
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

    # Clear stale results on the ordinary non-speculative path when the
    # source contains that branch.  This is optional because source variants
    # name the ordinary sampler path differently.
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
    rejection_start, rejection_close, rejection_indent = _find_call(
        source,
        rejection_pattern,
        lambda body: "self.rejection_sampler" in body,
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
        lambda body: (
            "self.input_batch.num_tokens_no_spec" in body
            and "self.input_batch.token_ids_cpu" in body
        ),
    )
    open_pos = source.find("(", proposer_start, proposer_close + 1)
    arg_indent = _next_line_indent(source, open_pos)
    if not arg_indent:
        arg_indent = proposer_indent + "    "

    # Insert before the line containing the call's closing parenthesis.  The
    # previous version inserted at the parenthesis itself, which could produce
    # ``verify_results=...,)`` and was needlessly sensitive to indentation.
    close_line_start = source.rfind("\n", 0, proposer_close) + 1
    before_close = source[:close_line_start].rstrip()
    comma_added = 0
    if before_close and not before_close.endswith(","):
        source = (
            source[: len(before_close)]
            + ","
            + source[len(before_close):]
        )
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


base.TRANSFORMS[
    Path("vllm_ascend/worker/model_runner_v1.py")
] = transform_runner


if __name__ == "__main__":
    raise SystemExit(base.main())
