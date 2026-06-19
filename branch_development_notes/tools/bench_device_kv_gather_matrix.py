#!/usr/bin/env python3
"""Run the raw kv_cache_block_gather benchmark matrix.

This runner wraps csrc/kv_cache_block_gather/benchmarks/
kv_cache_block_gather_benchmark.cpp after it has been compiled. It does not
build the binary; follow branch_development_notes/notes/reproduction.md first
to build the custom op and compile the benchmark in the Docker environment.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MATRIX = Path("branch_development_notes/benchmarks/device_kv_gather/matrix.json")
DEFAULT_OUTPUT_ROOT = Path("branch_development_notes/work/device_kv_gather_results")
FLOAT32_BYTES = 4

SHAPE_RE = re.compile(
    r"shape: pages=(?P<num_pages>\d+), "
    r"selected_blocks=(?P<selected_blocks>\d+), "
    r"dst_blocks=(?P<dst_blocks>\d+), "
    r"block_bytes=(?P<fragment_bytes>\d+), "
    r"useful_MiB=(?P<useful_mib>[0-9.]+), "
    r"src_pattern=(?P<src_pattern>\w+), "
    r"dst_pattern=(?P<dst_pattern>\w+)"
)

STATS_RE = re.compile(
    r"^(?P<backend>.+?)\s+"
    r"mean_ms=\s*(?P<mean_ms>[0-9.]+)\s+"
    r"p50_ms=\s*(?P<p50_ms>[0-9.]+)\s+"
    r"p90_ms=\s*(?P<p90_ms>[0-9.]+)"
    r"(?:\s+p95_ms=\s*(?P<p95_ms>[0-9.]+))?"
    r"(?:\s+p99_ms=\s*(?P<p99_ms>[0-9.]+))?"
    r"\s+GB/s=\s*(?P<gbps>[0-9.]+)"
)


@dataclass(frozen=True)
class BenchmarkRun:
    case_name: str
    device_id: int
    num_pages: int
    selected_blocks: int
    dst_blocks: int | None
    fragment_bytes: int
    elems_per_block: int
    src_pattern: str
    dst_pattern: str
    warmup: int
    iters: int

    def command(self, binary: Path) -> list[str]:
        cmd = [
            str(binary),
            str(self.device_id),
            "--num-pages",
            str(self.num_pages),
            "--selected-blocks",
            str(self.selected_blocks),
            "--elems-per-block",
            str(self.elems_per_block),
            "--src-pattern",
            self.src_pattern,
            "--dst-pattern",
            self.dst_pattern,
            "--warmup",
            str(self.warmup),
            "--iters",
            str(self.iters),
        ]
        if self.dst_blocks is not None:
            cmd.extend(["--dst-blocks", str(self.dst_blocks)])
        return cmd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--binary",
        default=os.getenv("KV_GATHER_BENCHMARK", "/tmp/kv_cache_block_gather_benchmark"),
        help="Path to compiled kv_cache_block_gather_benchmark binary.",
    )
    parser.add_argument("--matrix", type=Path, default=DEFAULT_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--device-id", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=None)
    parser.add_argument("--iters", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="Run only the first N expanded cases.")
    parser.add_argument("--dry-run", action="store_true", help="Write manifest and commands without executing.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def as_list(value: Any, default: list[Any] | None = None) -> list[Any]:
    if value is None:
        return [] if default is None else list(default)
    if isinstance(value, list):
        return value
    return [value]


def make_runs(matrix: dict[str, Any], args: argparse.Namespace) -> list[BenchmarkRun]:
    defaults = matrix.get("defaults", {})
    runs: list[BenchmarkRun] = []
    for case in matrix.get("cases", []):
        merged = {**defaults, **case}
        fragment_bytes_values = as_list(merged.get("fragment_bytes"))
        selected_blocks_values = as_list(merged.get("selected_blocks"))
        src_patterns = as_list(merged.get("src_patterns"), ["random"])
        dst_patterns = as_list(merged.get("dst_patterns"), ["random"])
        case_name = str(merged["name"])

        for fragment_bytes, selected_blocks, src_pattern, dst_pattern in itertools.product(
            fragment_bytes_values,
            selected_blocks_values,
            src_patterns,
            dst_patterns,
        ):
            fragment_bytes = int(fragment_bytes)
            if fragment_bytes % FLOAT32_BYTES != 0:
                raise ValueError(f"fragment_bytes must be divisible by {FLOAT32_BYTES}: {fragment_bytes}")
            selected_blocks = int(selected_blocks)
            dst_blocks = merged.get("dst_blocks")
            if dst_blocks is not None:
                dst_blocks = int(dst_blocks)
            device_id = args.device_id if args.device_id is not None else int(merged.get("device_id", 0))
            warmup = args.warmup if args.warmup is not None else int(merged.get("warmup", 3))
            iters = args.iters if args.iters is not None else int(merged.get("iters", 10))
            runs.append(
                BenchmarkRun(
                    case_name=case_name,
                    device_id=device_id,
                    num_pages=int(merged.get("num_pages", 4096)),
                    selected_blocks=selected_blocks,
                    dst_blocks=dst_blocks,
                    fragment_bytes=fragment_bytes,
                    elems_per_block=fragment_bytes // FLOAT32_BYTES,
                    src_pattern=str(src_pattern),
                    dst_pattern=str(dst_pattern),
                    warmup=warmup,
                    iters=iters,
                )
            )
    if args.limit is not None:
        return runs[: args.limit]
    return runs


def get_git_value(args: list[str]) -> str:
    try:
        return subprocess.check_output(["git", *args], text=True).strip()
    except Exception:
        return "unknown"


def prepare_output_dir(args: argparse.Namespace) -> Path:
    if args.output_dir is not None:
        output_dir = args.output_dir
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def parse_output(run: BenchmarkRun, stdout: str, stderr: str, returncode: int) -> list[dict[str, Any]]:
    shape: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        shape_match = SHAPE_RE.search(line)
        if shape_match:
            shape = shape_match.groupdict()
            continue
        stats_match = STATS_RE.search(line)
        if not stats_match:
            continue
        stats = stats_match.groupdict()
        row: dict[str, Any] = {
            "case_name": run.case_name,
            "status": "pass" if returncode == 0 else "fail",
            "backend": stats["backend"].strip(),
            "device_id": run.device_id,
            "num_pages": int(shape.get("num_pages", run.num_pages)),
            "selected_blocks": int(shape.get("selected_blocks", run.selected_blocks)),
            "dst_blocks": int(shape.get("dst_blocks", run.dst_blocks or run.selected_blocks)),
            "fragment_bytes": int(shape.get("fragment_bytes", run.fragment_bytes)),
            "useful_mib": float(shape.get("useful_mib", 0.0)),
            "src_pattern": shape.get("src_pattern", run.src_pattern),
            "dst_pattern": shape.get("dst_pattern", run.dst_pattern),
            "warmup": run.warmup,
            "iters": run.iters,
            "mean_ms": float(stats["mean_ms"]),
            "p50_ms": float(stats["p50_ms"]),
            "p90_ms": float(stats["p90_ms"]),
            "p95_ms": float(stats["p95_ms"]) if stats.get("p95_ms") is not None else None,
            "p99_ms": float(stats["p99_ms"]) if stats.get("p99_ms") is not None else None,
            "gbps": float(stats["gbps"]),
            "returncode": returncode,
        }
        rows.append(row)
    if rows:
        return rows
    return [
        {
            "case_name": run.case_name,
            "status": "fail" if returncode != 0 else "no_stats",
            "backend": "",
            "device_id": run.device_id,
            "num_pages": run.num_pages,
            "selected_blocks": run.selected_blocks,
            "dst_blocks": run.dst_blocks or run.selected_blocks,
            "fragment_bytes": run.fragment_bytes,
            "useful_mib": 0.0,
            "src_pattern": run.src_pattern,
            "dst_pattern": run.dst_pattern,
            "warmup": run.warmup,
            "iters": run.iters,
            "mean_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "gbps": None,
            "returncode": returncode,
            "stderr_tail": stderr[-1000:],
        }
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, Any]], manifest: dict[str, Any]) -> None:
    headers = [
        "case_name",
        "backend",
        "fragment_bytes",
        "selected_blocks",
        "src_pattern",
        "dst_pattern",
        "mean_ms",
        "p50_ms",
        "p95_ms",
        "p99_ms",
        "gbps",
        "status",
    ]
    with path.open("w", encoding="utf-8") as f:
        f.write("# Device KV Gather Matrix Results\n\n")
        f.write(f"- git_branch: `{manifest['git_branch']}`\n")
        f.write(f"- git_sha: `{manifest['git_sha']}`\n")
        f.write(f"- binary: `{manifest['binary']}`\n")
        f.write(f"- dry_run: `{manifest['dry_run']}`\n\n")
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            values = ["" if row.get(header) is None else str(row.get(header)) for header in headers]
            f.write("| " + " | ".join(values) + " |\n")


def main() -> int:
    args = parse_args()
    matrix = load_json(args.matrix)
    runs = make_runs(matrix, args)
    output_dir = prepare_output_dir(args)
    binary = Path(args.binary)

    if not args.dry_run and not binary.exists():
        print(f"benchmark binary does not exist: {binary}", file=sys.stderr)
        return 2

    commands = [run.command(binary) for run in runs]
    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "matrix": str(args.matrix),
        "binary": str(binary),
        "dry_run": args.dry_run,
        "run_count": len(runs),
        "git_branch": get_git_value(["branch", "--show-current"]),
        "git_sha": get_git_value(["rev-parse", "HEAD"]),
        "commands": commands,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")

    all_rows: list[dict[str, Any]] = []
    for index, run in enumerate(runs, start=1):
        cmd = run.command(binary)
        print(f"[{index}/{len(runs)}] {' '.join(cmd)}", flush=True)
        if args.dry_run:
            all_rows.append(
                {
                    "case_name": run.case_name,
                    "status": "dry_run",
                    "backend": "",
                    "device_id": run.device_id,
                    "num_pages": run.num_pages,
                    "selected_blocks": run.selected_blocks,
                    "dst_blocks": run.dst_blocks or run.selected_blocks,
                    "fragment_bytes": run.fragment_bytes,
                    "useful_mib": 0.0,
                    "src_pattern": run.src_pattern,
                    "dst_pattern": run.dst_pattern,
                    "warmup": run.warmup,
                    "iters": run.iters,
                }
            )
            continue
        proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False)
        (output_dir / f"run_{index:04d}.stdout").write_text(proc.stdout, encoding="utf-8")
        (output_dir / f"run_{index:04d}.stderr").write_text(proc.stderr, encoding="utf-8")
        all_rows.extend(parse_output(run, proc.stdout, proc.stderr, proc.returncode))
        if proc.returncode != 0:
            print(proc.stderr[-1000:], file=sys.stderr)

    write_jsonl(output_dir / "results.jsonl", all_rows)
    write_csv(output_dir / "results.csv", all_rows)
    write_markdown(output_dir / "summary.md", all_rows, manifest)
    print(f"wrote results to {output_dir}")
    return 0 if all(row.get("status") in {"pass", "dry_run"} for row in all_rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
