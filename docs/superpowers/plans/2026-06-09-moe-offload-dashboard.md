# MoE Offload Dashboard Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a research-only static dashboard generator that reads `benchmarks/results/*.json` and compares non-offloading MoE performance as the upper bound against `--ascend-moe-offload-gb 14` as the offload baseline.

**Architecture:** Add a focused Python script under `benchmarks/scripts/` with pure parsing/rendering helpers and a CLI entry point. Tests import the helpers directly and use temporary benchmark result files so no NPU or real benchmark run is required.

**Tech Stack:** Python standard library, pytest, static HTML/CSS/JS-free output.

---

### Task 1: Dashboard Parser And Renderer Tests

**Files:**
- Create: `tests/ut/benchmarks/test_generate_moe_offload_dashboard.py`
- Later create: `benchmarks/scripts/generate_moe_offload_dashboard.py`

- [ ] **Step 1: Write the failing tests**

Create tests that write two serving benchmark JSON files, load them through `load_dashboard_data()`, assert the labels preserve the semantic roles, assert throughput/TTFT/TPOT values are parsed, and assert `render_dashboard_html()` includes the comparison text.

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest -q tests/ut/benchmarks/test_generate_moe_offload_dashboard.py`

Expected: FAIL because `benchmarks.scripts.generate_moe_offload_dashboard` does not exist yet.

### Task 2: Implement Static Dashboard Generator

**Files:**
- Create: `benchmarks/scripts/generate_moe_offload_dashboard.py`
- Modify: `benchmarks/scripts/generate_moe_offload_dashboard.py`

- [ ] **Step 1: Implement the parser**

Add a `BenchmarkRun` dataclass and `load_dashboard_data(results_dir, upper_label, baseline_label)` that reads `*.json`, matches labels from `dashboard_label`, `variant`, `test_name`, or filename, and extracts:

- throughput from `output_throughput`, then `request_throughput`, then `tokens_per_second`, then `requests_per_second`
- TTFT from `median_ttft_ms`
- TPOT from `median_tpot_ms`

The parser must raise a clear `ValueError` when either required run is missing.

- [ ] **Step 2: Implement HTML rendering**

Add `render_dashboard_html(upper, baseline)` with a compact data-first table and small comparison bars. Treat non-offloading as the upper bound for throughput, and treat offload 14GB as the baseline. For latency metrics, lower is better.

- [ ] **Step 3: Add CLI**

Add command-line arguments:

- `--results-dir`, default `benchmarks/results`
- `--output`, default `<results-dir>/moe_offload_dashboard.html`
- `--upper-label`, default `non-offload`
- `--baseline-label`, default `offload-14GB`

### Task 3: Documentation And Local Hygiene

**Files:**
- Modify: `benchmarks/README.md`
- Modify: `.gitignore`

- [ ] **Step 1: Document usage**

Add a short README section showing how to generate the dashboard after placing benchmark JSON files in `benchmarks/results/`.

- [ ] **Step 2: Ignore visual companion state**

Add `.superpowers/` to `.gitignore` so the brainstorming browser artifacts remain local.

### Task 4: Verification

**Files:**
- Run-only verification.

- [ ] **Step 1: Run focused tests**

Run: `pytest -q tests/ut/benchmarks/test_generate_moe_offload_dashboard.py`

- [ ] **Step 2: Run compile check**

Run: `python -m compileall -q benchmarks/scripts tests/ut/benchmarks`

- [ ] **Step 3: Run whitespace check**

Run: `git diff --check`

- [ ] **Step 4: Confirm branch boundary**

Run: `git status --short --branch`

Confirm the branch is `research` and no push or sync to `dev` / `main` was performed.
