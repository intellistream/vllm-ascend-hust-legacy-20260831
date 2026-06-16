# SEW-MoE P0 Active Expert Observability Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. User constraint: do not commit changes for this plan; use local status checks instead of git commits.

**Goal:** Add trace-only active expert observability for logical top-k and grouped dispatch shapes, plus analyzer summaries that identify the next SEW-Compute or SEW-Offload target.

**Architecture:** Extend the existing `moe_offload` trace path instead of creating a new profiler. `TraceCollector` owns CPU-safe record extraction and JSONL serialization; `MoeOffloadRuntime` gates emission behind existing trace config; MoE execution calls logical tracing after `select_experts` and grouped tracing after token dispatch. The benchmark analyzer reads the same JSONL artifacts offline and renders a SEW-MoE section.

**Tech Stack:** Python 3, PyTorch tensors on CPU/NPU, pytest, JSONL artifacts, existing Ascend MoE/offload runtime.

---

### Task 1: Extend Trace Records For Logical And Grouped Events

**Files:**
- Modify: `vllm_ascend/moe_offload/trace_collector.py`
- Modify: `tests/ut/moe_offload/test_trace_collector.py`
- Modify: `tests/ut/moe_offload/test_trace_export.py`

- [ ] **Step 1: Write failing trace collector tests**

Add tests that expect `source`, `fanout`, `num_logical_experts`, grouped fields, and cumsum/count handling:

```python
def test_trace_collector_records_logical_active_expert_event():
    collector = TraceCollector(max_records=4)
    topk_ids = torch.tensor([[0, 2], [2, 3], [3, 3]], dtype=torch.int32)

    record = collector.record_logical(
        layer_id=7,
        step_id=11,
        topk_ids=topk_ids,
        num_logical_experts=4,
        mode="decode",
    )

    assert record.source == "logical_topk"
    assert record.num_logical_experts == 4
    assert record.num_tokens == 3
    assert record.top_k == 2
    assert record.fanout == 3
    assert record.active_experts == (0, 2, 3)
    assert record.expert_token_counts == {0: 1, 2: 2, 3: 3}
    assert record.group_list_type is None
    assert record.group_list_signature is None
    assert collector.latest_for_layer(7) == record


def test_trace_collector_records_grouped_count_event():
    collector = TraceCollector(max_records=4)
    group_list = torch.tensor([0, 3, 0, 2], dtype=torch.int64)

    record = collector.record_grouped(
        layer_id=8,
        step_id=12,
        group_list=group_list,
        group_list_type=1,
        physical_expert_count=4,
        mode="prefill",
    )

    assert record.source == "grouped_dispatch"
    assert record.num_tokens == 5
    assert record.top_k == 1
    assert record.fanout == 2
    assert record.active_experts == (1, 3)
    assert record.expert_token_counts == {1: 3, 3: 2}
    assert record.group_list_type == 1
    assert record.group_list_signature == "counts:0,3,0,2"
    assert record.physical_expert_count == 4


def test_trace_collector_records_grouped_cumsum_event():
    collector = TraceCollector(max_records=4)
    group_list = torch.tensor([0, 3, 3, 5], dtype=torch.int64)

    record = collector.record_grouped(
        layer_id=8,
        step_id=12,
        group_list=group_list,
        group_list_type=0,
        physical_expert_count=4,
    )

    assert record.expert_token_counts == {1: 3, 3: 2}
    assert record.group_list_signature == "cumsum:0,3,3,5"
```

- [ ] **Step 2: Run tests and verify they fail**

Run: `pytest tests/ut/moe_offload/test_trace_collector.py tests/ut/moe_offload/test_trace_export.py -q`

Expected: FAIL because `record_logical`, `record_grouped`, and new fields do not exist.

- [ ] **Step 3: Implement trace record extraction**

Update `TraceRecord` to include:

```python
source: str
num_logical_experts: int
fanout: int
group_list_type: int | None = None
group_list_signature: str | None = None
physical_expert_count: int | None = None
```

Keep `record(...)` as a compatibility wrapper that calls `record_logical(...)`. Add `record_logical(...)` and `record_grouped(...)`. For grouped records, interpret `group_list_type == 1` as per-expert counts and `group_list_type == 0` as cumulative counts; unsupported types should produce an empty count map and signature `unsupported:<type>`.

- [ ] **Step 4: Run trace tests and update old JSON expectations**

Run: `pytest tests/ut/moe_offload/test_trace_collector.py tests/ut/moe_offload/test_trace_export.py -q`

Expected: PASS after existing expected JSON dictionaries include the new deterministic fields.

- [ ] **Step 5: Local checkpoint**

Run: `git status --short`

Expected: modified trace tests and collector only. Do not commit.

### Task 2: Runtime APIs And MoE Integration

**Files:**
- Modify: `vllm_ascend/moe_offload/runtime.py`
- Modify: `vllm_ascend/ops/fused_moe/fused_moe.py`
- Modify: `vllm_ascend/ops/fused_moe/moe_comm_method.py`
- Modify: `tests/ut/moe_offload/test_runtime_trace_only.py`

- [ ] **Step 1: Write failing runtime tests**

Add tests for logical API compatibility and grouped API no-mutation:

```python
def test_runtime_traces_logical_and_grouped_active_experts_without_mutation():
    runtime = MoeOffloadRuntime(MoeOffloadConfig(enabled=True, trace_only=True))
    topk_ids = torch.tensor([[0, 2], [2, 3]], dtype=torch.int32)
    topk_weights = torch.randn(2, 2)
    group_list = torch.tensor([1, 2, 0, 1], dtype=torch.int64)

    returned_ids, returned_weights = runtime.trace_logical_active_experts(
        layer_id=5,
        topk_ids=topk_ids,
        topk_weights=topk_weights,
        num_logical_experts=4,
        mode="decode",
    )
    returned_group_list = runtime.trace_grouped_active_experts(
        layer_id=5,
        group_list=group_list,
        group_list_type=1,
        physical_expert_count=4,
        mode="decode",
    )

    records = runtime.trace_collector.records()
    assert returned_ids is topk_ids
    assert returned_weights is topk_weights
    assert returned_group_list is group_list
    assert [record.source for record in records] == ["logical_topk", "grouped_dispatch"]
    assert records[0].step_id == records[1].step_id
    assert records[0].fanout == 3
    assert records[1].expert_token_counts == {0: 1, 1: 2, 3: 1}
```

- [ ] **Step 2: Run test and verify it fails**

Run: `pytest tests/ut/moe_offload/test_runtime_trace_only.py -q`

Expected: FAIL because runtime APIs do not exist.

- [ ] **Step 3: Implement runtime trace APIs**

Add `trace_logical_active_experts(...)` and `trace_grouped_active_experts(...)`. Use one `step_id = next(self._step_counter)` for a logical/grouped pair by storing the most recent step id by layer until grouped tracing consumes it. Keep `trace_routing(...)` as a compatibility wrapper around `trace_logical_active_experts(...)`.

- [ ] **Step 4: Wire grouped trace after token dispatch**

In `MoECommMethod.fused_experts()`, after `token_dispatch_output` is available, call:

```python
get_moe_offload_runtime().trace_grouped_active_experts(
    layer_id=getattr(fused_experts_input.offload, "layer_id", -1) if fused_experts_input.offload else -1,
    group_list=token_dispatch_output.group_list,
    group_list_type=token_dispatch_output.group_list_type,
    physical_expert_count=fused_experts_input.routing.physical_expert_count,
)
```

The method must no-op when tracing is disabled or `group_list` is `None`.

- [ ] **Step 5: Rename apply hook to logical trace API**

In `AscendUnquantizedFusedMoEMethod.apply()`, replace `trace_routing(...)` with `trace_logical_active_experts(...)`, preserving returned tensors.

- [ ] **Step 6: Run runtime tests**

Run: `pytest tests/ut/moe_offload/test_runtime_trace_only.py tests/ut/moe_offload/test_trace_export.py -q`

Expected: PASS.

- [ ] **Step 7: Local checkpoint**

Run: `git status --short`

Expected: runtime, MoE integration, and tests modified. Do not commit.

### Task 3: Analyzer SEW-MoE Summary

**Files:**
- Modify: `benchmarks/scripts/analyze_ascend_moe_profile.py`
- Modify: `tests/ut/benchmarks/test_analyze_ascend_moe_profile.py`

- [ ] **Step 1: Write failing analyzer test**

Add a helper that writes JSONL active expert records next to a profile directory and assert the report contains SEW-MoE summary:

```python
def test_analyzer_summarizes_sew_moe_trace(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    trace_path = output.parent / "moe_offload_trace.jsonl"
    trace_path.write_text(
        "\n".join([
            json.dumps({
                "source": "logical_topk",
                "layer_id": 1,
                "step_id": 10,
                "mode": "decode",
                "num_tokens": 2,
                "top_k": 2,
                "num_logical_experts": 4,
                "fanout": 3,
                "active_experts": [0, 2, 3],
                "expert_token_counts": {"0": 1, "2": 2, "3": 1},
            }),
            json.dumps({
                "source": "grouped_dispatch",
                "layer_id": 1,
                "step_id": 10,
                "mode": "decode",
                "num_tokens": 4,
                "top_k": 1,
                "num_logical_experts": 0,
                "fanout": 3,
                "active_experts": [0, 1, 2],
                "expert_token_counts": {"0": 1, "1": 2, "2": 1},
                "group_list_type": 1,
                "group_list_signature": "counts:1,2,1",
                "physical_expert_count": 3,
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    report = analyzer.analyze_profile("decode", output, None)
    markdown = analyzer.render_markdown([report])

    assert report["sew_moe"]["record_count"] == 2
    assert report["sew_moe"]["fanout_by_source"]["logical_topk"]["max"] == 3
    assert report["sew_moe"]["top_group_list_signatures"][0]["signature"] == "counts:1,2,1"
    assert "SEW-MoE active expert trace" in markdown
    assert "counts:1,2,1" in markdown
```

- [ ] **Step 2: Run analyzer tests and verify failure**

Run: `pytest tests/ut/benchmarks/test_analyze_ascend_moe_profile.py -q`

Expected: FAIL because analyzer does not read SEW-MoE JSONL.

- [ ] **Step 3: Implement JSONL discovery and summary**

Add `_find_sew_moe_trace(output_dir)` that checks:

```python
output_dir.parent / "moe_offload_trace.jsonl"
output_dir / "moe_offload_trace.jsonl"
output_dir.parent / "sew_moe_trace.jsonl"
output_dir / "sew_moe_trace.jsonl"
```

Add `_summarize_sew_moe_trace(path)` that returns `{"record_count": 0, "note": "No SEW-MoE active expert trace found."}` when missing, otherwise fanout stats by source, top layers by fanout, top layers by grouped token count, and top `group_list_signature` counts.

- [ ] **Step 4: Render SEW-MoE markdown**

In `_render_phase`, include a compact section after kernel table. If no trace is found, render one sentence. If trace exists, render record count, fanout by source, and top group signatures.

- [ ] **Step 5: Run analyzer tests**

Run: `pytest tests/ut/benchmarks/test_analyze_ascend_moe_profile.py -q`

Expected: PASS.

- [ ] **Step 6: Local checkpoint**

Run: `git status --short`

Expected: analyzer and analyzer tests modified. Do not commit.

### Task 4: End-To-End Local Verification

**Files:**
- No additional files unless tests expose a defect.

- [ ] **Step 1: Run focused unit tests**

Run:

```bash
pytest \
  tests/ut/moe_offload/test_trace_collector.py \
  tests/ut/moe_offload/test_trace_export.py \
  tests/ut/moe_offload/test_runtime_trace_only.py \
  tests/ut/benchmarks/test_analyze_ascend_moe_profile.py \
  -q
```

Expected: PASS.

- [ ] **Step 2: Run a tiny local theory smoke**

Run a Python snippet that creates a trace JSONL from CPU tensors and analyzes a synthetic profiler directory. Expected output should show one logical and one grouped event with shared `step_id`, plus a SEW-MoE markdown section.

- [ ] **Step 3: Inspect worktree**

Run: `git status --short`

Expected: only planned files modified/added. Do not commit.

- [ ] **Step 4: Report result**

Summarize changed files, focused test commands, and whether the P0 theory is locally proven without NPU execution. Do not claim throughput improvement; P0 is observability only.

---

## Self-Review

- Spec coverage: logical active experts, grouped dispatch shape, JSONL emission, disabled no-op behavior, analyzer summaries, and non-offload/offload shared path are covered.
- Placeholder scan: no TBD/TODO/fill-later placeholders are used.
- Type consistency: `num_logical_experts`, `source`, `fanout`, `group_list_type`, `group_list_signature`, and `physical_expert_count` are consistent across collector, runtime, JSONL, and analyzer.
- User constraint: plan replaces commit steps with local checkpoints and explicitly says do not commit.
