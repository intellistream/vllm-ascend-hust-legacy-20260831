#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.

import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[3] / "benchmarks" / "scripts" / "analyze_ascend_moe_profile.py"


def load_analyzer_module():
    spec = importlib.util.spec_from_file_location("analyze_ascend_moe_profile", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_csv(path: Path, text: str):
    path.write_text(text.strip() + "\n", encoding="utf-8")


def make_profile_dir(tmp_path: Path, name: str) -> Path:
    output = tmp_path / name / "ASCEND_PROFILER_OUTPUT"
    output.mkdir(parents=True)
    write_csv(
        output / "op_statistic.csv",
        """
Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us),Ratio(%)
0,GroupedMatmul,MIX_AIC,8,8000,900,1000,1100,50
0,FusedInferAttentionScore,MIX_AIC,4,3000,700,750,800,18
0,RmsNorm,AI_VECTOR_CORE,20,1200,40,60,90,7
0,MoeGatingTopK,AI_VECTOR_CORE,8,900,90,112.5,140,5
""",
    )
    write_csv(
        output / "kernel_details.csv",
        """
Device_id,Model ID,Task ID,Stream ID,Name,Type,OP State,Accelerator Core,Start Time(us),Duration(us),Wait Time(us),Block Dim,Mix Block Dim,HF32 Eligible,Input Shapes,Input Data Types,Input Formats,Output Shapes,Output Data Types,Output Formats,Context ID,aicore_time(us),aic_total_cycles,aic_mac_time(us),aic_mac_ratio,aic_scalar_time(us),aic_scalar_ratio,aic_mte1_time(us),aic_mte1_ratio,aic_mte2_time(us),aic_mte2_ratio,aic_fixpipe_time(us),aic_fixpipe_ratio,aic_icache_miss_rate,aiv_time(us),aiv_total_cycles,aiv_vec_time(us),aiv_vec_ratio,aiv_scalar_time(us),aiv_scalar_ratio,aiv_mte2_time(us),aiv_mte2_ratio,aiv_mte3_time(us),aiv_mte3_ratio,aiv_icache_miss_rate,cube_utilization(%)
0,0,1,7,aclnnGroupedMatmulV5_GroupedMatmul,GroupedMatmul,dynamic,MIX_AIC,10,1000,100,1,0,NO,"2;4096",BF16,ND,"2;4096",BF16,ND,N/A,1000,10,700,0.7,10,0.01,0,0,180,0.18,10,0.01,0,0,0,0,0,0,0,0,0,0,0,0,81
0,0,2,7,aclnnRmsNorm,RmsNorm,dynamic,AI_VECTOR_CORE,20,30,20,1,0,NO,"1;4096",BF16,ND,"1;4096",BF16,ND,N/A,0,0,0,0,0,0,0,0,0,0,0,0,0,25,10,8,0.32,4,0.16,6,0.24,5,0.2,0,0
0,0,3,7,aclnnSlice,Slice,dynamic,AI_VECTOR_CORE,30,20,40,1,0,NO,"1;4096",BF16,ND,"1;4096",BF16,ND,N/A,0,0,0,0,0,0,0,0,0,0,0,0,0,18,10,5,0.27,5,0.27,4,0.22,4,0.22,0,0
""",
    )
    write_csv(
        output / "operator_details.csv",
        """
Name,Input Shapes,Call Stack,Host Self Duration(us),Host Total Duration(us),Device Self Duration(us),Device Total Duration(us),Device Self Duration With AICore(us),Device Total Duration With AICore(us)
vllm::decode_step,,,40,100,0,1200,0,1200
aten::rms_norm,,,30,50,0,90,0,90
""",
    )
    write_csv(
        output / "step_trace_time.csv",
        """
Device_id,Step,Computing,Communication(Not Overlapped),Overlapped,Communication,Free,Stage,Bubble,Communication(Not Overlapped and Exclude Receive),Preparing
0,,12000,0,0,0,3000,15000,0,0,50
""",
    )
    return output


def write_sew_trace(
    output: Path,
    signatures: list[tuple[str, int]],
    *,
    fanout: int = 3,
) -> None:
    records = []
    step_id = 0
    for signature, count in signatures:
        for _ in range(count):
            records.append({
                "source": "grouped_dispatch",
                "layer_id": 1,
                "step_id": step_id,
                "mode": "decode",
                "num_tokens": fanout,
                "top_k": 1,
                "num_logical_experts": 8,
                "fanout": fanout,
                "active_experts": list(range(fanout)),
                "expert_token_counts": {str(i): 1 for i in range(fanout)},
                "group_list_type": 1,
                "group_list_signature": signature,
                "physical_expert_count": fanout,
            })
            step_id += 1
    (output.parent / "moe_offload_trace.jsonl").write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


def write_pipeline_profile(
    output: Path,
    *,
    stage_t_ms: float,
    stage_r_ms: float,
    stage_c_ms: float,
    stage_m_ms: float,
    records: int = 4,
) -> None:
    lines = []
    for step_id in range(records):
        lines.append(json.dumps({
            "event": "moe_pipeline_timing",
            "layer_id": 1,
            "step_id": step_id,
            "stage_t_ms": stage_t_ms,
            "stage_r_ms": stage_r_ms,
            "stage_c_ms": stage_c_ms,
            "stage_m_ms": stage_m_ms,
        }))
    (output.parent / "sew_moe_profile.jsonl").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def append_compute_bucket_gate_profile(output: Path) -> None:
    profile_path = output.parent / "sew_moe_profile.jsonl"
    with profile_path.open("a", encoding="utf-8") as f:
        for payload in (
            {
                "enabled": True,
                "reason": "eligible",
                "bucket_id": 0,
                "signature": "counts:2,0,1,0",
                "original_expert_count": 4,
                "compact_expert_count": 2,
            },
            {
                "enabled": False,
                "reason": "requires_unquantized_path",
                "bucket_id": 1,
                "signature": "counts:1,1,1,1",
                "original_expert_count": 4,
                "compact_expert_count": 4,
            },
        ):
            f.write(json.dumps({
                "name": "compute_bucket_fast_path_gate",
                "event": "compute_bucket_fast_path_gate",
                "layer_id": 1,
                "seconds": 0.0001,
                "payload": payload,
            }) + "\n")


def test_builds_phase_report_with_moe_and_fusion_recommendations(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    benchmark = tmp_path / "decode.json"
    benchmark.write_text(
        json.dumps({
            "label": "decode-heavy",
            "median_ttft_ms": 410.0,
            "median_tpot_ms": 33.5,
            "output_throughput": 75.0,
        }),
        encoding="utf-8",
    )

    report = analyzer.analyze_profile("decode", output, benchmark)

    assert report["phase"] == "decode"
    assert report["benchmark"]["median_tpot_ms"] == 33.5
    assert report["op_hotspots"][0]["name"] == "GroupedMatmul"
    assert report["kernel_summary"]["top_by_duration"][0]["type"] == "GroupedMatmul"
    assert any(item["category"] == "moe_grouped_matmul" for item in report["optimization_opportunities"])
    assert any(item["category"] == "fusion_candidate" for item in report["optimization_opportunities"])
    assert any(item["category"] == "scheduler_wait" for item in report["optimization_opportunities"])


def test_renders_markdown_with_phase_specific_ttft_and_tpot_focus(tmp_path):
    analyzer = load_analyzer_module()
    mixed = analyzer.analyze_profile("mixed", make_profile_dir(tmp_path, "mixed"), None)
    prefill = analyzer.analyze_profile("prefill", make_profile_dir(tmp_path, "prefill"), None)
    decode = analyzer.analyze_profile("decode", make_profile_dir(tmp_path, "decode"), None)

    markdown = analyzer.render_markdown([mixed, prefill, decode])

    assert "Ascend MoE Profile Report" in markdown
    assert "TTFT focus" in markdown
    assert "TPOT focus" in markdown
    assert "GroupedMatmul" in markdown
    assert "RmsNorm" in markdown


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
    assert report["sew_moe"]["slot_budget_hint"]["min_slots_per_layer"] == 3
    assert report["sew_moe"]["slot_budget_hint"]["max_grouped_fanout"] == 3
    assert "SEW-MoE active expert trace" in markdown
    assert "counts:1,2,1" in markdown
    assert "Minimum per-layer slots" in markdown


def test_analyzer_reports_slot_budget_hint_from_grouped_fanout(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    trace_path = output.parent / "moe_offload_trace.jsonl"
    trace_path.write_text(
        "\n".join([
            json.dumps({
                "source": "grouped_dispatch",
                "layer_id": 3,
                "step_id": 1,
                "mode": "decode",
                "num_tokens": 4,
                "top_k": 1,
                "num_logical_experts": 0,
                "fanout": 4,
                "active_experts": [0, 1, 2, 3],
                "expert_token_counts": {"0": 1, "1": 1, "2": 1, "3": 1},
                "group_list_type": 1,
                "group_list_signature": "counts:1,1,1,1",
                "physical_expert_count": 4,
            }),
            json.dumps({
                "source": "grouped_dispatch",
                "layer_id": 7,
                "step_id": 2,
                "mode": "decode",
                "num_tokens": 2,
                "top_k": 1,
                "num_logical_experts": 0,
                "fanout": 2,
                "active_experts": [0, 1],
                "expert_token_counts": {"0": 1, "1": 1},
                "group_list_type": 1,
                "group_list_signature": "counts:1,1",
                "physical_expert_count": 2,
            }),
        ]) + "\n",
        encoding="utf-8",
    )

    report = analyzer.analyze_profile("decode", output, None)

    assert report["sew_moe"]["slot_budget_hint"] == {
        "min_slots_per_layer": 4,
        "max_grouped_fanout": 4,
        "mean_grouped_fanout": 3.0,
        "high_fanout_layers": [{"layer_id": 3, "max_fanout": 4}],
    }


def test_analyzer_recommends_p1_c_when_grouped_compute_dominates_stable_shapes(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    write_sew_trace(output, [("counts:1,2,1", 8), ("counts:2,1,1", 2)])
    write_pipeline_profile(output, stage_t_ms=0.1, stage_r_ms=0.5, stage_c_ms=6.0, stage_m_ms=0.4)

    report = analyzer.analyze_profile("decode", output, None)
    markdown = analyzer.render_markdown([report])

    assert report["p1_decision"]["target"] == "P1-C"
    assert report["p1_decision"]["signature_concentration_percent"] == 80.0
    assert report["p1_decision"]["compute_bucket_hint"]["coverage_percent"] == 100.0
    assert report["p1_decision"]["compute_bucket_hint"]["fallback_percent"] == 0.0
    assert report["p1_decision"]["compute_bucket_hint"]["top_signatures"] == [
        {"signature": "counts:1,2,1", "count": 8, "coverage_percent": 80.0},
        {"signature": "counts:2,1,1", "count": 2, "coverage_percent": 20.0},
    ]
    assert report["p1_decision"]["compute_bucket_plan"] == {
        "version": 1,
        "phase": "decode",
        "mode": "trace_only",
        "selection": "top_grouped_signatures",
        "total_grouped_records": 10,
        "coverage_percent": 100.0,
        "fallback_percent": 0.0,
        "gate": {
            "source": "grouped_dispatch",
            "requires_group_list_signature": True,
            "fallback": "existing_grouped_matmul_path",
        },
        "buckets": [
            {
                "bucket_id": 0,
                "signature": "counts:1,2,1",
                "sample_count": 8,
                "coverage_percent": 80.0,
                "active_expert_ids": [0, 1, 2],
                "compact_group_list": [1, 2, 1],
                "original_expert_count": 3,
                "compact_expert_count": 3,
            },
            {
                "bucket_id": 1,
                "signature": "counts:2,1,1",
                "sample_count": 2,
                "coverage_percent": 20.0,
                "active_expert_ids": [0, 1, 2],
                "compact_group_list": [2, 1, 1],
                "original_expert_count": 3,
                "compact_expert_count": 3,
            },
        ],
    }
    assert "Recommended P1 target: P1-C" in markdown
    assert "Stable grouped buckets" in markdown
    assert "Compute bucket plan: 2 buckets" in markdown


def test_analyzer_summarizes_compute_bucket_fast_path_gate_events(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    write_sew_trace(output, [("counts:2,0,1,0", 4)])
    write_pipeline_profile(output, stage_t_ms=0.1, stage_r_ms=0.5, stage_c_ms=6.0, stage_m_ms=0.4)
    append_compute_bucket_gate_profile(output)

    report = analyzer.analyze_profile("decode", output, None)
    markdown = analyzer.render_markdown([report])

    gate_summary = report["pipeline_profile"]["compute_bucket_fast_path_gate"]
    assert gate_summary == {
        "record_count": 2,
        "enabled_count": 1,
        "fallback_count": 1,
        "enabled_percent": 50.0,
        "mean_original_expert_count": 4.0,
        "mean_compact_expert_count": 3.0,
        "mean_compaction_ratio": 0.75,
        "top_fallback_reasons": [{"reason": "requires_unquantized_path", "count": 1}],
    }
    assert "Compute bucket gate: enabled=50.0%" in markdown
    assert "experts 4.0 -> 3.0" in markdown


def test_analyzer_recommends_p1_rm_when_routing_is_large_and_shapes_unstable(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    write_sew_trace(
        output,
        [
            ("counts:1,0,0", 1),
            ("counts:0,1,0", 1),
            ("counts:0,0,1", 1),
            ("counts:1,1,0", 1),
        ],
    )
    write_pipeline_profile(output, stage_t_ms=0.0, stage_r_ms=2.4, stage_c_ms=3.0, stage_m_ms=1.3)

    report = analyzer.analyze_profile("decode", output, None)

    assert report["p1_decision"]["target"] == "P1-RM"
    assert report["p1_decision"]["signature_concentration_percent"] == 25.0


def test_analyzer_computes_bucket_fallback_against_all_grouped_signatures(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    signatures = [("counts:hot", 18)]
    signatures.extend((f"counts:tail{i}", 1) for i in range(12))
    write_sew_trace(output, signatures)
    write_pipeline_profile(output, stage_t_ms=0.1, stage_r_ms=0.5, stage_c_ms=6.0, stage_m_ms=0.4)

    report = analyzer.analyze_profile("decode", output, None)

    assert report["p1_decision"]["target"] == "P1-C"
    assert report["sew_moe"]["grouped_signature_total_count"] == 30
    assert report["p1_decision"]["compute_bucket_hint"]["coverage_percent"] == 66.7
    assert report["p1_decision"]["compute_bucket_hint"]["fallback_percent"] == 33.3


def test_analyzer_recommends_p1_t_when_offload_transfer_dominates(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    write_sew_trace(output, [("counts:1,1,1,1,1,1,1,1", 4)], fanout=8)
    write_pipeline_profile(output, stage_t_ms=13.4, stage_r_ms=0.4, stage_c_ms=0.5, stage_m_ms=0.2)

    report = analyzer.analyze_profile("decode", output, None)

    assert report["p1_decision"]["target"] == "P1-T"
    assert report["pipeline_profile"]["fractions"]["t_frac"] > 0.9
    assert report["p1_decision"]["slot_sweep_hint"]["start_slots"] == 8
    assert report["p1_decision"]["slot_sweep_hint"]["stop_slots"] == 64
    markdown = analyzer.render_markdown([report])
    assert "--slot-range 8:64:8" in markdown


def test_analyzer_recommends_p1_h_when_transfer_and_compute_are_close(tmp_path):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    write_sew_trace(output, [("counts:2,1,1,0", 3), ("counts:1,2,0,1", 3)], fanout=4)
    write_pipeline_profile(output, stage_t_ms=4.0, stage_r_ms=1.0, stage_c_ms=2.3, stage_m_ms=0.8)

    report = analyzer.analyze_profile("decode", output, None)

    assert report["p1_decision"]["target"] == "P1-H"
    assert report["p1_decision"]["overlap_potential_ratio"] > 0.7


def test_phase_arg_accepts_optional_sew_trace_and_pipeline_paths(tmp_path):
    analyzer = load_analyzer_module()
    output = tmp_path / "ASCEND_PROFILER_OUTPUT"
    benchmark = tmp_path / "benchmark.json"
    trace = tmp_path / "moe_offload_trace.jsonl"
    pipeline = tmp_path / "sew_moe_profile.jsonl"

    phase, parsed_output, parsed_benchmark, parsed_trace, parsed_pipeline = analyzer._parse_phase_arg(
        f"decode:{output}:{benchmark}:{trace}:{pipeline}")

    assert phase == "decode"
    assert parsed_output == output
    assert parsed_benchmark == benchmark
    assert parsed_trace == trace
    assert parsed_pipeline == pipeline


def test_main_passes_optional_sew_paths_to_analyzer(tmp_path, monkeypatch):
    analyzer = load_analyzer_module()
    output = make_profile_dir(tmp_path, "decode")
    trace = tmp_path / "trace.jsonl"
    pipeline = tmp_path / "pipeline.jsonl"
    json_output = tmp_path / "report.json"

    calls = []

    def fake_analyze_profile(phase, profiler_output, benchmark_path=None, sew_moe_trace_path=None,
                             pipeline_profile_path=None):
        calls.append((phase, profiler_output, benchmark_path, sew_moe_trace_path, pipeline_profile_path))
        return {
            "phase": phase,
            "profiler_output": str(profiler_output),
            "benchmark": {},
            "op_hotspots": [],
            "operator_hotspots": [],
            "kernel_summary": {"by_type": [], "top_by_duration": []},
            "step_trace": {},
            "sew_moe": {"record_count": 0},
            "pipeline_profile": {"record_count": 0},
            "p1_decision": {"target": "INSUFFICIENT_DATA", "reason": "test"},
            "optimization_opportunities": [],
        }

    monkeypatch.setattr(analyzer, "analyze_profile", fake_analyze_profile)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "analyze_ascend_moe_profile.py",
            "--phase",
            f"decode:{output}::{trace}:{pipeline}",
            "--json-output",
            str(json_output),
        ],
    )

    assert analyzer.main() == 0
    assert calls == [("decode", output, None, trace, pipeline)]
