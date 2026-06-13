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
