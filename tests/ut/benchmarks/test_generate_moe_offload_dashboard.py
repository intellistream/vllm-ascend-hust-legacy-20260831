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

import pytest


SCRIPT_PATH = Path(__file__).parents[3] / "benchmarks" / "scripts" / "generate_moe_offload_dashboard.py"


def load_dashboard_module():
    spec = importlib.util.spec_from_file_location("generate_moe_offload_dashboard", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def write_result(path: Path, payload: dict):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_loads_upper_bound_and_offload_baseline_from_benchmark_json(tmp_path):
    dashboard = load_dashboard_module()

    write_result(
        tmp_path / "serving_qwen3_30b_non-offload.json",
        {
            "dashboard_label": "non-offload",
            "request_throughput": 2.5,
            "output_throughput": 480.0,
            "median_ttft_ms": 920.0,
            "median_tpot_ms": 41.0,
        },
    )
    write_result(
        tmp_path / "serving_qwen3_30b_offload-14GB.json",
        {
            "variant": "offload-14GB",
            "request_throughput": 2.2,
            "output_throughput": 390.0,
            "median_ttft_ms": 1180.0,
            "median_tpot_ms": 55.0,
        },
    )

    upper, baseline = dashboard.load_dashboard_data(
        tmp_path,
        upper_label="non-offload",
        baseline_label="offload-14GB",
    )

    assert upper.role == "upper_bound"
    assert upper.label == "non-offload"
    assert upper.throughput == 480.0
    assert upper.throughput_unit == "tok/s"
    assert upper.ttft_ms == 920.0
    assert upper.tpot_ms == 41.0

    assert baseline.role == "baseline"
    assert baseline.label == "offload-14GB"
    assert baseline.throughput == 390.0
    assert baseline.throughput_unit == "tok/s"
    assert baseline.ttft_ms == 1180.0
    assert baseline.tpot_ms == 55.0
    assert baseline.offload_gb == 14.0


def test_renders_data_first_dashboard_with_clear_metric_semantics(tmp_path):
    dashboard = load_dashboard_module()

    upper = dashboard.BenchmarkRun(
        label="non-offload",
        role="upper_bound",
        source=tmp_path / "upper.json",
        throughput=480.0,
        throughput_unit="tok/s",
        ttft_ms=920.0,
        tpot_ms=41.0,
        offload_gb=None,
    )
    baseline = dashboard.BenchmarkRun(
        label="offload-14GB",
        role="baseline",
        source=tmp_path / "baseline.json",
        throughput=390.0,
        throughput_unit="tok/s",
        ttft_ms=1180.0,
        tpot_ms=55.0,
        offload_gb=14.0,
    )

    html = dashboard.render_dashboard_html(upper, baseline)

    assert "MoE Offload Performance Dashboard" in html
    assert "MoE non-offloading upper bound" in html
    assert "--ascend-moe-offload-gb 14" in html
    assert "Throughput" in html
    assert "TTFT" in html
    assert "TPOT" in html
    assert "81.2%" in html
    assert "+28.3%" in html
    assert "+34.1%" in html


def test_missing_required_run_raises_clear_error(tmp_path):
    dashboard = load_dashboard_module()

    (tmp_path / "ShareGPT_prompt_subset.json").write_text("[]", encoding="utf-8")
    write_result(
        tmp_path / "serving_qwen3_30b_non-offload.json",
        {
            "dashboard_label": "non-offload",
            "output_throughput": 480.0,
            "median_ttft_ms": 920.0,
            "median_tpot_ms": 41.0,
        },
    )

    with pytest.raises(ValueError, match="Missing benchmark result for offload-14GB"):
        dashboard.load_dashboard_data(
            tmp_path,
            upper_label="non-offload",
            baseline_label="offload-14GB",
        )


def test_refuses_to_compare_throughput_with_different_units(tmp_path):
    dashboard = load_dashboard_module()

    upper = dashboard.BenchmarkRun(
        label="non-offload",
        role="upper_bound",
        source=tmp_path / "upper.json",
        throughput=480.0,
        throughput_unit="tok/s",
        ttft_ms=920.0,
        tpot_ms=41.0,
        offload_gb=None,
    )
    baseline = dashboard.BenchmarkRun(
        label="offload-14GB",
        role="baseline",
        source=tmp_path / "baseline.json",
        throughput=2.2,
        throughput_unit="req/s",
        ttft_ms=1180.0,
        tpot_ms=55.0,
        offload_gb=14.0,
    )

    with pytest.raises(ValueError, match="different units"):
        dashboard.render_dashboard_html(upper, baseline)
