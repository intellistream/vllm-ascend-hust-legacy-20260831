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

import argparse
import importlib.util
import json
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).parents[3] / "benchmarks" / "scripts" / "run_ascend_moe_profile_suite.py"


def load_suite_module():
    spec = importlib.util.spec_from_file_location("run_ascend_moe_profile_suite", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def make_args(tmp_path: Path, **overrides):
    values = {
        "base_url": "http://127.0.0.1:8005",
        "profile_url": "http://127.0.0.1:8005",
        "endpoint": "/v1/chat/completions",
        "backend": "openai-chat",
        "served_model_name": "qwen3-30b-a3b",
        "tokenizer": "/models/qwen3",
        "sharegpt_dataset": tmp_path / "sharegpt.json",
        "profiler_dir": tmp_path / "vllm_profile",
        "output_dir": tmp_path / "out",
        "bench_command": "vllm bench serve",
        "request_rate": "inf",
        "max_concurrency": 1,
        "num_warmups": 0,
        "mixed_num_prompts": 2,
        "mixed_output_len": 4,
        "prefill_num_prompts": 2,
        "prefill_input_len": 32,
        "prefill_output_len": 1,
        "decode_num_prompts": 2,
        "decode_input_len": 8,
        "decode_output_len": 16,
        "http_timeout": 1.0,
        "stop_analyse_delay_sec": 0.0,
        "skip_analyse": True,
        "dry_run": True,
        "sew_moe_trace_path": None,
        "sew_moe_profile_path": None,
        "require_sew_moe_artifacts": False,
        "run_slot_sweep": False,
        "slot_sweep_range": "8:64:8",
        "slot_sweep_policy": "lru",
        "slot_sweep_expert_bytes": 14_680_064,
        "slot_sweep_bandwidth_gbps": 24.0,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def test_dry_run_manifest_records_per_scenario_sew_artifact_paths(tmp_path):
    suite = load_suite_module()
    trace_path = tmp_path / "server" / "moe_offload_trace.jsonl"
    profile_path = tmp_path / "server" / "sew_moe_profile.jsonl"

    manifest = suite.run_suite(
        make_args(
            tmp_path,
            sew_moe_trace_path=trace_path,
            sew_moe_profile_path=profile_path,
        ))

    for scenario in manifest["scenarios"]:
        scenario_dir = tmp_path / "out" / scenario["name"]
        assert scenario["sew_moe_trace_jsonl"] == str(scenario_dir / "moe_offload_trace.jsonl")
        assert scenario["sew_moe_profile_jsonl"] == str(scenario_dir / "sew_moe_profile.jsonl")
        assert scenario["sew_moe_source_trace_jsonl"] == str(trace_path)
        assert scenario["sew_moe_source_profile_jsonl"] == str(profile_path)


def test_snapshot_jsonl_delta_writes_only_new_records(tmp_path):
    suite = load_suite_module()
    source = tmp_path / "source.jsonl"
    dest = tmp_path / "scenario" / "moe_offload_trace.jsonl"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text('{"old": 1}\n', encoding="utf-8")
    offset = source.stat().st_size
    with source.open("a", encoding="utf-8") as f:
        f.write('{"new": 2}\n{"new": 3}\n')

    wrote = suite._snapshot_jsonl_delta(source, offset, dest)

    assert wrote is True
    assert dest.read_text(encoding="utf-8") == '{"new": 2}\n{"new": 3}\n'


def test_write_analysis_report_exports_sew_p1_plan(tmp_path, monkeypatch):
    suite = load_suite_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    profiler_output = tmp_path / "profile" / "ASCEND_PROFILER_OUTPUT"
    profiler_output.mkdir(parents=True)
    manifest = {
        "scenarios": [
            {
                "name": "decode",
                "phase": "decode",
                "benchmark_json": str(tmp_path / "decode.json"),
                "sew_moe_trace_jsonl": str(tmp_path / "trace.jsonl"),
                "sew_moe_profile_jsonl": str(tmp_path / "profile.jsonl"),
                "profiler_outputs": [str(profiler_output)],
            }
        ]
    }

    class FakeAnalyzer:

        @staticmethod
        def analyze_profile(*_args):
            return {
                "phase": "decode",
                "p1_decision": {
                    "target": "P1-C",
                    "compute_bucket_plan": {
                        "version": 1,
                        "phase": "decode",
                        "buckets": [{"bucket_id": 0, "signature": "counts:hot"}],
                    },
                },
            }

        @staticmethod
        def render_markdown(_reports):
            return "# report\n"

    monkeypatch.setattr(suite, "_load_analyzer", lambda: FakeAnalyzer)

    suite._write_analysis_report(manifest, output_dir)

    plan = json.loads((output_dir / "sew_moe_p1_plan.json").read_text(encoding="utf-8"))
    assert plan == {
        "version": 1,
        "plans": [
            {
                "phase": "decode",
                "target": "P1-C",
                "compute_bucket_plan": {
                    "version": 1,
                    "phase": "decode",
                    "buckets": [{"bucket_id": 0, "signature": "counts:hot"}],
                },
            }
        ],
    }


def test_write_analysis_report_runs_slot_sweep_when_requested(tmp_path, monkeypatch):
    suite = load_suite_module()
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    scenario_dir = output_dir / "decode"
    scenario_dir.mkdir()
    profiler_output = tmp_path / "profile" / "ASCEND_PROFILER_OUTPUT"
    profiler_output.mkdir(parents=True)
    trace_path = scenario_dir / "moe_offload_trace.jsonl"
    trace_path.write_text(
        "\n".join([
            json.dumps({
                "source": "grouped_dispatch",
                "layer_id": 0,
                "step_id": 0,
                "active_experts": [0, 1],
            }),
            json.dumps({
                "source": "grouped_dispatch",
                "layer_id": 0,
                "step_id": 1,
                "active_experts": [0, 1],
            }),
        ]) + "\n",
        encoding="utf-8",
    )
    manifest = {
        "scenarios": [
            {
                "name": "decode",
                "phase": "decode",
                "benchmark_json": str(tmp_path / "decode.json"),
                "sew_moe_trace_jsonl": str(trace_path),
                "profiler_outputs": [str(profiler_output)],
            }
        ]
    }

    class FakeAnalyzer:

        @staticmethod
        def analyze_profile(*_args):
            return {
                "phase": "decode",
                "p1_decision": {
                    "target": "P1-T",
                    "slot_sweep_hint": {
                        "start_slots": 1,
                        "stop_slots": 2,
                        "step_slots": 1,
                    },
                },
            }

        @staticmethod
        def render_markdown(_reports):
            return "# report\n"

    monkeypatch.setattr(suite, "_load_analyzer", lambda: FakeAnalyzer)

    suite._write_analysis_report(
        manifest,
        output_dir,
        run_slot_sweep=True,
        slot_sweep_range="1:2:1",
        slot_sweep_policy="lru",
        slot_sweep_expert_bytes=10,
        slot_sweep_bandwidth_gbps=10.0,
    )

    sweep_path = scenario_dir / "slot_sweep_lru.json"
    sweep = json.loads(sweep_path.read_text(encoding="utf-8"))
    plan = json.loads((output_dir / "sew_moe_p1_plan.json").read_text(encoding="utf-8"))
    assert sweep["slot_range"] == "1:2:1"
    assert sweep["recommended_num_slots"] == 2
    assert sweep["recommended_prefetchable_miss_count"] == 0
    assert sweep["recommended_exposed_miss_count"] == 2
    assert sweep["recommended_prefetchable_host_to_hbm_bytes"] == 0
    assert sweep["recommended_exposed_host_to_hbm_bytes"] == 20
    assert plan["plans"][0]["slot_sweep_result_json"] == str(sweep_path)
    assert plan["plans"][0]["slot_sweep_result"]["recommended_num_slots"] == 2
    assert plan["plans"][0]["slot_sweep_result"]["recommended_exposed_miss_count"] == 2
