#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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
#

import argparse
import json
import os

import pytest

from tools.sew_offload.run_p1_experiments import run_experiment_matrix


def test_run_experiment_matrix_applies_env_and_writes_summary(tmp_path, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", "1")
    monkeypatch.setenv("VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS", "99")

    matrix_path = tmp_path / "experiments.json"
    output_dir = tmp_path / "runs"
    matrix_path.write_text(
        json.dumps({
            "version": 1,
            "source_plan": "/tmp/sew_moe_p1_plan.json",
            "experiments": [
                {
                    "name": "baseline",
                    "env": {},
                    "evidence": {
                        "phase": "decode",
                    },
                },
                {
                    "name": "p1_fixed_slot_recommended",
                    "env": {
                        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
                        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
                        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": "8",
                        "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": "1",
                        "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD": "8",
                    },
                    "evidence": {
                        "recommended_num_slots": 8,
                        "recommended_exposed_host_to_hbm_bytes": 400,
                    },
                },
                {
                    "name": "p1_compute_bucket_trace_only",
                    "env": {
                        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
                        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "1",
                        "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": "/tmp/sew_moe_p1_plan.json",
                    },
                    "evidence": {
                        "bucket_count": 2,
                    },
                },
                {
                    "name": "p1_compute_bucket_fast_path",
                    "env": {
                        "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": "/tmp/sew_moe_p1_plan.json",
                    },
                    "evidence": {
                        "bucket_count": 2,
                    },
                },
                {
                    "name": "p1_compute_bucket_plus_fixed_slot",
                    "env": {
                        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
                        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
                        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": "8",
                        "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": "1",
                        "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD": "8",
                        "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": "/tmp/sew_moe_p1_plan.json",
                    },
                    "evidence": {
                        "bucket_count": 2,
                        "recommended_num_slots": 8,
                        "recommended_exposed_host_to_hbm_bytes": 400,
                    },
                },
            ],
        }),
        encoding="utf-8",
    )
    calls = []

    def fake_run_smoke(args, _config, _requests):
        output_dir_for_case = os.path.abspath(args.output_dir)
        os.makedirs(output_dir_for_case, exist_ok=True)
        with open(os.path.join(output_dir_for_case, "outputs.jsonl"), "w", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "request_id": "inline_0000",
                    "output_text": "ok",
                    "output_token_ids": [1, 2],
                    "output_tokens": 2,
                }) + "\n"
            )
        calls.append({
            "name": args.experiment_name,
            "mode": args.mode,
            "num_slots": args.num_slots,
            "env_enabled": os.environ.get("VLLM_ASCEND_MOE_OFFLOAD_ENABLED", ""),
            "compute_bucket_plan": os.environ.get("VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH", ""),
            "arg_compute_bucket_plan": args.compute_bucket_plan_path,
            "output_dir": args.output_dir,
        })
        throughput_by_name = {
            "baseline": 10.0,
            "p1_fixed_slot_recommended": 12.0,
            "p1_compute_bucket_trace_only": 10.5,
            "p1_compute_bucket_fast_path": 11.0,
            "p1_compute_bucket_plus_fixed_slot": 13.0,
        }
        gate_events_by_name = {
            "baseline": [],
            "p1_fixed_slot_recommended": [],
            "p1_compute_bucket_trace_only": [],
            "p1_compute_bucket_fast_path": [
                {
                    "name": "compute_bucket_fast_path_gate",
                    "payload": {
                        "enabled": True,
                        "reason": "eligible",
                        "bucket_id": 0,
                        "original_expert_count": 4,
                        "compact_expert_count": 2,
                    },
                },
                {
                    "name": "compute_bucket_fast_path_gate",
                    "payload": {
                        "enabled": False,
                        "reason": "bucket_decision_fallback",
                        "bucket_id": None,
                        "original_expert_count": 4,
                        "compact_expert_count": 0,
                    },
                },
            ],
            "p1_compute_bucket_plus_fixed_slot": [
                {
                    "name": "compute_bucket_fast_path_gate",
                    "payload": {
                        "enabled": True,
                        "reason": "eligible",
                        "bucket_id": 1,
                        "original_expert_count": 4,
                        "compact_expert_count": 1,
                    },
                },
            ],
        }
        return {
            "status": "ok",
            "mode": args.mode,
            "output_throughput_tok_s": throughput_by_name[args.experiment_name],
            "moe_offload_profile": {
                "events": gate_events_by_name[args.experiment_name],
            },
        }

    monkeypatch.setattr("tools.sew_offload.run_p1_experiments.run_smoke", fake_run_smoke)

    args = argparse.Namespace(
        matrix=matrix_path,
        output_dir=output_dir,
        config="docs/sew-offload/benchmark_config.yaml",
        manifest=str(tmp_path / "manifest.jsonl"),
        model="/models/qwen3",
        inline_prompt="hello",
        inline_prompts_jsonl=None,
        inline_max_output_tokens=1,
        override_max_output_tokens=None,
        buckets="short_chat",
        max_requests=1,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        kv_cache_memory_mb=128,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        ignore_eos=True,
        resident_layer_ids="",
        release_original_expert_weights=False,
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        offload_params="experts",
        with_native_offload_backend=False,
    )

    summary = run_experiment_matrix(args, config={}, requests=[{"prompt": "hello", "max_output_tokens": 1}])

    assert [call["name"] for call in calls] == [
        "baseline",
        "p1_fixed_slot_recommended",
        "p1_compute_bucket_trace_only",
        "p1_compute_bucket_fast_path",
        "p1_compute_bucket_plus_fixed_slot",
    ]
    assert calls[0]["mode"] == "no_offload"
    assert calls[0]["env_enabled"] == ""
    assert calls[0]["compute_bucket_plan"] == ""
    assert calls[0]["arg_compute_bucket_plan"] == ""
    assert calls[1]["mode"] == "fixed_slot_sync"
    assert calls[1]["num_slots"] == 8
    assert calls[1]["env_enabled"] == "1"
    assert calls[1]["compute_bucket_plan"] == ""
    assert calls[1]["arg_compute_bucket_plan"] == ""
    assert calls[2]["mode"] == "trace_only"
    assert calls[2]["num_slots"] == 0
    assert calls[2]["env_enabled"] == "1"
    assert calls[2]["compute_bucket_plan"] == "/tmp/sew_moe_p1_plan.json"
    assert calls[2]["arg_compute_bucket_plan"] == "/tmp/sew_moe_p1_plan.json"
    assert calls[3]["mode"] == "compute_bucket_fast_path"
    assert calls[3]["num_slots"] == 0
    assert calls[3]["env_enabled"] == ""
    assert calls[3]["compute_bucket_plan"] == "/tmp/sew_moe_p1_plan.json"
    assert calls[3]["arg_compute_bucket_plan"] == "/tmp/sew_moe_p1_plan.json"
    assert calls[4]["mode"] == "fixed_slot_sync"
    assert calls[4]["num_slots"] == 8
    assert calls[4]["env_enabled"] == "1"
    assert calls[4]["compute_bucket_plan"] == "/tmp/sew_moe_p1_plan.json"
    assert calls[4]["arg_compute_bucket_plan"] == "/tmp/sew_moe_p1_plan.json"
    assert summary["experiments"][0]["evidence"] == {"phase": "decode"}
    assert summary["experiments"][1]["evidence"]["recommended_num_slots"] == 8
    assert summary["experiments"][2]["evidence"]["bucket_count"] == 2
    assert "correctness_vs_baseline" not in summary["experiments"][0]
    assert summary["experiments"][1]["correctness_vs_baseline"]["status"] == "ok"
    assert summary["experiments"][1]["correctness_vs_baseline"]["matched"] == 1
    assert summary["experiments"][2]["correctness_vs_baseline"]["status"] == "ok"
    assert summary["experiments"][3]["correctness_vs_baseline"]["status"] == "ok"
    assert summary["experiments"][4]["correctness_vs_baseline"]["status"] == "ok"
    assert (
        output_dir / "p1_compute_bucket_fast_path" / "correctness_compare.json"
    ).exists()
    assert summary["correctness_vs_baseline"] == {
        "status": "ok",
        "checked": 4,
        "failed": 0,
        "missing": 0,
    }
    assert summary["experiments"][0]["relative_to_baseline"]["output_throughput_tok_s_delta"] == 0.0
    assert summary["experiments"][0]["relative_to_baseline"]["output_throughput_tok_s_delta_percent"] == 0.0
    assert summary["experiments"][1]["relative_to_baseline"]["output_throughput_tok_s_delta"] == 2.0
    assert summary["experiments"][1]["relative_to_baseline"]["output_throughput_tok_s_delta_percent"] == 20.0
    assert summary["experiments"][2]["relative_to_baseline"]["output_throughput_tok_s_delta"] == 0.5
    assert summary["experiments"][2]["relative_to_baseline"]["output_throughput_tok_s_delta_percent"] == 5.0
    assert summary["experiments"][3]["relative_to_baseline"]["output_throughput_tok_s_delta"] == 1.0
    assert summary["experiments"][3]["relative_to_baseline"]["output_throughput_tok_s_delta_percent"] == 10.0
    assert summary["experiments"][4]["relative_to_baseline"]["output_throughput_tok_s_delta"] == 3.0
    assert summary["experiments"][4]["relative_to_baseline"]["output_throughput_tok_s_delta_percent"] == 30.0
    assert summary["best_by_output_throughput_tok_s"]["name"] == "p1_compute_bucket_plus_fixed_slot"
    assert [item["name"] for item in summary["throughput_delta_vs_baseline"]] == [
        "p1_compute_bucket_plus_fixed_slot",
        "p1_fixed_slot_recommended",
        "p1_compute_bucket_fast_path",
        "p1_compute_bucket_trace_only",
        "baseline",
    ]
    assert summary["throughput_delta_vs_baseline"][0]["output_throughput_tok_s_delta_percent"] == 30.0
    assert summary["experiments"][3]["compute_bucket_fast_path_gate_summary"] == {
        "total": 2,
        "enabled": 1,
        "fallback": 1,
        "enabled_percent": 50.0,
        "avg_original_expert_count": 4.0,
        "avg_compact_expert_count": 2.0,
        "avg_compaction_ratio": 0.5,
        "reasons": {
            "bucket_decision_fallback": 1,
            "eligible": 1,
        },
        "bucket_ids": {
            "0": 1,
        },
    }
    assert summary["experiments"][4]["compute_bucket_fast_path_gate_summary"]["enabled_percent"] == 100.0
    assert summary["compute_bucket_fast_path_gate_summary"] == {
        "total": 3,
        "enabled": 2,
        "fallback": 1,
        "enabled_percent": 66.66666666666666,
        "avg_original_expert_count": 4.0,
        "avg_compact_expert_count": 1.5,
        "avg_compaction_ratio": 0.375,
        "reasons": {
            "bucket_decision_fallback": 1,
            "eligible": 2,
        },
        "bucket_ids": {
            "0": 1,
            "1": 1,
        },
    }
    assert json.loads((output_dir / "p1_experiment_summary.json").read_text(encoding="utf-8")) == summary


def test_run_experiment_matrix_recommends_fastest_correct_case(tmp_path, monkeypatch):
    matrix_path = tmp_path / "experiments.json"
    output_dir = tmp_path / "runs"
    matrix_path.write_text(
        json.dumps({
            "version": 1,
            "experiments": [
                {"name": "baseline", "env": {}},
                {
                    "name": "p1_fast_but_wrong",
                    "env": {
                        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
                        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
                    },
                },
                {
                    "name": "p1_slower_but_correct",
                    "env": {
                        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
                        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
                    },
                },
            ],
        }),
        encoding="utf-8",
    )

    def fake_run_smoke(args, _config, _requests):
        case_output_dir = os.path.abspath(args.output_dir)
        os.makedirs(case_output_dir, exist_ok=True)
        tokens_by_name = {
            "baseline": [1, 2],
            "p1_fast_but_wrong": [9, 9],
            "p1_slower_but_correct": [1, 2],
        }
        throughput_by_name = {
            "baseline": 10.0,
            "p1_fast_but_wrong": 20.0,
            "p1_slower_but_correct": 15.0,
        }
        with open(os.path.join(case_output_dir, "outputs.jsonl"), "w", encoding="utf-8") as f:
            f.write(
                json.dumps({
                    "request_id": "inline_0000",
                    "output_text": "ok",
                    "output_token_ids": tokens_by_name[args.experiment_name],
                    "output_tokens": 2,
                }) + "\n"
            )
        return {
            "status": "ok",
            "mode": args.mode,
            "output_throughput_tok_s": throughput_by_name[args.experiment_name],
        }

    monkeypatch.setattr("tools.sew_offload.run_p1_experiments.run_smoke", fake_run_smoke)

    args = argparse.Namespace(
        matrix=matrix_path,
        output_dir=output_dir,
        config="docs/sew-offload/benchmark_config.yaml",
        manifest=str(tmp_path / "manifest.jsonl"),
        model="/models/qwen3",
        inline_prompt="hello",
        inline_prompts_jsonl=None,
        inline_max_output_tokens=1,
        override_max_output_tokens=None,
        buckets="short_chat",
        max_requests=1,
        max_model_len=128,
        max_num_seqs=1,
        max_num_batched_tokens=128,
        kv_cache_memory_mb=128,
        gpu_memory_utilization=0.9,
        enforce_eager=True,
        ignore_eos=True,
        resident_layer_ids="",
        release_original_expert_weights=False,
        offload_backend="prefetch",
        offload_group_size=4,
        offload_num_in_group=1,
        offload_prefetch_step=1,
        offload_params="experts",
        with_native_offload_backend=False,
    )

    summary = run_experiment_matrix(args, config={}, requests=[{"prompt": "hello", "max_output_tokens": 1}])

    assert summary["best_by_output_throughput_tok_s"]["name"] == "p1_fast_but_wrong"
    assert summary["recommended_correct_experiment"] == {
        "name": "p1_slower_but_correct",
        "status": "ok",
        "correctness_status": "ok",
        "output_throughput_tok_s": 15.0,
        "output_throughput_tok_s_delta": 5.0,
        "output_throughput_tok_s_delta_percent": 50.0,
    }
    assert summary["throughput_delta_vs_baseline"][0]["name"] == "p1_fast_but_wrong"
    assert summary["throughput_delta_vs_baseline_correct_only"][0]["name"] == "p1_slower_but_correct"
    assert json.loads((output_dir / "p1_experiment_summary.json").read_text(encoding="utf-8")) == summary
