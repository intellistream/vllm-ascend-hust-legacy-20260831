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

import json

from tools.sew_offload.materialize_p1_experiments import materialize_experiments


def test_materialize_p1_experiments_from_compute_and_slot_plan(tmp_path):
    plan_path = tmp_path / "sew_moe_p1_plan.json"
    plan_path.write_text(
        json.dumps({
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
                },
                {
                    "phase": "decode",
                    "target": "P1-T",
                    "slot_sweep_result": {
                        "recommended_num_slots": 16,
                        "recommended_miss_count": 12,
                        "recommended_host_to_hbm_bytes": 1200,
                        "recommended_prefetchable_miss_count": 8,
                        "recommended_exposed_miss_count": 4,
                        "recommended_prefetchable_host_to_hbm_bytes": 800,
                        "recommended_exposed_host_to_hbm_bytes": 400,
                    },
                    "slot_sweep_result_json": "/tmp/decode/slot_sweep_lru.json",
                },
            ],
        }),
        encoding="utf-8",
    )

    matrix = materialize_experiments(plan_path)

    assert matrix["version"] == 1
    assert matrix["source_plan"] == str(plan_path)
    assert [case["name"] for case in matrix["experiments"]] == [
        "baseline",
        "p1_compute_bucket_trace_only",
        "p1_compute_bucket_fast_path",
        "p1_fixed_slot_recommended",
        "p1_compute_bucket_plus_fixed_slot",
    ]
    assert matrix["experiments"][0]["env"] == {}
    assert matrix["experiments"][1]["env"] == {
        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "1",
        "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": str(plan_path),
    }
    assert matrix["experiments"][2]["env"] == {
        "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": str(plan_path),
    }
    assert matrix["experiments"][3]["env"] == {
        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": "16",
        "VLLM_ASCEND_MOE_OFFLOAD_POLICY": "lru",
        "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": "1",
        "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD": "16",
    }
    assert matrix["experiments"][3]["evidence"]["slot_sweep_result_json"] == "/tmp/decode/slot_sweep_lru.json"
    assert matrix["experiments"][3]["evidence"]["recommended_prefetchable_miss_count"] == 8
    assert matrix["experiments"][3]["evidence"]["recommended_exposed_miss_count"] == 4
    assert matrix["experiments"][3]["evidence"]["recommended_prefetchable_host_to_hbm_bytes"] == 800
    assert matrix["experiments"][3]["evidence"]["recommended_exposed_host_to_hbm_bytes"] == 400
    assert matrix["experiments"][4]["env"] == {
        "VLLM_ASCEND_MOE_OFFLOAD_ENABLED": "1",
        "VLLM_ASCEND_MOE_OFFLOAD_TRACE_ONLY": "0",
        "VLLM_ASCEND_MOE_OFFLOAD_NUM_SLOTS": "16",
        "VLLM_ASCEND_MOE_OFFLOAD_POLICY": "lru",
        "VLLM_ASCEND_MOE_OFFLOAD_LAYERED_RUNTIME": "1",
        "VLLM_ASCEND_MOE_OFFLOAD_FANOUT_THRESHOLD": "16",
        "VLLM_ASCEND_MOE_COMPUTE_BUCKET_PLAN_PATH": str(plan_path),
    }
    assert matrix["experiments"][4]["evidence"]["bucket_count"] == 1
    assert matrix["experiments"][4]["evidence"]["recommended_exposed_host_to_hbm_bytes"] == 400


def test_materialize_p1_experiments_writes_output(tmp_path):
    plan_path = tmp_path / "sew_moe_p1_plan.json"
    output_path = tmp_path / "experiments.json"
    plan_path.write_text(json.dumps({"version": 1, "plans": []}), encoding="utf-8")

    matrix = materialize_experiments(plan_path, output_path=output_path)

    assert matrix["experiments"][0]["name"] == "baseline"
    assert json.loads(output_path.read_text(encoding="utf-8")) == matrix
