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
import subprocess
import sys
from pathlib import Path

from tools.sew_offload.estimate_fixed_slot_memory import estimate_fixed_slot_memory


def test_estimate_fixed_slot_memory_reports_original_host_and_slot_bytes():
    summary = estimate_fixed_slot_memory(
        num_layers=2,
        num_experts_per_layer=3,
        expert_bytes=10,
        num_slots=2,
        original_weights_retained=True,
        host_store_enabled=True,
    )

    assert summary["original_expert_weight_bytes"] == 60
    assert summary["host_store_bytes"] == 60
    assert summary["slot_bank_bytes"] == 40
    assert summary["total_managed_bytes"] == 160
    assert summary["incremental_runtime_bytes"] == 100
    assert summary["original_weights_retained"] is True


def test_estimate_fixed_slot_memory_cli_outputs_json_summary(tmp_path):
    repo_root = Path(__file__).resolve().parents[3]
    output_path = tmp_path / "memory.json"

    subprocess.run(
        [
            sys.executable,
            "tools/sew_offload/estimate_fixed_slot_memory.py",
            "--num-layers",
            "2",
            "--num-experts-per-layer",
            "3",
            "--expert-bytes",
            "10",
            "--num-slots",
            "2",
            "--output",
            str(output_path),
        ],
        cwd=repo_root,
        check=True,
    )

    summary = json.loads(output_path.read_text(encoding="utf-8"))
    assert summary["slot_bank_bytes"] == 40
    assert summary["total_managed_bytes"] == 160
