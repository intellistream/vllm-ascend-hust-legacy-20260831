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

import pytest

from tools.sew_offload.compare_smoke_outputs import (
    compare_output_records,
    load_outputs_jsonl,
    write_comparison_summary,
)


def test_compare_output_records_accepts_exact_token_match():
    baseline = [
        {
            "request_id": "r0",
            "output_text": "hello",
            "output_token_ids": [1, 2, 3],
            "output_tokens": 3,
        }
    ]
    candidate = [
        {
            "request_id": "r0",
            "output_text": "hello",
            "output_token_ids": [1, 2, 3],
            "output_tokens": 3,
        }
    ]

    summary = compare_output_records(baseline, candidate)

    assert summary == {
        "status": "ok",
        "matched": 1,
        "mismatched": 0,
        "missing": 0,
        "extra": 0,
        "mismatches": [],
    }


def test_compare_output_records_reports_token_mismatch_without_tolerance():
    baseline = [
        {
            "request_id": "r0",
            "output_text": "hello",
            "output_token_ids": [1, 2, 3],
            "output_tokens": 3,
        }
    ]
    candidate = [
        {
            "request_id": "r0",
            "output_text": "help",
            "output_token_ids": [1, 2, 4],
            "output_tokens": 3,
        }
    ]

    summary = compare_output_records(baseline, candidate)

    assert summary["status"] == "failed"
    assert summary["matched"] == 0
    assert summary["mismatched"] == 1
    assert summary["mismatches"] == [
        {
            "request_id": "r0",
            "reason": "token_ids differ",
            "baseline_output_tokens": 3,
            "candidate_output_tokens": 3,
        }
    ]


def test_load_outputs_jsonl_rejects_duplicate_request_id(tmp_path):
    path = tmp_path / "outputs.jsonl"
    path.write_text(
        json.dumps({"request_id": "r0", "output_token_ids": [1], "output_tokens": 1}) + "\n"
        + json.dumps({"request_id": "r0", "output_token_ids": [1], "output_tokens": 1}) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="duplicate request_id"):
        load_outputs_jsonl(path)


def test_write_comparison_summary_writes_json(tmp_path):
    output_path = tmp_path / "comparison.json"

    summary = write_comparison_summary(
        baseline_outputs=[{"request_id": "r0", "output_token_ids": [1], "output_tokens": 1}],
        candidate_outputs=[{"request_id": "r0", "output_token_ids": [1], "output_tokens": 1}],
        output_path=output_path,
    )

    assert summary["status"] == "ok"
    assert json.loads(output_path.read_text(encoding="utf-8"))["matched"] == 1
