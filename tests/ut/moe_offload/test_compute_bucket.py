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

import torch

from vllm_ascend.moe_offload.compute_bucket import (
    ComputeBucketClassifier,
    ComputeBucketDecisionPath,
    load_compute_bucket_classifier,
)


def test_compute_bucket_classifier_loads_plan_and_matches_counts_signature(tmp_path):
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
                        "mode": "trace_only",
                        "total_grouped_records": 10,
                        "coverage_percent": 80.0,
                        "fallback_percent": 20.0,
                        "buckets": [
                            {
                                "bucket_id": 0,
                                "signature": "counts:1,2,1",
                                "sample_count": 8,
                                "coverage_percent": 80.0,
                            }
                        ],
                    },
                }
            ],
        }),
        encoding="utf-8",
    )

    classifier = load_compute_bucket_classifier(plan_path)
    decision = classifier.classify(
        group_list=torch.tensor([1, 2, 1], dtype=torch.int64),
        group_list_type=1,
        phase="decode",
    )

    assert decision.path is ComputeBucketDecisionPath.BUCKET
    assert decision.bucket_id == 0
    assert decision.signature == "counts:1,2,1"
    assert decision.reason == "signature_matched"
    assert decision.to_jsonable()["path"] == "bucket"
    assert decision.bucket.active_expert_ids == (0, 1, 2)
    assert decision.bucket.compact_group_list == (1, 2, 1)
    assert decision.to_jsonable()["bucket"]["active_expert_ids"] == [0, 1, 2]
    assert decision.to_jsonable()["bucket"]["compact_group_list"] == [1, 2, 1]


def test_compute_bucket_classifier_derives_active_expert_plan_from_sparse_counts():
    classifier = ComputeBucketClassifier.from_plan({
        "version": 1,
        "phase": "decode",
        "buckets": [
            {
                "bucket_id": 5,
                "signature": "counts:2,0,1,0",
                "sample_count": 9,
                "coverage_percent": 90.0,
            }
        ],
    })

    decision = classifier.classify(
        group_list=torch.tensor([2, 0, 1, 0], dtype=torch.int64),
        group_list_type=1,
        phase="decode",
    )

    assert decision.path is ComputeBucketDecisionPath.BUCKET
    assert decision.bucket.active_expert_ids == (0, 2)
    assert decision.bucket.compact_group_list == (2, 1)
    assert decision.bucket.original_expert_count == 4
    assert decision.bucket.compact_expert_count == 2


def test_compute_bucket_classifier_falls_back_for_unplanned_signature():
    classifier = ComputeBucketClassifier.from_plan({
        "version": 1,
        "phase": "decode",
        "buckets": [
            {
                "bucket_id": 2,
                "signature": "counts:2,0,1",
                "sample_count": 5,
                "coverage_percent": 50.0,
            }
        ],
    })

    decision = classifier.classify(
        group_list=torch.tensor([1, 1, 1], dtype=torch.int64),
        group_list_type=1,
        phase="decode",
    )

    assert decision.path is ComputeBucketDecisionPath.FALLBACK
    assert decision.bucket_id is None
    assert decision.signature == "counts:1,1,1"
    assert decision.reason == "signature_not_planned"


def test_compute_bucket_classifier_rejects_phase_mismatch_and_cumsum_signature():
    classifier = ComputeBucketClassifier.from_plan({
        "version": 1,
        "phase": "decode",
        "buckets": [
            {
                "bucket_id": 3,
                "signature": "cumsum:0,3,5",
                "sample_count": 4,
                "coverage_percent": 40.0,
            }
        ],
    })

    phase_decision = classifier.classify(
        group_list=torch.tensor([0, 3, 5], dtype=torch.int64),
        group_list_type=0,
        phase="prefill",
    )
    decode_decision = classifier.classify(
        group_list=torch.tensor([0, 3, 5], dtype=torch.int64),
        group_list_type=0,
        phase="decode",
    )

    assert phase_decision.path is ComputeBucketDecisionPath.FALLBACK
    assert phase_decision.reason == "phase_mismatch"
    assert decode_decision.path is ComputeBucketDecisionPath.BUCKET
    assert decode_decision.bucket_id == 3


def test_compute_bucket_classifier_derives_active_expert_plan_from_cumsum():
    classifier = ComputeBucketClassifier.from_plan({
        "version": 1,
        "phase": "decode",
        "buckets": [
            {
                "bucket_id": 4,
                "signature": "cumsum:0,2,2,5",
                "sample_count": 6,
                "coverage_percent": 60.0,
            }
        ],
    })

    decision = classifier.classify(
        group_list=torch.tensor([0, 2, 2, 5], dtype=torch.int64),
        group_list_type=0,
        phase="decode",
    )

    assert decision.path is ComputeBucketDecisionPath.BUCKET
    assert decision.bucket.active_expert_ids == (1, 3)
    assert decision.bucket.compact_group_list == (2, 5)
    assert decision.bucket.original_expert_count == 4
    assert decision.bucket.compact_expert_count == 2


def test_compute_bucket_classifier_mixed_phase_matches_prefill_and_decode():
    classifier = ComputeBucketClassifier.from_plan({
        "version": 1,
        "phase": "mixed",
        "buckets": [
            {
                "bucket_id": 1,
                "signature": "counts:1,0,2",
                "sample_count": 3,
            }
        ],
    })

    decode_decision = classifier.classify(
        group_list=torch.tensor([1, 0, 2], dtype=torch.int64),
        group_list_type=1,
        phase="decode",
    )
    prefill_decision = classifier.classify(
        group_list=torch.tensor([1, 0, 2], dtype=torch.int64),
        group_list_type=1,
        phase="prefill",
    )

    assert decode_decision.path is ComputeBucketDecisionPath.BUCKET
    assert prefill_decision.path is ComputeBucketDecisionPath.BUCKET
