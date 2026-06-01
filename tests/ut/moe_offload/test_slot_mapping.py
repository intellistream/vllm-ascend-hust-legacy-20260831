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

import pytest
import torch

from vllm_ascend.moe_offload.expert_key import ExpertKey
from vllm_ascend.moe_offload.slot_bank import ExpertSlotBank
from vllm_ascend.moe_offload.slot_mapping import ExpertSlotMapping, PreparedSlotWeights


def test_slot_mapping_builds_log2phy_for_active_ready_experts():
    bank = ExpertSlotBank(2, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))
    slot_a = bank.allocate_for(ExpertKey(3, 5), step_id=0)
    bank.mark_ready(slot_a.slot_id)
    slot_b = bank.allocate_for(ExpertKey(3, 7), step_id=1)
    bank.mark_ready(slot_b.slot_id)

    mapping = ExpertSlotMapping.from_slot_bank(
        layer_id=3,
        active_experts=(5, 7),
        num_logical_experts=8,
        slot_bank=bank,
        device=torch.device("cpu"),
    )

    assert mapping.logical_to_physical.tolist() == [-1, -1, -1, -1, -1, slot_a.slot_id, -1, slot_b.slot_id]
    topk_ids = torch.tensor([[5, 7], [7, 5]], dtype=torch.int64)
    assert mapping.remap_topk_ids(topk_ids).tolist() == [
        [slot_a.slot_id, slot_b.slot_id],
        [slot_b.slot_id, slot_a.slot_id],
    ]


def test_slot_mapping_rejects_missing_active_expert_before_backend():
    bank = ExpertSlotBank(1, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))
    slot = bank.allocate_for(ExpertKey(0, 1), step_id=0)
    bank.mark_ready(slot.slot_id)

    with pytest.raises(RuntimeError, match="active expert .* is not resident"):
        ExpertSlotMapping.from_slot_bank(
            layer_id=0,
            active_experts=(1, 2),
            num_logical_experts=3,
            slot_bank=bank,
            device=torch.device("cpu"),
        )


def test_slot_mapping_rejects_inactive_topk_ids():
    bank = ExpertSlotBank(1, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))
    slot = bank.allocate_for(ExpertKey(0, 1), step_id=0)
    bank.mark_ready(slot.slot_id)
    mapping = ExpertSlotMapping.from_slot_bank(
        layer_id=0,
        active_experts=(1,),
        num_logical_experts=3,
        slot_bank=bank,
        device=torch.device("cpu"),
    )

    with pytest.raises(RuntimeError, match="topk_ids contain experts without ready slots"):
        mapping.remap_topk_ids(torch.tensor([[1, 2]], dtype=torch.int64))


def test_prepared_slot_weights_exposes_stable_slot_backing_tensors():
    bank = ExpertSlotBank(2, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))
    initial_w13_ptr = bank.w13_slots.data_ptr()
    initial_w2_ptr = bank.w2_slots.data_ptr()
    slot = bank.allocate_for(ExpertKey(0, 1), step_id=0)
    bank.mark_ready(slot.slot_id)
    mapping = ExpertSlotMapping.from_slot_bank(
        layer_id=0,
        active_experts=(1,),
        num_logical_experts=2,
        slot_bank=bank,
        device=torch.device("cpu"),
    )

    prepared = PreparedSlotWeights.from_slot_bank(slot_bank=bank, mapping=mapping)

    assert prepared.w1 is bank.w13_slots
    assert prepared.w2 is bank.w2_slots
    assert prepared.w1.shape == (2, 2, 4)
    assert prepared.w2.shape == (2, 4, 2)
    assert prepared.log2phy is mapping.logical_to_physical
    assert prepared.physical_expert_count == 2
    assert prepared.w1.data_ptr() == initial_w13_ptr
    assert prepared.w2.data_ptr() == initial_w2_ptr


def test_prepared_slot_weights_backend_ready_rejects_physical_count_mismatch():
    bank = ExpertSlotBank(2, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))
    slot = bank.allocate_for(ExpertKey(0, 1), step_id=0)
    bank.mark_ready(slot.slot_id)
    mapping = ExpertSlotMapping.from_slot_bank(
        layer_id=0,
        active_experts=(1,),
        num_logical_experts=2,
        slot_bank=bank,
        device=torch.device("cpu"),
    )
    prepared = PreparedSlotWeights(
        w1=bank.w13_slots,
        w2=bank.w2_slots,
        log2phy=mapping.logical_to_physical,
        physical_expert_count=1,
        mapping=mapping,
    )

    with pytest.raises(ValueError, match="w1 physical expert count mismatch"):
        prepared.validate_backend_ready(expected_device_type="cpu")
