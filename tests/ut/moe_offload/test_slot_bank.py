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
from vllm_ascend.moe_offload.slot_bank import ExpertSlotBank, SlotState


def test_slot_bank_allocates_stable_slot_addresses():
    bank = ExpertSlotBank(
        num_slots=2,
        w13_shape=(2, 4),
        w2_shape=(4, 2),
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    original_ptrs = [(slot.w13.data_ptr(), slot.w2.data_ptr()) for slot in bank.slots]

    slot = bank.allocate_for(ExpertKey(0, 1), step_id=0)
    bank.mark_ready(slot.slot_id)
    same_slot = bank.allocate_for(ExpertKey(0, 1), step_id=1)

    assert slot.slot_id == same_slot.slot_id
    assert same_slot.state == SlotState.READY
    assert [(slot.w13.data_ptr(), slot.w2.data_ptr()) for slot in bank.slots] == original_ptrs


def test_slot_bank_evicts_lru_ready_slot_and_increments_version():
    bank = ExpertSlotBank(2, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))
    first = bank.allocate_for(ExpertKey(0, 0), step_id=0)
    bank.mark_ready(first.slot_id)
    second = bank.allocate_for(ExpertKey(0, 1), step_id=1)
    bank.mark_ready(second.slot_id)

    replacement = bank.allocate_for(ExpertKey(0, 2), step_id=2)

    assert replacement.slot_id == first.slot_id
    assert replacement.expert_key == ExpertKey(0, 2)
    assert replacement.state == SlotState.LOADING
    assert replacement.version == 2


def test_slot_bank_does_not_evict_computing_slots():
    bank = ExpertSlotBank(1, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))
    slot = bank.allocate_for(ExpertKey(0, 0), step_id=0)
    bank.mark_ready(slot.slot_id)
    bank.mark_computing(slot.slot_id)

    with pytest.raises(RuntimeError, match="no evictable expert slots"):
        bank.allocate_for(ExpertKey(0, 1), step_id=1)


def test_slot_bank_reports_backing_tensor_bytes():
    bank = ExpertSlotBank(2, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))

    expected_bytes = bank.w13_slots.numel() * bank.w13_slots.element_size()
    expected_bytes += bank.w2_slots.numel() * bank.w2_slots.element_size()
    assert bank.total_bytes == expected_bytes
