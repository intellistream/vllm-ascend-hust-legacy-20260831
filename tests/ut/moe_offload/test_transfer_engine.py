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

import torch

from vllm_ascend.moe_offload.expert_key import ExpertKey
from vllm_ascend.moe_offload.host_store import ExpertWeightBundle
from vllm_ascend.moe_offload.slot_bank import ExpertSlotBank, SlotState
from vllm_ascend.moe_offload.transfer_engine import TransferEngine


def test_transfer_engine_sync_load_copies_bundle_into_slot_and_marks_ready():
    bank = ExpertSlotBank(1, (2, 4), (4, 2), dtype=torch.float32, device=torch.device("cpu"))
    slot = bank.allocate_for(ExpertKey(0, 1), step_id=0)
    bundle = ExpertWeightBundle(
        layer_id=0,
        expert_id=1,
        w13=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        w2=torch.arange(8, dtype=torch.float32).reshape(4, 2),
    )

    TransferEngine().load_sync(bundle, slot)

    assert slot.state == SlotState.READY
    assert torch.equal(slot.w13, bundle.w13)
    assert torch.equal(slot.w2, bundle.w2)
    assert slot.w13.data_ptr() != bundle.w13.data_ptr()
    assert slot.w2.data_ptr() != bundle.w2.data_ptr()
