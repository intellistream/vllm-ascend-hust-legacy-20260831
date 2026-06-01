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

from vllm_ascend.moe_offload.host_store import ExpertWeightBundle
from vllm_ascend.moe_offload.layout import LayoutSignature, LayoutValidator


def _bundle(w13: torch.Tensor, w2: torch.Tensor) -> ExpertWeightBundle:
    return ExpertWeightBundle(layer_id=0, expert_id=1, w13=w13, w2=w2)


def test_layout_signature_tracks_shape_dtype_stride_and_device():
    tensor = torch.randn(2, 4, dtype=torch.float32)

    signature = LayoutSignature.from_tensor(tensor)

    assert signature.shape == (2, 4)
    assert signature.dtype == torch.float32
    assert signature.stride == tensor.stride()
    assert signature.device_type == "cpu"


def test_layout_validator_accepts_matching_bundle_and_slot():
    bundle = _bundle(torch.randn(2, 4), torch.randn(4, 2))
    slot = _bundle(torch.empty(2, 4), torch.empty(4, 2))

    LayoutValidator.validate_bundle_matches_slot(bundle, slot)


def test_layout_validator_rejects_shape_mismatch():
    bundle = _bundle(torch.randn(2, 4), torch.randn(4, 2))
    slot = _bundle(torch.empty(2, 5), torch.empty(4, 2))

    with pytest.raises(ValueError, match="w13 layout mismatch"):
        LayoutValidator.validate_bundle_matches_slot(bundle, slot)


def test_layout_validator_rejects_device_mismatch():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable for device mismatch construction.")
    bundle = _bundle(torch.randn(2, 4), torch.randn(4, 2))
    slot = _bundle(torch.empty(2, 4, device="cuda"), torch.empty(4, 2, device="cuda"))

    with pytest.raises(ValueError, match="device"):
        LayoutValidator.validate_backend_ready(slot, expected_device_type="cpu")


def test_layout_validator_allows_copy_compatible_cross_device_layout():
    bundle = _bundle(torch.randn(2, 4), torch.randn(4, 2))
    slot = _bundle(torch.empty(2, 4), torch.empty(4, 2))

    LayoutValidator.validate_copy_compatible(bundle, slot)
