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

from types import SimpleNamespace

import pytest
import torch

from vllm_ascend.moe_offload.host_store import ExpertWeightBundle, HostExpertStore


def test_host_expert_store_registers_unquantized_layer_by_expert_id():
    layer = SimpleNamespace(
        layer_id=7,
        w13_weight=torch.arange(3 * 2 * 4, dtype=torch.float32).reshape(3, 2, 4),
        w2_weight=torch.arange(3 * 4 * 2, dtype=torch.float32).reshape(3, 4, 2),
    )

    store = HostExpertStore()
    store.register_layer(layer)
    bundle = store.get(7, 2)

    assert isinstance(bundle, ExpertWeightBundle)
    assert bundle.layer_id == 7
    assert bundle.expert_id == 2
    assert torch.equal(bundle.w13, layer.w13_weight[2])
    assert torch.equal(bundle.w2, layer.w2_weight[2])
    assert bundle.w13.data_ptr() != layer.w13_weight[2].data_ptr()
    assert bundle.w2.data_ptr() != layer.w2_weight[2].data_ptr()


def test_host_expert_store_rejects_mismatched_expert_count():
    layer = SimpleNamespace(
        layer_id=0,
        w13_weight=torch.randn(2, 2, 4),
        w2_weight=torch.randn(3, 4, 2),
    )

    with pytest.raises(ValueError, match="same number of experts"):
        HostExpertStore().register_layer(layer)


def test_host_expert_store_rejects_empty_expert_layer():
    layer = SimpleNamespace(
        layer_id=0,
        w13_weight=torch.randn(0, 2, 4),
        w2_weight=torch.randn(0, 4, 2),
    )

    with pytest.raises(ValueError, match="at least one expert"):
        HostExpertStore().register_layer(layer)


def test_host_expert_store_missing_expert_raises_key_error():
    store = HostExpertStore()

    with pytest.raises(KeyError):
        store.get(0, 0)


def test_host_expert_store_reports_total_cloned_weight_bytes():
    layer = SimpleNamespace(
        layer_id=7,
        w13_weight=torch.zeros(3, 2, 4, dtype=torch.float32),
        w2_weight=torch.zeros(3, 4, 2, dtype=torch.float32),
    )
    store = HostExpertStore()

    store.register_layer(layer)

    expected_bytes = layer.w13_weight.numel() * layer.w13_weight.element_size()
    expected_bytes += layer.w2_weight.numel() * layer.w2_weight.element_size()
    assert store.total_bytes == expected_bytes


def test_host_expert_store_validates_complete_registered_layer():
    layer = SimpleNamespace(
        layer_id=7,
        w13_weight=torch.zeros(3, 2, 4, dtype=torch.float32),
        w2_weight=torch.zeros(3, 4, 2, dtype=torch.float32),
    )
    store = HostExpertStore()
    store.register_layer(layer)

    report = store.validate_complete_layers((7,))

    assert report.complete
    assert report.layers_checked == (7,)
    assert report.blockers == ()


def test_host_expert_store_self_check_reports_missing_layer_and_expert():
    store = HostExpertStore()

    missing_layer_report = store.validate_complete_layers((7,))
    assert not missing_layer_report.complete
    assert "host_store_missing_layers:[7]" in missing_layer_report.blockers

    layer = SimpleNamespace(
        layer_id=7,
        w13_weight=torch.zeros(3, 2, 4, dtype=torch.float32),
        w2_weight=torch.zeros(3, 4, 2, dtype=torch.float32),
    )
    store.register_layer(layer)
    store._weights.pop(next(key for key in store._weights if key.layer_id == 7 and key.expert_id == 1))

    missing_expert_report = store.validate_complete_layers((7,))
    assert not missing_expert_report.complete
    assert "host_store_missing_experts:layer=7,experts=[1]" in missing_expert_report.blockers


def test_host_expert_store_self_check_reports_layout_mismatch():
    layer = SimpleNamespace(
        layer_id=7,
        w13_weight=torch.zeros(3, 2, 4, dtype=torch.float32),
        w2_weight=torch.zeros(3, 4, 2, dtype=torch.float32),
    )
    store = HostExpertStore()
    store.register_layer(layer)
    store._weights[next(key for key in store._weights if key.layer_id == 7 and key.expert_id == 1)] = ExpertWeightBundle(
        layer_id=7,
        expert_id=1,
        w13=torch.zeros(2, 5, dtype=torch.float32),
        w2=torch.zeros(4, 2, dtype=torch.float32),
    )

    report = store.validate_complete_layers((7,))

    assert not report.complete
    assert "host_store_layout_mismatch:layer=7,expert=1,w13" in report.blockers


def test_host_expert_store_self_check_reports_stride_mismatch():
    layer = SimpleNamespace(
        layer_id=7,
        w13_weight=torch.zeros(3, 2, 4, dtype=torch.float32),
        w2_weight=torch.zeros(3, 4, 2, dtype=torch.float32),
    )
    store = HostExpertStore()
    store.register_layer(layer)
    key = next(key for key in store._weights if key.layer_id == 7 and key.expert_id == 1)
    store._weights[key] = ExpertWeightBundle(
        layer_id=7,
        expert_id=1,
        w13=torch.zeros(4, 2, dtype=torch.float32).t(),
        w2=torch.zeros(4, 2, dtype=torch.float32),
    )

    report = store.validate_complete_layers((7,))

    assert not report.complete
    assert "host_store_layout_mismatch:layer=7,expert=1,w13_stride" in report.blockers


def test_host_expert_store_self_check_reports_non_cpu_bundle():
    layer = SimpleNamespace(
        layer_id=7,
        w13_weight=torch.zeros(3, 2, 4, dtype=torch.float32),
        w2_weight=torch.zeros(3, 4, 2, dtype=torch.float32),
    )
    store = HostExpertStore()
    store.register_layer(layer)
    key = next(key for key in store._weights if key.layer_id == 7 and key.expert_id == 1)
    store._weights[key] = ExpertWeightBundle(
        layer_id=7,
        expert_id=1,
        w13=torch.zeros(2, 4, dtype=torch.float32, device="meta"),
        w2=torch.zeros(4, 2, dtype=torch.float32),
    )

    report = store.validate_complete_layers((7,))

    assert not report.complete
    assert "host_store_device_mismatch:layer=7,expert=1,w13=meta" in report.blockers
