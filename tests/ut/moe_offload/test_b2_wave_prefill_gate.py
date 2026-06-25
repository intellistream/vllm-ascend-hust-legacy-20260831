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

"""B2 wave-streamed prefill gate predicate (step 1: gate + config, no compute)."""

from vllm_ascend.moe_offload.config import MoeOffloadConfig
from vllm_ascend.moe_offload.runtime import MoeOffloadRuntime


def _runtime(num_slots=8, b2=True, resident=frozenset()):
    return MoeOffloadRuntime(
        MoeOffloadConfig(
            enabled=True,
            trace_only=False,
            num_slots=num_slots,
            b2_wave_prefill=b2,
            resident_layer_ids=resident,
        )
    )


class TestB2WavePrefillGate:
    def test_fires_on_prefill_high_fanout_offloaded_layer(self):
        rt = _runtime(num_slots=8, b2=True)
        # offloaded layer (not resident), prefill, fanout 51 > 8 -> B2.
        assert rt.should_use_b2_wave_prefill(
            layer_id=2, active_expert_count=51, is_prefill=True
        )

    def test_off_when_config_disabled(self):
        rt = _runtime(num_slots=8, b2=False)
        assert not rt.should_use_b2_wave_prefill(
            layer_id=2, active_expert_count=51, is_prefill=True
        )

    def test_off_for_decode(self):
        rt = _runtime(num_slots=8, b2=True)
        assert not rt.should_use_b2_wave_prefill(
            layer_id=2, active_expert_count=51, is_prefill=False
        )

    def test_off_when_fanout_fits_slots(self):
        rt = _runtime(num_slots=96, b2=True)
        # fanout 51 <= 96 -> B1 single wave already fits, no B2.
        assert not rt.should_use_b2_wave_prefill(
            layer_id=2, active_expert_count=51, is_prefill=True
        )

    def test_off_at_exact_capacity(self):
        rt = _runtime(num_slots=8, b2=True)
        assert not rt.should_use_b2_wave_prefill(
            layer_id=2, active_expert_count=8, is_prefill=True
        )
        # strictly greater triggers
        assert rt.should_use_b2_wave_prefill(
            layer_id=2, active_expert_count=9, is_prefill=True
        )

    def test_off_for_resident_layer(self):
        rt = _runtime(num_slots=8, b2=True, resident=frozenset({2}))
        # layer 2 is resident -> not an offloaded fixed-slot layer -> no B2.
        assert not rt.should_use_b2_wave_prefill(
            layer_id=2, active_expert_count=51, is_prefill=True
        )

    def test_default_config_off(self):
        # b2_wave_prefill defaults to False -> gate never fires.
        rt = MoeOffloadRuntime(
            MoeOffloadConfig(enabled=True, trace_only=False, num_slots=8)
        )
        assert not rt.should_use_b2_wave_prefill(
            layer_id=2, active_expert_count=51, is_prefill=True
        )
