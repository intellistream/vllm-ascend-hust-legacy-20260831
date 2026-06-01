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

from vllm_ascend.moe_offload.expert_key import ExpertKey
from vllm_ascend.moe_offload.policy import LruPolicy, StickyLayerLruPolicy, make_policy


def test_make_policy_returns_supported_policy_instances():
    assert isinstance(make_policy("lru"), LruPolicy)
    assert isinstance(make_policy("sticky_layer_lru"), StickyLayerLruPolicy)


def test_lru_policy_evicts_oldest_access():
    policy = LruPolicy()
    keys = [ExpertKey(0, 0), ExpertKey(0, 1)]
    last_used = {keys[0]: 7, keys[1]: 3}

    assert policy.choose_victim(keys, last_used=last_used, incoming=ExpertKey(0, 2)) == keys[1]


def test_sticky_layer_lru_keeps_same_layer_expert_when_possible():
    policy = StickyLayerLruPolicy()
    same_layer = ExpertKey(3, 0)
    other_layer = ExpertKey(2, 1)
    last_used = {same_layer: 1, other_layer: 9}

    assert policy.choose_victim([same_layer, other_layer], last_used=last_used, incoming=ExpertKey(3, 2)) == other_layer
