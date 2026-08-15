# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.

from vllm.config.speculative_capability import resolve_speculative_capability

from vllm_ascend.platform import NPUPlatform


def test_dspark_proposer_capability_resolves_before_device_allocation() -> None:
    capabilities = NPUPlatform.get_speculative_proposer_capabilities()

    assert capabilities["mtp"] == "builtin:mtp"
    assert capabilities["dspark"] == ("vllm_ascend.spec_decode.dspark_proposer.AscendDSparkProposer")

    capability = resolve_speculative_capability(
        requested_method="dspark",
        hf_config={"dspark_noise_token_id": 128799},
        platform=NPUPlatform.device_name,
        registered_proposers=capabilities,
    )

    assert capability.status == "enabled"
    assert capability.detected_checkpoint_method == "dspark"
    assert capability.resolved_method == "dspark"
