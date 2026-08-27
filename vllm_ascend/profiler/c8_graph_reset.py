#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Copyright 2026 The vLLM team.
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
#

from typing import Any

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper


class C8GraphResetWorkerExtension:
    """Explicit dev-only hook for a profiler-visible ACL graph recapture."""

    def reset_c8_acl_graphs_for_profiling(self) -> dict[str, Any]:
        wrappers = list(ACLGraphWrapper._all_instances)
        entries_before = sum(len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers)

        torch.accelerator.synchronize()
        ACLGraphWrapper.clear_all_graphs()
        torch.accelerator.synchronize()

        entries_after = sum(len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers)
        status = "PASS" if wrappers and entries_before > 0 and entries_after == 0 else "FAIL"
        capture_enabled = status == "PASS"
        set_cudagraph_capturing_enabled(capture_enabled)
        return {
            "schema_version": "vllm-ascend-c8-graph-reset/v1",
            "status": status,
            "wrapper_count": len(wrappers),
            "graph_entries_before": entries_before,
            "graph_entries_after": entries_after,
            "cudagraph_capturing_enabled_after": capture_enabled,
            "caller_precondition": "engine idle with no in-flight requests",
            "synchronization_scope": (
                "one synchronization before and after the explicit reset; never on the inference hot path"
            ),
        }

    def seal_c8_acl_graph_recapture_for_profiling(self) -> dict[str, Any]:
        """Disable runtime capture after proving the requested recapture."""
        wrappers = list(ACLGraphWrapper._all_instances)

        torch.accelerator.synchronize()
        entries_after_recapture = sum(len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers)
        set_cudagraph_capturing_enabled(False)
        torch.accelerator.synchronize()

        status = "PASS" if wrappers and entries_after_recapture > 0 else "FAIL"
        return {
            "schema_version": "vllm-ascend-c8-graph-recapture/v1",
            "status": status,
            "wrapper_count": len(wrappers),
            "graph_entries_after_recapture": entries_after_recapture,
            "cudagraph_capturing_enabled_after": False,
            "caller_precondition": "engine idle after exactly one requested recapture",
            "synchronization_scope": (
                "one synchronization before and after sealing the explicit recapture; never on the inference hot path"
            ),
        }
