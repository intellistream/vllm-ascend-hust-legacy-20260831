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

import gc
from typing import Any

import torch
from vllm.compilation.monitor import set_cudagraph_capturing_enabled
from vllm.platforms import current_platform

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.device_allocator.sleep_mem_optimized import (
    AclGraphSleepWakeupManager,
)


class C8GraphResetWorkerExtension:
    """Explicit dev-only hook for a profiler-visible ACL graph recapture."""

    def reset_c8_acl_graphs_for_profiling(self) -> dict[str, Any]:
        wrappers = list(ACLGraphWrapper._all_instances)
        entries_before = sum(len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers)
        graph_objects_reset = 0
        graph_pool_rebound = False
        old_graph_pool_count = len({repr(wrapper.graph_pool) for wrapper in wrappers})
        reset_error = None

        set_cudagraph_capturing_enabled(False)
        torch.accelerator.synchronize()
        try:
            # Dropping only the Python dictionaries can leave the shared graph
            # pool referenced. Explicitly release every graph before clearing
            # the remaining capture metadata.
            for wrapper in wrappers:
                for entry in wrapper.concrete_aclgraph_entries.values():
                    if entry.aclgraph is None:
                        continue
                    entry.aclgraph.reset()
                    graph_objects_reset += 1
            if not wrappers or entries_before == 0:
                raise RuntimeError("no captured ACL graphs were available for reset")
            if graph_objects_reset != entries_before:
                raise RuntimeError("not every captured ACL graph entry held a releasable graph object")

            AclGraphSleepWakeupManager.clear_all_attention_workspaces()
            AclGraphSleepWakeupManager.reset_all_graph_params()
            AclGraphSleepWakeupManager.reset_model_runner_graph_manager(self.model_runner)
            gc.collect()
            torch.npu.empty_cache()

            old_graph_pools = [wrapper.graph_pool for wrapper in wrappers]
            fresh_graph_pool = current_platform.graph_pool_handle()
            if any(fresh_graph_pool == old_pool for old_pool in old_graph_pools):
                raise RuntimeError("the replacement ACL graph pool is not fresh")
            current_platform.__class__._global_graph_pool = fresh_graph_pool
            for wrapper in wrappers:
                wrapper.graph_pool = fresh_graph_pool
            graph_pool_rebound = all(wrapper.graph_pool == fresh_graph_pool for wrapper in wrappers)
        except Exception as exc:  # pragma: no cover - exercised on real runtime
            reset_error = f"{type(exc).__name__}: {exc}"
        torch.accelerator.synchronize()

        entries_after = sum(len(wrapper.concrete_aclgraph_entries) for wrapper in wrappers)
        status = (
            "PASS"
            if (
                wrappers
                and entries_before > 0
                and graph_objects_reset == entries_before
                and entries_after == 0
                and graph_pool_rebound
                and reset_error is None
            )
            else "FAIL"
        )
        capture_enabled = status == "PASS"
        set_cudagraph_capturing_enabled(capture_enabled)
        return {
            "schema_version": "vllm-ascend-c8-graph-reset/v3",
            "status": status,
            "wrapper_count": len(wrappers),
            "graph_entries_before": entries_before,
            "graph_objects_reset": graph_objects_reset,
            "graph_entries_after": entries_after,
            "old_graph_pool_count": old_graph_pool_count,
            "graph_pool_rebound": graph_pool_rebound,
            "reset_error": reset_error,
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
