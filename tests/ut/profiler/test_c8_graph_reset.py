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

from unittest.mock import MagicMock

from vllm_ascend.compilation.acl_graph import ACLGraphWrapper
from vllm_ascend.profiler.c8_graph_reset import C8GraphResetWorkerExtension


class _FakeWrapper:
    def __init__(self, entries: int) -> None:
        self.concrete_aclgraph_entries = {key: MagicMock(aclgraph=MagicMock()) for key in range(entries)}
        self.first_run_finished = True

    def clear_graphs(self) -> None:
        self.concrete_aclgraph_entries.clear()


def test_reset_c8_acl_graphs_for_profiling(monkeypatch) -> None:
    wrappers = [_FakeWrapper(2), _FakeWrapper(3)]
    synchronize = MagicMock()
    set_capture = MagicMock()
    clear_workspaces = MagicMock()
    reset_graph_params = MagicMock(side_effect=lambda: [wrapper.clear_graphs() for wrapper in wrappers])
    reset_model_runner = MagicMock()
    collect = MagicMock()
    empty_cache = MagicMock()
    extension = C8GraphResetWorkerExtension()
    extension.model_runner = MagicMock()
    monkeypatch.setattr(ACLGraphWrapper, "_all_instances", wrappers)
    monkeypatch.setattr("torch.accelerator.synchronize", synchronize)
    monkeypatch.setattr("torch.npu.empty_cache", empty_cache)
    monkeypatch.setattr("vllm_ascend.profiler.c8_graph_reset.gc.collect", collect)
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.clear_all_attention_workspaces",
        clear_workspaces,
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.reset_all_graph_params",
        reset_graph_params,
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.reset_model_runner_graph_manager",
        reset_model_runner,
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.set_cudagraph_capturing_enabled",
        set_capture,
    )

    graphs = [entry.aclgraph for wrapper in wrappers for entry in wrapper.concrete_aclgraph_entries.values()]
    report = extension.reset_c8_acl_graphs_for_profiling()

    assert report["status"] == "PASS"
    assert report["wrapper_count"] == 2
    assert report["graph_entries_before"] == 5
    assert report["graph_objects_reset"] == 5
    assert report["graph_entries_after"] == 0
    assert report["reset_error"] is None
    assert report["cudagraph_capturing_enabled_after"] is True
    assert synchronize.call_count == 2
    for graph in graphs:
        graph.reset.assert_called_once_with()
    clear_workspaces.assert_called_once_with()
    reset_graph_params.assert_called_once_with()
    reset_model_runner.assert_called_once_with(extension.model_runner)
    collect.assert_called_once_with()
    empty_cache.assert_called_once_with()
    assert [call.args[0] for call in set_capture.call_args_list] == [False, True]


def test_reset_fails_closed_without_captured_graph(monkeypatch) -> None:
    synchronize = MagicMock()
    set_capture = MagicMock()
    extension = C8GraphResetWorkerExtension()
    extension.model_runner = MagicMock()
    monkeypatch.setattr(ACLGraphWrapper, "_all_instances", [])
    monkeypatch.setattr("torch.accelerator.synchronize", synchronize)
    monkeypatch.setattr("torch.npu.empty_cache", MagicMock())
    monkeypatch.setattr("vllm_ascend.profiler.c8_graph_reset.gc.collect", MagicMock())
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.clear_all_attention_workspaces",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.reset_all_graph_params",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.reset_model_runner_graph_manager",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.set_cudagraph_capturing_enabled",
        set_capture,
    )

    report = extension.reset_c8_acl_graphs_for_profiling()

    assert report["status"] == "FAIL"
    assert report["wrapper_count"] == 0
    assert report["graph_entries_before"] == 0
    assert report["graph_objects_reset"] == 0
    assert report["graph_entries_after"] == 0
    assert report["cudagraph_capturing_enabled_after"] is False
    assert synchronize.call_count == 2
    assert [call.args[0] for call in set_capture.call_args_list] == [False, False]


def test_reset_fails_closed_when_graph_release_raises(monkeypatch) -> None:
    wrapper = _FakeWrapper(1)
    wrapper.concrete_aclgraph_entries[0].aclgraph.reset.side_effect = RuntimeError("release failed")
    set_capture = MagicMock()
    extension = C8GraphResetWorkerExtension()
    extension.model_runner = MagicMock()
    monkeypatch.setattr(ACLGraphWrapper, "_all_instances", [wrapper])
    monkeypatch.setattr("torch.accelerator.synchronize", MagicMock())
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.set_cudagraph_capturing_enabled",
        set_capture,
    )

    report = extension.reset_c8_acl_graphs_for_profiling()

    assert report["status"] == "FAIL"
    assert report["graph_entries_after"] == 1
    assert report["reset_error"] == "RuntimeError: release failed"
    assert report["cudagraph_capturing_enabled_after"] is False
    assert [call.args[0] for call in set_capture.call_args_list] == [False, False]


def test_reset_fails_closed_when_a_graph_object_is_missing(monkeypatch) -> None:
    wrapper = _FakeWrapper(2)
    wrapper.concrete_aclgraph_entries[1].aclgraph = None
    set_capture = MagicMock()
    extension = C8GraphResetWorkerExtension()
    extension.model_runner = MagicMock()
    monkeypatch.setattr(ACLGraphWrapper, "_all_instances", [wrapper])
    monkeypatch.setattr("torch.accelerator.synchronize", MagicMock())
    monkeypatch.setattr("torch.npu.empty_cache", MagicMock())
    monkeypatch.setattr("vllm_ascend.profiler.c8_graph_reset.gc.collect", MagicMock())
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.clear_all_attention_workspaces",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.reset_all_graph_params",
        MagicMock(side_effect=wrapper.clear_graphs),
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.AclGraphSleepWakeupManager.reset_model_runner_graph_manager",
        MagicMock(),
    )
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.set_cudagraph_capturing_enabled",
        set_capture,
    )

    report = extension.reset_c8_acl_graphs_for_profiling()

    assert report["status"] == "FAIL"
    assert report["graph_entries_before"] == 2
    assert report["graph_objects_reset"] == 1
    assert report["cudagraph_capturing_enabled_after"] is False
    assert [call.args[0] for call in set_capture.call_args_list] == [False, False]


def test_seal_c8_acl_graph_recapture_for_profiling(monkeypatch) -> None:
    wrappers = [_FakeWrapper(4), _FakeWrapper(1)]
    synchronize = MagicMock()
    set_capture = MagicMock()
    monkeypatch.setattr(ACLGraphWrapper, "_all_instances", wrappers)
    monkeypatch.setattr("torch.accelerator.synchronize", synchronize)
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.set_cudagraph_capturing_enabled",
        set_capture,
    )

    report = C8GraphResetWorkerExtension().seal_c8_acl_graph_recapture_for_profiling()

    assert report["status"] == "PASS"
    assert report["wrapper_count"] == 2
    assert report["graph_entries_after_recapture"] == 5
    assert report["cudagraph_capturing_enabled_after"] is False
    assert synchronize.call_count == 2
    set_capture.assert_called_once_with(False)


def test_seal_fails_closed_and_disables_capture_without_recaptured_graph(
    monkeypatch,
) -> None:
    synchronize = MagicMock()
    set_capture = MagicMock()
    monkeypatch.setattr(ACLGraphWrapper, "_all_instances", [_FakeWrapper(0)])
    monkeypatch.setattr("torch.accelerator.synchronize", synchronize)
    monkeypatch.setattr(
        "vllm_ascend.profiler.c8_graph_reset.set_cudagraph_capturing_enabled",
        set_capture,
    )

    report = C8GraphResetWorkerExtension().seal_c8_acl_graph_recapture_for_profiling()

    assert report["status"] == "FAIL"
    assert report["graph_entries_after_recapture"] == 0
    assert report["cudagraph_capturing_enabled_after"] is False
    set_capture.assert_called_once_with(False)
