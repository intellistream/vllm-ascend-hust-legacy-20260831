from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from vllm.config import CUDAGraphMode

from vllm_ascend.worker import model_runner_v1


class RecordingObserver:
    def __init__(self):
        self.records = []
        self.error_count = 0

    def record(self, **fields):
        self.records.append(fields)

    def note_error(self):
        self.error_count += 1


def make_runner(*, dp_size=2, dp_rank=0, on_npu=False):
    runner = model_runner_v1.NPUModelRunner.__new__(model_runner_v1.NPUModelRunner)
    runner.dp_size = dp_size
    runner.dp_rank = dp_rank
    runner.vllm_config = object()
    runner.ascend_config = SimpleNamespace(dp_allreduce_on_npu=on_npu)
    runner._adm_observer = RecordingObserver()
    runner._dp_metadata_tensor = None if dp_size == 1 else torch.empty((2, dp_size), dtype=torch.int32)
    runner._dp_num_tokens_after_padding = None if dp_size == 1 else torch.empty(dp_size, dtype=torch.int32)
    return runner


def test_dp1_path_records_local_observation_without_collective():
    runner = make_runner(dp_size=1)
    result = runner._sync_metadata_across_dp(7)
    assert result == (7, None, CUDAGraphMode.NONE)
    assert len(runner._adm_observer.records) == 1
    record = runner._adm_observer.records[0]
    assert record["path"] == "dp1"
    assert record["snapshot_scope"] == "local"


def test_skip_path_records_local_observation_without_collective():
    runner = make_runner()
    with patch.object(model_runner_v1, "should_skip_allreduce_across_dp_group", return_value=True):
        result = runner._sync_metadata_across_dp(11)
    assert result[0] == 11
    assert result[1].tolist() == [11, 11]
    assert len(runner._adm_observer.records) == 1
    record = runner._adm_observer.records[0]
    assert record["path"] == "skip"
    assert record["snapshot_scope"] == "local"


def test_cpu_collective_records_global_observation():
    runner = make_runner()
    group = SimpleNamespace(cpu_group=object(), device_group=object())
    with (
        patch.object(model_runner_v1, "should_skip_allreduce_across_dp_group", return_value=False),
        patch.object(model_runner_v1, "get_dp_group", return_value=group),
        patch.object(model_runner_v1.dist, "all_reduce") as all_reduce,
    ):
        result = runner._sync_metadata_across_dp(13)
    assert result[0] == 13
    all_reduce.assert_called_once()
    assert len(runner._adm_observer.records) == 1
    record = runner._adm_observer.records[0]
    assert record["path"] == "cpu"
    assert record["snapshot_scope"] == "global"
    assert record["collective_enter_ns"] is not None
    assert record["collective_ns"] >= 0
    assert record["copy_to_host_ns"] is None


def test_collective_failure_is_observed_and_original_error_propagates():
    runner = make_runner()
    group = SimpleNamespace(cpu_group=object(), device_group=object())
    with (
        patch.object(model_runner_v1, "should_skip_allreduce_across_dp_group", return_value=False),
        patch.object(model_runner_v1, "get_dp_group", return_value=group),
        patch.object(model_runner_v1.dist, "all_reduce", side_effect=RuntimeError("collective failed")),
        pytest.raises(RuntimeError, match="collective failed"),
    ):
        runner._sync_metadata_across_dp(17)
    assert runner._adm_observer.records == []
    assert runner._adm_observer.error_count == 1