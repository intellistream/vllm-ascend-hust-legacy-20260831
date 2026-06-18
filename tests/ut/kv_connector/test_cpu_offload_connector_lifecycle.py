import json
import queue
import threading
from types import SimpleNamespace

import pytest

from vllm_ascend.distributed.kv_transfer.kv_pool.cpu_offload.cpu_offload_connector import (
    CPUOffloadingConnectorScheduler,
    CPUOffloadingConnectorWorker,
)


class _FakeRPCClient:
    def __init__(self):
        self.calls = []

    def call(self, name, *args):
        self.calls.append((name, args))


class _FakeEvent:
    def elapsed_time(self, other):
        return 1.25


def test_make_metadata_request_strips_unpickleable_request_hashers():
    request = SimpleNamespace(
        request_id="req-1",
        num_computed_tokens=128,
        _block_hasher=object(),
        get_hash_new_full_blocks=lambda: None,
    )

    metadata_request = CPUOffloadingConnectorScheduler._make_metadata_request(request)

    assert metadata_request is not request
    assert metadata_request.request_id == "req-1"
    assert metadata_request._block_hasher is None
    assert metadata_request.get_hash_new_full_blocks is None


def test_request_finished_skips_short_flush_requests():
    scheduler = CPUOffloadingConnectorScheduler.__new__(CPUOffloadingConnectorScheduler)
    scheduler.block_size = 128
    scheduler.finished_req_ids = []
    scheduler.zmq_rpc_client = _FakeRPCClient()
    request = SimpleNamespace(
        request_id="flush",
        num_computed_tokens=1,
        _block_hasher=None,
        get_hash_new_full_blocks=None,
    )

    assert scheduler.request_finished(request) is False

    assert scheduler.finished_req_ids == []
    assert scheduler.zmq_rpc_client.calls == []


def test_request_finished_records_full_block_requests():
    scheduler = CPUOffloadingConnectorScheduler.__new__(CPUOffloadingConnectorScheduler)
    scheduler.block_size = 128
    scheduler.finished_req_ids = []
    scheduler.zmq_rpc_client = _FakeRPCClient()
    request = SimpleNamespace(
        request_id="req-1",
        num_computed_tokens=128,
        _block_hasher=None,
        get_hash_new_full_blocks=None,
    )

    assert scheduler.request_finished(request) is True

    assert scheduler.finished_req_ids == ["req-1"]
    assert scheduler.zmq_rpc_client.calls[0][0] == "record_request_cache_and_free_slots"


def test_get_finished_commits_single_tp_saves():
    worker = CPUOffloadingConnectorWorker.__new__(CPUOffloadingConnectorWorker)
    worker.save_output_queue = queue.Queue()
    worker.save_output_queue.put("req-1")
    worker.requests = {"req-1": object()}
    worker.tp_world_size = 1
    committed = []
    worker._sending_finished = lambda done: committed.append(set(done))

    assert worker.get_finished() == {"req-1"}

    assert "req-1" not in worker.requests
    assert committed == [{"req-1"}]


def test_wait_for_save_surfaces_async_save_errors():
    worker = CPUOffloadingConnectorWorker.__new__(CPUOffloadingConnectorWorker)
    worker._save_done_condition = threading.Condition()
    worker._pending_save_req_ids = {"req-1"}
    worker._save_errors = {"req-1": "boom"}

    with pytest.raises(RuntimeError, match="CPU offload save failed"):
        worker.wait_for_save()


def test_wait_for_save_times_out_pending_saves(monkeypatch):
    worker = CPUOffloadingConnectorWorker.__new__(CPUOffloadingConnectorWorker)
    worker._save_done_condition = threading.Condition()
    worker._pending_save_req_ids = {"req-1"}
    worker._save_errors = {}
    monkeypatch.setenv("VLLM_ASCEND_CPU_OFFLOAD_SAVE_TIMEOUT_S", "0")

    with pytest.raises(TimeoutError, match="pending_req_ids"):
        worker.wait_for_save()


def test_collect_h2d_profile_events_writes_jsonl(tmp_path):
    worker = CPUOffloadingConnectorWorker.__new__(CPUOffloadingConnectorWorker)
    worker.h2d_profile_path = str(tmp_path / "h2d.jsonl")
    worker.current_layer = 3
    worker._h2d_profile_pending = [
        (_FakeEvent(), _FakeEvent(), 1024, 2, "tensor_copy"),
    ]

    worker._collect_h2d_profile_events()

    records = [
        json.loads(line)
        for line in (tmp_path / "h2d.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert records == [
        {
            "pid": records[0]["pid"],
            "layer": 3,
            "backend": "tensor_copy",
            "bytes": 1024,
            "segments": 2,
            "elapsed_ms": 1.25,
        }
    ]
    assert worker._h2d_profile_pending == []
