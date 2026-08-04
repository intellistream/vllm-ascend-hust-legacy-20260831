import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from vllm_ascend.worker.adm_observer import (
    ADMRuntimeObserver,
    ObservationViolation,
)


def make_observer(tmp_path, *, max_samples=4):
    return ADMRuntimeObserver(
        trace_dir=tmp_path,
        rank=0,
        dp_size=2,
        max_samples=max_samples,
        host_id="host-192-168-0-4",
        pid=123,
    )


def global_observation(**overrides):
    values = {
        "event_index": 0,
        "path": "cpu",
        "snapshot_scope": "global",
        "num_tokens": [8, 4],
        "cudagraph_mode": [2, 2],
        "collective_enter_ns": 1000,
        "pack_ns": 10,
        "collective_ns": 20,
        "copy_to_host_ns": None,
        "total_ns": 40,
    }
    values.update(overrides)
    return values


def local_observation(path):
    return {
        "event_index": 0,
        "path": path,
        "snapshot_scope": "local",
        "num_tokens": [8],
        "cudagraph_mode": [2],
        "collective_enter_ns": None,
        "pack_ns": 0,
        "collective_ns": None,
        "copy_to_host_ns": None,
        "total_ns": 1,
    }


def rank_file(tmp_path):
    return tmp_path / "rank-0-pid-123.jsonl"


def read_records(tmp_path):
    return [
        json.loads(line)
        for line in rank_file(tmp_path).read_text(
            encoding="utf-8"
        ).splitlines()
    ]


def test_disabled_observer_is_inert(tmp_path):
    observer = ADMRuntimeObserver.disabled()

    assert observer.record(path="cpu") is None
    assert observer.flush("explicit") is False
    assert list(tmp_path.iterdir()) == []


def test_global_cpu_observation_and_receipt(tmp_path):
    observer = make_observer(tmp_path)

    record = observer.record(**global_observation())
    assert record["schema_version"] == "adm-runtime-observation/v1"
    assert record["record_type"] == "observation"
    assert record["rank"] == 0
    assert record["dp_size"] == 2

    assert observer.flush("explicit") is True
    records = read_records(tmp_path)
    assert records[0] == record
    assert records[-1]["record_type"] == "receipt"
    assert records[-1]["sample_count"] == 1
    assert records[-1]["dropped_samples"] == 0
    assert records[-1]["trace_error_count"] == 0
    assert records[-1]["flush_reason"] == "explicit"


def test_bounded_recorder_reports_dropped_samples(tmp_path):
    observer = make_observer(tmp_path, max_samples=1)

    assert observer.record(**global_observation()) is not None
    assert observer.record(
        **global_observation(event_index=1)
    ) is None
    observer.flush("explicit")

    receipt = read_records(tmp_path)[-1]
    assert receipt["sample_count"] == 1
    assert receipt["dropped_samples"] == 1


def test_flush_is_deterministic_and_idempotent(tmp_path):
    observer = make_observer(tmp_path)
    observer.record(**global_observation())

    assert observer.flush("explicit") is True
    first = rank_file(tmp_path).read_bytes()
    assert observer.flush("explicit") is True
    assert rank_file(tmp_path).read_bytes() == first


@pytest.mark.parametrize("path", ["dp1", "skip"])
def test_local_paths_accept_one_rank_value(tmp_path, path):
    observer = make_observer(tmp_path)

    record = observer.record(**local_observation(path))

    assert record["path"] == path
    assert record["snapshot_scope"] == "local"
    assert record["num_tokens"] == [8]


def test_npu_path_requires_existing_copy_to_host_interval(tmp_path):
    observer = make_observer(tmp_path)

    record = observer.record(
        **global_observation(
            path="npu",
            copy_to_host_ns=7,
        )
    )

    assert record["copy_to_host_ns"] == 7


@pytest.mark.parametrize(
    ("overrides", "code"),
    [
        ({"path": "magic"}, "invalid_path"),
        ({"snapshot_scope": "local"}, "invalid_scope"),
        ({"num_tokens": [8]}, "rank_shape_mismatch"),
        ({"rank": 2}, "unknown_field"),
        ({"pack_ns": -1}, "invalid_timing"),
        ({"copy_to_host_ns": 1}, "invalid_path_timing"),
    ],
)
def test_invalid_global_observation_fails_closed(
    tmp_path, overrides, code
):
    observer = make_observer(tmp_path)
    values = global_observation()
    values.update(overrides)

    with pytest.raises(ObservationViolation) as error:
        observer.record(**values)

    assert error.value.code == code


def test_event_index_must_increase(tmp_path):
    observer = make_observer(tmp_path)
    observer.record(**global_observation(event_index=2))

    with pytest.raises(ObservationViolation) as error:
        observer.record(**global_observation(event_index=2))

    assert error.value.code == "non_monotonic_event"


def test_trace_error_is_reported_in_receipt(tmp_path):
    observer = make_observer(tmp_path)
    observer.note_error()
    observer.flush("trace_error")

    receipt = read_records(tmp_path)[-1]
    assert receipt["trace_error_count"] == 1
