from pathlib import Path

import pytest

from vllm_ascend import envs
from vllm_ascend.worker.adm_observer import (
    ADMRuntimeObserver,
    ObservationViolation,
)


TRACE_DIR = "VLLM_ASCEND_ADM_TRACE_DIR"
TRACE_MAX = "VLLM_ASCEND_ADM_TRACE_MAX_SAMPLES"


def clear_trace_env(monkeypatch):
    monkeypatch.delenv(TRACE_DIR, raising=False)
    monkeypatch.delenv(TRACE_MAX, raising=False)


def test_trace_variables_are_registered():
    assert TRACE_DIR in envs.env_variables
    assert TRACE_MAX in envs.env_variables


def test_from_env_is_disabled_without_trace_dir(monkeypatch):
    clear_trace_env(monkeypatch)

    assert ADMRuntimeObserver.from_env(rank=0, dp_size=2) is None


def test_from_env_constructs_rank_local_observer(
    monkeypatch, tmp_path
):
    clear_trace_env(monkeypatch)
    monkeypatch.setenv(TRACE_DIR, str(tmp_path))
    monkeypatch.setenv(TRACE_MAX, "7")

    observer = ADMRuntimeObserver.from_env(rank=1, dp_size=2)

    assert observer is not None
    assert observer.rank == 1
    assert observer.dp_size == 2
    assert observer.max_samples == 7
    assert Path(observer._trace_dir) == tmp_path


def test_from_env_rejects_non_positive_bound(
    monkeypatch, tmp_path
):
    clear_trace_env(monkeypatch)
    monkeypatch.setenv(TRACE_DIR, str(tmp_path))
    monkeypatch.setenv(TRACE_MAX, "0")

