# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.

import runpy
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILE_PATH = REPO_ROOT / "vllm_ascend/worker/pp_opt_profile.py"


def _install_fake_vllm_packages(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("vllm", "vllm.v1", "vllm.v1.worker"):
        package = types.ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)


def test_pp_opt_profile_falls_back_when_paired_core_module_is_absent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_vllm_packages(monkeypatch)
    profile_ns = runpy.run_path(str(PROFILE_PATH))

    @profile_ns["profile_worker_execute"]
    def sentinel() -> str:
        return "fallback"

    assert sentinel() == "fallback"
    assert profile_ns["record_active"]() is False
    assert profile_ns["mark_t2"]() is None
    assert profile_ns["mark_t3"]() is None
    assert profile_ns["set_microbatch_stats"](1, batch_size=2) is None


def test_pp_opt_profile_forwards_paired_core_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_vllm_packages(monkeypatch)
    paired_core = types.ModuleType("vllm.v1.worker.pp_opt_profile")
    hooks = {
        "mark_t2": lambda: "t2",
        "mark_t3": lambda: "t3",
        "profile_model_runner_execute": object(),
        "profile_worker_execute": object(),
        "profile_worker_sample_tokens": object(),
        "record_active": lambda: True,
        "set_microbatch_stats": object(),
    }
    for name, value in hooks.items():
        setattr(paired_core, name, value)
    monkeypatch.setitem(sys.modules, "vllm.v1.worker.pp_opt_profile", paired_core)

    profile_ns = runpy.run_path(str(PROFILE_PATH))

    for name, value in hooks.items():
        assert profile_ns[name] is value


def test_pp_opt_call_sites_import_compatibility_module() -> None:
    worker = (REPO_ROOT / "vllm_ascend/worker/worker.py").read_text(encoding="utf-8")
    model_runner = (REPO_ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8")

    assert "from vllm_ascend.worker import pp_opt_profile" in worker
    assert "from vllm_ascend.worker import pp_opt_profile" in model_runner


def test_pp_opt_worker_keeps_async_send_buffers_alive() -> None:
    worker = (REPO_ROOT / "vllm_ascend/worker/worker.py").read_text(encoding="utf-8")

    assert "VLLM_PP_OPT_OVERLAP_SENDS" in worker
    assert "self._pp_send_buffer_refs" in worker
    assert "value.clone() if isinstance(value, torch.Tensor)" in worker
    assert "self._pp_send_buffer_refs.append((pp_send_work, send_tensors))" in worker
    assert "@pp_opt_profile.profile_worker_execute" in worker
    assert "@pp_opt_profile.profile_worker_sample_tokens" in worker


def test_pp_opt_model_runner_records_rank_local_forward_window() -> None:
    model_runner = (REPO_ROOT / "vllm_ascend/worker/model_runner_v1.py").read_text(encoding="utf-8")

    assert "pp_opt_profile.mark_t2()" in model_runner
    assert "pp_opt_profile.mark_t3()" in model_runner
    assert "if pp_opt_profile.record_active():" in model_runner
    assert "torch.npu.synchronize()" in model_runner


def test_pp_opt_group_coordinator_tracks_nonblocking_sends() -> None:
    patch = (REPO_ROOT / "vllm_ascend/patch/worker/patch_distributed.py").read_text(encoding="utf-8")

    assert "if envs.VLLM_USE_PP_OPT_SCHEDULER:" in patch
    assert "self.size_send_work" in patch
    assert "self.object_send_work" in patch
    assert "self.tensor_send_works" in patch
