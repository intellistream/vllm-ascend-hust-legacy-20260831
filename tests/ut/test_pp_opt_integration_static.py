# SPDX-License-Identifier: Apache-2.0
# This file is a part of the vllm-ascend project.

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_pp_opt_worker_keeps_async_send_buffers_alive() -> None:
    worker = (REPO_ROOT / "vllm_ascend/worker/worker.py").read_text(
        encoding="utf-8"
    )

    assert "VLLM_PP_OPT_OVERLAP_SENDS" in worker
    assert "self._pp_send_buffer_refs" in worker
    assert "value.clone() if isinstance(value, torch.Tensor)" in worker
    assert "self._pp_send_buffer_refs.append((pp_send_work, send_tensors))" in worker
    assert "@pp_opt_profile.profile_worker_execute" in worker
    assert "@pp_opt_profile.profile_worker_sample_tokens" in worker


def test_pp_opt_model_runner_records_rank_local_forward_window() -> None:
    model_runner = (
        REPO_ROOT / "vllm_ascend/worker/model_runner_v1.py"
    ).read_text(encoding="utf-8")

    assert "pp_opt_profile.mark_t2()" in model_runner
    assert "pp_opt_profile.mark_t3()" in model_runner
    assert "if pp_opt_profile.record_active():" in model_runner
    assert "torch.npu.synchronize()" in model_runner


def test_pp_opt_group_coordinator_tracks_nonblocking_sends() -> None:
    patch = (
        REPO_ROOT / "vllm_ascend/patch/worker/patch_distributed.py"
    ).read_text(encoding="utf-8")

    assert "if envs.VLLM_USE_PP_OPT_SCHEDULER:" in patch
    assert "self.size_send_work" in patch
    assert "self.object_send_work" in patch
    assert "self.tensor_send_works" in patch
