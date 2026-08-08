# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Two-card coverage for the native PEARL speculative FIA ACLGraph."""

from __future__ import annotations

import os
import subprocess
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist


def _run_worker() -> None:
    from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
    from vllm_ascend.spec_decode.pearl.native_model import (
        NativeQwen2ForCausalLM,
        NativeTPContext,
    )

    dist.init_process_group(backend="hccl")
    rank = dist.get_rank()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.npu.set_device(local_rank)
    device = torch.device("npu")
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        max_position_embeddings=256,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=2,
    )
    try:
        with torch.inference_mode():
            torch.manual_seed(7)
            model = NativeQwen2ForCausalLM(
                config,
                NativeTPContext(group=dist.group.WORLD, rank=rank, size=2, leader_rank=0),
            )
            model.to(device=device, dtype=torch.bfloat16).eval()
            for parameter in model.parameters():
                parameter.data.normal_(mean=0.0, std=0.02)
            model.configure_cache(256)
            runner = NativeACLGraphRunner(model, enabled=True)

            positions0, metadata0 = model.make_attention_metadata([0], [0], slot_mapping=[0])
            model(
                torch.tensor([1], dtype=torch.long, device=device),
                positions0,
                metadata0,
            )
            positions1, metadata1 = model.make_attention_metadata([0], [1], slot_mapping=[1])
            runner.run_greedy(
                torch.tensor([2], dtype=torch.long, device=device),
                positions1,
                metadata1,
                vocabulary_size=31,
            )
            positions2, metadata2 = model.make_attention_metadata([0], [2], slot_mapping=[2])
            runner.run_greedy(
                torch.tensor([3], dtype=torch.long, device=device),
                positions2,
                metadata2,
                vocabulary_size=31,
            )
            positions3, metadata3 = model.make_attention_metadata(
                [0, 0],
                [3, 4],
                slot_mapping=[3, 4],
                use_fused_infer_attention=True,
            )
            runner.run_greedy(
                torch.tensor([4, 5], dtype=torch.long, device=device),
                positions3,
                metadata3,
                vocabulary_size=31,
            )
            positions4, metadata4 = model.make_attention_metadata(
                [0, 0],
                [5, 6],
                slot_mapping=[5, 6],
                use_fused_infer_attention=True,
            )
            token = runner.run_greedy(
                torch.tensor([6, 7], dtype=torch.long, device=device),
                positions4,
                metadata4,
                vocabulary_size=31,
            )
            torch.npu.synchronize()

            gathered = [torch.empty_like(token) for _ in range(2)]
            dist.all_gather(gathered, token)
            assert all(torch.equal(candidate, gathered[0]) for candidate in gathered)
            assert runner.capture_count == 2
            assert runner.replay_count == 4
            assert runner.failed_capture_count == 0
            for entry in runner.entries.values():
                assert len(entry.tasks) == 2
                assert len({task.workspace.data_ptr() for task in entry.tasks}) == 1
            if rank == 0:
                print("PEARL speculative FIA ACLGraph HCCL smoke test passed.")
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(torch.npu.device_count() < 2, reason="PEARL HCCL ACLGraph test requires two NPUs.")
def test_pearl_greedy_aclgraph_hccl() -> None:
    env = os.environ.copy()
    env.setdefault("HCCL_NPU_SOCKET_PORT_RANGE", "auto")
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "torch.distributed.run",
            "--standalone",
            "--nproc_per_node=2",
            __file__,
            "--worker",
        ],
        check=False,
        capture_output=True,
        env=env,
        text=True,
        timeout=180,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert "PEARL speculative FIA ACLGraph HCCL smoke test passed." in result.stdout


if __name__ == "__main__":
    if "--worker" not in sys.argv:
        raise SystemExit("Pass --worker when launching this file under torchrun.")
    _run_worker()
