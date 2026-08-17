# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import random

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch_npu
from torch.distributed.distributed_c10d import _get_default_group

from vllm_ascend.utils import enable_custom_op

enable_custom_op()


def _get_hcomm_name(rank: int) -> str:
    backend = _get_default_group()._get_backend(torch.device("npu"))
    # TORCH_DISTRIBUTED_DEBUG may wrap the HCCL process group.
    while hasattr(backend, "wrapped_pg"):
        backend = backend.wrapped_pg
    return backend.get_hccl_comm_name(rank)


def _make_expert_idx(
    tokens: int,
    top_k: int,
    global_experts: int,
    source_rank: int,
    generation: int,
) -> torch.Tensor:
    multipliers = (1, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31)
    multiplier = multipliers[generation % len(multipliers)]
    flat_idx = torch.arange(tokens * top_k, dtype=torch.int32)
    return ((flat_idx * multiplier + source_rank * (generation + 3)) % global_experts).reshape(tokens, top_k)


def _make_mask(tokens: int, source_rank: int, case: str) -> torch.Tensor | None:
    if case == "unmasked":
        return None
    if case == "all_active":
        return torch.ones(tokens, dtype=torch.bool)
    if case == "partial":
        return (torch.arange(tokens) + source_rank) % 3 != 0
    if case == "one_active":
        mask = torch.zeros(tokens, dtype=torch.bool)
        mask[source_rank % tokens] = True
        return mask
    if case == "all_inactive":
        return torch.zeros(tokens, dtype=torch.bool)
    raise ValueError(f"unknown mask case: {case}")


def _expected_for_case(
    *,
    rank: int,
    world_size: int,
    local_experts: int,
    tokens: int,
    top_k: int,
    hidden_size: int,
    generation: int,
    mask_case: str,
    probs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    global_experts = world_size * local_experts
    local_routes = _make_expert_idx(tokens, top_k, global_experts, rank, generation)
    local_mask = _make_mask(tokens, rank, mask_case)
    active = torch.ones(tokens, dtype=torch.bool) if local_mask is None else local_mask

    expected = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
    expected[:, 0] = ((local_routes + 1) * probs).sum(dim=-1).to(torch.bfloat16) * F.silu(
        torch.tensor(1, dtype=torch.bfloat16)
    )

    active_routes = []
    for source_rank in range(world_size):
        source_routes = _make_expert_idx(tokens, top_k, global_experts, source_rank, generation)
        source_mask = _make_mask(tokens, source_rank, mask_case)
        if source_mask is not None:
            source_routes = source_routes[source_mask]
        active_routes.append(source_routes.reshape(-1))
    global_counts = torch.bincount(torch.cat(active_routes).to(torch.int64), minlength=global_experts).to(torch.int32)
    local_counts = global_counts[rank * local_experts : (rank + 1) * local_experts]
    return expected, local_counts, active


def _invoke(
    *,
    x: torch.Tensor,
    weight1: list[torch.Tensor],
    weight2: list[torch.Tensor],
    expert_idx: torch.Tensor,
    scale1: list[torch.Tensor],
    scale2: list[torch.Tensor],
    bias: list[torch.Tensor],
    probs: torch.Tensor,
    group: str,
    out: torch.Tensor,
    expert_token_nums: torch.Tensor,
    mask: torch.Tensor | None,
) -> None:
    kwargs = dict(
        x=x,
        weight1=weight1,
        weight2=weight2,
        expert_idx=expert_idx,
        scale1=scale1,
        scale2=scale2,
        bias1=bias,
        bias2=bias,
        probs=probs,
        group=group,
        max_output_size=512,
        out=out,
        expert_token_nums=expert_token_nums,
    )
    if mask is not None:
        kwargs["x_active_mask"] = mask
    torch.ops._C_ascend.dispatch_ffn_combine(**kwargs)


def _check_result(
    *,
    rank: int,
    world_size: int,
    local_experts: int,
    tokens: int,
    top_k: int,
    hidden_size: int,
    generation: int,
    mask_case: str,
    probs_cpu: torch.Tensor,
    out: torch.Tensor,
    expert_token_nums: torch.Tensor,
) -> None:
    expected, expected_counts, active = _expected_for_case(
        rank=rank,
        world_size=world_size,
        local_experts=local_experts,
        tokens=tokens,
        top_k=top_k,
        hidden_size=hidden_size,
        generation=generation,
        mask_case=mask_case,
        probs=probs_cpu,
    )
    actual = out.cpu()
    if active.any():
        torch.testing.assert_close(actual[active], expected[active], rtol=0.02, atol=0.02)
    torch.testing.assert_close(expert_token_nums.cpu(), expected_counts, rtol=0, atol=0)


def _run_rank(
    rank: int,
    world_size: int,
    port: int,
    local_experts: int,
    active_mask_supported: bool,
    run_graph: bool,
) -> None:
    torch_npu.npu.set_device(rank)
    dist.init_process_group(
        backend="hccl",
        rank=rank,
        world_size=world_size,
        init_method=f"tcp://127.0.0.1:{port}",
    )

    try:
        # EP * local_experts == 128 is the boundary at which adding the
        # inactive-route sentinel changes the count-row alignment.
        tokens = 64
        top_k = 4
        hidden_size = 256
        ffn_size = 256
        gate_up_size = 2 * ffn_size
        global_experts = world_size * local_experts

        torch_npu.npu.config.allow_internal_format = True
        x_cpu = torch.zeros((tokens, hidden_size), dtype=torch.bfloat16)
        x_cpu[:, 0] = 1

        # GMM1 emits silu(1) in feature zero. GMM2 multiplies it by the
        # one-based global expert id, giving an independent routing oracle.
        weight1_cpu = torch.zeros((local_experts, hidden_size, gate_up_size), dtype=torch.bfloat16)
        weight1_cpu[:, 0, 0] = 1
        weight1_cpu[:, 0, ffn_size] = 1
        weight2_cpu = torch.zeros((local_experts, ffn_size, hidden_size), dtype=torch.bfloat16)
        for local_expert in range(local_experts):
            weight2_cpu[local_expert, 0, 0] = rank * local_experts + local_expert + 1

        probs_cpu = torch.arange(1, top_k + 1, dtype=torch.float32).repeat(tokens, 1)
        probs_cpu /= probs_cpu.sum(dim=-1, keepdim=True)

        x = x_cpu.npu()
        probs = probs_cpu.npu()
        weight1 = [torch_npu.npu_format_cast(weight1_cpu.npu(), 29)]
        weight2 = [torch_npu.npu_format_cast(weight2_cpu.npu(), 29)]
        scale1 = [torch.empty(0, dtype=torch.int64, device="npu")]
        scale2 = [torch.empty(0, dtype=torch.int64, device="npu")]
        bias = [torch.empty(0, dtype=torch.float32, device="npu")]
        out = torch.empty_like(x)
        expert_token_nums = torch.empty(local_experts, dtype=torch.int32, device="npu")

        x_before = x.clone()
        probs_before = probs.clone()
        weight1_before = weight1[0].clone()
        weight2_before = weight2[0].clone()

        direct_cases = [(0, "unmasked"), (5, "unmasked")]
        if active_mask_supported:
            direct_cases = [
                (0, "unmasked"),
                (0, "all_active"),
                (1, "partial"),
                (2, "one_active"),
                (3, "all_inactive"),
                (4, "unmasked"),
                (5, "partial"),
            ]

        for generation, mask_case in direct_cases:
            expert_idx_cpu = _make_expert_idx(tokens, top_k, global_experts, rank, generation)
            mask_cpu = _make_mask(tokens, rank, mask_case)
            expert_idx = expert_idx_cpu.npu()
            mask = None if mask_cpu is None else mask_cpu.npu()
            expert_idx_before = expert_idx.clone()
            mask_before = None if mask is None else mask.clone()

            out.fill_(torch.nan)
            expert_token_nums.fill_(-1)
            dist.barrier()
            _invoke(
                x=x,
                weight1=weight1,
                weight2=weight2,
                expert_idx=expert_idx,
                scale1=scale1,
                scale2=scale2,
                bias=bias,
                probs=probs,
                group=_get_hcomm_name(rank),
                out=out,
                expert_token_nums=expert_token_nums,
                mask=mask,
            )
            torch_npu.npu.synchronize()
            _check_result(
                rank=rank,
                world_size=world_size,
                local_experts=local_experts,
                tokens=tokens,
                top_k=top_k,
                hidden_size=hidden_size,
                generation=generation,
                mask_case=mask_case,
                probs_cpu=probs_cpu,
                out=out,
                expert_token_nums=expert_token_nums,
            )
            torch.testing.assert_close(expert_idx, expert_idx_before, rtol=0, atol=0)
            if mask is not None:
                torch.testing.assert_close(mask, mask_before, rtol=0, atol=0)

        if active_mask_supported and run_graph:
            # Capture one fixed-shape masked invocation, then mutate only the
            # graph-owned input buffers across all-active, one-active, partial,
            # and zero-active generations.
            expert_idx = torch.empty((tokens, top_k), dtype=torch.int32, device="npu")
            mask = torch.empty(tokens, dtype=torch.bool, device="npu")
            capture_generation = 6
            capture_case = "partial"
            expert_idx.copy_(_make_expert_idx(tokens, top_k, global_experts, rank, capture_generation))
            mask.copy_(_make_mask(tokens, rank, capture_case))
            out.fill_(torch.nan)
            expert_token_nums.fill_(-1)

            graph = torch.npu.NPUGraph()
            dist.barrier()
            with torch.npu.graph(
                graph,
                capture_error_mode="thread_local",
                auto_dispatch_capture=True,
            ):
                _invoke(
                    x=x,
                    weight1=weight1,
                    weight2=weight2,
                    expert_idx=expert_idx,
                    scale1=scale1,
                    scale2=scale2,
                    bias=bias,
                    probs=probs,
                    group=_get_hcomm_name(rank),
                    out=out,
                    expert_token_nums=expert_token_nums,
                    mask=mask,
                )
            # Capture records the operator for later execution; it does not
            # populate the poisoned output/count buffers.  Replay once before
            # validating the capture-generation inputs.
            dist.barrier()
            graph.replay()
            torch_npu.npu.synchronize()
            _check_result(
                rank=rank,
                world_size=world_size,
                local_experts=local_experts,
                tokens=tokens,
                top_k=top_k,
                hidden_size=hidden_size,
                generation=capture_generation,
                mask_case=capture_case,
                probs_cpu=probs_cpu,
                out=out,
                expert_token_nums=expert_token_nums,
            )

            replay_cases = [
                (7, "all_active"),
                (8, "one_active"),
                (9, "all_inactive"),
                (10, "partial"),
            ]
            for generation, mask_case in replay_cases:
                expert_idx_cpu = _make_expert_idx(tokens, top_k, global_experts, rank, generation)
                mask_cpu = _make_mask(tokens, rank, mask_case)
                expert_idx.copy_(expert_idx_cpu)
                mask.copy_(mask_cpu)
                out.fill_(torch.nan)
                expert_token_nums.fill_(-1)
                dist.barrier()
                graph.replay()
                torch_npu.npu.synchronize()
                _check_result(
                    rank=rank,
                    world_size=world_size,
                    local_experts=local_experts,
                    tokens=tokens,
                    top_k=top_k,
                    hidden_size=hidden_size,
                    generation=generation,
                    mask_case=mask_case,
                    probs_cpu=probs_cpu,
                    out=out,
                    expert_token_nums=expert_token_nums,
                )
                torch.testing.assert_close(expert_idx.cpu(), expert_idx_cpu, rtol=0, atol=0)
                torch.testing.assert_close(mask.cpu(), mask_cpu, rtol=0, atol=0)

        torch.testing.assert_close(x, x_before, rtol=0, atol=0)
        torch.testing.assert_close(probs, probs_before, rtol=0, atol=0)
        torch.testing.assert_close(weight1[0], weight1_before, rtol=0, atol=0)
        torch.testing.assert_close(weight2[0], weight2_before, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


def run_dispatch_ffn_combine_bf16_two_ranks(*, active_mask_supported: bool = True) -> None:
    world_size = 2
    # E=8 retains the merged operator's ordinary-domain regression. E=63
    # exercises sentinel 127 in a 128-int row. E=64 forces the sentinel to
    # expand the aligned row from 128 to 256 ints; graph replay uses this
    # sharper boundary.
    for local_experts in (8, 63, 64):
        port = 29501 + random.randint(0, 10000)
        mp.spawn(
            _run_rank,
            args=(
                world_size,
                port,
                local_experts,
                active_mask_supported,
                active_mask_supported and local_experts == 64,
            ),
            nprocs=world_size,
            join=True,
        )
