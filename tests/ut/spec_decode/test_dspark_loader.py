# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest
import torch

import vllm_ascend.models.deepseek_v4_dspark as dspark_model
from vllm_ascend.models.deepseek_v4_dspark import DeepseekV4DSparkModel, DSparkDeepseekV4ForCausalLM


@pytest.mark.parametrize("mix_placement, expected_experts", [(False, 4), (True, 6)])
def test_dspark_expert_mapping_adds_shared_slots(monkeypatch, mix_placement, expected_experts):
    captured = {}

    def fake_mapping(model, **kwargs):
        captured.update(model=model, **kwargs)
        return []

    monkeypatch.setattr(dspark_model, "fused_moe_make_expert_params_mapping", fake_mapping)
    monkeypatch.setattr(
        dspark_model,
        "get_ascend_config",
        lambda: SimpleNamespace(mix_placement=mix_placement),
        raising=False,
    )
    model = SimpleNamespace(config=SimpleNamespace(n_routed_experts=4, n_shared_experts=2))

    assert DeepseekV4DSparkModel.get_expert_mapping(model) == []
    assert captured["model"] is model
    assert captured["num_experts"] == expected_experts


@pytest.mark.parametrize(
    ("projection", "shape", "split_dim", "shard_id", "param_prefix"),
    [
        ("gate_proj", (4, 2), 0, "w1", "w13_"),
        ("down_proj", (2, 4), 1, "w2", "w2_"),
    ],
)
def test_dspark_loader_splits_fused_shared_expert_quant_weight(
    monkeypatch,
    projection,
    shape,
    split_dim,
    shard_id,
    param_prefix,
):
    routed_experts = 4
    shared_experts = 2
    source_name = f"model.layers.45.mlp.shared_experts.{projection}.weight_offset"
    mapped_name = f"model.layers.45.mlp.experts.{param_prefix}weight_offset"
    calls = []

    param = torch.nn.Parameter(torch.empty(1), requires_grad=False)

    def weight_loader(actual_param, weight, name, *, shard_id, expert_id, return_success):
        calls.append((actual_param, weight, name, shard_id, expert_id, return_success))
        return True

    param.weight_loader = weight_loader
    expert_mapping = [
        (
            f"model.layers.45.mlp.experts.{param_prefix}",
            f"model.layers.45.mlp.experts.{expert_id}.{projection}.",
            expert_id,
            shard_id,
        )
        for expert_id in range(routed_experts, routed_experts + shared_experts)
    ]
    draft = SimpleNamespace(
        model=SimpleNamespace(get_expert_mapping=lambda: expert_mapping),
        config=SimpleNamespace(
            expert_dtype="w8a8",
            num_attention_heads=1,
            n_routed_experts=routed_experts,
            n_shared_experts=shared_experts,
        ),
        has_own_embed_tokens=True,
        has_own_lm_head=True,
        named_parameters=lambda: [(mapped_name, param)],
        _remap_dspark_name=lambda _: source_name,
    )
    monkeypatch.setattr(
        dspark_model,
        "get_ascend_config",
        lambda: SimpleNamespace(mix_placement=True),
        raising=False,
    )
    monkeypatch.setattr(dspark_model, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(dspark_model, "get_tensor_model_parallel_rank", lambda: 0)
    weight = torch.arange(torch.tensor(shape).prod(), dtype=torch.int8).reshape(shape)

    result = DSparkDeepseekV4ForCausalLM.load_weights(
        draft,
        [(f"mtp.2.ffn.shared_experts.{projection}.weight_offset", weight)],
    )

    assert result == {mapped_name}
    assert [call[4] for call in calls] == [routed_experts, routed_experts + 1]
    assert [call[3] for call in calls] == [shard_id, shard_id]
    assert all(call[0] is param and call[2] == mapped_name and call[5] for call in calls)
    torch.testing.assert_close(
        torch.stack([call[1] for call in calls]),
        torch.stack(weight.chunk(shared_experts, dim=split_dim)),
    )
