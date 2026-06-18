#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Phase 4 correctness and stress regression guardrails."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from vllm_ascend.simllm.config import SimLLMConfig
from vllm_ascend.simllm.kv_reuse import KVReuseEngine
from vllm_ascend.simllm.sandwich import SandwichConfig
from vllm_ascend.simllm.similarity import MatchResult


def _scheduler_output(*prompt_lens: int) -> SimpleNamespace:
    reqs = []
    for idx, prompt_len in enumerate(prompt_lens):
        reqs.append(
            SimpleNamespace(
                req_id=f"req-{idx}",
                prompt_token_ids=list(range(prompt_len)),
                num_computed_tokens=0,
            )
        )
    return SimpleNamespace(scheduled_new_reqs=reqs)


def _runner_for_sandwich(tokens_per_req: tuple[int, ...]) -> MagicMock:
    runner = MagicMock()
    runner.input_batch.num_reqs = len(tokens_per_req)
    runner.input_batch.req_ids = [
        f"req-{idx}" for idx in range(len(tokens_per_req))
    ]
    runner.seq_lens = torch.tensor(tokens_per_req, dtype=torch.long)
    runner.query_start_loc = torch.tensor(
        [0] + list(tokens_per_req), dtype=torch.long
    ).cumsum(0)
    return runner


def _slot_mapping(num_layers: int, total_tokens: int) -> dict[str, torch.Tensor]:
    return {
        f"model.layers.{layer_idx}.self_attn": torch.arange(
            layer_idx * 100,
            layer_idx * 100 + total_tokens,
            dtype=torch.long,
        )
        for layer_idx in range(num_layers)
    }


def _matched_result(seq_len: int = 4) -> MatchResult:
    return MatchResult(
        matched=True,
        source_task_id="cached",
        cached_k=torch.zeros(1, 2, seq_len, 4),
        cached_v=torch.zeros(1, 2, seq_len, 4),
        similarity_score=1.0,
    )


class TestSequenceMismatch:
    """Cached KV alignment must be deterministic for shorter/longer prompts."""

    def test_prepare_injection_truncates_long_cached_kv(self):
        engine = KVReuseEngine(block_size=4, num_kv_heads=2, head_size=4)
        cached_k = torch.arange(1 * 2 * 6 * 4, dtype=torch.float32).reshape(
            1, 2, 6, 4,
        )
        cached_v = cached_k + 1000

        k_aligned, v_aligned = engine.prepare_injection(
            cached_k, cached_v, target_seq_len=4,
        )

        assert torch.equal(k_aligned, cached_k[:, :, :4, :])
        assert torch.equal(v_aligned, cached_v[:, :, :4, :])

    def test_prepare_injection_zero_pads_short_cached_kv(self):
        engine = KVReuseEngine(block_size=4, num_kv_heads=2, head_size=4)
        cached_k = torch.ones(1, 2, 3, 4)
        cached_v = torch.full((1, 2, 3, 4), 2.0)

        k_aligned, v_aligned = engine.prepare_injection(
            cached_k, cached_v, target_seq_len=5,
        )

        assert torch.equal(k_aligned[:, :, :3, :], cached_k)
        assert torch.equal(v_aligned[:, :, :3, :], cached_v)
        assert torch.equal(k_aligned[:, :, 3:, :], torch.zeros(1, 2, 2, 4))
        assert torch.equal(v_aligned[:, :, 3:, :], torch.zeros(1, 2, 2, 4))

    def test_scheduler_rewrite_and_injection_use_real_cached_coverage(
        self, monkeypatch,
    ):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        monkeypatch.setattr(
            patch_runner,
            "_kv_reuse_engine",
            KVReuseEngine(block_size=4, num_kv_heads=2, head_size=4),
        )

        scheduler_output = _scheduler_output(5, 4)
        short_k = torch.arange(1 * 2 * 3 * 4, dtype=torch.float32).reshape(
            1, 2, 3, 4,
        )
        short_v = short_k + 1000
        long_k = torch.arange(1 * 2 * 8 * 4, dtype=torch.float32).reshape(
            1, 2, 8, 4,
        )
        long_v = long_k + 2000

        runner = MagicMock()
        runner._simllm_match_results = {
            0: MatchResult(
                matched=True,
                source_task_id="cached-short",
                cached_k=short_k,
                cached_v=short_v,
                similarity_score=0.99,
            ),
            1: MatchResult(
                matched=True,
                source_task_id="cached-long",
                cached_k=long_k,
                cached_v=long_v,
                similarity_score=0.99,
            ),
        }

        patch_runner._simllm_rewrite_scheduler_output(runner, scheduler_output)
        patch_runner._simllm_build_injection_map_from_scheduler(
            runner, scheduler_output,
        )

        inj_map = patch_runner._simllm_injection_map
        assert inj_map is not None
        assert scheduler_output.scheduled_new_reqs[0].num_computed_tokens == 2
        assert scheduler_output.scheduled_new_reqs[1].num_computed_tokens == 3

        k_flat_0, v_flat_0, tok_start_0, covered_0 = inj_map[0]
        assert tok_start_0 == 0
        assert covered_0 == 3
        assert torch.equal(k_flat_0, short_k.squeeze(0).permute(1, 0, 2))
        assert torch.equal(v_flat_0, short_v.squeeze(0).permute(1, 0, 2))

        k_flat_1, v_flat_1, tok_start_1, covered_1 = inj_map[1]
        assert tok_start_1 == 5
        assert covered_1 == 4
        assert torch.equal(
            k_flat_1,
            long_k[:, :, :4, :].squeeze(0).permute(1, 0, 2),
        )
        assert torch.equal(
            v_flat_1,
            long_v[:, :, :4, :].squeeze(0).permute(1, 0, 2),
        )


class TestMixedBatchSandwich:
    """Matched/unmatched batches must stay in-place without re-queue."""

    def test_all_unique_batch_disables_only_middle_layers(
        self, monkeypatch,
    ):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        runner = _runner_for_sandwich((2, 3, 4))
        runner._simllm_match_results = {}
        slot_mapping = _slot_mapping(num_layers=4, total_tokens=9)
        original = {name: tensor.clone() for name, tensor in slot_mapping.items()}

        monkeypatch.setattr(
            patch_runner,
            "_sandwich_config",
            SandwichConfig(bottom_layers=1, top_layers=1, num_layers=4),
        )
        monkeypatch.setattr(
            patch_runner,
            "get_forward_context",
            lambda: SimpleNamespace(slot_mapping=slot_mapping),
        )

        patch_runner._simllm_apply_sandwich_slots(runner)

        assert torch.equal(slot_mapping["model.layers.0.self_attn"],
                           original["model.layers.0.self_attn"])
        assert torch.equal(slot_mapping["model.layers.3.self_attn"],
                           original["model.layers.3.self_attn"])
        assert (slot_mapping["model.layers.1.self_attn"] == -1).all()
        assert (slot_mapping["model.layers.2.self_attn"] == -1).all()

    def test_all_matched_batch_leaves_slots_unchanged(self, monkeypatch):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        runner = _runner_for_sandwich((2, 3, 4))
        runner._simllm_match_results = {
            idx: _matched_result() for idx in range(3)
        }
        slot_mapping = _slot_mapping(num_layers=4, total_tokens=9)
        original = {name: tensor.clone() for name, tensor in slot_mapping.items()}

        monkeypatch.setattr(
            patch_runner,
            "_sandwich_config",
            SandwichConfig(bottom_layers=1, top_layers=1, num_layers=4),
        )
        monkeypatch.setattr(
            patch_runner,
            "get_forward_context",
            lambda: SimpleNamespace(slot_mapping=slot_mapping),
        )

        patch_runner._simllm_apply_sandwich_slots(runner)

        for layer_name, tensor in slot_mapping.items():
            assert torch.equal(tensor, original[layer_name])

    def test_mixed_batch_disables_unmatched_rows_only(self, monkeypatch):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        runner = _runner_for_sandwich((2, 3, 4))
        runner._simllm_match_results = {1: _matched_result()}
        slot_mapping = _slot_mapping(num_layers=4, total_tokens=9)
        original = {name: tensor.clone() for name, tensor in slot_mapping.items()}

        monkeypatch.setattr(
            patch_runner,
            "_sandwich_config",
            SandwichConfig(bottom_layers=1, top_layers=1, num_layers=4),
        )
        monkeypatch.setattr(
            patch_runner,
            "get_forward_context",
            lambda: SimpleNamespace(slot_mapping=slot_mapping),
        )

        patch_runner._simllm_apply_sandwich_slots(runner)

        for keep_layer in (0, 3):
            layer_name = f"model.layers.{keep_layer}.self_attn"
            assert torch.equal(slot_mapping[layer_name], original[layer_name])

        for middle_layer in (1, 2):
            layer_name = f"model.layers.{middle_layer}.self_attn"
            tensor = slot_mapping[layer_name]
            assert (tensor[0:2] == -1).all()
            assert torch.equal(tensor[2:5], original[layer_name][2:5])
            assert (tensor[5:9] == -1).all()


class TestRequestAccounting:
    """Store-plan mapping must not lose or duplicate scheduled requests."""

    def test_store_plan_preserves_prefill_hash_order_with_reordered_batch(self):
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_build_store_plan,
        )

        plan = _simllm_build_store_plan(
            input_batch_req_ids=["decode-a", "prefill-b", "prefill-a"],
            prefill_req_ids=["prefill-a", "prefill-b"],
            num_hashes=2,
        )

        assert plan == [(2, 0), (1, 1)]

    def test_store_plan_does_not_duplicate_duplicate_batch_rows(self):
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_build_store_plan,
        )

        plan = _simllm_build_store_plan(
            input_batch_req_ids=["req-a", "req-b", "req-c"],
            prefill_req_ids=["req-a", "req-a", "req-c"],
            num_hashes=3,
        )

        row_indices = [row_idx for row_idx, _hash_idx in plan]
        assert len(row_indices) == len(set(row_indices))


class TestDisabledMode:
    """Disabled Sim-LLM must delegate without touching runtime state."""

    def test_execute_model_disabled_delegates_to_original(self, monkeypatch):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        sentinel = object()
        original = MagicMock(return_value=sentinel)
        runner = SimpleNamespace()
        scheduler_output = MagicMock()

        monkeypatch.setattr(
            patch_runner,
            "_simllm_config",
            SimLLMConfig(enabled=False),
        )
        monkeypatch.setattr(patch_runner, "_original_execute_model", original)

        result = patch_runner._simllm_execute_model(
            runner,
            scheduler_output,
            intermediate_tensors="intermediate",
            extra_flag=True,
        )

        assert result is sentinel
        original.assert_called_once_with(
            runner,
            scheduler_output,
            "intermediate",
            extra_flag=True,
        )
        assert not hasattr(runner, "_simllm_match_results")

    def test_model_forward_disabled_delegates_to_original(self, monkeypatch):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        sentinel = object()
        original = MagicMock(return_value=sentinel)
        runner = SimpleNamespace()

        monkeypatch.setattr(
            patch_runner,
            "_simllm_config",
            SimLLMConfig(enabled=False),
        )
        monkeypatch.setattr(patch_runner, "_original_model_forward", original)

        result = patch_runner._simllm_model_forward(
            runner,
            16,
            input_ids="ids",
            positions="positions",
            intermediate_tensors="intermediate",
            inputs_embeds="embeds",
            model_kw=True,
        )

        assert result is sentinel
        original.assert_called_once_with(
            runner,
            16,
            "ids",
            "positions",
            "intermediate",
            "embeds",
            model_kw=True,
        )
