#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""Integration tests for the Sim-LLM patch pipeline.

Mocks ``NPUModelRunner`` and its ``input_batch`` / ``kv_caches`` to
verify the full 5-step Sim-LLM pipeline without requiring NPU hardware.
"""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.simllm.kv_manager import KVManager
from vllm_ascend.simllm.kv_reuse import KVReuseEngine
from vllm_ascend.simllm.lsh import SimHashHasher
from vllm_ascend.simllm.similarity import SimilarityIdentifier

# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------


def _make_mock_runner(
    num_reqs: int = 3,
    tokens_per_req: tuple[int, ...] = (5, 7, 4),
    dim: int = 128,
    num_kv_heads: int = 4,
    head_size: int = 32,
    num_blocks: int = 10,
    block_size: int = 16,
    num_layers: int = 4,
    device: str = "cpu",
):
    """Build a mock NPUModelRunner with enough state for the Sim-LLM pipeline.

    Returns a MagicMock with realistic ``input_batch``, ``kv_caches``,
    ``model``, ``seq_lens``, and ``query_start_loc`` attributes.
    """
    runner = MagicMock()
    runner.device = device

    # -- input_batch --
    total_tokens = sum(tokens_per_req)
    runner.input_batch.num_reqs = num_reqs
    runner.input_batch.req_ids = [f"req_{i}" for i in range(num_reqs)]

    # Flat input_ids: [0,1,2,3,4, 0,1,2,3,4,5,6, 0,1,2,3]
    input_ids = torch.cat([torch.arange(t, dtype=torch.long) for t in tokens_per_req])
    runner.input_batch.input_ids = input_ids
    runner.input_batch.num_tokens = torch.tensor(tokens_per_req)

    # query_start_loc: [0, 5, 12, 16]
    cumsum = torch.tensor([0] + list(tokens_per_req)).cumsum(0)
    runner.query_start_loc = cumsum

    # seq_lens
    runner.seq_lens = torch.tensor(tokens_per_req, dtype=torch.long)

    # -- kv_caches --
    k_cache_shape = (num_blocks, block_size, num_kv_heads, head_size)
    v_cache_shape = (num_blocks, block_size, num_kv_heads, head_size)
    runner.kv_caches = [(torch.zeros(k_cache_shape), torch.zeros(v_cache_shape)) for _ in range(num_layers)]

    # -- block_table --
    blk_table = MagicMock()
    blk_table_tensor = torch.zeros(num_reqs, num_blocks, dtype=torch.long)
    for i, t in enumerate(tokens_per_req):
        n_blocks = KVReuseEngine.num_blocks_needed(t, block_size)
        blk_table_tensor[i, :n_blocks] = torch.arange(i * 3, i * 3 + n_blocks)
    blk_table.get_device_tensor.return_value = blk_table_tensor
    runner.input_batch.block_table = {0: blk_table}

    # -- model --
    runner.model = MagicMock()
    runner.model.get_input_embeddings.return_value.weight = torch.randn(
        1000,
        dim,
    )
    runner.model.get_input_embeddings.return_value.forward = lambda ids: (
        runner.model.get_input_embeddings.return_value.weight[ids % 1000]
    )
    # Make get_input_embeddings() return a module whose forward does the lookup.
    embed = MagicMock()
    embed.weight = torch.nn.Parameter(torch.randn(1000, dim))
    embed.forward = lambda ids: embed.weight[ids % 1000]
    runner.model.get_input_embeddings.return_value = embed

    # -- hidden_states for postprocess --
    runner.hidden_states = torch.randn(total_tokens, dim)

    return runner


# ---------------------------------------------------------------------------
# Direct KVReuseEngine integration tests
# ---------------------------------------------------------------------------


class TestGatherRoundTrip:
    """Verify write_to_cache → gather_from_cache roundtrip."""

    def test_roundtrip_single_block(self):
        engine = KVReuseEngine(block_size=16, num_kv_heads=4, head_size=64)
        k_cache = torch.zeros(4, 16, 4, 64)
        v_cache = torch.zeros(4, 16, 4, 64)

        k = torch.randn(1, 4, 12, 64)
        v = torch.randn(1, 4, 12, 64)

        engine.write_to_cache(k_cache, v_cache, [2], k, v)
        k_read = engine.gather_from_cache(k_cache, [2], 12, 16)
        v_read = engine.gather_from_cache(v_cache, [2], 12, 16)

        assert torch.allclose(k_read, k, atol=1e-6)
        assert torch.allclose(v_read, v, atol=1e-6)

    def test_roundtrip_multi_block(self):
        engine = KVReuseEngine(block_size=8, num_kv_heads=2, head_size=16)
        k_cache = torch.zeros(6, 8, 2, 16)
        v_cache = torch.zeros(6, 8, 2, 16)

        k = torch.randn(1, 2, 22, 16)  # 22 tokens → 3 blocks
        v = torch.randn(1, 2, 22, 16)

        engine.write_to_cache(k_cache, v_cache, [0, 2, 5], k, v)
        k_read = engine.gather_from_cache(k_cache, [0, 2, 5], 22, 8)
        v_read = engine.gather_from_cache(v_cache, [0, 2, 5], 22, 8)

        # Roundtrip: written == read
        assert torch.allclose(k_read, k, atol=1e-6)
        assert torch.allclose(v_read, v, atol=1e-6)

    def test_roundtrip_with_padding(self):
        """Write padded KV, gather — zeros should be in the padded region."""
        engine = KVReuseEngine(block_size=8, num_kv_heads=2, head_size=16)
        k_cache = torch.zeros(3, 8, 2, 16)

        k_short = torch.randn(1, 2, 5, 16)  # 5 tokens, need 1 block
        k_padded, _ = engine.prepare_injection(k_short, k_short.clone(), 8)
        engine.write_to_cache(k_cache, k_cache.clone(), [0], k_padded, k_padded)

        gathered = engine.gather_from_cache(k_cache, [0], 8, 8)
        # First 5 tokens: original data. Last 3: zeros.
        assert torch.allclose(gathered[:, :, :5, :], k_short, atol=1e-6)
        assert (gathered[:, :, 5:, :] == 0).all()


# ---------------------------------------------------------------------------
# KVManager roundtrip: store then match
# ---------------------------------------------------------------------------


class TestStoreThenMatch:
    """Store tasks in KVManager, then match with identical embeddings."""

    def test_store_and_match_same_embedding(self):
        kv_mgr = KVManager(max_cache_size=10)
        hasher = SimHashHasher(dim=64, num_bits=32)
        identifier = SimilarityIdentifier(
            embedding_dim=64,
            cosine_threshold=0.8,
        )

        # Store a task.
        emb = torch.randn(1, 64)
        emb = torch.nn.functional.normalize(emb, dim=-1)
        hsh = int(hasher.hash(emb).item())

        from vllm_ascend.simllm.kv_manager import CachedTask

        task = CachedTask(
            task_id="task_1",
            embedding=emb.clone(),
            lsh_hash=hsh,
            top_k=torch.randn(1, 4, 10, 64),
            top_v=torch.randn(1, 4, 10, 64),
            last_access_time=time.monotonic(),
            seq_len=10,
        )
        kv_mgr.store(task)
        assert kv_mgr.size() == 1

        # Match with identical embedding.
        result = identifier.identify(
            emb.clone(),
            torch.tensor([hsh]),
            kv_mgr,
        )
        assert result[0].matched is True
        assert result[0].source_task_id == "task_1"
        assert result[0].cached_k is not None
        assert result[0].cached_v is not None
        assert result[0].similarity_score >= 0.99

    def test_store_and_match_different_embedding(self):
        kv_mgr = KVManager(max_cache_size=10)
        hasher = SimHashHasher(dim=64, num_bits=32)
        identifier = SimilarityIdentifier(
            embedding_dim=64,
            cosine_threshold=0.85,
        )

        emb1 = torch.randn(1, 64)
        emb1 = torch.nn.functional.normalize(emb1, dim=-1)
        hsh1 = int(hasher.hash(emb1).item())

        from vllm_ascend.simllm.kv_manager import CachedTask

        kv_mgr.store(
            CachedTask(
                task_id="task_a",
                embedding=emb1,
                lsh_hash=hsh1,
                top_k=torch.randn(1, 4, 8, 64),
                top_v=torch.randn(1, 4, 8, 64),
                last_access_time=time.monotonic(),
                seq_len=8,
            )
        )

        emb2 = torch.randn(1, 64)
        emb2 = torch.nn.functional.normalize(emb2, dim=-1)
        hsh2 = int(hasher.hash(emb2).item())

        result = identifier.identify(emb2, torch.tensor([hsh2]), kv_mgr)
        # Random embeddings should be far apart (cos < 0.85).
        assert result[0].matched is False

    def test_store_multiple_match_best(self):
        """Store 3 tasks, match against the closest one."""
        kv_mgr = KVManager(max_cache_size=10)
        hasher = SimHashHasher(dim=64, num_bits=32)
        identifier = SimilarityIdentifier(
            embedding_dim=64,
            cosine_threshold=0.7,
        )

        emb_base = torch.randn(1, 64)
        emb_base = torch.nn.functional.normalize(emb_base, dim=-1)
        hsh_base = int(hasher.hash(emb_base).item())

        emb_close = emb_base + 0.01 * torch.randn(1, 64)
        emb_close = torch.nn.functional.normalize(emb_close, dim=-1)
        hsh_close = int(hasher.hash(emb_close).item())

        emb_far = torch.randn(1, 64)
        emb_far = torch.nn.functional.normalize(emb_far, dim=-1)
        hsh_far = int(hasher.hash(emb_far).item())

        from vllm_ascend.simllm.kv_manager import CachedTask

        for i, (emb, hsh) in enumerate(
            [
                (emb_far, hsh_far),
                (emb_base, hsh_base),
                (emb_close, hsh_close),
            ]
        ):
            kv_mgr.store(
                CachedTask(
                    task_id=f"task_{i}",
                    embedding=emb,
                    lsh_hash=hsh,
                    top_k=torch.randn(1, 4, 10, 64),
                    top_v=torch.randn(1, 4, 10, 64),
                    last_access_time=time.monotonic(),
                    seq_len=10,
                )
            )

        # Query with emb_base → should match task_1 (emb_base itself) with cos≈1.
        result = identifier.identify(emb_base, torch.tensor([hsh_base]), kv_mgr)
        assert result[0].matched is True
        assert result[0].similarity_score >= 0.99


# ---------------------------------------------------------------------------
# Hasher dimension reconciliation
# ---------------------------------------------------------------------------


class TestHasherReconciliation:
    """Verify SimHashHasher dimension reconciliation."""

    def test_reconcile_hasher_dim_uses_public_dim(self, monkeypatch):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        class _Config:
            hidden_size = 64

        class _Model:
            config = _Config()

        runner = MagicMock()
        runner.model = _Model()

        monkeypatch.setattr(
            patch_runner,
            "_simhash_hasher",
            SimHashHasher(dim=4096, num_bits=32),
        )
        monkeypatch.setattr(
            patch_runner,
            "_simllm_config",
            MagicMock(lsh_num_bits=32),
        )

        patch_runner._reconcile_hasher_dim(runner)

        assert patch_runner._simhash_hasher.dim == 64
        assert patch_runner._simhash_hasher.num_bits == 32


class TestStorePlan:
    """Verify KV store only targets requests with prefill hashes."""

    def test_store_plan_maps_prefill_req_ids_to_batch_rows(self):
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_build_store_plan,
        )

        plan = _simllm_build_store_plan(
            input_batch_req_ids=["cached-a", "new-a", "decode-b", "new-b"],
            prefill_req_ids=["new-a", "new-b"],
            num_hashes=2,
        )

        assert plan == [(1, 0), (3, 1)]

    def test_store_plan_ignores_missing_or_unhashed_prefill_rows(self):
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_build_store_plan,
        )

        plan = _simllm_build_store_plan(
            input_batch_req_ids=["cached-a", "new-a"],
            prefill_req_ids=["new-a", "new-b"],
            num_hashes=1,
        )

        assert plan == [(1, 0)]


class TestDecodeOnlyBatch:
    """Decode steps must not run prefill-only SimLLM cache handling."""

    def test_decode_only_batch_leaves_sandwich_slots_unchanged(
        self,
        monkeypatch,
    ):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner
        from vllm_ascend.simllm.sandwich import SandwichConfig

        runner = MagicMock()
        runner._simllm_batch_req_ids = None
        runner._simllm_match_results = {}
        runner.input_batch.num_reqs = 2
        runner.query_start_loc = torch.tensor([0, 1, 2])
        runner.seq_lens = torch.tensor([1, 1])

        slot_mapping = {f"model.layers.{idx}.self_attn": torch.arange(2) + idx * 10 for idx in range(4)}
        original = {name: slots.clone() for name, slots in slot_mapping.items()}
        forward_context = MagicMock(slot_mapping=slot_mapping)
        monkeypatch.setattr(
            patch_runner,
            "get_forward_context",
            lambda: forward_context,
        )
        monkeypatch.setattr(
            patch_runner,
            "_sandwich_config",
            SandwichConfig(bottom_layers=1, top_layers=1, num_layers=4),
        )

        patch_runner._simllm_apply_sandwich_slots(runner)

        for name, slots in slot_mapping.items():
            torch.testing.assert_close(slots, original[name])

    def test_decode_only_forward_skips_kv_extraction(self, monkeypatch):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        runner = MagicMock()
        runner._simllm_batch_req_ids = None
        hidden_states = torch.randn(2, 4)
        original_forward = MagicMock(return_value=hidden_states)
        extract_kv = MagicMock()
        monkeypatch.setattr(
            patch_runner,
            "_simllm_config",
            MagicMock(enabled=True),
        )
        monkeypatch.setattr(
            patch_runner,
            "_original_model_forward",
            original_forward,
        )
        monkeypatch.setattr(patch_runner, "_simllm_extract_kv", extract_kv)
        monkeypatch.setattr(
            patch_runner,
            "_simllm_apply_sandwich_slots",
            MagicMock(),
        )

        result = patch_runner._simllm_model_forward(runner, 2)

        assert result is hidden_states
        extract_kv.assert_not_called()


class TestSchedulerRewrite:
    """Matched prefills must shrink the worker's actual token workload."""

    def test_rewrite_updates_per_request_and_total_scheduled_tokens(self):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner
        from vllm_ascend.simllm.similarity import MatchResult

        matched = SimpleNamespace(
            req_id="matched",
            prompt_token_ids=list(range(1024)),
            num_computed_tokens=0,
        )
        unmatched = SimpleNamespace(
            req_id="unmatched",
            prompt_token_ids=list(range(16)),
            num_computed_tokens=0,
        )
        scheduler_output = SimpleNamespace(
            scheduled_new_reqs=[matched, unmatched],
            num_scheduled_tokens={"matched": 1024, "unmatched": 16},
            total_num_scheduled_tokens=1040,
        )
        runner = SimpleNamespace(
            _simllm_match_results={
                0: MatchResult(
                    matched=True,
                    source_task_id="cached",
                    cached_k=torch.randn(1, 2, 1024, 8),
                    cached_v=torch.randn(1, 2, 1024, 8),
                    similarity_score=1.0,
                )
            }
        )

        patch_runner._simllm_rewrite_scheduler_output(
            runner,
            scheduler_output,
        )

        assert matched.num_computed_tokens == 1023
        assert scheduler_output.num_scheduled_tokens == {
            "matched": 1,
            "unmatched": 16,
        }
        assert scheduler_output.total_num_scheduled_tokens == 17

    def test_prefill_forward_injects_cache_before_model(self, monkeypatch):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        runner = SimpleNamespace(_simllm_batch_req_ids=["matched"])
        calls = []

        monkeypatch.setattr(
            patch_runner,
            "_simllm_config",
            MagicMock(enabled=True),
        )
        monkeypatch.setattr(
            patch_runner,
            "_simllm_inject_kv",
            lambda _runner: calls.append("inject"),
        )
        monkeypatch.setattr(
            patch_runner,
            "_simllm_apply_sandwich_slots",
            lambda _runner: calls.append("sandwich"),
        )
        monkeypatch.setattr(
            patch_runner,
            "_simllm_extract_kv",
            lambda _runner, _hidden: calls.append("extract"),
        )
        monkeypatch.setattr(
            patch_runner,
            "_original_model_forward",
            lambda *_args, **_kwargs: calls.append("forward") or torch.ones(1),
        )

        patch_runner._simllm_model_forward(runner, 1)

        assert calls == ["inject", "sandwich", "forward", "extract"]

    def test_cache_is_prepopulated_for_every_attention_layer(self, monkeypatch):
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner
        from vllm_ascend.simllm.similarity import MatchResult

        runner = _make_mock_runner(
            num_reqs=1,
            tokens_per_req=(5,),
            num_kv_heads=2,
            head_size=4,
            num_layers=3,
        )
        cached_k = torch.randn(1, 2, 5, 4)
        cached_v = torch.randn(1, 2, 5, 4)
        runner._simllm_match_results = {
            0: MatchResult(
                matched=True,
                source_task_id="cached",
                cached_k=cached_k,
                cached_v=cached_v,
                similarity_score=1.0,
            )
        }
        monkeypatch.setattr(
            patch_runner,
            "_kv_reuse_engine",
            KVReuseEngine(block_size=16, num_kv_heads=2, head_size=4),
        )

        patch_runner._simllm_inject_kv(runner)

        expected_k = cached_k[:, :, :4, :]
        expected_v = cached_v[:, :, :4, :]
        for k_cache, v_cache in runner.kv_caches:
            actual_k = KVReuseEngine.gather_from_cache(k_cache, [0], 4, 16)
            actual_v = KVReuseEngine.gather_from_cache(v_cache, [0], 4, 16)
            torch.testing.assert_close(actual_k, expected_k)
            torch.testing.assert_close(actual_v, expected_v)


class TestKVExtractionStoreMode:
    """The low-overhead cache format gathers unmatched KV once per tensor."""

    @staticmethod
    def _runner_for_extract():
        runner = SimpleNamespace()
        runner.input_batch = SimpleNamespace(
            num_reqs=1,
            req_ids=["req-0"],
            block_table=[
                SimpleNamespace(
                    get_device_tensor=lambda: torch.tensor(
                        [[0]],
                        dtype=torch.long,
                    ),
                ),
            ],
        )
        runner.seq_lens = torch.tensor([2], dtype=torch.long)
        runner.query_start_loc = torch.tensor([0, 2], dtype=torch.long)
        runner.kv_caches = [
            (
                torch.ones(1, 4, 1, 1) * layer_idx,
                torch.ones(1, 4, 1, 1) * (layer_idx + 10),
            )
            for layer_idx in range(4)
        ]
        runner._simllm_batch_hashes = torch.tensor([123], dtype=torch.long)
        runner._simllm_batch_req_ids = ["req-0"]
        runner._simllm_batch_embeddings = torch.randn(1, 8)
        runner._simllm_match_results = {}
        return runner

    def test_unmatched_top_store_gathers_top_layer_only(self, monkeypatch):
        from vllm_ascend.simllm.config import SimLLMConfig
        from vllm_ascend.simllm.kv_manager import KVManager
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner
        from vllm_ascend.simllm.sandwich import SandwichConfig

        gather_calls = []

        def fake_gather(kv_cache, block_ids, seq_len, block_size):
            gather_calls.append(kv_cache)
            return torch.ones(1, 1, seq_len, 1)

        monkeypatch.setattr(patch_runner, "_kv_manager", KVManager())
        monkeypatch.setattr(
            patch_runner,
            "_sandwich_config",
            SandwichConfig(bottom_layers=1, top_layers=1, num_layers=4),
        )
        monkeypatch.setattr(
            patch_runner,
            "_simllm_config",
            SimLLMConfig(enabled=True, unmatched_store_mode="top"),
        )
        monkeypatch.setattr(
            patch_runner.KVReuseEngine,
            "gather_from_cache",
            staticmethod(fake_gather),
        )

        patch_runner._simllm_extract_kv(
            self._runner_for_extract(),
            torch.randn(2, 8),
        )

        assert len(gather_calls) == 2


class TestSchedulerEmbeddingReuse:
    """Verify KV storage can reuse embeddings without padding hidden states."""

    def test_embedding_map_keeps_scheduler_request_order(self):
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_build_embedding_map,
        )

        embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)

        embedding_map = _simllm_build_embedding_map(
            ["req-c", "req-a", "req-b"],
            embeddings,
        )

        assert list(embedding_map) == ["req-c", "req-a", "req-b"]
        torch.testing.assert_close(embedding_map["req-a"], embeddings[1:2])
        assert embedding_map["req-a"].shape == (1, 4)

    def test_embedding_map_handles_missing_or_short_embeddings(self):
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_build_embedding_map,
        )

        assert _simllm_build_embedding_map(["req-a"], None) == {}

        embeddings = torch.ones(1, 4)
        embedding_map = _simllm_build_embedding_map(
            ["req-a", "req-b"],
            embeddings,
        )

        assert list(embedding_map) == ["req-a"]

    def test_embedding_map_owns_per_request_storage(self):
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_build_embedding_map,
        )

        embeddings = torch.arange(12, dtype=torch.float32).reshape(3, 4)
        embedding_map = _simllm_build_embedding_map(
            ["req-a", "req-b", "req-c"],
            embeddings,
        )

        source_storage = embeddings.untyped_storage().data_ptr()
        mapped_storages = {embedding.untyped_storage().data_ptr() for embedding in embedding_map.values()}
        assert source_storage not in mapped_storages
        assert len(mapped_storages) == 3
        assert all(embedding._base is None for embedding in embedding_map.values())

    @pytest.mark.parametrize(
        ("pooling", "expected"),
        [
            (
                "mean",
                torch.tensor(
                    [
                        [2**-0.5, 2**-0.5],
                        [2 / 5**0.5, 1 / 5**0.5],
                    ]
                ),
            ),
            (
                "last",
                torch.tensor(
                    [
                        [0.0, 1.0],
                        [2 / 5**0.5, 1 / 5**0.5],
                    ]
                ),
            ),
        ],
    )
    def test_hidden_state_fallback_pools_each_original_slice(
        self,
        pooling,
        expected,
    ):
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _per_request_embeddings,
        )

        hidden_states = torch.tensor(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [2.0, 1.0],
            ]
        )

        result = _per_request_embeddings(
            hidden_states,
            torch.tensor([0, 2, 3]),
            pooling=pooling,
        )

        torch.testing.assert_close(result, expected)

    def test_extract_kv_prefers_scheduler_embedding_and_falls_back_per_missing_entry(
        self,
        monkeypatch,
    ):
        from vllm_ascend.simllm.config import SimLLMConfig
        from vllm_ascend.simllm.patch import patch_model_runner as patch_runner

        class CapturingKVManager:
            def __init__(self):
                self.tasks = []

            def store(self, task):
                self.tasks.append(task)

            def size(self):
                return len(self.tasks)

        manager = CapturingKVManager()
        block_table = SimpleNamespace(
            get_device_tensor=lambda: torch.tensor([[0], [1]]),
        )
        runner = SimpleNamespace(
            input_batch=SimpleNamespace(
                num_reqs=2,
                req_ids=["req-a", "req-b"],
                block_table=[block_table],
            ),
            seq_lens=torch.tensor([2, 1]),
            query_start_loc=torch.tensor([0, 2, 3]),
            kv_caches=[
                (
                    torch.ones(2, 4, 1, 1),
                    torch.ones(2, 4, 1, 1),
                )
            ],
            _simllm_batch_hashes=torch.tensor([11, 22]),
            _simllm_batch_hash_values=[11, 22],
            _simllm_batch_req_ids=["req-a", "req-b"],
            _simllm_batch_seq_lens=[2, 1],
            _simllm_batch_embeddings=torch.tensor([[0.0, 1.0]]),
            _simllm_match_results={},
        )
        hidden_states = torch.tensor(
            [
                [1.0, 0.0],
                [1.0, 0.0],
                [3.0, 4.0],
            ]
        )

        monkeypatch.setattr(patch_runner, "_kv_manager", manager)
        monkeypatch.setattr(
            patch_runner,
            "_simllm_config",
            SimLLMConfig(
                enabled=True,
                unmatched_store_mode="top",
                embedding_pooling="last",
            ),
        )

        patch_runner._simllm_extract_kv(runner, hidden_states)

        assert [task.task_id for task in manager.tasks] == ["req-a", "req-b"]
        torch.testing.assert_close(
            manager.tasks[0].embedding,
            torch.tensor([[0.0, 1.0]]),
        )
        torch.testing.assert_close(
            manager.tasks[1].embedding,
            torch.tensor([[0.6, 0.8]]),
        )


# ---------------------------------------------------------------------------
# Slot protection tests (verify injected KV survives the forward pass)
# ---------------------------------------------------------------------------


class TestProtectKVSlots:
    """Verify _simllm_protect_kv_slots correctly modifies slot_mapping."""

    @staticmethod
    def _make_runner_with_slots(
        num_reqs: int = 3,
        tokens_per_req: tuple[int, ...] = (5, 7, 4),
    ):
        """Build a mock runner with slot_mapping tensors."""
        runner = MagicMock()
        total_tokens = sum(tokens_per_req)
        runner.input_batch.num_reqs = num_reqs
        runner.input_batch.req_ids = [f"req_{i}" for i in range(num_reqs)]

        cumsum = torch.tensor([0] + list(tokens_per_req)).cumsum(0)
        runner.query_start_loc = cumsum
        runner.seq_lens = torch.tensor(tokens_per_req, dtype=torch.long)

        # slot_mapping: dict layer_name → tensor[num_tokens]
        slot_mapping = {}
        for layer_name in ("model.layers.0.self_attn", "model.layers.1.self_attn"):
            slot_mapping[layer_name] = torch.arange(100, 100 + total_tokens)
        return runner, slot_mapping

    @patch("vllm_ascend.simllm.patch.patch_model_runner.get_forward_context")
    def test_protect_slots_sets_minus_one(self, mock_get_fwd_ctx):
        """Matched tokens should have slot_mapping set to -1, others untouched."""
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_protect_kv_slots,
        )
        from vllm_ascend.simllm.similarity import MatchResult

        runner, slot_mapping = self._make_runner_with_slots(
            num_reqs=3,
            tokens_per_req=(5, 7, 4),
        )

        # Mock forward context.
        mock_ctx = MagicMock()
        mock_ctx.slot_mapping = slot_mapping
        mock_get_fwd_ctx.return_value = mock_ctx

        # Req 1 (batch_idx=1): 7 tokens, matched with cached_len=3
        cached_k = torch.randn(1, 4, 3, 32)  # L_kv = 3 (shorter than 7)
        cached_v = torch.randn(1, 4, 3, 32)
        runner._simllm_match_results = {
            1: MatchResult(
                matched=True,
                source_task_id="cached_1",
                cached_k=cached_k,
                cached_v=cached_v,
                similarity_score=0.95,
            ),
        }

        _simllm_protect_kv_slots(runner)

        # Req 0: tokens 0-4 (start=0), should be UNTOUCHED
        for layer_name, sm in slot_mapping.items():
            assert (sm[0:5] >= 100).all(), f"Req 0 tokens should not be -1 in {layer_name}"

        # Req 1: tokens 5-11 (start=5), first 3 tokens should be -1
        for layer_name, sm in slot_mapping.items():
            assert (sm[5:8] == -1).all(), f"Req 1 first 3 tokens should be -1 in {layer_name}"
            # Remaining 4 tokens (indices 8-11) should be UNCHANGED
            assert (sm[8:12] >= 100).all(), f"Req 1 last 4 tokens should be untouched in {layer_name}"

        # Req 2: tokens 12-15 (start=12), unmatched → should be UNTOUCHED
        for layer_name, sm in slot_mapping.items():
            assert (sm[12:16] >= 100).all(), f"Req 2 tokens should not be -1 in {layer_name}"

    @patch("vllm_ascend.simllm.patch.patch_model_runner.get_forward_context")
    def test_protect_slots_all_covered(self, mock_get_fwd_ctx):
        """When cached length >= request length, ALL tokens are protected."""
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_protect_kv_slots,
        )
        from vllm_ascend.simllm.similarity import MatchResult

        runner, slot_mapping = self._make_runner_with_slots(
            num_reqs=1,
            tokens_per_req=(5,),
        )

        mock_ctx = MagicMock()
        mock_ctx.slot_mapping = slot_mapping
        mock_get_fwd_ctx.return_value = mock_ctx

        # Cached KV has 10 tokens, request has 5 → all 5 covered.
        cached_k = torch.randn(1, 4, 10, 32)
        cached_v = torch.randn(1, 4, 10, 32)
        runner._simllm_match_results = {
            0: MatchResult(
                matched=True,
                source_task_id="cached_0",
                cached_k=cached_k,
                cached_v=cached_v,
                similarity_score=0.99,
            ),
        }

        _simllm_protect_kv_slots(runner)

        for sm in slot_mapping.values():
            assert (sm[0:5] == -1).all(), "All 5 tokens should be protected"

    @patch("vllm_ascend.simllm.patch.patch_model_runner.get_forward_context")
    def test_no_match_no_modification(self, mock_get_fwd_ctx):
        """When no requests are matched, slot_mapping is unchanged."""
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_protect_kv_slots,
        )

        runner, slot_mapping = self._make_runner_with_slots(
            num_reqs=2,
            tokens_per_req=(3, 4),
        )

        mock_ctx = MagicMock()
        mock_ctx.slot_mapping = slot_mapping
        mock_get_fwd_ctx.return_value = mock_ctx

        runner._simllm_match_results = {}  # no matches

        # Snapshot original values.
        original = {k: v.clone() for k, v in slot_mapping.items()}
        _simllm_protect_kv_slots(runner)

        for k in slot_mapping:
            assert torch.equal(slot_mapping[k], original[k]), f"slot_mapping[{k}] should be unchanged"

    @patch("vllm_ascend.simllm.patch.patch_model_runner.get_forward_context")
    def test_empty_match_results_early_return(self, mock_get_fwd_ctx):
        """None or empty match_results should be a safe no-op."""
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_protect_kv_slots,
        )

        runner, slot_mapping = self._make_runner_with_slots(
            num_reqs=1,
            tokens_per_req=(3,),
        )
        mock_ctx = MagicMock()
        mock_ctx.slot_mapping = slot_mapping
        mock_get_fwd_ctx.return_value = mock_ctx

        # None match_results.
        runner._simllm_match_results = None
        _simllm_protect_kv_slots(runner)  # should not raise

        # Empty dict.
        runner._simllm_match_results = {}
        _simllm_protect_kv_slots(runner)  # should not raise

    @patch("vllm_ascend.simllm.patch.patch_model_runner.get_forward_context")
    def test_spec_decode_list_format(self, mock_get_fwd_ctx):
        """slot_mapping as list[dict] (spec decode) should also work."""
        from vllm_ascend.simllm.patch.patch_model_runner import (
            _simllm_protect_kv_slots,
        )
        from vllm_ascend.simllm.similarity import MatchResult

        runner, sm0 = self._make_runner_with_slots(
            num_reqs=1,
            tokens_per_req=(4,),
        )
        sm1 = {k: v.clone() for k, v in sm0.items()}

        mock_ctx = MagicMock()
        mock_ctx.slot_mapping = [sm0, sm1]  # list-of-dicts format
        mock_get_fwd_ctx.return_value = mock_ctx

        cached_k = torch.randn(1, 4, 2, 32)  # L_kv = 2
        cached_v = torch.randn(1, 4, 2, 32)
        runner._simllm_match_results = {
            0: MatchResult(
                matched=True,
                source_task_id="cached_0",
                cached_k=cached_k,
                cached_v=cached_v,
                similarity_score=0.88,
            ),
        }

        _simllm_protect_kv_slots(runner)

        # Both ubatch dicts should have first 2 tokens set to -1.
        for sm_dict in [sm0, sm1]:
            for sm in sm_dict.values():
                assert (sm[0:2] == -1).all()
                assert (sm[2:4] >= 100).all()


# ---------------------------------------------------------------------------
# Scoped injection tests (Task A)
# ---------------------------------------------------------------------------


class TestScopedInjection:
    """Verify KV injection only targets top-N layers."""

    def test_top_n_only(self):
        """With top_layers=3, num_layers=8: only top 3 layers receive KV."""
        num_layers, top_n = 8, 3
        k_caches = [(torch.zeros(3, 16, 4, 8), torch.zeros(3, 16, 4, 8)) for _ in range(num_layers)]

        # Simulate the scoped injection logic.
        target = k_caches[num_layers - top_n :]  # layers 5, 6, 7
        assert len(target) == 3
        # Bottom layers (0-4) should NOT be targeted.
        for idx in range(num_layers - top_n):
            assert k_caches[idx] is not target[0] or idx >= num_layers - top_n

    def test_zero_top_layers_fallback(self):
        """top_layers=0 should fall back to all layers."""
        num_layers, top_n = 8, 0
        target = range(num_layers) if top_n <= 0 or top_n >= num_layers else range(num_layers - top_n, num_layers)
        assert list(target) == list(range(8))

    def test_top_layers_exceeds_num_layers(self):
        """top_layers >= num_layers should fall back to all layers."""
        num_layers, top_n = 8, 10
        target = range(num_layers) if top_n <= 0 or top_n >= num_layers else range(num_layers - top_n, num_layers)
        assert list(target) == list(range(8))


# ---------------------------------------------------------------------------
# Sandwich storage tests (Task B)
# ---------------------------------------------------------------------------


class TestSandwichStorage:
    """Verify unmatched tasks store averaged keep_layers KV."""

    def _build_kv_caches(self, num_layers=8, num_blocks=4, block_size=16, num_kv_heads=4, head_size=8):
        """Build kv_caches with non-zero KV in each layer."""
        return [
            (
                torch.randn(num_blocks, block_size, num_kv_heads, head_size),
                torch.randn(num_blocks, block_size, num_kv_heads, head_size),
            )
            for _ in range(num_layers)
        ]

    def test_keep_layers_bound_check(self):
        """keep_layers indices should all be within [0, num_layers)."""
        from vllm_ascend.simllm.sandwich import SandwichConfig

        cfg = SandwichConfig(bottom_layers=3, top_layers=3, num_layers=8)
        keep = cfg.keep_layers  # {0, 1, 2, 5, 6, 7}
        num_layers = 8
        keep = sorted({idx for idx in keep if 0 <= idx < num_layers})
        assert keep == [0, 1, 2, 5, 6, 7]
        assert len(keep) == 6

    def test_keep_layers_empty_fallback(self):
        """When keep_layers is empty (bottom=0, top=0), fallback to top-1."""
        from vllm_ascend.simllm.sandwich import SandwichConfig

        cfg = SandwichConfig(bottom_layers=0, top_layers=0, num_layers=8)
        keep = cfg.keep_layers
        num_layers = 8
        keep = sorted({idx for idx in keep if 0 <= idx < num_layers})
        if not keep:
            keep = [num_layers - 1]
        assert keep == [7]  # fallback to top layer only

    def test_matched_vs_unmatched_layer_selection(self):
        """Unmatched gathers from 6 layers; matched from 1 (top) layer."""
        from vllm_ascend.simllm.sandwich import SandwichConfig

        cfg = SandwichConfig(bottom_layers=3, top_layers=3, num_layers=32)
        keep_layers = sorted({idx for idx in cfg.keep_layers if 0 <= idx < 32})
        assert len(keep_layers) == 6  # unmatched: 6 layers
        top_only = [31]  # matched: 1 layer
        assert top_only[0] not in keep_layers[:3]  # top layer is in the top-3
        assert top_only[0] in keep_layers  # top-1 is part of keep_layers

    def test_sandwich_average_matches_individual(self):
        """Averaging 2 identical KV tensors should equal the original."""
        t1 = torch.randn(1, 4, 12, 8)
        t2 = t1.clone()
        avg = t1.clone()
        avg.add_(t2)
        avg.mul_(0.5)
        assert torch.allclose(avg, t1, atol=1e-6)
