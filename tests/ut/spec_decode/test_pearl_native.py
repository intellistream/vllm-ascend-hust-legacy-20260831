# SPDX-License-Identifier: Apache-2.0

from multiprocessing import Pipe
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

from vllm_ascend.spec_decode.pearl.api import PEARLConfig, PEARLEngine
from vllm_ascend.spec_decode.pearl.native_cache import NativePrefixCache
from vllm_ascend.spec_decode.pearl.native_engine import (
    NativePearlConfig,
    NativePearlEngine,
    PearlPipelineState,
    SamplingParams,
    _build_greedy_verdict,
    _build_parser,
    _build_stochastic_verdict,
    _finished,
    _gamma_from_decode_speeds,
    _normalize_sampling_params,
    _sample_logits,
    _truncate_completion,
)
from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
from vllm_ascend.spec_decode.pearl.native_model import (
    MIN_PAGED_ATTENTION_BLOCKS,
    PAGED_ATTENTION_BLOCK_SIZE,
    NativeAttention,
    NativeColumnLinear,
    NativeLMHead,
    NativeQwen2ForCausalLM,
    NativeRMSNorm,
    NativeRowLinear,
    NativeTPContext,
    load_native_qwen2_weights,
    prepare_native_model_config,
)


def test_preverify_acceptance_keeps_draft_and_target_states_in_sync():
    target = PearlPipelineState([1, 2, 3], prompt_length=2)
    draft = target.clone()
    next_window = [4, 5, 6, 7]
    draft.token_ids.extend(next_window)

    draft.apply_draft_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=next_window,
    )
    target.apply_target_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=next_window,
    )

    assert draft.token_ids == target.token_ids == [1, 2, 3, 4, 5, 6, 7]
    assert draft.committed_completion_token_ids == target.committed_completion_token_ids == [3, 4]
    assert not draft.pre_verify
    assert draft.accepted_draft_tokens == 1
    assert draft.verified_draft_tokens == 1


def test_postverify_rejection_rolls_back_the_same_pipeline_suffix_on_both_sides():
    target = PearlPipelineState(
        [1, 2, 3, 4, 5],
        prompt_length=2,
        pre_verify=False,
        committed_length=2,
    )
    draft = target.clone()
    next_window = [6, 7, 8, 9]
    draft.token_ids.extend(next_window)

    draft.apply_draft_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=next_window,
    )
    target.apply_target_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=next_window,
    )

    assert draft.token_ids == target.token_ids == [1, 2, 3, 4, 42]
    assert draft.committed_completion_token_ids == target.committed_completion_token_ids == [3, 4, 42]
    assert draft.pre_verify
    assert draft.accepted_draft_tokens == 2
    assert draft.verified_draft_tokens == 4


def test_native_qwen2_model_runs_with_a_single_tensor_parallel_rank_on_cpu():
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    model.configure_cache(16)

    hidden_states = model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))
    logits = model.compute_logits(hidden_states)
    greedy_tokens = model.compute_greedy_tokens(hidden_states, vocabulary_size=31)

    assert hidden_states.shape == (3, 16)
    assert logits.shape == (3, 32)
    assert torch.equal(greedy_tokens, logits[:, :31].argmax(dim=-1))


@pytest.mark.parametrize("architecture", ["Qwen2ForCausalLM", "Qwen3ForCausalLM", "LlamaForCausalLM"])
def test_native_model_runs_every_upstream_architecture_on_cpu(architecture):
    config = SimpleNamespace(
        architectures=[architecture],
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
        rope_parameters={"rope_theta": 10_000.0, "rope_type": "default"},
        rms_norm_eps=1e-6,
        intermediate_size=32,
        hidden_act="silu",
        attention_bias=architecture == "LlamaForCausalLM",
        mlp_bias=architecture == "LlamaForCausalLM",
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    model.configure_cache(16)

    hidden_states = model(torch.tensor([1, 2, 3]), torch.tensor([0, 1, 2]))

    assert hidden_states.shape == (3, 16)
    attention = model.layers[0].self_attn
    assert isinstance(attention.q_norm, NativeRMSNorm) == (architecture == "Qwen3ForCausalLM")
    assert (attention.o_proj.bias is not None) == (architecture == "LlamaForCausalLM")
    assert (model.layers[0].mlp.gate_up_proj.bias is not None) == (architecture == "LlamaForCausalLM")


def test_dynamic_tp_config_matches_upstream_padding_rules():
    config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        vocab_size=101,
        hidden_size=1280,
        num_attention_heads=10,
        num_key_value_heads=2,
        intermediate_size=1000,
    )

    prepared = prepare_native_model_config(config, tensor_parallel_size=3)

    assert prepared is not config
    assert prepared.head_dim == 128
    assert prepared.num_attention_heads == 15
    assert prepared.num_key_value_heads == 3
    assert prepared.intermediate_size == 1152
    assert prepared.vocab_size == 102
    assert prepared.valid_vocab_size == 101


def test_dynamic_tp_weight_loaders_zero_pad_the_final_partition():
    context = NativeTPContext(group=None, rank=2, size=3, leader_rank=0)
    column = NativeColumnLinear(2, 12, context)
    row = NativeRowLinear(12, 2, context)
    loaded_column = torch.arange(20, dtype=torch.float32).view(10, 2)
    loaded_row = torch.arange(20, dtype=torch.float32).view(2, 10)

    column.load_weight(loaded_column)
    row.load_weight(loaded_row)

    assert torch.equal(column.weight[:2], loaded_column[8:10])
    assert torch.count_nonzero(column.weight[2:]) == 0
    assert torch.equal(row.weight[:, :2], loaded_row[:, 8:10])
    assert torch.count_nonzero(row.weight[:, 2:]) == 0


def test_native_attention_allocates_vllm_compatible_paged_cache():
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
    )
    attention = NativeAttention(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )

    attention.configure_cache(PAGED_ATTENTION_BLOCK_SIZE + 1, use_paged_attention=True)

    assert attention.uses_paged_attention
    assert attention.key_cache is not None
    assert attention.key_cache.shape == (
        MIN_PAGED_ATTENTION_BLOCKS,
        PAGED_ATTENTION_BLOCK_SIZE,
        1,
        8,
    )
    assert attention.block_table is not None
    assert attention.block_table.dtype == torch.int32
    assert attention.block_table.tolist() == [[0, 1]]
    assert attention.context_lens is not None
    assert attention.context_lens.device.type == "cpu"


def test_native_attention_writes_a_packed_gqa_batch_with_one_cann_call():
    config = SimpleNamespace(
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=32,
        rope_theta=10_000.0,
    )
    attention = NativeAttention(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    attention.configure_cache(16, use_paged_attention=True)
    packed_qkv = torch.randn(4, 32)
    key = packed_qkv[:, 16:24].view(4, 1, 8)
    value = packed_qkv[:, 24:].view(4, 1, 8)

    with patch("vllm_ascend.spec_decode.pearl.native_model.DeviceOperator.reshape_and_cache") as cache_op:
        attention._write_to_cache(torch.arange(4), key, value)

    cache_op.assert_called_once()
    assert cache_op.call_args.kwargs["key"] is key
    assert cache_op.call_args.kwargs["value"] is value
    assert cache_op.call_args.kwargs["slot_mapping"].dtype == torch.int32


def test_native_lm_head_greedy_uses_one_tensor_parallel_collective():
    context = NativeTPContext(group=MagicMock(), rank=0, size=2, leader_rank=0)
    lm_head = NativeLMHead(vocab_size=8, hidden_size=2, context=context)
    lm_head.weight.data.copy_(torch.tensor([[1.0, 0.0], [0.0, 1.0], [2.0, 0.0], [0.0, 2.0]]))

    def copy_local_candidate(outputs, candidate, **_kwargs):
        outputs[0].copy_(candidate)
        outputs[1].copy_(candidate)

    with patch(
        "vllm_ascend.spec_decode.pearl.native_model.dist.all_gather",
        side_effect=copy_local_candidate,
    ) as all_gather:
        tokens = lm_head.greedy(torch.tensor([[1.0, 0.0]]), vocabulary_size=8)

    assert tokens.tolist() == [2]
    all_gather.assert_called_once()


def test_native_model_builds_disjoint_paged_metadata_for_a_static_batch():
    config = SimpleNamespace(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        max_position_embeddings=512,
        rope_theta=10_000.0,
        rms_norm_eps=1e-6,
        intermediate_size=32,
        tie_word_embeddings=False,
        num_hidden_layers=1,
    )
    model = NativeQwen2ForCausalLM(
        config,
        NativeTPContext(group=None, rank=0, size=1, leader_rank=0),
    )
    model.configure_cache(PAGED_ATTENTION_BLOCK_SIZE + 1, max_num_seqs=2)

    positions, metadata = model.make_attention_metadata([0, 1, 1], [128, 0, 128])

    assert positions.tolist() == [128, 0, 128]
    assert metadata.slot_mapping.tolist() == [128, 256, 384]
    assert metadata.context_lens.tolist() == [129, 1, 129]
    assert metadata.context_lens.device.type == "cpu"
    assert metadata.block_tables.tolist() == [[0, 1], [2, 3], [2, 3]]
    assert metadata.actual_seq_lengths_q == (1, 3)
    assert metadata.sequence_lens == (129, 129)
    assert metadata.request_block_tables.tolist() == [[0, 1], [2, 3]]

    _, remapped = model.make_attention_metadata(
        [0, 1],
        [128, 0],
        block_tables=[[4, 6], [3, 5]],
    )
    assert remapped.slot_mapping.tolist() == [6 * PAGED_ATTENTION_BLOCK_SIZE, 3 * PAGED_ATTENTION_BLOCK_SIZE]
    assert remapped.block_tables.tolist() == [[4, 6], [3, 5]]

    _, direct_slots = model.make_attention_metadata(
        [0, 1],
        [128, 0],
        block_tables=[[4, 6], [3, 5]],
        slot_mapping=[777, 888],
    )
    assert direct_slots.slot_mapping.tolist() == [777, 888]


def test_native_prefix_cache_shares_full_prompt_blocks_within_and_across_batches():
    cache = NativePrefixCache(num_blocks=8, blocks_per_sequence=2, block_size=4)
    shared_prefix = [1, 2, 3, 4]

    first = cache.allocate([shared_prefix + [5], shared_prefix + [9]])

    assert first.num_cached_tokens == [0, 4]
    assert first.block_tables[0][0] == first.block_tables[1][0]
    cache.release()

    second = cache.allocate([shared_prefix + [7]])

    assert second.num_cached_tokens == [4]
    assert second.block_tables[0][0] == first.block_tables[0][0]
    cache.release()


def test_native_prefix_cache_allocates_decode_pages_lazily():
    cache = NativePrefixCache(num_blocks=3, blocks_per_sequence=4, block_size=4)
    allocation = cache.allocate([[1], [2]])

    assert allocation.block_tables == [[0, -1, -1, -1], [1, -1, -1, -1]]
    updates = cache.ensure_capacity([0], [4])
    assert updates == [(0, 1, 2)]
    assert allocation.block_tables[0][1] == 2
    with pytest.raises(RuntimeError, match="no free physical blocks"):
        cache.ensure_capacity([1], [4])
    cache.release()


def test_target_ar_draft_rank_returns_without_allocating_or_running_the_model():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(
        max_num_seqs=1,
        max_num_batched_tokens=16,
        max_tokens=4,
        max_model_len=16,
    )
    engine.is_draft = True
    engine._allocate_cache = MagicMock()

    result = engine.generate_target_ar_batch(
        [[1, 2]],
        SamplingParams(temperature=0, max_tokens=4),
    )

    assert result is None
    engine._allocate_cache.assert_not_called()


def test_native_engine_builds_slots_from_the_cpu_cache_page_table():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.config = SimpleNamespace(kvcache_block_size=128)
    engine.cache_allocation = SimpleNamespace(block_tables=[[3, 7], [4, 9]])

    slots = engine._cache_slot_mapping([0, 1], [129, 2])

    assert slots == [7 * 128 + 1, 4 * 128 + 2]


@pytest.mark.parametrize("temperature", [0.0, 0.7])
def test_target_postverify_uses_speculative_fia_aclgraph(temperature):
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    state = PearlPipelineState(
        [1, 2, 3, 4, 5, 6],
        prompt_length=2,
        pre_verify=False,
        temperature=temperature,
    )
    engine._run_packed_greedy = MagicMock(return_value=torch.tensor([1, 2, 3, 4]))
    engine._run_packed_model = MagicMock(return_value=torch.randn(4, 32))

    engine._target_round_outputs_batch([state], [0])

    runner = engine._run_packed_greedy if temperature == 0 else engine._run_packed_model
    assert runner.call_args.kwargs["use_aclgraph"] is True
    assert runner.call_args.kwargs["use_fused_infer_attention"] is True


def test_target_preverify_uses_speculative_fia_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = False
    engine.gamma = 4
    engine.draft_vocab_size = 32
    engine.config = SimpleNamespace(enforce_eager=False)
    state = PearlPipelineState([1, 2, 3], prompt_length=2, pre_verify=True)
    engine._run_packed_greedy = MagicMock(return_value=torch.tensor([4]))

    engine._target_round_outputs_batch([state], [0])

    assert engine._run_packed_greedy.call_args.kwargs["use_aclgraph"] is True
    assert engine._run_packed_greedy.call_args.kwargs["use_fused_infer_attention"] is True


def test_greedy_graph_forwards_the_speculative_fia_backend():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.device = torch.device("cpu")
    engine._run_device_packed_greedy = MagicMock(return_value=torch.tensor([4]))

    engine._run_packed_greedy(
        [3],
        [0],
        [2],
        use_aclgraph=True,
        use_fused_infer_attention=True,
    )

    assert engine._run_device_packed_greedy.call_args.kwargs == {
        "use_aclgraph": True,
        "use_fused_infer_attention": True,
    }


def test_draft_round_uses_speculative_fia_aclgraph():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False)
    engine._run_device_packed_greedy = MagicMock(
        side_effect=[torch.tensor([4]), torch.tensor([5])],
    )
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    verification, continuation = engine._draft_round_batch([state], [0])

    assert verification == [[4]]
    assert continuation == [[4, 5]]
    assert all(call.kwargs["use_aclgraph"] is True for call in engine._run_device_packed_greedy.call_args_list)
    assert all(
        call.kwargs["use_fused_infer_attention"] is True for call in engine._run_device_packed_greedy.call_args_list
    )


def test_draft_round_snapshots_a_reused_aclgraph_output_buffer():
    engine = NativePearlEngine.__new__(NativePearlEngine)
    engine.is_draft = True
    engine.gamma = 2
    engine.device = torch.device("cpu")
    engine.config = SimpleNamespace(enforce_eager=False)
    shared_output = torch.tensor([0])

    def replay(*_args, **_kwargs):
        shared_output.add_(1)
        return shared_output

    engine._run_device_packed_greedy = MagicMock(side_effect=replay)
    state = PearlPipelineState([1, 2, 3], prompt_length=2)

    _, continuation = engine._draft_round_batch([state], [0])

    assert continuation == [[1, 2]]


def test_completion_is_truncated_at_the_first_eos_or_token_limit():
    assert _truncate_completion([10, 11, 99, 12], frozenset((99,)), 4) == [10, 11, 99]
    assert _truncate_completion([10, 11, 12], frozenset(), 2) == [10, 11]
    assert _truncate_completion([10, 99, 12], frozenset((99,)), 3, ignore_eos=True) == [10, 99, 12]


def test_sampling_params_are_per_request_and_reject_mixed_temperature_modes():
    params = SamplingParams(temperature=0.7, max_tokens=12, ignore_eos=True)

    assert _normalize_sampling_params(2, params, 64) == [params, params]
    with pytest.raises(ValueError, match="all zero or all non-zero"):
        _normalize_sampling_params(
            2,
            [SamplingParams(temperature=0), SamplingParams(temperature=1)],
            64,
        )

    state = PearlPipelineState(
        [1, 99],
        prompt_length=1,
        committed_length=2,
        max_tokens=2,
        ignore_eos=True,
    )
    assert not _finished(state, frozenset((99,)))
    state.token_ids.append(2)
    state.committed_length = 3
    assert _finished(state, frozenset((99,)))


def test_target_sampler_supports_greedy_and_exponential_race_sampling():
    logits = torch.tensor([[1.0, 3.0, 2.0], [2.0, 1.0, 3.0]])

    assert _sample_logits(logits, [0.0, 0.0]).tolist() == [1, 2]
    assert _sample_logits(
        logits,
        [1.0, 1.0],
        exponential_noise=torch.ones_like(logits),
    ).tolist() == [1, 2]


def test_stochastic_verdict_accepts_prefix_and_samples_masked_correction():
    target_logits = torch.tensor([[8.0, 1.0, 0.0]] * 5)
    draft_tokens = torch.zeros(5, dtype=torch.long)

    verdict = _build_stochastic_verdict(
        target_logits,
        draft_tokens,
        verification_sizes=[1, 4],
        gamma=4,
        temperatures=[1.0] * 5,
        random_values=torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        exponential_noise=torch.ones_like(target_logits),
    )

    assert verdict.tolist() == [[1, -1], [2, 1]]


def test_pipeline_state_reports_upstream_mat_segments():
    state = PearlPipelineState([1, 2, 3], prompt_length=2)
    assert state.acceptance_lengths == []
    state.apply_target_verification(
        gamma=4,
        accepted=1,
        correction_token_id=None,
        next_round_token_ids=[4, 5, 6, 7],
    )
    state.apply_target_verification(
        gamma=4,
        accepted=2,
        correction_token_id=42,
        next_round_token_ids=[8, 9, 10, 11],
    )

    assert state.acceptance_lengths == [4, 0]
    assert sum(state.acceptance_lengths) / len(state.acceptance_lengths) == 2.0


def test_auto_gamma_uses_the_upstream_draft_to_target_speed_ratio():
    assert _gamma_from_decode_speeds(700.0, 100.0) == 7
    assert _gamma_from_decode_speeds(50.0, 100.0) == 1
    assert _gamma_from_decode_speeds(10_000.0, 100.0) == 100

    config = NativePearlConfig("draft", "target", 1, 2, -1, 512, 32)
    assert config.gamma == -1


def test_direct_worker_cli_exposes_cache_capacity_controls():
    args = _build_parser().parse_args(
        [
            "--draft-model",
            "draft",
            "--target-model",
            "target",
            "--prompt",
            "hello",
            "--gpu-memory-utilization",
            "0.98",
            "--num-kvcache-blocks",
            "32",
            "--max-aclgraph-entries",
            "8",
        ]
    )

    assert args.gpu_memory_utilization == 0.98
    assert args.num_kvcache_blocks == 32
    assert args.max_aclgraph_entries == 8


def test_public_pearl_config_maps_upstream_fields_to_native_runtime():
    model_config = SimpleNamespace(
        architectures=["Qwen2ForCausalLM"],
        eos_token_id=[1, 2],
    )
    with patch(
        "vllm_ascend.spec_decode.pearl.api.AutoConfig.from_pretrained",
        side_effect=[model_config, model_config],
    ):
        config = PEARLConfig(
            "draft",
            "target",
            draft_tensor_parallel_size=1,
            target_tensor_parallel_size=2,
            max_model_len=512,
            max_num_batched_tokens=1024,
            max_num_seqs=8,
            gamma=4,
        )

    native = config.to_native()
    assert config.world_size == 3
    assert config.eos == [1, 2]
    assert config.draft_config.model == "draft"
    assert config.draft_config.devices == [0]
    assert config.target_config.model == "target"
    assert config.target_config.devices == [1, 2]
    assert config.target_config.master_rank == 1
    assert native.draft_model == "draft"
    assert native.target_tp_size == 2
    assert native.max_num_batched_tokens == 1024
    assert native.max_aclgraph_entries == 16
    assert native.max_num_seqs == 8


def test_public_pearl_config_rejects_nonpositive_worker_timeout():
    with pytest.raises(ValueError, match="worker_timeout_seconds"):
        PEARLConfig("draft", "target", worker_timeout_seconds=0)


def test_public_engine_collects_worker_replies_in_rank_order():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(worker_timeout_seconds=1.0)
    engine._processes = [MagicMock(), MagicMock()]
    for process in engine._processes:
        process.is_alive.return_value = True
    rank0, worker0 = Pipe(duplex=True)
    rank1, worker1 = Pipe(duplex=True)
    engine._connections = [rank0, rank1]
    try:
        worker1.send(("ready", 1))
        worker0.send(("ready", 0))

        assert engine._receive_all("test") == [("ready", 0), ("ready", 1)]
    finally:
        for connection in (rank0, rank1, worker0, worker1):
            connection.close()


def test_public_engine_times_out_with_pending_worker_ranks():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(worker_timeout_seconds=0.01)
    process = MagicMock()
    process.is_alive.return_value = True
    parent, worker = Pipe(duplex=True)
    engine._processes = [process]
    engine._connections = [parent]
    try:
        with pytest.raises(TimeoutError, match=r"pending ranks: \[0\]"):
            engine._receive_all("test timeout")
    finally:
        parent.close()
        worker.close()


def test_public_engine_broadcasts_upstream_log_command():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine._send_all = MagicMock()
    engine._receive_all = MagicMock(return_value=[("logged", 0), ("logged", 1)])

    engine.log("ready")

    engine._send_all.assert_called_once_with(("log", "ready", None, None))
    engine._receive_all.assert_called_once_with("worker logging")


def test_public_engine_chunks_queued_requests_by_sequence_and_token_limits():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(max_num_seqs=2, max_num_batched_tokens=5)
    params = SamplingParams(temperature=0, max_tokens=4)
    requests = [
        (0, [1, 2, 3], params),
        (1, [4, 5], params),
        (2, [6, 7, 8], params),
    ]

    chunks = engine._request_chunks(requests)

    assert [[request[0] for request in chunk] for chunk in chunks] == [[0, 1], [2]]


def test_public_engine_validates_pipeline_and_benchmark_cache_capacity_before_dispatch():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(gamma=4, max_model_len=10)
    engine._requests = [(0, [1, 2, 3, 4, 5], SamplingParams(temperature=0, max_tokens=2))]

    with pytest.raises(ValueError, match="verification window"):
        engine.generate()
    with pytest.raises(ValueError, match="benchmark steps"):
        engine.bench_generate(num_pearl_steps=1)


def test_public_engine_aggregates_aclgraph_metrics_from_every_worker():
    engine = PEARLEngine.__new__(PEARLEngine)
    engine.config = SimpleNamespace(
        gamma=4,
        max_model_len=32,
        max_num_seqs=1,
        max_num_batched_tokens=32,
        worker_timeout_seconds=1,
    )
    engine._requests = [(0, [1], SamplingParams(max_tokens=1))]
    engine.last_metrics = []
    engine.tokenizer = MagicMock()
    engine.tokenizer.decode.return_value = "result"
    engine._send_all = MagicMock()
    leader_result = {
        "completion_token_ids": [2],
        "num_acc_tokens": [1],
        "elapsed_seconds": 1.0,
    }
    base_metrics = {
        "aclgraph_captures": 2,
        "aclgraph_capture_attempts": 2,
        "aclgraph_replays": 3,
        "aclgraph_failed_captures": 0,
        "aclgraph_capacity_fallbacks": 0,
        "aclgraph_shape_fallbacks": 0,
    }
    failed_rank_metrics = {
        **base_metrics,
        "aclgraph_capture_attempts": 8,
        "aclgraph_failed_captures": 6,
        "aclgraph_capacity_fallbacks": 9,
        "aclgraph_shape_fallbacks": 11,
    }
    engine._receive_all = MagicMock(
        return_value=[
            ("result", None, failed_rank_metrics),
            ("result", [leader_result], base_metrics),
        ]
    )

    _, num_tokens, _, elapsed = engine._generate("pearl")

    assert num_tokens == [1]
    assert elapsed == 1.0
    assert engine.last_metrics[0]["aclgraph_capture_attempts"] == 8
    assert engine.last_metrics[0]["aclgraph_failed_captures"] == 6
    assert engine.last_metrics[0]["aclgraph_capacity_fallbacks"] == 9
    assert engine.last_metrics[0]["aclgraph_shape_fallbacks"] == 11


def test_greedy_verdict_finds_first_mismatch_in_packed_mixed_windows():
    target_tokens = torch.tensor([5, 10, 21, 32, 40, 51, 61, 71, 81])
    draft_tokens = torch.tensor([5, 10, 20, 30, 40, 50, 60, 70, 80])

    verdict = _build_greedy_verdict(
        target_tokens,
        draft_tokens,
        verification_sizes=[1, 4, 4],
        gamma=4,
    )

    assert verdict.tolist() == [[1, -1], [1, 21], [0, 51]]


def test_native_aclgraph_padding_preserves_tokens_and_uses_inactive_cache_slots():
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([9, 10]),
        context_lens=torch.tensor([3, 4], dtype=torch.int32),
        block_tables=torch.tensor([[1, 2], [3, 4]], dtype=torch.int32),
    )

    input_ids, positions, padded = NativeACLGraphRunner._pad_inputs(
        torch.tensor([5, 6]),
        torch.tensor([1, 2]),
        metadata,
        4,
    )

    assert input_ids.tolist() == [5, 6, 0, 0]
    assert positions.tolist() == [1, 2, 0, 0]
    assert padded.slot_mapping.tolist() == [9, 10, -1, -1]
    assert padded.context_lens.tolist() == [3, 4, 0, 0]
    assert padded.block_tables.tolist() == [[1, 2], [3, 4], [0, 0], [0, 0]]


def test_native_aclgraph_eager_greedy_path_includes_lm_head_sampling():
    model = MagicMock()
    hidden_states = torch.randn(2, 4)
    model.return_value = hidden_states
    model.compute_greedy_tokens.return_value = torch.tensor([3, 5])
    runner = NativeACLGraphRunner(model, enabled=False)
    metadata = SimpleNamespace(
        slot_mapping=torch.tensor([0, 1], dtype=torch.int32),
        context_lens=torch.tensor([1, 1], dtype=torch.int32),
        block_tables=torch.tensor([[0], [1]], dtype=torch.int32),
    )

    tokens = runner.run_greedy(
        torch.tensor([1, 2]),
        torch.tensor([0, 0]),
        metadata,
        vocabulary_size=8,
    )

    assert tokens.tolist() == [3, 5]
    model.compute_greedy_tokens.assert_called_once_with(hidden_states, 8)


def test_native_aclgraph_capacity_uses_eager_for_new_shapes():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1,),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.entries[("existing", 1)] = MagicMock()
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.capacity_fallback_count == 1
    runner._execute.assert_called_once()


def test_native_aclgraph_capacity_counts_failed_capture_attempts():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1,),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.capture_attempt_count = 1
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.entries == {}
    assert runner.capacity_fallback_count == 1
    runner._execute.assert_called_once()


def test_native_aclgraph_skips_dynamic_fia_tail_without_spending_capture_budget():
    model = MagicMock()
    metadata = SimpleNamespace(
        use_fused_infer_attention=True,
        actual_seq_lengths_q=(1, 2, 3, 4, 5, 6, 7),
    )
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(model, enabled=True, max_graph_entries=1)
    runner.set_expected_fia_batch_size(8)
    runner._execute = MagicMock(return_value=torch.tensor([7]))

    output = runner.run_greedy(
        torch.tensor([1]),
        torch.tensor([0]),
        metadata,
        vocabulary_size=8,
    )

    assert output.tolist() == [7]
    assert runner.capture_attempt_count == 0
    assert runner.shape_fallback_count == 1
    runner._execute.assert_called_once()


@pytest.mark.parametrize(
    ("actual_seq_lengths_q", "expected"),
    [
        ((1, 2, 3, 4), True),
        ((4, 8, 12, 16), True),
        ((1, 5, 6, 10), False),
    ],
)
def test_native_aclgraph_only_reuses_uniform_fia_segments(actual_seq_lengths_q, expected):
    with patch("vllm_ascend.spec_decode.pearl.native_graph.torch.npu.Stream"):
        runner = NativeACLGraphRunner(MagicMock(), enabled=True)
    runner.set_expected_fia_batch_size(4)

    assert runner._is_reusable_fia_shape(actual_seq_lengths_q) is expected


def test_native_aclgraph_rejects_nonpositive_entry_capacity():
    with pytest.raises(ValueError, match="max_graph_entries"):
        NativeACLGraphRunner(MagicMock(), enabled=False, max_graph_entries=0)


def test_native_weight_loader_uses_safe_open_keys_api(tmp_path):
    weight_file = tmp_path / "model.safetensors"
    weight_file.touch()
    checkpoint = MagicMock()
    checkpoint.keys.return_value = []
    model = MagicMock()
    model.packed_modules_mapping = {}

    with patch("vllm_ascend.spec_decode.pearl.native_model.safe_open") as safe_open:
        safe_open.return_value.__enter__.return_value = checkpoint
        load_native_qwen2_weights(model, str(tmp_path))

    checkpoint.keys.assert_called_once_with()
