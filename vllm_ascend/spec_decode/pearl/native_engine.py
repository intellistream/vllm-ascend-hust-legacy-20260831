# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native Ascend port of nano-PEARL's persistent two-model decode loop.

Run this module under ``torchrun``.  Unlike the OpenAI-compatible bridge,
all ranks form one HCCL world and retain their own model KV cache across every
PEARL round.  The round ordering matches nano-PEARL:

``pre-verify -> gamma draft tokens -> packed target verification -> rollback``.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from transformers import AutoConfig, AutoTokenizer

from vllm_ascend.spec_decode.pearl.native_cache import NativeCacheAllocation, NativePrefixCache
from vllm_ascend.spec_decode.pearl.native_graph import NativeACLGraphRunner
from vllm_ascend.spec_decode.pearl.native_model import (
    PAGED_ATTENTION_BLOCK_SIZE,
    NativeTPContext,
    build_native_model,
    load_native_model_weights,
)
from vllm_ascend.spec_decode.pearl.qwen_pair import validate_model_pair
from vllm_ascend.spec_decode.pearl.topology import PearlProcessGroups, PearlTopology

AUTO_GAMMA_BATCH_SIZES = (1, 2, 4, 8, 16, 32)
AUTO_GAMMA_WARMUP_STEPS = 5
AUTO_GAMMA_PROFILE_STEPS = 30
AUTO_GAMMA_PROFILE_SEQUENCE_LENGTH = 256


@dataclass
class PearlPipelineState:
    """Logical sequence state replicated by the draft and target model groups."""

    token_ids: list[int]
    prompt_length: int
    pre_verify: bool = True
    accepted_draft_tokens: int = 0
    verified_draft_tokens: int = 0
    committed_length: int | None = None
    temperature: float = 0.0
    max_tokens: int = 64
    ignore_eos: bool = False
    num_acc_tokens: list[int] | None = None
    cur_acc_tokens: int = 0

    def __post_init__(self) -> None:
        if self.committed_length is None:
            self.committed_length = len(self.token_ids)
        if self.num_acc_tokens is None:
            self.num_acc_tokens = []
        if self.temperature < 0 or self.max_tokens <= 0:
            raise ValueError("PEARL sampling requires non-negative temperature and positive max_tokens.")

    def clone(self) -> PearlPipelineState:
        return PearlPipelineState(
            token_ids=list(self.token_ids),
            prompt_length=self.prompt_length,
            pre_verify=self.pre_verify,
            accepted_draft_tokens=self.accepted_draft_tokens,
            verified_draft_tokens=self.verified_draft_tokens,
            committed_length=self.committed_length,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            ignore_eos=self.ignore_eos,
            num_acc_tokens=list(self.num_acc_tokens or ()),
            cur_acc_tokens=self.cur_acc_tokens,
        )

    @property
    def completion_token_ids(self) -> list[int]:
        return self.token_ids[self.prompt_length :]

    @property
    def committed_completion_token_ids(self) -> list[int]:
        assert self.committed_length is not None
        return self.token_ids[self.prompt_length : self.committed_length]

    def apply_target_verification(
        self,
        *,
        gamma: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: list[int],
    ) -> None:
        """Apply nano-PEARL's target-side append/rollback transition."""
        self._validate_verification(gamma, accepted, correction_token_id, next_round_token_ids)
        was_pre_verify = self.pre_verify
        assert self.committed_length is not None
        self.committed_length += accepted + (accepted < (1 if was_pre_verify else gamma))
        self.accepted_draft_tokens += accepted
        self.verified_draft_tokens += 1 if was_pre_verify else gamma
        self._record_acceptance(expected=1 if was_pre_verify else gamma, accepted=accepted)
        if accepted == (1 if was_pre_verify else gamma):
            self.token_ids.extend(next_round_token_ids)
            self.pre_verify = False
            return

        assert correction_token_id is not None
        if not was_pre_verify:
            rollout = gamma - accepted
            if rollout > 1:
                del self.token_ids[-(rollout - 1) :]
        self.token_ids.append(correction_token_id)
        self.pre_verify = True

    def apply_draft_verification(
        self,
        *,
        gamma: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: list[int],
    ) -> None:
        """Apply the mirrored draft-side rollback after a target verdict."""
        self._validate_verification(gamma, accepted, correction_token_id, next_round_token_ids)
        was_pre_verify = self.pre_verify
        assert self.committed_length is not None
        self.committed_length += accepted + (accepted < (1 if was_pre_verify else gamma))
        self.accepted_draft_tokens += accepted
        self.verified_draft_tokens += 1 if was_pre_verify else gamma
        self._record_acceptance(expected=1 if was_pre_verify else gamma, accepted=accepted)
        if accepted == (1 if was_pre_verify else gamma):
            self.pre_verify = False
            return

        assert correction_token_id is not None
        del self.token_ids[-gamma:]
        if not was_pre_verify:
            rollout = gamma - accepted
            if rollout > 1:
                del self.token_ids[-(rollout - 1) :]
        self.token_ids.append(correction_token_id)
        self.pre_verify = True

    def _validate_verification(
        self,
        gamma: int,
        accepted: int,
        correction_token_id: int | None,
        next_round_token_ids: list[int],
    ) -> None:
        expected = 1 if self.pre_verify else gamma
        if len(next_round_token_ids) != gamma:
            raise ValueError("Every PEARL next-round window must contain gamma draft tokens.")
        if not 0 <= accepted <= expected:
            raise ValueError("PEARL accepted length is outside the verification window.")
        if accepted == expected and correction_token_id is not None:
            raise ValueError("A fully accepted PEARL window cannot include a correction token.")
        if accepted < expected and correction_token_id is None:
            raise ValueError("A rejected PEARL window requires a target correction token.")

    def _record_acceptance(self, *, expected: int, accepted: int) -> None:
        assert self.num_acc_tokens is not None
        if accepted == expected:
            self.cur_acc_tokens += accepted
        else:
            # Upstream MAT treats the target correction as part of the output
            # segment terminated by this rejection.
            self.num_acc_tokens.append(self.cur_acc_tokens + accepted + 1)
            self.cur_acc_tokens = 0

    @property
    def acceptance_lengths(self) -> list[int]:
        if self.verified_draft_tokens == 0:
            return []
        return [*(self.num_acc_tokens or ()), self.cur_acc_tokens]


@dataclass(frozen=True)
class NativeSamplingParams:
    """Per-request sampling controls supported by upstream nano-PEARL."""

    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("PEARL temperature must be non-negative.")
        if self.max_tokens <= 0:
            raise ValueError("PEARL max_tokens must be positive.")


# Match upstream's public name while retaining a native-specific explicit name.
SamplingParams = NativeSamplingParams


@dataclass(frozen=True)
class NativePearlConfig:
    draft_model: str
    target_model: str
    draft_tp_size: int
    target_tp_size: int
    gamma: int
    max_model_len: int
    max_tokens: int
    max_num_seqs: int = 1
    max_num_batched_tokens: int = 16384
    gpu_memory_utilization: float = 0.9
    kvcache_block_size: int = PAGED_ATTENTION_BLOCK_SIZE
    num_kvcache_blocks: int = -1
    enable_prefix_caching: bool = True
    enforce_eager: bool = False
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.draft_tp_size <= 0 or self.target_tp_size <= 0:
            raise ValueError("Draft and target TP sizes must be positive.")
        if self.gamma == 0 or self.gamma < -1:
            raise ValueError("PEARL gamma must be positive, or -1 for automatic selection.")
        if self.max_model_len <= 0 or self.max_tokens <= 0 or self.max_num_seqs <= 0:
            raise ValueError("PEARL model, generation, and batch limits must be positive.")
        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError("PEARL max_num_batched_tokens must be at least max_model_len.")
        if not 0 < self.gpu_memory_utilization <= 1:
            raise ValueError("PEARL gpu_memory_utilization must be in (0, 1].")
        if self.kvcache_block_size != PAGED_ATTENTION_BLOCK_SIZE:
            raise ValueError(f"Native Ascend PEARL requires kvcache_block_size={PAGED_ATTENTION_BLOCK_SIZE}.")
        if self.num_kvcache_blocks == 0 or self.num_kvcache_blocks < -1:
            raise ValueError("PEARL num_kvcache_blocks must be positive, or -1 for automatic sizing.")
        if self.seed is not None and self.seed < 0:
            raise ValueError("PEARL sampling seed must be non-negative.")


class NativePearlEngine:
    """One rank of nano-PEARL's persistent HCCL runtime."""

    def __init__(self, config: NativePearlConfig) -> None:
        self.config = config
        self.gamma = config.gamma
        # torch.distributed's env:// rendezvous is established by torchrun.
        if not dist.is_initialized():
            dist.init_process_group(backend="hccl")
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.local_rank = int(__import__("os").environ.get("LOCAL_RANK", self.rank))
        torch.npu.set_device(self.local_rank)
        self.device = torch.device("npu")

        self.topology = PearlTopology.from_tensor_parallel_sizes(
            config.draft_tp_size,
            config.target_tp_size,
        )
        self.groups = PearlProcessGroups.create(self.topology, backend="hccl")
        if self.world_size != self.topology.world_size:
            raise ValueError(f"PEARL needs {self.topology.world_size} ranks, received {self.world_size}.")

        self.is_draft = self.groups.is_draft_worker
        self.model_group = self.groups.model_group
        self.model_context = NativeTPContext(
            group=self.model_group,
            rank=self.rank if self.is_draft else self.rank - config.draft_tp_size,
            size=config.draft_tp_size if self.is_draft else config.target_tp_size,
            leader_rank=self.topology.draft_leader_rank if self.is_draft else self.topology.target_leader_rank,
        )

        projection = validate_model_pair(config.draft_model, config.target_model)
        self.draft_vocab_size = projection.draft_vocab_size
        self.target_vocab_size = projection.target_vocab_size
        draft_model_config = AutoConfig.from_pretrained(config.draft_model)
        target_model_config = AutoConfig.from_pretrained(config.target_model)
        if _normalize_eos_tokens(draft_model_config.eos_token_id) != _normalize_eos_tokens(
            target_model_config.eos_token_id
        ):
            raise ValueError("Native PEARL requires identical draft and target EOS token IDs.")
        model_path = config.draft_model if self.is_draft else config.target_model
        model_config = draft_model_config if self.is_draft else target_model_config
        self.model = build_native_model(
            model_config,
            self.model_context,
            config.max_model_len,
            config.max_num_seqs,
            configure_cache=False,
        )
        load_native_model_weights(self.model, model_path)
        num_cache_blocks = self._resolve_num_cache_blocks()
        self.model.configure_cache(
            config.max_model_len,
            config.max_num_seqs,
            config.kvcache_block_size,
            num_cache_blocks,
        )
        attention = self.model.layers[0].self_attn
        assert attention.key_cache is not None
        self.prefix_cache = NativePrefixCache(
            num_blocks=attention.key_cache.shape[0],
            blocks_per_sequence=attention.blocks_per_sequence,
            block_size=attention.block_size,
        )
        self.cache_allocation: NativeCacheAllocation | None = None
        self.cache_block_tables: torch.Tensor | None = None
        self.model.eval()
        if config.seed is not None:
            torch.manual_seed(config.seed)
            torch.npu.manual_seed_all(config.seed)
        self.graph_runner = NativeACLGraphRunner(self.model, enabled=not config.enforce_eager)
        self.tokenizer = AutoTokenizer.from_pretrained(config.draft_model)
        # validate_model_pair above permits a target-only vocabulary suffix and
        # verifies that every draft token keeps the same target token ID.
        self.eos_token_ids = _normalize_eos_tokens(target_model_config.eos_token_id)
        self.gamma_profiles: dict[int, int] = {}
        if config.gamma == -1:
            self.gamma_profiles = self._profile_auto_gammas()
        dist.barrier()

    def _resolve_num_cache_blocks(self) -> int:
        if self.config.num_kvcache_blocks > 0:
            return self.config.num_kvcache_blocks
        attention = self.model.layers[0].self_attn
        bytes_per_element = attention.qkv_proj.weight.element_size()
        bytes_per_block = (
            2
            * self.config.kvcache_block_size
            * attention.num_kv_heads
            * attention.head_dim
            * bytes_per_element
            * len(self.model.layers)
        )
        free_memory, total_memory = torch.npu.mem_get_info(self.local_rank)
        used_memory = total_memory - free_memory
        cache_budget = max(0, int(total_memory * self.config.gpu_memory_utilization) - used_memory)
        max_required_blocks = (
            (self.config.max_model_len + self.config.kvcache_block_size - 1)
            // self.config.kvcache_block_size
            * self.config.max_num_seqs
        )
        num_blocks = min(max_required_blocks, cache_budget // bytes_per_block)
        if num_blocks < self.config.max_num_seqs:
            raise MemoryError(
                "PEARL cannot reserve one KV cache page per configured sequence; "
                "lower max_num_seqs or increase available NPU memory."
            )
        return num_blocks

    def _allocate_cache(
        self,
        prompts: list[list[int]],
        *,
        enable_prefix_caching: bool,
    ) -> NativeCacheAllocation:
        allocation = self.prefix_cache.allocate(
            prompts,
            enable_prefix_caching=enable_prefix_caching,
        )
        self.cache_allocation = allocation
        self.cache_block_tables = torch.tensor(
            allocation.block_tables,
            dtype=torch.int32,
            device=self.device,
        )
        return allocation

    def _ensure_cache_capacity(
        self,
        sequence_ids: list[int],
        positions: list[int],
    ) -> None:
        if self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before extending it.")
        updates = self.prefix_cache.ensure_capacity(sequence_ids, positions)
        if not updates:
            return
        update_sequences, update_logical_blocks, update_block_ids = zip(*updates)
        self.cache_block_tables[
            torch.tensor(update_sequences, dtype=torch.long, device=self.device),
            torch.tensor(update_logical_blocks, dtype=torch.long, device=self.device),
        ] = torch.tensor(update_block_ids, dtype=torch.int32, device=self.device)

    def _release_cache(self) -> None:
        self.prefix_cache.release()
        self.cache_allocation = None
        self.cache_block_tables = None

    def _cache_slot_mapping(
        self,
        sequence_ids: list[int],
        positions: list[int],
    ) -> list[int]:
        if self.cache_allocation is None:
            raise RuntimeError("Allocate the native PEARL KV cache before building slot mappings.")
        block_size = self.config.kvcache_block_size
        return [
            self.cache_allocation.block_tables[sequence_id][position // block_size] * block_size + position % block_size
            for sequence_id, position in zip(sequence_ids, positions)
        ]

    @torch.inference_mode()
    def generate(
        self,
        prompt_token_ids: list[int],
        sampling_params: NativeSamplingParams | None = None,
    ) -> dict[str, Any] | None:
        results = self.generate_batch([prompt_token_ids], sampling_params)
        return results[0] if results is not None else None

    @torch.inference_mode()
    def generate_batch(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None = None,
        *,
        max_rounds: int | None = None,
    ) -> list[dict[str, Any]] | None:
        """Generate a static batch with packed draft and target model calls."""
        if not prompt_token_ids or len(prompt_token_ids) > self.config.max_num_seqs:
            raise ValueError("PEARL batch size must be between one and max_num_seqs.")
        if sum(len(prompt) for prompt in prompt_token_ids) > self.config.max_num_batched_tokens:
            raise ValueError("PEARL prompts exceed max_num_batched_tokens.")
        request_params = _normalize_sampling_params(len(prompt_token_ids), sampling_params, self.config.max_tokens)
        if max_rounds is not None and max_rounds <= 0:
            raise ValueError("PEARL max_rounds must be positive when supplied.")
        self.gamma = self._auto_select_gamma(prompt_token_ids) if self.config.gamma == -1 else self.config.gamma
        for prompt, params in zip(prompt_token_ids, request_params):
            if not prompt:
                raise ValueError("PEARL generation requires non-empty prompts.")
            completion_capacity = params.max_tokens if max_rounds is None else 1 + (max_rounds + 1) * self.gamma
            if len(prompt) + completion_capacity + self.gamma > self.config.max_model_len:
                raise ValueError("Prompt plus PEARL completion exceeds max_model_len.")

        initial_tokens = [[int(token_id) for token_id in prompt] for prompt in prompt_token_ids]
        self._allocate_cache(
            initial_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )
        draft_states = [
            PearlPipelineState(
                tokens,
                len(tokens),
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                ignore_eos=params.ignore_eos,
            )
            for tokens, params in zip(initial_tokens, request_params)
        ]
        target_states = [state.clone() for state in draft_states]
        torch.npu.synchronize()
        prefill_started = time.perf_counter()
        target_tokens = self._prefill_and_sample_target_batch(initial_tokens, target_states)
        torch.npu.synchronize()
        prefill_elapsed = time.perf_counter() - prefill_started
        for draft_state, target_state, target_token in zip(draft_states, target_states, target_tokens):
            draft_state.token_ids.append(target_token)
            target_state.token_ids.append(target_token)
            assert draft_state.committed_length is not None and target_state.committed_length is not None
            draft_state.committed_length += 1
            target_state.committed_length += 1

        self._capture_decode_graphs(initial_tokens, target_tokens)
        torch.npu.synchronize()
        started = time.perf_counter()
        round_count = 0
        while True:
            local_states = draft_states if self.is_draft else target_states
            if max_rounds is None:
                active_indices = [
                    index for index, state in enumerate(local_states) if not _finished(state, self.eos_token_ids)
                ]
            elif round_count < max_rounds:
                active_indices = list(range(len(local_states)))
            else:
                active_indices = []
            if not active_indices:
                break
            pre_verify = [local_states[index].pre_verify for index in active_indices]
            verification_windows, next_windows = self._draft_round_batch(draft_states, active_indices)
            target_token_windows, target_logits = self._target_round_outputs_batch(target_states, active_indices)
            verification_sizes = [1 if value else self.gamma for value in pre_verify]
            draft_message = self._exchange_draft_windows(
                verification_windows,
                next_windows,
                verification_sizes,
            )
            verdict = self._verify_target_tokens_batch(
                target_token_windows,
                target_logits,
                draft_message,
                verification_sizes,
                [target_states[index].temperature for index in active_indices],
            )
            accepted, corrections, synchronized_next = self._broadcast_round_result(
                verdict,
                draft_message,
                sum(verification_sizes),
                len(active_indices),
            )
            for batch_index, sequence_index in enumerate(active_indices):
                if self.is_draft:
                    draft_states[sequence_index].apply_draft_verification(
                        gamma=self.gamma,
                        accepted=accepted[batch_index],
                        correction_token_id=corrections[batch_index],
                        next_round_token_ids=next_windows[batch_index],
                    )
                else:
                    target_states[sequence_index].apply_target_verification(
                        gamma=self.gamma,
                        accepted=accepted[batch_index],
                        correction_token_id=corrections[batch_index],
                        next_round_token_ids=synchronized_next[batch_index],
                    )
            round_count += 1

        torch.npu.synchronize()
        decode_elapsed = time.perf_counter() - started
        elapsed = prefill_elapsed + decode_elapsed
        results = []
        for sequence_index, state in enumerate(target_states):
            acceptance_lengths = state.acceptance_lengths
            completion_token_ids = _truncate_completion(
                state.committed_completion_token_ids,
                self.eos_token_ids,
                state.max_tokens if max_rounds is None else len(state.committed_completion_token_ids),
                state.ignore_eos if max_rounds is None else True,
            )
            results.append(
                {
                    "completion_token_ids": completion_token_ids,
                    "accepted_draft_tokens": state.accepted_draft_tokens,
                    "verified_draft_tokens": state.verified_draft_tokens,
                    "acceptance_rate": (
                        state.accepted_draft_tokens / state.verified_draft_tokens
                        if state.verified_draft_tokens
                        else 0.0
                    ),
                    "num_acc_tokens": acceptance_lengths,
                    "mean_accept_tokens": (
                        sum(acceptance_lengths) / len(acceptance_lengths) if acceptance_lengths else 0.0
                    ),
                    "temperature": state.temperature,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": self.cache_allocation.num_cached_tokens[sequence_index],
                    "gamma": self.gamma,
                    "aclgraph_captures": self.graph_runner.capture_count,
                    "aclgraph_replays": self.graph_runner.replay_count,
                    "aclgraph_failed_captures": self.graph_runner.failed_capture_count,
                }
            )
        self._release_cache()
        return results if self.rank == self.topology.target_leader_rank else None

    @torch.inference_mode()
    def generate_target_ar_batch(
        self,
        prompt_token_ids: list[list[int]],
        sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None = None,
    ) -> list[dict[str, Any]] | None:
        """Generate a static batch autoregressively on the target model only."""
        if not prompt_token_ids or len(prompt_token_ids) > self.config.max_num_seqs:
            raise ValueError("PEARL batch size must be between one and max_num_seqs.")
        if sum(len(prompt) for prompt in prompt_token_ids) > self.config.max_num_batched_tokens:
            raise ValueError("PEARL prompts exceed max_num_batched_tokens.")
        request_params = _normalize_sampling_params(len(prompt_token_ids), sampling_params, self.config.max_tokens)
        for prompt, params in zip(prompt_token_ids, request_params):
            if not prompt:
                raise ValueError("Target AR generation requires non-empty prompts.")
            if len(prompt) + params.max_tokens > self.config.max_model_len:
                raise ValueError("Prompt plus target completion exceeds max_model_len.")

        if self.is_draft:
            return None

        initial_tokens = [[int(token_id) for token_id in prompt] for prompt in prompt_token_ids]
        self._allocate_cache(
            initial_tokens,
            enable_prefix_caching=self.config.enable_prefix_caching,
        )
        states = [
            PearlPipelineState(
                tokens,
                len(tokens),
                temperature=params.temperature,
                max_tokens=params.max_tokens,
                ignore_eos=params.ignore_eos,
            )
            for tokens, params in zip(initial_tokens, request_params)
        ]
        torch.npu.synchronize()
        prefill_started = time.perf_counter()
        target_tokens = self._target_ar_prefill(initial_tokens, states)
        torch.npu.synchronize()
        prefill_elapsed = time.perf_counter() - prefill_started
        for state, target_token in zip(states, target_tokens):
            state.token_ids.append(target_token)
            assert state.committed_length is not None
            state.committed_length += 1
        self._capture_target_ar_graph(states)

        torch.npu.synchronize()
        started = time.perf_counter()
        while True:
            active_indices = [index for index, state in enumerate(states) if not _finished(state, self.eos_token_ids)]
            if not active_indices:
                break
            input_token_ids = [states[index].token_ids[-1] for index in active_indices]
            positions = [len(states[index].token_ids) - 1 for index in active_indices]
            next_tokens = self._run_packed_sample(
                input_token_ids,
                active_indices,
                positions,
                [states[index].temperature for index in active_indices],
            )
            if any(states[index].temperature > 0 for index in active_indices):
                dist.broadcast(
                    next_tokens,
                    src=self.topology.target_leader_rank,
                    group=self.groups.target_group,
                )
            for sequence_index, token_id in zip(active_indices, next_tokens.cpu().tolist()):
                states[sequence_index].token_ids.append(int(token_id))
                assert states[sequence_index].committed_length is not None
                states[sequence_index].committed_length += 1

        torch.npu.synchronize()
        decode_elapsed = time.perf_counter() - started
        elapsed = prefill_elapsed + decode_elapsed
        results = []
        for sequence_index, state in enumerate(states):
            results.append(
                {
                    "completion_token_ids": _truncate_completion(
                        state.committed_completion_token_ids,
                        self.eos_token_ids,
                        state.max_tokens,
                        state.ignore_eos,
                    ),
                    "accepted_draft_tokens": 0,
                    "verified_draft_tokens": 0,
                    "acceptance_rate": 0.0,
                    "num_acc_tokens": [],
                    "mean_accept_tokens": 0.0,
                    "temperature": state.temperature,
                    "max_tokens": state.max_tokens,
                    "ignore_eos": state.ignore_eos,
                    "elapsed_seconds": elapsed,
                    "prefill_elapsed_seconds": prefill_elapsed,
                    "decode_elapsed_seconds": decode_elapsed,
                    "cached_prompt_tokens": self.cache_allocation.num_cached_tokens[sequence_index],
                    "gamma": 0,
                    "aclgraph_captures": self.graph_runner.capture_count,
                    "aclgraph_replays": self.graph_runner.replay_count,
                    "aclgraph_failed_captures": self.graph_runner.failed_capture_count,
                }
            )
        self._release_cache()
        return results if self.rank == self.topology.target_leader_rank else None

    def _run_packed_hidden(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
        return self._run_device_packed_hidden(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )

    def _run_device_packed_hidden(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        self._ensure_cache_capacity(sequence_ids, positions)
        slot_mapping = self._cache_slot_mapping(sequence_ids, positions)
        position_tensor, attention_metadata = self.model.make_attention_metadata(
            sequence_ids,
            positions,
            self.cache_block_tables,
            slot_mapping,
            use_fused_infer_attention,
        )
        if use_aclgraph:
            hidden_states = self.graph_runner(input_ids, position_tensor, attention_metadata)
        else:
            hidden_states = self.model(input_ids, position_tensor, attention_metadata)
        if logit_indices is not None:
            hidden_states = hidden_states[logit_indices]
        return hidden_states

    def _run_packed_model(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        hidden_states = self._run_packed_hidden(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )
        return self.model.compute_logits(hidden_states)

    def _run_packed_greedy(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if logit_indices is not None or not use_aclgraph:
            hidden_states = self._run_packed_hidden(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph,
                logit_indices,
                use_fused_infer_attention,
            )
            return self.model.compute_greedy_tokens(hidden_states, self.draft_vocab_size)
        input_ids = torch.tensor(input_token_ids, dtype=torch.long, device=self.device)
        return self._run_device_packed_greedy(
            input_ids,
            sequence_ids,
            positions,
            use_aclgraph=True,
            use_fused_infer_attention=use_fused_infer_attention,
        )

    def _run_packed_sample(
        self,
        input_token_ids: list[int],
        sequence_ids: list[int],
        positions: list[int],
        temperatures: list[float],
        use_aclgraph: bool = True,
        logit_indices: list[int] | None = None,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if all(temperature == 0 for temperature in temperatures):
            return self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph,
                logit_indices,
                use_fused_infer_attention,
            )
        logits = self._run_packed_model(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph,
            logit_indices,
            use_fused_infer_attention,
        )[:, : self.draft_vocab_size]
        return _sample_logits(logits, temperatures)

    def _run_device_packed_greedy(
        self,
        input_ids: torch.Tensor,
        sequence_ids: list[int],
        positions: list[int],
        use_aclgraph: bool = True,
        use_fused_infer_attention: bool = False,
    ) -> torch.Tensor:
        if self.cache_allocation is None or self.cache_block_tables is None:
            raise RuntimeError("Allocate the native PEARL KV cache before running the model.")
        self._ensure_cache_capacity(sequence_ids, positions)
        slot_mapping = self._cache_slot_mapping(sequence_ids, positions)
        position_tensor, attention_metadata = self.model.make_attention_metadata(
            sequence_ids,
            positions,
            self.cache_block_tables,
            slot_mapping,
            use_fused_infer_attention,
        )
        if use_aclgraph:
            return self.graph_runner.run_greedy(
                input_ids,
                position_tensor,
                attention_metadata,
                self.draft_vocab_size,
            )
        hidden_states = self.model(input_ids, position_tensor, attention_metadata)
        return self.model.compute_greedy_tokens(hidden_states, self.draft_vocab_size)

    def _target_ar_prefill(
        self,
        prompts: list[list[int]],
        states: list[PearlPipelineState],
    ) -> list[int]:
        input_token_ids: list[int] = []
        sequence_ids: list[int] = []
        positions: list[int] = []
        last_token_indices: list[int] = []
        for sequence_id, prompt in enumerate(prompts):
            assert self.cache_allocation is not None
            cached_tokens = self.cache_allocation.num_cached_tokens[sequence_id]
            input_token_ids.extend(prompt[cached_tokens:])
            sequence_ids.extend([sequence_id] * (len(prompt) - cached_tokens))
            positions.extend(range(cached_tokens, len(prompt)))
            last_token_indices.append(len(input_token_ids) - 1)
        token_ids = self._run_packed_sample(
            input_token_ids,
            sequence_ids,
            positions,
            [state.temperature for state in states],
            use_aclgraph=False,
            logit_indices=last_token_indices,
        )
        if any(state.temperature > 0 for state in states):
            dist.broadcast(
                token_ids,
                src=self.topology.target_leader_rank,
                group=self.groups.target_group,
            )
        return [int(token_id) for token_id in token_ids.cpu().tolist()]

    def _capture_target_ar_graph(self, states: list[PearlPipelineState]) -> None:
        if not self.config.enforce_eager:
            sequence_ids = list(range(len(states)))
            input_ids = torch.tensor(
                [state.token_ids[-1] for state in states],
                dtype=torch.long,
                device=self.device,
            )
            self._run_device_packed_greedy(
                input_ids,
                sequence_ids,
                [len(state.token_ids) - 1 for state in states],
            )
        dist.barrier(group=self.groups.target_group)

    def _prefill_and_sample_target_batch(
        self,
        prompts: list[list[int]],
        states: list[PearlPipelineState],
    ) -> list[int]:
        input_token_ids: list[int] = []
        sequence_ids: list[int] = []
        positions: list[int] = []
        last_token_indices: list[int] = []
        for sequence_id, prompt in enumerate(prompts):
            assert self.cache_allocation is not None
            cached_tokens = self.cache_allocation.num_cached_tokens[sequence_id]
            input_token_ids.extend(prompt[cached_tokens:])
            sequence_ids.extend([sequence_id] * (len(prompt) - cached_tokens))
            positions.extend(range(cached_tokens, len(prompt)))
            last_token_indices.append(len(input_token_ids) - 1)
        if self.is_draft:
            # The target sample is the common committed frontier, but the draft
            # prefill still runs so its persistent KV cache is populated.
            self._run_packed_hidden(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph=False,
                logit_indices=last_token_indices,
            )
            token_ids = torch.zeros(len(prompts), dtype=torch.long, device=self.device)
        else:
            token_ids = self._run_packed_sample(
                input_token_ids,
                sequence_ids,
                positions,
                [state.temperature for state in states],
                use_aclgraph=False,
                logit_indices=last_token_indices,
            )
        dist.broadcast(token_ids, src=self.topology.target_leader_rank)
        return [int(token_id) for token_id in token_ids.cpu().tolist()]

    def _auto_select_gamma(self, prompts: list[list[int]]) -> int:
        if not self.gamma_profiles:
            raise RuntimeError("PEARL auto-gamma profiles were not initialized.")
        batch_size = len(prompts)
        bucket = next(
            (size for size in self.gamma_profiles if size >= batch_size),
            max(self.gamma_profiles),
        )
        return self.gamma_profiles[bucket]

    def _profile_auto_gammas(self) -> dict[int, int]:
        profile_length = min(
            AUTO_GAMMA_PROFILE_SEQUENCE_LENGTH,
            self.config.max_model_len - 1,
        )
        if profile_length <= 0:
            raise ValueError("PEARL max_model_len is too small for automatic gamma profiling.")
        batch_sizes = [
            batch_size
            for batch_size in AUTO_GAMMA_BATCH_SIZES
            if batch_size <= self.config.max_num_seqs
            and batch_size * profile_length <= self.config.max_num_batched_tokens
        ]
        if not batch_sizes:
            raise ValueError("PEARL batch limits cannot fit the automatic gamma profile.")

        profiles: dict[int, int] = {}
        for batch_size in batch_sizes:
            prompts = [[0] * profile_length for _ in range(batch_size)]
            self._allocate_cache(prompts, enable_prefix_caching=False)
            input_token_ids = [token for prompt in prompts for token in prompt]
            sequence_ids = [index for index in range(batch_size) for _ in range(profile_length)]
            positions = list(range(profile_length)) * batch_size
            last_token_indices = [(index + 1) * profile_length - 1 for index in range(batch_size)]
            decode_tokens = self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph=False,
                logit_indices=last_token_indices,
            )
            decode_positions = [profile_length] * batch_size
            decode_sequence_ids = list(range(batch_size))
            for _ in range(AUTO_GAMMA_WARMUP_STEPS):
                self._run_device_packed_greedy(
                    decode_tokens,
                    decode_sequence_ids,
                    decode_positions,
                )
            torch.npu.synchronize()
            dist.barrier()
            started = time.perf_counter()
            for _ in range(AUTO_GAMMA_PROFILE_STEPS):
                self._run_device_packed_greedy(
                    decode_tokens,
                    decode_sequence_ids,
                    decode_positions,
                )
            torch.npu.synchronize()
            elapsed = time.perf_counter() - started
            local_speed = AUTO_GAMMA_PROFILE_STEPS / elapsed
            speeds = torch.zeros(2, dtype=torch.float32, device=self.device)
            if self.rank == self.topology.draft_leader_rank:
                speeds[0] = local_speed
            if self.rank == self.topology.target_leader_rank:
                speeds[1] = local_speed
            dist.all_reduce(speeds)
            draft_speed, target_speed = (float(value) for value in speeds.cpu().tolist())
            profiles[batch_size] = _gamma_from_decode_speeds(draft_speed, target_speed)
            self._release_cache()
        return profiles

    def _capture_decode_graphs(self, prompts: list[list[int]], target_tokens: list[int]) -> None:
        if self.config.enforce_eager or self.gamma > 16:
            return
        sequence_ids = list(range(len(prompts)))
        first_decode_positions = [len(prompt) for prompt in prompts]
        input_ids = torch.tensor(target_tokens, dtype=torch.long, device=self.device)
        self._run_device_packed_greedy(
            input_ids,
            sequence_ids,
            first_decode_positions,
            use_fused_infer_attention=True,
        )
        if not self.is_draft and self.gamma > 1:
            packed_tokens = [token_id for token_id in target_tokens for _ in range(self.gamma)]
            packed_sequence_ids = [sequence_id for sequence_id in sequence_ids for _ in range(self.gamma)]
            packed_positions = [
                position for prompt in prompts for position in range(len(prompt) + 1, len(prompt) + self.gamma + 1)
            ]
            packed_input_ids = torch.tensor(packed_tokens, dtype=torch.long, device=self.device)
            self._run_device_packed_greedy(
                packed_input_ids,
                packed_sequence_ids,
                packed_positions,
                use_fused_infer_attention=True,
            )
        dist.barrier()

    def _draft_round_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
    ) -> tuple[list[list[int]], list[list[int]]]:
        if not self.is_draft:
            return [], []
        was_pre_verify = [states[index].pre_verify for index in active_indices]
        input_ids = torch.tensor(
            [states[index].token_ids[-1] for index in active_indices],
            dtype=torch.long,
            device=self.device,
        )
        first_positions = [len(states[index].token_ids) - 1 for index in active_indices]
        draft_steps: list[torch.Tensor] = []
        for step in range(self.gamma):
            positions = [position + step for position in first_positions]
            input_ids = self._run_device_packed_greedy(
                input_ids,
                active_indices,
                positions,
                use_aclgraph=not self.config.enforce_eager and self.gamma <= 16,
                use_fused_infer_attention=self.gamma <= 16,
            )
            # ACLGraph replays reuse one persistent output buffer. Preserve
            # each proposal before the next replay overwrites that buffer.
            draft_steps.append(input_ids.clone())
        draft_windows = torch.stack(draft_steps, dim=1).cpu().tolist()
        for sequence_index, token_ids in zip(active_indices, draft_windows):
            states[sequence_index].token_ids.extend(int(token_id) for token_id in token_ids)
        next_windows = [states[index].token_ids[-self.gamma :] for index in active_indices]
        verification_windows = []
        for sequence_index, is_pre_verify, next_window in zip(active_indices, was_pre_verify, next_windows):
            state = states[sequence_index]
            verification_windows.append(
                [next_window[0]] if is_pre_verify else state.token_ids[-2 * self.gamma + 1 : -self.gamma + 1]
            )
        return verification_windows, next_windows

    def _target_round_outputs_batch(
        self,
        states: list[PearlPipelineState],
        active_indices: list[int],
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if self.is_draft:
            return None, None
        input_token_ids: list[int] = []
        sequence_ids: list[int] = []
        positions: list[int] = []
        for sequence_index in active_indices:
            state = states[sequence_index]
            num_tokens = 1 if state.pre_verify else self.gamma
            start_position = len(state.token_ids) - num_tokens
            input_token_ids.extend(state.token_ids[-num_tokens:])
            sequence_ids.extend([sequence_index] * num_tokens)
            positions.extend(range(start_position, start_position + num_tokens))
        temperatures = [
            states[index].temperature
            for index in active_indices
            for _ in range(1 if states[index].pre_verify else self.gamma)
        ]
        # Speculative execution uses FIA graphs; the decode-only paged-attention
        # graph is reserved for target-only autoregressive generation.
        use_aclgraph = not self.config.enforce_eager and self.gamma <= 16
        if all(temperature == 0 for temperature in temperatures):
            target_tokens = self._run_packed_greedy(
                input_token_ids,
                sequence_ids,
                positions,
                use_aclgraph=use_aclgraph,
                use_fused_infer_attention=self.gamma <= 16,
            )
            return target_tokens, None
        logits = self._run_packed_model(
            input_token_ids,
            sequence_ids,
            positions,
            use_aclgraph=use_aclgraph,
            use_fused_infer_attention=self.gamma <= 16,
        )
        return None, logits[:, : self.draft_vocab_size]

    def _exchange_draft_windows(
        self,
        verification_windows: list[list[int]],
        next_windows: list[list[int]],
        verification_sizes: list[int],
    ) -> torch.Tensor | None:
        verification_size = sum(verification_sizes)
        continuation_size = self.gamma * len(verification_sizes)
        if self.groups.is_verification_worker:
            if self.rank == self.topology.draft_leader_rank:
                if [len(window) for window in verification_windows] != verification_sizes:
                    raise RuntimeError("PEARL draft window has an unexpected shape.")
                if any(len(window) != self.gamma for window in next_windows):
                    raise RuntimeError("PEARL continuation window has an unexpected shape.")
                message = torch.tensor(
                    [token for window in verification_windows for token in window]
                    + [token for window in next_windows for token in window],
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                message = torch.empty(verification_size + continuation_size, dtype=torch.long, device=self.device)
            dist.broadcast(message, src=self.topology.draft_leader_rank, group=self.groups.verification_group)
            return message
        return None

    def _verify_target_tokens_batch(
        self,
        target_tokens: torch.Tensor | None,
        target_logits: torch.Tensor | None,
        draft_message: torch.Tensor | None,
        verification_sizes: list[int],
        temperatures: list[float],
    ) -> torch.Tensor | None:
        if self.rank != self.topology.target_leader_rank:
            return None
        if draft_message is None:
            raise RuntimeError("The PEARL target leader did not receive draft verification tokens.")
        verification_size = sum(verification_sizes)
        packed_temperatures = [
            temperature for temperature, size in zip(temperatures, verification_sizes) for _ in range(size)
        ]
        if all(temperature == 0 for temperature in packed_temperatures):
            if target_tokens is None or target_tokens.shape != (verification_size,):
                raise RuntimeError("PEARL target and draft verification windows differ in length.")
            return _build_greedy_verdict(
                target_tokens,
                draft_message[:verification_size],
                verification_sizes,
                self.gamma,
            )
        if target_logits is None or target_logits.shape[0] != verification_size:
            raise RuntimeError("PEARL target logits do not match the packed verification window.")
        return _build_stochastic_verdict(
            target_logits,
            draft_message[:verification_size],
            verification_sizes,
            self.gamma,
            packed_temperatures,
        )

    def _broadcast_round_result(
        self,
        verdict: torch.Tensor | None,
        draft_message: torch.Tensor | None,
        verification_size: int,
        batch_size: int,
    ) -> tuple[list[int], list[int | None], list[list[int]]]:
        continuation_size = batch_size * self.gamma
        if self.rank == self.topology.target_leader_rank:
            if verdict is None or draft_message is None:
                raise RuntimeError("The PEARL target leader did not produce a round result.")
            result = torch.cat((verdict.flatten(), draft_message[verification_size:]))
        else:
            result = torch.empty(batch_size * 2 + continuation_size, dtype=torch.long, device=self.device)
        dist.broadcast(result, src=self.topology.target_leader_rank)
        values = [int(value) for value in result.cpu().tolist()]
        verdict_values = values[: batch_size * 2]
        continuation_values = values[batch_size * 2 :]
        accepted = verdict_values[::2]
        corrections = [None if value == -1 else value for value in verdict_values[1::2]]
        next_windows = [
            continuation_values[index : index + self.gamma] for index in range(0, continuation_size, self.gamma)
        ]
        return accepted, corrections, next_windows


def _normalize_eos_tokens(eos_token_id: int | list[int] | None) -> frozenset[int]:
    if eos_token_id is None:
        return frozenset()
    if isinstance(eos_token_id, int):
        return frozenset((eos_token_id,))
    return frozenset(int(token_id) for token_id in eos_token_id)


def _normalize_sampling_params(
    batch_size: int,
    sampling_params: NativeSamplingParams | Sequence[NativeSamplingParams] | None,
    default_max_tokens: int,
) -> list[NativeSamplingParams]:
    if sampling_params is None:
        params = [NativeSamplingParams(temperature=0.0, max_tokens=default_max_tokens)] * batch_size
    elif isinstance(sampling_params, NativeSamplingParams):
        params = [sampling_params] * batch_size
    else:
        params = list(sampling_params)
        if len(params) != batch_size:
            raise ValueError("PEARL requires one SamplingParams value per prompt.")
        if not all(isinstance(value, NativeSamplingParams) for value in params):
            raise TypeError("PEARL sampling_params must contain SamplingParams values.")
    temperatures = [value.temperature for value in params]
    if not (all(value == 0 for value in temperatures) or all(value > 0 for value in temperatures)):
        raise ValueError("A PEARL batch requires temperatures that are either all zero or all non-zero.")
    return params


def _sample_logits(
    logits: torch.Tensor,
    temperatures: Sequence[float],
    *,
    exponential_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Apply nano-PEARL's greedy or exponential-race target sampler."""
    if logits.ndim != 2 or logits.shape[0] != len(temperatures):
        raise ValueError("PEARL logits and temperatures must have the same batch dimension.")
    if all(temperature == 0 for temperature in temperatures):
        return logits.argmax(dim=-1)
    if not all(temperature > 0 for temperature in temperatures):
        raise ValueError("A PEARL sample requires temperatures that are either all zero or all non-zero.")
    temperature_tensor = torch.tensor(temperatures, dtype=torch.float32, device=logits.device).unsqueeze(1)
    probabilities = torch.softmax(logits.float() / temperature_tensor, dim=-1)
    if exponential_noise is None:
        exponential_noise = torch.empty_like(probabilities).exponential_(1)
    elif exponential_noise.shape != probabilities.shape:
        raise ValueError("PEARL exponential sampling noise must match the logits shape.")
    return probabilities.div(exponential_noise.clamp_min(1e-10)).argmax(dim=-1)


def _build_greedy_verdict(
    target_tokens: torch.Tensor,
    draft_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
) -> torch.Tensor:
    """Find each sequence's accepted greedy prefix without a host sync."""
    if gamma <= 0 or any(size not in (1, gamma) for size in verification_sizes):
        raise ValueError("PEARL verification windows must have length one or gamma.")
    verification_size = sum(verification_sizes)
    if target_tokens.shape != (verification_size,) or draft_tokens.shape != (verification_size,):
        raise ValueError("PEARL target and draft token tensors must match the packed verification size.")

    matches = target_tokens == draft_tokens
    return _build_verdict_from_acceptance(matches, target_tokens, verification_sizes, gamma)


def _build_stochastic_verdict(
    target_logits: torch.Tensor,
    draft_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
    temperatures: Sequence[float],
    *,
    random_values: torch.Tensor | None = None,
    exponential_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Verify greedy draft tokens against a sampled target distribution."""
    verification_size = sum(verification_sizes)
    if target_logits.ndim != 2 or target_logits.shape[0] != verification_size:
        raise ValueError("PEARL target logits must match the packed verification size.")
    if draft_tokens.shape != (verification_size,) or len(temperatures) != verification_size:
        raise ValueError("PEARL draft tokens and temperatures must match the packed verification size.")
    if not all(temperature > 0 for temperature in temperatures):
        raise ValueError("Stochastic PEARL verification requires positive target temperatures.")
    if draft_tokens.device.type == "cpu" and draft_tokens.numel() and int(draft_tokens.max()) >= target_logits.shape[1]:
        raise ValueError("A PEARL draft token lies outside the target verification vocabulary.")

    temperature_tensor = torch.tensor(temperatures, dtype=torch.float32, device=target_logits.device).unsqueeze(1)
    probabilities = torch.softmax(target_logits.float() / temperature_tensor, dim=-1)
    candidate_probabilities = probabilities.gather(1, draft_tokens.unsqueeze(1)).squeeze(1)
    if random_values is None:
        random_values = torch.rand_like(candidate_probabilities)
    elif random_values.shape != candidate_probabilities.shape:
        raise ValueError("PEARL verification random values must match the packed window.")
    accepted = random_values <= candidate_probabilities

    correction_logits = target_logits.clone()
    correction_logits.scatter_(1, draft_tokens.unsqueeze(1), -float("inf"))
    correction_tokens = _sample_logits(
        correction_logits,
        temperatures,
        exponential_noise=exponential_noise,
    )
    return _build_verdict_from_acceptance(
        accepted,
        correction_tokens,
        verification_sizes,
        gamma,
    )


def _build_verdict_from_acceptance(
    accepted_tokens: torch.Tensor,
    correction_tokens: torch.Tensor,
    verification_sizes: list[int],
    gamma: int,
) -> torch.Tensor:
    if gamma <= 0 or any(size not in (1, gamma) for size in verification_sizes):
        raise ValueError("PEARL verification windows must have length one or gamma.")
    verification_size = sum(verification_sizes)
    if accepted_tokens.shape != (verification_size,) or correction_tokens.shape != (verification_size,):
        raise ValueError("PEARL verification tensors must match the packed verification size.")
    match_matrix = torch.ones(
        (len(verification_sizes), gamma),
        dtype=torch.bool,
        device=accepted_tokens.device,
    )
    token_matrix = torch.zeros(
        (len(verification_sizes), gamma),
        dtype=torch.long,
        device=accepted_tokens.device,
    )
    offset = 0
    for row, size in enumerate(verification_sizes):
        match_matrix[row, :size] = accepted_tokens[offset : offset + size]
        token_matrix[row, :size] = correction_tokens[offset : offset + size]
        offset += size
    expected = torch.tensor(verification_sizes, dtype=torch.long, device=accepted_tokens.device)
    all_match = match_matrix.all(dim=1)
    first_mismatch = (~match_matrix).to(dtype=torch.int32).argmax(dim=1)
    accepted = torch.where(all_match, expected, first_mismatch)
    correction = token_matrix.gather(1, accepted.clamp(max=gamma - 1).unsqueeze(1)).squeeze(1)
    correction = torch.where(all_match, torch.full_like(correction, -1), correction)
    return torch.stack((accepted, correction), dim=1)


def _gamma_from_decode_speeds(draft_tokens_per_second: float, target_tokens_per_second: float) -> int:
    if draft_tokens_per_second <= 0 or target_tokens_per_second <= 0:
        raise ValueError("PEARL auto-gamma profiling produced a non-positive decode speed.")
    return max(1, round(draft_tokens_per_second / target_tokens_per_second))


def _finished(state: PearlPipelineState, eos_token_ids: frozenset[int]) -> bool:
    completion_token_ids = state.committed_completion_token_ids
    return len(completion_token_ids) >= state.max_tokens or (
        not state.ignore_eos and any(token_id in eos_token_ids for token_id in completion_token_ids)
    )


def _truncate_completion(
    completion_token_ids: list[int],
    eos_token_ids: frozenset[int],
    max_tokens: int,
    ignore_eos: bool = False,
) -> list[int]:
    truncated = completion_token_ids[:max_tokens]
    if ignore_eos:
        return truncated
    return truncated[: next((index + 1 for index, token_id in enumerate(truncated) if token_id in eos_token_ids), None)]


def _load_gsm8k_questions(dataset_path: str, max_samples: int) -> list[str]:
    import pyarrow.parquet as pq

    rows = pq.read_table(dataset_path, columns=["question"]).to_pylist()
    return [str(row["question"]) for row in rows[:max_samples]]


def _prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    formatted_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )
    return list(tokenizer.encode(formatted_prompt, add_special_tokens=False))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft-model", required=True)
    parser.add_argument("--target-model", required=True)
    parser.add_argument("--draft-tp-size", type=int, default=1)
    parser.add_argument("--target-tp-size", type=int, default=2)
    parser.add_argument("--gamma", type=int, default=4)
    parser.add_argument("--mode", choices=("pearl", "target-ar"), default="pearl")
    parser.add_argument("--max-model-len", type=int, default=1024)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--ignore-eos", action="store_true")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--num-kvcache-blocks", type=int, default=-1)
    parser.add_argument("--disable-prefix-caching", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--prompt")
    parser.add_argument("--repeat-prompt", type=int, default=1)
    parser.add_argument("--gsm8k", help="Path to a GSM8K parquet file.")
    parser.add_argument("--max-samples", type=int, default=1)
    parser.add_argument("--summary-only", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    if bool(args.prompt) == bool(args.gsm8k):
        raise ValueError("Pass exactly one of --prompt or --gsm8k.")
    if args.repeat_prompt <= 0 or (args.gsm8k and args.repeat_prompt != 1):
        raise ValueError("repeat-prompt must be positive and is only valid with --prompt.")
    config = NativePearlConfig(
        draft_model=args.draft_model,
        target_model=args.target_model,
        draft_tp_size=args.draft_tp_size,
        target_tp_size=args.target_tp_size,
        gamma=args.gamma,
        max_model_len=args.max_model_len,
        max_tokens=args.max_tokens,
        max_num_seqs=args.batch_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        num_kvcache_blocks=args.num_kvcache_blocks,
        enable_prefix_caching=not args.disable_prefix_caching,
        enforce_eager=args.enforce_eager,
        seed=args.seed,
    )
    engine = NativePearlEngine(config)
    sampling_params = NativeSamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        ignore_eos=args.ignore_eos,
    )
    prompts = [args.prompt] * args.repeat_prompt if args.prompt else _load_gsm8k_questions(args.gsm8k, args.max_samples)
    results: list[dict[str, Any]] = []
    total_elapsed = 0.0
    for start in range(0, len(prompts), args.batch_size):
        prompt_batch = prompts[start : start + args.batch_size]
        prompt_token_ids = [_prompt_token_ids(engine.tokenizer, prompt) for prompt in prompt_batch]
        if args.mode == "pearl":
            batch_results = engine.generate_batch(prompt_token_ids, sampling_params)
        else:
            batch_results = engine.generate_target_ar_batch(prompt_token_ids, sampling_params)
        if batch_results is not None:
            total_elapsed += batch_results[0]["elapsed_seconds"]
            for prompt, result in zip(prompt_batch, batch_results):
                result["prompt"] = prompt
                result["text"] = engine.tokenizer.decode(result["completion_token_ids"], skip_special_tokens=True)
                results.append(result)
    if engine.rank == engine.topology.target_leader_rank:
        total_verified = sum(result["verified_draft_tokens"] for result in results)
        total_accepted = sum(result["accepted_draft_tokens"] for result in results)
        generated_token_count = sum(len(result["completion_token_ids"]) for result in results)
        aggregate_mat = sum(result["mean_accept_tokens"] for result in results) / len(results) if results else 0.0
        print(
            json.dumps(
                {
                    "mode": args.mode,
                    "samples": [] if args.summary_only else results,
                    "num_samples": len(results),
                    "generated_token_count": generated_token_count,
                    "aggregate_acceptance_rate": total_accepted / total_verified if total_verified else 0.0,
                    "aggregate_mat": aggregate_mat,
                    "decode_throughput_tokens_per_second": (
                        generated_token_count / total_elapsed if total_elapsed else 0.0
                    ),
                },
                ensure_ascii=True,
            )
        )
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
