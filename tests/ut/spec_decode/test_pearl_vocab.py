# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm_ascend.spec_decode.draft_proposer import AscendDraftModelProposer
from vllm_ascend.spec_decode.pearl import PearlProposalBatch, PearlTargetVerifier, PearlVocabProjection
from vllm_ascend.spec_decode.pearl.qwen_pair import (
    build_speculative_config,
    validate_tensor_parallel_size,
)
from vllm_ascend.worker.model_runner_v1 import NPUModelRunner


class _Tokenizer:
    def __init__(self, tokens: list[str]):
        self.tokens = tokens

    def convert_ids_to_tokens(self, token_id: int) -> str:
        return self.tokens[token_id]


def test_projects_target_logits_to_draft_prefix_contiguously():
    projection = PearlVocabProjection(draft_vocab_size=3, target_vocab_size=5)
    target_logits = torch.arange(10, dtype=torch.float32).reshape(2, 5)

    projected = projection.project_target_logits(target_logits)

    assert projected.is_contiguous()
    assert torch.equal(projected, target_logits[:, :3])


def test_equal_vocab_projection_does_not_copy_logits():
    projection = PearlVocabProjection(draft_vocab_size=3, target_vocab_size=3)
    target_logits = torch.zeros((2, 3))

    assert projection.project_target_logits(target_logits) is target_logits


def test_projection_rejects_unexpected_target_vocab_size():
    projection = PearlVocabProjection(draft_vocab_size=3, target_vocab_size=5)

    with pytest.raises(ValueError, match="unexpected vocabulary size"):
        projection.project_target_logits(torch.zeros((2, 4)))


def test_draft_proposer_uses_ascend_lm_head_hook(monkeypatch):
    """Draft-model graph setup must not be bypassed by its upstream override."""
    invoked: list[object] = []

    def record_hook(self, target_language_model) -> None:
        invoked.extend((self, target_language_model))

    monkeypatch.setattr(
        "vllm_ascend.spec_decode.draft_proposer.AscendSpecDecodeBaseProposer._maybe_share_lm_head",
        record_hook,
    )
    proposer = object.__new__(AscendDraftModelProposer)
    target_language_model = object()

    proposer._maybe_share_lm_head(target_language_model)

    assert invoked == [proposer, target_language_model]


def test_projection_validates_candidate_range():
    projection = PearlVocabProjection(draft_vocab_size=3, target_vocab_size=5)
    projection.validate_draft_token_ids(torch.tensor([0, 2], dtype=torch.int64))

    with pytest.raises(ValueError, match="projected vocabulary range"):
        projection.validate_draft_token_ids(torch.tensor([3], dtype=torch.int64))


def test_projection_requires_identical_token_id_prefix():
    draft = _Tokenizer(["a", "b", "c"])
    target = _Tokenizer(["a", "b", "c", "d"])

    projection = PearlVocabProjection.from_tokenizers(
        draft_tokenizer=draft,
        target_tokenizer=target,
        draft_vocab_size=3,
        target_vocab_size=4,
    )

    assert projection.requires_projection


def test_projection_rejects_nonidentical_token_id_prefix():
    with pytest.raises(ValueError, match="found mismatches"):
        PearlVocabProjection.from_tokenizers(
            draft_tokenizer=_Tokenizer(["a", "b", "c"]),
            target_tokenizer=_Tokenizer(["a", "x", "c", "d"]),
            draft_vocab_size=3,
            target_vocab_size=4,
        )


def test_runner_projects_logits_before_rejection_sampling():
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.pearl_vocab_projection = PearlVocabProjection(
        draft_vocab_size=3,
        target_vocab_size=5,
    )

    projected = runner._project_pearl_target_logits(torch.zeros((2, 5)))

    assert projected is not None
    assert projected.shape == (2, 3)


def test_qwen_pair_uses_greedy_same_tp_speculative_configuration():
    config = build_speculative_config("/data/shared-models/Qwen2.5-0.5B-Instruct", 4, 2)

    assert config == {
        "method": "draft_model",
        "model": "/data/shared-models/Qwen2.5-0.5B-Instruct",
        "num_speculative_tokens": 4,
        "draft_tensor_parallel_size": 2,
        "use_heterogeneous_vocab": True,
        "draft_sample_method": "greedy",
    }


def test_qwen_pair_rejects_nondivisible_shared_tensor_parallel_size(monkeypatch):
    class _Config:
        def __init__(self, num_attention_heads: int):
            self.num_attention_heads = num_attention_heads

    monkeypatch.setattr(
        "vllm_ascend.spec_decode.pearl.qwen_pair.AutoConfig.from_pretrained",
        lambda model_path: _Config(14 if model_path == "draft" else 40),
    )

    with pytest.raises(ValueError, match="draft model has 14 attention heads"):
        validate_tensor_parallel_size("draft", "target", 4)


def test_target_verifier_projects_before_greedy_acceptance():
    verifier = PearlTargetVerifier.with_eos_tokens(
        PearlVocabProjection(draft_vocab_size=3, target_vocab_size=5),
        eos_token_ids=2,
    )
    proposals = PearlProposalBatch(request_slots=(7, 8), candidate_token_ids=((1, 2), (0,)))
    # The target-only suffix wins every row before the PEARL projection.
    target_logits = torch.tensor(
        [
            [0.0, 5.0, 1.0, 2.0, 100.0],
            [0.0, 1.0, 5.0, 2.0, 100.0],
            [5.0, 1.0, 0.0, 2.0, 100.0],
        ]
    )

    verdict = verifier.verify(proposals, target_logits)

    assert verdict.accepted_prefix_lengths == (2, 1)
    assert verdict.correction_token_ids == (None, None)
    assert verdict.finished == (True, False)


def test_target_verifier_returns_target_correction_after_first_rejection():
    verifier = PearlTargetVerifier.with_eos_tokens(
        PearlVocabProjection(draft_vocab_size=3, target_vocab_size=3),
        eos_token_ids=None,
    )
    proposals = PearlProposalBatch(request_slots=(7,), candidate_token_ids=((1, 2),))
    target_logits = torch.tensor([[0.0, 5.0, 1.0], [5.0, 1.0, 0.0]])

    verdict = verifier.verify(proposals, target_logits)

    assert verdict.accepted_prefix_lengths == (1,)
    assert verdict.correction_token_ids == (0,)
