"""Unit tests for AscendNgramProposerNPU.

Tests that the Ascend NPU ngram proposer:
1. Has a propose() signature matching the parent NgramProposerGPU.
2. Delegates to the parent's PyTorch kernel implementation correctly.
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import torch

from vllm_ascend.spec_decode.ngram_proposer_npu import AscendNgramProposerNPU


def test_propose_signature_matches_parent():
    """Verify that propose() has the same signature as the parent class."""
    from vllm.v1.spec_decode.ngram_proposer_gpu import NgramProposerGPU

    child_sig = inspect.signature(AscendNgramProposerNPU.propose)
    parent_sig = inspect.signature(NgramProposerGPU.propose)

    child_params = list(child_sig.parameters.keys())
    parent_params = list(parent_sig.parameters.keys())

    assert child_params == parent_params, (
        f"Signature mismatch:\n"
        f"  AscendNgramProposerNPU.propose params: {child_params}\n"
        f"  NgramProposerGPU.propose params:       {parent_params}\n"
        f"\n"
        f"The NPU proposer must accept the same parameters as the parent.\n"
        f"Missing: {set(parent_params) - set(child_params)}"
    )


def test_propose_delegates_to_parent():
    """Verify that propose() delegates to the parent's implementation."""
    mock_parent_propose = MagicMock(return_value=(
        torch.tensor([[1, 2, 3]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    ))

    proposer = AscendNgramProposerNPU.__new__(AscendNgramProposerNPU)
    # Manually set required attributes (normally set by __init__)
    proposer.k = 3
    proposer.min_n = 2
    proposer.max_n = 5
    proposer.max_model_len = 32
    proposer.max_num_seqs = 4
    proposer.device = "cpu"
    proposer.kernel = MagicMock()
    proposer.vllm_config = MagicMock()
    proposer.runner = SimpleNamespace()

    with patch.object(type(proposer).__bases__[0], "propose", mock_parent_propose):
        b = 1
        num_spec_tokens = 3
        num_tokens_no_spec = torch.tensor([5], dtype=torch.int32)
        token_ids_gpu = torch.zeros((b, 32), dtype=torch.int32)
        valid_sampled_token_ids_gpu = torch.full(
            (b, num_spec_tokens + 1), -1, dtype=torch.int32
        )
        valid_sampled_tokens_count = torch.tensor([0], dtype=torch.int32)

        result = proposer.propose(
            num_spec_tokens,
            num_tokens_no_spec,
            token_ids_gpu,
            valid_sampled_token_ids_gpu,
            valid_sampled_tokens_count,
        )

    mock_parent_propose.assert_called_once_with(
        num_spec_tokens,
        num_tokens_no_spec,
        token_ids_gpu,
        valid_sampled_token_ids_gpu,
        valid_sampled_tokens_count,
    )
    assert result is not None, "propose() must return a tuple (draft_tokens, num_valid)"
