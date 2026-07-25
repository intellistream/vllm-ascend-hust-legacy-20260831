"""Unit tests for AscendNgramProposerNPU.

Tests that the Ascend NPU ngram proposer:
1. Has a propose() signature matching the parent NgramProposerGPU.
2. Implements scatter-and-kernel logic correctly (no super() dependency).
"""

import inspect
from types import SimpleNamespace
from unittest.mock import MagicMock

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


def test_propose_uses_kernel_directly():
    """Verify that propose() calls self.kernel() with expected arguments
    instead of delegating to super().propose().

    NOTE: This test uses __new__ + manual attribute setup instead of the real
    __init__ because the real __init__ creates torch tensors on the NPU device
    (via _dummy_run). Since there is no NPU device in CI unit-test contexts, we
    bypass __init__ and set only the attributes the propose() method needs.
    If __init__ grows additional mandatory attributes, update this setup.
    """
    proposer = AscendNgramProposerNPU.__new__(AscendNgramProposerNPU)
    proposer.k = 3
    proposer.min_n = 2
    proposer.max_n = 5
    proposer.max_model_len = 32
    proposer.max_num_seqs = 4
    proposer.device = "cpu"
    proposer.vllm_config = MagicMock()
    proposer.runner = SimpleNamespace()

    # Replace the kernel with a mock
    mock_kernel = MagicMock(return_value=(
        torch.tensor([[21, 22, 23]], dtype=torch.int32),
        torch.tensor([3], dtype=torch.int32),
    ))
    proposer.kernel = mock_kernel

    b = 1
    num_spec_tokens = 3
    num_tokens_no_spec = torch.tensor([5], dtype=torch.int32, device="cpu")
    token_ids_gpu = torch.arange(32, dtype=torch.int32, device="cpu").unsqueeze(0)
    valid_sampled_token_ids_gpu = torch.tensor([[10, 11, 12, -1]], dtype=torch.int32, device="cpu")
    valid_sampled_tokens_count = torch.tensor([3], dtype=torch.int32, device="cpu")

    result = proposer.propose(
        num_spec_tokens,
        num_tokens_no_spec,
        token_ids_gpu,
        valid_sampled_token_ids_gpu,
        valid_sampled_tokens_count,
    )

    # Verify the kernel was called once
    assert mock_kernel.call_count == 1

    # Verify the result is returned correctly
    assert result is not None, "propose() must return a tuple (draft_tokens, num_valid)"
    draft_tokens, num_valid = result
    assert draft_tokens.shape == (b, proposer.k), f"Expected (1, {proposer.k}), got {draft_tokens.shape}"
    assert num_valid.shape == (b,), f"Expected (1,), got {num_valid.shape}"

    # Verify token_ids_gpu was updated with sampled tokens at the correct positions
    start_pos = num_tokens_no_spec[0].item()
    n_valid = valid_sampled_tokens_count[0].item()
    for i in range(n_valid):
        actual = token_ids_gpu[0, start_pos + i].item()
        expected = valid_sampled_token_ids_gpu[0, i].item()
        assert actual == expected, (
            f"At position {start_pos + i}: expected {expected}, got {actual}"
        )
