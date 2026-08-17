# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import torch

from tests.e2e.nightly.single_node.ops.dispatch_ffn_combine_bf16_common import (
    run_dispatch_ffn_combine_bf16_two_ranks,
)


@torch.inference_mode()
def test_dispatch_ffn_combine_bf16_two_ranks():
    # This escape hatch lets the same deterministic oracle validate the
    # pre-change A3 artifact, whose BF16 adapter intentionally ignored masks.
    baseline_only = os.getenv("VLLM_ASCEND_BF16_BASELINE_ONLY") == "1"
    run_dispatch_ffn_combine_bf16_two_ranks(active_mask_supported=not baseline_only)
