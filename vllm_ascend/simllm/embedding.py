# Embedding extraction — pooling strategies for task embedding from hidden states.
#
# Extracts a fixed-size embedding vector from model hidden states using
# configurable pooling strategies (mean, last, cls).
#
# Implemented in PLAN Phase 1 (Task 1.3).
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch


def extract_embedding(
    hidden_states: torch.Tensor,
    pooling: str = "mean",
) -> torch.Tensor:
    """Extract a pooled task embedding from hidden states.

    Args:
        hidden_states: [B, L, D] hidden states from the final transformer layer.
        pooling: Pooling strategy — "mean", "last", or "cls".

    Returns:
        [B, D] L2-normalized task embedding.
    """
    pass
