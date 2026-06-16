# SimHash LSH — locality-sensitive hashing for fast approximate similarity search.
#
# Uses random projection + sign bit packing (SimHash variant for cosine
# similarity). Determineistic given a fixed random seed.
#
# Implemented in PLAN Phase 1 (Task 1.2).
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import torch
import torch.nn.functional as F


class SimHashHasher:
    """SimHash-based locality-sensitive hasher for cosine similarity.

    Projects L2-normalized embeddings through a fixed random matrix
    and packs sign bits into int64 hash values.
    """

    def __init__(self, dim: int, num_bits: int = 64, seed: int = 42):
        gen = torch.Generator().manual_seed(seed)
        self.projections: torch.Tensor = torch.randn(dim, num_bits, generator=gen)
        self.num_bits: int = num_bits
        self.dim: int = dim

    @torch.no_grad()
    def hash(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Hash a batch of L2-normalized embeddings.

        Args:
            embeddings: [B, D] L2-normalized input embeddings.

        Returns:
            [B] int64 packed hash values.
        """
        pass


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity between two L2-normalized tensors.

    Args:
        a: [B, D] batch of normalized embeddings.
        b: [B, D] batch of normalized embeddings.

    Returns:
        [B] cosine similarity scores.
    """
    return F.cosine_similarity(a, b, dim=-1)
