# SandwichConfig — layer-selective KV retention.
#
# Retains KV only for bottom-N and top-N layers; discards middle-layer KV
# after the forward pass to save BlockTable memory.
#
# Implemented in PLAN Phase 1 (Task 1.5).
#
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SandwichConfig:
    """Layer-selective KV retention configuration."""

    bottom_layers: int = 3
    top_layers: int = 3
    num_layers: int = 32  # set from model config

    @property
    def keep_layers(self) -> set[int]:
        """Return the set of layer indices whose KV should be retained."""
        return set(range(self.bottom_layers)) | set(range(self.num_layers - self.top_layers, self.num_layers))

    def should_cache(self, layer_idx: int) -> bool:
        """Return True if layer_idx's KV should be written to BlockTable."""
        return layer_idx in self.keep_layers

    @property
    def retention_fraction(self) -> float:
        """Fraction of layers whose KV is retained."""
        return len(self.keep_layers) / self.num_layers
