# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""One unified-HCCL PEARL draft/target verification round."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from vllm_ascend.spec_decode.pearl.protocol import (
    PearlProposalBatch,
    PearlVerificationBatch,
    broadcast_proposals,
    broadcast_verifications,
)
from vllm_ascend.spec_decode.pearl.topology import PearlProcessGroups
from vllm_ascend.spec_decode.pearl.verifier import PearlTargetVerifier


@dataclass(frozen=True)
class PearlRoundExecutor:
    """Execute PEARL's proposal, verification, and correction collectives.

    The executor is invoked identically by every member of a unified HCCL
    world. Only the draft leader provides a proposal; only the target leader
    provides target logits. Every rank receives the resulting correction batch.
    """

    groups: PearlProcessGroups
    device: torch.device | str
    verifier: PearlTargetVerifier

    @torch.inference_mode()
    def execute(
        self,
        proposals: PearlProposalBatch | None,
        target_logits: torch.Tensor | None,
        *,
        temperature: float = 0.0,
        generator: torch.Generator | None = None,
    ) -> tuple[PearlProposalBatch | None, PearlVerificationBatch]:
        """Run one PEARL candidate window through target verification."""
        rank = self.groups.rank
        topology = self.groups.topology
        if rank == topology.draft_leader_rank:
            if proposals is None:
                raise ValueError("The PEARL draft leader must provide proposals.")
        elif proposals is not None:
            raise ValueError("Only the PEARL draft leader may provide proposals.")

        received_proposals: PearlProposalBatch | None = None
        if self.groups.is_verification_worker:
            received_proposals = broadcast_proposals(
                proposals,
                source_rank=topology.draft_leader_rank,
                group=self.groups.verification_group,
                device=self.device,
            )

        if rank == topology.target_leader_rank:
            if target_logits is None:
                raise ValueError("The PEARL target leader must provide target logits.")
            assert received_proposals is not None
            verifications = self.verifier.verify(
                received_proposals,
                target_logits,
                temperature=temperature,
                generator=generator,
            )
        else:
            if target_logits is not None:
                raise ValueError("Only the PEARL target leader may provide target logits.")
            verifications = None

        return received_proposals, broadcast_verifications(
            verifications,
            source_rank=topology.target_leader_rank,
            device=self.device,
        )
