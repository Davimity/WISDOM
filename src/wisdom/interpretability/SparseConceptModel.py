"""Minimal non-negative sparse bottleneck for frozen surface embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SparseConceptModel(nn.Module):
    """Encode exact-zero concepts and reconstruct standardized WISDOM embeddings."""

    def __init__(self, embedding_dim: int, concept_count: int) -> None:
        """Build one linear ReLU encoder and one linear decoder.

        Args:
            embedding_dim: Frozen WISDOM surface width ``H``.
            concept_count: Probe or final concept width ``K``.

        Raises:
            ValueError: If either dimension is not positive.
        """
        super().__init__()
        if embedding_dim < 1 or concept_count < 1:
            raise ValueError("embedding and concept dimensions must be positive")

        self.encoder = nn.Linear(embedding_dim, concept_count)
        self.decoder = nn.Linear(concept_count, embedding_dim)
        self.normalize_decoder()

    def forward(self, standardized: Tensor) -> tuple[Tensor, Tensor]:
        """Produce non-negative concepts and a standardized reconstruction.

        Args:
            standardized: Standardized frozen embeddings ``[N,H]``.

        Returns:
            Exact-zero ReLU concepts ``[N,K]`` and reconstruction ``[N,H]``.
        """
        concepts      = torch.relu(self.encoder(standardized))
        reconstruction = self.decoder(concepts)
        return concepts, reconstruction

    def normalize_decoder(self) -> None:
        """Project every decoder concept direction onto the unit L2 sphere.

        L1 activation penalties otherwise admit the scale transformation ``c/a`` and ``a*W_d``.
        Unit decoder columns remove that degeneracy while retaining each concept direction.
        """
        with torch.no_grad():
            norms = self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1.0e-12)
            self.decoder.weight.div_(norms)
