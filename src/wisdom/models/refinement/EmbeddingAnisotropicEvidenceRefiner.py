"""Representation-guided local conductance, optionally detached only in its guide branch."""

import torch

from torch import Tensor
from wisdom.models.refinement.GeometricAnisotropicEvidenceRefiner import (
    GeometricAnisotropicEvidenceRefiner,
)
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext


class EmbeddingAnisotropicEvidenceRefiner(GeometricAnisotropicEvidenceRefiner):
    """Use local spatial compatibility times cosine similarity of surface representations."""

    def __init__(
        self,
        strength      : float = 0.5,
        steps         : int = 2,
        temperature   : float = 0.5,
        embedding_detach: bool = True,
        geometry_sigma: float = 2.0,
    ) -> None:
        """Configure fixed conductance and the explicit representation-gradient policy.

        Args:
            strength: Convex mixing fraction in [0,1].
            steps: Nonnegative number of updates.
            temperature: Positive cosine-distance bandwidth.
            embedding_detach: Detach only the embeddings used to construct conductance.
            geometry_sigma: Positive spatial bandwidth in Å.

        Raises:
            ValueError: If temperature or inherited mixing/bandwidth settings are invalid.
        """
        super().__init__(strength, steps, geometry_sigma)
        if temperature <= 0.0:
            raise ValueError("surface refiner embedding temperature must be positive")
        self.temperature      = temperature
        self.embedding_detach = embedding_detach

    def conductance(
        self, context: SurfaceEvidenceContext, left: Tensor, right: Tensor,
    ) -> Tensor:
        """Build symmetric exp(-spatial²/(2 sigma²)-(1-cosine)/temperature).

        Args:
            context: Borrowed surface embeddings and positions.
            left: First edge endpoints [E].
            right: Second edge endpoints [E].

        Returns:
            Local conductances [E]. Detached guides do not detach the evidence path.

        Raises:
            ValueError: If surface positions are unavailable.
        """
        if context.positions is None:
            raise ValueError("embedding evidence refinement requires surface positions")

        # stopgrad applies to the guide only; logits still connect the local head to the backbone.

        embeddings = (context.evidence_features.detach() if self.embedding_detach
                      else context.evidence_features)
        guide = torch.nn.functional.normalize(embeddings.float(), dim=-1)
        cosine = (guide[left] * guide[right]).sum(-1).clamp(-1.0, 1.0)
        distance = (context.positions[left].float() - context.positions[right].float())
        return (
            -distance.square().sum(-1) / (2.0 * self.geometry_sigma ** 2)
            -(1.0 - cosine) / self.temperature
        ).exp()
