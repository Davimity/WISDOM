"""Small symmetric conductance scorer with end-to-end representation gradients."""

import torch

from torch import Tensor, nn
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext
from wisdom.models.refinement.GeometricAnisotropicEvidenceRefiner import (
    GeometricAnisotropicEvidenceRefiner,
)


class LearnedAnisotropicEvidenceRefiner(GeometricAnisotropicEvidenceRefiner):
    """Learn edge compatibility, not a second surface encoder or a new prediction head."""

    def __init__(
        self,
        embedding_dim: int,
        hidden_dim   : int = 32,
        strength     : float = 0.5,
        steps        : int = 2,
        geometry_sigma: float = 2.0,
    ) -> None:
        """Build an eight-channel point projection and a tiny symmetric pair scorer.

        Args:
            embedding_dim: Width H of the existing surface embeddings.
            hidden_dim: Positive edge MLP width, conventionally 16 or 32.
            strength: Convex evidence mixing fraction in [0,1].
            steps: Nonnegative number of residual updates.
            geometry_sigma: Positive spatial scale in Å for the scorer input.

        Raises:
            ValueError: If inherited mixing/bandwidth settings are invalid.
        """
        super().__init__(strength, steps, geometry_sigma)
        self.projection = nn.Linear(embedding_dim, 8)
        self.scorer = nn.Sequential(nn.Linear(19, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, 1))
        self.last_statistics: tuple[Tensor, Tensor] | None = None

    def conductance(
        self, context: SurfaceEvidenceContext, left: Tensor, right: Tensor,
    ) -> Tensor:
        """Score symmetric projected pairs and geometry with a bounded sigmoid.

        Args:
            context: Embeddings and generic geometry, never targets.
            left: First edge endpoints [E].
            right: Second edge endpoints [E].

        Returns:
            sigmoid(MLP([abs(u_i-u_j),u_i*u_j,d/sigma,1-dot,delta_k²])) [E].
            u=tanh(project(H)) remains differentiable to H and all scorer parameters.
            Only two detached summary scalars survive for logging; no edge graph is retained.
        """
        projected = torch.tanh(self.projection(
            context.embeddings.to(self.projection.weight.dtype),
        )).float()
        distance, alignment, difference = context.geometry(left, right)
        features = torch.cat((
            (projected[left] - projected[right]).abs(),
            projected[left] * projected[right],
            (distance.sqrt() / self.geometry_sigma)[:, None],
            alignment[:, None], difference[:, None],
        ), dim=-1)
        weights = torch.sigmoid(self.scorer(features.to(next(self.scorer.parameters()).dtype)))
        weights = weights.reshape(-1).float()
        self.last_statistics = (
            (weights.mean().detach(), weights.std(unbiased=False).detach())
            if weights.numel() else None
        )
        return weights

    def parameter_values(self) -> dict[str, float]:
        """Report the last batch's conductance moments, not epoch-averaged statistics.

        Returns:
            Mean/std in [0,1] from the last forward; empty for an edgeless batch.
        """
        if self.last_statistics is None:
            return {}
        mean, std = self.last_statistics
        return {
            "surface_refiner_conductance_mean": float(mean.cpu()),
            "surface_refiner_conductance_std":  float(std.cpu()),
        }
