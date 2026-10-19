"""Sparse surface-to-atom feedback over the existing compact transfer relation."""

import torch

from torch import Tensor, nn
from lambdaforge.nn import Scatter
from wisdom.models.ArchitectureImplementation import ArchitectureImplementation, FeedbackWeighting


class SurfaceAtomFeedback(nn.Module):
    """Aggregate surface states onto their candidate atoms and update scalar atom states."""

    def __init__(
        self, hidden_dim: int, weighting: str = "point", implementation: str = "controlled",
    ) -> None:
        """Construct one distinct surface-to-atom residual update.

        Args:
            hidden_dim: Shared atomic and surface state width.
            weighting: point reproduces a distance-weighted sample mean; area additionally
                weights represented surface area, making exact area subdivision invariant.
            implementation: controlled preserves uncovered atoms and accumulates in FP32;
                legacy retains archived point-weighted arithmetic and uncovered-atom behavior.

        Raises:
            ValueError: If ``hidden_dim`` is not positive.
        """
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("surface-to-atom feedback width must be positive")
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization = nn.LayerNorm(hidden_dim)
        self.hidden_dim    = hidden_dim
        self.weighting      = FeedbackWeighting(weighting)
        self.implementation = ArchitectureImplementation(implementation)
        if self.implementation is ArchitectureImplementation.LEGACY and (
            self.weighting is not FeedbackWeighting.POINT
        ):
            raise ValueError("legacy feedback supports historical point weighting only")

    def forward(
        self,
        atom_states   : Tensor,
        surface_states: Tensor | None,
        neighbors     : Tensor,
        distances     : Tensor,
        mask          : Tensor,
        radius        : float,
        area_weights  : Tensor | None = None,
    ) -> Tensor:
        """Return atoms updated by a distance-weighted mean of adjacent surface states.

        Args:
            atom_states: Scalar atom embeddings ``[N,H]``.
            surface_states: First-round surface embeddings ``[M,H]``. None is the explicit
                parameter-matched null-message control: no S1 representation is inspected.
            neighbors: Compact atom candidates ``[M,J]`` with ``-1`` padding.
            distances: Center distances ``[M,J]`` in ångströms.
            mask: Valid candidate mask ``[M,J]``.
            radius: Positive physical transfer cutoff in ångströms.
            area_weights: Represented areas ``[M]`` in Å²; required for area weighting.

        Returns:
            Second-round scalar atom states ``[N,H]``. Controlled uncovered atoms remain
            exactly unchanged. The covered update is LayerNorm(A+MLP([A,message])); no new
            atom-to-atom message pass occurs. Null control uses identical coverage and weights.
        """
        if radius <= 0.0:
            raise ValueError("surface-to-atom radius must be positive")
        if atom_states.ndim != 2 or atom_states.shape[1] != self.hidden_dim:
            raise ValueError("feedback atom states must have shape [N,H]")
        if surface_states is not None and (
            surface_states.shape != (len(neighbors), self.hidden_dim)
        ):
            raise ValueError("feedback surface states must have shape [M,H]")
        if neighbors.shape != distances.shape or neighbors.shape != mask.shape:
            raise ValueError("feedback transfer tables must share shape [M,J]")

        valid = mask.bool() & (neighbors >= 0) & (distances <= radius)
        if not torch.any(valid):
            return atom_states
        point_ids = torch.arange(len(neighbors), device=neighbors.device)[:, None]
        point_ids = point_ids.expand_as(neighbors)[valid]
        atom_ids  = neighbors[valid]
        controlled = self.implementation is ArchitectureImplementation.CONTROLLED
        weights = torch.exp(-(distances[valid].float() if controlled
                              else distances[valid]) / radius)
        if self.weighting is FeedbackWeighting.AREA:
            if area_weights is None:
                raise ValueError("area-weighted feedback requires represented surface areas")
            weights = weights * area_weights[point_ids].float()
        # Accumulate a geometric integral with a normalized measure; zero-area support is
        # uncovered. Tiny positive denominators must not be clipped at BF16/FP16 epsilon.

        if surface_states is None:
            weighted = weights.new_zeros((len(point_ids), self.hidden_dim))
        else:
            values = surface_states[point_ids].float() if controlled else surface_states[point_ids]
            weighted = values * weights[:, None]
        totals    = Scatter.sum(weighted, atom_ids, len(atom_states))
        norms     = Scatter.sum(weights, atom_ids, len(atom_states))
        floor     = torch.finfo(norms.dtype).tiny if controlled else torch.finfo(weighted.dtype).eps
        feedback = totals / norms.clamp_min(floor)[:, None]
        update   = self.update(torch.cat((atom_states, feedback), dim=1))
        updated  = self.normalization(atom_states + update)
        return torch.where((norms > 0)[:, None], updated, atom_states) if controlled else updated
