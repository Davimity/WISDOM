"""Sparse surface-to-atom feedback over the existing compact transfer relation."""

import torch
from lambdaforge.nn import Scatter
from torch import Tensor, nn


class SurfaceAtomFeedback(nn.Module):
    """Aggregate surface states onto their candidate atoms and update scalar atom states."""

    def __init__(self, hidden_dim: int) -> None:
        """Construct one distinct surface-to-atom residual update.

        Args:
            hidden_dim: Shared atomic and surface state width.

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

    def forward(
        self,
        atom_states   : Tensor,
        surface_states: Tensor,
        neighbors     : Tensor,
        distances     : Tensor,
        mask          : Tensor,
        radius        : float,
    ) -> Tensor:
        """Return atoms updated by a distance-weighted mean of adjacent surface states.

        Args:
            atom_states: Scalar atom embeddings ``[N,H]``.
            surface_states: First-round surface embeddings ``[M,H]``.
            neighbors: Compact atom candidates ``[M,J]`` with ``-1`` padding.
            distances: Center distances ``[M,J]`` in ångströms.
            mask: Valid candidate mask ``[M,J]``.
            radius: Positive physical transfer cutoff in ångströms.

        Returns:
            Second-round scalar atom states ``[N,H]``.
        """
        if radius <= 0.0:
            raise ValueError("surface-to-atom radius must be positive")
        if atom_states.ndim != 2 or atom_states.shape[1] != self.hidden_dim:
            raise ValueError("feedback atom states must have shape [N,H]")
        if surface_states.ndim != 2 or surface_states.shape[1] != self.hidden_dim:
            raise ValueError("feedback surface states must have shape [M,H]")
        if neighbors.shape != distances.shape or neighbors.shape != mask.shape:
            raise ValueError("feedback transfer tables must share shape [M,J]")

        valid = mask.bool() & (distances <= radius)
        if not torch.any(valid):
            return atom_states
        point_ids = torch.arange(len(surface_states), device=surface_states.device)[:, None]
        point_ids = point_ids.expand_as(neighbors)[valid]
        atom_ids  = neighbors[valid]
        weights   = torch.exp(-distances[valid] / radius)
        weighted  = surface_states[point_ids] * weights[:, None]
        totals    = Scatter.sum(weighted, atom_ids, len(atom_states))
        norms     = Scatter.sum(weights, atom_ids, len(atom_states)).clamp_min(
            torch.finfo(weighted.dtype).eps
        )
        feedback = totals / norms[:, None]
        update   = self.update(torch.cat((atom_states, feedback), dim=1))
        return self.normalization(atom_states + update)
