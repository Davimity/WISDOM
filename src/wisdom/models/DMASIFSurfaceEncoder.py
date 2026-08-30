"""Compact dMaSIF-like quasi-geodesic surface propagation."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class DMASIFSurfaceEncoder(nn.Module):
    """Aggregate bounded local patches with invariant quasi-geodesic geometric filters."""

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0) -> None:
        """Build a dependency-light molecular-surface convolution hypothesis.

        The original dMaSIF uses PyKeOps and learned quasi-geodesic convolutions. This compact
        implementation preserves its relevant hypothesis—normal-aware local surface filtering—but
        consumes WISDOM's precomputed bounded neighborhoods and does not copy the old non-commercial
        reference code or require its obsolete CUDA/PyTorch stack.

        Args:
            hidden_dim: Scalar surface feature width ``H``.
            layers: Number of residual local-filter blocks.
            dropout: Pointwise dropout probability in ``[0,1)``.

        Raises:
            ValueError: If dimensions or dropout are invalid.
        """
        super().__init__()
        if hidden_dim < 1 or layers < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("dMaSIF encoder dimensions or dropout are invalid")

        self.geometry_scorers = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 16), nn.SiLU(), nn.Linear(16, 1))
            for _ in range(layers)
        )
        self.updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )

    def forward(
        self,
        features : Tensor,
        positions: Tensor,
        normals  : Tensor,
        neighbors: Tensor,
        mask     : Tensor,
    ) -> Tensor:
        """Propagate scalar features through normal-aware bounded local surface patches.

        Args:
            features: Initial surface features ``[M,H]``.
            positions: Surface coordinates ``[M,3]`` in ångströms.
            normals: Outward unit normals ``[M,3]``.
            neighbors: Globally offset neighbor IDs ``[M,Ks]`` with ``-1`` sentinels.
            mask: Validity mask ``[M,Ks]``.

        Returns:
            Updated scalar embeddings ``[M,H]``.
        """
        safe_neighbors = neighbors.clamp_min(0)
        offsets         = positions[safe_neighbors] - positions[:, None, :]
        distances       = torch.linalg.vector_norm(offsets, dim=2).clamp_min(1.0e-8)
        source_offset   = torch.sum(offsets * normals[:, None, :], dim=2) / distances
        target_offset   = torch.sum(offsets * normals[safe_neighbors], dim=2) / distances
        normal_cosine   = torch.sum(
            normals[:, None, :] * normals[safe_neighbors],
            dim=2,
        )
        scale = distances[mask].median().clamp_min(1.0e-6)
        geometry = torch.stack(
            (distances / scale, source_offset, target_offset, normal_cosine),
            dim=2,
        )

        values = features
        for scorer, update in zip(self.geometry_scorers, self.updates, strict=True):
            scores  = scorer(geometry).squeeze(-1).masked_fill(~mask, -torch.inf)
            weights = torch.softmax(scores, dim=1).masked_fill(~mask, 0.0)
            context = torch.sum(weights.unsqueeze(-1) * values[safe_neighbors], dim=1)
            values  = values + update(torch.cat((values, context), dim=1))
        return values
