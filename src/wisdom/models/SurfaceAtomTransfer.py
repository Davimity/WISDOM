"""Learned bounded transfer from atomic embeddings to molecular-surface points."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class SurfaceAtomTransfer(nn.Module):
    """Aggregate at most J atom embeddings per point with invariant geometric attention."""

    def __init__(
        self,
        hidden_dim : int,
        radius     : float = 6.0,
        score_width: int = 16,
        chunk_size : int = 8192,
    ) -> None:
        """Build the scalar geometric scorer and bounded execution policy.

        Args:
            hidden_dim: Atomic embedding width ``H`` and returned surface-context width.
            radius: Preprocessing surface-to-atom cutoff in ångströms used for normalization.
            score_width: Hidden width of the two-layer scalar attention scorer.
            chunk_size: Maximum surface points evaluated together; peak gathered activations are
                proportional to ``chunk_size * J * H`` rather than all proteins' old edge count.

        Raises:
            ValueError: If a width, radius, or chunk size is non-positive.
        """
        super().__init__()
        if hidden_dim < 1 or radius <= 0.0 or score_width < 1 or chunk_size < 1:
            raise ValueError("transfer dimensions, radius, and chunk size must be positive")

        self.scorer = nn.Sequential(
            nn.Linear(3, score_width),
            nn.SiLU(),
            nn.Linear(score_width, 1),
        )
        self.hidden_dim = hidden_dim
        self.radius     = float(radius)
        self.chunk_size = chunk_size

    def forward(
        self,
        atom_embeddings    : Tensor,
        neighbors          : Tensor,
        distances          : Tensor,
        normal_offsets     : Tensor,
        tangential_distances: Tensor,
        mask               : Tensor,
    ) -> Tensor:
        """Produce one learned local-chemistry vector per surface point.

        The scorer receives ``(d/r, z/r, rho/r)`` where ``d`` is distance, ``z`` is signed normal
        offset, ``rho`` is tangential magnitude, and ``r`` is the physical cutoff. Masked softmax
        weights combine only valid atom embeddings. These three scalars are unchanged by a common
        rigid rotation or translation of atoms and surface points.

        Args:
            atom_embeddings: Encoded atom features ``float [N,H]``.
            neighbors: Atom IDs ``long [M,J]`` with ``-1`` sentinels.
            distances: Center distances ``float [M,J]`` in ångströms.
            normal_offsets: Signed normal components ``float [M,J]`` in ångströms.
            tangential_distances: Tangential magnitudes ``float [M,J]`` in ångströms.
            mask: Boolean validity array ``[M,J]``.

        Returns:
            Weighted atomic context ``float [M,H]`` in exact surface point order.

        Raises:
            ValueError: If tensor shapes disagree, valid endpoints leave ``[0,N)``, or a point has
                no valid neighbor.
        """
        point_count = neighbors.shape[0]
        if neighbors.ndim != 2:
            raise ValueError("surface atom neighbors must have shape [M,J]")
        if any(
            value.shape != neighbors.shape
            for value in (distances, normal_offsets, tangential_distances, mask)
        ):
            raise ValueError("surface atom geometry and mask must share shape [M,J]")
        if atom_embeddings.ndim != 2 or atom_embeddings.shape[1] != self.hidden_dim:
            raise ValueError("atom embeddings must have shape [N,H]")
        if torch.any(mask.sum(dim=1) == 0):
            raise ValueError("every surface point requires at least one active atom")
        if neighbors[mask].numel() and (
            neighbors[mask].min() < 0 or neighbors[mask].max() >= len(atom_embeddings)
        ):
            raise ValueError("surface atom neighbor is out of range")

        outputs: list[Tensor] = []
        for start in range(0, point_count, self.chunk_size):
            stop       = min(start + self.chunk_size, point_count)
            chunk_mask = mask[start:stop]
            chunk_ids  = neighbors[start:stop].clamp_min(0)
            geometry   = torch.stack(
                (
                    distances[start:stop],
                    normal_offsets[start:stop],
                    tangential_distances[start:stop],
                ),
                dim=-1,
            ) / self.radius
            scores  = self.scorer(geometry).squeeze(-1)
            scores  = scores.masked_fill(~chunk_mask, -torch.inf)
            weights = torch.softmax(scores, dim=1).masked_fill(~chunk_mask, 0.0)
            gathered = atom_embeddings[chunk_ids]
            outputs.append(torch.sum(weights.unsqueeze(-1) * gathered, dim=1))

        return torch.cat(outputs, dim=0)
