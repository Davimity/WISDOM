"""Bounded atom-to-surface transfer with separable semantic residuals."""

import torch

from collections.abc import Mapping
from torch import Tensor, nn


class SurfaceAtomTransfer(nn.Module):
    """Aggregate nearby atom embeddings using invariant distance, orientation, and content."""

    def __init__(
        self,
        hidden_dim : int,
        radius     : float = 6.0,
        score_width: int = 16,
        chunk_size : int = 8192,
    ) -> None:
        """Build three small scalar scorers and a bounded execution policy.

        Args:
            hidden_dim: Atomic embedding width ``H`` and returned context width.
            radius: Physical atom-to-surface cutoff in ångströms.
            score_width: Hidden width of each residual scorer.
            chunk_size: Maximum surface points processed together.

        Raises:
            ValueError: If a width, radius, or chunk size is non-positive.
        """
        super().__init__()
        if hidden_dim < 1 or radius <= 0.0 or score_width < 1 or chunk_size < 1:
            raise ValueError("transfer dimensions, radius, and chunk size must be positive")

        self.distance_scorer = nn.Sequential(
            nn.Linear(1, score_width), nn.SiLU(), nn.Linear(score_width, 1)
        )
        self.orientation_scorer = nn.Sequential(
            nn.Linear(3, score_width), nn.SiLU(), nn.Linear(score_width, 1, bias=False)
        )
        self.content_scorer = nn.Sequential(
            nn.Linear(hidden_dim + 1, score_width),
            nn.SiLU(),
            nn.Linear(score_width, 1, bias=False),
        )

        self.hidden_dim = hidden_dim
        self.radius     = float(radius)
        self.chunk_size = chunk_size

    def forward(
        self,
        atom_embeddings     : Tensor,
        neighbors           : Tensor,
        distances           : Tensor,
        normal_offsets      : Tensor,
        tangential_distances: Tensor,
        mask                : Tensor,
        gates               : Mapping[str, Tensor] | None = None,
    ) -> Tensor:
        """Return one weighted atomic context vector per surface point.

        The base distance score is always available. Signed normal offset ``z`` and tangential
        distance ``rho`` enter only through ``transfer.orientation``; learned atom content enters
        only through ``transfer.atom_content``. All geometric quantities are rigid-motion
        invariant, and masked softmax excludes padding and atoms outside ``radius``.

        Args:
            atom_embeddings: Encoded atoms ``float [N,H]``.
            neighbors: Atom IDs ``long [M,J]`` with ``-1`` padding.
            distances: Center distances ``float [M,J]`` in ångströms.
            normal_offsets: Signed normal components ``float [M,J]`` in ångströms.
            tangential_distances: Tangential magnitudes ``float [M,J]`` in ångströms.
            mask: Valid-neighbour mask ``bool [M,J]``.
            gates: One-forward semantic gate values. ``None`` enables both residuals for direct
                transfer diagnostics outside a complete WISDOM model.

        Returns:
            Atomic context ``float [M,H]`` in unchanged surface order.
        """
        self._validate(
            atom_embeddings, neighbors, distances, normal_offsets, tangential_distances, mask
        )
        if gates is None:
            gates = {
                "transfer.orientation": atom_embeddings.new_ones(()),
                "transfer.atom_content": atom_embeddings.new_ones(()),
            }

        outputs: list[Tensor] = []
        for start in range(0, len(neighbors), self.chunk_size):
            stop       = min(start + self.chunk_size, len(neighbors))
            atom_ids   = neighbors[start:stop].clamp_min(0)
            valid      = mask[start:stop] & (distances[start:stop] <= self.radius)
            gathered   = atom_embeddings[atom_ids]
            normalized = (distances[start:stop] / self.radius).unsqueeze(-1)
            orientation = torch.stack(
                (
                    distances[start:stop] / self.radius,
                    normal_offsets[start:stop] / self.radius,
                    tangential_distances[start:stop] / self.radius,
                ),
                dim=-1,
            )

            base_score = self.distance_scorer(normalized).squeeze(-1)
            orientation_score = (
                self.orientation_scorer(orientation)
                - self.orientation_scorer(torch.zeros_like(orientation))
            ).squeeze(-1)
            content_input = torch.cat((normalized, gathered), dim=-1)
            content_score = (
                self.content_scorer(content_input)
                - self.content_scorer(
                    torch.cat((normalized, torch.zeros_like(gathered)), dim=-1)
                )
            ).squeeze(-1)
            scores = (
                base_score
                + gates["transfer.orientation"] * orientation_score
                + gates["transfer.atom_content"] * content_score
            )
            scores  = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).eps
            )
            outputs.append(torch.sum(weights.unsqueeze(-1) * gathered, dim=1))

        return torch.cat(outputs, dim=0)

    def _validate(
        self,
        atom_embeddings     : Tensor,
        neighbors           : Tensor,
        distances           : Tensor,
        normal_offsets      : Tensor,
        tangential_distances: Tensor,
        mask                : Tensor,
    ) -> None:
        """Check shape invariants required for safe bounded gathering.

        Args:
            atom_embeddings: Encoded atom matrix ``[N,H]``.
            neighbors: Atom-neighbour table ``[M,J]``.
            distances: Distance table ``[M,J]``.
            normal_offsets: Normal-offset table ``[M,J]``.
            tangential_distances: Tangential-distance table ``[M,J]``.
            mask: Validity table ``[M,J]``.

        Raises:
            ValueError: If widths or table shapes disagree.
        """
        if atom_embeddings.ndim != 2 or atom_embeddings.shape[1] != self.hidden_dim:
            raise ValueError("atom embeddings must have shape [N,H]")
        if neighbors.ndim != 2 or any(
            value.shape != neighbors.shape
            for value in (distances, normal_offsets, tangential_distances, mask)
        ):
            raise ValueError("surface atom tables must share shape [M,J]")
