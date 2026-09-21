"""Bounded atom-to-surface transfer with separable semantic residuals."""

import torch

from torch import Tensor, nn
from collections.abc import Mapping


class SurfaceAtomTransfer(nn.Module):
    """Aggregate nearby atoms with optional candidate-specific surface-shape conditioning."""

    def __init__(
        self,
        hidden_dim       : int,
        radius           : float = 6.0,
        score_width      : int = 16,
        chunk_size       : int = 8192,
        point_feature_dim: int = 0,
        neutral_residual_initialization: bool = False,
        physics_distance_initialization: bool = False,
    ) -> None:
        """Build separable scalar scorers and a bounded execution policy.

        Args:
            hidden_dim: Atomic embedding width ``H`` and returned context width.
            radius: Physical atom-to-surface cutoff in ångströms.
            score_width: Hidden width of each residual scorer.
            chunk_size: Maximum surface points processed together.
            point_feature_dim: Width ``G`` of optional gated surface geometry. Zero preserves the
                original distance/orientation/content transfer exactly.
            neutral_residual_initialization: Start orientation/content corrections at zero.
            physics_distance_initialization: Start the base score at ``-distance/radius`` plus a
                zero-initialized learned residual; both parts remain trainable.

        Raises:
            ValueError: If a width, radius, or chunk size is non-positive.
        """
        super().__init__()
        if (
            hidden_dim < 1
            or radius <= 0.0
            or score_width < 1
            or chunk_size < 1
            or point_feature_dim < 0
        ):
            raise ValueError(
                "transfer dimensions/chunk size must be positive and point width non-negative"
            )

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
        if neutral_residual_initialization:
            self.initialize_neutral_residuals()
        if physics_distance_initialization:
            self.distance_prior: nn.Parameter | None = nn.Parameter(torch.tensor(-1.0))
            self.initialize_physics_distance()
        else:
            self.distance_prior = None

        # The optional fourth scorer is deliberately an interaction, not a point-only logit. A
        # low-rank Hadamard product lets surface geometry change the relative candidate scores
        # without materializing a wider [M,J,H+G] tensor.

        if point_feature_dim:
            self.point_geometry_projection: nn.Linear | None = nn.Linear(
                point_feature_dim,
                score_width,
                bias=False,
            )
            self.point_atom_projection: nn.Linear | None = nn.Linear(
                hidden_dim,
                score_width,
                bias=False,
            )
            self.point_pair_projection: nn.Linear | None = nn.Linear(
                3,
                score_width,
                bias=False,
            )
            self.point_geometry_scorer: nn.Linear | None = nn.Linear(
                score_width,
                1,
                bias=False,
            )
        else:
            self.point_geometry_projection = None
            self.point_atom_projection     = None
            self.point_pair_projection     = None
            self.point_geometry_scorer     = None

        self.hidden_dim        = hidden_dim
        self.radius            = float(radius)
        self.chunk_size        = chunk_size
        self.point_feature_dim = point_feature_dim

    def initialize_neutral_residuals(self) -> None:
        """Zero orientation/content output layers while retaining distance-based transfer.

        These two scorers are additive corrections to a valid distance score. Their zero start
        therefore has a direct physical interpretation and does not block subsequent gradients.
        """
        orientation_output = self.orientation_scorer[-1]
        content_output     = self.content_scorer[-1]
        if not isinstance(orientation_output, nn.Linear) or not isinstance(
            content_output, nn.Linear
        ):
            raise RuntimeError("transfer residual scorers must end in linear layers")
        nn.init.zeros_(orientation_output.weight)
        nn.init.zeros_(content_output.weight)

    def initialize_physics_distance(self) -> None:
        """Reset the distance residual around the trainable monotone ``-d/R`` prior.

        Raises:
            RuntimeError: If this transfer was not constructed with the physical prior parameter.
        """
        if self.distance_prior is None:
            raise RuntimeError("physical distance initialization was not configured")
        distance_output = self.distance_scorer[-1]
        if not isinstance(distance_output, nn.Linear) or distance_output.bias is None:
            raise RuntimeError("distance scorer must end in one biased linear layer")
        nn.init.zeros_(distance_output.weight)
        nn.init.zeros_(distance_output.bias)
        with torch.no_grad():
            self.distance_prior.fill_(-1.0)

    def forward(
        self,
        atom_embeddings     : Tensor,
        neighbors           : Tensor,
        distances           : Tensor,
        normal_offsets      : Tensor,
        tangential_distances: Tensor,
        mask                : Tensor,
        gates               : Mapping[str, Tensor] | None = None,
        point_features      : Tensor | None = None,
    ) -> Tensor:
        """Return one weighted atomic context vector per surface point.

        The base distance score is always available. Signed normal offset ``z`` and tangential
        distance ``rho`` enter through ``transfer.orientation``; learned atom content enters
        through ``transfer.atom_content``. When configured, gated surface geometry ``g_p`` enters
        only through a candidate-specific low-rank interaction
        ``u(g_p)^T [v(h_a) + w(d_pa,z_pa,rho_pa)]``. Consequently it can change relative softmax
        weights, unlike adding one point-only scalar to every candidate. All geometric quantities
        are rigid-motion invariant, and masked softmax excludes padding and atoms outside
        ``radius``.

        Args:
            atom_embeddings: Encoded atoms ``float [N,H]``.
            neighbors: Atom IDs ``long [M,J]`` with ``-1`` padding.
            distances: Center distances ``float [M,J]`` in ångströms.
            normal_offsets: Signed normal components ``float [M,J]`` in ångströms.
            tangential_distances: Tangential magnitudes ``float [M,J]`` in ångströms.
            mask: Valid-neighbour mask ``bool [M,J]``.
            gates: One-forward semantic gate values. ``None`` enables every configured residual
                for direct transfer diagnostics outside a complete WISDOM model.
            point_features: Existing transformed and semantically gated surface descriptors
                ``float [M,G]``. Required only when ``point_feature_dim`` is non-zero.

        Returns:
            Atomic context ``float [M,H]`` in unchanged surface order.
        """
        self._validate(
            atom_embeddings,
            neighbors,
            distances,
            normal_offsets,
            tangential_distances,
            mask,
            point_features,
        )
        if gates is None:
            gates = {
                "transfer.orientation":  atom_embeddings.new_ones(()),
                "transfer.atom_content": atom_embeddings.new_ones(()),
            }
            if self.point_feature_dim:
                gates["transfer.point_geometry"] = atom_embeddings.new_ones(())

        # The orientation residual is centered at one shared all-zero vector. Its response does
        # not depend on a surface point or neighbour, so compute it once for the complete forward.

        zero_orientation = torch.zeros(
            (1, 3),
            device=atom_embeddings.device,
            dtype=atom_embeddings.dtype,
        )
        orientation_origin = self.orientation_scorer(zero_orientation)

        outputs: list[Tensor] = []
        for start in range(0, len(neighbors), self.chunk_size):
            stop                = min(start + self.chunk_size, len(neighbors))
            chunk_distances     = distances[start:stop]
            distance_normalized = chunk_distances / self.radius
            normal_normalized   = normal_offsets[start:stop] / self.radius
            tangent_normalized  = tangential_distances[start:stop] / self.radius

            atom_ids = neighbors[start:stop].clamp_min(0)
            valid    = mask[start:stop] & (chunk_distances <= self.radius)
            gathered = atom_embeddings[atom_ids]
            normalized = distance_normalized.unsqueeze(-1)
            orientation = torch.stack(
                (
                    distance_normalized,
                    normal_normalized,
                    tangent_normalized,
                ),
                dim=-1,
            )

            base_score = self.distance_scorer(normalized).squeeze(-1)
            if self.distance_prior is not None:
                base_score = base_score + self.distance_prior * distance_normalized
            orientation_score = (
                self.orientation_scorer(orientation)
                - orientation_origin
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

            # Surface geometry must interact with a candidate-dependent quantity. The point branch
            # has no bias, so closing every curvature gate makes this residual exactly zero and
            # recovers the original transfer rather than merely approximating it.

            if self.point_feature_dim:
                assert point_features is not None
                assert self.point_geometry_projection is not None
                assert self.point_atom_projection is not None
                assert self.point_pair_projection is not None
                assert self.point_geometry_scorer is not None

                point_state = torch.tanh(
                    self.point_geometry_projection(point_features[start:stop])
                ).unsqueeze(1)
                pair_state = torch.nn.functional.silu(
                    self.point_atom_projection(gathered)
                    + self.point_pair_projection(orientation)
                )
                point_geometry_score = self.point_geometry_scorer(
                    point_state * pair_state
                ).squeeze(-1)
                scores = (
                    scores
                    + gates["transfer.point_geometry"] * point_geometry_score
                )

            scores  = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            weights = torch.softmax(scores, dim=1)
            weights = torch.where(valid, weights, torch.zeros_like(weights))
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(
                torch.finfo(weights.dtype).eps
            )

            # Batched matrix multiplication contracts [M,1,J] with [M,J,H] directly. It avoids
            # materializing the former [M,J,H] weighted tensor while preserving the same sum.

            outputs.append(torch.bmm(weights.unsqueeze(1), gathered).squeeze(1))

        return torch.cat(outputs, dim=0)

    def _validate(
        self,
        atom_embeddings     : Tensor,
        neighbors           : Tensor,
        distances           : Tensor,
        normal_offsets      : Tensor,
        tangential_distances: Tensor,
        mask                : Tensor,
        point_features      : Tensor | None,
    ) -> None:
        """Check shape invariants required for safe bounded gathering.

        Args:
            atom_embeddings: Encoded atom matrix ``[N,H]``.
            neighbors: Atom-neighbour table ``[M,J]``.
            distances: Distance table ``[M,J]``.
            normal_offsets: Normal-offset table ``[M,J]``.
            tangential_distances: Tangential-distance table ``[M,J]``.
            mask: Validity table ``[M,J]``.
            point_features: Optional surface geometry matrix ``[M,G]``.

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
        if self.point_feature_dim and (
            point_features is None
            or point_features.ndim != 2
            or point_features.shape != (len(neighbors), self.point_feature_dim)
        ):
            raise ValueError("point features must have configured shape [M,G]")
