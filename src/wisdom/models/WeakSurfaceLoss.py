"""Weakly supervised surface priors derived only from protein labels and geometry."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder


class WeakSurfaceLoss(nn.Module):
    """Combine independently weighted weak surface constraints without reading local targets."""

    def __init__(
        self,
        negative_weight         : float = 0.0,
        positive_existence_weight: float = 0.0,
        regional_positive_weight : float = 0.0,
        regional_length          : float = 2.5,
        regional_ranking_weight  : float = 0.0,
        ranking_margin           : float = 0.5,
        cardinality_weight       : float = 0.0,
        minimum_positive_area    : float = 0.0,
        maximum_positive_area    : float | None = None,
        total_variation_weight   : float = 0.0,
        dirichlet_weight         : float = 0.0,
    ) -> None:
        """Configure the sequential weak-loss families from the experimental plan.

        Args:
            negative_weight: Weight of area-normalized negative-bag BCE.
            positive_existence_weight: Weight requiring at least one positive local logit in each
                globally positive protein.
            regional_positive_weight: Weight requiring one positive spectrally smoothed region.
            regional_length: Heat-diffusion length in ångströms for regional terms.
            regional_ranking_weight: Weight of positive-versus-negative regional hinge ranking.
            ranking_margin: Required logit margin between positive and negative regional scores.
            cardinality_weight: Weight of lower/optional upper positive-area constraints.
            minimum_positive_area: Lower positive probability mass fraction for positive proteins.
            maximum_positive_area: Optional broad upper mass fraction; ``None`` disables it.
            total_variation_weight: Weight of probability differences over stored surface edges.
            dirichlet_weight: Weight of normalized intrinsic spectral high-frequency energy.

        Raises:
            ValueError: If a weight, length, margin, or area bound is invalid, or if both smoothness
                alternatives are enabled simultaneously.
        """
        weights = (
            negative_weight,
            positive_existence_weight,
            regional_positive_weight,
            regional_ranking_weight,
            cardinality_weight,
            total_variation_weight,
            dirichlet_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("weak surface loss weights must be non-negative")
        if regional_length < 0.0 or ranking_margin < 0.0:
            raise ValueError("regional length and ranking margin must be non-negative")
        if not 0.0 <= minimum_positive_area <= 1.0:
            raise ValueError("minimum positive area must lie in [0,1]")
        if maximum_positive_area is not None and not (
            minimum_positive_area <= maximum_positive_area <= 1.0
        ):
            raise ValueError("maximum positive area must follow the lower bound within [0,1]")
        if total_variation_weight > 0.0 and dirichlet_weight > 0.0:
            raise ValueError("total variation and Dirichlet smoothness must be compared separately")

        super().__init__()

        self.negative_weight           = float(negative_weight)
        self.positive_existence_weight = float(positive_existence_weight)
        self.regional_positive_weight  = float(regional_positive_weight)
        self.regional_length           = float(regional_length)
        self.regional_ranking_weight   = float(regional_ranking_weight)
        self.ranking_margin            = float(ranking_margin)
        self.cardinality_weight        = float(cardinality_weight)
        self.minimum_positive_area      = float(minimum_positive_area)
        self.maximum_positive_area      = maximum_positive_area
        self.total_variation_weight     = float(total_variation_weight)
        self.dirichlet_weight           = float(dirichlet_weight)

    @property
    def active(self) -> bool:
        """Return whether at least one weak-loss term contributes to optimization."""
        return any(
            weight > 0.0
            for weight in (
                self.negative_weight,
                self.positive_existence_weight,
                self.regional_positive_weight,
                self.regional_ranking_weight,
                self.cardinality_weight,
                self.total_variation_weight,
                self.dirichlet_weight,
            )
        )

    @property
    def needs_surface_neighbors(self) -> bool:
        """Return whether total variation requires stored surface-neighbour arrays."""
        return self.total_variation_weight > 0.0

    def forward(
        self,
        logits           : Tensor,
        area_weights     : Tensor,
        owners           : Tensor,
        protein_targets  : Tensor,
        operators        : Sequence[Mapping[str, Tensor]],
        surface_ptr      : Tensor,
        surface_neighbors: Tensor | None = None,
        neighbor_mask    : Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Evaluate every configured prior and return named unweighted terms plus their total.

        Args:
            logits: Local surface logits with shape ``[M]``.
            area_weights: Positive represented-area weights with shape ``[M]``.
            owners: Point-to-protein owner indices with shape ``[M]``.
            protein_targets: Global binary labels with shape ``[B]``.
            operators: Per-protein mass/eigenpair packs used by regional and Dirichlet terms.
            surface_ptr: Point prefix boundaries with shape ``[B+1]``.
            surface_neighbors: Optional padded global neighbour indices ``[M,K]``.
            neighbor_mask: Optional valid-neighbour mask aligned with ``surface_neighbors``.

        Returns:
            Mapping of scalar differentiable terms ``negative``, ``positive_existence``,
            ``regional_positive``, ``regional_ranking``, ``cardinality``, ``total_variation``,
            ``dirichlet``, and their configured weighted ``total``.

        Raises:
            ValueError: If tensors are misaligned or total variation lacks neighbour arrays.
        """
        # Weak reductions combine logits with stored geometric areas. CUDA autocast may keep the
        # logits in BF16 while promoting softplus and area products to FP32; accumulating that
        # mixed pair with ``index_add_`` is illegal. These scalar priors are also more stable in
        # FP32, so the complete loss boundary is evaluated explicitly in that dtype while the cast
        # remains differentiable with respect to the mixed-precision model output.

        local_logits = logits.reshape(-1).float()
        areas        = area_weights.reshape(-1).float()
        point_owners = owners.reshape(-1).long()
        targets      = protein_targets.reshape(-1)
        if local_logits.shape != areas.shape or local_logits.shape != point_owners.shape:
            raise ValueError("weak surface losses require aligned point arrays")
        if len(targets) == 0 or point_owners.min() < 0 or point_owners.max() >= len(targets):
            raise ValueError("weak surface losses received an invalid protein owner")

        zero            = local_logits.sum() * 0.0
        normalized_area = self._normalized_area(areas, point_owners, len(targets))
        positive        = targets > 0.5
        negative        = ~positive
        maximum         = self._protein_maximum(local_logits, point_owners, len(targets))

        terms = {
            "negative": self._negative(
                local_logits,
                normalized_area,
                point_owners,
                negative,
                zero,
            ),
            "positive_existence": self._positive_existence(maximum, positive, zero),
            "regional_positive": zero,
            "regional_ranking":  zero,
            "cardinality":       self._cardinality(
                local_logits, normalized_area, point_owners, positive, len(targets), zero
            ),
            "total_variation":   zero,
            "dirichlet":         zero,
        }

        if self.regional_positive_weight > 0.0 or self.regional_ranking_weight > 0.0:
            regional_logits = DiffusionSurfaceEncoder.diffuse(
                local_logits,
                operators,
                surface_ptr,
                self.regional_length,
            )
            regional_maximum = self._protein_maximum(
                regional_logits,
                point_owners,
                len(targets),
            )
            terms["regional_positive"] = self._positive_existence(
                regional_maximum,
                positive,
                zero,
            )
            terms["regional_ranking"] = self._regional_ranking(
                regional_maximum,
                positive,
                negative,
                zero,
            )

        if self.total_variation_weight > 0.0:
            if surface_neighbors is None or neighbor_mask is None:
                raise ValueError("total variation requires surface neighbours and their mask")
            terms["total_variation"] = self._total_variation(
                torch.sigmoid(local_logits),
                surface_neighbors,
                neighbor_mask,
            )
        if self.dirichlet_weight > 0.0:
            terms["dirichlet"] = self._dirichlet(
                torch.sigmoid(local_logits),
                operators,
                surface_ptr,
            )

        terms["total"] = (
            self.negative_weight * terms["negative"]
            + self.positive_existence_weight * terms["positive_existence"]
            + self.regional_positive_weight * terms["regional_positive"]
            + self.regional_ranking_weight * terms["regional_ranking"]
            + self.cardinality_weight * terms["cardinality"]
            + self.total_variation_weight * terms["total_variation"]
            + self.dirichlet_weight * terms["dirichlet"]
        )
        return terms

    @staticmethod
    def _normalized_area(areas: Tensor, owners: Tensor, protein_count: int) -> Tensor:
        """Normalize represented area independently inside each protein.

        Args:
            areas: Positive point areas ``[M]``.
            owners: Point owners ``[M]``.
            protein_count: Number of proteins ``B``.

        Returns:
            Point weights ``[M]`` summing to one for each owner.
        """
        sums = areas.new_zeros(protein_count).index_add_(0, owners, areas)
        return areas / sums[owners].clamp_min(torch.finfo(areas.dtype).eps)

    @staticmethod
    def _protein_maximum(values: Tensor, owners: Tensor, protein_count: int) -> Tensor:
        """Reduce one point field to one differentiable maximum per protein.

        Args:
            values: Point field ``[M]``.
            owners: Point owners ``[M]``.
            protein_count: Number of proteins ``B``.

        Returns:
            Per-protein maxima ``[B]``.
        """
        output = values.new_full((protein_count,), float("-inf"))
        return output.scatter_reduce(0, owners, values, reduce="amax", include_self=True)

    @staticmethod
    def _negative(
        logits         : Tensor,
        normalized_area: Tensor,
        owners         : Tensor,
        negative       : Tensor,
        zero           : Tensor,
    ) -> Tensor:
        """Compute area-weighted negative-bag BCE without local annotations.

        Args:
            logits: Point logits ``[M]``.
            normalized_area: Within-protein area fractions ``[M]``.
            owners: Point owners ``[M]``.
            negative: Protein-level negative mask ``[B]``.
            zero: Differentiable scalar zero fallback.

        Returns:
            Mean negative-bag BCE or ``zero`` when the batch has no negative protein.
        """
        if not torch.any(negative):
            return zero
        per_protein = logits.new_zeros(len(negative)).index_add_(
            0,
            owners,
            normalized_area * F.softplus(logits),
        )
        return per_protein[negative].mean()

    @staticmethod
    def _positive_existence(maximum: Tensor, positive: Tensor, zero: Tensor) -> Tensor:
        """Require one positive local or regional score in every positive bag.

        Args:
            maximum: Per-protein maximum logits ``[B]``.
            positive: Protein-level positive mask ``[B]``.
            zero: Differentiable scalar zero fallback.

        Returns:
            Mean ``softplus(-maximum)`` over positive proteins or ``zero`` when absent.
        """
        return F.softplus(-maximum[positive]).mean() if torch.any(positive) else zero

    def _regional_ranking(
        self,
        regional_maximum: Tensor,
        positive        : Tensor,
        negative        : Tensor,
        zero            : Tensor,
    ) -> Tensor:
        """Rank every positive regional maximum above every negative maximum by a margin.

        Args:
            regional_maximum: Per-protein regional scores ``[B]``.
            positive: Positive protein mask ``[B]``.
            negative: Negative protein mask ``[B]``.
            zero: Differentiable scalar zero fallback.

        Returns:
            Mean pairwise hinge loss, or ``zero`` when either class is absent.
        """
        if not torch.any(positive) or not torch.any(negative):
            return zero
        differences = (
            self.ranking_margin
            - regional_maximum[positive][:, None]
            + regional_maximum[negative][None, :]
        )
        return F.relu(differences).mean()

    def _cardinality(
        self,
        logits         : Tensor,
        normalized_area: Tensor,
        owners         : Tensor,
        positive       : Tensor,
        protein_count  : int,
        zero           : Tensor,
    ) -> Tensor:
        """Apply broad lower and optional upper positive-area priors to positive bags.

        Args:
            logits: Point logits ``[M]``.
            normalized_area: Within-protein area fractions ``[M]``.
            owners: Point owners ``[M]``.
            positive: Positive protein mask ``[B]``.
            protein_count: Number of proteins ``B``.
            zero: Differentiable scalar zero fallback.

        Returns:
            Mean squared bound violation over positive proteins, or ``zero`` when absent.
        """
        if not torch.any(positive):
            return zero
        mass = logits.new_zeros(protein_count).index_add_(
            0,
            owners,
            normalized_area * torch.sigmoid(logits),
        )[positive]
        penalty = F.relu(self.minimum_positive_area - mass).square()
        if self.maximum_positive_area is not None:
            penalty = penalty + F.relu(mass - self.maximum_positive_area).square()
        return penalty.mean()

    @staticmethod
    def _total_variation(
        probabilities: Tensor,
        neighbors    : Tensor,
        mask         : Tensor,
    ) -> Tensor:
        """Measure absolute probability variation over the stored bounded surface topology.

        Args:
            probabilities: Point probabilities ``[M]``.
            neighbors: Padded global neighbour IDs ``[M,K]``.
            mask: Valid-neighbour mask ``bool [M,K]``.

        Returns:
            Mean absolute difference over valid directed neighbour entries.
        """
        valid = mask.bool()
        if not torch.any(valid):
            return probabilities.sum() * 0.0
        source = probabilities[:, None].expand_as(neighbors)[valid]
        target = probabilities[neighbors[valid]]
        return torch.mean(torch.abs(source - target))

    @staticmethod
    def _dirichlet(
        probabilities: Tensor,
        operators    : Sequence[Mapping[str, Tensor]],
        surface_ptr  : Tensor,
    ) -> Tensor:
        """Measure normalized intrinsic high-frequency energy of each probability map.

        For ``q=sigmoid(l)``, coefficients are ``c=Phi^T M q`` and the normalized energy is
        ``sum_k lambda_k c_k² / (sum_k c_k² + epsilon)``. This respects intrinsic surface geometry
        and avoids constructing a learned or dense graph.

        Args:
            probabilities: Concatenated point probabilities ``[M]``.
            operators: Per-protein mass and eigenpairs.
            surface_ptr: Point prefix boundaries ``[B+1]``.

        Returns:
            Mean normalized spectral energy over proteins.
        """
        energies: list[Tensor] = []
        for index, operator in enumerate(operators):
            start        = int(surface_ptr[index])
            stop         = int(surface_ptr[index + 1])
            local        = probabilities[start:stop].float()
            mass         = operator["mass"].float()
            eigenvalues  = operator["eigenvalues"].float()
            eigenvectors = operator["eigenvectors"].float()
            coefficients = eigenvectors.T @ (mass * local)
            power        = coefficients.square()
            numerator    = torch.sum(eigenvalues * power)
            denominator  = torch.sum(power).clamp_min(torch.finfo(power.dtype).eps)
            energies.append(numerator / denominator)
        return torch.stack(energies).mean().to(probabilities.dtype)
