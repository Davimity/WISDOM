"""Ground-truth-free consistency metrics for aligned surface predictions."""

import math
import torch

from torch import Tensor
from scipy.stats import pearsonr, spearmanr


class ViewConsistencyMetricSuite:
    """Quantify agreement between predictions on two physically aligned surface views."""

    def compute(
        self,
        reference_logits : Tensor,
        view_logits      : Tensor,
        correspondence   : dict[str, Tensor],
    ) -> dict[str, float | None]:
        """Measure rank, probability, distributional, hotspot, and variance agreement.

        Let ``l_i`` and ``m_i`` be logits at matched physical points and let
        ``p_i=sigmoid(l_i)``, ``q_i=sigmoid(m_i)``. The normalized maps are
        ``P_i=p_i/sum_j p_j`` and ``Q_i=q_i/sum_j q_j``. This method reports Spearman correlation
        of logits, Pearson correlation of probabilities, Jensen-Shannon divergence of ``P`` and
        ``Q``, total-variation distance ``0.5*sum_i |P_i-Q_i|``, overlap of the top 5% and 10% of
        matched points, and mean point-wise prediction variance. The scalar
        ``view_surface_consistency`` is the arithmetic mean of their bounded agreement forms.

        No target or DNA annotation is accepted. Consequently these values can measure confidence
        but cannot establish biological correctness by themselves.

        Args:
            reference_logits: Finite surface logits ``[M]`` from the reference discretization.
            view_logits: Finite surface logits ``[N]`` from the second discretization.
            correspondence: Output of ``SurfaceViewCorrespondence.align`` containing aligned
                ``reference_indices`` and ``view_indices``.

        Returns:
            Flat metrics. Correlations remain ``None`` when a constant map makes them undefined;
            all distance, overlap, variance, count, and combined metrics remain finite.

        Raises:
            ValueError: If logits or correspondence indices are empty, non-finite, misaligned, or
                outside their source map.
        """
        logits_a = reference_logits.detach().view(-1).float().cpu()
        logits_b = view_logits.detach().view(-1).float().cpu()
        indices_a = correspondence.get("reference_indices", torch.empty(0)).view(-1).long().cpu()
        indices_b = correspondence.get("view_indices", torch.empty(0)).view(-1).long().cpu()

        if not len(logits_a) or not len(logits_b) or not len(indices_a):
            raise ValueError("view consistency requires non-empty logits and correspondence")
        if len(indices_a) != len(indices_b):
            raise ValueError("view correspondence index vectors must be aligned")
        if not torch.isfinite(logits_a).all() or not torch.isfinite(logits_b).all():
            raise ValueError("view logits must be finite")
        if (
            indices_a.min() < 0
            or indices_b.min() < 0
            or indices_a.max() >= len(logits_a)
            or indices_b.max() >= len(logits_b)
        ):
            raise ValueError("view correspondence index lies outside its surface map")

        aligned_logits_a = logits_a[indices_a]
        aligned_logits_b = logits_b[indices_b]
        probabilities_a  = torch.sigmoid(aligned_logits_a).to(torch.float64)
        probabilities_b  = torch.sigmoid(aligned_logits_b).to(torch.float64)

        normalized_a = probabilities_a / probabilities_a.sum()
        normalized_b = probabilities_b / probabilities_b.sum()
        midpoint     = 0.5 * (normalized_a + normalized_b)
        epsilon      = torch.finfo(torch.float64).eps
        js_divergence = 0.5 * (
            torch.sum(normalized_a * torch.log((normalized_a + epsilon) / (midpoint + epsilon)))
            + torch.sum(normalized_b * torch.log((normalized_b + epsilon) / (midpoint + epsilon)))
        ) / math.log(2.0)
        total_variation = 0.5 * torch.sum(torch.abs(normalized_a - normalized_b))
        point_variance  = torch.var(
            torch.stack((probabilities_a, probabilities_b), dim=0),
            dim=0,
            unbiased=False,
        ).mean()

        spearman = self._correlation(aligned_logits_a, aligned_logits_b, "spearman")
        pearson  = self._correlation(probabilities_a, probabilities_b, "pearson")
        overlap_05 = self._top_overlap(probabilities_a, probabilities_b, 0.05)
        overlap_10 = self._top_overlap(probabilities_a, probabilities_b, 0.10)

        components = [
            1.0 - float(js_divergence),
            1.0 - float(total_variation),
            1.0 - min(1.0, 4.0 * float(point_variance)),
            overlap_05,
            overlap_10,
        ]
        if spearman is not None:
            components.append(0.5 * (spearman + 1.0))
        if pearson is not None:
            components.append(0.5 * (pearson + 1.0))

        return {
            "view_matched_points":             float(len(indices_a)),
            "view_logit_spearman":             spearman,
            "view_probability_pearson":        pearson,
            "view_map_js_divergence":          float(js_divergence),
            "view_map_l1":                     float(2.0 * total_variation),
            "view_map_total_variation":        float(total_variation),
            "view_top_05_overlap":             overlap_05,
            "view_top_10_overlap":             overlap_10,
            "view_regional_probability_variance": float(point_variance),
            "view_surface_consistency":        math.fsum(components) / len(components),
        }

    @staticmethod
    def _correlation(first: Tensor, second: Tensor, kind: str) -> float | None:
        """Return one correlation while preserving constant-map undefinedness.

        Args:
            first: First aligned finite vector.
            second: Second aligned finite vector.
            kind: ``spearman`` for rank correlation or ``pearson`` for linear correlation.

        Returns:
            Correlation in ``[-1,1]`` or ``None`` when fewer than two values or a constant vector
            makes the statistic undefined.

        Raises:
            ValueError: If ``kind`` is not one of the two supported statistics.
        """
        if kind not in {"spearman", "pearson"}:
            raise ValueError("view correlation kind must be spearman or pearson")
        if len(first) < 2 or first.min() == first.max() or second.min() == second.max():
            return None

        statistic = (
            spearmanr(first.numpy(), second.numpy()).statistic
            if kind == "spearman"
            else pearsonr(first.numpy(), second.numpy()).statistic
        )
        value = float(statistic)
        return value if math.isfinite(value) else None

    @staticmethod
    def _top_overlap(first: Tensor, second: Tensor, fraction: float) -> float:
        """Measure overlap between equal-size high-probability sets.

        Args:
            first: First aligned probability vector.
            second: Second aligned probability vector.
            fraction: Retained fraction in ``(0,1]``.

        Returns:
            Intersection size divided by retained-set size, in ``[0,1]``.
        """
        count       = max(1, math.ceil(fraction * len(first)))
        first_top   = set(torch.topk(first, count).indices.tolist())
        second_top  = set(torch.topk(second, count).indices.tolist())
        return len(first_top & second_top) / count
