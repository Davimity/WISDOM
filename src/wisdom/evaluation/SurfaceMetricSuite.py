"""Definition-aware diagnostics for weakly supervised surface predictions."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from scipy.stats import spearmanr

from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite


class SurfaceMetricSuite:
    """Compare local model scores with evaluation-only DNA surface targets."""

    METRIC_NAMES = ("auprc", "auroc", "balanced_accuracy", "f1")
    TOP_FRACTIONS = (0.05, 0.10, 0.25)

    def compute(
        self,
        probabilities         : Tensor,
        targets               : Tensor,
        valid_mask            : Tensor,
        surface_batch         : Tensor,
        protein_targets       : Tensor,
        surface_area_weights  : Tensor,
        protein_probabilities: Tensor | None = None,
        attention_weights     : Tensor | None = None,
        head_disagreement     : Tensor | None = None,
    ) -> dict[str, float | None]:
        """Compute pooled-point and per-positive-protein localization diagnostics.

        ``surface_micro_*`` pools every unambiguous point before measuring performance. It tests
        whether local scores remain low on curated negative proteins as well as whether they find
        positive interfaces, but proteins with more points contribute more observations.
        ``surface_positive_macro_*`` first measures each globally positive protein independently
        and then takes an arithmetic mean, so every evaluable positive protein has equal weight.
        Top-fraction recall asks which fraction of the true interface lies among the highest-scored
        5%, 10%, or 25% of valid points. Enrichment divides the selected positive-point rate by the
        protein's overall positive-point rate; one is random ranking and values above one indicate
        concentration around the interface. Normalized AUPRC subtracts each positive protein's
        random-ranking baseline (its interface prevalence) and divides by the remaining headroom.

        Globally negative proteins have no positive interface by the benchmark definition. Their
        area-weighted mean local probability therefore measures false positive mass, while their
        maximum probability measures the strongest false hotspot. Both are averaged per protein so
        surface discretization size cannot make one protein dominate the result.

        The point predictions are never transformed into a loss here. AUPRC and AUROC assess score
        ranking without choosing a threshold. Balanced accuracy and F1 apply probability threshold
        0.5; they are useful calibration diagnostics but are not model-selection objectives.

        Args:
            probabilities: Local sigmoid scores with finite shape ``[M]`` and values in ``[0,1]``.
            targets: Hard DNA-interface targets with binary shape ``[M]``.
            valid_mask: Boolean shape ``[M]`` excluding the physical ambiguity band and unavailable
                local ground truth.
            surface_batch: Integer owner of every point with shape ``[M]`` and values in ``[0,B)``.
            protein_targets: Global binary labels with shape ``[B]``.
            surface_area_weights: Positive represented-area weights with shape ``[M]``. They are
                normalized independently within every protein before negative mass is computed.
            protein_probabilities: Optional global sigmoid scores ``[B]``. When supplied, their
                Spearman rank correlation with per-positive-protein surface AUPRC is reported.
            attention_weights: Optional point attention weights ``[M]`` produced by attention
                pooling. When supplied, the suite measures whether attention itself identifies the
                interface and whether it agrees with the local evidence logits.
            head_disagreement: Optional absolute direct-versus-surface protein probability gap
                ``[B]`` used only as a ground-truth-free uncertainty signal.

        Returns:
            Micro and positive-protein macro ranking metrics, normalized AUPRC, top-fraction
            recall/enrichment, negative false-positive mass/peak, optional attention diagnostics,
            confidence-to-surface Spearman correlation, and evaluable counts. Undefined metrics
            remain ``None``.

        Raises:
            ValueError: If arrays are misaligned, non-finite, non-binary, or reference an invalid
                protein owner.
        """
        scores     = probabilities.detach().view(-1).float().cpu()
        truth      = targets.detach().view(-1).long().cpu()
        valid      = valid_mask.detach().view(-1).bool().cpu()
        owners     = surface_batch.detach().view(-1).long().cpu()
        proteins   = protein_targets.detach().view(-1).long().cpu()
        area       = surface_area_weights.detach().view(-1).float().cpu()
        attentions = (
            attention_weights.detach().view(-1).float().cpu()
            if attention_weights is not None
            else None
        )
        confidence = (
            protein_probabilities.detach().view(-1).float().cpu()
            if protein_probabilities is not None
            else None
        )
        disagreement = (
            head_disagreement.detach().view(-1).float().cpu()
            if head_disagreement is not None
            else None
        )

        point_count = len(scores)
        if point_count == 0 or any(
            len(values) != point_count
            for values in (truth, valid, owners, area)
        ):
            raise ValueError("surface metrics require non-empty aligned point arrays")
        if not torch.isfinite(scores).all() or torch.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("surface probabilities must be finite values in [0,1]")
        if not torch.all((truth == 0) | (truth == 1)):
            raise ValueError("surface targets must contain only zero and one")
        if not torch.isfinite(area).all() or torch.any(area <= 0.0):
            raise ValueError("surface area weights must be finite and positive")
        if not len(proteins) or not torch.all((proteins == 0) | (proteins == 1)):
            raise ValueError("protein targets must be a non-empty binary vector")
        if confidence is not None and (
            confidence.shape != proteins.shape
            or not torch.isfinite(confidence).all()
            or torch.any((confidence < 0.0) | (confidence > 1.0))
        ):
            raise ValueError("protein probabilities must be finite aligned values in [0,1]")
        if attentions is not None and (
            attentions.shape != scores.shape
            or not torch.isfinite(attentions).all()
            or torch.any(attentions < 0.0)
        ):
            raise ValueError("attention weights must be finite aligned non-negative values")
        if disagreement is not None and (
            disagreement.shape != proteins.shape
            or not torch.isfinite(disagreement).all()
            or torch.any(disagreement < 0.0)
        ):
            raise ValueError("head disagreement must be finite, non-negative, and protein-aligned")
        if owners.min() < 0 or owners.max() >= len(proteins):
            raise ValueError("surface point owner is outside the protein target vector")
        if not torch.all(owners[1:] >= owners[:-1]):
            raise ValueError("surface point owners must remain grouped in protein order")

        binary_suite = BinaryMetricSuite()
        output       = self._empty_output()
        valid_count  = int(valid.sum())

        output["surface_valid_points"] = float(valid_count)
        if valid_count:
            micro = binary_suite.compute(scores[valid], truth[valid], self.METRIC_NAMES)
            for name in self.METRIC_NAMES:
                output[f"surface_micro_{name}"] = micro[name]

        per_protein: dict[str, list[float]] = {
            name: []
            for name in self.METRIC_NAMES
        }
        top_recall             : dict[float, list[float]] = {
            fraction: [] for fraction in self.TOP_FRACTIONS
        }
        enrichment             : dict[float, list[float]] = {
            fraction: [] for fraction in self.TOP_FRACTIONS
        }
        positive_confidence       : list[float] = []
        positive_surface_auprc   : list[float] = []
        normalized_auprc         : list[float] = []
        negative_positive_mass  : list[float] = []
        negative_peak           : list[float] = []
        attention_auprc         : list[float] = []
        attention_logit_spearman: list[float] = []
        attention_entropy       : list[float] = []
        positive_map_entropy    : list[float] = []
        positive_head_disagreement: list[float] = []
        owner_counts = torch.bincount(owners, minlength=len(proteins))
        owner_starts = torch.cat((torch.zeros(1, dtype=torch.long), owner_counts.cumsum(dim=0)))

        for protein_id in range(len(proteins)):
            start          = int(owner_starts[protein_id])
            stop           = int(owner_starts[protein_id + 1])
            protein_valid  = valid[start:stop]
            if not protein_valid.any():
                continue

            protein_scores = scores[start:stop][protein_valid]
            protein_truth  = truth[start:stop][protein_valid]
            protein_area   = area[start:stop][protein_valid]
            protein_area   = protein_area / protein_area.sum()

            if proteins[protein_id] == 0:
                negative_positive_mass.append(float((protein_area * protein_scores).sum()))
                negative_peak.append(float(protein_scores.max()))
                continue

            metrics = binary_suite.compute(
                protein_scores,
                protein_truth,
                self.METRIC_NAMES,
            )
            if metrics["auprc"] is None:
                continue

            for name in self.METRIC_NAMES:
                value = metrics[name]
                if value is not None:
                    per_protein[name].append(value)

            positives      = int(protein_truth.sum())
            prevalence     = positives / len(protein_truth)
            bounded_probability = protein_scores.clamp(1.0e-7, 1.0 - 1.0e-7)
            probability_entropy = -(
                bounded_probability * bounded_probability.log()
                + (1.0 - bounded_probability) * (1.0 - bounded_probability).log()
            ) / math.log(2.0)
            positive_map_entropy.append(float((protein_area * probability_entropy).sum()))
            positive_surface_auprc.append(float(metrics["auprc"]))
            if disagreement is not None:
                positive_head_disagreement.append(float(disagreement[protein_id]))
            if prevalence < 1.0:
                normalized_auprc.append(
                    (float(metrics["auprc"]) - prevalence) / (1.0 - prevalence)
                )

            for fraction in self.TOP_FRACTIONS:
                selected_count = max(1, math.ceil(fraction * len(protein_truth)))
                selected       = torch.topk(
                    protein_scores,
                    selected_count,
                    sorted=False,
                ).indices
                selected_positives = int(protein_truth[selected].sum())
                top_recall[fraction].append(selected_positives / positives)
                enrichment[fraction].append(
                    (selected_positives / selected_count) / prevalence
                )

            if confidence is not None:
                positive_confidence.append(float(confidence[protein_id]))

            if attentions is not None:
                protein_attention = attentions[start:stop][protein_valid]
                attention_metric  = binary_suite.compute(
                    protein_attention,
                    protein_truth,
                    ("auprc",),
                )["auprc"]
                if attention_metric is not None:
                    attention_auprc.append(attention_metric)

                correlation = self._spearman(protein_scores, protein_attention)
                if correlation is not None:
                    attention_logit_spearman.append(correlation)

                attention_sum = protein_attention.sum()
                if attention_sum > 0.0:
                    normalized_attention = protein_attention / attention_sum
                    entropy = -torch.sum(
                        normalized_attention
                        * normalized_attention.clamp_min(torch.finfo(torch.float32).eps).log()
                    )
                    maximum_entropy = math.log(len(normalized_attention))
                    attention_entropy.append(
                        float(entropy) / maximum_entropy if maximum_entropy > 0.0 else 0.0
                    )

        evaluated = len(per_protein["auprc"])
        output["surface_positive_proteins"] = float(evaluated)
        for name, values in per_protein.items():
            if values:
                output[f"surface_positive_macro_{name}"] = math.fsum(values) / len(values)
        if normalized_auprc:
            output["surface_positive_macro_normalized_auprc"] = (
                math.fsum(normalized_auprc) / len(normalized_auprc)
            )

        output["surface_negative_proteins"] = float(len(negative_positive_mass))
        if negative_positive_mass:
            output["surface_negative_positive_mass"] = (
                math.fsum(negative_positive_mass) / len(negative_positive_mass)
            )
            output["surface_negative_peak"] = math.fsum(negative_peak) / len(negative_peak)

        if attention_auprc:
            output["surface_attention_positive_macro_auprc"] = (
                math.fsum(attention_auprc) / len(attention_auprc)
            )
        if attention_logit_spearman:
            output["surface_attention_logit_spearman"] = (
                math.fsum(attention_logit_spearman) / len(attention_logit_spearman)
            )
        if attention_entropy:
            output["surface_attention_entropy"] = (
                math.fsum(attention_entropy) / len(attention_entropy)
            )

        for fraction in self.TOP_FRACTIONS:
            label = int(100 * fraction)
            if top_recall[fraction]:
                output[f"surface_positive_macro_top_{label}_recall"] = (
                    math.fsum(top_recall[fraction]) / len(top_recall[fraction])
                )
                output[f"surface_positive_macro_top_{label}_enrichment"] = (
                    math.fsum(enrichment[fraction]) / len(enrichment[fraction])
                )

        if (
            len(positive_confidence) >= 2
            and len(set(positive_confidence)) > 1
            and len(set(positive_surface_auprc)) > 1
        ):
            correlation = float(spearmanr(positive_confidence, positive_surface_auprc).statistic)
            output["surface_confidence_quality_spearman"] = (
                correlation if math.isfinite(correlation) else None
            )

        self._selective_metrics(
            output,
            positive_surface_auprc,
            positive_map_entropy,
            "map_entropy",
        )
        if positive_head_disagreement:
            self._selective_metrics(
                output,
                positive_surface_auprc,
                positive_head_disagreement,
                "head_disagreement",
            )
        return output

    @staticmethod
    def _selective_metrics(
        output     : dict[str, float | None],
        quality    : list[float],
        uncertainty: list[float],
        name       : str,
    ) -> None:
        """Measure whether one local-GT-free signal identifies unreliable surface maps.

        Args:
            output: Public metric mapping updated in place.
            quality: Per-positive-protein surface AUPRC values.
            uncertainty: Aligned scores where larger means less confidence.
            name: Stable metric-family prefix.
        """
        if len(quality) != len(uncertainty) or not quality:
            return

        uncertainty_tensor = torch.tensor(uncertainty)
        quality_tensor     = torch.tensor(quality)
        error_correlation  = SurfaceMetricSuite._spearman(
            uncertainty_tensor,
            1.0 - quality_tensor,
        )
        output[f"surface_{name}_error_spearman"] = error_correlation

        order = torch.argsort(uncertainty_tensor)
        for coverage in (0.90, 0.80, 0.70):
            retained = max(1, math.ceil(coverage * len(order)))
            label    = round(100 * coverage)
            output[f"surface_{name}_auprc_at_coverage_{label}"] = float(
                quality_tensor[order[:retained]].mean()
            )

    @staticmethod
    def _spearman(first: Tensor, second: Tensor) -> float | None:
        """Measure finite rank agreement between two aligned point fields.

        Args:
            first: First one-dimensional point field.
            second: Second aligned one-dimensional point field.

        Returns:
            Spearman correlation in ``[-1,1]``, or ``None`` when either field is constant or has
            fewer than two observations.
        """
        if len(first) < 2 or len(torch.unique(first)) < 2 or len(torch.unique(second)) < 2:
            return None

        value = float(spearmanr(first.numpy(), second.numpy()).statistic)
        return value if math.isfinite(value) else None

    def _empty_output(self) -> dict[str, float | None]:
        """Create the stable local-metric schema before definition-aware calculation.

        Returns:
            Mapping containing every public surface metric as ``None`` and both counts as zero.
        """
        output: dict[str, float | None] = {
            "surface_valid_points":                    0.0,
            "surface_positive_proteins":               0.0,
            "surface_negative_proteins":               0.0,
            "surface_positive_macro_normalized_auprc": None,
            "surface_negative_positive_mass":          None,
            "surface_negative_peak":                   None,
            "surface_attention_positive_macro_auprc":  None,
            "surface_attention_logit_spearman":        None,
            "surface_attention_entropy":               None,
            "surface_map_entropy_error_spearman":       None,
            "surface_head_disagreement_error_spearman": None,
        }
        for name in self.METRIC_NAMES:
            output[f"surface_micro_{name}"]          = None
            output[f"surface_positive_macro_{name}"] = None
        for fraction in self.TOP_FRACTIONS:
            label = int(100 * fraction)
            output[f"surface_positive_macro_top_{label}_recall"]     = None
            output[f"surface_positive_macro_top_{label}_enrichment"] = None
        for name in ("map_entropy", "head_disagreement"):
            for coverage in (90, 80, 70):
                output[f"surface_{name}_auprc_at_coverage_{coverage}"] = None
        output["surface_confidence_quality_spearman"] = None
        return output
