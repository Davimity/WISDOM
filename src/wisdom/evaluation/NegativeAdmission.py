"""Frozen-predictor negative-extreme calibration, distinct from training and probability fitting."""

import numpy as np

from typing import Any
from collections.abc import Mapping, Sequence
from wisdom.models.negative.SurfaceRegions import SurfaceRegions


class NegativeAdmission:
    """VN01 N3 finite-rank admission and development-only threshold control."""

    def __init__(self, maxima: Sequence[float], formal: bool = False) -> None:
        """Freeze one maximum operational logit per independent negative protein.

        Args:
            maxima: Finite scalar extremes from a fixed predictor, never per-region pseudo-samples.
            formal: Whether documented sampling/history also supports exchangeability. Disjoint
                IDs alone cannot establish this assumption; false labels the result exploratory.

        Raises:
            ValueError: No calibration negatives or invalid extrema.
        """
        self.maxima = np.asarray(maxima, dtype=np.float64)
        if self.maxima.ndim != 1 or not len(self.maxima) or not np.isfinite(self.maxima).all():
            raise ValueError("N3 requires finite maxima from a nonempty negative reserve")
        self.formal = formal

    def pvalues(self, scores: Sequence[float]) -> np.ndarray:
        """Calculate (1 + count(T_b >= score))/(n+1), preserving ties conservatively.

        Args:
            scores: Finite operational regional maxima; original predictor scores stay unchanged.

        Returns:
            Tail ranks [R] in [1/(n+1),1]. They are not biological probabilities or FDR estimates.
        """
        values = np.asarray(scores, dtype=np.float64)
        if not np.isfinite(values).all():
            raise ValueError("N3 regional maxima must be finite")
        ordered = np.sort(self.maxima)
        return (1 + len(ordered) - np.searchsorted(ordered, values, side="left")) / (
            len(ordered) + 1)

    @staticmethod
    def development_threshold(scores: Sequence[float], labels: Sequence[int]) -> float:
        """Fit F1's global threshold using development labels only, not superficial targets.

        Args:
            scores: Global protein logits from the fixed predictor on a development cohort.
                They equal the surface maximum only for single-head MAX pooling.
            labels: Corresponding global binary labels containing both classes.

        Returns:
            Threshold maximizing balanced accuracy; ties favor specificity, then larger threshold.

        Raises:
            ValueError: Missing classes, invalid scores, or inconsistent lengths.
        """
        values, target = np.asarray(scores), np.asarray(labels)
        if values.shape != target.shape or set(target.tolist()) != {0, 1} or \
                not np.isfinite(values).all():
            raise ValueError("N3 F1 threshold needs a finite development sample with both classes")
        candidates = [*np.unique(values), float(np.nextafter(values.max(), np.inf))]

        def quality(threshold: float) -> tuple[float, float, float]:
            """Rank a fixed development decision by balanced accuracy and specificity.

            Args:
                threshold: Candidate global-logit cutoff evaluated on development only.

            Returns:
                Balanced accuracy, specificity and cutoff for deterministic tie breaking.
            """
            prediction = values >= threshold
            recall = float(prediction[target == 1].mean())
            specificity = float((~prediction[target == 0]).mean())
            return (recall + specificity) / 2, specificity, threshold

        return float(max(candidates, key=quality))

    @staticmethod
    def verify_reserve(reserve: Sequence[Mapping[str, Any]],
                       used: Sequence[Mapping[str, Any]]) -> None:
        """Reject calibration overlap with any training, selection or evaluation population.

        Args:
            reserve: Negative-only rows with identifier and nonempty leakage_group.
            used: Every already used/evaluated protein under the same grouping protocol.

        Raises:
            ValueError: Empty reserve, any nonnegative row, missing group, repeated ID/group,
                or overlapping identifier/leakage group. Independence still needs history audit.
        """
        identifiers = [str(r["identifier"]) for r in reserve]
        groups = [str(r["leakage_group"]) for r in reserve]
        if (not reserve or any(int(r["label"]) != 0 for r in reserve)
                or not all(g and g != "unspecified" for g in groups)
                or len(set(identifiers)) != len(identifiers) or len(set(groups)) != len(groups)):
            raise ValueError("N3 needs one independent negative per nonempty leakage group")
        if set(identifiers) & {str(r["identifier"]) for r in used} or \
                set(groups) & {str(r["leakage_group"]) for r in used}:
            raise ValueError("N3 calibration overlaps training, model selection or evaluation")

    def summary(self, alpha: float) -> dict[str, Any]:
        """Describe finite resolution and the conditional scope of the proposed guarantee.

        Args:
            alpha: Predeclared tail-admission level in (0,1), not selected from positive local GT.

        Returns:
            Audit record that never calls an exploratory reference formally calibrated.
        """
        if not 0 < alpha < 1:
            raise ValueError("N3 alpha must lie in (0,1)")
        return {"alpha": alpha, "negative_count": len(self.maxima),
            "minimum_pvalue": 1 / (len(self.maxima) + 1),
            "admission_resolvable": alpha >= 1 / (len(self.maxima) + 1),
            "scope": ("marginal_under_exchangeability" if self.formal
                      else "exploratory_no_guarantee"),
            "ranking_scores_changed": False}

    def evaluate(self, snapshot: Mapping[str, Mapping[str, Any]],
                 geometry: Mapping[str, SurfaceRegions], levels: Sequence[float],
                 lower: float = .25, development_threshold: float | None = None) -> dict[str, Any]:
        """Apply F0/F1/F2 to the same fixed field and common lower-superlevel extension.

        Args:
            snapshot: Frozen evaluation logits/labels/area and optional hard local targets/masks.
            geometry: Fixed evaluation surface connectivity keyed by identifier.
            levels: Predeclared finite-rank admission levels, not tuned against evaluation targets.
            lower: Common extension probability cutoff; does not become the global decision cutoff.
            development_threshold: Optional F1 logit threshold selected on a separate cohort.

        Returns:
            Per-protein admission counts, proxy areas and local Dice/IoU/precision/recall with
            explicit unavailable values, plus negative any-admission rates and sensitivity.
            No predictor score or model parameter is replaced by a calibrated decision.
        """
        procedures: dict[str, tuple[str, float]] = {"F0": ("logit", 0.0)}
        if development_threshold is not None:
            procedures["F1"] = ("logit", development_threshold)
        procedures.update({f"F2_{alpha:g}": ("rank", alpha) for alpha in levels})
        output: dict[str, Any] = {"calibration": [self.summary(a) for a in levels],
            "F1_status": "available" if development_threshold is not None else
                         "unavailable_no_independent_development_cohort", "procedures": {}}
        for name, (kind, cutoff) in procedures.items():
            records = []
            for identifier, values in snapshot.items():
                logits = values["logits"].numpy()
                probability = values["logits"].sigmoid().numpy()
                regions = geometry[identifier].components(probability, lower)
                scores = [float(logits[r["anchor"]]) for r in regions]
                accepted = (self.pvalues(scores) <= cutoff if kind == "rank"
                            else np.asarray(scores) >= cutoff)
                mask = np.zeros(len(logits), dtype=bool)
                for region, keep in zip(regions, accepted, strict=True):
                    if keep:
                        mask[region["support"]] = True
                row: dict[str, Any] = {"identifier": identifier, "label": values["label"],
                    "operational_maximum": float(logits.max()),
                    "protein_logit": values["protein_logit"], "regions": int(accepted.sum()),
                    "protein_prediction": (bool(self.pvalues([float(logits.max())])[0] <= cutoff)
                        if kind == "rank" else values["protein_logit"] >= cutoff),
                    "any_admission": bool(mask.any()),
                    "area": float(geometry[identifier].weights[mask].sum())}
                for metric in ("dice", "iou", "precision", "recall"):
                    row[metric] = None
                if "local_target" in values:
                    valid = values["local_valid"].numpy().astype(bool)
                    truth = values["local_target"].numpy().astype(bool)
                    tp = int((mask & truth & valid).sum())
                    predicted = int((mask & valid).sum())
                    positive_points = int((truth & valid).sum())
                    union = predicted + positive_points - tp
                    total = predicted + positive_points
                    row.update(dice=2 * tp / total if total else None,
                        iou=tp / union if union else None,
                        precision=tp / predicted if predicted else None,
                        recall=tp / positive_points if positive_points else None)
                records.append(row)
            negatives = [r for r in records if r["label"] == 0]
            positives = [r for r in records if r["label"] == 1]
            output["procedures"][name] = {"cutoff_kind": kind, "cutoff": cutoff,
                "negative_count": len(negatives), "positive_count": len(positives),
                "false_admission_rate": sum(r["any_admission"] for r in negatives) / len(negatives)
                    if negatives else None,
                "sensitivity": sum(r["any_admission"] for r in positives) / len(positives)
                    if positives else None, "records": records}
            output["procedures"][name]["global"] = self.binary_metrics(records)
        return output

    @staticmethod
    def binary_metrics(records: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
        """Summarize thresholded decisions without overwriting the model's ranking evidence.

        Args:
            records: Fixed protein labels and protein_prediction decisions for one procedure.

        Returns:
            Accuracy, balanced accuracy, precision, recall, specificity, F1 and MCC. Zero
            denominators stay unavailable; ranking AUPRC/AUROC deliberately remain untouched.
        """
        target = np.asarray([r["label"] for r in records], dtype=bool)
        prediction = np.asarray([r["protein_prediction"] for r in records], dtype=bool)
        tp, fp = int((prediction & target).sum()), int((prediction & ~target).sum())
        tn, fn = int((~prediction & ~target).sum()), int((~prediction & target).sum())
        recall = tp / (tp + fn) if tp + fn else None
        specificity = tn / (tn + fp) if tn + fp else None
        denominator = ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) ** .5
        return {"accuracy": (tp + tn) / len(records) if records else None,
            "balanced_accuracy": (recall + specificity) / 2
                if recall is not None and specificity is not None else None,
            "precision": tp / (tp + fp) if tp + fp else None,
            "recall": recall, "specificity": specificity,
            "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None,
            "mcc": (tp * tn - fp * fn) / denominator if denominator else None}
