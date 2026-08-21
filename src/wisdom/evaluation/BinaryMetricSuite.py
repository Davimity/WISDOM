"""Definition-aware use of LambdaForge binary classification metrics."""

from __future__ import annotations

import math
from collections.abc import Callable

import torch
from lambdaforge.metrics import (
    BinaryAccuracy,
    BinaryAUPRC,
    BinaryAUROC,
    BinaryBalancedAccuracy,
    BinaryCohenKappa,
    BinaryF1,
    BinaryMCC,
    BinaryPrecision,
    BinaryRecall,
    BinarySpecificity,
)
from lambdaforge.metrics.Metric import Metric
from torch import Tensor


class BinaryMetricSuite:
    """Compute required metrics while preserving mathematically undefined values."""

    def __init__(self, threshold: float = 0.5) -> None:
        """Set the explicit probability threshold for discrete predictions.

        Args:
            threshold: Probability at or above which a prediction is positive.

        Raises:
            ValueError: If the threshold does not lie strictly between zero and one.
        """
        if not 0.0 < threshold < 1.0:
            raise ValueError("classification threshold must lie in (0,1)")
        self.threshold = float(threshold)

    def compute(self, probabilities: Tensor, targets: Tensor) -> dict[str, float | None]:
        """Run public LambdaForge metrics only where their denominators are defined.

        Args:
            probabilities: Finite continuous scores with shape ``[N]`` in ``[0,1]``.
            targets: Binary integer or Boolean values with shape ``[N]``.

        Returns:
            Mapping of accuracy, balanced accuracy, precision, recall, specificity, F1, MCC,
            Cohen's kappa, AUROC, and AUPRC. Undefined metrics are JSON ``null``, never zero.

        Raises:
            ValueError: If inputs are empty, misaligned, non-finite, non-probabilistic, or
                nonbinary.
        """
        scores = probabilities.detach().view(-1).float().cpu()
        truth  = targets.detach().view(-1).long().cpu()
        if not len(scores) or scores.shape != truth.shape:
            raise ValueError("binary metrics require non-empty aligned vectors")
        if not torch.isfinite(scores).all() or torch.any((scores < 0.0) | (scores > 1.0)):
            raise ValueError("binary metric scores must be finite probabilities")
        if not torch.all((truth == 0) | (truth == 1)):
            raise ValueError("binary metric targets must contain only zero and one")

        predicted = scores >= self.threshold
        positive  = truth == 1
        negative  = ~positive
        tp        = int((predicted & positive).sum())
        tn        = int((~predicted & negative).sum())
        fp        = int((predicted & negative).sum())
        fn        = int((~predicted & positive).sum())
        has_both_classes = bool(positive.any() and negative.any())

        constructors: dict[str, tuple[Callable[[], Metric], bool]] = {
            "accuracy": (lambda: BinaryAccuracy("probability", "target", self.threshold), True),
            "balanced_accuracy": (
                lambda: BinaryBalancedAccuracy("probability", "target", self.threshold),
                has_both_classes,
            ),
            "precision": (
                lambda: BinaryPrecision("probability", "target", self.threshold),
                tp + fp > 0,
            ),
            "recall": (
                lambda: BinaryRecall("probability", "target", self.threshold),
                tp + fn > 0,
            ),
            "specificity": (
                lambda: BinarySpecificity("probability", "target", self.threshold),
                tn + fp > 0,
            ),
            "f1": (
                lambda: BinaryF1("probability", "target", self.threshold),
                2 * tp + fp + fn > 0,
            ),
            "mcc": (
                lambda: BinaryMCC("probability", "target", self.threshold),
                (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn) > 0,
            ),
            "cohen_kappa": (
                lambda: BinaryCohenKappa("probability", "target", self.threshold),
                self._kappa_defined(tp, tn, fp, fn),
            ),
            "auroc": (lambda: BinaryAUROC("probability", "target"), has_both_classes),
            "auprc": (lambda: BinaryAUPRC("probability", "target"), has_both_classes),
        }
        output: dict[str, float | None] = {}
        metric_outputs = {"probability": scores}
        metric_batch   = {"target": truth}
        for name, (constructor, defined) in constructors.items():
            if not defined:
                output[name] = None
                continue
            metric = constructor()
            update = metric.update
            compute = metric.compute
            update(metric_outputs, metric_batch)
            value        = float(compute())
            output[name] = value if math.isfinite(value) else None
        return output

    @staticmethod
    def undefined() -> dict[str, float | None]:
        """Return the complete metric schema for an empty evaluable subset.

        Returns:
            Every supported metric mapped to ``None`` so an absent subset is explicit rather than
            silently replaced by zero or omitted from a report.
        """
        return {
            name: None
            for name in (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "specificity",
                "f1",
                "mcc",
                "cohen_kappa",
                "auroc",
                "auprc",
            )
        }

    @staticmethod
    def _kappa_defined(tp: int, tn: int, fp: int, fn: int) -> bool:
        """Test whether Cohen's kappa expected-agreement denominator is non-zero.

        Args:
            tp: True-positive count.
            tn: True-negative count.
            fp: False-positive count.
            fn: False-negative count.

        Returns:
            True when sample count is positive and expected agreement differs from one.
        """
        count = tp + tn + fp + fn
        if count == 0:
            return False
        expected = ((tp + fp) * (tp + fn) + (fn + tn) * (fp + tn)) / (count * count)
        return not math.isclose(expected, 1.0)
