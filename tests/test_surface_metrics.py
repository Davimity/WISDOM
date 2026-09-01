"""Scientific tests for evaluation-only WISDOM surface metrics."""

import pytest
import torch

from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite
from wisdom.evaluation.SurfaceMetricSuite import SurfaceMetricSuite


def test_binary_metric_suite_exposes_mcc_for_composite_hpo() -> None:
    """Perfect binary decisions expose MCC under the exact ``val_mcc`` source name."""
    metrics = BinaryMetricSuite().compute(
        torch.tensor([0.95, 0.80, 0.20, 0.05]),
        torch.tensor([1, 1, 0, 0]),
    )

    assert metrics["mcc"] == pytest.approx(1.0)


def test_binary_metric_subset_avoids_unrequested_surface_work() -> None:
    """Surface evaluation can request its four metrics without computing the protein-only suite."""
    metrics = BinaryMetricSuite().compute(
        torch.tensor([0.9, 0.8, 0.2, 0.1]),
        torch.tensor([1, 1, 0, 0]),
        SurfaceMetricSuite.METRIC_NAMES,
    )

    assert set(metrics) == set(SurfaceMetricSuite.METRIC_NAMES)
    assert all(value == pytest.approx(1.0) for value in metrics.values())


def test_surface_metrics_separate_point_micro_and_positive_protein_macro() -> None:
    """Perfect rankings remain perfect under both documented aggregation views."""
    metrics = SurfaceMetricSuite().compute(
        probabilities=torch.tensor([0.95, 0.80, 0.10, 0.90, 0.20, 0.30, 0.10, 0.99]),
        targets=torch.tensor([1, 1, 0, 1, 0, 0, 0, 0]),
        valid_mask=torch.tensor([True, True, True, True, True, True, True, False]),
        surface_batch=torch.tensor([0, 0, 0, 1, 1, 2, 2, 2]),
        protein_targets=torch.tensor([1, 1, 0]),
    )

    assert metrics["surface_valid_points"] == 7.0
    assert metrics["surface_positive_proteins"] == 2.0
    for name in SurfaceMetricSuite.METRIC_NAMES:
        assert metrics[f"surface_micro_{name}"] == pytest.approx(1.0)
        assert metrics[f"surface_positive_macro_{name}"] == pytest.approx(1.0)


def test_surface_metrics_keep_undefined_positive_macro_values_explicit() -> None:
    """An all-negative benchmark does not invent a positive-protein localization score."""
    metrics = SurfaceMetricSuite().compute(
        probabilities=torch.tensor([0.2, 0.1]),
        targets=torch.tensor([0, 0]),
        valid_mask=torch.tensor([True, True]),
        surface_batch=torch.tensor([0, 0]),
        protein_targets=torch.tensor([0]),
    )

    assert metrics["surface_positive_proteins"] == 0.0
    assert metrics["surface_positive_macro_auprc"] is None
    assert metrics["surface_micro_auprc"] is None
