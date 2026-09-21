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
        surface_area_weights=torch.ones(8),
        protein_probabilities=torch.tensor([0.95, 0.80, 0.10]),
    )

    assert metrics["surface_valid_points"] == 7.0
    assert metrics["surface_positive_proteins"] == 2.0
    for name in SurfaceMetricSuite.METRIC_NAMES:
        assert metrics[f"surface_micro_{name}"] == pytest.approx(1.0)
        assert metrics[f"surface_positive_macro_{name}"] == pytest.approx(1.0)
    for fraction in (5, 10, 25):
        assert metrics[f"surface_positive_macro_top_{fraction}_recall"] > 0.0
        assert metrics[f"surface_positive_macro_top_{fraction}_enrichment"] >= 1.0
    assert metrics["surface_confidence_quality_spearman"] is None
    assert metrics["surface_positive_macro_normalized_auprc"] == pytest.approx(1.0)
    assert metrics["surface_negative_positive_mass"] == pytest.approx(0.2)
    assert metrics["surface_negative_peak"] == pytest.approx(0.3)


def test_surface_metrics_relate_protein_confidence_to_localization_quality() -> None:
    """Positive-protein confidence is rank-correlated with independently measured surface AUPRC."""
    metrics = SurfaceMetricSuite().compute(
        probabilities=torch.tensor([0.90, 0.10, 0.20, 0.80]),
        targets=torch.tensor([1, 0, 1, 0]),
        valid_mask=torch.ones(4, dtype=torch.bool),
        surface_batch=torch.tensor([0, 0, 1, 1]),
        protein_targets=torch.tensor([1, 1]),
        surface_area_weights=torch.ones(4),
        protein_probabilities=torch.tensor([0.90, 0.60]),
    )

    assert metrics["surface_confidence_quality_spearman"] == pytest.approx(1.0)


def test_surface_metrics_keep_undefined_positive_macro_values_explicit() -> None:
    """An all-negative benchmark does not invent a positive-protein localization score."""
    metrics = SurfaceMetricSuite().compute(
        probabilities=torch.tensor([0.2, 0.1]),
        targets=torch.tensor([0, 0]),
        valid_mask=torch.tensor([True, True]),
        surface_batch=torch.tensor([0, 0]),
        protein_targets=torch.tensor([0]),
        surface_area_weights=torch.tensor([1.0, 3.0]),
    )

    assert metrics["surface_positive_proteins"] == 0.0
    assert metrics["surface_positive_macro_auprc"] is None
    assert metrics["surface_micro_auprc"] is None
    assert metrics["surface_negative_positive_mass"] == pytest.approx(0.125)
    assert metrics["surface_negative_peak"] == pytest.approx(0.2)


def test_surface_metrics_diagnose_attention_independently_from_local_logits() -> None:
    """Attention pooling reports whether its weights identify and agree with local evidence."""
    metrics = SurfaceMetricSuite().compute(
        probabilities=torch.tensor([0.9, 0.7, 0.2, 0.1]),
        targets=torch.tensor([1, 1, 0, 0]),
        valid_mask=torch.ones(4, dtype=torch.bool),
        surface_batch=torch.zeros(4, dtype=torch.long),
        protein_targets=torch.tensor([1]),
        surface_area_weights=torch.ones(4),
        attention_weights=torch.tensor([0.4, 0.3, 0.2, 0.1]),
    )

    assert metrics["surface_attention_positive_macro_auprc"] == pytest.approx(1.0)
    assert metrics["surface_attention_logit_spearman"] == pytest.approx(1.0)
    assert metrics["surface_attention_entropy"] is not None


def test_surface_metrics_measure_selective_localization_uncertainty() -> None:
    """Head disagreement and map entropy expose validation risk-coverage diagnostics."""
    metrics = SurfaceMetricSuite().compute(
        probabilities=torch.tensor([0.9, 0.1, 0.5, 0.5, 0.8, 0.2, 0.6, 0.7]),
        targets=torch.tensor([1, 0, 0, 1, 1, 0, 1, 0]),
        valid_mask=torch.ones(8, dtype=torch.bool),
        surface_batch=torch.tensor([0, 0, 1, 1, 2, 2, 3, 3]),
        protein_targets=torch.ones(4, dtype=torch.long),
        surface_area_weights=torch.ones(8),
        head_disagreement=torch.tensor([0.1, 0.8, 0.2, 0.9]),
    )

    assert metrics["surface_head_disagreement_error_spearman"] > 0.0
    assert metrics["surface_head_disagreement_auprc_at_coverage_70"] is not None
    assert metrics["surface_map_entropy_auprc_at_coverage_70"] is not None
