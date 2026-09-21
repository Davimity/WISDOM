"""Tests for formal V2 surface-view correspondence and consistency metrics."""

import pytest
import torch

from wisdom.evaluation.SurfaceViewCorrespondence import SurfaceViewCorrespondence
from wisdom.evaluation.ViewConsistencyMetricSuite import ViewConsistencyMetricSuite


def test_surface_view_correspondence_uses_geometry_normals_and_mutuality() -> None:
    """Only bounded, normal-compatible, mutually nearest physical points are paired."""
    reference_positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [4.0, 0.0, 0.0]]
    )
    reference_normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]]
    )
    view_positions = torch.tensor(
        [[0.05, 0.0, 0.0], [1.05, 0.0, 0.0], [4.05, 0.0, 0.0]]
    )
    view_normals = torch.tensor(
        [[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]
    )

    correspondence = SurfaceViewCorrespondence().align(
        reference_positions,
        reference_normals,
        view_positions,
        view_normals,
        maximum_distance=0.2,
        minimum_normal_cosine=0.5,
    )

    assert correspondence["reference_indices"].tolist() == [0, 2]
    assert correspondence["view_indices"].tolist() == [0, 2]
    assert correspondence["distances"].tolist() == pytest.approx([0.05, 0.05], abs=1.0e-6)
    assert correspondence["normal_cosines"].tolist() == pytest.approx([1.0, 1.0])


def test_identical_aligned_maps_have_unit_consistency() -> None:
    """Every bounded agreement component reaches one for an identical prediction map."""
    logits = torch.linspace(-3.0, 3.0, 20)
    indices = torch.arange(20)
    metrics = ViewConsistencyMetricSuite().compute(
        logits,
        logits.clone(),
        {"reference_indices": indices, "view_indices": indices},
    )

    assert metrics["view_logit_spearman"] == pytest.approx(1.0)
    assert metrics["view_probability_pearson"] == pytest.approx(1.0)
    assert metrics["view_map_js_divergence"] == pytest.approx(0.0)
    assert metrics["view_map_total_variation"] == pytest.approx(0.0)
    assert metrics["view_top_05_overlap"] == pytest.approx(1.0)
    assert metrics["view_top_10_overlap"] == pytest.approx(1.0)
    assert metrics["view_regional_probability_variance"] == pytest.approx(0.0)
    assert metrics["view_surface_consistency"] == pytest.approx(1.0)


def test_opposite_aligned_maps_expose_disagreement() -> None:
    """Reversing a non-constant map reduces the composite and separates hotspot sets."""
    first   = torch.linspace(-4.0, 4.0, 40)
    second  = -first
    indices = torch.arange(40)
    metrics = ViewConsistencyMetricSuite().compute(
        first,
        second,
        {"reference_indices": indices, "view_indices": indices},
    )

    assert metrics["view_logit_spearman"] == pytest.approx(-1.0)
    assert metrics["view_probability_pearson"] == pytest.approx(-1.0)
    assert metrics["view_top_05_overlap"] == pytest.approx(0.0)
    assert metrics["view_top_10_overlap"] == pytest.approx(0.0)
    assert metrics["view_surface_consistency"] < 0.5


def test_constant_maps_preserve_undefined_correlations() -> None:
    """A constant pair keeps correlations unavailable without losing its valid agreement score."""
    logits  = torch.zeros(8)
    indices = torch.arange(8)
    metrics = ViewConsistencyMetricSuite().compute(
        logits,
        logits,
        {"reference_indices": indices, "view_indices": indices},
    )

    assert metrics["view_logit_spearman"] is None
    assert metrics["view_probability_pearson"] is None
    assert metrics["view_surface_consistency"] == pytest.approx(1.0)
