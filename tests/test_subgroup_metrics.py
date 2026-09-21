"""Tests for model-performance strata that expose dataset shortcuts."""

import pytest
import torch

from wisdom.evaluation.SubgroupMetricSuite import SubgroupMetricSuite
from wisdom.Training import _subgroup_coupling


def test_subgroup_metrics_preserve_definition_aware_scores() -> None:
    """Size and phenotype groups expose G/S while positive-only prevalence keeps G undefined."""
    labels        = torch.tensor([0, 1, 0, 1, 0, 1, 0, 1])
    probabilities = torch.tensor([0.05, 0.95, 0.15, 0.85, 0.25, 0.75, 0.35, 0.65])
    atoms         = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])
    point_counts  = torch.tensor([4, 5, 6, 7, 8, 9, 10, 11])
    owners        = torch.repeat_interleave(torch.arange(8), point_counts)

    local_targets: list[int] = []
    local_scores : list[float] = []
    for protein, count in enumerate(point_counts.tolist()):
        positive = bool(labels[protein])
        local_targets.extend([1 if positive and point == 0 else 0 for point in range(count)])
        local_scores.extend([0.9 if positive and point == 0 else 0.1 for point in range(count)])

    metrics = SubgroupMetricSuite().compute(
        probabilities,
        labels,
        atoms,
        point_counts,
        torch.tensor(local_scores),
        torch.tensor(local_targets),
        torch.ones(len(owners), dtype=torch.bool),
        owners,
        ["G_SHARED"] * 8,
        ["not_applicable", "I_SHARED"] * 4,
        ["core"] * 8,
    )

    assert metrics["subgroup_atom_count_q1_count"] == pytest.approx(2.0)
    assert metrics["subgroup_global_phenotype_g_shared_global_score"] == pytest.approx(1.0)
    assert metrics["subgroup_global_phenotype_g_shared_surface_auprc"] == pytest.approx(1.0)
    assert metrics["subgroup_surface_prevalence_q1_global_score"] is None
    assert metrics["subgroup_interface_phenotype_i_shared_surface_count"] == pytest.approx(4.0)


def test_subgroup_coupling_tracks_each_stratum_without_affecting_global_selection() -> None:
    """Independent subgroup curves emit W and replace duplicate best-checkpoint epochs safely."""
    curves: dict[str, list[dict[str, float]]] = {}
    first = _subgroup_coupling(
        {
            "subgroup_atom_count_q1_global_score": 0.60,
            "subgroup_atom_count_q1_surface_auprc": 0.50,
        },
        curves,
        epoch=1,
    )
    second = _subgroup_coupling(
        {
            "subgroup_atom_count_q1_global_score": 0.70,
            "subgroup_atom_count_q1_surface_auprc": 0.65,
        },
        curves,
        epoch=1,
    )

    assert first["subgroup_atom_count_q1_wisdom_score"] == pytest.approx(0.605)
    assert len(curves["subgroup_atom_count_q1"]) == 1
    assert second["subgroup_atom_count_q1_global_at_selection"] == pytest.approx(0.70)
    assert second["subgroup_atom_count_q1_surface_at_global_selection"] == pytest.approx(0.65)
