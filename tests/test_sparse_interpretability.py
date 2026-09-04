"""Focused contracts for post-HPO sparse concept discovery."""

# ruff: noqa: I001

from __future__ import annotations

import json
import torch

from pathlib import Path

from wisdom.interpretability.EmbeddingScaler import EmbeddingScaler
from wisdom.interpretability.SparseConceptDiscovery import SparseConceptDiscovery
from wisdom.interpretability.SparseConceptModel import SparseConceptModel


def test_embedding_scaler_uses_supplied_training_population_only() -> None:
    """Validation outliers cannot alter statistics fitted explicitly from train."""
    training   = torch.tensor([[1.0, 2.0], [3.0, 6.0]])
    validation = torch.tensor([[1000.0, -1000.0]])
    scaler     = EmbeddingScaler.fit(training)

    assert torch.allclose(scaler.mean, torch.tensor([2.0, 4.0]))
    assert torch.allclose(scaler.transform(training).mean(dim=0), torch.zeros(2))
    assert not torch.allclose(scaler.transform(validation), torch.zeros_like(validation))


def test_sparse_model_has_exact_zeros_and_unit_decoder_columns() -> None:
    """ReLU exposes measurable L0 sparsity and normalization removes L1 scale degeneracy."""
    model = SparseConceptModel(embedding_dim=5, concept_count=3)
    with torch.no_grad():
        model.encoder.weight.fill_(-1.0)
        model.encoder.bias.zero_()

    concepts, reconstruction = model(torch.ones(4, 5))

    assert torch.count_nonzero(concepts) == 0
    assert reconstruction.shape == (4, 5)
    assert torch.allclose(model.decoder.weight.norm(dim=0), torch.ones(3), atol=1.0e-6)


def test_seed_matching_is_permutation_invariant() -> None:
    """Hungarian decoder matching recovers identical concepts in a different order."""
    first  = SparseConceptModel(embedding_dim=3, concept_count=3)
    second = SparseConceptModel(embedding_dim=3, concept_count=3)
    with torch.no_grad():
        first.decoder.weight.copy_(torch.eye(3))
        second.decoder.weight.copy_(torch.eye(3)[:, [2, 0, 1]])

    result = SparseConceptDiscovery._stability((first, second), threshold=0.99)

    assert result["stable"] == 3
    assert result["unstable"] == 0


def test_concept_training_cannot_change_frozen_predictor_weights() -> None:
    """Sparse optimization owns no reference to, or optimizer state for, predictor parameters."""
    predictor = torch.nn.Linear(4, 1)
    predictor.requires_grad_(False)
    before = {name: value.clone() for name, value in predictor.state_dict().items()}
    concept = SparseConceptModel(embedding_dim=4, concept_count=2)
    optimizer = torch.optim.Adam(concept.parameters(), lr=0.01)
    values = torch.randn(32, 4)

    for _ in range(5):
        codes, reconstruction = concept(values)
        loss = (reconstruction - values).square().mean() + 0.1 * codes.mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        concept.normalize_decoder()

    assert all(torch.equal(before[name], value) for name, value in predictor.state_dict().items())


def test_larger_activation_penalty_produces_sparser_codes() -> None:
    """A stronger mean-ReLU penalty decreases both activation rate and activation magnitude."""
    torch.manual_seed(12)
    values = torch.randn(256, 4)

    def fit(sparse_lambda: float) -> tuple[float, float]:
        """Fit one deterministic comparison model and return its two sparsity summaries."""
        torch.manual_seed(91)
        model     = SparseConceptModel(embedding_dim=4, concept_count=6)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        for _epoch in range(160):
            concepts, reconstruction = model(values)
            loss = (reconstruction - values).square().mean() + sparse_lambda * concepts.mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            model.normalize_decoder()

        with torch.no_grad():
            concepts, _ = model(values)
        return float((concepts > 0.0).float().mean()), float(concepts.mean())

    unregularized_rate, unregularized_mean = fit(0.0)
    sparse_rate, sparse_mean               = fit(1.0)

    assert sparse_rate < unregularized_rate
    assert sparse_mean < unregularized_mean


def test_concept_report_measures_local_and_protein_knockout_effects() -> None:
    """Zeroing a concept reports its exact frozen-head effect without using labels."""
    discovery = SparseConceptDiscovery()
    scaler    = EmbeddingScaler(torch.zeros(2), torch.ones(2))
    model     = SparseConceptModel(embedding_dim=2, concept_count=2)
    head      = torch.nn.Linear(2, 1, bias=False)
    values    = torch.tensor([[2.0, 1.0], [1.0, 2.0], [3.0, 1.0], [1.0, 3.0]])

    with torch.no_grad():
        model.encoder.weight.copy_(torch.eye(2))
        model.encoder.bias.zero_()
        model.decoder.weight.copy_(torch.eye(2))
        model.decoder.bias.zero_()
        head.weight.copy_(torch.tensor([[1.0, 0.0]]))

    metadata = {
        "local_logits": values[:, 0],
        "positions": torch.arange(12, dtype=torch.float32).reshape(4, 3),
        "point_ids": torch.arange(4),
        "proteins": ["FIRST_A", "SECOND_A"],
        "ptr": torch.tensor([0, 2, 4]),
    }
    concept_rows, top_rows = discovery._concept_reports(
        model,
        values,
        values,
        metadata,
        metadata,
        head,
        scaler,
        near_dead_threshold=0.001,
        dominant_threshold=0.95,
        redundancy_threshold=0.95,
        stability_scores=torch.ones(2),
        stability_threshold=0.80,
        top_activations_per_concept=2,
    )

    assert concept_rows[0]["local_logit_knockout_effect"] > 0.0
    assert concept_rows[0]["protein_logit_knockout_effect"] > 0.0
    assert concept_rows[1]["local_logit_knockout_effect"] == 0.0
    assert concept_rows[1]["protein_logit_knockout_effect"] == 0.0
    assert len(top_rows) == 4


def test_sampling_manifest_preserves_exact_split_point_membership(tmp_path: Path) -> None:
    """The compact audit records every sampled original point without consulting labels."""
    path = tmp_path / "sampling.jsonl"
    metadata = {
        "ptr": torch.tensor([0, 2, 3]),
        "point_ids": torch.tensor([3, 8, 2]),
        "proteins": ["FIRST_A", "SECOND_A"],
    }

    SparseConceptDiscovery._write_sampling_manifest(path, {"train": metadata})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {
            "split": "train",
            "protein": "FIRST_A",
            "surface_point_indices": [3, 8],
        },
        {
            "split": "train",
            "protein": "SECOND_A",
            "surface_point_indices": [2],
        },
    ]
