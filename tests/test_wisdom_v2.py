from __future__ import annotations

import pytest
import torch
from test_wisdom_v1 import _model, _sample

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.models.PoolingType import PoolingType
from wisdom.models.WisdomV2 import WisdomV2


def _v2(pooling_type: PoolingType | str, **pooling_parameters: object) -> WisdomV2:
    return WisdomV2(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=2,
        surface_layers=2,
        dropout=0.0,
        curvature_features=6,
        pooling_type=pooling_type,
        **pooling_parameters,
    )


def _inputs(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        name: batch[name]
        for name in (
            "atomic_numbers",
            "residue_type_ids",
            "atom_edge_index",
            "atom_edge_types",
            "surface_curvatures",
            "surface_edge_index",
            "surface_atom_edge_index",
            "surface_area_weights",
            "surface_batch",
        )
    }


def _grid_edges(side: int) -> torch.Tensor:
    directed_edges: list[tuple[int, int]] = []
    for row in range(side):
        for column in range(side):
            source = row * side + column
            for row_delta, column_delta in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                target_row    = row + row_delta
                target_column = column + column_delta
                if 0 <= target_row < side and 0 <= target_column < side:
                    directed_edges.append((source, target_row * side + target_column))
    return torch.tensor(directed_edges, dtype=torch.long).T.contiguous()


def test_max_pooling_is_exactly_the_v1_control() -> None:
    v1 = _model().eval()
    v2 = _v2(PoolingType.MAX).eval()
    v2.load_state_dict(v1.state_dict(), strict=False)
    batch = WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0)))

    with torch.no_grad():
        v1_outputs = v1(**_inputs(batch))
        v2_outputs = v2(**_inputs(batch))

    assert torch.equal(v1_outputs["logits"], v2_outputs["logits"])
    assert torch.equal(v1_outputs["surface_logits"], v2_outputs["surface_logits"])


@pytest.mark.parametrize(
    ("pooling_type", "parameters"),
    [
        (PoolingType.MAX, {}),
        (PoolingType.MEAN, {}),
        (PoolingType.ATTENTION, {"attention_hidden_dim": 4}),
        (PoolingType.TOPK, {"topk_fraction": 0.5}),
        (PoolingType.LOCAL_MEAN_MAX, {"regional_levels": 2}),
        (PoolingType.LOG_SUM_EXP, {"log_sum_exp_beta": 3.0}),
    ],
)
def test_pooling_variants_produce_finite_maps_diagnostics_and_gradients(
    pooling_type: PoolingType,
    parameters: dict[str, object],
) -> None:
    model = _v2(pooling_type, **parameters)
    batch = WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0)))
    outputs = model(**_inputs(batch))

    assert outputs["logits"].shape == (2,)
    assert outputs["surface_logits"].shape == (5,)
    assert outputs["surface_probabilities"].shape == (5,)
    assert outputs["localization_scores"].shape == (5,)
    assert outputs["positive_area_fraction"].shape == (2,)
    assert outputs["localization_entropy"].shape == (2,)
    assert outputs["maximum_surface_probability"].shape == (2,)
    assert all(torch.isfinite(value).all() for value in outputs.values())
    assert torch.allclose(outputs["localization_scores"][:2].sum(), torch.tensor(1.0))
    assert torch.allclose(outputs["localization_scores"][2:].sum(), torch.tensor(1.0))
    assert torch.all(
        (outputs["localization_entropy"] >= 0.0)
        & (outputs["localization_entropy"] <= 1.0)
    )
    if pooling_type is PoolingType.ATTENTION:
        assert torch.allclose(outputs["attention_weights"][:2].sum(), torch.tensor(1.0))
        assert torch.allclose(outputs["attention_weights"][2:].sum(), torch.tensor(1.0))

    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], batch["target"])
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(
        value is not None and torch.isfinite(value).all() for value in gradients
    )


def test_masked_pooling_keeps_proteins_independent_inside_a_batch() -> None:
    model = _v2(PoolingType.LOG_SUM_EXP, log_sum_exp_beta=2.0).eval()
    first  = _sample(3, 2, 1.0)
    second = _sample(2, 5, 0.0)

    together = WisdomCollator()((first, second))
    separate = (WisdomCollator()((first,)), WisdomCollator()((second,)))
    with torch.no_grad():
        combined_logits = model(**_inputs(together))["logits"]
        separate_logits = torch.cat([model(**_inputs(batch))["logits"] for batch in separate])

    assert torch.allclose(combined_logits, separate_logits, rtol=1.0e-5, atol=1.0e-6)


def test_local_area_mean_suppresses_an_isolated_peak_but_preserves_a_region() -> None:
    model      = _v2(PoolingType.LOCAL_MEAN_MAX, regional_levels=1).eval()
    edges      = _grid_edges(3)
    embeddings = torch.zeros(9, model.hidden_dim)
    weights    = torch.ones(9)
    batch      = torch.zeros(9, dtype=torch.long)

    isolated = torch.full((9,), -2.0)
    isolated[4] = 9.0
    coherent = torch.tensor([-2.0, 5.0, 6.0, 4.0, 8.0, 5.0, -2.0, 4.0, 3.0])

    isolated_logit = model.pool_surface_logits(
        isolated, embeddings, edges, weights, batch
    )["logits"]
    coherent_logit = model.pool_surface_logits(
        coherent, embeddings, edges, weights, batch
    )["logits"]

    assert isolated_logit.item() < 1.0
    assert coherent_logit.item() >= 5.0


def test_local_consensus_respects_represented_surface_area() -> None:
    model      = _v2(PoolingType.LOCAL_MEAN_MAX, regional_levels=1).eval()
    edges      = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]])
    logits     = torch.tensor([-2.0, 8.0, -2.0])
    embeddings = torch.zeros(3, model.hidden_dim)
    batch      = torch.zeros(3, dtype=torch.long)

    small_peak = model.pool_surface_logits(
        logits,
        embeddings,
        edges,
        torch.tensor([1.0, 0.01, 1.0]),
        batch,
    )["logits"]
    large_peak = model.pool_surface_logits(
        logits,
        embeddings,
        edges,
        torch.tensor([1.0, 10.0, 1.0]),
        batch,
    )["logits"]

    assert small_peak.item() < 0.0
    assert large_peak.item() > small_peak.item()


def test_uniform_localization_has_unit_normalized_entropy() -> None:
    model = _v2(PoolingType.MAX).eval()
    torch.nn.init.zeros_(model.local_head.weight)
    torch.nn.init.zeros_(model.local_head.bias)

    first  = _sample(3, 2, 1.0)
    second = _sample(2, 3, 0.0)
    first["surface_area_weights"]  = torch.ones(2)
    second["surface_area_weights"] = torch.ones(3)
    batch = WisdomCollator()((first, second))

    with torch.no_grad():
        entropy = model(**_inputs(batch))["localization_entropy"]

    assert torch.allclose(entropy, torch.ones(2), rtol=1.0e-5, atol=1.0e-6)
