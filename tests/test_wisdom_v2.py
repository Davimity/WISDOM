from __future__ import annotations

import numpy as np
import pytest
import torch
from test_wisdom_v1 import _model, _model_inputs, _sample

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.models.PoolingType import PoolingType
from wisdom.models.WisdomV2 import WisdomV2


def _v2(pooling_type: PoolingType | str, **pooling_parameters: object) -> WisdomV2:
    """Create a compact v2 instance with only the pooling hypothesis variable.

    Args:
        pooling_type: Closed pooling rule under test.
        **pooling_parameters: Pooling-specific constructor overrides.

    Returns:
        Small WISDOM v2 model with the fixture-compatible backbone.
    """
    return WisdomV2(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=2,
        projection_depth=1,
        surface_layers=2,
        dropout=0.0,
        curvature_features=6,
        atom_spatial_k=2,
        surface_atom_k=16,
        diffusion_spectral_modes=8,
        pooling_type=pooling_type,
        **pooling_parameters,
    )


def _line_operator(point_count: int) -> dict[str, torch.Tensor]:
    """Construct an exact spectral path-graph operator for regional-pooling tests.

    Args:
        point_count: Number of points in the one-dimensional path.

    Returns:
        Mass/eigenpair operator mapping compatible with ``diffuse_scalar``.
    """
    laplacian = np.zeros((point_count, point_count), dtype=np.float32)
    for source in range(point_count - 1):
        target = source + 1
        laplacian[source, source] += 1.0
        laplacian[target, target] += 1.0
        laplacian[source, target] -= 1.0
        laplacian[target, source] -= 1.0

    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    diagonal = torch.arange(point_count)
    return {
        "mass": torch.ones(point_count),
        "eigenvalues": torch.from_numpy(eigenvalues),
        "eigenvectors": torch.from_numpy(eigenvectors),
        "gradient_index": torch.stack((diagonal, diagonal)),
        "gradient_x": torch.zeros(point_count),
        "gradient_y": torch.zeros(point_count),
    }


def test_max_pooling_is_exactly_the_v1_control() -> None:
    """V2 MAX reproduces the v1 scientific control exactly."""
    v1 = _model().eval()
    v2 = _v2(PoolingType.MAX).eval()
    v2.load_state_dict(v1.state_dict(), strict=False)
    batch = dict(WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0))))

    with torch.no_grad():
        v1_outputs = v1(**_model_inputs(batch))
        v2_outputs = v2(**_model_inputs(batch))

    assert torch.equal(v1_outputs["logits"], v2_outputs["logits"])
    assert torch.equal(v1_outputs["surface_logits"], v2_outputs["surface_logits"])


@pytest.mark.parametrize(
    ("pooling_type", "parameters"),
    [
        (PoolingType.MAX, {}),
        (PoolingType.MEAN, {}),
        (PoolingType.ATTENTION, {"attention_hidden_dim": 4}),
        (PoolingType.TOPK, {"topk_fraction": 0.5}),
        (PoolingType.LOCAL_MEAN_MAX, {"regional_diffusion_scale": 2.0}),
        (PoolingType.LOG_SUM_EXP, {"log_sum_exp_beta": 3.0}),
    ],
)
def test_pooling_variants_produce_finite_maps_diagnostics_and_gradients(
    pooling_type: PoolingType,
    parameters  : dict[str, object],
) -> None:
    """Every pooling hypothesis supports a finite complete optimizer step."""
    model = _v2(pooling_type, **parameters)
    batch = dict(WisdomCollator()((_sample(3, 2, 1.0), _sample(2, 3, 0.0))))
    outputs = model(**_model_inputs(batch))

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

    loss = torch.nn.functional.binary_cross_entropy_with_logits(outputs["logits"], batch["target"])
    loss.backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients and all(torch.isfinite(value).all() for value in gradients)


def test_masked_pooling_keeps_proteins_independent_inside_a_batch() -> None:
    """One protein cannot affect another protein's pooled output."""
    model  = _v2(PoolingType.LOG_SUM_EXP, log_sum_exp_beta=2.0).eval()
    first  = _sample(3, 2, 1.0)
    second = _sample(2, 5, 0.0)

    together = WisdomCollator()((first, second))
    separate = (WisdomCollator()((first,)), WisdomCollator()((second,)))
    with torch.no_grad():
        combined_logits = model(**_model_inputs(dict(together)))["logits"]
        separate_logits = torch.cat(
            [model(**_model_inputs(dict(batch)))["logits"] for batch in separate]
        )

    assert torch.allclose(combined_logits, separate_logits, rtol=1.0e-5, atol=1.0e-6)


def test_regional_diffusion_suppresses_an_isolated_peak() -> None:
    """Short physical diffusion rewards a coherent region over one isolated extreme."""
    model      = _v2(PoolingType.LOCAL_MEAN_MAX, regional_diffusion_scale=1.5).eval()
    embeddings = torch.zeros(9, model.hidden_dim)
    weights    = torch.ones(9)
    batch      = torch.zeros(9, dtype=torch.long)
    operators  = [_line_operator(9)]
    pointer    = torch.tensor([0, 9])

    isolated = torch.full((9,), -2.0)
    isolated[4] = 9.0
    coherent = torch.tensor([-2.0, 5.0, 6.0, 4.0, 8.0, 5.0, -2.0, 4.0, 3.0])

    isolated_logit = model.pool_surface_logits(
        isolated,
        embeddings,
        weights,
        batch,
        operators,
        pointer,
    )["logits"]
    coherent_logit = model.pool_surface_logits(
        coherent,
        embeddings,
        weights,
        batch,
        operators,
        pointer,
    )["logits"]

    assert isolated_logit.item() < coherent_logit.item()
    assert coherent_logit.item() > 3.0


def test_uniform_localization_has_unit_normalized_entropy() -> None:
    """A uniform point distribution has normalized entropy one for each non-trivial bag."""
    model = _v2(PoolingType.MAX).eval()
    torch.nn.init.zeros_(model.local_head.weight)
    torch.nn.init.zeros_(model.local_head.bias)

    first  = _sample(3, 2, 1.0)
    second = _sample(2, 3, 0.0)
    first["surface_area_weights"]  = torch.ones(2)
    second["surface_area_weights"] = torch.ones(3)
    batch = dict(WisdomCollator()((first, second)))

    with torch.no_grad():
        entropy = model(**_model_inputs(batch))["localization_entropy"]

    assert torch.allclose(entropy, torch.ones(2), rtol=1.0e-5, atol=1.0e-6)
