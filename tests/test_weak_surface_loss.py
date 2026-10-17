import pytest
import torch

from wisdom.models.WeakSurfaceLoss import WeakSurfaceLoss


def _operators() -> list[dict[str, torch.Tensor]]:
    return [
        {
            "mass":         torch.tensor([0.5, 0.5]),
            "eigenvalues":  torch.tensor([0.0, 1.0]),
            "eigenvectors": torch.eye(2),
        },
        {
            "mass":         torch.tensor([0.5, 0.5]),
            "eigenvalues":  torch.tensor([0.0, 1.0]),
            "eigenvectors": torch.eye(2),
        },
    ]


def test_weak_surface_losses_use_only_bag_labels_and_geometry() -> None:
    logits  = torch.tensor([-2.0, -1.0, 1.0, 3.0], requires_grad=True)
    areas   = torch.tensor([1.0, 3.0, 1.0, 1.0])
    owners  = torch.tensor([0, 0, 1, 1])
    targets = torch.tensor([0.0, 1.0])
    ptr     = torch.tensor([0, 2, 4])

    loss = WeakSurfaceLoss(
        negative_weight=0.3,
        positive_existence_weight=0.2,
        regional_positive_weight=0.1,
        regional_length=0.0,
        regional_ranking_weight=0.4,
        ranking_margin=0.5,
        cardinality_weight=0.2,
        minimum_positive_area=0.1,
    )
    terms = loss(logits, areas, owners, targets, _operators(), ptr)

    expected_negative = 0.25 * torch.nn.functional.softplus(logits[0])
    expected_negative += 0.75 * torch.nn.functional.softplus(logits[1])
    expected_existence = torch.nn.functional.softplus(-logits[3])
    expected_ranking = torch.relu(torch.tensor(0.5) - logits[3] + logits[1])

    assert float(terms["negative"].detach()) == pytest.approx(float(expected_negative.detach()))
    assert float(terms["positive_existence"].detach()) == pytest.approx(
        float(expected_existence.detach())
    )
    assert float(terms["regional_positive"].detach()) == pytest.approx(
        float(expected_existence.detach())
    )
    assert float(terms["regional_ranking"].detach()) == pytest.approx(
        float(expected_ranking.detach())
    )
    assert float(terms["cardinality"].detach()) == pytest.approx(0.0)

    terms["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


def test_weak_surface_smoothness_alternatives_are_finite() -> None:
    logits    = torch.tensor([-2.0, 2.0, -1.0, 1.0], requires_grad=True)
    areas     = torch.ones(4)
    owners    = torch.tensor([0, 0, 1, 1])
    targets   = torch.tensor([0.0, 1.0])
    ptr       = torch.tensor([0, 2, 4])
    neighbors = torch.tensor([[1], [0], [3], [2]])
    mask      = torch.ones_like(neighbors, dtype=torch.bool)

    total_variation = WeakSurfaceLoss(total_variation_weight=0.1)(
        logits,
        areas,
        owners,
        targets,
        _operators(),
        ptr,
        neighbors,
        mask,
    )
    dirichlet = WeakSurfaceLoss(dirichlet_weight=0.1)(
        logits,
        areas,
        owners,
        targets,
        _operators(),
        ptr,
    )

    assert float(total_variation["total_variation"].detach()) > 0.0
    assert float(dirichlet["dirichlet"].detach()) > 0.0
    assert torch.isfinite(total_variation["total"])
    assert torch.isfinite(dirichlet["total"])


def test_weak_surface_loss_rejects_two_smoothness_priors() -> None:
    with pytest.raises(ValueError, match="compared separately"):
        WeakSurfaceLoss(total_variation_weight=0.1, dirichlet_weight=0.1)


@pytest.mark.parametrize("smoothness", ["total_variation", "dirichlet"])
def test_v6h_all_compatible_losses_sum_and_backpropagate(smoothness: str) -> None:
    """Combine six active priors without local targets, retaining the exact weighted sum.

    Args:
        smoothness: The one enabled smoothness alternative; TV and Dirichlet remain exclusive.

    The positive bag has insufficient mass and lower logits than the negative bag, so the
    cardinality and ranking terms really activate instead of passing vacuously at zero.
    """
    logits = torch.tensor([1.0, 2.0, -8.0, -9.0], requires_grad=True)
    weights = {"negative": 0.3, "positive_existence": 0.03, "regional_positive": 0.3,
               "regional_ranking": 0.1, "cardinality": 0.03,
               "total_variation": 0.005 if smoothness == "total_variation" else 0.0,
               "dirichlet": 0.001 if smoothness == "dirichlet" else 0.0}
    loss = WeakSurfaceLoss(
        negative_weight=weights["negative"],
        positive_existence_weight=weights["positive_existence"],
        regional_positive_weight=weights["regional_positive"], regional_length=2.5,
        regional_ranking_weight=weights["regional_ranking"], ranking_margin=0.5,
        cardinality_weight=weights["cardinality"], minimum_positive_area=0.02,
        total_variation_weight=weights["total_variation"],
        dirichlet_weight=weights["dirichlet"],
    )
    neighbors = torch.tensor([[1], [0], [3], [2]])
    terms = loss(
        logits, torch.ones(4), torch.tensor([0, 0, 1, 1]), torch.tensor([0.0, 1.0]),
        _operators(), torch.tensor([0, 2, 4]), neighbors, torch.ones_like(neighbors).bool(),
    )
    assert all(terms[name] > 0 for name, weight in weights.items() if weight > 0)
    assert torch.allclose(terms["total"], sum(weights[name] * terms[name] for name in weights))
    terms["total"].backward()
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    assert torch.count_nonzero(logits.grad) == 4


def test_weak_surface_loss_accepts_bfloat16_logits_with_float32_geometry() -> None:
    logits  = torch.tensor([-2.0, -1.0, 1.0, 3.0], dtype=torch.bfloat16, requires_grad=True)
    areas   = torch.tensor([1.0, 3.0, 1.0, 1.0], dtype=torch.float32)
    owners  = torch.tensor([0, 0, 1, 1])
    targets = torch.tensor([0.0, 1.0])
    ptr     = torch.tensor([0, 2, 4])

    terms = WeakSurfaceLoss(
        negative_weight=0.3,
        cardinality_weight=0.2,
        minimum_positive_area=0.1,
    )(logits, areas, owners, targets, _operators(), ptr)

    assert terms["total"].dtype == torch.float32
    assert torch.isfinite(terms["total"])

    terms["total"].backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()


@pytest.mark.parametrize("family", ["total_variation", "dirichlet"])
def test_smoothness_constant_oscillating_fields_and_logit_gradients(family: str) -> None:
    """Use a mass-orthonormal constant/contrast basis to test actual smoothness terms.

    Args:
        family: Local directed-neighbour L1 TV or normalized spectral Dirichlet energy.

    A constant has zero nonconstant-mode energy. A discontinuity increases the penalty
    and backward must reach its logits. Identity eigenvectors would not test this
    scientific contract because their first vector is not a constant surface mode.
    """
    operator = {
        "mass": torch.tensor([0.5, 0.5]),
        "eigenvalues": torch.tensor([0.0, 1.0]),
        "eigenvectors": torch.tensor([[1.0, 1.0], [1.0, -1.0]]),
    }
    neighbors = torch.tensor([[1], [0]])
    mask = torch.ones_like(neighbors, dtype=torch.bool)
    loss = WeakSurfaceLoss(**{f"{family}_weight": 1.0})
    values = []
    for data in ([0.0, 0.0], [-2.0, 2.0]):
        logits = torch.tensor(data, requires_grad=True)
        terms = loss(
            logits, torch.ones(2), torch.zeros(2, dtype=torch.long),
            torch.ones(1), [operator], torch.tensor([0, 2]), neighbors, mask,
        )
        values.append(float(terms[family].detach()))
        terms["total"].backward()
        assert logits.grad is not None and torch.isfinite(logits.grad).all()
        if data[0] != data[1]:
            assert logits.grad.abs().sum() > 0
    assert values[0] == pytest.approx(0.0, abs=1e-7)
    assert values[1] > values[0]


def test_cardinality_is_squared_expected_mass_hinge_not_hard_area() -> None:
    """Verify exact area normalization, squared lower bound and gradient flow."""
    logits = torch.tensor([-8.0, -7.0], requires_grad=True)
    areas = torch.tensor([1.0, 3.0])
    minimum = 0.01
    terms = WeakSurfaceLoss(cardinality_weight=0.1, minimum_positive_area=minimum)(
        logits, areas, torch.zeros(2, dtype=torch.long), torch.ones(1),
        [], torch.tensor([0, 2]),
    )
    mass = (areas / areas.sum() * logits.sigmoid()).sum()
    assert torch.allclose(terms["cardinality"], torch.relu(minimum - mass).square())
    terms["total"].backward()
    assert logits.grad is not None and logits.grad.abs().sum() > 0
