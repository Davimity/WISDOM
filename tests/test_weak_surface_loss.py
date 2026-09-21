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
