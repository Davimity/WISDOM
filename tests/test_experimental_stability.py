import pytest
import torch

from wisdom.models.ArchitectureSpike import ArchitectureSpike
from wisdom.models.GlobalSurfaceHeads import GlobalSurfaceHeads
from wisdom.models.InitializationProfile import InitializationProfile
from wisdom.models.ModelWeightAverage import ModelWeightAverage
from wisdom.models.OptimizationProfile import OptimizationProfile
from wisdom.models.SurfaceAtomFeedback import SurfaceAtomFeedback
from wisdom.models.VectorAtomicState import VectorAtomicState
from wisdom.models.WeakLossProfile import WeakLossProfile
from wisdom.models.WisdomV1 import WisdomV1


def test_experimental_profiles_change_only_the_declared_hypothesis() -> None:
    point_geometry = ArchitectureSpike.POINT_GEOMETRY.overrides()
    weak_negative  = WeakLossProfile.NEGATIVE_030.overrides()

    assert point_geometry == {
        "surface_geometry_transfer": True,
        "interaction_round":         "single",
        "vector_atomic_channels":    0,
    }
    assert weak_negative["negative_surface_lambda"] == pytest.approx(0.30)
    assert weak_negative["regional_positive_lambda"] == pytest.approx(0.0)
    assert weak_negative["total_variation_lambda"] == pytest.approx(0.0)

    positive_existence = WeakLossProfile.POSITIVE_EXIST_030.overrides()
    assert positive_existence["negative_surface_lambda"] == pytest.approx(0.30)
    assert positive_existence["positive_existence_lambda"] == pytest.approx(0.30)
    assert positive_existence["regional_positive_lambda"] == pytest.approx(0.0)


def test_custom_experimental_profiles_preserve_explicit_parameters() -> None:
    assert ArchitectureSpike.CUSTOM.overrides() == {}
    assert WeakLossProfile.CUSTOM.overrides() == {}


def test_initialization_and_optimizer_profiles_compose_without_overwriting() -> None:
    """Keep independently selected initialization and optimizer policies active together."""
    initialization = InitializationProfile.GATE_WARMUP.overrides()
    optimization   = OptimizationProfile.EMA_0999.overrides()
    combined       = initialization | optimization

    assert combined == {
        "gate_warmup_fraction": 0.10,
        "gate_ramp_fraction":   0.20,
        "weight_averaging":     "ema",
        "ema_decay":            0.999,
    }


def test_initialization_controls_preserve_valid_model_contract() -> None:
    model = WisdomV1(
        hidden_dim=8,
        embedding_dim=4,
        residue_embedding_dim=4,
        atomic_layers=1,
        projection_depth=1,
        surface_layers=4,
        dropout=0.0,
        curvature_features=6,
        embedding_initialization="fan_scaled",
        gate_initial_active=0.5,
        diffusion_time_initialization="broader_lengths",
        diffusion_residual_initialization="layerscale",
        atomic_residual_initialization="rezero",
        neutral_edge_initialization=True,
        neutral_transfer_initialization=True,
        physics_distance_initialization=True,
        local_head_weight_std=0.01,
        linear_initialization="activation_aware",
    )

    diffusion_times = [
        float(block.diffusion_times[0].detach()) for block in model.surface_encoder.blocks
    ]
    assert diffusion_times == pytest.approx([0.25, 1.0, 4.0, 16.0], rel=1.0e-5)
    active_probability = model.semantic_gates.probability_active(
        model.semantic_gates.names[0]
    )
    assert float(active_probability.detach()) == pytest.approx(0.5)
    assert torch.count_nonzero(model.atomic_encoder.residual_scales[0]) == 0
    distance_prior = model.surface_atom_transfer.distance_prior
    assert distance_prior is not None
    assert float(distance_prior.detach()) == pytest.approx(-1.0)


def test_physical_embeddings_are_deterministic_neutral_and_trainable() -> None:
    first = WisdomV1(
        hidden_dim=8,
        embedding_dim=8,
        atomic_layers=1,
        projection_depth=1,
        surface_layers=1,
        dropout=0.0,
        curvature_features=6,
        embedding_initialization="physical",
    )
    second = WisdomV1(
        hidden_dim=8,
        embedding_dim=8,
        atomic_layers=1,
        projection_depth=1,
        surface_layers=1,
        dropout=0.0,
        curvature_features=6,
        embedding_initialization="physical",
    )

    assert torch.equal(first.atomic_number_embedding.weight, second.atomic_number_embedding.weight)
    assert torch.equal(first.residue_type_embedding.weight, second.residue_type_embedding.weight)
    assert torch.count_nonzero(first.atomic_number_embedding.weight[0]) == 0
    assert first.atomic_number_embedding.weight.requires_grad
    assert first.residue_type_embedding.weight.requires_grad


@pytest.mark.parametrize("head_type", ["dual", "global_context", "film"])
def test_global_surface_heads_return_direct_and_conditioned_predictions(head_type: str) -> None:
    heads      = GlobalSurfaceHeads(8, head_type=head_type, context_dim=4, dropout=0.0)
    embeddings = torch.randn(7, 8)
    areas      = torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0])
    owners     = torch.tensor([0, 0, 0, 1, 1, 1, 1])

    output = heads(embeddings, areas, owners, protein_count=2)

    assert output["direct_logits"].shape == (2,)
    assert output["global_context"].shape == (2, 8)
    if head_type == "dual":
        assert "conditioned_surface_logits" not in output
    else:
        assert output["conditioned_surface_logits"].shape == (7,)


def test_model_weight_average_swaps_and_restores_optimizer_weights() -> None:
    model   = torch.nn.Linear(2, 1, bias=False)
    average = ModelWeightAverage(model, mode="ema", ema_decay=0.5)
    initial = model.weight.detach().clone()

    with torch.no_grad():
        model.weight.add_(2.0)
    optimized = model.weight.detach().clone()
    average.update(model, epoch=1)
    average.apply(model)

    assert torch.allclose(model.weight, initial + 1.0)
    average.restore(model)
    assert torch.allclose(model.weight, optimized)


def test_vector_atomic_state_is_rotation_equivariant_and_scalar_invariant() -> None:
    torch.manual_seed(7)
    module    = VectorAtomicState(hidden_dim=6, vector_channels=3)
    states    = torch.randn(4, 6)
    positions = torch.tensor(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]
    )
    edges     = torch.tensor([[0, 0, 0], [1, 2, 3]])
    distances = torch.tensor([1.0, 2.0, 3.0])
    rotation  = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])

    scalar, vectors = module(states, positions, edges, distances)
    rotated_scalar, rotated_vectors = module(states, positions @ rotation.T, edges, distances)

    assert torch.allclose(scalar, rotated_scalar, atol=1.0e-6)
    assert torch.allclose(rotated_vectors, vectors @ rotation.T, atol=1.0e-6)


def test_surface_atom_feedback_preserves_shapes_and_gradients() -> None:
    module   = SurfaceAtomFeedback(hidden_dim=5)
    atoms    = torch.randn(3, 5, requires_grad=True)
    surfaces = torch.randn(4, 5, requires_grad=True)
    neighbors = torch.tensor([[0, 1], [1, 2], [2, -1], [0, 2]])
    distances = torch.tensor([[1.0, 2.0], [1.0, 2.0], [1.0, 0.0], [3.0, 1.0]])
    mask      = neighbors >= 0

    updated = module(atoms, surfaces, neighbors, distances, mask, radius=3.0)
    updated.square().mean().backward()

    assert updated.shape == atoms.shape
    assert atoms.grad is not None
    assert surfaces.grad is not None
