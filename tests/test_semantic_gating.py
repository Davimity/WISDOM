"""Focused contracts for WISDOM semantic Hard-Concrete gating."""

import torch

from wisdom.models.gating.HardConcreteGate import HardConcreteGate
from wisdom.models.gating.SemanticGateRegistry import SemanticGateRegistry
from wisdom.models.WisdomV1 import WisdomV1


def test_hard_concrete_initial_probability_and_bounds() -> None:
    """The fixed initialization starts near 0.95 and every sample remains bounded."""
    gate    = HardConcreteGate()
    samples = torch.stack([gate(training=True) for _ in range(512)])

    assert torch.isclose(gate.probability_active(), torch.tensor(0.95), atol=1.0e-6)
    assert torch.all((samples >= 0.0) & (samples <= 1.0))
    assert (samples == 0.0).any() or (samples == 1.0).any()


def test_hard_concrete_gradient_monotonicity_and_eval_determinism() -> None:
    """Increasing log-alpha raises activity and evaluation never samples."""
    gate = HardConcreteGate()
    low  = gate.probability_active()
    with torch.no_grad():
        gate.log_alpha.add_(1.0)
    high = gate.probability_active()
    first = gate(training=False)
    second = gate(training=False)
    high.backward()

    assert high > low
    assert torch.equal(first, second)
    assert gate.log_alpha.grad is not None and gate.log_alpha.grad > 0.0


def test_registry_hierarchy_normalizes_l0_and_supports_overrides() -> None:
    """Effective child costs include ancestors without changing forward control flow."""
    registry = SemanticGateRegistry((('parent', None), ('child', 'parent')))

    expected = (
        registry.probability_active('parent')
        + registry.probability_active('parent') * registry.probability_active('child')
    ) / 2.0
    assert torch.allclose(registry.regularization(), expected)
    assert 0.0 <= float(registry.regularization().detach()) <= 1.0

    registry.set_override('all_on')
    values = registry.sample(training=True)
    assert values['parent'].item() == 1.0 and values['child'].item() == 1.0

    registry.force('child', 0)
    assert registry.sample(training=False)['child'].item() == 0.0
    registry.force('child', None)


def test_registry_state_round_trip_preserves_deterministic_gates() -> None:
    """Checkpoint state reproduces gate parameters independently of implicit ordering."""
    definitions = (('parent', None), ('child', 'parent'))
    source      = SemanticGateRegistry(definitions)
    restored    = SemanticGateRegistry(definitions)
    with torch.no_grad():
        source.gates[0].log_alpha.fill_(-8.0)
        source.gates[1].log_alpha.fill_(8.0)
    restored.load_state_dict(source.state_dict())

    assert all(
        torch.equal(source.deterministic_value(name), restored.deterministic_value(name))
        for name in source.names
    )


def test_v1_curvature_gates_are_per_descriptor_and_scale() -> None:
    """Each physical curvature channel has one stable global gate and fixed input width."""
    model = WisdomV1(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=1,
        projection_depth=1,
        surface_layers=1,
        curvature_features=6,
    )
    names = set(model.semantic_gates.names)

    for descriptor in ('mean', 'gaussian', 'curvedness', 'shape_index'):
        for scale in range(2):
            assert f'surface.curvature.{descriptor}.scale_{scale}' in names

    assert model.surface_projection.output.in_features == model.hidden_dim + 8
