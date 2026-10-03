"""Mathematical contracts for the controlled V5 pooling characterization."""

import copy
from pathlib import Path

import pytest
import torch
from lambdaforge.work import WorkConfig, WorkRunner
from test_wisdom_v2 import _line_operator
from yaml import safe_load

from wisdom.models.BoundedScalar import BoundedScalar
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.PoolingType import PoolingType
from wisdom.models.ProteinPoolingHead import ProteinPoolingHead


def _pool(head, logits, areas=None, embeddings=None, counts=None):
    """Evaluate scalar evidence with explicit disjoint bag ownership and exact test spectra."""
    if areas is None:
        areas = torch.ones_like(logits)
    if embeddings is None:
        embeddings = torch.stack((logits, logits.square(), logits.sin()), dim=1)
    if counts is None:
        counts = [len(logits)]
    ptr = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()])
    owners = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    operators = [_line_operator(n) for n in counts]
    return head(logits, embeddings, areas, owners, operators, ptr)


def _head(family, area="point", **kwargs):
    """Build a deterministic tiny pooling head, without a trainable backbone."""
    return ProteinPoolingHead(3, family, 0, pooling_area_mode=area, **kwargs).eval()


@pytest.mark.parametrize("family", list(PoolingType))
@pytest.mark.parametrize("area", ["point", "area"])
def test_family_shapes_gradients_and_batch_isolation(family, area):
    head = _head(family, area)
    values = torch.tensor([-2.0, 0.3, 1.2, 2.0, -0.5, 1.0], requires_grad=True)
    masses = torch.tensor([1.0, 2.0, 3.0, 1.0, 4.0, 1.0])
    combined = _pool(head, values, masses, counts=[3, 3])["logits"]
    separate = torch.cat(
        [
            _pool(head, values[:3], masses[:3])["logits"],
            _pool(head, values[3:], masses[3:])["logits"],
        ]
    )
    assert combined.shape == (2,)
    assert torch.isfinite(combined).all()
    assert torch.allclose(combined, separate, atol=2e-6)
    combined.sum().backward()
    assert values.grad is not None and torch.isfinite(values.grad).all()
    assert values.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "family",
    [
        "mean",
        "attention",
        "topk",
        "log_sum_exp",
        "linear_softmax",
        "autopool",
        "gem",
        "max_mean",
    ],
)
def test_uniform_area_matches_point_measure(family):
    point = _head(family, topk_fraction=0.5)
    area = _head(family, "area", topk_fraction=0.5)
    area.load_state_dict(point.state_dict())
    values = torch.tensor([-3.0, 1.0, 2.0, 4.0])
    assert torch.allclose(_pool(point, values)["logits"], _pool(area, values)["logits"], atol=1e-6)


@pytest.mark.parametrize(
    "family",
    [
        "mean",
        "attention",
        "topk",
        "log_sum_exp",
        "linear_softmax",
        "autopool",
        "gem",
        "max_mean",
    ],
)
def test_area_measure_is_invariant_to_point_subdivision(family):
    head = _head(family, "area", topk_fraction=0.27, gem_power=8)
    values = torch.tensor([-2.0, 0.7, 2.3])
    mass = torch.tensor([2.0, 5.0, 3.0])
    embeddings = torch.tensor([[0.2, -0.5, 1.0], [1.0, 0.1, 0.7], [0.4, 0.6, -0.3]])
    original = _pool(head, values, mass, embeddings)["logits"]
    subdivided = _pool(
        head,
        values[[0, 1, 1, 2]],
        torch.tensor([2.0, 2.5, 2.5, 3.0]),
        embeddings[[0, 1, 1, 2]],
    )["logits"]
    assert torch.allclose(original, subdivided, atol=2e-6)


def test_fractional_area_topk_has_exact_boundary_and_not_ceiling_count():
    values = torch.tensor([8.0, 4.0, 0.0])
    areas = torch.tensor([1.0, 3.0, 6.0])
    result = _pool(_head("topk", "area", topk_fraction=0.25), values, areas)["logits"]
    assert result.item() == pytest.approx((0.1 * 8 + 0.15 * 4) / 0.25)
    # Uniform area and a non-integral target differ intentionally from legacy ceil(f*N).
    uniform = _pool(_head("topk", "area", topk_fraction=0.5), values)["logits"]
    point = _pool(_head("topk", "point", topk_fraction=0.5), values)["logits"]
    assert uniform.item() == pytest.approx((8 + 0.5 * 4) / 1.5)
    assert point.item() == pytest.approx(6)


def test_known_limits_and_probability_space_controls():
    values = torch.tensor([-2.0, 0.5, 3.0])
    maximum = _pool(_head("max"), values)["logits"]
    # A zero length bypasses even a deliberately truncated spectrum.
    ptr = torch.tensor([0, 3])
    op = _line_operator(3)
    op["eigenvectors"] = op["eigenvectors"][:, :1]
    op["eigenvalues"] = op["eigenvalues"][:1]
    assert DiffusionSurfaceEncoder.diffuse(values, [op], ptr, 0.0) is values
    assert torch.equal(
        _pool(_head("local_mean_max", regional_diffusion_scale=0), values)["logits"], maximum
    )
    assert torch.allclose(
        _pool(_head("log_sum_exp", log_sum_exp_beta=10000), values)["logits"], maximum, atol=0.001
    )
    for area in ["point", "area"]:
        masses = torch.tensor([1.0, 2.0, 5.0])
        measure = masses / masses.sum() if area == "area" else torch.ones(3) / 3
        expected = torch.sum(measure * values.sigmoid())
        gem = _pool(_head("gem", area, gem_power=1), values, masses)["logits"].sigmoid()
        auto = _pool(_head("autopool", area, autopool_alpha=0), values, masses)["logits"].sigmoid()
        assert torch.allclose(gem, expected[None], atol=1e-6)
        assert torch.allclose(auto, expected[None], atol=1e-6)
        mean = _pool(_head("mean", area), values, masses)["logits"]
        assert torch.equal(
            _pool(_head("max_mean", area, max_mean_lambda=0), values, masses)["logits"], mean
        )
        assert torch.equal(
            _pool(_head("max_mean", area, max_mean_lambda=1), values, masses)["logits"], maximum
        )


@pytest.mark.parametrize(
    ("family", "kwargs", "bounds"),
    [
        ("log_sum_exp", {"log_sum_exp_mode": "learned", "log_sum_exp_beta_init": 5}, (0.25, 200)),
        (
            "local_mean_max",
            {"regional_scale_mode": "learned", "regional_diffusion_scale_init": 1.5},
            (0.05, 12),
        ),
        ("autopool", {"autopool_alpha_mode": "learned", "autopool_alpha_init": 1}, (0, 50)),
        ("gem", {"gem_power_mode": "learned", "gem_power_init": 4}, (1, 32)),
        ("max_mean", {"max_mean_lambda_mode": "learned", "max_mean_lambda_init": 0.5}, (0, 1)),
    ],
)
def test_learned_scalar_gradient_domain_and_state_roundtrip(family, kwargs, bounds):
    head = _head(family, "area", **kwargs)
    values = torch.tensor([-0.7, 0.1, 2.3, 0.8], requires_grad=True)
    optimizer = torch.optim.AdamW(head.parameters(), lr=0.01)
    for _ in range(3):
        optimizer.zero_grad()
        _pool(head, values, torch.tensor([2.0, 1.0, 3.0, 2.0]))["logits"].sum().backward()
        scalar = head.learned_scalar
        assert scalar is not None and scalar.raw.grad is not None
        assert torch.isfinite(scalar.raw.grad) and scalar.raw.grad.abs() > 1e-8
        optimizer.step()
        assert bounds[0] < scalar().item() < bounds[1]
    restored = _head(family, "area", **kwargs)
    restored.load_state_dict(copy.deepcopy(head.state_dict()))
    assert restored.parameter_values() == head.parameter_values()
    assert torch.equal(_pool(restored, values)["logits"], _pool(head, values)["logits"])


def test_multiscale_convex_weights_receive_gradients_and_serialize():
    head = _head("multiscale_regional_max")
    values = torch.tensor([-2.0, 1.0, 5.0, -1.0], requires_grad=True)
    _pool(head, values)["logits"].sum().backward()
    assert head.scale_logits.grad is not None
    assert torch.isfinite(head.scale_logits.grad).all()
    assert head.scale_logits.grad.abs().sum() > 0
    assert sum(head.parameter_values().values()) == pytest.approx(1.0)
    restored = _head("multiscale_regional_max")
    restored.load_state_dict(head.state_dict())
    assert torch.equal(_pool(restored, values)["logits"], _pool(head, values)["logits"])


@pytest.mark.parametrize("variant", ["simple", "gated"])
def test_attention_uniform_prior_subdivision_and_bfloat16_audit(variant):
    point = _head("attention", attention_variant=variant)
    area = _head("attention", "area", attention_variant=variant)
    area.load_state_dict(point.state_dict())
    values = torch.tensor([-1.0, 0.2, 2.0])
    embeddings = torch.randn(3, 3).bfloat16()
    first = _pool(point, values, embeddings=embeddings)
    second = _pool(area, values, embeddings=embeddings)
    assert torch.allclose(first["attention_weights"], second["attention_weights"], atol=1e-6)
    assert torch.allclose(first["logits"], second["logits"], atol=1e-6)
    divided = _pool(
        area, values[[0, 0, 1, 2]], torch.tensor([0.5, 0.5, 1.0, 1.0]), embeddings[[0, 0, 1, 2]]
    )
    assert torch.allclose(second["logits"], divided["logits"], atol=1e-6)


@pytest.mark.parametrize("family", ["log_sum_exp", "linear_softmax", "autopool", "gem"])
@pytest.mark.parametrize("values", [[-1000.0, -999.0], [999.0, 1000.0], [-1000.0, 1000.0]])
def test_extreme_logits_remain_finite_and_differentiable(family, values):
    local = torch.tensor(values, requires_grad=True)
    head = _head(family, "area", log_sum_exp_beta=160, gem_power=32, autopool_alpha=50)
    result = _pool(head, local, torch.tensor([0.3, 0.7]))["logits"]
    result.sum().backward()
    assert torch.isfinite(result).all()
    assert local.grad is not None and torch.isfinite(local.grad).all()


def test_legacy_checkpoint_keys_and_curriculum_restore():
    head = _head("attention")
    assert all(key.startswith("attention.scorer.") for key in head.state_dict())
    restored = _head("attention")
    restored.load_state_dict(head.state_dict(), strict=True)
    for family in ["max", "mean", "topk", "local_mean_max", "log_sum_exp"]:
        assert not _head(family).state_dict()
    scheduled = _head("log_sum_exp", log_sum_exp_mode="curriculum")
    scheduled.set_log_sum_exp_beta(80)
    resumed = _head("log_sum_exp", log_sum_exp_mode="curriculum")
    resumed.load_state_dict(scheduled.state_dict())
    assert resumed.parameter_values()["pooling_beta"] == 80
    learned = _head("log_sum_exp", log_sum_exp_mode="learned")
    with pytest.raises(ValueError, match="schedule"):
        learned.set_log_sum_exp_beta(1)


@pytest.mark.parametrize(
    "family",
    [
        "max",
        "mean",
        "attention",
        "topk",
        "local_mean_max",
        "log_sum_exp",
    ],
)
def test_legacy_defaults_preserve_the_original_measure(family):
    legacy = _head(family, "legacy", topk_fraction=0.5)
    explicit = _head(family, "area" if family == "mean" else "point", topk_fraction=0.5)
    explicit.load_state_dict(legacy.state_dict(), strict=True)
    values = torch.tensor([-0.5, 2.0, 1.0, -0.2])
    areas = torch.tensor([1.0, 3.0, 2.0, 1.0])
    assert torch.equal(
        _pool(legacy, values, areas)["logits"], _pool(explicit, values, areas)["logits"]
    )


def test_bounded_scalars_have_finite_initial_gradients_and_valid_domains():
    for initial, lower, upper, log in [(1.0, 0.25, 200.0, True), (0.5, 0.0, 1.0, False)]:
        scalar = BoundedScalar(initial, lower, upper, log)
        assert scalar().item() == pytest.approx(initial)
        scalar().backward()
        assert scalar.raw.grad is not None and scalar.raw.grad > 0
        for raw in [-1000.0, 1000.0]:
            with torch.no_grad():
                scalar.raw.fill_(raw)
            assert lower <= scalar().item() <= upper


def test_v5_grids_use_exact_frozen_values_and_four_seeds():
    """Use native planning to verify one paired Study and the reviewed ledger per YAML."""
    root = Path(__file__).parents[1] / "experiments"
    legacy = safe_load((root / "wisdom_v5.yaml").read_text())["with"]
    expected = {
        "wisdom_v5a.yaml": [1, 2, 10, 16, 9, 18],
        "wisdom_v5b.yaml": [8, 18, 4, 2, 16, 18, 12, 1],
    }
    for name, counts in expected.items():
        raw = safe_load((root / name).read_text())
        assert "steps" not in raw and "search" not in raw
        assert raw["seeds"] == [4, 7, 32, 54]
        assert raw["resources"] == {"gpu": 2, "cpu": 36, "memory": "96GiB", "time": "168h"}
        assert raw["execution"] == {"max_time": "168h"}
        for key, value in legacy.items():
            assert raw["with"][key] == value, (name, key)
        # Planning invokes the installed public LF expander but executes no scientific Work.
        plan = WorkRunner().plan(WorkConfig.from_yaml(root / name))
        assert [len(level) for level in plan.levels] == [sum(counts) * 4]
        assert len(plan.preflight["studies"]) == 1
        facts = plan.preflight["studies"][0]
        assert facts["candidates"] == sum(counts)
        assert facts["required_runs"] == sum(counts) * 4
        assert facts["shared_seeds"] == [4, 7, 32, 54]
        assert facts["study_time_budget_seconds"] == 604800
        assert facts["scheduler_wall_time_seconds"] == 604800
        assert facts["requested_gpus"] == 2 and facts["max_parallel"] is None
        assert facts["reference"] == ({"pooling_type": "max"} if "v5a" in name else None)
        assert [item["candidates"] for item in facts["conditional_branches"]["counts"]] == counts


def test_v5_inactive_parameters_are_absent_from_native_candidates():
    """Check exact active keys, including transitive conditions on adaptive scalars."""
    root = Path(__file__).parents[1] / "experiments"
    fixed = {
        "max": set(),
        "mean": {"pooling_area_mode"},
        "attention": {"pooling_area_mode", "attention_hidden_dim"},
        "topk": {"pooling_area_mode", "topk_fraction"},
        "local_mean_max": {"regional_diffusion_scale"},
        "log_sum_exp": {"pooling_area_mode", "log_sum_exp_beta"},
    }
    adaptive = {
        "attention": {"pooling_area_mode", "attention_variant", "attention_hidden_dim"},
        "log_sum_exp": {"pooling_area_mode", "log_sum_exp_mode"},
        "local_mean_max": {"regional_scale_mode", "regional_diffusion_scale_init"},
        "linear_softmax": {"pooling_area_mode"},
        "autopool": {"pooling_area_mode", "autopool_alpha_mode"},
        "gem": {"pooling_area_mode", "gem_power_mode"},
        "max_mean": {"pooling_area_mode", "max_mean_lambda_mode"},
        "multiscale_regional_max": set(),
    }
    for name, branches in [("v5a", fixed), ("v5b", adaptive)]:
        config = WorkConfig.from_yaml(root / f"wisdom_{name}.yaml")
        run = config.levels[0].runs[0]
        for candidate in run.variants:
            family = candidate["pooling_type"]
            keys = {"pooling_type"} | branches[family]
            if name == "v5b":
                if family == "log_sum_exp":
                    keys |= (
                        {"log_sum_exp_beta_init"}
                        if candidate["log_sum_exp_mode"] == "learned"
                        else {"pooling_curriculum_end_beta", "pooling_curriculum_hold_fraction"}
                    )
                for scalar_family, scalar in [
                    ("autopool", "autopool_alpha"),
                    ("gem", "gem_power"),
                    ("max_mean", "max_mean_lambda"),
                ]:
                    if family == scalar_family:
                        keys.add(
                            scalar if candidate[f"{scalar}_mode"] == "fixed" else f"{scalar}_init"
                        )
            assert set(candidate) == keys, (name, candidate)
        # Reference matching is candidate-level, not one duplicated baseline per family.
        if name == "v5a":
            assert [dict(c) for c in run.variants if c["pooling_type"] == "max"] == [
                {"pooling_type": "max"}
            ]
        else:
            assert all(c["pooling_type"] != "max" for c in run.variants)


def test_v5b_curriculum_changes_before_validation_patience_can_expire():
    """Ensure the configured schedule has a chance to act without disabling early stopping."""
    root = Path(__file__).parents[1] / "experiments"
    run = WorkConfig.from_yaml(root / "wisdom_v5b.yaml").levels[0].runs[0]
    curriculum = [c for c in run.variants if c.get("log_sum_exp_mode") == "curriculum"]
    assert len(curriculum) == 10
    assert run.parameters["epochs"] == 500 and run.parameters["patience"] == 30
    for candidate in curriculum:
        assert candidate["pooling_curriculum_hold_fraction"] == 0.0
        # Training computes progress=(epoch-1)/epochs; zero hold changes beta at epoch 2.
        fraction = (2 - 1) / run.parameters["epochs"]
        beta = 1.0 + fraction * (candidate["pooling_curriculum_end_beta"] - 1.0)
        assert 1.0 < beta < candidate["pooling_curriculum_end_beta"]
