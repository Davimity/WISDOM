"""Mathematical contracts for the controlled V5 pooling characterization."""

import copy
import inspect
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
from wisdom.Training import _create_model


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


@pytest.mark.parametrize(
    "family,settings,initial,fixed_setting",
    [
        ("log_sum_exp", {"log_sum_exp_mode": "learned", "log_sum_exp_beta_init": 0.25,
                         "log_sum_exp_beta_bounds": (0.005, 2560)}, 0.25, "log_sum_exp_beta"),
        ("local_mean_max", {"regional_scale_mode": "learned", "regional_diffusion_scale_init": 16,
                            "regional_diffusion_scale_bounds": (0.05, 128)},
         16, "regional_diffusion_scale"),
        ("autopool", {"autopool_alpha_mode": "learned", "autopool_alpha_init": 500,
                      "autopool_alpha_bounds": (0, 2000)}, 500, "autopool_alpha"),
        ("gem", {"gem_power_mode": "learned", "gem_power_init": 2,
                 "gem_power_bounds": (1, 2048)}, 2, "gem_power"),
        ("max_mean", {"max_mean_lambda_mode": "learned", "max_mean_lambda_init": 0.95},
         0.95, "max_mean_lambda"),
    ],
)
def test_v5b_warm_starts_match_fixed_operators_and_receive_bce_gradients(
    family, settings, initial, fixed_setting
):
    """Only the scalar is trainable: its initial forward equals the paired fixed reference."""
    learned = _head(family, "area", **settings)
    fixed = _head(family, "area", **{fixed_setting: initial})
    # Avoid a genuine MAX/probability saturation so the test diagnoses autograd, not biology.
    logits = torch.tensor([-0.008, -0.003, 0.001, 0.006], requires_grad=True)
    owners = torch.zeros(4, dtype=torch.int64)
    ptr = torch.tensor([0, 4])
    op = _line_operator(4)
    op["eigenvalues"] = op["eigenvalues"] / 1000
    mass = torch.ones(4)
    features = torch.randn(4, 3)
    result = learned(logits, features, mass, owners, [op], ptr)["logits"]
    baseline = fixed(logits, features, mass, owners, [op], ptr)["logits"]
    assert torch.allclose(result, baseline, atol=2e-6, rtol=2e-5)
    assert learned.learned_scalar is not None
    scalar = learned.learned_scalar
    assert scalar().item() == pytest.approx(initial, rel=2e-6)
    optimizer = torch.optim.AdamW(learned.parameters(), lr=0.01, weight_decay=0)
    assert any(p is scalar.raw for group in optimizer.param_groups for p in group["params"])
    torch.nn.functional.binary_cross_entropy_with_logits(result, torch.ones(1)).backward()
    assert scalar.raw.grad is not None and torch.isfinite(scalar.raw.grad)
    assert scalar.raw.grad.abs() > 0
    assert logits.grad is not None and torch.isfinite(logits.grad).all()
    before = scalar().item()
    optimizer.step()
    assert scalar().item() != before
    # A scalar pooler cannot secretly learn from the embeddings: changing them has no effect.
    first = learned(logits, features, mass, owners, [op], ptr)["logits"]
    second = learned(logits, 100 * features, mass, owners, [op], ptr)["logits"]
    assert torch.equal(first, second)


def test_attention_uses_embeddings_while_pooling_values_remain_local_logits():
    """Attention is the distinct embedding-conditioned family already explored in V5a."""
    for variant in ("simple", "gated"):
        head = _head("attention", attention_variant=variant)
        embeddings = torch.randn(4, 3, requires_grad=True)
        values = torch.tensor([-2.0, -0.5, 0.7, 3.0], requires_grad=True)
        result = _pool(head, values, embeddings=embeddings)
        assert torch.allclose(
            result["logits"], (result["attention_weights"] * values).sum()[None]
        )
        result["logits"].sum().backward()
        assert embeddings.grad is not None and embeddings.grad.abs().sum() > 0


def test_expanded_constructor_bounds_restore_without_reinterpreting_historical_models():
    """Saved constructor limits restore exact weights; omitted limits retain historical domains."""
    options = {
        "hidden_dim": 4, "embedding_dim": 2, "atomic_layers": 1, "surface_layers": 1,
        "pooling_type": "autopool", "autopool_alpha_mode": "learned",
        "autopool_alpha_init": 500, "autopool_alpha_bounds": (0, 2000),
    }
    model, saved = _create_model(2, options)
    restored, _ = _create_model(2, saved)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert saved["autopool_alpha_bounds"] == (0, 2000)
    assert restored.pooling_head.parameter_values() == model.pooling_head.parameter_values()
    historical = _head("autopool", autopool_alpha_mode="learned", autopool_alpha_init=5)
    assert historical.learned_scalar.upper_bound == 50
    assert _head("local_mean_max", regional_scale_mode="learned").learned_scalar.upper_bound == 12
    # Physical domain restrictions are scientific invariants, not arbitrary numerical checks.
    with pytest.raises(ValueError, match="non-negative"):
        _head("autopool", autopool_alpha_mode="learned", autopool_alpha_bounds=(-1, 10))
    with pytest.raises(ValueError, match="at least one"):
        _head("gem", gem_power_mode="learned", gem_power_bounds=(0.5, 10))


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
    # Freeze the reviewed scientific ledger directly: a retired legacy YAML is not an input.
    ledger = {
        "model_version": 2, "subset": "full", "initialization_profile": "baseline",
        "optimization_profile": "baseline", "hidden_dim": 128, "embedding_dim": 32,
        "residue_embedding_dim": 64, "atomic_layers": 2, "projection_depth": 3,
        "surface_layers": 4, "atom_spatial_k": 8, "surface_atom_k": 16,
        "diffusion_spectral_modes": 32, "surface_atom_radius": 4.0,
        "dropout": 0.2, "learning_rate": 0.001, "weight_decay": 0.0005,
        "gate_lambda": 0.04, "weak_loss_profile": "baseline", "architecture_spike": "baseline",
    }
    expected = {
        "v5/wisdom_v5a.yaml": [1, 2, 32, 26, 16, 54, 2, 26, 24, 22, 1],
        "v5/wisdom_v5b.yaml": [1, 1, 1, 1, 1],
    }
    for name, counts in expected.items():
        raw = safe_load((root / name).read_text())
        assert "steps" not in raw and "search" not in raw
        assert raw["seeds"] == [4, 7, 32, 54]
        # Host allocations are operational choices, not frozen scientific hyperparameters.
        config = WorkConfig.from_yaml(root / name)
        resources = config.resources
        assert resources.gpu_count >= 1 and resources.cpu_cores >= 1
        for key, value in ledger.items():
            assert raw["with"][key] == value, (name, key)
        # Planning invokes the installed public LF expander but executes no scientific Work.
        plan = WorkRunner().plan(config)
        assert [len(level) for level in plan.levels] == [sum(counts) * 4]
        assert len(plan.preflight["studies"]) == 1
        facts = plan.preflight["studies"][0]
        assert facts["candidates"] == sum(counts)
        assert facts["required_runs"] == sum(counts) * 4
        assert facts["shared_seeds"] == [4, 7, 32, 54]
        assert facts["study_time_budget_seconds"] == (3600000 if "v5a" in name else 604800)
        assert facts["scheduler_wall_time_seconds"] == resources.runtime_seconds
        assert facts["requested_gpus"] == resources.gpu_count and facts["max_parallel"] is None
        assert facts["reference"] == ({"pooling_type": "max"} if "v5a" in name else None)
        assert [item["candidates"] for item in facts["conditional_branches"]["counts"]] == counts


def test_v5_inactive_parameters_are_absent_from_native_candidates():
    """Check exact active keys, including transitive conditions on adaptive scalars."""
    root = Path(__file__).parents[1] / "experiments"
    characterization = {
        "max": set(),
        "mean": {"pooling_area_mode"},
        "attention": {"pooling_area_mode", "attention_variant", "attention_hidden_dim"},
        "topk": {"pooling_area_mode", "topk_fraction"},
        "local_mean_max": {"regional_diffusion_scale"},
        "log_sum_exp": {"pooling_area_mode", "log_sum_exp_mode"},
        "linear_softmax": {"pooling_area_mode"},
        "autopool": {"pooling_area_mode", "autopool_alpha"},
        "gem": {"pooling_area_mode", "gem_power"},
        "max_mean": {"pooling_area_mode", "max_mean_lambda"},
        "multiscale_regional_max": set(),
    }
    adaptive = {
        "log_sum_exp": {"pooling_area_mode", "log_sum_exp_mode", "log_sum_exp_beta_bounds"},
        "local_mean_max": {
            "regional_scale_mode", "regional_diffusion_scale_init",
            "regional_diffusion_scale_bounds",
        },
        "autopool": {"autopool_alpha_mode", "autopool_alpha_bounds"},
        "gem": {"gem_power_mode", "gem_power_bounds"},
        "max_mean": {"max_mean_lambda_mode"},
    }
    for name, branches in [("v5a", characterization), ("v5b", adaptive)]:
        config = WorkConfig.from_yaml(root / "v5" / f"wisdom_{name}.yaml")
        run = config.levels[0].runs[0]
        for candidate in run.variants:
            family = candidate["pooling_type"]
            keys = {"pooling_type"} | branches[family]
            if family == "log_sum_exp":
                mode = candidate["log_sum_exp_mode"]
                keys |= {
                    "fixed": {"log_sum_exp_beta"},
                    "learned": {"log_sum_exp_beta_init"},
                    "curriculum": {
                        "pooling_curriculum_end_beta",
                        "pooling_curriculum_hold_fraction",
                    },
                }[mode]
            if name == "v5b":
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


def test_v5a_curriculum_changes_before_validation_patience_can_expire():
    """Ensure the configured schedule has a chance to act without disabling early stopping."""
    root = Path(__file__).parents[1] / "experiments"
    run = WorkConfig.from_yaml(root / "v5/wisdom_v5a.yaml").levels[0].runs[0]
    curriculum = [c for c in run.variants if c.get("log_sum_exp_mode") == "curriculum"]
    assert len(curriculum) == 22
    assert run.parameters["epochs"] == 500 and run.parameters["patience"] == 30
    for candidate in curriculum:
        assert candidate["pooling_curriculum_hold_fraction"] == 0.0
        # Training computes progress=(epoch-1)/epochs; zero hold changes beta at epoch 2.
        fraction = (2 - 1) / run.parameters["epochs"]
        beta = 1.0 + fraction * (candidate["pooling_curriculum_end_beta"] - 1.0)
        if candidate["pooling_curriculum_end_beta"] == 1.0:
            assert beta == 1.0  # Explicit constant-beta schedule control, not a delayed change.
        else:
            assert 1.0 < beta < candidate["pooling_curriculum_end_beta"]


def test_v5a_covers_every_family_and_expands_old_upper_edges():
    """Require both mathematical endpoints and substantial continuation past old grid edges."""
    root = Path(__file__).parents[1] / "experiments"
    raw = safe_load((root / "v5/wisdom_v5a.yaml").read_text())
    space = raw["sweep"]["space"]
    assert set(space["pooling_type"]["values"]) == {family.value for family in PoolingType}
    assert set(space["attention_variant"]["values"]) == {"simple", "gated"}
    assert set(space["log_sum_exp_mode"]["values"]) == {"fixed", "curriculum"}
    assert {0.6, 0.8, 1.0}.issubset(space["topk_fraction"]["values"])
    assert {0.0, 1.0}.issubset(space["max_mean_lambda"]["values"])
    for key, lower, upper in [
        ("attention_hidden_dim", 4, 512),
        ("regional_diffusion_scale", 0, 64),
        ("log_sum_exp_beta", 0.01, 1280),
        ("pooling_curriculum_end_beta", 1, 1280),
        ("autopool_alpha", 0, 1000),
        ("gem_power", 1, 1024),
    ]:
        values = space[key]["values"]
        assert min(values) == lower and max(values) == upper
        assert values == sorted(set(values))


@pytest.mark.parametrize("name", ["v5a", "v5b"])
def test_every_authored_pooling_candidate_has_finite_forward_and_backward(name):
    """Execute all native candidate heads, including extreme controls and bounded initializations.

    Args:
        name: Characterization or learned-scalar refinement configuration name.

    These synthetic numerical checks establish executability, not scientific superiority.
    """
    root = Path(__file__).parents[1] / "experiments"
    run = WorkConfig.from_yaml(root / "v5" / f"wisdom_{name}.yaml").levels[0].runs[0]
    accepted = set(inspect.signature(ProteinPoolingHead).parameters)
    for candidate in run.variants:
        parameters = {key: value for key, value in candidate.items() if key in accepted}
        head = ProteinPoolingHead(3, dropout=0, **parameters).eval()
        if candidate.get("log_sum_exp_mode") == "curriculum":
            head.set_log_sum_exp_beta(candidate["pooling_curriculum_end_beta"])
        values = torch.tensor([-2.0, -0.25, 0.5, 1.75], requires_grad=True)
        areas = torch.tensor([1.0, 3.0, 2.0, 1.0])
        pooled = _pool(head, values, areas)["logits"]
        assert torch.isfinite(pooled).all(), candidate
        pooled.sum().backward()
        assert values.grad is not None and torch.isfinite(values.grad).all(), candidate
        assert all(p.grad is None or torch.isfinite(p.grad).all() for p in head.parameters())


@pytest.mark.parametrize("area", ["point", "area"])
def test_expanded_fixed_pooling_endpoints_recover_their_controls(area):
    """Verify exact endpoints and finite high-sharpness limits on moderate local logits.

    Args:
        area: Point-count or represented-area measure used consistently by compared controls.
    """
    values = torch.tensor([-2.0, -0.25, 0.5, 1.75])
    areas = torch.tensor([1.0, 3.0, 2.0, 1.0])
    mean = _pool(_head("mean", area), values, areas)["logits"]
    maximum = values.max()[None]
    topk = _pool(_head("topk", area, topk_fraction=1), values, areas)["logits"]
    assert torch.allclose(topk, mean, atol=1e-6)
    for weight, target in [(0.0, mean), (1.0, maximum)]:
        mixture = _pool(_head("max_mean", area, max_mean_lambda=weight), values, areas)["logits"]
        assert torch.allclose(mixture, target, atol=1e-6)
    for family, settings, tolerance in [
        ("log_sum_exp", {"log_sum_exp_beta": 1280}, 0.01),
        ("autopool", {"autopool_alpha": 1000}, 0.01),
        ("gem", {"gem_power": 1024}, 0.02),
    ]:
        pooled = _pool(_head(family, area, **settings), values, areas)["logits"]
        assert torch.allclose(pooled, maximum, atol=tolerance), (family, area, pooled)
