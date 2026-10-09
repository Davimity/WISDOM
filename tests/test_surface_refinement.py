"""Mathematical, autograd, compatibility and native-planning contracts for V5c."""

import copy
import inspect
from dataclasses import replace
from pathlib import Path

import pytest
import torch
from lambdaforge.work import WorkConfig, WorkRunner
from test_wisdom_v1 import _sample
from test_wisdom_v2 import _line_operator, _v2
from torch.utils.data import DataLoader
from yaml import safe_load

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.evaluation.SurfaceFaithfulnessAudit import SurfaceFaithfulnessAudit
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.refinement import build_surface_evidence_refiner
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext
from wisdom.models.refinement.SurfaceEvidenceRefinerType import SurfaceEvidenceRefinerType
from wisdom.Training import Training, _create_model, _evaluate, _model_inputs


def _parameters(**overrides):
    """Use actual Training defaults so tests cannot invent a second configuration contract."""
    import_signature = inspect.signature(Training.run)
    parameters = {
        name: value.default
        for name, value in import_signature.parameters.items()
        if name.startswith("surface_refiner_")
    }
    return {**parameters, **overrides}


def _context(counts=(4, 3), embeddings=None):
    """Make tiny exact spectra and stored path neighbors for two disconnected proteins."""
    total = sum(counts)
    if embeddings is None:
        embeddings = torch.randn(total, 8, requires_grad=True)
    owners = torch.repeat_interleave(torch.arange(len(counts)), torch.tensor(counts))
    neighbors = torch.full((total, 2), -1, dtype=torch.long)
    start = 0
    for count in counts:
        for index in range(start, start + count):
            if index > start:
                neighbors[index, 0] = index - 1
            if index + 1 < start + count:
                neighbors[index, 1] = index + 1
        start += count
    return SurfaceEvidenceContext(
        embeddings=embeddings,
        area_weights=torch.ones(total),
        owners=owners,
        surface_ptr=torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()]),
        operators=[_line_operator(count) for count in counts],
        positions=torch.stack(
            (torch.arange(total).float(), torch.zeros(total), torch.zeros(total)), dim=1
        ),
        normals=torch.tensor([0.0, 0.0, 1.0]).expand(total, 3),
        curvatures=torch.zeros(total, 2, 3),
        neighbors=neighbors,
        neighbor_mask=neighbors >= 0,
    )


@pytest.mark.parametrize("kind", list(SurfaceEvidenceRefinerType))
def test_all_refiners_preserve_shape_finiteness_gradients_and_disjoint_domains(kind):
    """A protein BCE crosses the operator, and changing protein two cannot alter protein one."""
    torch.manual_seed(4)
    context = _context()
    refiner = build_surface_evidence_refiner(kind, 8, **_parameters())
    raw = torch.tensor([-1.3, 0.5, 1.7, -0.2, 0.3, -0.5, 1.2], requires_grad=True)
    refined = refiner(raw, context)
    assert refined.shape == raw.shape and torch.isfinite(refined).all()
    changed = raw.clone()
    changed[4:] = changed[4:] + 5.0
    assert torch.allclose(refiner(changed, context)[:4], refined[:4], atol=1e-6)
    torch.nn.functional.binary_cross_entropy_with_logits(
        torch.stack((refined[:4].mean(), refined[4:].mean())),
        torch.tensor([1.0, 0.0]),
    ).backward()
    assert raw.grad is not None and torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    learned = kind in {
        SurfaceEvidenceRefinerType.LEARNED_HEAT,
        SurfaceEvidenceRefinerType.LEARNED_ANISOTROPIC,
    }
    assert bool(list(refiner.parameters())) == learned
    if learned:
        for parameter in refiner.parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            assert parameter.grad.abs().sum() > 0


def test_identity_zero_heat_and_historical_regional_share_exact_control_and_primitive():
    """Zero returns the same tensor, not a truncated projection; regional remains heat/MAX."""
    context = _context((4,))
    raw = torch.tensor([-1.0, 2.0, 0.2, -0.5], requires_grad=True)
    for kind, kwargs in (("none", {}), ("heat", {"surface_refiner_heat_length": 0.0})):
        refiner = build_surface_evidence_refiner(kind, 8, **_parameters(**kwargs))
        assert refiner(raw, context) is raw
        assert not refiner.state_dict()
    heat = build_surface_evidence_refiner("heat", 8, **_parameters())
    expected = DiffusionSurfaceEncoder.diffuse(raw, context.operators, context.surface_ptr, 3.0)
    assert torch.equal(heat(raw, context), expected)
    regional = _v2("local_mean_max", regional_diffusion_scale=3.0).pooling_head
    pooled = regional(
        raw,
        context.embeddings,
        context.area_weights,
        context.owners,
        context.operators,
        context.surface_ptr,
    )
    assert torch.equal(pooled["logits"], heat(raw, context).max().reshape(1))


@pytest.mark.parametrize(
    "kind",
    ["geometric_anisotropic", "embedding_anisotropic", "learned_anisotropic", "graph_tv", "crf"],
)
def test_graph_isolation_and_no_neighbor_fallback(kind):
    """Padding and deliberately corrupt cross-protein neighbors cannot leak evidence."""
    context = _context((2, 1))
    context = replace(
        context,
        neighbors=torch.tensor([[1, 2], [0, 2], [0, -1]]),
        neighbor_mask=torch.tensor([[True, True], [True, True], [True, False]]),
    )
    raw = torch.tensor([-0.8, 1.2, 50.0], requires_grad=True)
    refiner = build_surface_evidence_refiner(kind, 8, **_parameters())
    result = refiner(raw, context)
    assert result[2] == raw[2]
    assert torch.equal(context.edges()[0], torch.tensor([0]))
    assert torch.equal(context.edges()[1], torch.tensor([1]))


@pytest.mark.parametrize("detach", [True, False])
def test_embedding_detach_affects_only_conductance_branch(detach):
    """Detached conductance blocks its guide gradients, but the local-head path still learns."""
    embeddings = torch.randn(4, 8, requires_grad=True)
    context = _context((4,), embeddings)
    refiner = build_surface_evidence_refiner(
        "embedding_anisotropic",
        8,
        **_parameters(surface_refiner_embedding_detach=detach),
    )
    raw = torch.tensor([-1.0, 0.5, 2.0, -0.3], requires_grad=True)
    refiner(raw, context).square().sum().backward()
    assert (embeddings.grad is None) == detach
    assert raw.grad is not None
    embeddings.grad = None
    connected_raw = embeddings.sum(-1)
    refiner(connected_raw, context).square().sum().backward()
    assert embeddings.grad is not None and embeddings.grad.abs().sum() > 0


def test_geometric_normalization_learned_symmetry_and_convex_range():
    """Sparse conductances are symmetric; normalized mean preserves a constant field."""
    context = _context((4,))
    left, right = context.edges()
    for kind in ("geometric_anisotropic", "embedding_anisotropic", "learned_anisotropic"):
        refiner = build_surface_evidence_refiner(kind, 8, **_parameters())
        weights = refiner.conductance(context, left, right)
        assert torch.allclose(weights, refiner.conductance(context, right, left))
        assert ((weights >= 0) & (weights <= 1)).all()
        assert torch.allclose(refiner(torch.ones(4), context), torch.ones(4))
        values = refiner(torch.tensor([-4.0, 2.0, -1.0, 6.0]), context)
        assert values.min() >= -4.0 and values.max() <= 6.0


def test_tv_stabilized_steps_reduce_energy_and_crf_has_attractive_sign():
    """TV decreases its documented energy; Potts smoothing attracts neighbors' binary beliefs."""
    context = _context((2,))
    raw = torch.tensor([2.0, -1.0], requires_grad=True)
    tv = build_surface_evidence_refiner("graph_tv", 8, **_parameters())
    left, right = context.edges()
    weight = context.weights(left, right, 2.0, 0.25, 1.0)

    def energy(values):
        return (
            0.5 * (values - raw).square().sum()
            + 0.05 * (weight * torch.sqrt((values[left] - values[right]).square() + 0.1**2)).sum()
        )

    assert energy(tv(raw, context)) < energy(raw)
    crf = build_surface_evidence_refiner(
        "crf",
        8,
        **_parameters(surface_refiner_crf_steps=1, surface_refiner_crf_damping=1.0),
    )
    expected = raw + 0.5 * (2.0 * raw.flip(0).sigmoid() - 1.0)
    assert torch.allclose(crf(raw, context), expected, atol=1e-6)
    for kind, kwargs in (
        ("graph_tv", {"surface_refiner_tv_lambda": 0.0}),
        ("crf", {"surface_refiner_crf_strength": 0.0}),
    ):
        assert build_surface_evidence_refiner(kind, 8, **_parameters(**kwargs))(raw, context) is raw


@pytest.mark.parametrize("kind", ["learned_heat", "learned_anisotropic"])
def test_learned_refiner_checkpoint_round_trip_and_heat_bounds(kind, tmp_path):
    """Weights and constructor parameters reconstruct the exact refined field and physical scale."""
    parameters = dict(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=1,
        projection_depth=1,
        surface_layers=1,
        curvature_features=6,
        dropout=0.0,
        surface_refiner_type=kind,
    )
    model, saved = _create_model(2, parameters)
    path = tmp_path / "model.pt"
    torch.save({"parameters": saved, "weights": model.state_dict()}, path)
    checkpoint = torch.load(path, weights_only=True)
    restored, _ = _create_model(2, checkpoint["parameters"])
    restored.load_state_dict(checkpoint["weights"], strict=True)
    context = _context((4,))
    raw = torch.tensor([-1.0, 0.5, 1.0, -2.0])
    assert torch.equal(model.surface_refiner(raw, context), restored.surface_refiner(raw, context))
    if kind == "learned_heat":
        assert 0.05 <= float(restored.surface_refiner.length().detach()) <= 12.0


@pytest.mark.parametrize("kind", list(SurfaceEvidenceRefinerType))
@pytest.mark.parametrize("pooling", ["max", "log_sum_exp", "attention"])
def test_v2_integrated_forward_pools_refined_and_bce_reaches_local_head(kind, pooling):
    """Exercise the actual atomic-transfer-DiffusionNet-local-refiner-pooling model."""
    context = _context((4, 3))
    samples = [_sample(3, count, label) for count, label in ((4, 1.0), (3, 0.0))]
    start = 0
    for sample, count in zip(samples, (4, 3), strict=True):
        sample["surface_positions"] = context.positions[start : start + count]
        sample["surface_normals"] = context.normals[start : start + count]
        sample["surface_neighbors"] = context.neighbors[start : start + count] - start
        sample["surface_neighbor_mask"] = context.neighbor_mask[start : start + count]
        sample["surface_neighbor_distances"] = torch.ones(count, 2)
        start += count
    batch = dict(WisdomCollator()(samples))
    model = _v2(pooling, surface_refiner_type=kind).eval()
    output = model(**_model_inputs(batch))
    assert output["surface_logits"] is output["refined_surface_logits"]
    assert torch.equal(output["surface_probabilities"], output["surface_logits"].sigmoid())
    if kind == "none":
        assert output["surface_logits"] is output["raw_surface_logits"]
        assert not any(name.startswith("surface_refiner.") for name in model.state_dict())
    expected = model.pool_refined_surface_logits(
        output["surface_logits"],
        output["surface_embeddings"],
        batch["surface_area_weights"],
        batch["surface_batch"],
        batch["surface_operators"],
        batch["surface_ptr"],
    )
    assert torch.equal(output["logits"], expected["logits"])
    if pooling == "attention":
        raw_pool = model.pooling_head(
            output["raw_surface_logits"],
            output["surface_embeddings"],
            batch["surface_area_weights"],
            batch["surface_batch"],
            batch["surface_operators"],
            batch["surface_ptr"],
        )
        assert torch.equal(output["attention_weights"], raw_pool["attention_weights"])
    torch.nn.functional.binary_cross_entropy_with_logits(
        output["logits"],
        batch["target"],
    ).backward()
    assert model.local_head.weight.grad is not None
    assert model.local_head.weight.grad.abs().sum() > 0
    if kind != "none":
        assert SurfaceFaithfulnessAudit().compute(model, output, batch) == ({}, 0)
        with pytest.raises(ValueError, match="original surface context"):
            model.pool_surface_logits(
                output["raw_surface_logits"],
                output["surface_embeddings"],
                batch["surface_area_weights"],
                batch["surface_batch"],
                batch["surface_operators"],
                batch["surface_ptr"],
            )


def test_native_v5c_sweep_has_30_candidates_120_runs_and_no_inactive_children():
    """Use native LF expansion to check independent factors, paired seeds and frozen backbone."""
    root = Path(__file__).parents[1] / "experiments"
    config = WorkConfig.from_yaml(root / "wisdom_v5c.yaml")
    plan = WorkRunner().plan(config)
    assert len(plan.levels[0]) == 120
    assert plan.preflight["studies"][0]["candidates"] == 30
    assert plan.preflight["studies"][0]["shared_seeds"] == [4, 7, 32, 54]
    authored = safe_load((root / "wisdom_v5c.yaml").read_text())
    baseline = safe_load((root / "wisdom_v5b.yaml").read_text())
    assert authored["with"]["faithfulness_audit"] is False
    assert authored["with"]["visualization"]["mode"] == "none"
    assert authored["with"]["evaluate_test"] is False
    for name, value in baseline["with"].items():
        if name not in {"faithfulness_audit"}:
            assert authored["with"][name] == value
    candidates = list(config.levels[0].runs[0].variants)
    assert len(candidates) == 30
    allowed = {
        "none": set(),
        "heat": {"heat_length"},
        "learned_heat": {"heat_length_init"},
        "geometric_anisotropic": {"strength", "steps"},
        "embedding_anisotropic": {"strength", "steps", "embedding_temperature", "embedding_detach"},
        "learned_anisotropic": {"steps", "learned_hidden_dim"},
        "graph_tv": {"tv_lambda", "tv_steps", "tv_step_size", "tv_epsilon"},
        "crf": {"crf_strength", "crf_steps", "crf_damping"},
    }
    for parameters in candidates:
        kind = parameters["surface_refiner_type"]
        actual = {
            name.removeprefix("surface_refiner_")
            for name in parameters
            if name.startswith("surface_refiner_") and name != "surface_refiner_type"
        }
        assert actual == allowed[kind]
        assert parameters["pooling_type"] in {"max", "log_sum_exp", "attention"}


@pytest.mark.parametrize("kind", ["none", "heat"])
def test_raw_refined_metrics_use_identical_targets_and_gain_is_not_clipped(kind):
    """A deterministic bad refinement must report a negative gain rather than hide it."""
    samples = [_sample(3, 4, 1.0), _sample(3, 3, 0.0)]
    samples[0]["surface_target_hard"] = torch.tensor([0, 1, 0, 1])
    samples[1]["surface_target_hard"] = torch.zeros(3, dtype=torch.long)
    for sample in samples:
        sample["surface_valid_mask"] = torch.ones(
            len(sample["surface_target_hard"]), dtype=torch.bool
        )
    loader = DataLoader(samples, batch_size=2, collate_fn=WisdomCollator())
    model = _v2("max", surface_refiner_type=kind).eval()
    batch = next(iter(loader))
    expected = model(**_model_inputs(batch))
    expected["raw_surface_logits"] = torch.tensor([-3.0, 3.0, -2.0, 2.0, -1.0, -2.0, -3.0])
    expected["surface_logits"] = (
        expected["raw_surface_logits"] if kind == "none" else -expected["raw_surface_logits"]
    )
    expected = {name: value.detach() for name, value in expected.items()}
    model.forward = lambda **kwargs: copy.copy(expected)
    _, metrics = _evaluate(model, loader, torch.device("cpu"), None)
    assert metrics["surface_raw_positive_macro_auprc"] == 1.0
    assert (metrics["surface_refinement_gain"] == 0.0 if kind == "none"
            else metrics["surface_refinement_gain"] < 0.0)
    assert metrics["surface_refinement_gain"] == (
        metrics["surface_positive_macro_auprc"] - metrics["surface_raw_positive_macro_auprc"]
    )
    assert metrics["surface_raw_negative_proteins"] == metrics["surface_negative_proteins"]
