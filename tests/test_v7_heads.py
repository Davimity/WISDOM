"""Small deterministic controls for the four V7 hypotheses, without fitting real data."""

# ruff: noqa: I001
# Imports follow the project's shortest-line-first convention rather than isort grouping.

import io
import copy
import torch
import pytest

from pathlib import Path
from torch.nn import functional as F
from test_wisdom_v2 import _v2, _line_operator
from test_wisdom_v1 import _sample, _model_inputs
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.Training import _calibrate_local_head_bias
from wisdom.models.ProteinTaskLoss import ProteinTaskLoss
from wisdom.models.GlobalSurfaceHeads import GlobalSurfaceHeads
from wisdom.analysis.WisdomAnalysisProfile import WisdomAnalysisProfile
from wisdom.evaluation.SurfaceFaithfulnessAudit import SurfaceFaithfulnessAudit
from wisdom.Training import _create_model, _evaluate, _validation_utility, _train_wisdom


HEADS = ("single", "dual", "global_context", "film")


def model(head, pooling="max", refiner="none", **options):
    """Build a controlled tiny head with optional constructor overrides."""
    return _v2(pooling, head_type=head, head_implementation="controlled",
               surface_refiner_type=refiner, surface_refiner_heat_length=6.0, **options)


def batch():
    """Collate unequal proteins, nonuniform represented areas and nontrivial spectra."""
    samples = [_sample(3, 5, 1.0), _sample(4, 7, 0.0)]
    for sample in samples:
        count = len(sample["surface_area_weights"])
        sample["surface_area_weights"] = torch.linspace(0.2, 2.0, count)
    result = dict(WisdomCollator()(samples))
    result["surface_operators"] = [_line_operator(5), _line_operator(7)]
    return result


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.parametrize("pooling", ["max", "log_sum_exp", "attention"])
@pytest.mark.parametrize("refiner", ["none", "heat"])
def test_shapes_pooling_backward_and_strict_restore(head, pooling, refiner):
    """Every hypothesis supports public poolings/refiners and an exact checkpoint roundtrip."""
    torch.manual_seed(37)
    predictor = model(head, pooling, refiner).eval()
    data = batch()
    output = predictor(**_model_inputs(data))
    assert output["logits"].shape == (2,)
    assert output["surface_logits"].shape == (12,)
    pooled = predictor.pool_refined_surface_logits(
        output["surface_logits"], output["surface_evidence_features"],
        data["surface_area_weights"], data["surface_batch"],
        data["surface_operators"], data["surface_ptr"],
    )["logits"]
    torch.testing.assert_close(output.get("surface_protein_logits", output["logits"]), pooled)
    if head != "single":
        assert torch.equal(output["logits"], output["direct_protein_logits"])
        expected_gap = (output["logits"].sigmoid() - pooled.sigmoid()).abs()
        torch.testing.assert_close(output["head_disagreement"], expected_gap)
    loss = ProteinTaskLoss(head, "normalized")(output, data["target"])["total"]
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)
    if head != "single":
        direct = predictor.global_surface_heads.global_head[-1]
        assert direct.weight.grad.abs().sum() > 0
    buffer = io.BytesIO()
    torch.save(predictor.state_dict(), buffer)
    buffer.seek(0)
    restored = model(head, pooling, refiner).eval()
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    again = restored(**_model_inputs(data))
    for name, value in output.items():
        torch.testing.assert_close(value, again[name], rtol=0, atol=0)


@pytest.mark.parametrize("head", HEADS)
def test_common_parameters_and_refiner_rng_are_paired(head):
    """Additional heads never consume the RNG used to initialize shared model components."""
    torch.manual_seed(8)
    reference = model("single", "attention", "learned_anisotropic")
    torch.manual_seed(8)
    candidate = model(head, "attention", "learned_anisotropic")
    shared = reference.state_dict()
    assert any(name.startswith("surface_refiner.") for name in shared)
    for name, value in shared.items():
        assert torch.equal(value, candidate.state_dict()[name]), name


@pytest.mark.parametrize("head", ["global_context", "film"])
def test_copied_conditioned_head_starts_from_single_and_skips_old_forward(head):
    """Identity modulation and copied classifier reproduce single, with no discarded forward."""
    torch.manual_seed(41)
    reference = model("single").eval()
    torch.manual_seed(41)
    candidate = model(head, local_head_initialization="copy").eval()
    calls = []
    handle = candidate.local_head.register_forward_hook(lambda *args: calls.append(1))
    data = _model_inputs(batch())
    expected = reference(**data)
    actual = candidate(**data)
    handle.remove()
    assert calls == []
    assert not candidate.local_head.weight.requires_grad
    torch.testing.assert_close(expected["surface_logits"], actual["surface_logits"], rtol=0, atol=0)
    torch.testing.assert_close(expected["logits"], actual["surface_protein_logits"], rtol=0, atol=0)


@pytest.mark.parametrize("head", HEADS)
def test_batch_isolation(head):
    """Changing every atom/point of protein two cannot alter protein one's predictions."""
    torch.manual_seed(16)
    predictor = model(head, "attention", "heat").eval()
    data = batch()
    changed = copy.deepcopy(data)
    changed["surface_curvatures"][5:] *= -15
    atom_second = changed["atom_batch"] == 1
    changed["atomic_numbers"][atom_second] = 26
    actual = predictor(**_model_inputs(data))
    altered = predictor(**_model_inputs(changed))
    torch.testing.assert_close(actual["logits"][:1], altered["logits"][:1], rtol=0, atol=0)
    torch.testing.assert_close(actual["surface_logits"][:5], altered["surface_logits"][:5],
                               rtol=0, atol=0)


@pytest.mark.parametrize("head", ["dual", "global_context", "film"])
@pytest.mark.parametrize("reduction", ["area_mean", "mean", "max"])
def test_reductions_and_active_parameter_routes(head, reduction):
    """Exact segmented reductions and only the selected trainable head modules are present."""
    torch.manual_seed(91)
    heads = GlobalSurfaceHeads(4, head, 3, global_reduction=reduction)
    points = torch.randn(7, 4, requires_grad=True)
    areas = torch.tensor([1., 2., 3., 4., 2., 1., 5.])
    owners = torch.tensor([0, 0, 0, 1, 1, 1, 1])
    result = heads(points, areas, owners, 2)
    expected = []
    for index in range(2):
        local = points[owners == index]
        weights = areas[owners == index]
        value = ((local * weights[:, None]).sum(0) / weights.sum()
                 if reduction == "area_mean" else
                 (local.mean(0) if reduction == "mean" else local.max(0).values))
        expected.append(value)
    torch.testing.assert_close(result["global_context"], torch.stack(expected))
    count = sum(p.numel() for p in heads.parameters())
    assert count == {"dual": 33, "global_context": 56, "film": 85}[head]
    (result["direct_logits"].sum() + result.get("conditioned_surface_logits", points * 0).sum()
     ).backward()
    assert all(p.grad is not None for p in heads.parameters())
    if head == "dual":
        assert not hasattr(heads, "context_projection")
    if head != "film":
        assert not hasattr(heads, "film_head")


@pytest.mark.parametrize("head", ["global_context", "film"])
def test_detach_cuts_only_context_input_route(head):
    """Detached local loss still updates the context projection and pointwise evidence path."""
    torch.manual_seed(29)
    detached = GlobalSurfaceHeads(4, head, 3, detach_context=True)
    # Nonzero FiLM modulation exposes its context gradient after the intentional zero-start step.
    if head == "film":
        torch.nn.init.normal_(detached.film_head.weight, std=0.1)
    connected = copy.deepcopy(detached)
    connected.detach_context = False
    points = torch.randn(7, 4, requires_grad=True)
    arguments = (torch.ones(7), torch.tensor([0, 0, 0, 1, 1, 1, 1]), 2)
    first = detached(points, *arguments)
    second = connected(points, *arguments)
    torch.testing.assert_close(first["conditioned_surface_logits"],
                               second["conditioned_surface_logits"])
    first["conditioned_surface_logits"].square().sum().backward(retain_graph=True)
    first_gradient = points.grad.clone()
    points.grad.zero_()
    second["conditioned_surface_logits"].square().sum().backward()
    assert first_gradient.abs().sum() > 0
    assert not torch.allclose(first_gradient, points.grad)
    assert detached.context_projection[0].weight.grad.abs().sum() > 0
    assert all(p.grad is None for p in detached.global_head.parameters())
    torch.testing.assert_close(detached.context_projection[0].weight.grad,
                               connected.context_projection[0].weight.grad)


def test_normalized_loss_priors_and_single_equivalence():
    """Equal branch predictions retain the exact task/prior ratio of original single BCE."""
    direct = torch.tensor([0.3, -0.5], requires_grad=True)
    surface = direct.detach().clone().requires_grad_()
    labels = torch.tensor([1., 0.])
    original = F.binary_cross_entropy_with_logits(direct, labels)
    single = ProteinTaskLoss("single", "normalized", surface_fraction=0.25)
    assert torch.equal(single({"logits": direct}, labels)["total"], original)
    multi = {"logits": direct, "surface_protein_logits": surface}
    normalized = ProteinTaskLoss("dual", "normalized")(multi, labels)["total"]
    legacy = ProteinTaskLoss("dual")(multi, labels)["total"]
    prior = 0.13 * direct.square().mean()
    assert torch.equal(normalized + prior, original + prior)
    assert torch.equal(legacy, 2 * original)
    normalized.backward()
    assert direct.grad.abs().sum() > 0 and surface.grad.abs().sum() > 0
    torch.testing.assert_close(direct.grad, surface.grad)


@pytest.mark.parametrize("fraction", [0.0, 0.25, 0.5, 0.75])
def test_exact_normalized_loss(fraction):
    """Every accepted fraction matches the explicit convex combination, including rho=0."""
    output = {"logits": torch.tensor([-2., 1.]),
              "surface_protein_logits": torch.tensor([0.4, -0.8])}
    labels = torch.tensor([0., 1.])
    result = ProteinTaskLoss("dual", "normalized", fraction)(output, labels)
    expected = (1 - fraction) * result["direct"] + fraction * result["surface"]
    assert torch.equal(result["total"], expected)


@pytest.mark.parametrize("options", [
    {"surface_fraction": 1.0, "balance": "normalized"},
    {"surface_fraction": -0.1}, {"surface_fraction": float("nan")},
    {"direct_weight": 0.0}, {"surface_weight": -1.0},
    {"surface_fraction": 0.25}, {"balance": "normalized", "direct_weight": 2.0},
])
def test_invalid_or_unsupervised_primary_head_is_rejected(options):
    """Invalid coefficients never launch a comparison with an unsupervised selection branch."""
    with pytest.raises(ValueError):
        ProteinTaskLoss("dual", **options)


@pytest.mark.parametrize("head", HEADS)
def test_mixed_precision_forward_backward(head):
    """CPU BF16 dense operations coexist with FP32 surface operators and reductions."""
    torch.manual_seed(14)
    predictor = model(head, "attention", "heat")
    data = batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = predictor(**_model_inputs(data))
        loss = ProteinTaskLoss(head, "normalized")(output, data["target"])["total"]
    assert torch.isfinite(loss)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


@pytest.mark.parametrize("head", HEADS)
def test_legacy_checkpoint_missing_new_options_restores_strictly(head):
    """Absent new options retain historical module keys and default weighted-sum semantics."""
    torch.manual_seed(7)
    old = _v2("max", head_type=head).eval()
    parameters = dict(hidden_dim=8, embedding_dim=4, atomic_layers=2, projection_depth=1,
                      surface_layers=2, dropout=0., curvature_features=6, atom_spatial_k=2,
                      surface_atom_k=16, diffusion_spectral_modes=8, head_type=head)
    restored, _ = _create_model(2, parameters)
    restored.load_state_dict(old.state_dict(), strict=True)
    restored.eval()
    data = _model_inputs(batch())
    torch.testing.assert_close(old(**data)["logits"], restored(**data)["logits"], rtol=0, atol=0)
    if head != "single":
        assert hasattr(restored.global_surface_heads, "context_local_head")
        assert hasattr(restored.global_surface_heads, "film_local_head")


def test_explicit_metric_branches_and_unavailable_refiner_audit():
    """Reports never confuse local re-pooling with dependence of the direct classifier."""
    predictor = model("dual", "max", "heat").eval()
    data = batch()
    primary, local = _evaluate(predictor, [data], torch.device("cpu"), None,
                               faithfulness_audit=True)
    assert primary["auprc"] == primary["direct_auprc"]
    assert "surface_derived_auprc" in primary
    assert _validation_utility(primary) == pytest.approx(
        0.7 * primary["direct_auprc"] + 0.3 * primary["direct_auroc"])
    assert not any("faithfulness_" in name for name in local)
    status = SurfaceFaithfulnessAudit.status(predictor)
    assert status["surface_branch"]["status"] == "unavailable"
    assert "operator" in status["surface_branch"]["reason"]
    assert status["direct_branch"]["status"] == "unavailable"
    assert "embedding" in status["direct_branch"]["reason"]
    catalog = WisdomAnalysisProfile().build()["metrics"]
    assert not catalog["val_direct_auprc"]["discovery"]
    assert "ONLY" in catalog["val_surface_branch_faithfulness_deletion_top_10"]["description"]


def test_surface_targets_do_not_affect_forward_or_task_gradients():
    """Local GT is evaluation-only: replacing it cannot change forward or protein-only losses."""
    predictor = model("global_context").eval()
    data = batch()
    data["surface_target_hard"] = torch.ones(12)
    first = predictor(**_model_inputs(data))
    loss = ProteinTaskLoss("global_context", "normalized")
    gradients = torch.autograd.grad(loss(first, data["target"])["total"],
                                    predictor.global_surface_heads.context_local_head.weight)
    data["surface_target_hard"].zero_()
    second = predictor(**_model_inputs(data))
    again = torch.autograd.grad(loss(second, data["target"])["total"],
                                predictor.global_surface_heads.context_local_head.weight)
    assert torch.equal(gradients[0], again[0])


@pytest.mark.parametrize("head", HEADS)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device unavailable")
def test_cuda_amp_forward_backward(head):
    """Exercise real CUDA mixed precision with the same FP32 sparse-operator boundary."""
    device = torch.device("cuda")
    predictor = model(head, "attention", "heat").to(device)
    data = _model_inputs(batch())
    inputs = {key: ([{name: tensor.to(device) for name, tensor in op.items()} for op in value]
                    if key == "surface_operators" else value.to(device))
              for key, value in data.items()}
    with torch.autocast("cuda", dtype=torch.float16):
        output = predictor(**inputs)
        loss = ProteinTaskLoss(head, "normalized")(output, torch.tensor([1., 0.], device=device))
    loss["total"].backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


def test_pending_ledger_stops_before_reading_dataset(tmp_path):
    """A valid YAML plan cannot accidentally run provisional V7 scientific parameters."""
    with pytest.raises(ValueError, match="carry-forward ledger is pending"):
        _train_wisdom(None, Path(tmp_path / "does-not-exist"), weak_loss_profile="PENDING_V6H")


@pytest.mark.parametrize("head", ["dual", "global_context", "film"])
def test_new_heads_respect_orthogonal_initialization(head):
    """Initialize new dense heads with the selected policy, then restore FiLM identity."""
    predictor = model(head, linear_initialization="orthogonal")
    matrix = predictor.global_surface_heads.global_head[1].weight
    torch.testing.assert_close(matrix @ matrix.T, torch.eye(8), rtol=1e-5, atol=1e-6)
    if head == "film":
        assert predictor.global_surface_heads.film_head.weight.count_nonzero() == 0
        assert predictor.global_surface_heads.film_head.bias.count_nonzero() == 0


def test_audit_ignores_independent_direct_logits():
    """A surface-only intervention must not be presented as direct-head functional evidence."""
    predictor = model("dual").eval()
    data = batch()
    output = predictor(**_model_inputs(data))
    first = SurfaceFaithfulnessAudit().compute(predictor, output, data)
    output["logits"] = torch.tensor([-1000., 1000.])
    second = SurfaceFaithfulnessAudit().compute(predictor, output, data)
    assert first == second


@pytest.mark.parametrize("head", ["dual", "global_context", "film"])
def test_calibration_uses_effective_local_head_and_surface_output(head):
    """Calibrate the effective local classifier, not a disconnected direct output."""
    predictor = model(head).eval()
    data = batch()
    before = predictor(**_model_inputs(data))["logits"].detach()
    inactive_bias = predictor.local_head.bias.detach().clone()
    bias = _calibrate_local_head_bias(predictor, [data], torch.device("cpu"), None, 1)
    result = predictor(**_model_inputs(data))
    torch.testing.assert_close(before, result["logits"], rtol=0, atol=0)
    assert float(result["surface_protein_logits"].sigmoid().mean().detach()) == pytest.approx(
        0.5, abs=1e-6)
    if head != "dual":
        assert torch.equal(inactive_bias, predictor.local_head.bias)
    assert torch.isfinite(torch.tensor(bias))


@pytest.mark.parametrize("options", [
    {"head_type": "single", "head_dropout": 0.1},
    {"head_type": "single", "global_reduction": "max"},
    {"head_type": "dual", "global_context_dim": 8},
    {"head_type": "dual", "detach_global_context": False},
    {"head_type": "dual", "film_modulation_scale": 2.0},
    {"head_type": "dual", "local_head_initialization": "copy"},
])
def test_controlled_heads_reject_inactive_nondefault_factors(options):
    """A conditional study cannot silently claim to vary parameters unused by a selected head."""
    with pytest.raises(ValueError):
        _v2("max", head_implementation="controlled", **options)


@pytest.mark.parametrize("initialization", ["current", "orthogonal", "activation_aware"])
def test_direct_classifier_initialization_is_paired_across_multi_heads(initialization):
    """Even custom reinitialization cannot shift common direct-head weights by variant size."""
    states = []
    for head in ("dual", "global_context", "film"):
        torch.manual_seed(42)
        predictor = model(head, linear_initialization=initialization)
        states.append(predictor.global_surface_heads.global_head.state_dict())
    for name, reference in states[0].items():
        assert torch.equal(reference, states[1][name]), name
        assert torch.equal(reference, states[2][name]), name
