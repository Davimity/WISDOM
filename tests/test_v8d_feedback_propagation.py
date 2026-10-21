"""Finite V8d controls on synthetic disjoint structures, with no scientific Study execution."""

# ruff: noqa: I001

import io
import torch
import pytest

from pathlib import Path
from test_wisdom_v2 import _v2
from test_v8_architectures import batch, model as historical_model
from lambdaforge.work import WorkConfig, WorkRunner
from wisdom.Training import _model_inputs, _create_model
from wisdom.models.FeedbackPropagationMode import FeedbackPropagationMode
from wisdom.models.SurfaceAtomFeedback import SurfaceAtomFeedback
from wisdom.models.PostFeedbackPropagation import PostFeedbackPropagation


MODES = [f"d{index}" for index in range(9)]


def model(mode, head="single", pooling="max", **parameters):
    """Construct a small end-to-end model, preserving both historical head contracts."""
    return _v2(
        pooling,
        feedback_propagation=mode,
        architecture_implementation="controlled",
        head_implementation="controlled",
        head_type=head,
        feedback_weighting="area",
        gate_cost_normalization="baseline_count",
        surface_refiner_type="heat",
        surface_refiner_heat_length=6.0,
        **parameters,
    )


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("head,pooling", [("single", "max"), ("film", "attention")])
def test_modes_shapes_gradients_and_strict_checkpoints(mode, head, pooling):
    """All nine modes keep final heads, finite end-to-end gradients and exact saved-state reuse."""
    torch.manual_seed(151)
    predictor = model(mode, head, pooling).eval()
    data = batch()
    out = predictor(**_model_inputs(data))
    assert out["logits"].shape == (2,) and out["surface_logits"].shape == (12,)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], data["target"])
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)
    for prefix in ("atomic_encoder.", "surface_encoder."):
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for name, p in predictor.named_parameters()
            if name.startswith(prefix)
        )
    post = predictor.post_feedback_propagation
    if post is not None:
        assert post.scale.grad is not None and post.scale.grad.abs() > 0
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0 for p in post.transform.parameters()
        )
        assert not (
            {id(p) for p in post.parameters()}
            & {id(p) for p in predictor.atomic_encoder.parameters()}
        )
    if mode not in {"d0", "d1", "d2", "d7"}:
        assert any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in predictor.surface_atom_feedback.parameters()
        )
    stream = io.BytesIO()
    torch.save(predictor.state_dict(), stream)
    stream.seek(0)
    restored = model(mode, head, pooling).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(
        out["logits"], restored(**_model_inputs(data))["logits"], rtol=0, atol=0
    )


@pytest.mark.parametrize(
    "index,spike",
    tuple(
        enumerate(
            (
                "baseline",
                "depth_control",
                "depth_control_parameter_matched",
                "bidirectional",
            )
        )
    ),
)
@pytest.mark.parametrize("head", ["single", "film"])
def test_d0_d3_exactly_reproduce_existing_v8b(index, spike, head):
    """Historical paths retain every state key, weight and scalar output without migration."""
    torch.manual_seed(151)
    reference = historical_model(spike, head).eval()
    torch.manual_seed(151)
    candidate = model(f"d{index}", head, "attention").eval()
    assert candidate.state_dict().keys() == reference.state_dict().keys()
    candidate.load_state_dict(reference.state_dict(), strict=True)
    inputs = _model_inputs(batch())
    for name in ("logits", "surface_logits"):
        torch.testing.assert_close(
            candidate(**inputs)[name], reference(**inputs)[name], rtol=0, atol=0
        )


@pytest.mark.parametrize("mode", ["d4", "d5", "d6", "d7", "d8"])
def test_new_modules_preserve_common_initialization_and_parameter_counts(mode):
    """Added modules cannot shift shared RNG; local and graph capacities are recorded honestly."""
    torch.manual_seed(157)
    reference = model("d3").state_dict()
    torch.manual_seed(157)
    predictor = model(mode)
    for name, value in reference.items():
        if name in predictor.state_dict():
            assert torch.equal(value, predictor.state_dict()[name]), name
    post = predictor.post_feedback_propagation
    width = predictor.hidden_dim
    layers = FeedbackPropagationMode(mode).layers()
    expected = (
        4 * width**2 + 3 * width if mode == "d8" else layers * (2 * width**2 + 2 * width + 88)
    ) + 1
    assert sum(p.numel() for p in post.parameters()) == expected
    if mode != "d8":
        assert isinstance(post.transform.input_projection, torch.nn.Identity)


def test_feedback_reaches_an_uncovered_neighbor_only_after_graph_propagation():
    """Central hypothesis: A's surface update reaches B via an edge, not through nodewise U."""
    torch.manual_seed(163)
    atoms = torch.randn(3, 4)
    surface = torch.randn(1, 4, requires_grad=True)
    feedback = SurfaceAtomFeedback(4, weighting="area")
    tables = (torch.tensor([[0]]), torch.tensor([[1.0]]), torch.tensor([[True]]), 4.0)
    real = feedback(atoms, surface, *tables, area_weights=torch.ones(1))
    null = feedback(atoms, None, *tables, area_weights=torch.ones(1))
    torch.testing.assert_close(real[1:], atoms[1:], rtol=0, atol=0)
    graph = (
        torch.tensor([[0], [1]]),
        torch.tensor([True]),
        torch.tensor([True]),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
        torch.ones(1),
    )
    gates = {name: torch.tensor(1.0) for name in model("d3").semantic_gates.names}
    post = PostFeedbackPropagation(4, layers=1, dropout=0, initial_scale=1.0)
    propagated = post(real, *graph, gates)
    alternative = post(null, *graph, gates)
    assert not torch.allclose(propagated[1], alternative[1])
    propagated[1].square().sum().backward()
    assert surface.grad is not None and surface.grad.abs().sum() > 0
    local = PostFeedbackPropagation(4, nodewise=True, dropout=0, initial_scale=1.0)
    torch.testing.assert_close(
        local(real, *graph, gates)[1], local(null, *graph, gates)[1], rtol=0, atol=0
    )


@pytest.mark.parametrize("mode", ["d4", "d5", "d6", "d7", "d8"])
def test_zero_gamma_is_exact_identity_without_outer_normalization(mode):
    """Zero scale bypasses the complete nonlinear transform, not merely its edge residual."""
    predictor = model(mode, post_feedback_initial_scale=0.0).eval()
    data = batch()
    captured = []
    hook = predictor.post_feedback_propagation.register_forward_hook(
        lambda module, args, out: captured.append((args[0].detach(), out.detach()))
    )
    predictor(**_model_inputs(data))
    hook.remove()
    torch.testing.assert_close(*captured[0], rtol=0, atol=0)


@pytest.mark.parametrize("mode", MODES)
def test_empty_support_isolation_and_rigid_permutation(mode):
    """Unequal proteins, missing candidates and coherent atom relabeling preserve isolation."""
    predictor = model(mode).eval()
    data = batch()
    data["surface_atom_mask"] = data["surface_atom_mask"].clone()
    data["surface_atom_mask"][0] = False
    inputs = _model_inputs(data)
    expected = predictor(**inputs)["logits"]
    moved = dict(inputs)
    permutation = torch.tensor([2, 0, 1, 6, 3, 5, 4])
    inverse = torch.argsort(permutation)
    atom_names = (
        "atomic_numbers",
        "residue_type_ids",
        "atom_role_ids",
        "atom_hybridization_ids",
        "formal_charges",
        "atom_aromaticity",
        "atom_hbond_donor",
        "atom_hbond_acceptor",
        "residue_hydropathy",
        "residue_polarity",
        "atom_positions",
        "atom_batch",
    )
    for name in atom_names:
        moved[name] = inputs[name][permutation]
    moved["atom_positions"] = (
        moved["atom_positions"] @ torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        + 11
    )
    moved["atom_edge_index"] = inverse[inputs["atom_edge_index"]].sort(dim=0).values
    neighbors = inputs["surface_atom_neighbors"]
    moved["surface_atom_neighbors"] = torch.where(
        neighbors >= 0, inverse[neighbors.clamp_min(0)], neighbors
    )
    torch.testing.assert_close(expected, predictor(**moved)["logits"], rtol=3e-5, atol=3e-6)
    data["atomic_numbers"] = data["atomic_numbers"].clone()
    data["atomic_numbers"][3:] = 8
    torch.testing.assert_close(
        expected[0], predictor(**_model_inputs(data))["logits"][0], rtol=0, atol=0
    )
    assert not any("target" in name for name in _model_inputs(data))


@pytest.mark.parametrize("mode", ["d3", "d5", "d6", "d8"])
def test_checkpointing_preserves_outputs_rng_and_parameter_gradients(mode):
    """Non-reentrant recomputation keeps live feedback and the single shared gate sample."""
    parameters = {
        "hidden_dim": 8,
        "embedding_dim": 4,
        "surface_layers": 2,
        "atomic_layers": 2,
        "dropout": 0.2,
        "feedback_propagation": mode,
        "architecture_implementation": "controlled",
        "feedback_weighting": "area",
    }
    plain, _ = _create_model(2, parameters)
    checked, _ = _create_model(
        2,
        parameters
        | {
            "checkpoint_surface": True,
            "checkpoint_post_feedback": True,
        },
    )
    plain.train()
    checked.train()
    checked.load_state_dict(plain.state_dict(), strict=True)
    observed = []
    for predictor in (plain, checked):
        torch.manual_seed(167)
        inputs = _model_inputs(batch())
        output = predictor(**inputs)["logits"]
        output.square().sum().backward()
        observed.append(
            (
                output.detach(),
                {
                    name: p.grad.clone()
                    for name, p in predictor.named_parameters()
                    if p.grad is not None
                },
                torch.get_rng_state(),
            )
        )
    torch.testing.assert_close(observed[0][0], observed[1][0], rtol=0, atol=0)
    assert torch.equal(observed[0][2], observed[1][2])
    assert observed[0][1].keys() == observed[1][1].keys()
    for name in observed[0][1]:
        torch.testing.assert_close(observed[0][1][name], observed[1][1][name], rtol=1e-5, atol=1e-7)


@pytest.mark.parametrize("budget", [(4, 4), (2, 2), (3, 1), (1, 3)])
def test_explicit_surface_budget_and_checkpoint_constructor(budget):
    """Compact depth is a labelled scientific factor, never a resource-driven fallback."""
    predictor, parameters = _create_model(
        2,
        {
            "hidden_dim": 8,
            "embedding_dim": 4,
            "feedback_propagation": "d5",
            "architecture_implementation": "controlled",
            "surface_stage_layers": budget,
        },
    )
    assert len(predictor.surface_encoder.blocks) == budget[0]
    assert len(predictor.second_surface_encoder.blocks) == budget[1]
    restored, _ = _create_model(2, parameters)
    restored.load_state_dict(predictor.state_dict(), strict=True)


def test_one_forward_shares_sparse_operators_and_candidate_tables():
    """Two surface stages consume identical tensor objects, not duplicated physical geometry."""
    predictor = model("d5").eval()
    stages = []
    tables = []
    handles = [
        encoder.register_forward_pre_hook(lambda module, args: stages.append(args[1]))
        for encoder in (predictor.surface_encoder, predictor.second_surface_encoder)
    ]
    handles += [
        transfer.register_forward_pre_hook(lambda module, args: tables.append(args[1]))
        for transfer in (predictor.surface_atom_transfer, predictor.second_surface_transfer)
    ]
    predictor(**_model_inputs(batch()))
    for handle in handles:
        handle.remove()
    assert stages[0] is stages[1] and tables[0] is tables[1]
    assert stages[0][0]["stacked_gradient"] is stages[1][0]["stacked_gradient"]


@pytest.mark.parametrize("mode", ["d5", "d6", "d8"])
def test_bfloat16_checkpointing_remains_finite(mode):
    """Dense AMP and FP32 sparse/feedback accumulation compose with recomputation."""
    predictor = model(mode, checkpoint_surface=True, checkpoint_post_feedback=True).train()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = predictor(**_model_inputs(batch()))
        loss = output["logits"].square().sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


def test_null_feedback_and_oom_are_explicit_not_successful_fallbacks(monkeypatch):
    """D6 passes None to U, and OOM propagates to the existing LF Attempt boundary unchanged."""
    predictor = model("d6").eval()
    observed = []
    hook = predictor.surface_atom_feedback.register_forward_pre_hook(
        lambda module, args, kwargs: observed.append(args[1]),
        with_kwargs=True,
    )
    predictor(**_model_inputs(batch()))
    hook.remove()
    assert observed == [None]

    def exhausted(*args, **kwargs):
        raise torch.cuda.OutOfMemoryError("synthetic injected resource failure")

    monkeypatch.setattr(predictor.post_feedback_propagation, "forward", exhausted)
    with pytest.raises(torch.cuda.OutOfMemoryError, match="synthetic injected"):
        predictor(**_model_inputs(batch()))


def test_invalid_or_unbounded_mechanisms_are_not_silent_fallbacks():
    """Reject unrelated V9/V8 combinations and unsupported budgets without altering topology."""
    for options in (
        {"feedback_propagation": "d9"},
        {"feedback_propagation": "d5", "atomic_encoder_type": "gine"},
        {"feedback_propagation": "d5", "surface_stage_layers": [4, 4, 4]},
    ):
        with pytest.raises(ValueError):
            _v2("max", architecture_implementation="controlled", **options)


def test_native_v8d_plan_is_frozen_paired_and_not_historical_recovery():
    """All prepared main comparisons preserve the audited V8b ledger and remain new Runs."""
    root = Path(__file__).parents[1] / "experiments/v8"
    config = WorkConfig.from_yaml(root / "wisdom_v8d.yaml")
    settings = config.to_dict()
    assert settings["name"] == "wisdom-v8d" and config.planned_runs == 32
    assert not {"steps", "search"} & settings.keys()
    assert settings["with"]["evaluate_test"] is False
    assert settings["with"]["surface_layers"] == 4
    historical = WorkConfig.from_yaml(root / "wisdom_v8b.yaml").to_dict()
    for name, value in historical["with"].items():
        assert settings["with"][name] == value
    settings["with"]["dataset"] = {"file": str(Path(__file__).resolve())}
    fixture = WorkConfig.from_mapping(settings, source=config.source)
    assert not fixture.validation_errors(check_inputs=True)
    plan = WorkRunner().run(fixture, dry_run=True).to_dict()
    assert sum(len(level) for level in plan["levels"]) == 32
