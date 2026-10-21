"""Audited V8 controls on synthetic structures, without fitting or opening production splits."""

# ruff: noqa: I001

import io
import copy
import torch
import pytest

from pathlib import Path
from inspect import signature
from test_wisdom_v2 import _v2
from collections import Counter
from test_v7_heads import batch as _v7_batch
from lambdaforge.work import WorkConfig, WorkRunner
from wisdom.models.WeakSurfaceLoss import WeakSurfaceLoss
from wisdom.models.ArchitectureSpike import ArchitectureSpike
from wisdom.models.VectorAtomicState import VectorAtomicState
from wisdom.models.SurfaceAtomFeedback import SurfaceAtomFeedback
from wisdom.Training import Training, _create_model, _model_inputs


SPIKES = (
    "baseline",
    "point_geometry",
    "depth_control",
    "depth_control_parameter_matched",
    "bidirectional",
    "vector_8",
    "vector_16",
)


def batch():
    """Extend the deterministic unequal-protein fixture with its stored atom coordinates."""
    data = _v7_batch()
    data["atom_positions"] = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [3.0, 0.0, 0.0],
            [10.0, 0.0, 0.0],
            [11.0, 0.0, 0.0],
            [13.0, 0.0, 0.0],
            [16.0, 0.0, 0.0],
        ],
    )
    return data


def model(spike, head="single", **kwargs):
    """Construct one small corrected V8 candidate, keeping Heat6 and the head fixed."""
    return _v2(
        "attention",
        **ArchitectureSpike(spike).overrides(),
        architecture_implementation="controlled",
        feedback_weighting="area",
        gate_cost_normalization="baseline_count",
        head_type=head,
        head_implementation="controlled",
        surface_refiner_type="heat",
        surface_refiner_heat_length=6.0,
        **kwargs,
    )


@pytest.mark.parametrize("spike", SPIKES)
@pytest.mark.parametrize("head", ["single", "dual", "global_context", "film"])
def test_forward_backward_heat_priors_and_strict_checkpoint(spike, head):
    """All candidates accept every V7 head, global-label priors and exact strict restoration."""
    torch.manual_seed(151)
    predictor = model(spike, head).eval()
    data = batch()
    out = predictor(**_model_inputs(data))
    assert out["surface_logits"].shape == (12,) and out["logits"].shape == (2,)
    weak = WeakSurfaceLoss(negative_weight=0.03, positive_existence_weight=0.03)
    prior = weak(
        out["surface_logits"],
        data["surface_area_weights"],
        data["surface_batch"],
        data["target"],
        data["surface_operators"],
        data["surface_ptr"],
    )["total"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(out["logits"], data["target"])
    (loss + prior + 0.04 * predictor.gate_regularization()).backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)
    stream = io.BytesIO()
    torch.save(predictor.state_dict(), stream)
    stream.seek(0)
    restored = model(spike, head).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(
        out["logits"], restored(**_model_inputs(data))["logits"], rtol=0, atol=0
    )


@pytest.mark.parametrize("initialization", ["current", "orthogonal", "activation_aware"])
def test_shared_initialization_and_second_round_parameters_are_paired(initialization):
    """Additional modules never shift shared parameter draws, including custom reinitialization."""
    torch.manual_seed(167)
    baseline = model("baseline", "film", linear_initialization=initialization).state_dict()
    candidates = {}
    for spike in SPIKES:
        torch.manual_seed(167)
        candidate = model(spike, "film", linear_initialization=initialization)
        candidates[spike] = candidate.state_dict()
        for name, value in baseline.items():
            assert torch.equal(value, candidate.state_dict()[name]), (spike, name)
    for name, value in candidates["depth_control"].items():
        if name.startswith("second_surface_"):
            assert torch.equal(value, candidates["bidirectional"][name]), name
            assert torch.equal(value, candidates["depth_control_parameter_matched"][name]), name


def test_uncovered_atoms_remain_exactly_identical_and_covered_atoms_learn():
    """Demonstrate the historical coverage bug and the controlled residual identity fix."""
    torch.manual_seed(8)
    module = SurfaceAtomFeedback(4)
    atoms = torch.randn(3, 4, requires_grad=True)
    surface = torch.randn(2, 4, requires_grad=True)
    neighbors = torch.tensor([[0, -1], [1, -1]])
    distances = torch.zeros(2, 2)
    mask = neighbors >= 0
    out = module(atoms, surface, neighbors, distances, mask, 4.0)
    assert torch.equal(out[2], atoms[2])
    grad = torch.autograd.grad(out[2].sum(), atoms, retain_graph=True)[0]
    assert torch.equal(grad[2], torch.ones(4)) and grad[:2].count_nonzero() == 0
    (out * torch.randn_like(out)).sum().backward()
    assert surface.grad.abs().sum() > 0
    historical = SurfaceAtomFeedback(4, implementation="legacy")
    historical.load_state_dict(module.state_dict(), strict=True)
    assert not torch.equal(historical(atoms, surface, neighbors, distances, mask, 4.0)[2], atoms[2])
    assert torch.equal(module(atoms, surface, neighbors, distances, mask & False, 4.0), atoms)


def test_area_feedback_is_invariant_to_exact_surface_subdivision():
    """Exact area subdivision preserves the represented surface integral."""
    module = SurfaceAtomFeedback(4, weighting="area")
    atoms = torch.randn(2, 4)
    surfaces = torch.randn(2, 4)
    ids = torch.tensor([[0, 1], [0, 1]])
    distances = torch.tensor([[0.0, 1.0], [2.0, 1.0]])
    areas = torch.tensor([3.0, 1.0])
    mask = torch.ones_like(ids, dtype=torch.bool)
    original = module(atoms, surfaces, ids, distances, mask, 4.0, areas)
    duplicate = torch.tensor([0, 0, 1])
    split = module(
        atoms,
        surfaces[duplicate],
        ids[duplicate],
        distances[duplicate],
        mask[duplicate],
        4.0,
        torch.tensor([1.5, 1.5, 1.0]),
    )
    torch.testing.assert_close(original, split, rtol=1e-6, atol=1e-6)
    tiny = module(atoms, surfaces, ids, distances, mask, 4.0, areas * 1e-25)
    torch.testing.assert_close(original, tiny, rtol=1e-5, atol=1e-6)
    assert torch.equal(module(atoms, surfaces, ids, distances, mask, 4.0, areas * 0), atoms)


def test_null_control_has_equal_parameters_without_receiving_surface_information():
    """Matched control calls the identical update with None, not a detached real S1 message."""
    candidate = model("bidirectional")
    control = model("depth_control_parameter_matched")
    assert sum(p.numel() for p in candidate.parameters() if p.requires_grad) == sum(
        p.numel() for p in control.parameters() if p.requires_grad
    )
    seen = []
    hook = control.surface_atom_feedback.register_forward_pre_hook(
        lambda m, args: seen.append(args[1])
    )
    control(**_model_inputs(batch()))
    hook.remove()
    assert seen == [None]
    feedback = candidate.surface_atom_feedback
    atoms = torch.randn(4, 8, requires_grad=True)
    ids = torch.tensor([[0], [1]])
    output = feedback(atoms, None, ids, torch.ones(2, 1), ids >= 0, 4.0, torch.ones(2))
    assert torch.equal(output[2:], atoms[2:])
    output.square().sum().backward()
    assert feedback.update[0].weight.grad is not None
    assert feedback.update[0].weight.grad[:, 8:].count_nonzero() == 0


def test_gradient_returns_from_second_surface_through_atoms_to_first_surface():
    """The additional return path is differentiable; depth control receives no real atom update."""
    predictor = model("bidirectional").eval()
    observed = {}

    def surface_hook(module, args, output):
        """Retain S1 for an explicit feedback Jacobian test."""
        observed["first"] = output

    def feedback_hook(module, args, output):
        """Retain the updated atom tensor consumed by second transfer."""
        observed["atoms"] = output

    handles = [
        predictor.surface_encoder.register_forward_hook(surface_hook),
        predictor.surface_atom_feedback.register_forward_hook(feedback_hook),
    ]
    out = predictor(**_model_inputs(batch()))
    jacobian = torch.autograd.grad(
        (observed["atoms"] * torch.randn_like(observed["atoms"])).sum(),
        observed["first"],
        retain_graph=True,
    )[0]
    assert torch.isfinite(jacobian).all() and jacobian.abs().sum() > 0
    out["logits"].square().sum().backward()
    assert predictor.surface_atom_feedback.update[0].weight.grad.abs().sum() > 0
    assert predictor.second_surface_transfer.content_scorer[0].weight.grad.abs().sum() > 0
    for handle in handles:
        handle.remove()


def vector_fixture():
    """Provide a nonsymmetric four-atom graph whose edge ordering changes under renumbering."""
    states = torch.randn(4, 5)
    positions = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.2, 0.0], [0.3, 2.0, 1.0], [0.1, -1.0, 3.0]])
    edges = torch.tensor([[0, 0, 1, 2], [1, 3, 2, 3]])
    distances = (positions[edges[0]] - positions[edges[1]]).norm(dim=1)
    return states, positions, edges, distances


def test_vector_rotation_translation_permutation_and_edge_reversal():
    """Controlled coefficients remove numbering-dependent signs while vectors remain equivariant."""
    torch.manual_seed(3)
    module = VectorAtomicState(5, 3)
    states, positions, edges, distances = vector_fixture()
    scalar, vectors = module(states, positions, edges, distances)
    rotation = torch.linalg.qr(torch.randn(3, 3))[0]
    moved, rotated = module(states, positions @ rotation.T + 17.0, edges, distances)
    torch.testing.assert_close(scalar, moved, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(rotated, vectors @ rotation.T, rtol=1e-5, atol=2e-6)
    flipped_scalar, flipped_vectors = module(states, positions, edges.flip(0), distances)
    torch.testing.assert_close(scalar, flipped_scalar)
    torch.testing.assert_close(vectors, flipped_vectors)
    order = torch.tensor([3, 1, 0, 2])
    inverse = order.argsort()
    remapped = inverse[edges].sort(dim=0).values
    permuted, permuted_vectors = module(states[order], positions[order], remapped, distances)
    torch.testing.assert_close(scalar, permuted[inverse], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(vectors, permuted_vectors[inverse], rtol=1e-5, atol=1e-6)
    historical = VectorAtomicState(5, 3, implementation="legacy")
    historical.load_state_dict(module.state_dict(), strict=True)
    _, old = historical(states, positions, edges, distances)
    _, reversed_old = historical(states, positions, edges.flip(0), distances)
    assert not torch.allclose(old, reversed_old)
    empty, empty_vectors = module(states, positions, edges[:, :0], distances[:0])
    assert torch.equal(empty, states) and empty_vectors.count_nonzero() == 0
    duplicate_positions = positions * 0
    _, zero = module(states, duplicate_positions, edges, distances * 0)
    assert zero.count_nonzero() == 0 and torch.isfinite(zero).all()


@pytest.mark.parametrize("spike", SPIKES)
def test_protein_isolation_and_point_permutation(spike):
    """Feedback, vectors and point geometry never cross disjoint protein ownership."""
    predictor = model(spike).eval()
    original = batch()
    modified = copy.deepcopy(original)
    modified["surface_curvatures"][5:] += 10
    modified["atom_positions"][modified["atom_batch"] == 1] += 100
    modified["atomic_numbers"][modified["atom_batch"] == 1] = 26
    first = predictor(**_model_inputs(original))
    second = predictor(**_model_inputs(modified))
    torch.testing.assert_close(
        first["surface_logits"][:5], second["surface_logits"][:5], rtol=0, atol=0
    )
    # Simultaneously permute every surface row and its operators inside each protein.
    order = torch.tensor([4, 2, 0, 3, 1, 11, 9, 7, 5, 10, 8, 6])
    inverse = order.argsort()
    permuted = copy.deepcopy(original)
    for key, value in original.items():
        if key.startswith("surface_") and isinstance(value, torch.Tensor) and len(value) == 12:
            permuted[key] = value[order]
    for i, ids in enumerate((order[:5], order[5:] - 5)):
        op = permuted["surface_operators"][i]
        undo = ids.argsort()
        op["mass"] = op["mass"][ids]
        op["eigenvectors"] = op["eigenvectors"][ids]
        op["gradient_index"] = undo[op["gradient_index"]]
    after = predictor(**_model_inputs(permuted))
    torch.testing.assert_close(
        first["surface_logits"], after["surface_logits"][inverse], rtol=2e-5, atol=2e-6
    )


def test_fixed_gate_denominator_preserves_shared_gradients():
    """Adding one geometry gate does not dilute the baseline gate coefficients."""
    base = model("baseline")
    point = model("point_geometry")
    assert len(point.semantic_gates.names) == len(base.semantic_gates.names) + 1
    name = "atom.formal_charge"
    base.gate_regularization().backward()
    point.gate_regularization().backward()
    index = base.semantic_gates.names.index(name)
    assert torch.equal(
        base.semantic_gates.gates[index].log_alpha.grad,
        point.semantic_gates.gates[index].log_alpha.grad,
    )
    assert point.gate_regularization() > base.gate_regularization()


@pytest.mark.parametrize("spike", SPIKES)
def test_whole_model_atom_renumbering_and_rigid_motion(spike):
    """Renumbering atoms and moving the coordinate frame preserve final scalar predictions."""
    torch.manual_seed(151)
    predictor = model(spike).eval()
    data = batch()
    expected = predictor(**_model_inputs(data))
    moved = copy.deepcopy(data)
    rotation = torch.linalg.qr(torch.randn(3, 3))[0]
    moved["atom_positions"] = data["atom_positions"] @ rotation.T + 19.0
    moved["surface_positions"] = data["surface_positions"] @ rotation.T + 19.0
    moved["surface_normals"] = data["surface_normals"] @ rotation.T
    actual = predictor(**_model_inputs(moved))
    torch.testing.assert_close(expected["logits"], actual["logits"], atol=2e-6, rtol=2e-5)
    order = torch.tensor([2, 0, 1, 6, 4, 3, 5])
    inverse = order.argsort()
    permuted = copy.deepcopy(data)
    for key, value in data.items():
        if isinstance(value, torch.Tensor) and value.ndim and len(value) == 7:
            permuted[key] = value[order]
    permuted["atom_edge_index"] = inverse[data["atom_edge_index"]].sort(dim=0).values
    ids = data["surface_atom_neighbors"]
    permuted["surface_atom_neighbors"] = torch.where(ids >= 0, inverse[ids.clamp_min(0)], ids)
    actual = predictor(**_model_inputs(permuted))
    torch.testing.assert_close(
        expected["surface_logits"], actual["surface_logits"], rtol=2e-5, atol=2e-6
    )


def test_geometry_gate_off_is_exact_baseline_and_branch_receives_gradients():
    """New point geometry is a true candidate interaction and vanishes exactly when switched off."""
    torch.manual_seed(167)
    baseline = model("baseline").eval()
    torch.manual_seed(167)
    conditioned = model("point_geometry").eval()
    baseline.semantic_gates.set_override("all_on")
    conditioned.semantic_gates.set_override("all_on")
    conditioned.semantic_gates.force("transfer.point_geometry", 0)
    data = batch()
    expected = baseline(**_model_inputs(data))
    actual = conditioned(**_model_inputs(data))
    torch.testing.assert_close(expected["surface_logits"], actual["surface_logits"], rtol=0, atol=0)
    conditioned.semantic_gates.force("transfer.point_geometry", None)
    active = conditioned(**_model_inputs(data))
    active["logits"].square().sum().backward()
    projection = conditioned.surface_atom_transfer.point_geometry_projection
    assert projection.weight.grad.abs().sum() > 0 and torch.isfinite(projection.weight.grad).all()


@pytest.mark.parametrize("spike", SPIKES)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_amp_paths_and_peak_memory(spike):
    """Measure bounded synthetic CUDA allocations and exercise mixed-precision gradients."""
    predictor = model(spike).cuda().eval()
    data = _model_inputs(batch())
    moved = {
        key: (
            [{k: v.cuda() for k, v in operator.items()} for operator in value]
            if key == "surface_operators"
            else value.cuda()
        )
        for key, value in data.items()
    }
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.float16):
        output = predictor(**moved)
        loss = output["logits"].square().sum()
    loss.backward()
    assert torch.isfinite(loss)
    assert torch.cuda.max_memory_allocated() > 0
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


@pytest.mark.parametrize("spike", SPIKES)
def test_amp_is_finite_and_surface_gt_is_not_an_input(spike):
    """CPU BF16 supports the full return path; local targets cannot enter model inputs."""
    predictor = model(spike).eval()
    data = batch()
    data["surface_target_hard"] = torch.randn(12, requires_grad=True)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = predictor(**_model_inputs(data))
        loss = result["logits"].square().sum()
    loss.backward()
    assert torch.isfinite(loss) and data["surface_target_hard"].grad is None
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


def test_v8_campaigns_are_native_paired_two_line_fixed_designs():
    """Exact conditional candidates bind two frozen V7 lines to 16/32/24 mandatory Runs."""
    root = Path(__file__).parents[1] / "experiments/v8"
    ledgers = []
    for suffix, count, reference in (
        ("a", 16, "baseline"),
        ("b", 32, "depth_control"),
        ("c", 24, "baseline"),
    ):
        authored = WorkConfig.from_yaml(root / f"wisdom_v8{suffix}.yaml")
        fixture = authored.to_dict()
        fixture["with"]["dataset"] = {"file": str(Path(__file__).resolve())}
        # Only the typed binding changes. A dry-run never reads this stand-in as protein data
        # and publishes no fake dataset; production selectors are checked separately by LF CLI.

        config = WorkConfig.from_mapping(fixture, source=authored.source)
        assert config.planned_runs == count and not config.validation_errors(check_inputs=True)
        work = config.levels[0].runs[0]
        native = work.study_design.to_dict()
        assert work.seeds == (151, 157, 163, 167) and work.search_policy is None
        assert native["reference"] == {
            "pooling_type": "max", "head_type": "single", "architecture_spike": reference,
        }
        requirements = native["evidence"]["requirements"]
        assert Counter(r["seed"] for r in requirements) == dict.fromkeys(work.seeds, count // 4)
        resolved = config.to_dict()
        common = resolved["with"]
        ledgers.append(common)
        assert common["weak_loss_profile"] == "custom"
        assert common["evaluate_test"] is False
        assert common["architecture_implementation"] == "controlled"
        assert common["gate_cost_normalization"] == "baseline_count"
        assert common["feedback_weighting"] == "area"
        assert not {"head_type", "head_loss_balance", "surface_loss_fraction"} & common.keys()
        assert resolved["resources"]["gpu"] >= 1
        assert resolved.get("execution", {}).get("max_parallel") in {None, 1}
        assert resolved.get("execution", {}).get("runs_per_gpu") in {None, 1}
        assert not {"steps", "search"} & resolved.keys()

        # Sparse candidates, not the domain cross-product, define the mandatory evidence.
        # Each line must cover every architecture exactly once without inactive head inputs.

        candidates = resolved["sweep"]["candidates"]
        assert len(candidates) == count // 4
        assert len({repr(sorted(item.items())) for item in candidates}) == len(candidates)
        film_fields = {
            "head_loss_balance": "normalized", "surface_loss_fraction": 0.5,
            "head_dropout": 0.0, "global_reduction": "area_mean", "global_context_dim": 16,
            "detach_global_context": True, "local_head_initialization": "copy",
            "film_modulation_scale": 1.0,
        }
        attention_fields = {"attention_variant": "simple", "attention_hidden_dim": 256}
        variants = {
            "a": {"baseline", "point_geometry"},
            "b": {"baseline", "depth_control", "depth_control_parameter_matched", "bidirectional"},
            "c": {"baseline", "vector_8", "vector_16"},
        }[suffix]
        seen = {"max": set(), "attention": set()}
        for candidate in candidates:
            line = candidate["pooling_type"]
            seen[line].add(candidate["architecture_spike"])
            assert (common.keys() | candidate.keys()) <= signature(Training.run).parameters.keys()
            assert ArchitectureSpike(candidate["architecture_spike"]).overrides()
            if line == "max":
                assert candidate["head_type"] == "single"
                assert candidate["pooling_area_mode"] == "point"
                assert candidate["negative_surface_lambda"] == 0
                assert candidate["regional_ranking_lambda"] == 0
                assert candidate["regional_positive_lambda"] == 0.3
                assert not (film_fields.keys() | attention_fields.keys()) & candidate.keys()
            else:
                assert line == "attention" and candidate["head_type"] == "film"
                assert candidate["pooling_area_mode"] == "area"
                assert candidate["negative_surface_lambda"] == 0.03
                assert candidate["regional_positive_lambda"] == 0.03
                assert candidate["regional_ranking_lambda"] == 0.03
                assert all(candidate[key] == value for key, value in film_fields.items())
                assert all(candidate[key] == value for key, value in attention_fields.items())
            assert candidate.keys() == (resolved["sweep"]["space"].keys()
                                        if line == "attention" else
                                        resolved["sweep"]["space"].keys() - film_fields.keys()
                                        - attention_fields.keys())
        assert seen["max"] == seen["attention"] == variants
        plan = WorkRunner().run(config, dry_run=True).to_dict()
        assert sum(len(level) for level in plan["levels"]) == count
    assert ledgers[0] == ledgers[1] == ledgers[2]
    assert not (root / "wisdom_v8.yaml").exists()


@pytest.mark.parametrize(
    "spike",
    ["baseline", "point_geometry", "depth_control", "bidirectional", "vector_8", "vector_16"],
)
def test_historical_constructor_keys_remain_strictly_restorable(spike):
    """Absent V8 policy keys retain legacy layout and restore all archived trainable tensors."""
    parameters = dict(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=1,
        surface_layers=1,
        **ArchitectureSpike(spike).overrides(),
    )
    old, saved = _create_model(2, parameters)
    restored, _ = _create_model(2, saved)
    restored.load_state_dict(old.state_dict(), strict=True)
    assert (
        old.architecture_implementation.value
        == restored.architecture_implementation.value
        == "legacy"
    )
