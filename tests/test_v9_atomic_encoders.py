"""Public-LF atomic compositions on synthetic structures; no scientific training or test data."""

# ruff: noqa: I001

import io
import torch
import pytest

from pathlib import Path
from test_wisdom_v2 import _v2
from test_v8_architectures import batch
from lambdaforge.work import WorkConfig, WorkRunner
from wisdom.Training import _model_inputs, _create_model
from wisdom.models.AtomicEncoderType import AtomicEncoderType
from wisdom.models.AtomicGraphEncoder import AtomicGraphEncoder


KINDS = [kind.value for kind in AtomicEncoderType]


def model(kind):
    """Construct a tiny frozen MAX/Heat6 model while changing only its atomic operator."""
    return _v2(
        "max",
        architecture_implementation="controlled",
        head_implementation="controlled",
        gate_cost_normalization="baseline_count",
        surface_refiner_type="heat",
        surface_refiner_heat_length=6.0,
        atomic_encoder_type=kind,
        atomic_radial_count=4,
        atomic_vector_channels=4,
        atomic_basis_count=2,
        atomic_spatial_cutoff=6.0,
        atomic_degree_log_means=(0.8, 1.1),
        atomic_allow_dense_global=True,
    )


@pytest.mark.parametrize("kind", KINDS)
def test_forward_backward_and_strict_restore(kind):
    """Every public operator composes with Heat6 and the original protein BCE, without local GT."""
    torch.manual_seed(173)
    predictor = model(kind).eval()
    data = batch()
    result = predictor(**_model_inputs(data))
    assert result["logits"].shape == (2,)
    assert result["surface_logits"].shape == (12,)
    assert torch.isfinite(result["logits"]).all()
    loss = torch.nn.functional.binary_cross_entropy_with_logits(result["logits"], data["target"])
    (loss + 0.04 * predictor.gate_regularization()).backward()
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)
    stream = io.BytesIO()
    torch.save(predictor.state_dict(), stream)
    stream.seek(0)
    restored = model(kind).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(
        result["logits"], restored(**_model_inputs(data))["logits"], rtol=0, atol=0
    )


@pytest.mark.parametrize("kind", KINDS)
def test_shared_surface_initialization_and_protein_isolation(kind):
    """Operator construction cannot shift the common surface weights or mix separate proteins."""
    torch.manual_seed(179)
    reference = model("baseline").state_dict()
    torch.manual_seed(179)
    predictor = model(kind).eval()
    for name, value in reference.items():
        if not name.startswith("atomic_encoder."):
            assert torch.equal(value, predictor.state_dict()[name]), (kind, name)
    data = batch()
    expected = predictor(**_model_inputs(data))["logits"][0]
    data["atomic_numbers"] = data["atomic_numbers"].clone()
    data["atomic_numbers"][3:] = 8
    actual = predictor(**_model_inputs(data))["logits"][0]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.parametrize("kind", KINDS)
def test_atom_permutation_and_rigid_motion(kind):
    """Relabeling atoms and rotating/translating fixed geometry preserve scalar predictions."""
    predictor = model(kind).eval()
    data = batch()
    expected = predictor(**_model_inputs(data))["logits"]
    permutation = torch.tensor([2, 0, 1, 6, 3, 5, 4])
    inverse = torch.argsort(permutation)
    inputs = _model_inputs(data)
    atom_fields = (
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
    for name in atom_fields:
        inputs[name] = inputs[name][permutation]
    inputs["atom_edge_index"] = inverse[inputs["atom_edge_index"]].sort(dim=0).values
    neighbors = inputs["surface_atom_neighbors"]
    inputs["surface_atom_neighbors"] = torch.where(
        neighbors >= 0, inverse[neighbors.clamp_min(0)], neighbors
    )
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    inputs["atom_positions"] = inputs["atom_positions"] @ rotation.T + 11
    torch.testing.assert_close(expected, predictor(**inputs)["logits"], rtol=3e-4, atol=3e-5)


@pytest.mark.parametrize("kind", KINDS)
def test_cpu_bfloat16_composition_has_finite_backward(kind):
    """Mixed dense precision keeps the complete atomic/surface composition differentiable."""
    predictor = model(kind).eval()
    data = batch()
    with torch.autocast("cpu", dtype=torch.bfloat16):
        output = predictor(**_model_inputs(data))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["logits"], data["target"]
        )
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in predictor.parameters() if p.grad is not None)


@pytest.mark.parametrize(
    "kind", [value.value for value in AtomicEncoderType if value.needs_positions()]
)
def test_geometric_candidates_are_reflection_invariant(kind):
    """Polar-vector/distance mechanisms do not introduce an accidental frame or chirality cue."""
    predictor = model(kind).eval()
    inputs = _model_inputs(batch())
    expected = predictor(**inputs)["logits"]
    inputs["atom_positions"] = inputs["atom_positions"] * torch.tensor([-1.0, 1.0, 1.0])
    torch.testing.assert_close(expected, predictor(**inputs)["logits"], rtol=3e-4, atol=3e-5)


def test_masked_sources_are_closed_and_not_penalized():
    """Structural controls keep the reference gate denominator but cannot activate absent inputs."""
    predictor = model("gcn_structural")
    names = AtomicEncoderType.GCN_STRUCTURAL.inactive_sources()
    gates = predictor.semantic_gates
    gates.set_override("all_on")
    sampled = gates.sample(training=True)
    for name in names:
        assert sampled[name] == gates.effective_probability(name) == 0
        with pytest.raises(ValueError, match="absent source"):
            gates.force(name, 1)
    assert len(gates.names) == predictor.baseline_gate_count


@pytest.mark.parametrize("kind", KINDS)
def test_closed_relations_cannot_transmit_neighbor_content(kind):
    """A closed branch cannot leak a neighbour's changed embedding through biases or roots."""
    predictor = model(kind).eval()
    predictor.semantic_gates.force("atomic_graph.covalent", 0)
    predictor.semantic_gates.force("atomic_graph.spatial", 0)
    captured = []
    handle = predictor.atomic_encoder.register_forward_hook(
        lambda module, inputs, output: captured.append(output.detach().clone())
    )
    data = batch()
    predictor(**_model_inputs(data))
    data["atomic_numbers"] = data["atomic_numbers"].clone()
    data["atomic_numbers"][1:] = 8
    predictor(**_model_inputs(data))
    handle.remove()
    torch.testing.assert_close(captured[0][0], captured[1][0], rtol=0, atol=0)


def test_mpnn_pna_messages_are_paired_and_degree_statistics_are_training_only():
    """PNA reuses the exact MPNN message parameters/order; only aggregation and update differ."""
    torch.manual_seed(181)
    mpnn = model("mpnn_mean").atomic_encoder
    torch.manual_seed(181)
    pna = model("pna_full").atomic_encoder
    source = pna.covalent_message_sources[0]
    for name, value in mpnn.covalent[0].message_mlps[0].state_dict().items():
        assert torch.equal(value, source.message_mlps[0].state_dict()[name])
    data = batch()
    stats = AtomicGraphEncoder.training_degree_statistics([data], 8)
    assert all(value >= 0 for value in stats)
    other = dict(data, target=1 - data["target"])
    assert stats == AtomicGraphEncoder.training_degree_statistics([other], 8)


def test_global_requires_explicit_permission_and_does_not_silently_shrink():
    """Dense global attention is a separate approved exception with a hard byte guard."""
    with pytest.raises(ValueError, match="allow_dense_global"):
        _v2(
            "max", architecture_implementation="controlled", atomic_encoder_type="structural_global"
        )
    predictor = _v2(
        "max",
        architecture_implementation="controlled",
        atomic_encoder_type="structural_global",
        atomic_allow_dense_global=True,
        atomic_maximum_pairwise_bytes=1,
    ).eval()
    with pytest.raises(MemoryError):
        predictor(**_model_inputs(batch()))


def test_native_campaign_counts_and_frozen_context():
    """Six independent fixed studies enumerate exactly 172 Runs, with no hidden Cartesian axes."""
    root = Path(__file__).parents[1] / "experiments/v9"
    total = 0
    for suffix, candidates in (("a", 10), ("b", 7), ("c", 9), ("d", 6), ("e", 5), ("f", 6)):
        authored = WorkConfig.from_yaml(root / f"wisdom_v9{suffix}.yaml")
        fixture = authored.to_dict()
        fixture["with"]["dataset"] = {"file": str(Path(__file__).resolve())}
        config = WorkConfig.from_mapping(fixture, source=authored.source)
        assert not config.validation_errors(check_inputs=True)
        assert config.planned_runs == 4 * candidates
        work = config.levels[0].runs[0]
        assert work.seeds == (173, 179, 181, 191) and work.search_policy is None
        assert work.study_design.to_dict()["reference"] == {"atomic_encoder_type": "baseline"}
        common = fixture["with"]
        assert common["regional_positive_lambda"] == 0.3 and common["head_type"] == "single"
        assert common["evaluate_test"] is False and common["surface_refiner_heat_length"] == 6
        assert common["architecture_spike"] == "baseline"
        assert common["surface_geometry_transfer"] is False
        assert common["interaction_round"] == "single"
        assert common["vector_atomic_channels"] == 0
        assert common["feedback_propagation"] == "none"
        assert common["surface_encoder_type"] == "diffusion"
        assert common["surface_representation_mode"] == "learned"
        assert common["pooling_type"] == "max" and common["pooling_area_mode"] == "point"
        assert fixture["resources"]["gpu"] == 2 and fixture["resources"]["cpu"] == 36
        assert fixture["resources"]["memory"] == "96GiB"
        assert "execution" not in fixture
        assert work.execution_policy.max_parallel is None
        assert work.execution_policy.runs_per_gpu is None
        assert common["atomic_allow_dense_global"] is (suffix == "f")
        assert not {"steps", "search"} & fixture.keys()
        plan = WorkRunner().run(config, dry_run=True).to_dict()
        assert sum(len(level) for level in plan["levels"]) == candidates * 4
        total += candidates * 4
    assert total == 172 and not (root / "wisdom_v9.yaml").exists()


def test_v9_keeps_the_reviewed_single_pass_shared_ledger():
    """Every operator screen freezes the same V8/V7 core without importing a noisy V8 winner."""
    root = Path(__file__).parents[1] / "experiments"
    previous = WorkConfig.from_yaml(root / "v8/wisdom_v8b.yaml").to_dict()["with"]
    fields = (
        "dataset", "model_version", "subset", "initialization_profile", "optimization_profile",
        "hidden_dim", "embedding_dim", "residue_embedding_dim", "atomic_layers", "projection_depth",
        "surface_layers", "atom_spatial_k", "surface_atom_k", "diffusion_spectral_modes",
        "surface_atom_radius", "dropout", "learning_rate", "weight_decay", "gate_lambda",
        "surface_refiner_type", "surface_refiner_heat_length", "weak_loss_profile",
        "positive_existence_lambda", "cardinality_lambda", "total_variation_lambda",
        "dirichlet_lambda", "regional_loss_length", "regional_ranking_margin",
        "head_implementation", "architecture_implementation", "gate_cost_normalization",
        "batch_size", "epochs", "patience", "minimum_delta", "precision",
        "surface_metrics", "surface_metrics_interval", "faithfulness_audit", "evaluate_test",
        "optimization_diagnostics", "data_workers",
    )
    reference = None
    for suffix in "abcdef":
        values = WorkConfig.from_yaml(root / f"v9/wisdom_v9{suffix}.yaml").to_dict()["with"]
        assert {name: values[name] for name in fields} == {name: previous[name] for name in fields}
        scientific = {name: value for name, value in values.items()
                      if name != "atomic_allow_dense_global"}
        if reference is None:
            reference = scientific
        assert scientific == reference


def test_old_constructor_defaults_remain_strictly_restorable():
    """Old checkpoints need no new options or new atomic state keys."""
    first, parameters = _create_model(2, {"hidden_dim": 8, "embedding_dim": 4, "surface_layers": 1})
    second, _ = _create_model(2, parameters)
    second.load_state_dict(first.state_dict(), strict=True)
    assert second.atomic_encoder_type is AtomicEncoderType.BASELINE
