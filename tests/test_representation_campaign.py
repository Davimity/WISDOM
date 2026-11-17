"""Small CPU controls for opt-in V10/V11/V13 scientific interventions."""

# ruff: noqa: I001

import copy
import json
import yaml
import torch
import pytest

from pathlib import Path
from test_wisdom_v1 import _sample
from test_v8_architectures import batch, model
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.Training import _model_inputs, _create_model
from wisdom.evaluation.SurfaceMapAudit import SurfaceMapAudit
from wisdom.models.surface.DiffusionBlock import DiffusionBlock
from wisdom.models.surface.SurfaceLocalMessages import SurfaceLocalMessages
from wisdom.models.communication.TransferExperiment import TransferExperiment
from wisdom.models.atomic.representation.AtomicRepresentation import AtomicRepresentation


def chemical_batch():
    """Provide two local residue namespaces and explicit valid/missing chemical measurements."""
    data = batch()
    data["atom_residue_indices"] = torch.tensor([0, 0, 1, 0, 1, 1, 1])
    members = []
    sasa = []
    external = []
    for count in (3, 4):
        members.append(
            {
                "atom_name_id": torch.arange(count),
                "atom_identity_id": torch.arange(count),
                "torsion": torch.arange(14).reshape(2, 7).float() / 10,
                "torsion_valid": torch.tensor([[True] * 7, [False] * 7]),
                "functional_group_bits": torch.zeros(count, dtype=torch.long),
                "chemistry_known": torch.ones(count, dtype=torch.bool),
                "ring_member": torch.zeros(count, dtype=torch.bool),
            }
        )
        sasa.append({"sasa": torch.arange(count).float(), "valid": torch.ones(count).bool()})
        external.append(
            {
                "embedding": torch.arange(960).reshape(2, 480).float(),
                "valid": torch.tensor([True, False]),
            }
        )
    data["structural_assets"] = {
        "atomic_context": {"members": members},
        "atomic_sasa": {"members": sasa},
        "external_embeddings_esm2": {"members": external},
    }
    return data


def test_input_blocks_masks_fit_and_projection_bias():
    """Only valid training residues fit context moments; invalid projected context stays zero."""
    data = chemical_batch()
    sources = (
        "atom_name",
        "residue_atom",
        "backbone_torsions",
        "sidechain_torsions",
        "covalent_statistics",
        "functional_groups",
        "atomic_sasa",
    )
    representation = AtomicRepresentation(sources, external_model="esm2", external_mode="context")
    audit = representation.fit([data])
    assert audit["external_context"]["count"] == 2
    assert audit["atomic_sasa"]["count"] == 7
    gates = {name: torch.tensor(1.0) for name, _ in representation.gate_definitions}
    output = representation(data, gates)
    assert output.shape == (7, 97)
    assert torch.isfinite(output).all()
    assert output[[2, 4, 5, 6], -32:].count_nonzero() == 0
    assert representation.raw_blocks(data)["backbone_torsions"][2].count_nonzero() == 0
    closed = representation(data, {name: torch.tensor(0.0) for name in gates})
    assert closed.count_nonzero() == 0
    restored = copy.deepcopy(representation)
    restored.load_state_dict(representation.state_dict(), strict=True)
    torch.testing.assert_close(output, restored(data, gates), rtol=0, atol=0)


def test_static_adaptation_rank_norm_and_unknown():
    """Twenty centered pretrained rows have rank at most nineteen and a trainable zero UNK."""
    module = AtomicRepresentation(external_model="esm2", external_mode="static")
    table = torch.nn.Embedding(21, 64)
    module.initialize_residue_table(
        table,
        {"weights": torch.randn(20, 480), "alphabet": json.dumps(list("ARNDCQEGHILKMFPSTWYV"))},
    )
    assert module.initialization_audit["rank"] <= 19
    assert table.weight[0].count_nonzero() == 0
    torch.testing.assert_close(table.weight[1:].square().sum(1).mean(), torch.tensor(64.0))
    assert table.weight.requires_grad


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"summary": "mean_std"},
        {"summary": "mean_max"},
        {"summary": "mean_std_max"},
        {"summary": "mean_support"},
        {"summary": "duplicate_mean"},
        {"message": "pair"},
        {"message": "point_pair", "point_dim": 3},
        {"message": "low_rank"},
        {"heads": 2},
        {"heads": 4, "shared_head_weights": True},
        {"normalization": "entmax15"},
        {"learned_radius": True},
    ],
)
def test_transfer_initial_mean_empty_rows_and_gradients(options):
    """Neutral additions reproduce weighted means and mask empty rows, including projection bias."""
    torch.manual_seed(5)
    module = TransferExperiment(8, **options)
    scores = torch.randn(3, 4)
    atoms = torch.randn(3, 4, 8, requires_grad=True)
    geometry = torch.randn(3, 4, 3)
    distances = torch.arange(4).float().expand(3, -1)
    valid = torch.tensor([[True] * 4, [True, False, False, False], [False] * 4])
    output = module(scores, atoms, geometry, distances, valid, torch.randn(3, 3))
    assert output.shape == (3, 8) and torch.isfinite(output).all()
    assert output[2].count_nonzero() == 0
    if not options.get("learned_radius") and options.get("normalization") != "entmax15":
        torch.testing.assert_close(output[0], (scores[0].softmax(0)[:, None] * atoms[0]).sum(0))
    torch.testing.assert_close(output[1], atoms[1, 0])
    output.square().sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in module.parameters() if p.grad is not None)
    assert atoms.grad[2].count_nonzero() == 0


def test_zero_start_pair_branch_learns_on_second_step():
    """Only the final correction layer starts at zero; hidden pair features become trainable."""
    module = TransferExperiment(4, message="pair")
    optimizer = torch.optim.SGD(module.parameters(), lr=0.1)
    inputs = (
        torch.randn(2, 3),
        torch.randn(2, 3, 4),
        torch.randn(2, 3, 3),
        torch.ones(2, 3),
        torch.ones(2, 3).bool(),
    )
    for _ in range(2):
        optimizer.zero_grad()
        module(*inputs).square().sum().backward()
        optimizer.step()
    assert module.value_correction[0].weight.grad.abs().sum() > 0


def test_opt_in_empty_options_preserve_common_model_and_forward():
    """An empty declaration allocates no extension and preserves exact baseline output."""
    torch.manual_seed(11)
    reference = model("baseline").eval()
    torch.manual_seed(11)
    candidate = model("baseline", atomic_representation={}, surface_options={}).eval()
    assert reference.state_dict().keys() == candidate.state_dict().keys()
    for key, value in reference.state_dict().items():
        assert torch.equal(value, candidate.state_dict()[key]), key
    torch.testing.assert_close(
        reference(**_model_inputs(batch()))["logits"],
        candidate(**_model_inputs(batch()))["logits"],
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize(
    "options",
    [
        {"transfer_options": {"summary": "mean_std"}},
        {"transfer_options": {"message": "point_pair"}},
        {"transfer_options": {"heads": 2}},
        {"surface_options": {"use_gradient": False}},
        {"surface_options": {"readout_mode": "concat", "readouts": 2}},
        {
            "surface_options": {
                "readout_mode": "mixture",
                "readouts": 3,
                "scale_multipliers": [0.5, 1.0, 2.0],
            }
        },
    ],
)
def test_complete_model_extensions_restore_strictly(options):
    """Every implemented family passes end-to-end gradients and exact checkpoint restoration."""
    predictor = model("baseline", **options).eval()
    output = predictor(**_model_inputs(batch()))
    output["logits"].sum().backward()
    restored = model("baseline", **options).eval()
    restored.load_state_dict(predictor.state_dict(), strict=True)
    torch.testing.assert_close(
        output["logits"], restored(**_model_inputs(batch()))["logits"], rtol=0, atol=0
    )


def test_local_branch_two_step_gradient_and_isolated_points():
    """A zero residual scale preserves diffusion initially without killing branch learning."""
    branch = SurfaceLocalMessages(4, "attention")
    operator = {
        "local_edge_index": torch.tensor([[0], [1]]),
        "positions": torch.randn(3, 3),
        "normals": torch.randn(3, 3),
        "area": torch.ones(3),
    }
    features = torch.randn(3, 4)
    assert branch(features, operator).count_nonzero() == 0
    optimizer = torch.optim.SGD(branch.parameters(), lr=0.1)
    for _ in range(2):
        optimizer.zero_grad()
        output = features + branch(features, operator)
        output.square().sum().backward()
        optimizer.step()
    assert branch.message[0].weight.grad.abs().sum() > 0
    assert branch(features, operator)[2].count_nonzero() == 0


def test_gradient_ablation_keeps_nominal_mlp_width():
    """Zeroing derivative features is not an implicit capacity reduction."""
    block = DiffusionBlock(8)
    count = sum(p.numel() for p in block.parameters())
    block.configure(use_gradient=False)
    assert block.mlp[0].in_features == 24
    assert count == sum(p.numel() for p in block.parameters())


@pytest.mark.parametrize("options", [
    {"sources": ["atom_name", "residue_atom"]},
    {"sources": ["backbone_torsions", "sidechain_torsions"]},
    {"sources": ["covalent_statistics", "functional_groups", "atomic_sasa"]},
    {"external_model": "esm2", "external_mode": "context"},
    {"external_model": "esm2", "external_mode": "replace"},
])
@pytest.mark.parametrize("atomic_encoder", ["baseline", "schnet"])
def test_atomic_representation_full_backbone_restore(options, atomic_encoder):
    """New inputs and E4 table removal reach either backbone and restore exactly.

    Args:
        options: Declared physical or external-source representation, including residue removal.
        atomic_encoder: Historical baseline or 64-channel SchNet with an unchanged surface width.
    """
    data = chemical_batch()
    encoder_options = (
        {"atomic_encoder_type": "schnet", "atomic_hidden_dim": 64,
         "atomic_radial_encoding": "gaussian", "atomic_radial_count": 16}
        if atomic_encoder == "schnet" else {})
    predictor = model("baseline", atomic_representation=options, **encoder_options).eval()
    if atomic_encoder == "schnet":
        assert predictor.atomic_encoder.input_projection.out_features == 64
        assert predictor.atomic_encoder.covalent[0].channels == 64
        assert predictor.atomic_encoder.readout.out_features == 8
    predictor.atomic_representation.fit([data])
    output = predictor(**_model_inputs(data))
    output["logits"].sum().backward()
    restored = model("baseline", atomic_representation=options, **encoder_options).eval()
    restored.load_state_dict(predictor.state_dict(), strict=True)
    torch.testing.assert_close(output["logits"], restored(**_model_inputs(data))["logits"],
                               rtol=0, atol=0)
    if options.get("external_mode") == "replace":
        assert predictor.residue_type_embedding.weight.numel() == 0
        assert "atom.residue_type" in predictor.semantic_gates.inactive_names


def test_physical_random_control_matches_norm_and_common_initialization():
    """Norm-matched physical/random tables change chemical information, not shared head draws."""
    predictors = []
    for mode in ("current", "physical_norm", "physical_random"):
        torch.manual_seed(101)
        predictors.append(model("baseline", embedding_initialization=mode))
    for candidate in predictors[1:]:
        for name, weight in predictors[0].state_dict().items():
            if not name.startswith(("atomic_number_embedding", "residue_type_embedding")):
                assert torch.equal(weight, candidate.state_dict()[name]), name
        for table in (candidate.atomic_number_embedding, candidate.residue_type_embedding):
            torch.testing.assert_close(table.weight[1:].square().sum(1).mean(),
                                       torch.tensor(float(table.embedding_dim)))
            assert table.weight[0].count_nonzero() == 0


def test_surface_map_correspondence_filters_before_retaining_three():
    """A closer wrong-facing point cannot hide farther physically compatible candidates."""
    audit = SurfaceMapAudit(maximum_distance=2., query_neighbors=4)
    positions = torch.tensor([[0., 0., 0.]])
    normals = torch.tensor([[0., 0., 1.]])
    source = torch.tensor([[.1, 0., 0.], [.2, 0., 0.], [.3, 0., 0.], [.4, 0., 0.]])
    ns = normals.repeat(4, 1)
    ns[0] *= -1
    ids, weights, valid = audit.correspondence(positions, normals, source, ns)
    assert ids.tolist() == [[1, 2, 3]] and valid.all()
    torch.testing.assert_close(weights.sum(1), torch.ones(1).double())


def test_surface_map_identity_area_and_undefined_correlation():
    """Identity maps have zero error and unit area IoU; a constant correlation stays unavailable."""
    audit = SurfaceMapAudit()
    geometry = {"surface_positions": torch.tensor([[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]]),
                "surface_normals": torch.tensor([[0., 0., 1.]]).repeat(3, 1),
                "surface_area_weights": torch.tensor([1., 2., 5.])}
    values = torch.tensor([1., 1., 1.])
    metrics = audit.compare(geometry, geometry, values, values)
    assert metrics["reference_area_coverage"] == 1
    assert metrics["reference_probability_mae"] == 0
    assert metrics["reference_top_5_area_iou"] == 1
    assert metrics["reference_spearman"] is None


def test_frozen_view_audit_complete_forward_never_uses_original_point_targets():
    """Published alternate arrays replace all point-bound inputs; malformed old GT is ignored."""
    sample = _sample(3, 5, 1.)
    point_fields = {key: value.clone() for key, value in sample.items()
                    if key.startswith(("surface_", "diffusion_"))}
    sample.update({"surface_view__" + key: value for key, value in point_fields.items()})
    sample["surface_target_hard"] = torch.tensor([float("nan")])
    sample["surface_target_soft"] = torch.tensor([float("nan")])
    predictor = model("baseline").eval()
    result = SurfaceMapAudit().evaluate(predictor, sample, WisdomCollator(), torch.device("cpu"))
    assert result["rigid_tensor_max_logit_error"] == 0
    assert result["reference_logit_mae"] == 0
    assert result["constant_calibration"]["reference_spearman"] is None


@pytest.mark.parametrize("atomic_encoder", ["baseline", "schnet"])
def test_closed_input_sources_recover_control_and_do_not_dilute_gate_cost(atomic_encoder):
    """Closed sources recover the control within widened-GEMM roundoff and preserve gate cost.

    Args:
        atomic_encoder: Baseline or narrower SchNet, both under the same paired input-source test.
    """
    torch.manual_seed(181)
    encoder_options = (
        {"atomic_encoder_type": "schnet", "atomic_hidden_dim": 64}
        if atomic_encoder == "schnet" else {})
    reference = model("baseline", **encoder_options).eval()
    torch.manual_seed(181)
    augmented = model("baseline", atomic_representation={
        "sources": ["atom_name", "backbone_torsions"], "force_sources": "closed"},
        **encoder_options).eval()
    data = chemical_batch()
    torch.testing.assert_close(reference(**_model_inputs(data))["logits"],
                               augmented(**_model_inputs(data))["logits"], rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(reference.gate_regularization(), augmented.gate_regularization())


def test_v10_freezes_the_reviewed_schnet_backbone_without_reopening_v9_axes():
    """Every current V10 control/intervention inherits one explicit provisional V9f choice.

    The original confirmation template is historical, not part of this representation campaign.
    The decision record describes four development seeds, not independent confirmation or an
    assertion that the enriched @8 representation is identical to the @6 evidence.
    """
    root = Path(__file__).parents[1] / "experiments/v10"
    decision = json.loads((root / "backbone_selection.json").read_text())
    expected = decision["parameters"]
    assert decision["source_execution_id"] == "execution-d384d8f6b5064a26"
    assert decision["source_configuration"] == "full_rbf16_64"
    assert decision["source_seeds"] == [173, 179, 181, 191]
    assert len(decision["source_records"]) == 4
    assert expected["atomic_hidden_dim"] == 64
    assert expected["atomic_output_dim"] == expected["hidden_dim"] == 128
    paths = [root / f"wisdom_v10{letter}.yaml" for letter in "abcdefgh"]
    paths.extend(path for path in root.glob("wisdom_v10*.yaml.example")
                 if "legacy" not in path.name)
    for path in paths:
        document = yaml.safe_load(path.read_text())
        assert {key: document["with"][key] for key in expected} == expected
        assert not set(expected).intersection(document["sweep"]["space"])
        assert document["with"]["dataset"] == {"dataset": "wisdom-dna-reduced@8"}
        assert document["with"]["evaluate_test"] is False
        assert document["with"]["surface_refiner_heat_length"] == 6.0
        assert document["with"]["regional_positive_lambda"] == 0.3
        assert not {"search", "steps"}.intersection(document)
    viewer = yaml.safe_load((root / "visualization/wisdom_v10.yaml").read_text())["with"]
    assert viewer["dataset"] == {"dataset": "wisdom-dna-reduced@8"}
    assert viewer["source_seed"] == 211


def test_v10_selected_production_widths_preserve_atom_inputs_and_surface_output():
    """The 114-channel chemical input and 128-channel surface interface survive V10 extensions."""
    options = {
        "hidden_dim": 128, "embedding_dim": 32, "residue_embedding_dim": 64,
        "atomic_encoder_type": "schnet", "atomic_radial_encoding": "gaussian",
        "atomic_radial_count": 16, "atomic_hidden_dim": 64, "atomic_output_dim": 128,
        "architecture_implementation": "controlled", "gate_cost_normalization": "baseline_count",
        "atomic_layers": 2, "surface_layers": 1, "diffusion_spectral_modes": 4,
    }
    torch.manual_seed(211)
    control, _ = _create_model(2, options)
    torch.manual_seed(211)
    augmented, parameters = _create_model(2, {
        **options, "atomic_representation": {"sources": ["atom_name", "residue_atom"]}})
    projection = augmented.atomic_encoder.input_projection
    assert control.atomic_encoder.input_projection.in_features == 114
    assert projection.out_features == 64
    assert projection.in_features == 114 + 16
    assert augmented.atomic_encoder.readout.in_features == 64
    assert augmented.atomic_encoder.readout.out_features == 128
    torch.testing.assert_close(projection.weight[:, :114],
                               control.atomic_encoder.input_projection.weight, rtol=0, atol=0)
    torch.testing.assert_close(projection.bias,
                               control.atomic_encoder.input_projection.bias, rtol=0, atol=0)
    data = chemical_batch()
    augmented.atomic_representation.fit([data])
    output = augmented.eval()(**_model_inputs(data))
    output["logits"].sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in augmented.parameters() if p.grad is not None)
    restored, _ = _create_model(2, parameters)
    restored.load_state_dict(augmented.state_dict(), strict=True)
    torch.testing.assert_close(restored.eval()(**_model_inputs(data))["logits"],
                               output["logits"], rtol=0, atol=0)
