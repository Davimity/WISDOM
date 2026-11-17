"""Technical verification of V9f; synthetic optimization is not scientific validation."""

# ruff: noqa: I001

import io
import json
import yaml
import torch
import runpy
import pytest
import hashlib
import numpy as np

from pathlib import Path
from lambdaforge.nn import SchNetLayer
from test_v9_atomic_encoders import batch
from test_wisdom_v1 import _sample, _write_npz
from lambdaforge.work import WorkConfig, WorkRunner
from wisdom.Training import _create_model, _model_inputs
from wisdom.analysis.SchNetCampaignAudit import SchNetCampaignAudit
from wisdom.models.atomic.SchNetFilterAudit import SchNetFilterAudit


def predictor(mode="gaussian", count=16, width=128, kind="schnet"):
    """Create a tiny-geometry model with actual campaign widths and fixed surface modules.

    Args:
        mode: Direct or Gaussian radial input category.
        count: Gaussian channel count, or explicit one-channel direct record.
        width: SchNet internal channels; transfer remains 128 channels.
        kind: Full or five-metadata-masked SchNet.

    Returns:
        Fresh controlled WISDOM V2 model; all random state belongs to the caller's seed.
    """
    return _create_model(2, {
        "hidden_dim": 128, "embedding_dim": 4, "residue_embedding_dim": 4,
        "atomic_layers": 2, "projection_depth": 1, "surface_layers": 1,
        "dropout": 0.0, "curvature_features": 6, "diffusion_spectral_modes": 4,
        "atomic_encoder_type": kind, "atomic_radial_encoding": mode,
        "atomic_radial_count": count, "atomic_hidden_dim": width, "atomic_output_dim": 128,
        "architecture_implementation": "controlled", "head_implementation": "controlled",
    })[0]


@pytest.mark.parametrize("mode,count", [("direct", 1), ("gaussian", 8),
                                         ("gaussian", 16), ("gaussian", 32)])
@pytest.mark.parametrize("width", [64, 96, 128])
@pytest.mark.parametrize("kind", ["schnet", "schnet_radial"])
def test_shapes_paired_initialization_bfloat16_backward_restore(mode, count, width, kind):
    """All representations preserve the surface interface, finite training and checkpoint keys.

    Args:
        mode: Radial representation.
        count: Exact number of radial features.
        width: Actual latent width, including future 96-channel support outside the matrix.
        kind: Full/restricted metadata policy.
    """
    torch.manual_seed(173)
    model = predictor(mode, count, width, kind).eval()
    audit = SchNetFilterAudit.configuration(model)
    torch.manual_seed(173)
    reference = predictor()
    assert audit["shared_initialization_sha256"] == (
        SchNetFilterAudit.configuration(reference)["shared_initialization_sha256"])
    encoder = model.atomic_encoder
    assert encoder.input_projection.out_features == width
    assert encoder.covalent[0].channels == width
    assert encoder.covalent[0].filter[0].in_features == count + 3
    assert encoder.spatial[0].filter[0].in_features == count + 4
    assert encoder.covalent[0].filter[-1].out_features == width
    assert encoder.normalizations[0].normalized_shape == (width,)
    if width == 128:
        assert isinstance(encoder.readout, torch.nn.Identity)
    else:
        assert (encoder.readout.in_features, encoder.readout.out_features) == (width, 128)
    data = batch()
    captured = []
    handle = encoder.register_forward_hook(lambda module, inputs, output: captured.append(output))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = model(**_model_inputs(data))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            result["logits"], data["target"])
    loss.backward()
    handle.remove()
    assert captured[0].shape == (7, 128)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    restored = predictor(mode, count, width, kind).eval()
    restored.load_state_dict(torch.load(stream, weights_only=True), strict=True)
    torch.testing.assert_close(model(**_model_inputs(data))["logits"],
                               restored(**_model_inputs(data))["logits"], rtol=0, atol=0)


def test_rbf16_exact_v9d_reference_and_historical_checkpoint():
    """Omitted new options and explicit V9f RBF-16/128 have identical weights and predictions."""
    torch.manual_seed(179)
    explicit = predictor()
    # Reconstruct the old public signature without any new radial or width options.

    torch.manual_seed(179)
    original = _create_model(2, {
        "hidden_dim": 128, "embedding_dim": 4, "residue_embedding_dim": 4,
        "atomic_layers": 2, "projection_depth": 1, "surface_layers": 1,
        "dropout": 0.0, "curvature_features": 6, "diffusion_spectral_modes": 4,
        "atomic_encoder_type": "schnet", "atomic_radial_count": 16,
        "architecture_implementation": "controlled", "head_implementation": "controlled",
    })[0]
    assert original.state_dict().keys() == explicit.state_dict().keys()
    for key in original.state_dict():
        assert torch.equal(original.state_dict()[key], explicit.state_dict()[key]), key
    explicit.load_state_dict(original.state_dict(), strict=True)
    for operator in (explicit.atomic_encoder.covalent[0], explicit.atomic_encoder.spatial[0]):
        assert type(operator) is SchNetLayer
        torch.testing.assert_close(operator.radial_basis.centers, torch.linspace(0, 6, 16))
        assert 0.5 / operator.radial_basis.width**2 == pytest.approx(3.125)
    assert explicit.atomic_encoder.covalent[0].cutoff is None
    assert explicit.atomic_encoder.spatial[0].cutoff.cutoff == 6
    data = _model_inputs(batch())
    torch.testing.assert_close(original.eval()(**data)["logits"],
                               explicit.eval()(**data)["logits"], rtol=0, atol=0)


@pytest.mark.parametrize("mode,count", [("direct", 1), ("gaussian", 8), ("gaussian", 32)])
def test_distance_gates_cutoffs_and_covalent_support(mode, count):
    """Closed distance gates remove all distance dependence, not the complete relation.

    Args:
        mode: Direct or Gaussian expansion.
        count: Actual radial dimension.
    """
    encoder = predictor(mode, count, 64).atomic_encoder
    distances = torch.tensor([0.1, 2.0, 5.9, 6.0, 8.0])
    for operator in (encoder.covalent[0], encoder.spatial[0]):
        radial, support = operator.radial_inputs(distances, distance_weights=torch.tensor(0.0))
        assert torch.count_nonzero(radial) == 0
        assert torch.equal(support, torch.ones_like(distances))
        radial, partial = operator.radial_inputs(distances, distance_weights=torch.tensor(0.3))
        opened, full = operator.radial_inputs(distances, distance_weights=torch.tensor(1.0))
        torch.testing.assert_close(radial, 0.3 * opened)
        torch.testing.assert_close(partial, 0.7 + 0.3 * full)
    assert torch.equal(encoder.covalent[0].radial_inputs(distances)[1],
                       torch.ones_like(distances))
    assert torch.equal(encoder.spatial[0].radial_inputs(distances)[1][-2:], torch.zeros(2))
    if mode == "direct":
        torch.testing.assert_close(opened, distances[:, None] / 6)


@pytest.mark.parametrize("mode,count", [("direct", 1), ("gaussian", 16)])
def test_restricted_metadata_and_closed_distance_gates_have_no_alternative_path(mode, count):
    """Changing unavailable metadata or gated distances cannot alter complete protein logits.

    Args:
        mode: Direct or Gaussian SchNet.
        count: Effective radial input count.
    """
    model = predictor(mode, count, 64, "schnet_radial").eval()
    for relation in ("covalent", "spatial"):
        model.semantic_gates.force(f"edge.{relation}.distance", 0)
    inputs = _model_inputs(batch())
    expected = model(**inputs)["logits"]
    inputs["atom_edge_distance"] = inputs["atom_edge_distance"] * 20 + 50
    inputs["atom_edge_bond_order"] = torch.full_like(inputs["atom_edge_bond_order"], 3)
    inputs["atom_edge_same_residue"] = ~inputs["atom_edge_same_residue"].bool()
    inputs["atom_edge_same_chain"] = ~inputs["atom_edge_same_chain"].bool()
    inputs["atom_edge_residue_separation"] = inputs["atom_edge_residue_separation"] + 50
    actual = model(**inputs)["logits"]
    torch.testing.assert_close(expected, actual, rtol=0, atol=0)


@pytest.mark.parametrize("mode,count", [("direct", 1), ("gaussian", 16)])
def test_masked_metadata_filters_and_audit_are_target_free(mode, count, tmp_path):
    """Restricted sources stay unavailable and best-checkpoint probes preserve RNG/weights.

    Args:
        mode: Direct or Gaussian inputs.
        count: Radial width.
        tmp_path: Disposable native-output equivalent for bounded diagnostic files.
    """
    model = predictor(mode, count, 64, "schnet_radial").eval()
    for name in model.atomic_encoder_type.inactive_sources():
        assert model.semantic_gates.sample(training=False)[name] == 0
    before = {name: tensor.clone() for name, tensor in model.state_dict().items()}
    rng = torch.get_rng_state().clone()
    report = SchNetFilterAudit.write(model, tmp_path)
    assert torch.equal(rng, torch.get_rng_state())
    assert len(report["profiles"]) == 8
    assert "shared_checkpoint_sha256" in report
    for name, tensor in model.state_dict().items():
        assert torch.equal(before[name], tensor), name
    with np.load(tmp_path / "radial-filters.npz", allow_pickle=False) as values:
        assert values["covalent_layer0_intra_residue_single_before_support"].shape == (121, 64)
    assert (tmp_path / "radial-filters.png").stat().st_size > 0


def test_campaign_migration_preserves_historical_attention_configuration():
    """Migration changes only the future name; historical metadata/config bytes are preserved."""
    root = Path(__file__).parents[1] / "experiments/v9"
    metadata = json.loads((root / "campaign_migration.json").read_text())
    legacy = metadata["historical_yaml"]
    assert hashlib.sha256(legacy.encode()).hexdigest() == metadata["historical_yaml_sha256"]
    old = WorkConfig.from_mapping(yaml.safe_load(legacy),
                                 source=root / "wisdom_v9g.yaml").to_dict()
    future = WorkConfig.from_yaml(root / "wisdom_v9g.yaml").to_dict()
    assert future.pop("name") == "wisdom-v9g"
    assert old.pop("name") == "wisdom-v9f"
    assert old == future
    new = WorkConfig.from_yaml(root / "wisdom_v9f.yaml")
    assert new.name == "wisdom-v9f-schnet" and new.planned_runs == 52


@pytest.mark.parametrize("cells", [[2], [1, 4], [1, 6, 7, 8, 9], [1, 10, 11, 12, 13]])
def test_cell_and_block_authoring_is_native_and_does_not_submit(cells, tmp_path):
    """Selected rows retain every seed, exact scientific context, and comment documentation.

    Args:
        cells: Explicit main-matrix rows, including contemporary controls where required.
        tmp_path: Disposable output configuration; no production evidence is touched.
    """
    root = Path(__file__).parents[1]
    configure = runpy.run_path(str(root / "scripts/configure_schnet_campaign.py"))["configure"]
    # A disposable file binding avoids requiring the production Registry in a temporary project.
    # Native validation checks existence only; no fake DatasetVersion is registered.

    source = tmp_path / "source.yaml"
    source.write_text((root / "experiments/v9/wisdom_v9f.yaml").read_text().replace(
        "{dataset: wisdom-dna-reduced@6}", "{file: '" + str(Path(__file__).resolve()) + "'}"))
    output = tmp_path / "selection.yaml"
    assert configure(source, output, cells) == 4 * len(cells)
    native = WorkConfig.from_yaml(output)
    assert native.planned_runs == 4 * len(cells)
    assert native.to_dict()["seeds"] == [173, 179, 181, 191]
    with pytest.raises(FileExistsError):
        configure(source, output, cells)


@pytest.mark.parametrize("mode,count", [("direct", 1), ("gaussian", 32)])
def test_reduced_schnet_permutation_rigid_motion_and_protein_independence(mode, count):
    """Internal width reduction cannot introduce frame, index or batch-neighbor information.

    Args:
        mode: Active radial encoding.
        count: Active radial input width.
    """
    model = predictor(mode, count, 64).eval()
    inputs = _model_inputs(batch())
    expected = model(**inputs)["logits"]
    permutation = torch.tensor([2, 0, 1, 6, 3, 5, 4])
    inverse = torch.argsort(permutation)
    for name in ("atomic_numbers", "residue_type_ids", "atom_role_ids", "atom_hybridization_ids",
                 "formal_charges", "atom_aromaticity", "atom_hbond_donor", "atom_hbond_acceptor",
                 "residue_hydropathy", "residue_polarity", "atom_positions", "atom_batch"):
        inputs[name] = inputs[name][permutation]
    inputs["atom_edge_index"] = inverse[inputs["atom_edge_index"]].sort(dim=0).values
    neighbors = inputs["surface_atom_neighbors"]
    inputs["surface_atom_neighbors"] = torch.where(
        neighbors >= 0, inverse[neighbors.clamp_min(0)], neighbors)
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    inputs["atom_positions"] = inputs["atom_positions"] @ rotation.T + 11
    actual = model(**inputs)["logits"]
    torch.testing.assert_close(expected, actual, rtol=3e-4, atol=3e-5)
    inputs["atomic_numbers"] = inputs["atomic_numbers"].clone()
    inputs["atomic_numbers"][3:] = 8
    changed = model(**inputs)["logits"]
    torch.testing.assert_close(changed[0], actual[0], rtol=0, atol=0)


@pytest.mark.parametrize("mode,count,width", [("direct", 1, 64), ("gaussian", 16, 128)])
def test_native_short_training_smoke(mode, count, width, tmp_path):
    """Exercise actual Training.run, parameter forwarding, evaluation and restored-filter artifacts.

    Args:
        mode: Direct or Gaussian model family.
        count: Active radial dimension.
        width: Internal atomic channels, independent of the small surface width used here.
        tmp_path: Disposable synthetic two-class manifests and LF execution root.
    """
    rows = ["file,label,split"]
    for split in ("train", "val", "test"):
        for label in (0, 1):
            path = tmp_path / f"{split}{label}.npz"
            _write_npz(path, _sample(4, 6, float(label)))
            rows.append(f"{path.name},{label},{split}")
    manifest = tmp_path / "manifest.csv"
    manifest.write_text("\n".join(rows) + "\n")
    config = WorkConfig.from_mapping({
        "name": "synthetic-schnet-smoke", "run": "wisdom.Training.Training", "seeds": [173],
        "with": {
            "dataset": {"file": str(manifest)}, "model_version": 2,
            "hidden_dim": 8, "embedding_dim": 4, "atomic_layers": 1, "surface_layers": 1,
            "diffusion_spectral_modes": 4, "atom_spatial_k": 2,
            "atomic_encoder_type": "schnet", "architecture_implementation": "controlled",
            "atomic_radial_encoding": mode, "atomic_radial_count": count,
            "atomic_hidden_dim": width, "atomic_output_dim": 8,
            "epochs": 2, "data_workers": 0, "precision": "float32", "evaluate_test": False,
            "surface_metrics": False, "visualization": {"mode": "none"},
        },
    }, source=tmp_path / "smoke.yaml")
    result = WorkRunner(execution_root=tmp_path / "runs").run(config)
    assert result.status == "succeeded"
    assert list((tmp_path / "runs").rglob("radial-filters.json"))


def test_predeclared_audit_uses_checkpoint_metrics_and_paired_complete_support(tmp_path):
    """Latest good-looking metrics, failed Attempts and missing seeds cannot inflate contrasts.

    Args:
        tmp_path: Disposable descriptive report directory, not a scientific result store.
    """
    rows = []
    for cell, (kind, mode, count, width) in enumerate(SchNetCampaignAudit.CELLS, 1):
        for seed in SchNetCampaignAudit.SEEDS:
            rows.append({
                "run_id": f"run-{cell}-{seed}", "status": "succeeded", "seed": seed,
                "attempt_number": 1,
                "parameters": {"atomic_encoder_type": kind, "atomic_radial_encoding": mode,
                               "atomic_radial_count": count, "atomic_hidden_dim": width},
                "metrics": {"surface_score": 999, "trainable_parameter_count": 100 - cell},
                "artifacts": [{"name": "best-model", "sha256": f"sha-{cell}-{seed}",
                               "metadata": {"epoch": 5, "metrics": {
                                   "surface_score": 0.1 + cell / 100,
                                   "global_score": 0.5, "surface_raw_positive_macro_auprc": 0.1}}}],
            })
    # A newer failed Attempt invalidates the formerly completed evidence for this one seed.

    rows.append({**rows[-1], "attempt_number": 2, "status": "failed"})
    audit = SchNetCampaignAudit({"execution_id": "synthetic-only", "runs": rows})
    contrast = audit.contrast({2: 1, 4: -1}, "surface_score")
    assert contrast["n"] == 4 and contrast["mean"] == pytest.approx(-0.02)
    assert contrast["positive_seeds"] == 0 and contrast["negative_seeds"] == 4
    assert audit.contrast({13: 1, 4: -1}, "surface_score")["n"] == 3
    report = audit.write(tmp_path / "audit")
    assert len(report["coverage"]["13"]["completed_seeds"]) == 3
    assert "13" not in report["descriptively_dominated_by_cell_for_each_cost"][
        "trainable_parameter_count"]
    assert report["paired_vs_cell04"]["2"]["surface_score"]["mean"] == pytest.approx(-0.02)
    assert report["descriptive_interactions"]["RBF32-versus-direct_width-interaction"][
        "mean"] == pytest.approx(0.0)
    assert (tmp_path / "audit/radial-width-cost.png").is_file()
    with pytest.raises(FileExistsError):
        audit.write(tmp_path / "audit")
