"""Small deterministic contracts for late fusion, frozen fields and explicit zinc evidence."""

import hashlib
import json
from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.work import WorkConfig, WorkRunner
from test_wisdom_v1 import _sample, _write_npz

from wisdom.data.TaskSpecification import TaskSpecification
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.features.SurfaceFeatureProjector import SurfaceFeatureProjector
from wisdom.features.SurfaceFeatureSchema import SurfaceFeatureSchema
from wisdom.features.SurfaceFeatureSidecar import SurfaceFeatureSidecar
from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.WisdomV2 import WisdomV2
from wisdom.preprocessing.common.snapshots import snapshot_structures
from wisdom.preprocessing.zinc.audit import write_diversity_report
from wisdom.preprocessing.zinc.evidence import load_evidence
from wisdom.preprocessing.zinc.ZincAnnotation import ZincAnnotation
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination
from wisdom.preprocessing.zinc.ZincValidation import ZincValidation
from wisdom.Training import _model_inputs
from wisdom.utils.structure.BiologicalAssembly import BiologicalAssembly
from wisdom.utils.structure.ProteinStructure import ProteinStructure


def _base(path: Path) -> None:
    """Write a tiny point-ordered chemistry fixture; it is not a production benchmark."""
    _write_npz(path, _sample(3, 3, 1.0))
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays.update(
        atom_positions=np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32
        ),
        atom_names=np.asarray(["ND1", "OD1", "N"]),
        residue_names=np.asarray(["HIS", "ASP", "ALA"]),
        formal_charges=np.asarray([0, -1, 0], dtype=np.int8),
        atomic_numbers=np.asarray([7, 8, 7]),
        metadata_json=np.asarray(
            json.dumps(
                {
                    "preprocessing_schema_version": "3.0",
                    "coordinate_origin": [0, 0, 0],
                    "config": {"surface_atom_radius": 6.0},
                }
            )
        ),
    )
    np.savez_compressed(path, **arrays)


@pytest.mark.parametrize("mode,width", [("learned", 8), ("explicit", 3), ("hybrid", 11)])
def test_attention_consumes_the_actual_evidence_width(mode, width, monkeypatch):
    """Attention uses H/K/H+K and explicit mode never executes any learned encoder."""
    sample = _sample(3, 3, 1.0)
    if mode != "learned":
        sample["surface_explicit_features"] = torch.randn(3, 3)
    batch = WisdomCollator(atom_spatial_k=8, surface_atom_k=8)([sample])
    model = WisdomV2(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=1,
        surface_layers=1,
        pooling_type="attention",
        surface_representation_mode=mode,
        explicit_feature_dim=0 if mode == "learned" else 3,
        dropout=0.0,
    )
    if mode == "explicit":

        def forbidden(*args, **kwargs):
            """Fail if the explicit control accidentally calls a learned encoder."""
            raise AssertionError("explicit mode called a learned encoder")

        monkeypatch.setattr(model.atomic_encoder, "forward", forbidden)
        monkeypatch.setattr(model.surface_encoder, "forward", forbidden)
        assert not next(model.atomic_encoder.parameters()).requires_grad
    output = model(**_model_inputs(batch))
    assert output["surface_evidence_features"].shape == (3, width)
    if mode == "explicit":
        assert "surface_learned_embeddings" not in output
        assert "surface_embeddings" not in output
    else:
        assert output["surface_learned_embeddings"].shape == (3, 8)
        assert output["surface_embeddings"] is output["surface_learned_embeddings"]
    if mode != "learned":
        assert output["surface_explicit_features"].shape == (3, 3)
    assert torch.isfinite(output["logits"]).all()
    output["logits"].sum().backward()
    assert model.local_head.weight.grad is not None


def test_learned_default_preserves_identical_initialization_and_predictions():
    """Opting into the default vocabulary does not alter historical weights or outputs."""
    kwargs = dict(hidden_dim=8, embedding_dim=4, atomic_layers=1, surface_layers=1, dropout=0.0)
    torch.manual_seed(7)
    first = WisdomV1(**kwargs).eval()
    torch.manual_seed(7)
    second = WisdomV1(**kwargs, surface_representation_mode="learned").eval()
    batch = WisdomCollator(atom_spatial_k=8, surface_atom_k=8)([_sample(3, 3, 1.0)])
    with torch.no_grad():
        left, right = first(**_model_inputs(batch)), second(**_model_inputs(batch))
    assert torch.equal(left["surface_logits"], right["surface_logits"])
    assert torch.equal(left["logits"], right["logits"])


def test_feature_sidecars_preserve_base_bytes_and_reject_alignment_drift(tmp_path):
    """Feature projection reads no annotation and binds the exact immutable base bytes."""
    base, sidecar = tmp_path / "base.npz", tmp_path / "fields.npz"
    _base(base)
    original = base.read_bytes()
    names = SurfaceFeatureSchema.resolve(group="generic_minimal")
    SurfaceFeatureSidecar.write(base, sidecar, names)
    values = SurfaceFeatureSidecar.read(sidecar, base, names)
    assert values.shape == (3, 3)
    assert base.read_bytes() == original
    # Modifying only the NPZ container is enough to invalidate the scientific point binding.
    with base.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        SurfaceFeatureSidecar.read(sidecar, base, names)


def test_normalization_depends_only_on_the_declared_training_population(tmp_path):
    """An arbitrary validation distribution cannot change frozen training means/stds."""
    base, sidecar = tmp_path / "base.npz", tmp_path / "fields.npz"
    _base(base)
    names = ("formal_charge_density", "zn_lewis_strict")
    SurfaceFeatureSidecar.write(base, sidecar, names)
    stats = SurfaceFeatureSidecar.fit_statistics([("TRAIN", base, sidecar)], names)
    training = SurfaceFeatureSidecar.read(sidecar, base, names)
    assert stats["sources"] == [
        {"id": "TRAIN", "base_npz_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
         "feature_sha256": hashlib.sha256(sidecar.read_bytes()).hexdigest(), "point_count": 3}
    ]
    assert np.allclose(
        SurfaceFeatureSidecar.normalize(training, names, stats).mean(0), 0, atol=1e-6
    )
    SurfaceFeatureSidecar.normalize(np.full_like(training, 1000), names, stats)
    assert stats == SurfaceFeatureSidecar.fit_statistics([("TRAIN", base, sidecar)], names)
    with pytest.raises(ValueError, match="train only"):
        SurfaceFeatureSidecar.normalize(training, names, {**stats, "split": "validation"})


def test_strict_lewis_excludes_amide_N_and_observed_zinc(tmp_path):
    """The zinc-motivated heuristic distinguishes atom identity but never reads actual Zn."""
    base = tmp_path / "base.npz"
    _base(base)
    with np.load(base, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    projector = SurfaceFeatureProjector()
    assert np.all(projector.project(arrays, ["zn_lewis_strict"]) > 0)
    arrays["atomic_numbers"][0] = 30
    with pytest.raises(ValueError, match="observed Zn"):
        projector.project(arrays, ["zn_lewis_strict"])


def test_unknown_optional_fields_fail_without_silently_padding():
    """Researcher-authored channel errors must not change the scientific input silently."""
    with pytest.raises(ValueError, match="unknown surface features"):
        SurfaceFeatureSchema.resolve(["surface_target_soft"])
    with pytest.raises(ValueError, match="require explicit_feature_dim"):
        WisdomV1(surface_representation_mode="explicit")


def test_zinc_task_ingestion_uses_declared_target_not_DNA(tmp_path):
    """Generic ingestion keeps the DNA fallback but accepts explicit Zn metadata."""
    _write_npz(tmp_path / "base.npz", _sample(3, 3, 0.0))
    specification = {
        "task_name": "zinc_binding",
        "global_target_key": "zinc_binding",
        "local_annotation_asset": "zinc_annotation",
    }
    DatasetIndex.write(
        tmp_path / "index.jsonl",
        [
            DatasetMember(
                member_id="FIXTURE_A",
                partitions={"split": "train"},
                targets={"zinc_binding": 0},
                metadata={"task_specification": specification},
                assets={"universal_npz": DatasetAsset(path="base.npz")},
            )
        ],
    )
    data = WisdomDataset(tmp_path, "train", include_surface_targets=False)
    assert data[0]["target"].item() == 0
    assert data.task_specification.task_name == "zinc_binding"
    assert TaskSpecification.from_metadata({}).global_target_key == "dna_binding"


def _assembly() -> BiologicalAssembly:
    """Build two peptide residues coordinating one Zn plus an unrelated metal for unit tests."""
    source = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("AQ")
    for number, name, element, x in ((1, "HIS", "N", -2.0), (2, "ASP", "O", 2.0)):
        residue = gemmi.Residue()
        residue.name, residue.seqid = name, gemmi.SeqId(number, " ")
        residue.entity_type = gemmi.EntityType.Polymer
        residue.het_flag = "A"
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = (
            "ND1" if number == 1 else "OD1",
            gemmi.Element(element),
            gemmi.Position(x, 0, 0),
        )
        atom.occ = 1.0
        residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    metal_chain = gemmi.Chain("Z")
    for number, x in ((1, 0.0), (2, 20.0)):
        residue = gemmi.Residue()
        residue.name, residue.seqid, residue.het_flag = "ZN", gemmi.SeqId(number, " "), "H"
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos, atom.occ = (
            "ZN",
            gemmi.Element("Zn"),
            gemmi.Position(x, 0, 0),
            1.0,
        )
        residue.add_atom(atom)
        metal_chain.add_residue(residue)
    model.add_chain(metal_chain)
    source.add_model(model)
    return BiologicalAssembly(source, source[0], "1")


def test_zinc_coordination_counts_distinct_residues_and_complete_chain_names():
    """Only the selected-copy occupied site is accepted; a remote Zn is not a protein positive."""
    sites = ZincCoordination().analyse(_assembly(), "AQ", 1)
    assert len(sites) == 2
    assert sites[0]["accepted"] and len(sites[0]["selected_residues"]) == 2
    assert not sites[1]["accepted"]
    assert not ZincCoordination(minimum_occupancy=1.1).analyse(_assembly(), "AQ", 1)


def test_missing_explicit_negative_evidence_is_not_ligand_absence(tmp_path):
    """Negative labels without experiments or an experimental NOT annotation are rejected."""
    row = {
        "identifier": "FAKE_AQ",
        "label": 0,
        "sequence": "HD",
        "assembly_id": "1",
        "protein_copy": 1,
        "origin": "fixture",
        "label_evidence": {"kind": "no_Zn_in_PDB"},
    }
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="explicit sequence-bound"):
        load_evidence(path)
    row["label_evidence"] = {
        "kind": "experimental_non_binding",
        "scope": "zinc_binding",
        "reference": "fixture-not-a-real-experiment",
        "assay": "fixture",
        "sequence_sha256": hashlib.sha256(b"HD").hexdigest(),
    }
    path.write_text(json.dumps(row) + "\n")
    assert load_evidence(path)[0]["protein_chain"] == "AQ"


def test_positive_zinc_without_local_GT_is_unavailable_not_all_negative(tmp_path):
    """Training can retain an inaccessible positive site, but evaluation must fail explicitly."""
    base = tmp_path / "base.npz"
    _base(base)
    sites = ZincCoordination().analyse(_assembly(), "AQ", 1)
    row = {
        "identifier": "FIXTURE_AQ",
        "label": 1,
        "split": "train",
        "sites": sites,
        "protein_chain": "AQ",
        "protein_copy": 1,
        "assembly_id": "1",
        "assembly_rotation": np.eye(3).tolist(),
        "assembly_translation": [1000.0, 0.0, 0.0],
        "structure_sha256": "a" * 64,
        "label_evidence": {"kind": "fixture"},
    }
    output = tmp_path / "annotation.npz"
    ZincAnnotation().write(base, output, row)
    assert ZincAnnotation.validate(output, base)
    with np.load(output, allow_pickle=False) as archive:
        assert not archive["local_gt_available"]
        assert not archive["surface_valid_mask"].any()
    with pytest.raises(ValueError, match="no usable GT"):
        ZincAnnotation().write(base, output, {**row, "split": "validation"})


def test_optional_fields_publish_through_native_LF_and_support_inference(tmp_path, monkeypatch):
    """An isolated toy publication tests native cache/map/statistics assets, not real science."""
    root = tmp_path / "input"
    root.mkdir()
    _base(root / "base.npz")
    DatasetIndex.write(
        root / "index.jsonl",
        [
            DatasetMember(
                member_id=identifier,
                partitions={"split": split},
                targets={"dna_binding": label},
                metadata={"dilutions": ["replicate-00/train-25"] if identifier == "FIXTURE_A"
                          else []},
                assets={"universal_npz": DatasetAsset(path="base.npz")},
            )
            for identifier, split, label in (("FIXTURE_A", "train", 1),
                                              ("FIXTURE_B", "train", 0),
                                              ("FIXTURE_C", "validation", 1))
        ],
    )
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "registry"))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "published"))
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "local")
    config = tmp_path / "fields.yaml"
    config.write_text(
        "run: wisdom.features.SurfaceFeatures.SurfaceFeatures\n"
        f"with:\n  dataset: {{file: {root}}}\n  dataset_name: toy-fields\n"
        "  dataset_version: '1'\n  feature_group: generic_minimal\n  workers: 1\n"
    )
    result = WorkRunner(execution_root=tmp_path / "runs").run(WorkConfig.from_yaml(config))
    assert result.status == "succeeded", result.to_dict()
    indexes = list((tmp_path / "published").rglob("index.jsonl"))
    assert len(indexes) == 1
    data = WisdomDataset(
        indexes[0].parent,
        "train",
        include_surface_targets=False,
        surface_features=("formal_charge_density", "hbond_donor_density"),
    )
    assert data[0]["surface_explicit_features"].shape == (3, 2)
    assert {source["id"] for source in data.feature_statistics["sources"]} == {
        "FIXTURE_A", "FIXTURE_B"}
    reduced = WisdomDataset(indexes[0].parent, "val", subset="replicate-00/train-25",
        include_surface_targets=False, surface_features=("formal_charge_density",))
    assert len(reduced) == 1  # Validation is not filtered by a training dilution.
    assert [source["id"] for source in reduced.feature_statistics["sources"]] == ["FIXTURE_A"]
    # Learned mode does not consult or decode optional field files.
    learned = WisdomDataset(indexes[0].parent, "train", include_surface_targets=False)
    assert "surface_explicit_features" not in learned[0]


def test_checkpoint_roundtrip_preserves_late_fusion_and_excludes_GT(tmp_path):
    """Reloaded late-fusion parameters reproduce predictions, with GT outside model kwargs."""
    sample = _sample(3, 3, 1.0)
    sample["surface_explicit_features"] = torch.randn(3, 3)
    sample["surface_target_hard"] = torch.ones(3, dtype=torch.bool)
    sample["surface_target_soft"] = torch.ones(3)
    sample["surface_valid_mask"] = torch.ones(3, dtype=torch.bool)
    batch = WisdomCollator(atom_spatial_k=8, surface_atom_k=8)([sample])
    parameters = dict(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=1,
        surface_layers=1,
        surface_representation_mode="hybrid",
        explicit_feature_dim=3,
        pooling_type="attention",
        dropout=0,
    )
    model = WisdomV2(**parameters).eval()
    expected = model(**_model_inputs(batch))["logits"]
    torch.save({"model_parameters": parameters, "model": model.state_dict()}, tmp_path / "model.pt")
    state = torch.load(tmp_path / "model.pt", weights_only=True)
    restored = WisdomV2(**state["model_parameters"]).eval()
    restored.load_state_dict(state["model"])
    torch.testing.assert_close(restored(**_model_inputs(batch))["logits"], expected)
    assert not any("target" in name or "valid_mask" in name for name in _model_inputs(batch))


def test_zinc_audit_records_duplicates_cross_split_groups_and_invalid_targets(tmp_path):
    """Hard split/identity failures remain in the report even when an archive is also absent."""
    records = [
        {
            "id": "DUP_A",
            "partitions": {"split": split, "leakage_group": "G0"},
            "targets": {"zinc_binding": 1},
            "assets": {},
            "metadata": {},
        }
        for split in ("train", "test")
    ]
    records.append({**records[0], "id": "BAD_A", "targets": {"zinc_binding": 3}})
    report = ZincValidation().audit_members(records, tmp_path)
    failures = [entry["failure"] for entry in report["failures"]]
    assert report["verdict"] == "FAIL"
    assert "duplicate member identity" in failures
    assert "invalid binary target or supervised split" in failures
    records[1]["id"] = "OTHER_A"
    report = ZincValidation().audit_members(records, tmp_path)
    assert any("crosses splits" in entry["failure"] for entry in report["failures"])


@pytest.mark.parametrize("view", ["replicate-00/train-nan", "replicate-00/train-101", "bad"])
def test_zinc_audit_records_invalid_dilutions_without_losing_the_report(tmp_path, view):
    """Malformed view names are per-member failures, not sorting-time crashes."""
    record = {
        "id": "BAD_A",
        "partitions": {"split": "train", "leakage_group": "G0"},
        "targets": {"zinc_binding": 1},
        "assets": {},
        "metadata": {"dilutions": [view]},
    }
    report = ZincValidation().audit_members([record], tmp_path)
    assert report["verdict"] == "FAIL"
    assert (tmp_path / "report.json").is_file()
    assert not report["local_support"]


def test_zinc_diversity_report_preserves_rare_chemistry_and_unavailable_SMD(tmp_path):
    """Sparse support is disclosed, not presented as zero or removed to hide the long tail."""
    records = [
        {
            "identifier": f"FIX{i}_A",
            "label": i % 2,
            "leakage_group": f"G{i}",
            "split": "train",
            "origin": "fixture",
            "sequence_length": 10,
            "sites": [{"accepted": True, "selected_residue_names": ["HIS", "ASP"],
                       "selected_residue_counts": {"HIS": 2, "ASP": 1}, "interchain": False}],
        }
        for i in range(4)
    ]
    report = write_diversity_report(tmp_path, records, records)
    assert report["covariates"]["sequence_length"]["smd"] is None
    assert report["site_residue_classes"]["ASP1/HIS2"] == 2
    assert report["selected_leakage_groups"] == 4
    assert (tmp_path / "diversity.svg").is_file()
    assert "not a significance test" in (tmp_path / "diversity.md").read_text()


def test_native_zinc_preprocessing_publication_and_tiny_backward(tmp_path, monkeypatch):
    """Synthetic sources exercise real geometry, Zn references, native publication and ingestion.

    Artificial evidence and deliberately assigned toy groups test software only; they are never
    real experimental negatives or a scientific benchmark, and their Registry is isolated.
    """
    design = tmp_path / "design"
    design.mkdir()
    rows = []
    for split_index, split in enumerate(("train", "validation", "test")):
        for label in (0, 1):
            index = 2 * split_index + label
            identifier = f"F{index:03d}_AQ"
            source = _assembly().source.clone()
            source.name = identifier
            first_name = ("ALA", "GLU", "CYS")[split_index]
            source[0][0][0].name = first_name
            source[0][0][1].name = "HIS" if label else "ASP"
            if not label:
                del source[0]["Z"]
            source.setup_entities()
            entity = source.get_entity_of(source[0][0].get_polymer())
            entity.full_sequence = [first_name, "HIS" if label else "ASP"]
            assembly = gemmi.Assembly("1")
            generator = gemmi.Assembly.Gen()
            generator.chains = [chain.name for chain in source[0]]
            operator = gemmi.Assembly.Operator()
            operator.name = "1"
            generator.operators.append(operator)
            assembly.generators.append(generator)
            source.assemblies.append(assembly)
            path = design / f"f{index:03d}.cif"
            source.make_mmcif_document().write_file(str(path))
            structure = ProteinStructure(path)
            sequence = structure.sequence(structure.structure[0][0])
            sites = ZincCoordination().analyse(structure.assembly("1"), "AQ", 1)
            rows.append(
                {
                    "identifier": identifier,
                    "pdb_id": f"f{index:03d}",
                    "protein_chain": "AQ",
                    "protein_copy": 1,
                    "assembly_id": "1",
                    "sequence": sequence,
                    "label": label,
                    "origin": "synthetic-unit-test",
                    "source_structure": str(path),
                    "structure_sha256": structure.sha256(),
                    "sites": sites,
                    "split": split,
                    "assembly_rotation": np.eye(3).tolist(),
                    "assembly_translation": [0, 0, 0],
                    "leakage_group": f"TOY{index}",
                    "global_phenotype": "ZN_NOISE",
                    "local_phenotype": "ZN_NOISE",
                    "coordination_parameters": {
                        "cutoff": 3.0,
                        "minimum_occupancy": 0.5,
                        "minimum_donors": 2,
                        "minimum_residues": 2,
                    },
                    "label_evidence": {"kind": "structural_coordination_candidate"}
                    if label
                    else {
                        "kind": "experimental_non_binding",
                        "scope": "zinc_binding",
                        "reference": "synthetic-unit-test-NOT-real-experiment",
                        "assay": "fixture",
                        "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    },
                }
            )
    snapshot_structures(design / "structures", rows)
    (design / "selection.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    (design / "dilutions.json").write_text(json.dumps({"replicates": {}}))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "registry"))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "published"))
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "local")
    config = tmp_path / "preprocess.yaml"
    config.write_text(
        "run: wisdom.preprocessing.zinc.ZincPreprocessing.ZincPreprocessing\n"
        f"with:\n  design: {{file: {design}}}\n  dataset_name: toy-zinc\n"
        "  workers: 1\n  surface_resolution: 2.0\n  diffusion_spectral_modes_max: 8\n"
    )
    result = WorkRunner(execution_root=tmp_path / "runs").run(WorkConfig.from_yaml(config))
    assert result.status == "succeeded", result.to_dict()
    indexes = list((tmp_path / "published").rglob("index.jsonl"))
    assert len(indexes) == 1
    published = indexes[0].parent
    assert ZincValidation().audit(published, tmp_path / "audit")["verdict"] == "PASS"
    dataset = WisdomDataset(published, "train")
    batch = WisdomCollator(atom_spatial_k=8, surface_atom_k=8)([dataset[0], dataset[1]])
    assert batch["target"].tolist() == [0, 1]
    assert not torch.any(batch["atomic_numbers"] == 30)
    model = WisdomV2(
        hidden_dim=8,
        embedding_dim=4,
        atomic_layers=1,
        surface_layers=1,
        diffusion_spectral_modes=8,
        pooling_type="attention",
    )
    prediction = model(**_model_inputs(batch))["logits"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction, batch["target"])
    loss.backward()
    assert torch.isfinite(loss)
