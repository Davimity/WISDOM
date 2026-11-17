"""Historical Zn membership is immutable even when modern local references disagree."""

import json
from pathlib import Path

import gemmi
import numpy as np
import pytest
import torch
import yaml
from lambdaforge.data import DatasetIndex
from lambdaforge.work import WorkConfig, WorkRunner
from lambdaforge.work.cache import WorkCache
from test_surface_features_and_zinc import _assembly, _base

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.models.WisdomV2 import WisdomV2
from wisdom.preprocessing.common.clustering.descriptors import rigid_transform
from wisdom.preprocessing.zinc.stages.evidence import load_evidence
from wisdom.preprocessing.zinc.stages.legacy import read_membership
from wisdom.preprocessing.zinc.ZincAnnotation import ZincAnnotation
from wisdom.preprocessing.zinc.ZincValidation import ZincValidation
from wisdom.Training import _model_inputs

ROOT = Path(__file__).resolve().parents[1]


def input_files(root):
    """Create six inherited toy records; these are software fixtures, not assay evidence."""
    root.mkdir(parents=True, exist_ok=True)
    lists = {"train": ["F000_AQ", "F001_AQ"],
             "validation": ["F002_AQ", "F003_AQ"], "test": ["F004_AQ", "F005_AQ"],
             "positives": ["F001_AQ", "F003_AQ", "F005_AQ"],
             "negatives": ["F000_AQ", "F002_AQ", "F004_AQ"]}
    lists["all"] = [identifier for split in ("train", "validation", "test")
                    for identifier in lists[split]]
    files = {name: root / f"{name}.txt" for name in lists}
    for name, values in lists.items():
        files[name].write_text("\n".join(values) + "\n")
    return files


def test_original_tfg_membership_is_exact_balanced_and_untouched():
    """Check all real TXT sets without retrieving coordinates or revising historical labels."""
    root = ROOT / "data/zn_old"
    files = {name: root / filename for name, filename in {
        "train": "train.txt", "validation": "val.txt", "test": "test.txt",
        "positives": "positives.txt", "negatives": "negatives.txt", "all": "all.txt"}.items()}
    if not all(path.is_file() for path in files.values()):
        pytest.skip("Historical researcher-owned Zn inputs are not distributed in this checkout")
    original = {name: path.read_bytes() for name, path in files.items()}
    rows, audit = read_membership(files)
    assert len(rows) == 450
    assert audit["split_counts"] == {"train": {"0": 74, "1": 74},
                                    "validation": {"0": 74, "1": 74},
                                    "test": {"0": 77, "1": 77}}
    assert not audit["selection_performed"]
    assert not audit["homology_independence_verified"]
    assert {name: path.read_bytes() for name, path in files.items()} == original


@pytest.mark.parametrize("defect", ["split_overlap", "label_overlap", "missing", "duplicate"])
def test_legacy_input_errors_never_trigger_silent_repair(tmp_path, defect):
    """Broken original membership fails instead of dropping records or assigning new splits."""
    files = input_files(tmp_path)
    if defect == "split_overlap":
        files["validation"].write_text("F000_AQ\nF003_AQ\n")
    elif defect == "label_overlap":
        files["positives"].write_text(files["positives"].read_text() + "F000_AQ\n")
    elif defect == "missing":
        files["all"].write_text("F000_AQ\n")
    else:
        files["train"].write_text("F000_AQ\nF000_AQ\n")
    with pytest.raises(ValueError):
        read_membership(files)


@pytest.mark.parametrize("label", [0, 1])
def test_unsupported_legacy_local_GT_is_unavailable_not_relabelled(tmp_path, label):
    """Inherited negatives with Zn and positives without Zn cannot supply invented local GT."""
    base, target = tmp_path / "base.npz", tmp_path / "annotation.npz"
    _base(base)
    row = {"identifier": "TOY_AQ", "label": label, "split": "test", "assembly_id": "1",
           "protein_chain": "AQ", "protein_copy": 1, "structure_sha256": "fixture",
           "assembly_rotation": np.eye(3).tolist(), "assembly_translation": [0, 0, 0],
           "label_evidence": {"kind": "legacy_tfg_manifest"},
           "sites": [] if label else [{"accepted": False, "selected_donor_count": 1}]}
    ZincAnnotation().write(base, target, row)
    assert ZincAnnotation.validate(target, base)
    with np.load(target, allow_pickle=False) as arrays:
        assert not arrays["local_gt_available"].item()
        assert not arrays["surface_valid_mask"].any()
        assert not arrays["surface_target_hard"].any()
        assert json.loads(str(arrays["annotation_metadata_json"].item()))["label"] == label
    row["label_evidence"] = {"kind": "experimental_non_binding"}
    with pytest.raises(ValueError):
        ZincAnnotation().write(base, target, row)


def test_legacy_negative_support_is_not_accepted_by_modern_selection(tmp_path):
    """The historical exception is explicit and cannot weaken the modern negative contract."""
    path = tmp_path / "evidence.jsonl"
    path.write_text(json.dumps({"identifier": "TOY_AQ", "sequence": "AC", "label": 0,
        "assembly_id": "1", "protein_copy": 1,
        "label_evidence": {"kind": "legacy_tfg_manifest", "reference": "inputs/",
            "membership_sha256": "fixture", "scope": "zinc_binding",
            "sequence_sha256": "fixture", "experimental_confirmation": False}}) + "\n")
    with pytest.raises(ValueError, match="explicit sequence-bound"):
        load_evidence(path)
    assert load_evidence(path, allow_legacy=True)[0]["label"] == 0


def test_legacy_transform_uses_stored_zero_occupancy_coordinates_only_when_requested():
    """Reproduce 1C78_B's all-zero occupancies without weakening occupied evidence defaults."""
    chain = gemmi.Chain("B")
    residue = gemmi.Residue()
    residue.name, residue.seqid, residue.label_seq = "ALA", gemmi.SeqId(9, " "), 9
    residue.entity_type = gemmi.EntityType.Polymer
    for name, xyz in (("N", (0, 0, 0)), ("CA", (1, 0, 0)), ("C", (0, 1, 0))):
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = name, gemmi.Element("C"), gemmi.Position(*xyz)
        atom.occ = 0
        residue.add_atom(atom)
    chain.add_residue(residue)
    assembled = chain.clone()
    rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
    translation = np.array([10, -2, 4], dtype=float)
    for atom in assembled[0]:
        atom.pos = gemmi.Position(*(rotation @ np.array(list(atom.pos)) + translation))
    with pytest.raises(ValueError, match="three matching heavy atoms"):
        rigid_transform(chain, assembled)
    actual_rotation, actual_translation = rigid_transform(chain, assembled, include_unoccupied=True)
    np.testing.assert_allclose(actual_rotation, rotation, atol=1e-12)
    np.testing.assert_allclose(actual_translation, translation, atol=1e-12)


def test_legacy_recipe_matches_current_structural_recipe_without_dilution():
    """Every shared structural/frozen-encoder override equals the current full Zn pipeline."""
    root = ROOT / "experiments/preprocess/zinc"
    legacy = yaml.safe_load((root / "zinc_legacy_preprocess.yaml").read_text())
    modern = yaml.safe_load((root / "zinc_preprocess.yaml").read_text())["steps"][2]["with"]
    settings = legacy["with"]
    for field in modern.keys() & settings.keys() - {"dataset_name", "dataset_version", "workers"}:
        assert settings[field] == modern[field], field
    assert "design" not in settings and "steps" not in legacy
    assert settings["training_subset"] == "full"
    assert settings["dataset_name"] == "wisdom-zinc-tfg"


def test_native_legacy_publication_preserves_every_split_and_fails_on_later_label_changes(
    tmp_path, monkeypatch,
):
    """Use the real offline Work/geometry/publication/validator path, not a cluster job."""
    files = input_files(tmp_path / "inputs")
    sources = {}
    for index in range(6):
        source = _assembly().source.clone()
        atom = gemmi.Atom()
        atom.name, atom.element, atom.pos = "CA", gemmi.Element("C"), gemmi.Position(1, 1, 0)
        source[0][0][0].add_atom(atom)
        # One legacy negative has a real contact and one legacy positive has no deposited Zn.
        # Their membership and global labels must survive this contradiction unchanged.

        if index in (2, 3, 4):
            del source[0]["Z"]
        if index == 0:
            for residue in source[0][0]:
                for atom in residue:
                    atom.occ = 0
        source.setup_entities()
        entity = source.get_entity_of(source[0][0].get_polymer())
        entity.full_sequence = ["HIS", "ASP"]
        assembly = gemmi.Assembly("1")
        generator = gemmi.Assembly.Gen()
        generator.chains = [chain.name for chain in source[0]]
        operator = gemmi.Assembly.Operator()
        operator.name = "1"
        generator.operators.append(operator)
        assembly.generators.append(generator)
        source.assemblies.append(assembly)
        path = tmp_path / f"f{index:03d}.cif"
        source.make_mmcif_document().write_file(str(path))
        sources[f"f{index:03d}"] = path

    def fetch(cache, url, **options):
        """Replace only remote acquisition with exact local CIF fixtures."""
        return sources[url.rsplit("/", 1)[-1].split(".")[0].lower()]

    monkeypatch.setattr(WorkCache, "fetch", fetch)
    monkeypatch.setenv("LAMBDAFORGE_DATASET_REGISTRY", str(tmp_path / "registry"))
    monkeypatch.setenv("LAMBDAFORGE_DATASET_ROOT", str(tmp_path / "published"))
    monkeypatch.setenv("LAMBDAFORGE_CLUSTER", "local")
    config = tmp_path / "preprocess.yaml"
    settings = {f"legacy_{name}": {"file": str(path)} for name, path in files.items()}
    settings.update(dataset_name="toy-zinc-tfg", workers=1, surface_resolution=2.0,
                    diffusion_spectral_modes_max=8)
    config.write_text(yaml.safe_dump({"run": "wisdom.preprocessing.zinc.ZincPreprocessing."
        "ZincPreprocessing", "with": settings}))
    result = WorkRunner(execution_root=tmp_path / "runs").run(WorkConfig.from_yaml(config))
    assert result.status == "succeeded", result.to_dict()
    [index] = list((tmp_path / "published").rglob("index.jsonl"))
    published = index.parent
    expected, _ = read_membership(files)
    actual = list(DatasetIndex(index))
    assert {member.member_id: (member.targets["zinc_binding"], member.partitions["split"])
            for member in actual} == {
        row["identifier"]: (row["label"], row["split"]) for row in expected}
    verdict = ZincValidation().audit(published, tmp_path / "audit")
    assert verdict["verdict"] == "PASS", verdict
    assert verdict["protocol"] == "legacy_tfg"
    assert not verdict["homology_independence_verified"]
    assert verdict["warnings"]
    assert any("no occupied heavy atoms" in warning for warning in verdict["warnings"])
    assert not next(member for member in actual if member.member_id == "F003_AQ").targets[
        "local_ground_truth"]
    dataset = WisdomDataset(published, "train", include_surface_targets=True)
    batch = WisdomCollator(atom_spatial_k=8, surface_atom_k=8)([dataset[0], dataset[1]])
    model = WisdomV2(hidden_dim=8, embedding_dim=4, atomic_layers=1, surface_layers=1,
                     diffusion_spectral_modes=8, pooling_type="attention")
    logits = model(**_model_inputs(batch))["logits"]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, batch["target"])
    loss.backward()
    assert torch.isfinite(loss)
    assert not torch.any(batch["atomic_numbers"] == 30)
    # Tampering belongs only to the isolated toy dataset, never to user-managed assets.

    for member in actual:
        if "selection_evidence" in member.assets:
            original = published / member.assets["selection_evidence"].path / "inputs/train.txt"
            original.write_text("F001_AQ\n")
    assert ZincValidation().audit(published, tmp_path / "tampered-audit")["verdict"] == "FAIL"
