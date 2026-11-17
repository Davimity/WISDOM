"""The reduced DNA recipe must publish every requested structural family, not omit failures."""

import importlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import yaml
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from test_structural_enrichment import _complete_base, _work

from wisdom.preprocessing.common.structure.assets.external.ExternalProteinRepresentation import (
    ExternalProteinRepresentation,
)
from wisdom.preprocessing.common.structure.assets.external.StructureTokens import (
    StructureTokens,
)
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.dna.DNAValidation import DNAValidation
from wisdom.preprocessing.dna.preprocessing import publication

preprocessing_module = importlib.import_module(
    "wisdom.preprocessing.dna.preprocessing.Preprocessing"
)
STRUCTURAL_FAMILIES = frozenset(
    {
        "atomic_context",
        "extended_surface_atom",
        "sequence_context",
        "covalent_environment",
        "atomic_sasa",
        "residue_graph",
        "surface_local_graph",
        "spectral_extension",
        "surface_view",
        "surface_environment",
        "structure_tokens",
    }
)


def _recipe():
    """Read the actual researcher-facing recipe without publishing or resolving datasets."""
    root = Path(__file__).resolve().parents[1]
    config = yaml.safe_load((root / "experiments/preprocess/dna/dna_preprocess.yaml").read_text())
    return next(step["with"] for step in config["steps"] if step["name"] == "preprocess")


def test_dna_recipe_enables_all_structural_families_without_changing_population():
    options = _recipe()
    assert options["include_full_train"] is False
    assert options["train_dilutions"] == ["replicate-00/train-25"]
    assert options["include_validation"] is True and options["include_test"] is False
    for name in STRUCTURAL_FAMILIES - {"spectral_extension", "structure_tokens"}:
        assert options[name] is True
    assert options["spectral_modes"] == 256
    assert options["view_id"] == 1 and options["view_resolution"] == 1
    assert "external_model" not in options  # Keep model-qualified bundle assets.
    assert options["protein_models"] == {"file": "../../../data/models"}
    assert options["protein_model_kinds"] == ["esm2"]
    assert options["protein_static_tables"] is False
    assert options["structure_tokens"] is False


@pytest.mark.parametrize("failure", [None, "tool", "checkpoint", "static"])
def test_work_preflights_dependencies_and_forwards_exact_family_contract(
    tmp_path, monkeypatch, failure
):
    options = _recipe()
    models = Mock()
    models.requested_families = frozenset()
    models.requires_structure_tokens = False
    monkeypatch.setattr(preprocessing_module, "ProteinModelAssets", Mock(return_value=models))
    options["protein_models"] = tmp_path / "models"
    for name in ("train", "validation", "test", "catalog", "dilutions", "structures"):
        options[name] = tmp_path / name
    work = _work(tmp_path / "run")
    work.resources = SimpleNamespace(cpu=12)
    work.resuming = False
    work.tools = SimpleNamespace(
        require=Mock(return_value=SimpleNamespace(name="foldseek", version="test"))
    )
    if failure == "tool":
        options["structure_tokens"] = True
        work.tools.require.side_effect = FileNotFoundError("foldseek")
    if failure == "checkpoint":
        options["protein_models"] = None
        options["external_model"] = "esm2"
    if failure == "static":
        options["protein_models"] = None
        options["static_embedding_table"] = True
    manifests = Mock()
    manifests.load.return_value = [{"identifier": "tiny", "split": "train"}]
    monkeypatch.setattr(preprocessing_module, "DatasetManifests", Mock(return_value=manifests))
    snapshot = Mock(return_value=tmp_path / "snapshot")
    geometry = Mock(return_value=(tmp_path / "geometry", tmp_path / "geometry.json"))
    annotation = Mock(return_value=(tmp_path / "annotation", tmp_path / "annotation.json"))
    publish = Mock(return_value={"dataset": "test-only"})
    monkeypatch.setattr(preprocessing_module, "validate_structure_snapshot", snapshot)
    monkeypatch.setattr(preprocessing_module, "generate_geometry", geometry)
    monkeypatch.setattr(preprocessing_module, "generate_annotations", annotation)
    monkeypatch.setattr(preprocessing_module, "publish_dataset", publish)
    if failure:
        with pytest.raises(
            (ValueError, RuntimeError),
            match={
                "tool": "Foldseek is required",
                "checkpoint": "external_checkpoint",
                "static": "static_embedding_table requires",
            }[failure],
        ):
            preprocessing_module.Preprocessing.run(work, **options)
        snapshot.assert_not_called()
        geometry.assert_not_called()
        publish.assert_not_called()
        return
    result = preprocessing_module.Preprocessing.run(work, **options)
    assert result == {"dataset": "test-only"}
    work.tools.require.assert_not_called()
    manifest_args = manifests.load.call_args.kwargs
    assert manifest_args["include_full_train"] is False
    assert manifest_args["train_dilutions"] == ["replicate-00/train-25"]
    assert manifest_args["include_validation"] is True and manifest_args["include_test"] is False
    sidecars = publish.call_args.kwargs["structural_sidecars"]
    assert publish.call_args.kwargs["protein_models"] is models
    models.preflight.assert_called_once()
    assert sidecars.requested_families == STRUCTURAL_FAMILIES - {"structure_tokens"}
    assert sidecars.producers["spectral_extension"].parameters == {"modes": 256}
    assert geometry.call_args.kwargs["surface_atom_radius"] == 6
    assert geometry.call_args.kwargs["surface_atom_k_max"] == 32


@pytest.mark.parametrize(
    "omit_asset,include_models", [(False, False), (True, False), (False, True), (True, True)]
)
def test_dna_publication_joins_complete_assets_and_blocks_missing_families(
    tmp_path,
    pdb_path,
    monkeypatch,
    omit_asset,
    include_models,
):
    """Use real small geometry/producers; mock only task audit and official descriptor output."""
    annotation = tmp_path / "annotation"
    base = _complete_base(annotation, pdb_path)
    original = base.read_bytes()
    shutil.copyfile(pdb_path, annotation / "source.pdb")
    work = _work(tmp_path / "run")
    work.tools = SimpleNamespace(
        require=Mock(return_value=SimpleNamespace(name="foldseek", version="test"))
    )
    sidecars = StructuralSidecars(
        True,
        True,
        True,
        sequence_context=True,
        atomic_sasa=True,
        sasa_samples=24,
        covalent_environment=True,
        residue_graph=True,
        surface_environment=True,
        surface_local_graph=True,
        spectral_modes=12,
        surface_view=True,
        structure_tokens=True,
    )
    assert sidecars.requested_families == STRUCTURAL_FAMILIES
    neural_families = (
        {
            f"{prefix}_{kind}"
            for kind in ("esm2", "esmc", "saprot")
            for prefix in ("external_embeddings", "static_embedding_table")
        }
        if include_models
        else set()
    )
    required_families = STRUCTURAL_FAMILIES | neural_families

    def model_assets(work, bases, assets):
        """Join actual model-asset writers without fitting or loading large neural checkpoints."""
        paths = {"tiny": {}}
        for kind in ("esm2", "esmc", "saprot"):
            extractor = SimpleNamespace(
                kind=kind,
                dimension=4,
                metadata={
                    "extractor": kind,
                    "dimension": 4,
                    "frozen": True,
                    "checkpoint_sha256": "a" * 64,
                },
                encode=lambda words: np.ones((len(words), 4), np.float32),
                static_table=lambda: (np.ones((20, 4), np.float32), list("ACDEFGHIKLMNPQRSTVWY")),
            )
            producer = ExternalProteinRepresentation(extractor)
            for static, prefix in (
                (False, "external_embeddings"),
                (True, "static_embedding_table"),
            ):
                family = f"{prefix}_{kind}"
                if omit_asset and family == "static_embedding_table_esmc":
                    continue
                path = tmp_path / f"{family}.npz"
                producer.write(
                    bases["tiny"],
                    path,
                    assets["tiny"]["sequence_context"],
                    assets["tiny"]["structure_tokens"] if kind == "saprot" else None,
                    static=static,
                    family=family,
                )
                paths["tiny"][family] = path
        return paths

    models = (
        SimpleNamespace(requested_families=neural_families, generate=model_assets)
        if include_models
        else None
    )

    def write_index(root, rows, report):
        DatasetIndex.write(
            root / "members.jsonl",
            [
                DatasetMember(
                    "tiny",
                    assets={
                        name: DatasetAsset(
                            path=filename, sha256="sha256:" + ProteinArchive.sha256(root / filename)
                        )
                        for name, filename in (
                            ("universal_npz", "tiny.npz"),
                            ("source_structure", "source.pdb"),
                        )
                    },
                    partitions={"split": "train"},
                    targets={"dna_binding": 0},
                    metadata={"dilutions": ["replicate-00/train-25"]},
                )
            ],
        )

    def tokens(self, base, output, sequence, protein):
        with np.load(sequence, allow_pickle=False) as stored:
            count = len(stored["residue_chain"])
        StructuralAsset.write(
            output,
            base,
            {"token": np.full(count, "#", dtype="U1"), "valid": np.zeros(count, np.bool_)},
            "structure_tokens",
            {
                "tool": "foldseek",
                "version": "test",
                "method": "official-structureto3didescriptor",
            },
            {"algorithm": "saprot-official-3di-join-v1"},
        )

    def audit(self, root, output):
        member = next(iter(DatasetIndex(root / "members.jsonl")))
        assert member.assets.keys() >= required_families
        assert member.targets["dna_binding"] == 0 and member.partitions["split"] == "train"
        assert member.metadata["dilutions"] == ["replicate-00/train-25"]
        assert set(member.metadata["required_structural_assets"]) == required_families
        paths = {name: root / asset.path for name, asset in member.assets.items()}
        StructuralSidecars.validate_assets(paths["universal_npz"], paths)
        return {"verdict": "PASS"}

    monkeypatch.setattr(StructureTokens, "write", tokens)
    monkeypatch.setattr(publication, "_write_index", write_index)
    monkeypatch.setattr(publication, "_write_evidence", lambda root, rows: root.mkdir())
    monkeypatch.setattr(publication.DNAValidation, "audit", audit)
    if omit_asset:
        if not include_models:
            monkeypatch.setattr(sidecars, "generate", lambda *args, **kwargs: {"tiny": {}})
        with pytest.raises(RuntimeError, match="lacks required structural assets"):
            publication.publish_dataset(
                work,
                [{"identifier": "tiny", "leakage_group": "g"}],
                annotation,
                tmp_path / "geometry.json",
                tmp_path / "annotation.json",
                "test",
                "8",
                sidecars,
                1,
                protein_models=models,
            )
        work.outputs.dataset.assert_not_called()
        return
    publication.publish_dataset(
        work,
        [{"identifier": "tiny", "leakage_group": "g"}],
        annotation,
        tmp_path / "geometry.json",
        tmp_path / "annotation.json",
        "test",
        "8",
        sidecars,
        1,
        protein_models=models,
    )
    metadata = work.outputs.dataset.call_args.kwargs["metadata"]
    assert set(metadata["required_structural_assets"]) == required_families
    assert set(metadata["optional_universal_assets"]) == required_families
    report = json.loads((work.run_dir / "structural-enrichment-report" / "report.json").read_text())
    assert report["status"] == "validated" and report["completed"] == 1
    assert base.read_bytes() == original


def test_independent_dna_audit_rejects_missing_declared_structural_assets(tmp_path):
    """The standalone audit honours the member's completeness contract, not only present files."""
    member = DatasetMember("tiny", metadata={"required_structural_assets": ["atomic_sasa"]})
    with pytest.raises(ValueError, match="missing required structural assets"):
        DNAValidation._validate_member(tmp_path, member)
