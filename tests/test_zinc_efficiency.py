"""Exact assembly summaries, spawn-safe inspection and shared zinc asset contracts."""

import hashlib
import importlib
import multiprocessing
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path
from unittest.mock import Mock

import gemmi
import numpy as np
import pytest
import yaml
from test_surface_features_and_zinc import _assembly

from wisdom.preprocessing.common.structure.assets.surface.SurfaceEnvironment import (
    SurfaceEnvironment,
)
from wisdom.preprocessing.zinc.stages.discovery import inspect_entry
from wisdom.preprocessing.zinc.stages.structures import _surface_points
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination
from wisdom.preprocessing.zinc.ZincPreprocessing import ZincPreprocessing
from wisdom.preprocessing.zinc.ZincValidation import ZincValidation
from wisdom.utils.structure.ProteinStructure import ProteinStructure


@pytest.mark.parametrize("cutoff", [1.5, 2.0, 3.0, 3.2])
@pytest.mark.parametrize("minimum_donors,minimum_residues", [(1, 1), (2, 2), (3, 2)])
def test_summary_equals_all_copy_site_evidence(cutoff, minimum_donors, minimum_residues):
    """Repeated names, shared sites and zero-site copies preserve exact acceptance counts."""
    assembly = _assembly()
    original = assembly.model[0].clone()
    distant = original.clone()
    distant.name = "EMPTY"
    for residue in distant:
        for atom in residue:
            atom.pos.x += 100
    assembly.model.add_chain(original)
    assembly.model.add_chain(distant)
    verifier = ZincCoordination(cutoff, 0.5, minimum_donors, minimum_residues)
    counts = Counter()
    expected = []
    for chain in assembly.protein_chains():
        counts[chain.name] += 1
        sites = verifier.analyse(assembly, chain.name, counts[chain.name])
        accepted = [site for site in sites if site["accepted"]]
        expected.append({
            "protein_chain": chain.name,
            "protein_copy": counts[chain.name],
            "accepted_sites": len(accepted),
            "donor_count": sum(site["selected_donor_count"] for site in accepted),
        })
    assert verifier.summarize_assembly(assembly) == expected
    assert expected[-1]["accepted_sites"] == 0


def test_summary_constructs_one_donor_index_and_respects_occupancy(monkeypatch):
    """Index construction is per assembly; low-occupancy donors remain excluded."""
    assembly = _assembly()
    assembly.model[0][0][0].occ = 0.25
    module = importlib.import_module("wisdom.preprocessing.zinc.ZincCoordination")
    tree = Mock(wraps=module.cKDTree)
    monkeypatch.setattr(module, "cKDTree", tree)
    summary = ZincCoordination().summarize_assembly(assembly)
    assert summary == [{"protein_chain": "AQ", "protein_copy": 1,
                        "accepted_sites": 0, "donor_count": 0}]
    assert tree.call_count == 1


def test_mmcif_is_parsed_once_with_identical_coordinates(cif_path, monkeypatch):
    """The reused CIF block preserves Gemmi geometry, sequence and metadata extraction."""
    expected = gemmi.read_structure(str(cif_path))
    expected.setup_entities()
    expected.assign_label_seq_id()
    read = Mock(wraps=gemmi.cif.read)
    monkeypatch.setattr(gemmi.cif, "read", read)
    actual = ProteinStructure(cif_path)
    assert read.call_count == 1
    assert actual.structure.make_mmcif_document().as_string() == (
        expected.make_mmcif_document().as_string()
    )
    assert actual.sha256() == hashlib.sha256(cif_path.read_bytes()).hexdigest()


def test_pure_entry_inspection_matches_spawned_process(cif_path, tmp_path):
    """Native process inspection needs no Work, cache, lock or GPU in its child payload."""
    structure = gemmi.read_structure(str(cif_path))
    structure.setup_entities()
    for chain in structure[0]:
        entity = structure.get_entity_of(chain.get_polymer())
        if entity is not None:
            entity.full_sequence = [residue.name for residue in chain.get_polymer()]
    assembly = gemmi.Assembly("1")
    generator = gemmi.Assembly.Gen()
    generator.chains = [chain.name for chain in structure[0]]
    operator = gemmi.Assembly.Operator()
    operator.name = "1"
    generator.operators.append(operator)
    assembly.generators.append(generator)
    structure.assemblies.append(assembly)
    path = tmp_path / "source.cif"
    structure.make_mmcif_document().write_file(str(path))
    entry = {"pdb_id": "test", "path": str(path)}
    inspect = partial(inspect_entry, coordination_cutoff=3.0, minimum_occupancy=0.5,
                      minimum_donors=2, minimum_residues=2)
    expected = inspect(entry)
    assert expected and any(not row["candidate_verified"] for row in expected)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=1, mp_context=context) as pool:
        assert pool.submit(inspect, entry).result(timeout=45) == expected


def test_reusing_parsed_source_preserves_surface_points(cif_path):
    """No changed conformer, centering, float32 rounding or assembly rotation is introduced."""
    structure = ProteinStructure(cif_path)
    row = {"identifier": "TEST_A", "pdb_id": "test", "protein_chain": "A"}
    args = (cif_path, row, structure.sha256(), 1.0, 1.4, np.eye(3), np.zeros(3))
    expected = _surface_points(*args)
    actual = _surface_points(*args, structure=structure.structure)
    np.testing.assert_array_equal(actual, expected)


def test_missing_model_fails_before_geometry(tmp_path, monkeypatch):
    """An unavailable required model cannot waste geometry work or silently omit a family."""
    module = importlib.import_module("wisdom.preprocessing.zinc.ZincPreprocessing")
    monkeypatch.setattr(module, "StructureTokens", Mock())
    geometry = Mock(side_effect=AssertionError("geometry started before model preflight"))
    monkeypatch.setattr(module, "generate_geometry", geometry)
    work = Mock()
    with pytest.raises(FileNotFoundError, match=r"checkpoints\.json"):
        ZincPreprocessing.run(work, design=tmp_path, protein_models=tmp_path)
    geometry.assert_not_called()
    work.outputs.dataset.assert_not_called()


def test_independent_audit_detects_missing_required_family(tmp_path):
    """Asset requirements survive publication and are audited independently of byte hashes."""
    member = {"id": "missing", "assets": {}, "targets": {"zinc_binding": 0},
              "partitions": {"split": "train", "leakage_group": "g"},
              "metadata": {"required_structural_assets": ["sequence_context"]}}
    result = ZincValidation().audit_members([member], tmp_path)
    assert result["verdict"] == "FAIL"
    assert "missing required structural assets" in str(result["failures"])


def test_zinc_recipe_includes_atomic_residue_and_all_v15_inputs():
    """The production recipe covers V14/V15 requirements rather than only basic geometry."""
    root = Path(__file__).parents[1]
    recipe = yaml.safe_load(
        (root / "experiments/preprocess/zinc/zinc_preprocess.yaml").read_text())
    step = next(step for step in recipe["steps"] if step["name"] == "preprocess")
    options = step["with"]
    assert options["atomic_checkpoint"] == {
        "file": "../../../data/external-models/slae/autoencoder.ckpt"}
    assert options["atomic_device"] == options["external_device"] == "cuda"
    assert step["resources"]["gpu"] == 1 and options["workers"] > 1
    assert options["protein_models"] == {"file": "../../../data/models"}
    assert options["protein_model_kinds"] == ["esm2"]
    assert options["protein_static_tables"] is False
    assert options["sequence_context"] and options["structure_tokens"] is False
    assert options["surface_environment"]
    assert options["environment_options"]["horizons"] == [4.0, 15.0]
    fields = set()
    for path in (root / "experiments/v15").glob("*.yaml"):
        study = yaml.safe_load(path.read_text())
        for value in study["sweep"]["space"]["surface_options"]["values"]:
            fields.update(value.get("environment", {}).get("features", ()))
    assert fields == {"travel_depth", "occlusion_near", "occlusion_far", "empty_radius"}
    assert fields <= set(SurfaceEnvironment.FEATURES)
