"""Optional universal enrichment must not change historical geometry or task supervision."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gemmi
import numpy as np
import pytest
import yaml
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from scipy.spatial.transform import Rotation
from test_wisdom_v1 import _sample, _write_npz

from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.preprocessing.common.StructuralEnrichment import StructuralEnrichment
from wisdom.preprocessing.common.structure.AtomicContext import AtomicContext
from wisdom.preprocessing.common.structure.AtomicStructureBuilder import AtomicStructureBuilder
from wisdom.preprocessing.common.structure.context_chemistry import CHI_PATHS, EXACT_ATOMS
from wisdom.preprocessing.common.structure.ExtendedSurfaceAtom import ExtendedSurfaceAtom
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.ProteinPreprocessor import ProteinPreprocessor
from wisdom.preprocessing.common.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.StructuralSidecars import StructuralSidecars
from wisdom.preprocessing.common.structure.StructureResolver import StructureResolver
from wisdom.preprocessing.common.structure.SurfaceAtomNeighborhoodBuilder import (
    SurfaceAtomNeighborhoodBuilder,
)
from wisdom.preprocessing.dna.preprocessing import publication
from wisdom.preprocessing.zinc import ZincPreprocessing as zinc_module
from wisdom.utils.structure.models.Atom import Atom
from wisdom.utils.structure.models.Chain import Chain
from wisdom.utils.structure.models.Protein import Protein
from wisdom.utils.structure.models.Residue import Residue


def _chemistry():
    """Return canonical, interrupted and unknown residues with no task atoms or labels."""
    coordinates = (
        ((0, 1, 0), (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 0, 1), (1, 0, 1)),
        ((2, 0.5, 0.25), (2, 1, 0), (3, 1, 1), (3, 1, 2), (2, 2, 0), (2, 2, 1)),
        ((20, 0, 0), (20, 1, 0), (21, 1, 1), (21, 1, 2)),
        ((30, 0, 0), (31, 0, 0)),
    )
    residue_names = ("SER", "SER", "ALA", "UNK")
    atom_names = (("N", "CA", "C", "O", "CB", "OG"),) * 2 + (("N", "CA", "C", "O"), ("QX", "QY"))
    residues = []
    offset = 0
    for i, (name, names, xyz) in enumerate(
        zip(residue_names, atom_names, coordinates, strict=True)
    ):
        atoms = tuple(
            Atom(offset + j, atom, 7 if atom == "N" else 8 if atom.startswith("O") else 6, position)
            for j, (atom, position) in enumerate(zip(names, xyz, strict=True))
        )
        residues.append(Residue(name, 4 + i, "A" if i == 1 else "", True, atoms))
        offset += len(atoms)
    protein = Protein("synthetic", (Chain("AQ", tuple(residues)),))
    return protein, AtomicStructureBuilder(6).build(protein)


def _save(base, arrays):
    """Persist a unit-only base subset; publication tests use the complete real schema below."""
    metadata = {
        "preprocessing_schema_version": "3.0",
        "source_hash": "unit-test",
        "config_hash": "unit-test",
    }
    np.savez_compressed(base, **arrays, metadata_json=np.asarray(json.dumps(metadata)))


def test_torsions_connectivity_and_unknown_masks():
    protein, arrays = _chemistry()
    context = AtomicContext().build(arrays, protein)
    assert context["torsion_valid"][0, 1:3].all()
    assert context["torsion_valid"][1, 0]
    assert not context["torsion_valid"][0, 0]
    assert not context["torsion_valid"][1, 1:3].any()
    assert not context["torsion_valid"][2, :3].any()  # Residue order is not a peptide bond.
    assert context["torsion_valid"][:2, 3].all()
    assert not context["torsion_applicable"][3].any()
    assert not context["chemistry_known"][-2:].any()
    assert not context["atom_identity_id"][-2:].any()
    assert context["author_number"].tolist() == [4, 5, 6, 7]
    assert context["insertion_code"].tolist() == ["", "A", "", ""]
    assert context["chain_author_id"].tolist() == ["AQ"]
    assert np.all(context["torsion"][~context["torsion_valid"]] == 0)
    assert not context["occupancy_valid"].any()


def test_known_dihedral_and_degeneracy():
    angle, valid = AtomicContext.torsion(np.asarray([[0, 1, 0], [0, 0, 0], [1, 0, 0], [1, 0, 1]]))
    assert valid and angle == pytest.approx(np.pi / 2)
    assert AtomicContext.torsion(np.zeros((4, 3))) == (0, False)


def test_rotations_frames_and_covalent_angles():
    _, arrays = _chemistry()
    rotation = Rotation.from_euler("xyz", [20, 35, 70], degrees=True).as_matrix()
    transformed = {**arrays, "atom_positions": arrays["atom_positions"] @ rotation.T + [3, 4, 5]}
    left, right = AtomicContext(False).build(arrays), AtomicContext(False).build(transformed)
    np.testing.assert_allclose(left["torsion"], right["torsion"], atol=1e-6)
    frames, mask = AtomicContext.frames(arrays)
    rotated, other_mask = AtomicContext.frames(transformed)
    np.testing.assert_array_equal(mask, other_mask)
    np.testing.assert_allclose(rotated[mask], frames[mask] @ rotation.T, atol=1e-6)
    np.testing.assert_allclose(
        frames[mask] @ frames[mask].transpose(0, 2, 1),
        np.broadcast_to(np.eye(3), (mask.sum(), 3, 3)),
        atol=1e-6,
    )
    triplets, angles = AtomicContext.bond_angles(arrays)
    assert np.any(np.isclose(angles, np.pi / 2))
    covalent = {
        tuple(edge) for edge in arrays["atom_edge_index"][:, arrays["atom_edge_is_covalent"]].T
    }
    assert all(
        tuple(sorted((i, j))) in covalent and tuple(sorted((j, k))) in covalent
        for i, j, k in triplets
    )
    reflected = {**arrays, "atom_positions": arrays["atom_positions"] * [-1, 1, 1]}
    mirror, _ = AtomicContext.frames(reflected)
    expected = frames[mask] * [-1, 1, 1]
    expected[:, 2] *= -1
    np.testing.assert_allclose(mirror[mask], expected, atol=1e-6)


def test_context_hash_schema_quality_and_source_correspondence(tmp_path):
    protein, arrays = _chemistry()
    base, output = tmp_path / "base.npz", tmp_path / "context.npz"
    _save(base, arrays)
    before = base.read_bytes()
    context = AtomicContext()
    context.write(base, output, protein)
    assert context.validate(output, base, protein)
    assert base.read_bytes() == before
    with np.load(output, allow_pickle=False) as archive:
        assert all(archive[name].dtype != object for name in archive.files)
        assert "atom_names" not in archive.files and "residue_names" not in archive.files
    wrong = {**arrays, "atomic_numbers": arrays["atomic_numbers"] + 1}
    with pytest.raises(ValueError, match="atom order/geometry"):
        context.build(wrong, protein)
    _save(base, wrong)
    with pytest.raises(ValueError, match="hash mismatch"):
        context.validate(output, base)


@pytest.mark.parametrize("radius,maximum", [(6, 32), (12, 96), (12, 3)])
def test_csr_dense_exact_historic_and_truncation(radius, maximum):
    rng = np.random.default_rng(9)
    atoms = rng.normal(size=(100, 3)) * 3
    points = np.asarray([[0, 0, 0], [1, 1, 0], [50, 50, 50]], np.float32)
    normals = np.asarray([[0, 0, 1]] * 3, np.float32)
    arrays = dict(atom_positions=atoms, surface_positions=points, surface_normals=normals)
    builder = ExtendedSurfaceAtom(radius, maximum)
    values = builder.build(arrays)
    assert values["point_ptr"][-1] == len(values["atom_index"])
    assert values["point_ptr"][-1] == values["point_ptr"][-2]  # No padding atom in empty CSR row.
    np.testing.assert_array_equal(
        np.diff(values["point_ptr"]) + values["discarded_count"], values["candidate_count"]
    )
    dense = builder.dense(values, maximum)
    historic = SurfaceAtomNeighborhoodBuilder(radius, maximum).build(atoms, points[:2], normals[:2])
    for name in historic:
        np.testing.assert_array_equal(dense[name][:2], historic[name])
    assert np.all(dense["surface_atom_neighbors"][-1] == -1)
    assert not dense["surface_atom_mask"][-1].any()


def test_csr_ties_boundaries_rotation_and_fingerprint(tmp_path):
    atoms = np.asarray([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [2, 0, 0]], np.float32)
    arrays = dict(
        atom_positions=atoms,
        surface_positions=np.zeros((1, 3), np.float32),
        surface_normals=np.asarray([[0, 0, 1]], np.float32),
    )
    values = ExtendedSurfaceAtom(1, 2).build(arrays)
    assert values["atom_index"].tolist() == [0, 1]
    assert values["candidate_count"].tolist() == [3]
    assert values["discarded_count"].tolist() == [1]
    base, sidecar = tmp_path / "base.npz", tmp_path / "extended.npz"
    _save(base, arrays)
    builder = ExtendedSurfaceAtom(1, 2)
    builder.write(base, sidecar)
    assert builder.validate(sidecar, base)
    rotation = Rotation.from_euler("z", 90, degrees=True).as_matrix()
    shifted = {
        key: value @ rotation.T + (0 if "normals" in key else np.asarray([7, 6, 5]))
        for key, value in arrays.items()
    }
    transformed = builder.build(shifted)
    for key in ("distance", "normal_offset", "tangential"):
        np.testing.assert_allclose(transformed[key], values[key], atol=1e-6)
    _save(base, {**arrays, "atom_positions": atoms + 1})
    with pytest.raises(ValueError, match="hash mismatch"):
        builder.validate(sidecar, base)


def test_opt_in_csr_batch_uses_base_atom_counts_not_candidate_maximum():
    """Unrepresented atoms still reserve their namespace; no cross-protein candidate is created."""
    arrays = dict(
        atom_positions=np.asarray([[0, 0, 0], [1, 0, 0], [100, 0, 0]], np.float32),
        surface_positions=np.asarray([[0, 0, 1]], np.float32),
        surface_normals=np.asarray([[0, 0, 1]], np.float32),
    )
    values = ExtendedSurfaceAtom(2, 96).build(arrays)
    combined = ExtendedSurfaceAtom.batch([values, values], [3, 3])
    assert combined["atom_ptr"].tolist() == [0, 3, 6]
    assert combined["surface_ptr"].tolist() == [0, 1, 2]
    assert combined["atom_index"].tolist() == [0, 1, 3, 4]
    assert combined["point_ptr"].tolist() == [0, 2, 4]
    with pytest.raises(ValueError, match="namespace"):
        ExtendedSurfaceAtom.batch([values], [1])


def test_reader_observed_quality_and_missing_columns(cif_path, tmp_path):
    document = gemmi.cif.read(str(cif_path))
    document.sole_block().set_pair("_exptl.method", "'X-RAY DIFFRACTION'")
    document.write_file(str(cif_path))
    source = StructureResolver(tmp_path).resolve(str(cif_path), tmp_path)
    protein, _ = ProteinReader(PreprocessConfig(chains=("A",)), preserve_quality=True).read(source)
    atoms = [a for chain in protein.chains for residue in chain.residues for a in residue.atoms]
    assert any(a.occupancy is not None for a in atoms)
    assert any(a.b_iso is not None for a in atoms)
    alternate = next(a for a in atoms if a.name == "CB" and a.alternate_count == 2)
    assert alternate.altloc == "B" and alternate.occupancy == pytest.approx(0.6)

    # Predicted-model confidence is sometimes stored in B's column; do not relabel it as an
    # experimental displacement. Source occupancy remains merely an explicitly supplied field.
    document = gemmi.cif.read(str(cif_path))
    document.sole_block().set_pair("_exptl.method", "'THEORETICAL MODEL'")
    document.write_file(str(cif_path))
    predicted, _ = ProteinReader(PreprocessConfig(chains=("A",)), preserve_quality=True).read(
        replace(source, sha256=ProteinArchive.sha256(cif_path))
    )
    assert all(
        a.b_iso is None
        for chain in predicted.chains
        for residue in chain.residues
        for a in residue.atoms
    )
    # Gemmi can still parse coordinates without these columns: its implicit defaults must not
    # become observed source values in our optional sidecar.
    document = gemmi.cif.read(str(cif_path))
    document.sole_block().find_values("_atom_site.occupancy").erase()
    document.sole_block().find_values("_atom_site.B_iso_or_equiv").erase()
    document.write_file(str(cif_path))
    protein, _ = ProteinReader(PreprocessConfig(chains=("A",)), preserve_quality=True).read(
        replace(source, sha256=hashlib.sha256(cif_path.read_bytes()).hexdigest())
    )
    assert all(
        a.occupancy is None and a.b_iso is None
        for chain in protein.chains
        for residue in chain.residues
        for a in residue.atoms
    )


def test_complete_archive_enrichment_resume_and_disabled_no_read(pdb_path, tmp_path):
    record = {"key": str(pdb_path), "identifier": str(pdb_path), "output_name": "tiny.npz"}
    config = PreprocessConfig(chains=("A",), diffusion_spectral_modes_max=8)
    ProteinPreprocessor(config).process(record, pdb_path, tmp_path, tmp_path)
    base = tmp_path / "tiny.npz"
    original = base.read_bytes()
    builds = []

    def managed_file(name, build, validate):
        path = tmp_path / name
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            build(path)
            builds.append(name)
        assert validate(path)
        return path

    def mapped(rows, fn, **kwargs):
        return [fn(row) for row in rows]

    work = SimpleNamespace(
        log=lambda text: None, map=mapped, checkpoints=SimpleNamespace(file=managed_file)
    )
    sidecars = StructuralSidecars(True, True, True)
    result = sidecars.generate(work, {"tiny": base}, 1)
    assert set(result["tiny"]) == {"atomic_context", "extended_surface_atom"}
    assert len(builds) == 2
    assert sidecars.generate(work, {"tiny": base}, 2) == result
    assert len(builds) == 2
    StructuralSidecars.validate_assets(base, result["tiny"])
    assert base.read_bytes() == original
    assert ProteinArchive(config).can_resume(
        base, hashlib.sha256(pdb_path.read_bytes()).hexdigest()
    )
    assert StructuralSidecars().generate(work, {"missing": tmp_path / "absent"}, 1) == {
        "missing": {}
    }

    # Task IDs/labels never enter the enrichment key. Two independent tasks sharing exactly the
    # same universal archive reuse the identical common sidecars, not task-specific chemistry.
    same = sidecars.generate(work, {"dna-member": base, "zinc-member": base}, 1)
    assert same["dna-member"] == same["zinc-member"] == result["tiny"]
    assert len(builds) == 2
    wider = StructuralSidecars(True, True, True, 14).generate(work, {"tiny": base}, 1)
    assert wider["tiny"]["atomic_context"] == result["tiny"]["atomic_context"]
    assert wider["tiny"]["extended_surface_atom"] != result["tiny"]["extended_surface_atom"]
    assert len(builds) == 3

    # A compatible imported context must survive later candidate-only enrichment even when
    # the original coordinate path is no longer available to reopen.
    existing = sidecars.generate(work, {"tiny": base}, 1, existing=result)
    assert existing == result
    assert len(builds) == 3


@pytest.mark.parametrize("name", sorted(CHI_PATHS))
def test_canonical_chi_paths_match_gemmi_dihedral(name):
    """All supported side-chain conventions agree with an independent geometry implementation."""
    names = [atom for residue, atom in EXACT_ATOMS if residue == name]
    rng = np.random.default_rng(53)
    positions = rng.normal(size=(len(names), 3)).astype(np.float32)
    atoms = tuple(
        Atom(i, atom, 6, tuple(xyz))
        for i, (atom, xyz) in enumerate(zip(names, positions, strict=True))
    )
    protein = Protein("chi", (Chain("A", (Residue(name, 1, "", True, atoms),)),))
    arrays = AtomicStructureBuilder(6).build(protein)
    context = AtomicContext(False).build(arrays)
    for column, template in enumerate(CHI_PATHS[name], start=3):
        expected = gemmi.calculate_dihedral(
            *[gemmi.Position(*positions[names.index(atom)]) for atom in template.split()]
        )
        assert context["torsion_valid"][0, column]
        assert context["torsion"][0, column] == pytest.approx(expected, abs=1e-6)


def test_disulfide_and_unknown_vs_false_chemistry():
    """Actual covalent S-S support differs from spatial proximity and unknown component roles."""
    residues = tuple(
        Residue("CYS", i + 1, "", True, (Atom(i, "SG", 16, (float(i) * 2.05, 0, 0)),))
        for i in range(2)
    )
    arrays = AtomicStructureBuilder(6).build(Protein("ss", (Chain("A", residues),)))
    context = AtomicContext(False).build(arrays)
    assert context["heavy_degree"].tolist() == [1, 1]
    assert np.all(context["functional_group_bits"] & (1 << 10))
    assert not np.any(context["functional_group_bits"] & (1 << 6))
    generic = {**arrays, "atom_edge_bond_type": np.ones_like(arrays["atom_edge_bond_type"])}
    assert np.all(AtomicContext(False).build(generic)["functional_group_bits"] & (1 << 10))
    spatial_only = {
        **arrays,
        "atom_edge_is_covalent": np.zeros_like(arrays["atom_edge_is_covalent"]),
        "atom_edge_bond_type": np.zeros_like(arrays["atom_edge_bond_type"]),
    }
    context = AtomicContext(False).build(spatial_only)
    assert not context["heavy_degree"].any()
    assert not np.any(context["functional_group_bits"] & (1 << 10))


def test_atomic_identity_symmetry_missing_atoms_and_masks():
    """Exact names stay distinct; an absent chi-defining atom does not invent an angle."""
    names = ["N", "CA", "C", "CB", "CG", "OD1", "OD2"]
    rng = np.random.default_rng(1)
    atoms = tuple(Atom(i, name, 6, tuple(rng.normal(size=3))) for i, name in enumerate(names))
    residue = Residue("ASP", 1, "", True, atoms)
    protein = Protein("asp", (Chain("A", (residue,)),))
    arrays = AtomicStructureBuilder(6).build(protein)
    context = AtomicContext(False).build(arrays)
    assert context["atom_identity_id"][-1] != context["atom_identity_id"][-2]
    assert context["torsion_valid"][0, 4]
    truncated = replace(protein, chains=(Chain("A", (replace(residue, atoms=atoms[:-2]),)),))
    context = AtomicContext(False).build(AtomicStructureBuilder(6).build(truncated))
    assert context["torsion_applicable"][0, 4]
    assert not context["torsion_valid"][0, 4]


def test_source_order_repeated_chains_and_independent_quality_flag():
    """Chain indices distinguish duplicate author names, and quality can be disabled separately."""
    protein, _ = _chemistry()
    first = protein.chains[0].residues[0]
    second = replace(
        first,
        atoms=tuple(
            replace(
                atom,
                index=atom.index + len(first.atoms),
                position=tuple(np.asarray(atom.position) + 20),
            )
            for atom in first.atoms
        ),
    )
    duplicated = replace(protein, chains=(Chain("AQ", (first,)), Chain("AQ", (second,))))
    arrays = AtomicStructureBuilder(6).build(duplicated)
    context = AtomicContext(False).build(arrays, duplicated)
    assert context["chain_author_id"].tolist() == ["AQ", "AQ"]
    assert context["residue_chain_index"].tolist() == [0, 1]
    assert "occupancy" not in context


def test_both_task_yaml_enrichment_contract():
    """Shared families are deliberately enabled without changing the historical base settings."""
    root = Path(__file__).parents[1]
    for task, version in (("dna", "8"), ("zinc", "2")):
        settings = yaml.safe_load(
            (root / f"experiments/preprocess/{task}/{task}_preprocess.yaml").read_text()
        )
        preprocess = next(
            step["with"] for step in settings["steps"] if step["name"] == "preprocess"
        )
        assert preprocess["dataset_version"] == version
        assert preprocess["extended_radius"] == 12 and preprocess["extended_k_max"] == 96
        assert preprocess["atomic_context"] and preprocess["extended_surface_atom"]
        assert preprocess.get("surface_atom_radius", 6) == 6
        assert preprocess.get("surface_atom_k_max", 32) == 32


def _work(tmp_path):
    """Emulate managed ownership without touching a real Registry or scheduler."""

    def managed_file(name, build, validate):
        path = tmp_path / "checkpoints" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            build(path)
        assert validate(path)
        return path

    def directory(name, **kwargs):
        path = tmp_path / name
        path.mkdir(parents=True, exist_ok=True)
        return path

    outputs = SimpleNamespace(
        artifact=lambda name, path, **kwargs: path,
        directory=directory,
        dataset=Mock(return_value={"dataset": "unit-test-only"}),
        dataset_preflight=Mock(),
    )
    return SimpleNamespace(
        run_dir=tmp_path,
        log=Mock(),
        outputs=outputs,
        checkpoints=SimpleNamespace(file=managed_file),
        metrics=SimpleNamespace(log=Mock()),
        map=lambda rows, fn, **kwargs: [fn(row) for row in rows],
        resume_map=lambda rows, fn, **kwargs: [fn(row) for row in rows],
    )


def _complete_base(root, pdb_path):
    """Build the real universal fixture once; task tests never publish it to a Registry."""
    record = {"key": str(pdb_path), "identifier": str(pdb_path), "output_name": "tiny.npz"}
    ProteinPreprocessor(PreprocessConfig(chains=("A",), diffusion_spectral_modes_max=8)).process(
        record, pdb_path, root, root
    )
    return root / "tiny.npz"


def test_dna_publication_attaches_common_assets_before_audit(tmp_path, pdb_path, monkeypatch):
    """The actual DNA publication stage carries sidecars and advertises the real base schema."""
    annotation = tmp_path / "annotation"
    base = _complete_base(annotation, pdb_path)
    original = base.read_bytes()
    work = _work(tmp_path / "run")

    def index(root, rows, report):
        DatasetIndex.write(
            root / "members.jsonl",
            [
                DatasetMember(
                    "tiny",
                    assets={
                        "universal_npz": DatasetAsset(
                            path="tiny.npz", sha256="sha256:" + ProteinArchive.sha256(base)
                        )
                    },
                )
            ],
        )

    def audit(self, root, output):
        member = next(iter(DatasetIndex(root / "members.jsonl")))
        paths = {key: root / asset.path for key, asset in member.assets.items()}
        assert set(paths) == {"universal_npz", "atomic_context", "extended_surface_atom"}
        StructuralSidecars.validate_assets(paths["universal_npz"], paths)
        return {"verdict": "PASS"}

    monkeypatch.setattr(publication, "_write_index", index)
    monkeypatch.setattr(publication, "_write_evidence", lambda root, rows: root.mkdir())
    monkeypatch.setattr(publication.DNAValidation, "audit", audit)
    publication.publish_dataset(
        work,
        [{"identifier": "tiny", "leakage_group": "g"}],
        annotation,
        tmp_path / "geometry.json",
        tmp_path / "annotations.json",
        "dna",
        "new",
        StructuralSidecars(True, True, True),
        1,
    )
    settings = work.outputs.dataset.call_args.kwargs
    assert settings["metadata"]["structural_schema"] == ProteinArchive.SCHEMA_VERSION == "3.0"
    assert base.read_bytes() == original


def test_zinc_publication_attaches_identical_universal_families(tmp_path, pdb_path, monkeypatch):
    """The real Zn orchestration passes generic options and publishes native optional assets."""
    geometry = tmp_path / "geometry"
    base = _complete_base(geometry, pdb_path)
    original = base.read_bytes()
    report = tmp_path / "geometry.json"
    report.write_text(json.dumps({"records": [{"identifier": "tiny_A", "output": "tiny.npz"}]}))
    design = tmp_path / "design"
    design.mkdir()
    row = {
        "identifier": "tiny_A",
        "pdb_id": "tiny",
        "label": 0,
        "split": "train",
        "leakage_group": "g",
        "global_phenotype": "p",
        "local_phenotype": "not_applicable",
    }
    (design / "selection.jsonl").write_text(json.dumps(row) + "\n")
    (design / "dilutions.json").write_text('{"replicates":{}}')
    work = _work(tmp_path / "run")
    monkeypatch.setattr(zinc_module, "validate_structure_snapshot", lambda *args: design)
    monkeypatch.setattr(zinc_module, "generate_geometry", lambda *args: (geometry, report))

    def annotate(self, base, path, row):
        with path.open("wb") as stream:
            np.savez_compressed(stream, local_gt_available=np.asarray(False))

    def audit(self, members, output):
        for member in members:
            StructuralSidecars.validate_assets(member["assets"]["universal_npz"], member["assets"])
        return {"verdict": "PASS"}

    monkeypatch.setattr(zinc_module.ZincAnnotation, "write", annotate)
    monkeypatch.setattr(zinc_module.ZincAnnotation, "validate", lambda *args, **kwargs: True)
    monkeypatch.setattr(zinc_module.ZincValidation, "audit_members", audit)
    zinc_module.ZincPreprocessing.run(
        work, design=design, workers=1, atomic_context=True, extended_surface_atom=True
    )
    member = work.outputs.dataset.call_args.kwargs["members"][0]
    assert member["targets"]["zinc_binding"] == 0
    assert {"atomic_context", "extended_surface_atom"} <= member["assets"].keys()
    assert base.read_bytes() == original


def test_historical_loader_ignores_unrequested_enrichment_assets(tmp_path):
    """Adding optional assets cannot decode new arrays or change old training inputs by default."""
    _write_npz(tmp_path / "base.npz", _sample(3, 3, 0.0))
    (tmp_path / "unused.bin").write_bytes(b"intentionally not an NPZ")
    member = DatasetMember(
        "fixture",
        partitions={"split": "train"},
        targets={"dna_binding": 0},
        assets={"universal_npz": DatasetAsset(path="base.npz")},
    )
    DatasetIndex.write(tmp_path / "index.jsonl", [member])
    before = WisdomDataset(tmp_path, "train")[0]
    assets = {
        **dict(member.assets),
        "atomic_context": DatasetAsset(path="unused.bin"),
        "extended_surface_atom": DatasetAsset(path="unused.bin"),
    }
    DatasetIndex.write(tmp_path / "index.jsonl", [replace(member, assets=assets)])
    after = WisdomDataset(tmp_path, "train")[0]
    assert before.keys() == after.keys()
    for name in before:
        if hasattr(before[name], "numpy"):
            np.testing.assert_array_equal(before[name].numpy(), after[name].numpy())
        else:
            assert before[name] == after[name]


def test_existing_context_survives_native_geometry_reuse_publication(
    tmp_path, pdb_path, monkeypatch
):
    """Candidate-only publication retains valid source context and reports its actual schema."""
    dataset = tmp_path / "dataset"
    base = _complete_base(dataset, pdb_path)
    metadata = ProteinArchive(PreprocessConfig()).read_metadata(base)
    protein = StructuralSidecars.source_protein(metadata)
    context = dataset / "context.npz"
    AtomicContext().write(base, context, protein)
    before = context.read_bytes()
    member = DatasetMember(
        "tiny",
        partitions={"split": "train"},
        targets={"dna_binding": 0},
        assets={
            "universal_npz": DatasetAsset(path="tiny.npz"),
            "atomic_context": DatasetAsset(path="context.npz"),
        },
    )
    DatasetIndex.write(dataset / "index.jsonl", [member])
    monkeypatch.setattr(StructuralSidecars, "source_protein", staticmethod(lambda *args: None))
    work = _work(tmp_path / "work")
    StructuralEnrichment.run(work, dataset, "unit-only", "new", atomic_context=False, workers=1)
    published = work.outputs.dataset.call_args.kwargs
    assert published["metadata"]["atomic_context_schema"] == "1.0"
    assert published["metadata"]["base_geometry_recomputed"] is False
    assert published["members"][0]["targets"] == {"dna_binding": 0}
    assert context.read_bytes() == before
    assert set(published["members"][0]["assets"]) == {
        "universal_npz",
        "atomic_context",
        "extended_surface_atom",
    }
