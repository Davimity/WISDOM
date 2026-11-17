"""Small offline assets exercise physical correspondence, isolation and historical regression."""

import importlib
import json
import os
import shutil
import sys
from dataclasses import replace
from itertools import pairwise
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock
from zipfile import BadZipFile

import gemmi
import numpy as np
import pytest
import torch
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember
from lambdaforge.work.tools import ToolService
from scipy.spatial import cKDTree
from test_structural_enrichment import _chemistry, _complete_base, _save, _work

from wisdom.data.StructuralInputs import StructuralInputs
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.evaluation.SurfaceViewCorrespondence import SurfaceViewCorrespondence
from wisdom.preprocessing.common.StructuralEnrichment import StructuralEnrichment
from wisdom.preprocessing.common.structure.assets.context.AtomicContext import AtomicContext
from wisdom.preprocessing.common.structure.assets.context.AtomicSASA import AtomicSASA
from wisdom.preprocessing.common.structure.assets.context.CovalentEnvironment import (
    CovalentEnvironment,
)
from wisdom.preprocessing.common.structure.assets.context.SequenceContext import SequenceContext
from wisdom.preprocessing.common.structure.assets.external.ExternalProteinRepresentation import (
    ExternalProteinRepresentation,
)
from wisdom.preprocessing.common.structure.assets.external.ProteinRepresentationExtractor import (
    ProteinRepresentationExtractor,
)
from wisdom.preprocessing.common.structure.assets.external.StructureTokens import (
    StructureTokens,
)
from wisdom.preprocessing.common.structure.assets.graphs.ResidueGraph import ResidueGraph
from wisdom.preprocessing.common.structure.assets.graphs.SurfaceLocalGraph import SurfaceLocalGraph
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars
from wisdom.preprocessing.common.structure.assets.surface.SpectralExtension import SpectralExtension
from wisdom.preprocessing.common.structure.assets.surface.SurfaceView import SurfaceView
from wisdom.preprocessing.common.structure.geometry.AtomicStructureBuilder import (
    AtomicStructureBuilder,
)
from wisdom.preprocessing.common.structure.geometry.SurfaceBuilder import SurfaceBuilder
from wisdom.preprocessing.common.structure.io.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.io.StructureResolver import StructureResolver
from wisdom.preprocessing.common.structure.io.VerifiedStructureSource import VerifiedStructureSource
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.Training import _create_model, _model_inputs


@pytest.mark.parametrize("resolution", [0.3, 1.0, 2.7])
def test_voxel_vectorization_preserves_nonunit_spacing_and_first_index_ties(resolution):
    """Require exact former-loop point/owner selection, including duplicate-distance ties."""
    rng = np.random.default_rng(92)
    points = rng.normal(size=(600, 3)) * 3
    points = np.concatenate((points, points[:40]))
    owners = np.arange(len(points), dtype=np.int32)
    origin = points.min(axis=0)
    cells = np.floor((points - origin) / resolution).astype(np.int64)
    order = np.lexsort((owners, cells[:, 2], cells[:, 1], cells[:, 0]))
    ordered = cells[order]
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(ordered, axis=0), axis=1)) + 1, len(order)]
    chosen = []
    for start, stop in pairwise(boundaries):
        group = order[start:stop]
        center = origin + (cells[group[0]] + 0.5) * resolution
        distances = np.sum((points[group] - center) ** 2, axis=1)
        chosen.append(group[np.argmin(distances)])
    actual, actual_owners = SurfaceBuilder(resolution)._voxel_select(points, owners)
    assert np.array_equal(actual, points[chosen])
    assert np.array_equal(actual_owners, owners[chosen])


def test_sequence_physical_residues_insertions_and_multiple_chains(tmp_path):
    protein, arrays = _chemistry()
    sequence = SequenceContext().build(arrays, protein)
    assert len(sequence["residue_chain"]) == 4
    assert sequence["observed_tokens"].tolist()[:3] == ["S", "S", "A"]
    assert sequence["author_number"][:3].tolist() == [4, 5, 6]
    assert sequence["insertion_code"][1] == "A"
    assert len(sequence["complete_tokens"]) == 0
    assert not sequence["complete_valid"].any()
    assert sequence["segment_index"][0] == sequence["segment_index"][1]
    assert sequence["segment_index"][2] != sequence["segment_index"][1]
    base, path = tmp_path / "base.npz", tmp_path / "sequence.npz"
    _save(base, arrays)
    SequenceContext().write(base, path, protein)
    assert SequenceContext().validate(path, base)
    values, header = StructuralAsset.read(path, base, "sequence_context")
    values["observed_index"][0] = len(values["observed_tokens"])
    StructuralAsset.write(path, base, values, "sequence_context", {}, header)
    with pytest.raises(ValueError, match="range"):
        SequenceContext().validate(path, base)


def test_missing_source_preserves_verified_context_identity(tmp_path):
    protein, arrays = _chemistry()
    context = AtomicContext().build(arrays, protein)
    actual = SequenceContext().build(arrays, context=context)
    assert actual["author_valid"].all()
    assert actual["author_number"].tolist() == [4, 5, 6, 7]
    contradictory = replace(
        protein,
        chains=(
            replace(
                protein.chains[0],
                residues=(
                    replace(protein.chains[0].residues[0], number=999),
                    *protein.chains[0].residues[1:],
                ),
            ),
        ),
    )
    with pytest.raises(ValueError, match="conflicts"):
        SequenceContext().build(arrays, contradictory, context=context)


def test_sasa_isolated_atom_and_buried_center():
    producer = AtomicSASA(samples=48)
    arrays = {
        "atom_positions": np.asarray([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
        "vdw_radii": np.asarray([1.0, 3.0]),
    }
    values = producer.build(arrays)
    assert values["sasa"][0] == 0
    assert values["sasa"][1] == pytest.approx(4 * np.pi * 4.4**2)
    assert values["valid"].all()


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("samples", [24, 48, 960])
def test_sasa_survivor_pruning_is_bitwise_identical_to_full_quadrature(dtype, samples):
    """Pruning buried directions preserves the former exact sphere test, including equality."""
    rng = np.random.default_rng(17)
    atoms = np.concatenate((rng.uniform(-4, 4, (80, 3)), [[0, 0, 0], [0, 0, 0], [20, 0, 0]]))
    radii = np.concatenate((rng.uniform(1.2, 1.9, 80), [1.0, 3.0, 1.7]))
    arrays = {"atom_positions": atoms.astype(dtype), "vdw_radii": radii.astype(dtype)}
    atoms = arrays["atom_positions"].astype(np.float64)
    expanded = arrays["vdw_radii"].astype(np.float64) + 1.4
    directions = SurfaceBuilder(1).fibonacci_sphere(samples)
    tree = cKDTree(atoms)
    expected = np.empty(len(atoms), np.float32)
    for index, (atom, radius) in enumerate(zip(atoms, expanded, strict=True)):
        exposed = np.ones(samples, np.bool_)
        points = atom + radius * directions
        for other in tree.query_ball_point(atom, float(radius + expanded.max())):
            if other != index:
                exposed &= np.sum((points - atoms[other]) ** 2, axis=1) >= expanded[other] ** 2
        expected[index] = 4 * np.pi * radius ** 2 * exposed.mean()
    actual = AtomicSASA(samples=samples).build(arrays)
    assert np.array_equal(actual["sasa"], expected)
    assert actual["valid"].all()


def test_residue_minimum_contacts_do_not_use_ca_or_truncated_candidates(tmp_path):
    _, arrays = _chemistry()
    arrays["surface_positions"] = np.asarray([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]], np.float32)
    producer = ResidueGraph(contact_radius=3, surface_radius=3)
    result = producer.build(arrays)
    assert len(result["residue_chain"]) == 4
    assert np.array_equal(result["atom_residue"], arrays["residue_indices"])
    assert result["surface_ptr"][-1] == result["surface_ptr"][-2]
    ids = result["surface_residue"][: result["surface_ptr"][1]]
    assert len(ids) == len(set(ids))
    for column, (left, right) in enumerate(result["edge_index"].T):
        assert left < right
        a = arrays["atom_positions"][arrays["residue_indices"] == left]
        b = arrays["atom_positions"][arrays["residue_indices"] == right]
        expected = np.min(np.linalg.norm(a[:, None] - b[None, :], axis=2))
        assert result["edge_distance"][column] == pytest.approx(expected)
    base, path = tmp_path / "base.npz", tmp_path / "residue.npz"
    _save(base, arrays)
    producer.write(base, path)
    assert producer.validate(path, base)


def test_portable_equivalent_source_and_wrong_structure(tmp_path, pdb_path):
    local_source = tmp_path / "source.pdb"
    local_source.write_bytes(pdb_path.read_bytes())
    pdb_path = local_source
    base = _complete_base(tmp_path, pdb_path)
    metadata_path = tmp_path / "portable.cif"
    gemmi.read_structure(str(pdb_path)).make_mmcif_document().write_file(str(metadata_path))
    # Move the source out of its old path without touching researcher data; fixtures own it.
    moved = pdb_path.with_name("removed-original.pdb")
    pdb_path.rename(moved)
    protein, _, level = VerifiedStructureSource.recover(base, metadata_path)
    assert protein is not None and level == "equivalent"
    assert all(
        atom.occupancy is None for c in protein.chains for r in c.residues for atom in r.atoms
    )
    structure = gemmi.read_structure(str(metadata_path))
    structure[0][0][0][0].pos.x += 1
    structure.make_mmcif_document().write_file(str(metadata_path))
    with pytest.raises(ValueError, match=r"geometry|frame"):
        VerifiedStructureSource.recover(base, metadata_path)


def test_new_sidecars_publish_and_load_selectively_without_base_rebuild(tmp_path, pdb_path):
    base = _complete_base(tmp_path, pdb_path)
    original = base.read_bytes()
    work = _work(tmp_path)
    producer = StructuralSidecars(
        True,
        True,
        True,
        sequence_context=True,
        atomic_sasa=True,
        sasa_samples=24,
        residue_graph=True,
        surface_local_graph=True,
    )
    assets = producer.generate(work, {"protein": base}, 1)["protein"]
    assert set(assets) == {
        "atomic_context",
        "extended_surface_atom",
        "sequence_context",
        "atomic_sasa",
        "residue_graph",
        "surface_local_graph",
    }
    member = DatasetMember(
        "protein",
        partitions={"split": "train"},
        targets={"dna_binding": 0},
        assets={
            key: DatasetAsset(path=str(path.relative_to(tmp_path)))
            for key, path in {"universal_npz": base, **assets}.items()
        },
    )
    DatasetIndex.write(tmp_path / "index.jsonl", [member])
    historical = WisdomDataset(tmp_path, "train")[0]
    requested = WisdomDataset(tmp_path, "train", structural_assets=tuple(assets))[0]
    assert not any("__" in key for key in historical)
    for key, value in historical.items():
        if isinstance(value, torch.Tensor):
            torch.testing.assert_close(value, requested[key], rtol=0, atol=0)
    batch = WisdomCollator(diffusion_spectral_modes=8)([requested, requested])
    assert batch["structural_assets"]["residue_graph"]["namespace"] == "protein-local"
    csr = batch["structural_assets"]["extended_surface_atom"]["disjoint_csr"]
    n = len(requested["atomic_numbers"])
    p = csr["point_ptr"][len(requested["surface_area_weights"])]
    assert torch.all(csr["atom_index"][:p] < n)
    assert torch.all(csr["atom_index"][p:] >= n)
    assert base.read_bytes() == original
    predictor, _ = _create_model(
        1,
        {
            "hidden_dim": 8,
            "embedding_dim": 4,
            "atomic_layers": 1,
            "surface_layers": 1,
            "diffusion_spectral_modes": 8,
        },
    )
    predictor.eval()
    with torch.no_grad():
        original_prediction = predictor(
            **_model_inputs(WisdomCollator(diffusion_spectral_modes=8)([historical, historical]))
        )["logits"]
        requested_prediction = predictor(**_model_inputs(batch))["logits"]
    torch.testing.assert_close(original_prediction, requested_prediction, rtol=0, atol=0)
    reused = producer.generate(work, {"protein": base}, 2, existing={"protein": assets})
    assert reused["protein"] == assets
    with pytest.raises(ValueError, match="lacks requested"):
        StructuralInputs(("surface_view",)).register("missing", base, assets)


def test_views_preserve_physical_atoms_and_have_own_operators(tmp_path, pdb_path):
    base = _complete_base(tmp_path, pdb_path)
    with np.load(base) as stored:
        original = {key: stored[key] for key in stored.files if key != "metadata_json"}
    zero, _ = SurfaceView(0).build(base)
    for key in (
        "surface_positions",
        "surface_normals",
        "surface_curvatures",
        "surface_atom_neighbors",
    ):
        np.testing.assert_array_equal(zero[key], original[key])
    first, _ = SurfaceView(1).build(base)
    second, _ = SurfaceView(1).build(base)
    np.testing.assert_array_equal(first["surface_positions"], second["surface_positions"])
    assert not np.array_equal(first["surface_positions"], original["surface_positions"])
    assert first["diffusion_eigenvectors"].shape[0] == len(first["surface_positions"])
    path = tmp_path / "view.npz"
    SurfaceView(1).write(base, path)
    assert SurfaceView(1).validate(path, base)


def test_filtered_surface_graph_rejects_opposite_cavity_sides():
    arrays = {
        "surface_positions": np.asarray([[0.0, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.2, 0.0]]),
        "surface_normals": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
    }
    result = SurfaceLocalGraph().build(arrays)
    assert (0, 1) not in map(tuple, result["surface_edge_index"].T)
    assert np.all(result["surface_edge_index"][0] < result["surface_edge_index"][1])


def test_external_representation_offline_contract_and_static_separation(tmp_path):
    protein, arrays = _chemistry()
    base, sequence = tmp_path / "base.npz", tmp_path / "sequence.npz"
    _save(base, arrays)
    SequenceContext().write(base, sequence, protein)
    extractor = SimpleNamespace(
        kind="esm2",
        dimension=3,
        metadata={"checkpoint_sha256": "a" * 64, "frozen": True, "dimension": 3},
        encode=lambda tokens: np.asarray(
            [[ord(token), 1.0, len(tokens)] for token in tokens], np.float32
        ),
        static_table=lambda: (np.ones((20, 3), np.float32), list("ACDEFGHIKLMNPQRSTVWY")),
    )
    producer = ExternalProteinRepresentation(extractor)
    contextual, static = tmp_path / "external.npz", tmp_path / "static.npz"
    producer.write(base, contextual, sequence)
    producer.write(base, static, sequence, static=True)
    assert producer.validate(contextual, base)
    assert producer.validate(static, base, static=True)
    with np.load(contextual) as stored:
        assert stored["embedding"].shape == (4, 3)
        assert stored["valid"][:3].all()
    with np.load(static) as stored:
        assert stored["weights"].shape == (20, 3)
    assert (
        StructuralAsset.read(contextual, base, "external_embeddings")[1]["parameters"][
            "sequence_policy"
        ]
        == "fragments"
    )


@pytest.mark.parametrize("deposited_methionine", ["MET", "MSE"])
def test_deposited_sequence_missing_positions_insertions_and_ligands(
    tmp_path, deposited_methionine,
):
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("AQ")
    for i, (number, insertion, name, label) in enumerate(
        ((5, " ", "ALA", 1), (5, "A", "ALA", 3), (9, " ", "MSE", 4))
    ):
        residue = gemmi.Residue()
        residue.name = name
        residue.seqid = gemmi.SeqId(number, insertion)
        residue.label_seq = label
        residue.subchain = "Axp"
        residue.entity_type = gemmi.EntityType.Polymer
        atom = gemmi.Atom()
        atom.name = "CA"
        atom.element = gemmi.Element("C")
        atom.pos = gemmi.Position(i * 4.0, 0.0, 0.0)
        residue.add_atom(atom)
        chain.add_residue(residue)
    ligand = gemmi.Residue()
    ligand.name = "ZN"
    ligand.seqid = gemmi.SeqId(100, " ")
    ligand.subchain = "zinc"
    ligand.entity_type = gemmi.EntityType.NonPolymer
    atom = gemmi.Atom()
    atom.name, atom.element, atom.pos = "ZN", gemmi.Element("Zn"), gemmi.Position(1.0, 3.0, 0.0)
    ligand.add_atom(atom)
    chain.add_residue(ligand)
    model.add_chain(chain)
    structure.add_model(model)
    entity = gemmi.Entity("1")
    entity.entity_type = gemmi.EntityType.Polymer
    entity.polymer_type = gemmi.PolymerType.PeptideL
    entity.subchains = ["Axp"]
    entity.full_sequence = ["ALA", "GLY", "ALA", deposited_methionine]
    structure.entities.append(entity)
    path = tmp_path / "deposited.cif"
    structure.make_mmcif_document().write_file(str(path))
    source = StructureResolver(tmp_path).resolve(str(path), tmp_path)
    protein, _ = ProteinReader(PreprocessConfig(include_metals=True)).read(source)
    arrays = AtomicStructureBuilder(6).build(protein)
    result = SequenceContext().build(arrays, protein, path)
    assert result["complete_tokens"].tolist() == list("AGAM")
    assert result["complete_index"].tolist() == [0, 2, 3, -1]
    assert result["observed_tokens"].tolist() == list("AAM")
    assert result["is_protein_polymer"].tolist() == [True, True, True, False]
    assert result["author_number"].tolist() == [5, 5, 9, 100]
    assert result["insertion_code"].tolist() == ["", "A", "", ""]
    # A mislabeled full sequence cannot silently align identical alanines to the wrong position.
    document = gemmi.cif.read(str(path))
    table = document.sole_block().find_values("_entity_poly_seq.mon_id")
    table[0] = "GLY"
    document.write_file(str(path))
    with pytest.raises(ValueError, match="sequence disagrees"):
        SequenceContext().build(arrays, protein, path)


@pytest.mark.parametrize("component,letter", [
    ("MSE", "M"), ("PTR", "Y"), ("PCA", "E"), ("LLP", "K"), ("SEP", "S"), ("TPO", "T"),
])
def test_modified_deposited_monomers_join_their_exact_physical_residues(
    tmp_path, component, letter,
):
    """Parent normalization is chemical dictionary lookup, not sequence realignment or repair."""
    structure = gemmi.Structure()
    model = gemmi.Model(1)
    chain = gemmi.Chain("A")
    residue = gemmi.Residue()
    residue.name, residue.seqid, residue.label_seq = component, gemmi.SeqId(151, " "), 1
    residue.subchain, residue.entity_type = "Axp", gemmi.EntityType.Polymer
    atom = gemmi.Atom()
    atom.name, atom.element, atom.pos = "CA", gemmi.Element("C"), gemmi.Position(0, 0, 0)
    residue.add_atom(atom)
    chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    entity = gemmi.Entity("1")
    entity.entity_type, entity.polymer_type = gemmi.EntityType.Polymer, gemmi.PolymerType.PeptideL
    entity.subchains, entity.full_sequence = ["Axp"], [component]
    structure.entities.append(entity)
    path = tmp_path / "modified.cif"
    structure.make_mmcif_document().write_file(str(path))
    source = StructureResolver(tmp_path).resolve(str(path), tmp_path)
    protein, _ = ProteinReader(PreprocessConfig()).read(source)
    arrays = AtomicStructureBuilder(6).build(protein)
    values = SequenceContext().build(arrays, protein, path)
    assert values["canonical_letter"].tolist() == [letter]
    assert values["complete_tokens"].tolist() == [letter]
    assert values["complete_index"].tolist() == [0]
    assert values["complete_valid"].all()


def test_sequence_preflight_fails_before_costly_assets_and_preserves_completed_files(
    tmp_path, monkeypatch, pdb_path,
):
    """Source joins fail up front; successful per-family checkpoints are reused on the retry."""
    base = _complete_base(tmp_path / "geometry", pdb_path)
    work = _work(tmp_path / "work")
    producer = StructuralSidecars(sequence_context=True, atomic_sasa=True)
    original_build = SequenceContext.build
    write = producer.producers["atomic_sasa"].write
    writes = []

    def counted(*args, **kwargs):
        """Count actual atomic-area computation, not checkpoint validation."""
        writes.append(1)
        return write(*args, **kwargs)

    def invalid(*args, **kwargs):
        """Emulate a real source disagreement, never silently masking an invalid join."""
        raise ValueError("deposited sequence disagrees with a physical residue")

    monkeypatch.setattr(producer.producers["atomic_sasa"], "write", counted)
    first = producer.generate(work, {"GOOD": base}, 1)
    preserved = {name: path.read_bytes() for name, path in first["GOOD"].items()}
    assert writes == [1]
    monkeypatch.setattr(SequenceContext, "build", invalid)
    with pytest.raises(RuntimeError, match=r"preflight failed.*before expensive"):
        producer.generate(work, {"GOOD": base, "BAD": base}, 1)
    assert writes == [1]
    failures = json.loads((tmp_path / "work/sequence-preflight/failures.json").read_text())
    assert {entry["identifier"] for entry in failures} == {"GOOD", "BAD"}
    assert {name: path.read_bytes() for name, path in first["GOOD"].items()} == preserved
    monkeypatch.setattr(SequenceContext, "build", original_build)
    second = producer.generate(work, {"GOOD": base, "BAD": base}, 1)
    assert writes == [1]
    assert {name: path.read_bytes() for name, path in second["GOOD"].items()} == preserved


def test_asset_binding_partial_files_and_numeric_corruption(tmp_path):
    _, arrays = _chemistry()
    base, other, path = (tmp_path / name for name in ("base.npz", "other.npz", "sasa.npz"))
    _save(base, arrays)
    changed = dict(arrays)
    changed["atom_positions"] = arrays["atom_positions"] + 0.1
    _save(other, changed)
    producer = AtomicSASA(samples=24)
    producer.write(base, path)
    with pytest.raises(ValueError, match="hash mismatch"):
        producer.validate(path, other)
    values, header = StructuralAsset.read(path, base, "atomic_sasa")
    values["sasa"][0] += 1
    StructuralAsset.write(path, base, values, "atomic_sasa", producer.parameters, header)
    with pytest.raises(ValueError, match="reconstruction"):
        producer.validate(path, base)
    path.write_bytes(path.read_bytes()[:30])
    with pytest.raises((ValueError, BadZipFile, EOFError)):
        producer.validate(path, base)


def test_cov_histograms_and_spectral_prefix_extension(tmp_path, pdb_path):
    base = _complete_base(tmp_path, pdb_path)
    for producer in (CovalentEnvironment(), SpectralExtension(4), SpectralExtension(12)):
        output = tmp_path / (type(producer).__name__ + str(producer.parameters)[:10] + ".npz")
        producer.write(base, output)
        assert producer.validate(output, base)
    with np.load(base) as stored:
        atom_count = len(stored["atomic_numbers"])
    values, _ = StructuralAsset.read(output, base, "spectral_extension")
    assert len(values["eigenvalues"]) == 12 and atom_count > 0


def test_view_transport_and_rigid_tangent_components(tmp_path, pdb_path):
    base = _complete_base(tmp_path, pdb_path)
    output = tmp_path / "view.npz"
    SurfaceView(1).write(base, output)
    service = SurfaceViewCorrespondence()
    match = service.align_assets(base, output)
    with np.load(base) as stored:
        arrays = {key: stored[key] for key in stored.files if key != "metadata_json"}
    with np.load(output) as stored:
        view = stored["surface_positions"]
    constant, mask = service.transport(
        torch.ones(len(arrays["surface_positions"])), match, len(view)
    )
    assert torch.all(constant[mask] == 1)
    assert match["reference_area_coverage"] > 0
    smooth = torch.from_numpy(arrays["surface_positions"][:, 0])
    transported, mask = service.transport(smooth, match, len(view))
    assert torch.max(torch.abs(transported[mask] - torch.from_numpy(view[:, 0])[mask])) <= 1.5
    localized = torch.exp(-(smooth**2))
    assert torch.isfinite(service.transport(localized, match, len(view))[0]).all()
    rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    transformed = SurfaceView.rigid_transform(arrays, rotation, np.asarray([1.0, 2.0, 3.0]))
    rows = arrays["diffusion_gradient_index"][0]
    original_frames = np.asarray(
        [SurfaceBuilder._tangent_basis(n) for n in arrays["surface_normals"]]
    )
    changed_frames = np.asarray(
        [SurfaceBuilder._tangent_basis(n) for n in transformed["surface_normals"]]
    )
    old_vector = (
        original_frames[rows, 0] * arrays["diffusion_gradient_x"][:, None]
        + original_frames[rows, 1] * arrays["diffusion_gradient_y"][:, None]
    )
    new_vector = (
        changed_frames[rows, 0] * transformed["diffusion_gradient_x"][:, None]
        + changed_frames[rows, 1] * transformed["diffusion_gradient_y"][:, None]
    )
    np.testing.assert_allclose(new_vector, old_vector @ rotation.T, atol=1e-5)


@pytest.mark.parametrize("kind", ["esm2", "esmc", "saprot"])
def test_local_extractor_adapter_contract_without_real_model_download(
    tmp_path,
    monkeypatch,
    kind,
):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    dimension = 960 if kind == "esmc" else 480
    letters = list("ACDEFGHIKLMNPQRSTVWY")
    words = (
        [a + b for a in letters for b in "acdefghiklmnpqrstvwy"] if kind == "saprot" else letters
    )
    vocabulary = {word: i + 2 for i, word in enumerate(words)}
    tokenizer = SimpleNamespace(
        get_vocab=lambda: vocabulary,
        cls_token_id=0,
        eos_token_id=1,
        convert_tokens_to_ids=lambda tokens: [vocabulary[token] for token in tokens],
    )
    model = MagicMock()
    model.config = SimpleNamespace(
        hidden_size=dimension, num_hidden_layers=30 if kind == "esmc" else 12
    )
    model.eval.return_value, model.to.return_value = model, model
    model.side_effect = lambda input_ids, **kwargs: SimpleNamespace(
        hidden_states=(input_ids[:, :, None].float().expand(-1, -1, dimension),)
    )
    model.get_input_embeddings.return_value = SimpleNamespace(
        weight=torch.arange((len(words) + 2) * dimension).reshape(-1, dimension).float()
    )
    backend = SimpleNamespace(
        __version__="offline-test-double",
        AutoModel=MagicMock(),
        AutoModelForMaskedLM=MagicMock(),
        AutoTokenizer=MagicMock(),
        EsmcForMaskedLM=MagicMock(),
        EsmcTokenizer=MagicMock(),
    )
    backend.AutoModel.from_pretrained.return_value = model
    backend.AutoModelForMaskedLM.from_pretrained.return_value = SimpleNamespace(esm=model)
    backend.AutoTokenizer.from_pretrained.return_value = tokenizer
    backend.EsmcTokenizer.from_pretrained.return_value = tokenizer
    backend.EsmcForMaskedLM.from_pretrained.return_value = SimpleNamespace(esmc=model)
    original_import = importlib.import_module

    def requested_module(name):
        if name in {"transformers", "esm.models.esmc.model", "esm.models.esmc.tokenizer"}:
            return backend
        return original_import(name)

    monkeypatch.setattr(importlib, "import_module", requested_module)
    extractor = ProteinRepresentationExtractor(
        kind, checkpoint, "unit-only", "frozen-test", maximum_length=2
    )
    tokens = ["Aa", "Cc", "Dd"] if kind == "saprot" else list("ACD")
    result = extractor.encode(tokens)
    np.testing.assert_array_equal(result[:, 0], [vocabulary[token] for token in tokens])
    assert result.shape == (3, dimension) and model.call_count == 2
    assert model.requires_grad_.call_args.args == (False,)
    loader = backend.EsmcForMaskedLM if kind == "esmc" else backend.AutoModelForMaskedLM
    assert loader.from_pretrained.call_args.kwargs["local_files_only"] is True
    weights, amino = extractor.static_table()
    assert weights.shape == (20, dimension) and amino == letters
    previous_hash = extractor.metadata["checkpoint_sha256"]
    (checkpoint / "config.json").write_text('{"changed":true}')
    assert (
        ProteinRepresentationExtractor(kind, checkpoint, "unit-only", "frozen-test").metadata[
            "checkpoint_sha256"
        ]
        != previous_hash
    )
    with pytest.raises(ValueError, match="tokenizer lacks"):
        extractor.encode(["not-a-residue"])


def test_universal_publication_preserves_arbitrary_task_metadata(tmp_path, pdb_path):
    base = _complete_base(tmp_path, pdb_path)
    member = DatasetMember(
        "third-task",
        targets={"activity_score": 12.5},
        partitions={"split": "validation", "leakage_group": "frozen-group"},
        metadata={"dilutions": ["unchanged"], "custom": {"keep": True}},
        assets={"universal_npz": DatasetAsset(path=base.name)},
    )
    DatasetIndex.write(tmp_path / "index.jsonl", [member])
    work = _work(tmp_path)
    original = base.read_bytes()
    StructuralEnrichment.run(
        work,
        tmp_path,
        "test-only-family",
        "new",
        atomic_context=False,
        extended_surface_atom=False,
        sequence_context=True,
        workers=1,
    )
    published = work.outputs.dataset.call_args.kwargs["members"][0]
    assert published["targets"] == {"activity_score": 12.5}
    assert published["partitions"] == dict(member.partitions)
    assert published["metadata"] == dict(member.metadata)
    assert "sequence_context" in published["assets"]
    assert base.read_bytes() == original
    report = json.loads((tmp_path / "structural-enrichment-report/report.json").read_text())
    assert report["completed"] == 1 and report["failed"] == 0


def test_enrichment_failure_preserves_other_members_and_reports_them(tmp_path, pdb_path):
    base = _complete_base(tmp_path, pdb_path)
    work = _work(tmp_path)
    original = work.checkpoints.file

    def managed_file(name, build, validate=lambda path: True):
        return original(name, build, validate)

    work.checkpoints.file = managed_file
    producer = StructuralSidecars(atomic_sasa=True, sasa_samples=24)
    with pytest.raises(RuntimeError, match="1/2"):
        producer.generate(work, {"good": base, "missing": tmp_path / "missing.npz"}, 1)
    assert producer.report["completed"] == 1 and producer.report["failed"] == 1
    assert list((tmp_path / "checkpoints").rglob("atomic_sasa.npz"))
    assert list((tmp_path / "checkpoints").rglob("failures.json"))


def test_requested_csr_budget_is_checked_before_model_use(tmp_path, pdb_path):
    base = _complete_base(tmp_path, pdb_path)
    producer = StructuralSidecars(extended_surface_atom=True, extended_radius=8, extended_k_max=12)
    assets = producer.generate(_work(tmp_path), {"protein": base}, 1)["protein"]
    for radius, count in ((9, 12), (8, 13)):
        with pytest.raises(ValueError, match="exceeds stored"):
            StructuralInputs(("extended_surface_atom",), radius, count).register(
                "protein", base, assets
            )


@pytest.mark.external_tools
def test_official_foldseek_descriptor_on_small_verified_structure(tmp_path, pdb_path):
    search_path = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")
    if not shutil.which("foldseek", path=search_path):
        pytest.skip("official Foldseek binary is not installed")
    structure = gemmi.Structure()
    structure.name = "synthetic-helix"
    model, chain = gemmi.Model(1), gemmi.Chain("AQ")
    for i in range(24):
        residue = gemmi.Residue()
        residue.name, residue.seqid, residue.het_flag = "ALA", gemmi.SeqId(i + 1, " "), "A"
        for j, name in enumerate(("N", "CA", "C", "O", "CB")):
            atom = gemmi.Atom()
            atom.name, atom.element = (
                name,
                gemmi.Element("N" if name == "N" else "O" if name == "O" else "C"),
            )
            angle = i * np.pi * 100 / 180
            atom.pos = gemmi.Position(3 * np.cos(angle) + 0.4 * j, 3 * np.sin(angle), i * 1.5)
            residue.add_atom(atom)
        chain.add_residue(residue)
    model.add_chain(chain)
    structure.add_model(model)
    source = tmp_path / "helix.cif"
    structure.setup_entities()
    structure.make_mmcif_document().write_file(str(source))
    resolved = StructureResolver(tmp_path).resolve(str(source), tmp_path)
    protein, _ = ProteinReader(PreprocessConfig()).read(resolved)
    arrays = AtomicStructureBuilder(6).build(protein)
    base = tmp_path / "helix.npz"
    _save(base, arrays)
    quality = "exact"
    sequence, output = tmp_path / "sequence.npz", tmp_path / "3di.npz"
    SequenceContext().write(base, sequence, protein, source, quality=quality)
    work = SimpleNamespace(
        tools=ToolService(SimpleNamespace(emit=lambda *a, **k: None)),
        log=lambda *a, **k: None,
        temp_dir=tmp_path / "scratch",
    )
    producer = StructureTokens(work)
    producer.write(base, output, sequence, protein)
    assert producer.validate(output, base)
    with np.load(output) as stored:
        assert stored["valid"].sum() > 0


@pytest.mark.parametrize("kind", ["esm2", "esmc", "saprot"])
def test_real_external_model_explicitly_enabled_only(kind):
    config = os.environ.get("WISDOM_EXTERNAL_TEST_CONFIG")
    if not config:
        pytest.skip("set WISDOM_EXTERNAL_TEST_CONFIG to explicit local checkpoint configuration")
    with open(config) as handle:
        specifications = json.load(handle)
    if kind not in specifications:
        pytest.skip("this optional local checkpoint was not staged")
    spec = specifications[kind]
    extractor = ProteinRepresentationExtractor(
        kind, Path(spec["checkpoint"]), spec["identifier"], spec["revision"], maximum_length=2
    )
    values = extractor.encode(list("ACD") if kind != "saprot" else ["Aa", "Cc", "Dd"])
    assert values.shape == (3, extractor.dimension) and np.isfinite(values).all()
    weights, alphabet = extractor.static_table()
    assert weights.shape == (20, extractor.dimension) and len(alphabet) == 20
