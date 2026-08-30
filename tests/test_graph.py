from pathlib import Path

import numpy as np
import pytest

from wisdom.preprocessing.structure.AtomicStructureBuilder import AtomicStructureBuilder
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.structure.StructureResolver import StructureResolver
from wisdom.preprocessing.structure.SurfaceAtomNeighborhoodBuilder import (
    SurfaceAtomNeighborhoodBuilder,
)
from wisdom.utils.structure.enums.BondType import BondType
from wisdom.utils.structure.enums.ConnectionType import ConnectionType
from wisdom.utils.structure.models.Atom import Atom
from wisdom.utils.structure.models.Chain import Chain
from wisdom.utils.structure.models.Protein import Protein
from wisdom.utils.structure.models.Residue import Residue


def _atom(index: int, x: float, name: str = "X") -> Atom:
    return Atom(index, name, 6, (x, 0.0, 0.0))


def _protein_with_explicit() -> Protein:
    atoms = (_atom(0, 0.0), _atom(1, 1.0), _atom(2, 10.0))
    residue = Residue("UNK", 1, "", False, atoms)
    return Protein(
        id="graph",
        chains=(Chain("A", (residue,)),),
        explicit_connections=(
            (atoms[1], atoms[2], ConnectionType.COVALENT, BondType.DOUBLE),
            (atoms[0], atoms[2], ConnectionType.COVALENT, BondType.TRIPLE),
        ),
    )


def test_bounded_spatial_ranks_and_covalent_edges_outside_radius() -> None:
    graph = AtomicStructureBuilder(1.1, max_neighbors=1).build(_protein_with_explicit())
    pairs = [tuple(pair) for pair in graph["atom_edge_index"].T.tolist()]
    covalent = dict(zip(pairs, graph["atom_edge_is_covalent"].tolist(), strict=True))
    ranks    = dict(zip(pairs, graph["atom_edge_spatial_rank"].tolist(), strict=True))

    assert covalent[(0, 1)] and ranks[(0, 1)] == 1
    assert covalent[(0, 2)] and ranks[(0, 2)] == 0
    assert covalent[(1, 2)] and ranks[(1, 2)] == 0
    assert len(pairs) == len(set(pairs))
    assert np.all(graph["atom_edge_index"][0] < graph["atom_edge_index"][1])


def test_smaller_atomic_k_is_a_deterministic_nested_subset() -> None:
    atoms   = tuple(_atom(index, float(index)) for index in range(6))
    protein = Protein(
        id="line",
        chains=(Chain("A", (Residue("UNK", 1, "", False, atoms),)),),
    )
    first  = AtomicStructureBuilder(10.0, max_neighbors=4).build(protein)
    second = AtomicStructureBuilder(10.0, max_neighbors=4).build(protein)

    assert all(np.array_equal(first[name], second[name]) for name in first)
    pairs = first["atom_edge_index"].T
    ranks = first["atom_edge_spatial_rank"]
    k1    = {tuple(pair) for pair in pairs[(ranks > 0) & (ranks <= 1)]}
    k3    = {tuple(pair) for pair in pairs[(ranks > 0) & (ranks <= 3)]}

    assert k1 < k3
    assert len(k3) <= 6 * 3


def test_explicit_double_and_triple_are_preserved() -> None:
    graph = AtomicStructureBuilder(0.1).build(_protein_with_explicit())
    pair_to_type = dict(
        zip(
            map(tuple, graph["atom_edge_index"].T.tolist()),
            graph["atom_edge_bond_type"].tolist(),
            strict=True,
        )
    )
    assert pair_to_type[(0, 2)] == BondType.TRIPLE
    assert pair_to_type[(1, 2)] == BondType.DOUBLE


def test_templates_peptides_aromatics_and_disulfides(pdb_path: Path) -> None:
    config = PreprocessConfig(chains=["A"])
    source = StructureResolver(pdb_path.parent).resolve(str(pdb_path), pdb_path.parent)
    protein, _ = ProteinReader(config).read(source)
    graph = AtomicStructureBuilder(0.1).build(protein)
    types = graph["atom_edge_bond_type"].tolist()
    assert types.count(BondType.PEPTIDE) == 2
    assert BondType.DISULFIDE in types

    names = ["CG", "CD1", "CE1", "CZ", "CE2", "CD2"]
    atoms = tuple(_atom(index, float(index), name) for index, name in enumerate(names))
    aromatic = Protein(
        id="phe",
        chains=(Chain("A", (Residue("PHE", 1, "", True, atoms),)),),
    )
    aromatic_graph = AtomicStructureBuilder(0.1).build(aromatic)
    assert BondType.AROMATIC in aromatic_graph["atom_edge_bond_type"]


def test_surface_atom_table_never_falls_back_outside_physical_radius() -> None:
    with pytest.raises(ValueError, match="no atom within"):
        SurfaceAtomNeighborhoodBuilder(radius=1.0, max_neighbors=4).build(
            atom_positions=np.asarray([[0.0, 0.0, 0.0]]),
            surface_positions=np.asarray([[3.0, 0.0, 0.0]]),
            surface_normals=np.asarray([[1.0, 0.0, 0.0]]),
        )


def test_surface_atom_table_matches_brute_force_and_is_nested() -> None:
    atoms = np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    points = np.asarray([[1.0, 0.0, 0.0]])
    normals = np.asarray([[1.0, 0.0, 0.0]])

    table = SurfaceAtomNeighborhoodBuilder(radius=5.0, max_neighbors=3).build(
        atoms,
        points,
        normals,
    )

    assert table["surface_atom_neighbors"].tolist() == [[0, 1, 2]]
    assert np.allclose(table["surface_atom_distances"], [[1.0, 1.0, 3.0]])
    assert table["surface_atom_neighbors"][:, :1].tolist() == [[0]]
    assert np.all(table["surface_atom_mask"])


def test_surface_atom_distance_ties_use_persisted_precision_then_atom_id() -> None:
    """Ensure float32 distance ties use atom ID rather than hidden float64 digits.

    The first atom is microscopically farther away in float64, but both distances become exactly
    one after conversion to the persisted float32 dtype. The serialized ordering contract must
    therefore use atom ID zero first, matching archive validation after reopening the NPZ.
    """
    atoms   = np.asarray([[1.00000004, 0.0, 0.0], [1.0, 0.0, 0.0]])
    points  = np.asarray([[0.0, 0.0, 0.0]])
    normals = np.asarray([[1.0, 0.0, 0.0]])

    table = SurfaceAtomNeighborhoodBuilder(radius=2.0, max_neighbors=2).build(
        atoms,
        points,
        normals,
    )

    assert table["surface_atom_distances"].tolist() == [[1.0, 1.0]]
    assert table["surface_atom_neighbors"].tolist() == [[0, 1]]


def test_surface_atom_tangent_uses_stable_vector_projection() -> None:
    """A nearly radial offset must retain its small tangential component after storage."""
    atoms   = np.asarray([[6.0, 0.003, 0.0]], dtype=np.float32)
    points  = np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32)
    normals = np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32)

    table = SurfaceAtomNeighborhoodBuilder(radius=7.0, max_neighbors=1).build(
        atoms,
        points,
        normals,
    )

    assert table["surface_atom_normal_offsets"][0, 0] == pytest.approx(6.0)
    assert table["surface_atom_tangential_distances"][0, 0] == pytest.approx(0.003)
