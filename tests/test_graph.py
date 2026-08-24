from pathlib import Path

import numpy as np
import pytest

from wisdom.preprocessing.structure.AtomicStructureBuilder import AtomicStructureBuilder
from wisdom.preprocessing.structure.dataclasses.Atom import Atom
from wisdom.preprocessing.structure.dataclasses.Chain import Chain
from wisdom.preprocessing.structure.dataclasses.Protein import Protein
from wisdom.preprocessing.structure.dataclasses.Residue import Residue
from wisdom.preprocessing.structure.enums.BondType import BondType
from wisdom.preprocessing.structure.enums.ConnectionType import ConnectionType
from wisdom.preprocessing.structure.enums.Relation import Relation
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.structure.StructureCache import StructureCache
from wisdom.preprocessing.structure.SurfaceBuilder import SurfaceBuilder


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


def test_union_relation_masks_and_covalent_outside_radius() -> None:
    graph = AtomicStructureBuilder(1.1).build(_protein_with_explicit())
    pairs = [tuple(pair) for pair in graph["atom_edge_index"].T.tolist()]
    masks = dict(zip(pairs, graph["atom_edge_relation_mask"].tolist(), strict=True))
    assert masks[(0, 1)] == Relation.SPATIAL | Relation.COVALENT
    assert masks[(0, 2)] == Relation.COVALENT
    assert masks[(1, 2)] == Relation.COVALENT
    assert len(pairs) == len(set(pairs))
    assert np.all(graph["atom_edge_index"][0] < graph["atom_edge_index"][1])


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
    source = StructureCache(pdb_path.parent).resolve(str(pdb_path), pdb_path.parent)
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


def test_surface_atom_graph_never_falls_back_to_knn() -> None:
    with pytest.raises(ValueError, match="no atom within"):
        SurfaceBuilder(resolution=1.0, atom_radius=1.0).build(
            np.asarray([[0.0, 0.0, 0.0]]),
            np.asarray([1.7]),
        )
