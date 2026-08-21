from dataclasses import replace
from pathlib import Path

import gemmi
import numpy as np
import pytest

from preprocess.AtomicStructureBuilder import AtomicStructureBuilder
from preprocess.enums.AtomRole import AtomRole
from preprocess.enums.BondType import BondType
from preprocess.PreprocessConfig import PreprocessConfig
from preprocess.ProteinReader import ProteinReader
from preprocess.StructureCache import StructureCache


def _read(path: Path, config: PreprocessConfig | None = None):
    selected_config = config or PreprocessConfig()
    source = StructureCache(path.parent, download=False).resolve(str(path), path.parent)
    return ProteinReader(selected_config).read(source)


def test_underscore_chain_identifier_format(tmp_path: Path) -> None:
    cache = StructureCache(tmp_path, download=False)
    with pytest.raises(ValueError, match="identifier"):
        cache.resolve("XYZ#A,B", tmp_path)

    # A cached file is sufficient to inspect the public parsed source without network access.
    cached = tmp_path / "xyz.cif.gz"
    cached.write_bytes(b"cached")
    source = cache.resolve("XYZ_AB", tmp_path)
    assert source.protein_id == "xyz"
    assert source.chains == ("A", "B")
    assert source.name == "xyz_AB"


@pytest.mark.parametrize("fixture_name", ["pdb_path", "cif_path", "gz_pdb_path"])
def test_reads_pdb_mmcif_and_gzip(request: pytest.FixtureRequest, fixture_name: str) -> None:
    path: Path = request.getfixturevalue(fixture_name)
    protein, metadata = _read(path)
    atoms = [
        atom for chain in protein.chains for residue in chain.residues for atom in residue.atoms
    ]
    assert len(atoms) == 21
    assert sum(len(chain.residues) for chain in protein.chains) == 4
    assert np.allclose(np.mean([atom.position for atom in atoms], axis=0), 0.0)
    assert metadata.source_format in {"pdb", "mmcif"}


def test_chain_selection_and_source_selector_precedence(pdb_path: Path) -> None:
    config = PreprocessConfig(chains=["B"])
    source = StructureCache(pdb_path.parent, download=False).resolve(str(pdb_path), pdb_path.parent)
    chain_b, metadata_b = ProteinReader(config).read(source)
    chain_a, metadata_a = ProteinReader(config).read(replace(source, chains=("A",)))
    assert [chain.id for chain in chain_b.chains] == ["B"]
    assert [chain.id for chain in chain_a.chains] == ["A"]
    assert metadata_b.selected_chains == ("B",)
    assert metadata_a.selected_chains == ("A",)


def test_invalid_model_chain_and_filtering(pdb_path: Path) -> None:
    with pytest.raises(ValueError, match="model_index"):
        _read(pdb_path, PreprocessConfig(model_index=1))

    config = PreprocessConfig()
    source = StructureCache(pdb_path.parent, download=False).resolve(str(pdb_path), pdb_path.parent)
    with pytest.raises(ValueError, match="requested chains"):
        ProteinReader(config).read(replace(source, chains=("Z",)))

    inclusive = PreprocessConfig(
        include_hydrogens=True,
        include_waters=True,
        include_nonpolymer=True,
        include_metals=True,
    )
    protein, _ = _read(pdb_path, inclusive)
    residues = [residue for chain in protein.chains for residue in chain.residues]
    atoms = [atom for residue in residues for atom in residue.atoms]
    roles = AtomicStructureBuilder(inclusive.atom_radius).build(protein)["atom_role_ids"]
    assert any(atom.atomic_number == 1 for atom in atoms)
    assert AtomRole.WATER in roles
    assert AtomRole.METAL in roles
    assert any(residue.name == "LIG" for residue in residues)
    assert protein.explicit_connections

    metals_only, _ = _read(pdb_path, PreprocessConfig(include_metals=True))
    metal_arrays = AtomicStructureBuilder(6.0).build(metals_only)
    metal_residues = [residue for chain in metals_only.chains for residue in chain.residues]
    assert AtomRole.METAL in metal_arrays["atom_role_ids"]
    assert not any(residue.name == "LIG" for residue in metal_residues)


def test_altloc_uses_highest_occupancy_deterministically(pdb_path: Path) -> None:
    protein, _ = _read(
        pdb_path,
        PreprocessConfig(chains=["A"], center_coordinates=False),
    )
    beta = next(
        atom
        for chain in protein.chains
        for residue in chain.residues
        if residue.number == 1
        for atom in residue.atoms
        if atom.name == "CB"
    )
    assert beta.position == pytest.approx((1.45, 1.7, 0.0))


def test_mmcif_explicit_bond_order(cif_path: Path) -> None:
    document = gemmi.cif.read(str(cif_path))
    block = document.sole_block()
    block.set_mmcif_category(
        "_struct_conn.",
        {
            "id": ["ligand_double"],
            "conn_type_id": ["covale"],
            "ptnr1_label_asym_id": ["Ax2"],
            "ptnr1_label_comp_id": ["LIG"],
            "ptnr1_label_seq_id": ["."],
            "ptnr1_label_atom_id": ["C1"],
            "ptnr1_auth_asym_id": ["A"],
            "ptnr1_auth_seq_id": ["103"],
            "pdbx_ptnr1_PDB_ins_code": ["?"],
            "ptnr2_label_asym_id": ["Ax2"],
            "ptnr2_label_comp_id": ["LIG"],
            "ptnr2_label_seq_id": ["."],
            "ptnr2_label_atom_id": ["O1"],
            "ptnr2_auth_asym_id": ["A"],
            "ptnr2_auth_seq_id": ["103"],
            "pdbx_ptnr2_PDB_ins_code": ["?"],
            "pdbx_value_order": ["doub"],
        },
    )
    explicit_path = cif_path.with_name("explicit.cif")
    document.write_file(str(explicit_path))
    protein, _ = _read(
        explicit_path,
        PreprocessConfig(
            include_nonpolymer=True,
            center_coordinates=False,
        ),
    )
    assert any(connection[3] == BondType.DOUBLE for connection in protein.explicit_connections)
