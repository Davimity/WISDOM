from pathlib import Path

import numpy as np
import pytest

from wisdom.utils.structure.ProteinStructure import ProteinStructure


def test_protein_structure_owns_digest_and_experimental_metadata(cif_path: Path) -> None:
    structure = ProteinStructure(cif_path)

    assert len(structure.sha256()) == 64
    assert structure.resolution is None
    assert structure.release_year is None
    assert structure.experimental_method == "unavailable"


def test_protein_structure_resolves_assembly_chains_and_dna_atoms(
    tmp_path: Path,
    pdb_path: Path,
) -> None:
    source = pdb_path.read_text(encoding="utf-8")
    assembly = "\n".join(
        (
            "REMARK 350 BIOMOLECULE: 1",
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A, D",
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000",
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000",
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000",
        )
    )
    dna = "\n".join(
        (
            "ATOM     28  P    DA D   1       1.000   4.000   0.000  1.00 20.00           P  ",
            "ATOM     29  O5'  DA D   1       2.000   4.000   0.000  1.00 20.00           O  ",
            "ATOM     30  C5'  DA D   1       3.000   4.000   0.000  1.00 20.00           C  ",
            "TER",
            "END",
        )
    )
    path = tmp_path / "protein-dna.pdb"
    content = source.replace(
        "HEADER    WISDOM TEST STRUCTURE",
        f"HEADER    WISDOM TEST STRUCTURE\n{assembly}",
    ).replace("END\n", dna + "\n")
    path.write_text(
        content,
        encoding="utf-8",
    )

    structure = ProteinStructure(path)
    assembled = structure.assembly("1")
    deposited, protein_copy = assembled.protein_copy("A", 1)
    positions, radii, owners = assembled.dna_atoms()

    assert deposited.name == "A"
    assert protein_copy.name == "A"
    assert len(assembled.protein_chains()) == 1
    assert [chain.name for chain in assembled.dna_chains()] == ["D"]
    assert positions.shape == (3, 3)
    assert radii.shape == (3,)
    assert owners.tolist() == [0, 0, 0]
    assert np.isfinite(positions).all()
    assert np.all(radii > 0.0)


def test_protein_structure_rejects_absent_assembly_copy(
    tmp_path: Path,
    pdb_path: Path,
) -> None:
    source = pdb_path.read_text(encoding="utf-8")
    assembly = "\n".join(
        (
            "REMARK 350 BIOMOLECULE: 1",
            "REMARK 350 APPLY THE FOLLOWING TO CHAINS: A",
            "REMARK 350   BIOMT1   1  1.000000  0.000000  0.000000        0.00000",
            "REMARK 350   BIOMT2   1  0.000000  1.000000  0.000000        0.00000",
            "REMARK 350   BIOMT3   1  0.000000  0.000000  1.000000        0.00000",
        )
    )
    path = tmp_path / "protein.pdb"
    content = source.replace(
        "HEADER    WISDOM TEST STRUCTURE",
        f"HEADER    WISDOM TEST STRUCTURE\n{assembly}",
    )
    path.write_text(
        content,
        encoding="utf-8",
    )

    structure = ProteinStructure(path)
    assembled = structure.assembly("1")

    with pytest.raises(ValueError, match="contains 1"):
        assembled.protein_copy("A", 2)
