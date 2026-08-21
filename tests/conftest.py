from __future__ import annotations

import gzip
from pathlib import Path

import gemmi
import pytest


@pytest.fixture
def pdb_path() -> Path:
    return Path(__file__).parent / "data" / "tiny.pdb"


@pytest.fixture
def cif_path(tmp_path: Path, pdb_path: Path) -> Path:
    output = tmp_path / "tiny.cif"
    structure = gemmi.read_structure(str(pdb_path))
    structure.make_mmcif_document().write_file(str(output))
    return output


@pytest.fixture
def gz_pdb_path(tmp_path: Path, pdb_path: Path) -> Path:
    output = tmp_path / "tiny.pdb.gz"
    with pdb_path.open("rb") as source, gzip.open(output, "wb") as target:
        target.write(source.read())
    return output
