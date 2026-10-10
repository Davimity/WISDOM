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


def pytest_collection_modifyitems(items: list[pytest.Item]) -> None:
    """Keep production-data, GPU and browser checks distinct from offline CPU contracts.

    Args:
        items: Collected tests; markers affect selection, not scientific failure handling.
    """
    production = {
        "test_dataset_splits_are_disjoint_and_cover_master_manifest",
        "test_training_catalog_resolves_before_every_dry_run",
    }
    for item in items:
        if item.originalname in production:
            item.add_marker(pytest.mark.slow)
        if item.originalname == "test_cuda_forward_backward_when_available":
            item.add_marker(pytest.mark.gpu)
        if "browser" in item.name or "chromium" in item.name:
            item.add_marker(pytest.mark.browser)
