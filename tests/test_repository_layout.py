"""Maintain source navigation, portable report copies and the deliberate documentation boundary."""

import hashlib
import json
import re
import runpy
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def test_source_directories_remain_small_and_cohesive():
    """Do not rebuild a flat bag of unrelated classes after moving existing concepts."""
    for directory in (ROOT / "src/wisdom").rglob("*"):
        if directory.is_dir():
            modules = [p for p in directory.glob("*.py") if p.stem != "__init__"]
            assert len(modules) <= 12, (directory, len(modules))
    assert {p.stem for p in (ROOT / "src/wisdom/models").glob("*.py")} == {
        "WisdomV1", "WisdomV2", "WisdomV3", "__init__",
    }


def test_bundled_report_is_a_real_content_identified_pdf():
    """A checkout carries the actual report, not a machine-specific symbolic link or placeholder."""
    pdf = ROOT / "docs/WISDOM-memoria.pdf"
    metadata = json.loads(pdf.with_suffix(".json").read_text())
    content = pdf.read_bytes()
    assert not pdf.is_symlink() and content.startswith(b"%PDF-")
    assert metadata["bytes"] == len(content)
    assert metadata["sha256"] == hashlib.sha256(content).hexdigest()
    assert not list((ROOT / "docs").glob("*.md"))


def test_practical_documentation_has_no_broken_local_file_links():
    """Documentation consolidation must not leave readers following deleted audit notes."""
    for relative in ("README.md", "README.es.md", "experiments/README.md"):
        document = ROOT / relative
        for target in re.findall(r"\]\(([^)]+)\)", document.read_text()):
            if target.startswith(("http:", "https:", "#")):
                continue
            path = target.split("#", 1)[0]
            assert (document.parent / path).is_file(), (relative, target)


def test_report_sync_detects_content_changes_not_file_dates(tmp_path):
    """A copied report checks cleanly until either the author PDF or the copied metadata changes.

    Args:
        tmp_path: Isolated author/documentation directory, never the published report.
    """
    synchronize = runpy.run_path(str(ROOT / "scripts/sync_report.py"))["synchronize"]
    source = tmp_path / "author.pdf"
    target = tmp_path / "docs/WISDOM-memoria.pdf"
    source.write_bytes(b"%PDF-1.7\nfirst report\n")
    synchronize(source, target)
    original = target.read_bytes()
    assert synchronize(source, target, check=True)["bytes"] == len(original)
    source.touch()
    synchronize(source, target, check=True)
    source.write_bytes(b"%PDF-1.7\nsecond report\n")
    with pytest.raises(ValueError, match="stale"):
        synchronize(source, target, check=True)
    assert target.read_bytes() == original
    synchronize(source, target)
    target.with_suffix(".json").write_text("{}")
    with pytest.raises(ValueError, match="stale"):
        synchronize(source, target, check=True)


def test_report_sync_rejects_non_pdf_and_missing_copy(tmp_path):
    """Avoid replacing the report with a mistaken source or hiding a missing published copy.

    Args:
        tmp_path: Isolated invalid source and unused destination paths.
    """
    synchronize = runpy.run_path(str(ROOT / "scripts/sync_report.py"))["synchronize"]
    source = tmp_path / "source.pdf"
    target = tmp_path / "docs/report.pdf"
    source.write_text("not a PDF")
    with pytest.raises(ValueError, match="must be a PDF"):
        synchronize(source, target)
    assert not target.exists()
    source.write_bytes(b"%PDF-1.7\nreport\n")
    with pytest.raises(ValueError, match="missing"):
        synchronize(source, target, check=True)
