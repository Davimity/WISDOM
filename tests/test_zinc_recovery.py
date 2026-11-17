"""Durable Zn observations and early scientific feasibility without public acquisition."""

import gzip
import json
import pickle
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
from lambdaforge.work import WorkCache
from lambdaforge.work.checkpoints import CheckpointCollection

from wisdom.preprocessing.zinc.stages import discovery
from wisdom.preprocessing.zinc.stages.negatives import fetch_json
from wisdom.preprocessing.zinc.stages.sources import freeze_source
from wisdom.preprocessing.zinc.ZincSelection import ZincSelection


def source_work(tmp_path):
    """Use actual LF byte integrity and checkpoint services with no real remote requests."""
    return SimpleNamespace(cache=WorkCache(tmp_path / "cache"),
        checkpoints=CheckpointCollection(tmp_path / "checkpoints"), log=Mock())


def test_public_inventory_survives_cache_change_without_querying_a_later_revision(tmp_path):
    """A new code-scoped cache cannot replace the Run's first public observation."""
    work = source_work(tmp_path)
    payload = b'{"result_set": [{"identifier": "1ABC"}], "total_count": 1}'
    url = "https://example.org/query"
    with patch("urllib.request.urlopen", return_value=BytesIO(payload)) as request:
        assert fetch_json(work, url, "session/query.json", tmp_path, 1000)["total_count"] == 1
        request.assert_called_once()
    work.cache = WorkCache(tmp_path / "new-code-cache")
    with patch("urllib.request.urlopen", side_effect=AssertionError("requeried frozen inventory")):
        assert fetch_json(work, url, "session/query.json", tmp_path, 1000)["total_count"] == 1
    receipt = tmp_path / "sources/session/query.json.source.json"
    assert json.loads(receipt.read_text())["url"] == url


def test_concurrent_readers_share_one_durable_observation(tmp_path):
    """Native checkpoint locking publishes one immutable source for every metadata reader."""
    work = source_work(tmp_path)
    work.cache.fetch = Mock(return_value=tmp_path / "response.json")
    work.cache.fetch.return_value.write_bytes(b'{"results": []}')
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(lambda _: fetch_json(work, "https://example.org/a", "a",
            tmp_path, 1000, persist_source=False), range(16)))
    assert results == [{"results": []}] * 16
    assert work.cache.fetch.call_count == 1
    assert not (tmp_path / "sources").exists()


def test_lost_source_cannot_adopt_new_public_bytes_and_url_changes_are_rejected(tmp_path):
    """An observed digest is evidence, not permission to rebuild from a later revision."""
    work = source_work(tmp_path)
    url = "https://example.org/evidence"
    with patch("urllib.request.urlopen", return_value=BytesIO(b'{"revision": 1}')):
        source = freeze_source(work, url, "session/a.json", 1000)
    Path(source).unlink()  # Only the isolated temporary fixture is removed, not project data.
    work.cache = WorkCache(tmp_path / "empty-cache")
    with (patch("urllib.request.urlopen", return_value=BytesIO(b'{"revision": 2}')),
          pytest.raises(RuntimeError, match="byte-for-byte")):
        freeze_source(work, url, "session/a.json", 1000)
    assert not Path(source).exists()
    with pytest.raises(RuntimeError, match="changed request"):
        freeze_source(work, url + "?different=1", "session/a.json", 1000)


def test_coordinate_checkpoint_is_spawn_safe_and_inspection_key_binds_bytes(tmp_path):
    """A native checkpoint reference survives pickle and restores without its download cache."""
    work = source_work(tmp_path)
    with patch("urllib.request.urlopen", return_value=BytesIO(gzip.compress(b"exact cif bytes"))):
        entry = discovery.download_coordinate("1abc", work, 1000, "session")
    restored = pickle.loads(pickle.dumps(entry))
    assert restored["path"].scope == "checkpoint"
    assert Path(restored["path"]).read_bytes() == b"exact cif bytes"
    work.cache = WorkCache(tmp_path / "new-cache")
    with patch("urllib.request.urlopen", side_effect=AssertionError("download repeated")):
        assert discovery.download_coordinate("1abc", work, 1000, "session") == entry


def test_repeated_negative_structures_stop_before_positive_scan(tmp_path, monkeypatch):
    """Several chains of one biological sequence cannot fill independent split roles."""
    source = tmp_path / "evidence.jsonl"
    source.write_text("Synthetic evidence loaded below, never a published benchmark.")
    rows = [{"identifier": f"{index}NEG_A", "label": 0, "sequence": "ACDE"}
            for index in range(4)]
    monkeypatch.setattr(discovery, "load_evidence", lambda _: rows)
    work = source_work(tmp_path)
    work.outputs = Mock()
    work.outputs.directory.return_value = tmp_path
    with (patch("urllib.request.urlopen", side_effect=AssertionError("bulk scan started")),
          pytest.raises(RuntimeError, match="only 1 distinct negative sequences")):
        discovery.discover_candidates(work, source, "session")
    audit = json.loads((tmp_path / "negative-feasibility.json").read_text())
    assert audit["negative_chains"] == 4
    assert not audit["necessary_sequence_support"]
    assert not audit["independent_groups_verified"]


def test_legacy_retry_refuses_to_mix_saved_inspections_with_a_new_inventory(tmp_path, monkeypatch):
    """Missing legacy observation bytes need explicit LF recovery, not a new same-session query."""
    source = tmp_path / "negative.jsonl"
    source.write_text("Reviewed fixture provided by patched loader.")
    monkeypatch.setattr(discovery, "load_evidence", lambda _: [
        {"label": 0, "sequence": sequence} for sequence in ("ACDE", "FGHI", "KLMN")])
    work = source_work(tmp_path)
    work.outputs = Mock()
    work.outputs.directory.return_value = tmp_path
    (tmp_path / "checkpoints/maps/zinc-coordinate-downloads").mkdir(parents=True)
    with (patch("urllib.request.urlopen", side_effect=AssertionError("fresh public query")),
          pytest.raises(RuntimeError, match="Legacy Zn Discovery")):
        discovery.discover_candidates(work, source, "session")


def test_offline_selection_checks_negative_sequence_support_before_geometry(tmp_path, monkeypatch):
    """An acquisition pilot cannot bypass benchmark feasibility by forwarding it to Selection."""
    module = __import__("wisdom.preprocessing.zinc.ZincSelection", fromlist=["load_evidence"])
    monkeypatch.setattr(module, "load_evidence", lambda _: [
        {"label": 0, "sequence": "ACDE"}] * 4)
    work = SimpleNamespace(tools=Mock())
    with pytest.raises(RuntimeError, match="three distinct explicit"):
        ZincSelection.run(work, raw_path=tmp_path / "raw.jsonl")
