"""Offline Zn acquisition ownership, frozen-selection and workflow contracts."""

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest
import yaml

from wisdom.preprocessing.zinc import discovery
from wisdom.preprocessing.zinc.ZincDiscovery import ZincDiscovery
from wisdom.preprocessing.zinc.ZincSelection import ZincSelection


def test_skipped_selection_forwards_design_without_touching_scientific_services(tmp_path):
    """An explicit skip declares only the exact supplied design; no tools/cache/map run."""
    work = SimpleNamespace(outputs=Mock(), log=Mock())
    result = ZincSelection.run(work, skip=True, existing_design=tmp_path)
    assert result == {"skipped": True, "design": str(tmp_path)}
    work.outputs.artifact.assert_called_once_with("zinc-design", tmp_path, role="dataset-design")


def test_skipped_selection_requires_explicit_design():
    """Skipping must not silently reconstruct missing evidence or rerun scientific selection."""
    with pytest.raises(ValueError, match="existing_design"):
        ZincSelection.run(SimpleNamespace(), skip=True)


def test_skipped_discovery_forwards_snapshot_without_acquisition(tmp_path):
    """A frozen RAW directory is a native dependency, not an instruction to crawl again."""
    work = SimpleNamespace(outputs=Mock(), log=Mock())
    ZincDiscovery.run(work, skip=True, existing_raw=tmp_path)
    assert work.outputs.artifact.call_count == 2


def test_shared_discovery_owns_native_map_query_and_raw_evidence(tmp_path, monkeypatch):
    """Freeze bytes; scientific identity ignores human labels and execution settings."""
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"result_set": [{"identifier": "2XYZ"}, {"identifier": "1ABC"}]}))
    negative = tmp_path / "negative.jsonl"
    negative.write_text("reviewed source bytes")
    root = tmp_path / "raw"
    root.mkdir()
    digest = hashlib.sha256(query.read_bytes()).hexdigest()
    negatives = [{"identifier": "3NEG_A", "label": 0, "structure_sha256": digest}]
    monkeypatch.setattr(discovery, "load_evidence", lambda path: negatives)
    monkeypatch.setattr(
        discovery,
        "candidates_for_entry",
        lambda work, pdb, **parameters: [
            {
                "identifier": pdb.upper() + "_A",
                "label": 1,
                "structure_sha256": digest,
                "assembly_alternatives": [{"accepted_sites": 1}],
            }
        ],
    )
    calls = []

    def resume_map(items, function, **options):
        """Exercise the configured callable while retaining native-map options for inspection."""
        calls.append((items, options))
        return [function(item) for item in items]

    def checkpoint(key, build):
        """Use real receipt builders without any public acquisition."""
        path = tmp_path / "checkpoints" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        build(path)
        return path

    work = SimpleNamespace(
        cache=Mock(),
        outputs=Mock(),
        log=Mock(),
        resume_map=resume_map,
        checkpoints=SimpleNamespace(file=checkpoint),
    )
    work.cache.fetch.return_value = query
    work.outputs.directory.return_value = root
    result = discovery.discover_candidates(
        work, negative, "release-test", maximum_entries=1, acquire_functional_metadata=False
    )
    assert result["raw_path"] == root / "raw.jsonl"
    assert result["positive_candidates"] == result["explicit_negatives"] == 1
    assert calls[0][0] == ["1abc"]
    assert calls[0][1]["name"] == "zinc-discovery"
    assert calls[0][1]["executor"] == "thread"
    assert calls[0][1]["workers"] == 8
    rows = [json.loads(line) for line in result["raw_path"].read_text().splitlines()]
    assert [row["label"] for row in rows] == [1, 0]
    assert json.loads((root / "provenance.json").read_text())["pilot"] is True
    assert work.cache.fetch.call_args_list[0].kwargs["key"] == (
        "zinc-discovery/release-test/rcsb-query.json"
    )
    request_url = work.cache.fetch.call_args_list[0].args[0]
    request = json.loads(parse_qs(urlsplit(request_url).query)["json"][0])
    assert request["query"]["parameters"] == {
        "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
        "operator": "exact_match",
        "value": "ZN",
    }
    assert request["request_options"] == {"return_all_hits": True}
    assert request["return_type"] == "entry"
    assert (root / "structures/index.json").is_file()
    first = json.loads((root / "provenance.json").read_text())
    root = tmp_path / "second-raw"
    root.mkdir()
    work.outputs.directory.return_value = root
    discovery.discover_candidates(
        work,
        negative,
        "another-session",
        workers=2,
        maximum_entries=1,
        acquire_functional_metadata=False,
        snapshot_label="another-label",
        started_at="later",
    )
    second = json.loads((root / "provenance.json").read_text())
    assert first["snapshot_id"] == second["snapshot_id"]
    receipts = list((root / "sources").rglob("*.source.json"))
    assert receipts and all(json.loads(path.read_text())["first_observed_at"] for path in receipts)

    # A later public response is a distinct observation even with identical scientific policy.
    query.write_text(
        json.dumps(
            {
                "result_set": [{"identifier": "2XYZ"}, {"identifier": "1ABC"}],
                "source_revision": "changed",
            }
        )
    )
    digest = hashlib.sha256(query.read_bytes()).hexdigest()
    negatives[0]["structure_sha256"] = digest
    root = tmp_path / "changed-raw"
    root.mkdir()
    work.outputs.directory.return_value = root
    discovery.discover_candidates(
        work, negative, "changed-session", maximum_entries=1, acquire_functional_metadata=False
    )
    assert json.loads((root / "provenance.json").read_text())["snapshot_id"] != first["snapshot_id"]


def test_standalone_discovery_delegates_to_same_stage(monkeypatch, tmp_path):
    """The optional acquisition-only Work keeps its historical return contract and outputs."""
    stage = Mock(
        return_value={"raw_path": tmp_path, "positive_candidates": 2, "explicit_negatives": 3}
    )
    monkeypatch.setattr("wisdom.preprocessing.zinc.ZincDiscovery.discover_candidates", stage)
    session = tmp_path / "session.json"
    session.write_text(json.dumps({"session_id": "test", "started_at": "frozen-time"}))
    work = SimpleNamespace(checkpoints=Mock(), _start_session=ZincDiscovery._start_session)
    work.checkpoints.file.return_value = session
    result = ZincDiscovery.run(work, negative_evidence=tmp_path, snapshot_label="release-test")
    assert result == {"positive_candidates": 2, "explicit_negatives": 3}
    assert stage.call_args.args[0] is work


def test_complete_zinc_yaml_has_five_native_steps_and_no_preexisting_dataset_lookup():
    """A fresh build acquires explicit negatives without a preexisting file or dataset."""
    path = Path(__file__).parents[1] / "experiments/preprocess/zinc/zinc_preprocess.yaml"
    steps = yaml.safe_load(path.read_text())["steps"]
    assert [step["name"] for step in steps] == [
        "discover",
        "select",
        "preprocess",
        "validate",
        "visualize",
    ]
    assert steps[0]["with"]["skip"] is False
    assert "negative_evidence" not in steps[0]["with"]
    assert steps[0]["with"]["negative_evidence_go_terms"] == ["GO:0008270", "GO:0046872"]
    assert "raw_path" not in steps[0]["with"]
    assert steps[1]["with"]["raw_path"] == {"from": "discover.raw"}
    assert steps[1]["with"]["positive_policy"] == "all"
    assert steps[2]["with"]["design"] == {"from": "select.zinc-design"}
    assert steps[3]["with"]["dataset"] == {"from": "preprocess.dataset"}
    assert steps[4]["with"]["dataset"] == {"from": "preprocess.dataset"}
    assert steps[4]["with"]["skip"] is False
