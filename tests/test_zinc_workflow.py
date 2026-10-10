"""Offline Zn acquisition ownership, frozen-selection and workflow contracts."""

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

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


def test_shared_discovery_owns_native_map_query_and_raw_evidence(tmp_path, monkeypatch):
    """Selection and standalone discovery share one acquisition without a nested Work runner."""
    query = tmp_path / "query.json"
    query.write_text(json.dumps({"result_set": [{"identifier": "2XYZ"},
                                               {"identifier": "1ABC"}]}))
    negative = tmp_path / "negative.jsonl"
    negative.write_text("reviewed source bytes")
    root = tmp_path / "raw"
    root.mkdir()
    negatives = [{"identifier": "3NEG_A", "label": 0}]
    monkeypatch.setattr(discovery, "load_evidence", lambda path: negatives)
    monkeypatch.setattr(discovery, "candidates_for_entry",
                        lambda work, pdb, **parameters: [{"identifier": pdb, "label": 1}])
    calls = []

    def resume_map(items, function, **options):
        """Exercise the configured callable while retaining native-map options for inspection."""
        calls.append((items, options))
        return [function(item) for item in items]

    work = SimpleNamespace(cache=Mock(), outputs=Mock(), log=Mock(), resume_map=resume_map)
    work.cache.fetch.return_value = query
    work.outputs.directory.return_value = root
    result = discovery.discover_candidates(work, negative, "release-test", maximum_entries=1)
    assert result["raw_path"] == root / "raw.jsonl"
    assert result["positive_candidates"] == result["explicit_negatives"] == 1
    assert calls[0][0] == ["1abc"]
    assert calls[0][1]["name"] == "zinc-discovery"
    assert calls[0][1]["executor"] == "thread"
    assert calls[0][1]["workers"] == 8
    rows = [json.loads(line) for line in result["raw_path"].read_text().splitlines()]
    assert [row["label"] for row in rows] == [1, 0]
    assert json.loads((root / "provenance.json").read_text())["pilot"] is True
    assert work.cache.fetch.call_args.kwargs["key"] == "zinc-discovery/release-test/rcsb-query.json"


def test_standalone_discovery_delegates_to_same_stage(monkeypatch, tmp_path):
    """The optional acquisition-only Work keeps its historical return contract and outputs."""
    stage = Mock(return_value={"raw_path": tmp_path, "positive_candidates": 2,
                               "explicit_negatives": 3})
    monkeypatch.setattr("wisdom.preprocessing.zinc.ZincDiscovery.discover_candidates", stage)
    work = SimpleNamespace()
    result = ZincDiscovery.run(work, negative_evidence=tmp_path, release_id="release-test")
    assert result == {"positive_candidates": 2, "explicit_negatives": 3}
    assert stage.call_args.args[0] is work


def test_complete_zinc_yaml_has_three_native_steps_and_no_preexisting_dataset_lookup():
    """A fresh build acquires explicit negatives without a preexisting file or dataset."""
    path = Path(__file__).parents[1] / "experiments/preprocess/zinc/zinc_preprocess.yaml"
    steps = yaml.safe_load(path.read_text())["steps"]
    assert [step["name"] for step in steps] == ["select", "preprocess", "visualize"]
    assert steps[0]["with"]["skip"] is False
    assert "negative_evidence" not in steps[0]["with"]
    assert steps[0]["with"]["negative_go_terms"] == ["GO:0008270", "GO:0046872"]
    assert "raw_path" not in steps[0]["with"]
    assert steps[1]["with"]["design"] == {"from": "select.zinc-design"}
    assert steps[2]["with"]["dataset"] == {"from": "preprocess.dataset"}
    assert steps[2]["with"]["skip"] is False
