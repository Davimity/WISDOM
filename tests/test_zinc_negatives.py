"""Offline scientific safeguards for autonomous Zn negative acquisition."""

import hashlib
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock

import pytest

from wisdom.preprocessing.zinc import discovery, negatives
from wisdom.preprocessing.zinc.evidence import load_evidence


def denial(**overrides):
    """Return a synthetic referenced experimental denial, never published as benchmark data."""
    return {"id": "annotation-1", "geneProductId": "UniProtKB:P00001",
            "goId": "GO:0008270", "qualifier": "NOT|enables", "goEvidence": "IDA",
            "reference": "PMID:123", "extensions": None, **overrides}


def test_negative_acquisition_filters_inference_context_isoforms_and_child_terms(
    tmp_path, monkeypatch,
):
    """Only exact experimental whole-product denials can enter the structural mapping."""
    records = [denial(), denial(goId="GO:0046872"), denial(goEvidence="ISS"),
               denial(extensions=[{"relation": "in_taxon"}]),
               denial(geneProductId="UniProtKB:P00001-2"), denial(goId="GO:child"),
               denial(qualifier="enables"), denial(reference="")]
    monkeypatch.setattr(negatives, "fetch_json", lambda *args:
                        {"results": [{"ancestors": ["GO:0046872"]}]})
    queries = []

    def annotations(work, parameters, *args):
        """Expose exact API traversal parameters and simulated unexpected server records."""
        queries.append(parameters)
        return records if parameters["goId"] == "GO:0008270" else [records[1]]

    monkeypatch.setattr(negatives, "annotations", annotations)
    mapped = []

    def resume_map(items, function, **options):
        """Observe accepted subjects without acquiring public structures."""
        mapped.extend(items)
        assert options["executor"] == "thread"
        assert options["name"] == "zinc-negative-mapping"
        return [{"rows": [], "sources": [], "exclusions": []} for item in items]

    work = SimpleNamespace(log=Mock(), resume_map=resume_map)
    output = negatives.discover_negatives(work, tmp_path, "fixture", 3, 2,
                                          ["GO:0008270", "GO:0046872"])
    assert output.read_text() == ""
    assert len(mapped) == 1
    assert len(mapped[0]["annotations"]) == 2
    assert all(query["goUsage"] == "exact" for query in queries)
    assert all(query["qualifier"] == "NOT|enables" for query in queries)
    audit = json.loads((tmp_path / "negative-discovery.json").read_text())
    assert audit["experimental_subjects"] == 1
    assert audit["exclusion_counts"]["unsupported_subject_or_isoform"] == 1
    assert audit["exclusion_counts"]["not_an_exact_whole_product_denial"] == 3


def test_unrelated_negative_term_is_rejected_before_network(tmp_path):
    """A NOT iron-binding annotation cannot deny Zn binding."""
    with pytest.raises(ValueError, match="ancestors"):
        negatives.discover_negatives(SimpleNamespace(), tmp_path, "fixture", 1, 2,
                                      ["GO:0005506"])


def test_empty_negative_inventory_stops_before_positive_query(tmp_path, monkeypatch):
    """A costly positive scan is not started to mask the absence of reliable negatives."""
    source = tmp_path / "negative-evidence.jsonl"
    source.write_text("")
    monkeypatch.setattr(discovery, "discover_negatives", lambda *args: source)
    work = SimpleNamespace(outputs=Mock(), cache=Mock())
    work.outputs.directory.return_value = tmp_path
    with pytest.raises(RuntimeError, match="positive structure scan has not started"):
        discovery.discover_candidates(work, None, "fixture")
    work.cache.fetch.assert_not_called()


def test_quickgo_pagination_and_native_rate_limiter(tmp_path, monkeypatch):
    """All pages are retained and shared LF limiters, not floats, reach native fetch."""
    first, second = tmp_path / "page1.json", tmp_path / "page2.json"
    first.write_text(json.dumps({"numberOfHits": 2, "pageInfo": {"total": 2},
                                 "results": [denial()]}))
    second.write_text(json.dumps({"numberOfHits": 2, "pageInfo": {"total": 2},
                                  "results": [denial(id="annotation-2")]}))
    work = SimpleNamespace(cache=Mock(), log=Mock())
    work.cache.fetch.side_effect = [first, second]
    rows = negatives.annotations(work, {"goId": "GO:0008270", "goUsage": "exact"},
                                 "fixture", tmp_path, 2)
    assert len(rows) == 2
    assert len(list((tmp_path / "sources").rglob("*.json"))) == 2
    assert work.cache.fetch.call_args.kwargs["rate_limit"] is work.cache.rate_limit.return_value
    second.write_text(json.dumps({"numberOfHits": 3, "pageInfo": {"total": 2},
                                  "results": [denial(id="annotation-2")]}))
    work.cache.fetch.side_effect = [first, second]
    with pytest.raises(RuntimeError, match="changed during pagination"):
        negatives.annotations(work, {}, "fixture-other", tmp_path, 2)


def test_subject_mapping_requires_complete_exact_sequence_and_keeps_chain_names(
    tmp_path, monkeypatch,
):
    """A matching AQ chain is one chain; mutated/tagged chains do not inherit evidence."""
    monkeypatch.setattr(negatives, "annotations", lambda *args: [])
    entry = {"sequence": {"value": "ACDE"},
             "uniProtKBCrossReferences": [{"database": "PDB", "id": "1ABC"}]}
    monkeypatch.setattr(negatives, "fetch_json", lambda *args: entry)
    exact   = SimpleNamespace(name="AQ", sequence="ACDE")
    tagged  = SimpleNamespace(name="B", sequence="MACDE")
    mutated = SimpleNamespace(name="C", sequence="ACDF")
    model   = MagicMock()
    model.__getitem__.return_value = [exact, tagged, mutated]
    model.assemblies = [SimpleNamespace(name="1")]
    assembly = SimpleNamespace(protein_chains=lambda: [exact, exact, tagged, mutated])
    structure = SimpleNamespace(structure=model, sequence=lambda chain: chain.sequence,
                                assembly=lambda identifier: assembly,
                                sha256=lambda: "coordinate-hash")
    monkeypatch.setattr(negatives, "ProteinStructure", lambda path: structure)
    work = SimpleNamespace(cache=Mock(), log=Mock())
    work.cache.fetch.return_value = tmp_path / "fixture.cif"
    result = negatives.map_subject({"accession": "P00001", "annotations": [denial()]},
                                   work, tmp_path, "fixture", 2)
    assert [row["identifier"] for row in result["rows"]] == ["1ABC_AQ"]
    assert result["rows"][0]["protein_copy"] == 1
    source = tmp_path / "generated.jsonl"
    source.write_text(json.dumps(result["rows"][0]) + "\n")
    assert load_evidence(source)[0]["label"] == 0
    digest = hashlib.sha256(b"ACDE").hexdigest()
    assert result["rows"][0]["label_evidence"]["sequence_sha256"] == digest


def test_experimental_positive_conflict_quarantines_subject(tmp_path, monkeypatch):
    """Conflicting experimental Zn assertion and denial do not become a chosen label."""
    monkeypatch.setattr(negatives, "annotations", lambda *args: [denial(qualifier="enables")])
    work = SimpleNamespace(cache=Mock())
    result = negatives.map_subject({"accession": "P00001", "annotations": [denial()]},
                                   work, tmp_path, "fixture", 2)
    assert result["rows"] == []
    assert result["exclusions"][0]["reason"] == "contradictory_experimental_Zn_annotation"
    work.cache.fetch.assert_not_called()


@pytest.mark.parametrize("term", ["GO:0008270", "GO:0046872", "GO:0043169", "GO:0043167"])
def test_curated_parent_denial_is_accepted_but_unrelated_denial_is_not(tmp_path, term):
    """NOT of a reviewed parent entails Zn denial; NOT of a sibling metal does not."""
    row = {"identifier": "1ABC_A", "sequence": "ACDE", "label": 0,
           "protein_copy": 1, "assembly_id": "1", "origin": "fixture",
           "label_evidence": {"kind": "curated_not_annotation", "term": term,
                              "qualifier": "NOT", "evidence_code": "IDA",
                              "reference": "PMID:123", "scope": "zinc_binding",
                              "sequence_sha256": hashlib.sha256(b"ACDE").hexdigest()}}
    source = tmp_path / "evidence.jsonl"
    source.write_text(json.dumps(row))
    assert load_evidence(source)[0]["label"] == 0
    row["label_evidence"]["term"] = "GO:0005506"
    source.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="explicit sequence-bound"):
        load_evidence(source)
