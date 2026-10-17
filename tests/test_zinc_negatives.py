"""Offline scientific safeguards for autonomous Zn negative acquisition."""

import hashlib
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch
from urllib.error import HTTPError

import pytest
from lambdaforge.work import WorkCache

from wisdom.preprocessing.zinc import discovery, negatives
from wisdom.preprocessing.zinc.evidence import load_evidence


def denial(**overrides):
    """Return a synthetic referenced experimental denial, never published as benchmark data."""
    return {
        "id": "annotation-1",
        "geneProductId": "UniProtKB:P00001",
        "goId": "GO:0008270",
        "qualifier": "NOT|enables",
        "goEvidence": "IDA",
        "reference": "PMID:123",
        "extensions": None,
        **overrides,
    }


def test_negative_acquisition_filters_inference_context_isoforms_and_child_terms(
    tmp_path,
    monkeypatch,
):
    """Only exact experimental whole-product denials can enter the structural mapping."""
    records = [
        denial(),
        denial(goId="GO:0046872"),
        denial(goEvidence="ISS"),
        denial(extensions=[{"relation": "in_taxon"}]),
        denial(geneProductId="UniProtKB:P00001-2"),
        denial(goId="GO:child"),
        denial(qualifier="enables"),
        denial(reference=""),
    ]
    monkeypatch.setattr(
        negatives, "fetch_json", lambda *args: {"results": [{"ancestors": ["GO:0046872"]}]}
    )
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
    output = negatives.discover_negatives(
        work, tmp_path, "fixture", 3, 2, ["GO:0008270", "GO:0046872"]
    )
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
        negatives.discover_negatives(SimpleNamespace(), tmp_path, "fixture", 1, 2, ["GO:0005506"])


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
    first.write_text(
        json.dumps({"numberOfHits": 2, "pageInfo": {"total": 2}, "results": [denial()]})
    )
    second.write_text(
        json.dumps(
            {"numberOfHits": 2, "pageInfo": {"total": 2}, "results": [denial(id="annotation-2")]}
        )
    )

    def checkpoint(key, build):
        """Materialize a receipt through the same build callback as a native checkpoint."""
        path = tmp_path / "checkpoints" / key
        path.parent.mkdir(parents=True, exist_ok=True)
        if not path.exists():
            build(path)
        return path

    work = SimpleNamespace(cache=Mock(), log=Mock(), checkpoints=SimpleNamespace(file=checkpoint))
    work.cache.fetch.side_effect = [first, second]
    rows = negatives.annotations(
        work, {"goId": "GO:0008270", "goUsage": "exact"}, "fixture", tmp_path, 2
    )
    assert len(rows) == 2
    assert len(list((tmp_path / "sources").rglob("*.source.json"))) == 2
    assert work.cache.fetch.call_args.kwargs["rate_limit"] is work.cache.rate_limit.return_value
    options = work.cache.fetch.call_args.kwargs
    assert options["retries"] == 5
    assert options["retry_backoff"] == 2.0
    assert options["validate"] is negatives.valid_json_response
    second.write_text(
        json.dumps(
            {"numberOfHits": 3, "pageInfo": {"total": 2}, "results": [denial(id="annotation-2")]}
        )
    )
    work.cache.fetch.side_effect = [first, second]
    with pytest.raises(RuntimeError, match="changed during pagination"):
        negatives.annotations(work, {}, "fixture-other", tmp_path, 2)


def test_rejected_json_query_reports_the_server_reason_without_partial_output(tmp_path):
    """Native permanent HTTP failures retain the offending search attribute in the Work log."""
    url = "https://search.rcsb.org/rcsbsearch/v2/query?json=invalid"
    cause = HTTPError(url, 400, "Bad Request", {}, BytesIO(
        b'{"message":"search is not enabled on [wrong.attribute] attribute"}'
    ))
    work = SimpleNamespace(cache=WorkCache(tmp_path / "cache"), log=Mock())

    with patch("urllib.request.urlopen", side_effect=cause) as download:
        with pytest.raises(RuntimeError, match=r"wrong\.attribute.*correct its fields") as caught:
            negatives.fetch_json(work, url, "fixture/query.json", tmp_path, 2)
        download.assert_called_once()

    assert url in str(caught.value)
    assert "fixture/query.json" in str(caught.value)
    assert caught.value.__cause__.__cause__ is cause
    assert not (tmp_path / "sources").exists()
    assert work.log.call_args.kwargs["level"] == "error"


@pytest.mark.parametrize("content,expected", [
    (b'{"results": []}', True),
    (b"<html>Service unavailable</html>", False),
    (b'{"results":', False),
    (b"[]", False),
    (b"\xff\xff", False),
])
def test_native_json_validation_rejects_error_pages_and_truncated_content(
    tmp_path, content, expected
):
    """The response format guard is shared by new acquisition and native cache restoration."""
    response = tmp_path / "response.json"
    response.write_bytes(content)
    assert negatives.valid_json_response(response) is expected


def test_public_json_uses_native_transient_retries_and_restores_without_network(tmp_path):
    """A temporary server failure is retried by LF and the successful response is reused."""
    url = "https://example.org/evidence.json"
    work = SimpleNamespace(cache=WorkCache(tmp_path / "cache"), log=Mock(),
                           checkpoints=Mock())
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    work.checkpoints.file.return_value = receipt
    responses = [HTTPError(url, 503, "Unavailable", {}, BytesIO(b"try later")),
                 BytesIO(b'{"results": []}')]
    with patch("urllib.request.urlopen", side_effect=responses) as download:
        assert negatives.fetch_json(work, url, "fixture.json", tmp_path, 1000) == {
            "results": []
        }
        assert download.call_count == 2
        assert negatives.fetch_json(work, url, "fixture.json", tmp_path, 1000) == {
            "results": []
        }
        assert download.call_count == 2


def test_invalid_json_is_not_published_to_native_cache_and_can_be_refetched(tmp_path):
    """An HTTP-200 HTML error page cannot become a reusable scientific source."""
    url = "https://example.org/evidence.json"
    work = SimpleNamespace(cache=WorkCache(tmp_path / "cache"), log=Mock(),
                           checkpoints=Mock())
    receipt = tmp_path / "receipt.json"
    receipt.write_text("{}")
    work.checkpoints.file.return_value = receipt
    with patch("urllib.request.urlopen", side_effect=[
        BytesIO(b"<html>proxy error</html>"), BytesIO(b'{"results": []}')
    ]) as download:
        with pytest.raises(RuntimeError, match="Semantic validation rejected"):
            negatives.fetch_json(work, url, "fixture.json", tmp_path, 1000)
        assert not (tmp_path / "sources").exists()
        assert negatives.fetch_json(work, url, "fixture.json", tmp_path, 1000) == {
            "results": []
        }
        assert download.call_count == 2


def test_subject_mapping_requires_complete_exact_sequence_and_keeps_chain_names(
    tmp_path,
    monkeypatch,
):
    """A matching AQ chain is one chain; mutated/tagged chains do not inherit evidence."""
    monkeypatch.setattr(negatives, "annotations", lambda *args: [])
    entry = {
        "sequence": {"value": "ACDE"},
        "uniProtKBCrossReferences": [{"database": "PDB", "id": "1ABC"}],
    }
    monkeypatch.setattr(negatives, "fetch_json", lambda *args: entry)
    exact = SimpleNamespace(name="AQ", sequence="ACDE")
    tagged = SimpleNamespace(name="B", sequence="MACDE")
    mutated = SimpleNamespace(name="C", sequence="ACDF")
    model = MagicMock()
    model.__getitem__.return_value = [exact, tagged, mutated]
    model.assemblies = [SimpleNamespace(name="1")]
    assembly = SimpleNamespace(protein_chains=lambda: [exact, exact, tagged, mutated])
    structure = SimpleNamespace(
        structure=model,
        sequence=lambda chain: chain.sequence,
        assembly=lambda identifier: assembly,
        sha256=lambda: "coordinate-hash",
    )
    monkeypatch.setattr(negatives, "ProteinStructure", lambda path: structure)
    work = SimpleNamespace(cache=Mock(), log=Mock())
    work.cache.fetch.return_value = tmp_path / "fixture.cif"
    result = negatives.map_subject(
        {"accession": "P00001", "annotations": [denial()]}, work, tmp_path, "fixture", 2
    )
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
    result = negatives.map_subject(
        {"accession": "P00001", "annotations": [denial()]}, work, tmp_path, "fixture", 2
    )
    assert result["rows"] == []
    assert result["exclusions"][0]["reason"] == "contradictory_experimental_Zn_annotation"
    work.cache.fetch.assert_not_called()


@pytest.mark.parametrize("term", ["GO:0008270", "GO:0046872", "GO:0043169", "GO:0043167"])
def test_curated_parent_denial_is_accepted_but_unrelated_denial_is_not(tmp_path, term):
    """NOT of a reviewed parent entails Zn denial; NOT of a sibling metal does not."""
    row = {
        "identifier": "1ABC_A",
        "sequence": "ACDE",
        "label": 0,
        "protein_copy": 1,
        "assembly_id": "1",
        "origin": "fixture",
        "label_evidence": {
            "kind": "curated_not_annotation",
            "term": term,
            "qualifier": "NOT",
            "evidence_code": "IDA",
            "reference": "PMID:123",
            "scope": "zinc_binding",
            "sequence_sha256": hashlib.sha256(b"ACDE").hexdigest(),
        },
    }
    source = tmp_path / "evidence.jsonl"
    source.write_text(json.dumps(row))
    assert load_evidence(source)[0]["label"] == 0
    row["label_evidence"]["term"] = "GO:0005506"
    source.write_text(json.dumps(row))
    with pytest.raises(ValueError, match="explicit sequence-bound"):
        load_evidence(source)
