"""Offline safeguards for observed-universe coverage, confidence and acquisition bias."""

import hashlib
import json
from types import SimpleNamespace

import pytest
from test_surface_features_and_zinc import _assembly

from wisdom.preprocessing.common import population
from wisdom.preprocessing.common.shortcut import shortcut_baseline
from wisdom.preprocessing.zinc import metadata
from wisdom.preprocessing.zinc.audit import write_benchmark_report
from wisdom.preprocessing.zinc.evidence import load_evidence
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination


def rows():
    """Return auditable synthetic members, not real biological or experimental evidence."""
    sites = ZincCoordination().analyse(_assembly(), "AQ", 1)
    return [
        {
            "identifier": f"F{index}_AQ",
            "label": index % 2,
            "quality_eligible": True,
            "quality_exclusion_reason": "",
            "sequence_sha256": f"sequence-{index}",
            "leakage_group": f"group-{index}",
            "global_phenotype": "ZG_NOISE",
            "local_phenotype": "ZL_NOISE",
            "origin": "positive_source" if index % 2 else "negative_source",
            "split": ("train", "validation", "test")[index // 2],
            "sites": sites if index % 2 else [],
            "zn_site_count": int(index % 2),
            "pfam": ["PF1"] if index in (1, 3) else [],
            "sequence_length": 10 + index % 2,
            "radius_of_gyration_normalized": 2,
            "aspect_ratio": 1,
            "packing_density": 0.5,
            "negative_evidence_tier": "direct_Zn_NOT" if not index % 2 else "not_applicable",
        }
        for index in range(6)
    ]


def test_keep_all_positives_avoids_quadratic_ranking(monkeypatch):
    """Keeping the whole eligible universe is not a quota selection problem."""

    def forbidden(*args):
        """Fail if an unnecessary greedy ranking is attempted."""
        raise AssertionError("all-positive retention must not run quota ranking")

    monkeypatch.setattr(population, "_diverse", forbidden)
    selected, audit = population.select_population(
        rows(), 1, True, False, 2026, keep_all_positives=True
    )
    assert len(selected) == 6
    assert audit["omitted"] == []


def test_diversity_quota_counts_individual_domains_and_is_deterministic():
    """A shared domain in different lists is one category; missing is not diversity."""
    candidates = [
        {**rows()[1], "identifier": name, "pfam": domains}
        for name, domains in (("A", ["PF1"]), ("B", ["PF1", "PF2"]), ("C", []))
    ]
    first = population._diverse(candidates, 2, 2026, False, additional_strata=("pfam",))
    second = population._diverse(candidates, 2, 2026, False, additional_strata=("pfam",))
    assert first == second
    assert {row["identifier"] for row in first} == {"A", "B"}


def test_funnels_reconcile_exclusions_and_missing_family_is_not_a_category(tmp_path):
    """Only equal-unit stages yield retention; multiple annotation domains overlap honestly."""
    raw = rows()
    rejected = {
        **raw[-1],
        "identifier": "REJECT_A",
        "quality_eligible": False,
        "quality_exclusion_reason": "resolution;secondary_reason",
    }
    raw.append(rejected)
    audit = {
        "discovery": {"snapshot_id": "synthetic", "pilot": True},
        "phenotypes": {},
        "splits": {
            name: {"positive": 1, "negative": 1, "leakage_groups": 2}
            for name in ("train", "validation", "test")
        },
    }
    write_benchmark_report(
        tmp_path,
        raw,
        raw[:-1],
        audit,
        {
            "rcsb_entries_containing_Zn": 20,
            "inspected_entries": 10,
            "assembly_copies": 8,
            "candidate_chains": 4,
        },
        {},
    )
    summary = audit["benchmark"]
    exclusions = summary["exclusions"]["positive"]
    assert exclusions["raw"] == exclusions["eligible"] + exclusions["excluded"]
    assert exclusions["primary_reasons"] == {"resolution": 1}
    assert sum(exclusions["all_reasons_nonadditive"].values()) == 2
    positive = summary["funnels"]["positive"]
    assert positive[1]["retained_percent"] == 50
    assert positive[2]["retained_percent"] is None  # entries != copies
    families = summary["distributions"]["pfam"]["selected"]["1"]
    assert families["categories"] == {"PF1": 2}
    assert families["missing"] == 1
    assert summary["family_independence"]["pfam"]["PF1"]["train_and_heldout"]
    assert summary["family_independence"]["pfam"]["PF1"]["groups_by_split"] == {
        "train": 1,
        "validation": 1,
        "test": 0,
    }
    assert summary["split_distributions"]["pfam"]["test"]["1"]["missing"] == 1
    assert (tmp_path / "coverage.csv").is_file()
    assert "not biological specificity" in (tmp_path / "report.md").read_text() or (
        "does not prove physiological" in (tmp_path / "report.md").read_text()
    )


def test_structural_shortcut_excludes_source_and_never_opens_test():
    """An acquisition-perfect source field cannot enter the structural control classifier."""
    records = rows()
    records += [
        {**row, "identifier": row["identifier"] + "_copy"}
        for row in records
        if row["split"] != "test"
    ]
    baseline = shortcut_baseline(records, 2026, structural_only=True)
    assert baseline["available"]
    assert not {"origin", "method", "resolution", "release_year"} & set(baseline["features"])
    assert baseline == shortcut_baseline(
        [row for row in records if row["split"] != "test"], 2026, structural_only=True
    )


def test_metadata_respects_author_chain_and_exact_uniprot_scope(monkeypatch, tmp_path):
    """A partial UniProt reference describes a link but cannot transfer enzyme annotations."""
    entity = {
        "rcsb_polymer_entity_container_identifiers": {
            "auth_asym_ids": ["AQ"],
            "reference_sequence_identifiers": [
                {"database_name": "UniProt", "database_accession": "P1"}
            ],
        },
        "rcsb_polymer_entity_annotation": [{"type": "Pfam", "annotation_id": "PF1"}],
        "rcsb_entity_source_organism": [{"ncbi_taxonomy_id": 9606}],
        "polymer_entity_instances": [
            {"rcsb_polymer_instance_annotation": [{"type": "CATH", "annotation_id": "3.40"}]}
        ],
    }
    canonical = {
        "primaryAccession": "P1",
        "sequence": {"value": "ACDE"},
        "proteinDescription": {"recommendedName": {"ecNumbers": [{"value": "1.2.3.4"}]}},
        "uniProtKBCrossReferences": [{"database": "InterPro", "id": "IP1"}],
    }

    def response(work, url, *args):
        """Simulate source responses, not biological label acquisition."""
        return (
            {"data": {"entry": {"polymer_entities": [entity]}}}
            if "graphql" in url
            else {"results": [canonical]}
        )

    monkeypatch.setattr(metadata, "fetch_json", response)
    records = [
        {"identifier": "1ABC_AQ", "sequence": "ACDE", "label": 1},
        {"identifier": "1ABC_AQ", "sequence": "MACDE", "label": 0},
        {"identifier": "1ABC_B", "sequence": "ACDE", "label": 0},
    ]
    enriched = metadata.enrich_metadata(
        {"pdb_id": "1abc", "rows": records}, SimpleNamespace(), tmp_path, "fixture", 2
    )["rows"]
    assert [row["label"] for row in enriched] == [1, 0, 0]
    assert enriched[0]["exact_uniprot"] == ["P1"]
    assert enriched[0]["ec"] == ["1.2.3.4"]
    assert enriched[1]["exact_uniprot"] == enriched[1]["ec"] == []
    assert enriched[1]["pfam"] == ["PF1"]
    assert enriched[2]["pfam"] == []
    assert enriched[2]["functional_metadata_status"] == "unavailable"


def test_confidence_tier_does_not_change_acceptance():
    """A transparent stronger-support tier is not a new physiological label policy."""
    sites = ZincCoordination().analyse(_assembly(), "AQ", 1)
    assert sites[0]["accepted"] and sites[0]["confidence_tier"] == "structural_basic"
    assert not sites[1]["accepted"] and sites[1]["confidence_tier"] == "not_accepted"


@pytest.mark.parametrize("partial", [False, True])
def test_missing_metadata_is_allowed_but_partial_public_response_is_not(
    monkeypatch, tmp_path, partial
):
    """Absent annotations are missing support, not a new family or silently partial evidence."""
    payload = {"data": {"entry": None}}
    if partial:
        payload["errors"] = ["synthetic GraphQL error"]
    monkeypatch.setattr(metadata, "fetch_json", lambda *args: payload)
    job = {"pdb_id": "1abc", "rows": [{"identifier": "1ABC_AQ", "sequence": "ACDE"}]}
    if partial:
        with pytest.raises(RuntimeError, match="incomplete GraphQL"):
            metadata.enrich_metadata(job, SimpleNamespace(), tmp_path, "fixture", 2)
    else:
        record = metadata.enrich_metadata(job, SimpleNamespace(), tmp_path, "fixture", 2)["rows"][0]
        assert record["pfam"] == record["interpro"] == record["ec"] == []
        assert record["functional_metadata_status"] == "unavailable"


def test_unverified_positive_bridge_does_not_override_explicit_negative(tmp_path):
    """A co-deposited, non-coordinating chain hypothesis is not positive evidence."""
    digest = hashlib.sha256(b"ACDE").hexdigest()
    row = {
        "identifier": "1ABC_AQ",
        "sequence": "ACDE",
        "assembly_id": "1",
        "protein_copy": 1,
        "label": 1,
        "origin": "rcsb",
        "candidate_verified": False,
        "label_evidence": {"kind": "structural_coordination_candidate"},
    }
    negative = {
        **row,
        "label": 0,
        "origin": "fixture",
        "label_evidence": {
            "kind": "experimental_non_binding",
            "scope": "zinc_binding",
            "assay": "synthetic",
            "reference": "NOT-real-evidence",
            "sequence_sha256": digest,
        },
    }
    path = tmp_path / "raw.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(negative))
    result = load_evidence(path)
    assert len(result) == 1 and result[0]["label"] == 0
    assert not result[0].get("evidence_conflict")
