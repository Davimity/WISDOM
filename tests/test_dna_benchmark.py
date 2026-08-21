"""Focused scientific tests for DNA curation, sidecars, and metric semantics."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from lambdaforge.preprocessing import PreprocessingRecord
from lambdaforge.tasks import TaskContext

from wisdom.dna.DNAAnnotationSink import DNAAnnotationSink
from wisdom.dna.DNAAnnotationTransform import DNAAnnotationTransform
from wisdom.dna.DNACandidateCurator import DNACandidateCurator
from wisdom.dna.DNADatasetSource import DNADatasetSource
from wisdom.dna.DNALabel import DNALabel
from wisdom.dna.DNASelectionSink import DNASelectionSink
from wisdom.dna.EvidenceKind import EvidenceKind
from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite
from wisdom.evaluation.PointCloudExporter import PointCloudExporter


class _UnusedContext:
    """Provide a deliberate placeholder for transforms that do not resolve task paths."""


def _candidate(path: Path, evidence: list[str]) -> PreprocessingRecord:
    """Create one auditable tiny candidate record.

    Args:
        path: Protein-DNA fixture path.
        evidence: Closed evidence values.

    Returns:
        Candidate preprocessing record.
    """
    return PreprocessingRecord(
        key="TEST_1_A",
        value={
            "pdb_id": "TEST",
            "assembly_id": "1",
            "protein_chain": "A",
            "dna_chains": ["D"],
            "structure_path": str(path),
            "structure_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "evidence": evidence,
            "evidence_sources": ["fixture"],
            "sequence_cluster_id": "fixture-cluster-A",
        },
    )


def _catalog_row(
    identifier    : str,
    label         : int,
    structure_path: Path,
    cluster       : str,
    sequence_hash : str,
    structure_hash: str,
) -> dict[str, object]:
    """Create one complete curated row for sink invariant tests.

    Args:
        identifier: Stable ``PDB_CHAIN`` identity.
        label: Binary protein target.
        structure_path: Existing exact source assembly.
        cluster: Leakage-prevention sequence cluster.
        sequence_hash: Exact sequence digest surrogate.
        structure_hash: Exact selected-chain structure digest surrogate.

    Returns:
        JSON-compatible catalog row containing mandatory provenance.
    """
    positive = label == 1
    return {
        "candidate_key": f"{identifier}_1",
        "base_identifier": identifier,
        "label": label,
        "label_state": "positive" if positive else "negative",
        "label_reason": "fixture evidence",
        "included": True,
        "exclusion_reason": None,
        "evidence": [
            EvidenceKind.BIOLOGICAL_ASSEMBLY_CONTACT.value
            if positive
            else EvidenceKind.CURATED_NOT_DNA_BINDING.value
        ],
        "evidence_sources": ["fixture"],
        "source_database": "fixture",
        "source_record": identifier,
        "source_version": "fixture-v1",
        "source_url": "https://example.invalid/fixture",
        "source_checksum": "sha256:" + "0" * 64,
        "published_partition": "development",
        "query_version": "fixture-v1",
        "query_date_utc": "2026-08-13T00:00:00+00:00",
        "structure_path": str(structure_path),
        "structure_sha256": hashlib.sha256(structure_path.read_bytes()).hexdigest(),
        "protein_chain": "A",
        "dna_chains": ["D"] if positive else [],
        "sequence_cluster_id": cluster,
        "label_conflict_cluster_id": f"90:{cluster}",
        "sequence_sha256": sequence_hash,
        "protein_structure_sha256": structure_hash,
        "contact_pair_count": 2 if positive else 0,
        "positive_assertion": positive,
        "local_gt_expected": True,
        "local_gt_method": "dna_distance" if positive else "global_negative",
        "no_positive_uniprot_annotation": not positive,
        "no_known_pdb_dna_complex": not positive,
        "no_biolip_dna_binding": not positive,
        "interface_residue_count": 1 if positive else 0,
        "interface_residue_fraction": 0.1 if positive else 0.0,
        "interface_region_count": 1 if positive else 0,
        "largest_interface_region": 1 if positive else 0,
        "sequence_length": 10,
        "total_chain_count": 2 if positive else 1,
        "aspect_ratio": 1.5,
        "resolution_angstrom": 2.0,
        "tier": "core",
    }


def test_positive_requires_contact_and_absence_is_not_negative() -> None:
    fixture = Path(__file__).parent / "data" / "dna_complex.pdb"
    curator = DNACandidateCurator(minimum_residues=1, minimum_interface_residues=1)

    positive = curator.transform(
        _candidate(fixture, [EvidenceKind.BIOLOGICAL_ASSEMBLY_CONTACT]),
        _UnusedContext(),  # type: ignore[arg-type]
    )
    unknown = curator.transform(
        _candidate(fixture, [EvidenceKind.ABSENCE_OF_DNA]),
        _UnusedContext(),  # type: ignore[arg-type]
    )

    assert positive.value["label_state"] == DNALabel.POSITIVE
    assert positive.value["contact_pair_count"] > 0
    assert unknown.value["label_state"] == DNALabel.UNKNOWN


def test_dyprol_source_preserves_chain_case_and_preflights_duplicate_keys(
    tmp_path   : Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep case-sensitive mmCIF chains distinct and reject only ambiguous repeated keys."""
    train = tmp_path / "train.txt"
    test  = tmp_path / "test.txt"
    train.write_text(">7S01_D\nACDE\n1000\n>7S01_d\nFGHI\n0100\n", encoding="utf-8")
    test.write_text(">6V7B_a\nKLMN\n0010\n", encoding="utf-8")

    source = DNADatasetSource(mode="fixture")
    train_digest = hashlib.sha256(train.read_bytes()).hexdigest()
    test_digest  = hashlib.sha256(test.read_bytes()).hexdigest()
    monkeypatch.setattr(source, "DYPROL_TRAIN_SHA256", train_digest)
    monkeypatch.setattr(source, "DYPROL_TEST_SHA256", test_digest)
    monkeypatch.setattr(
        source.client,
        "zip_member",
        lambda url, size, member, path: train if member == source.DYPROL_TRAIN_MEMBER else test,
    )

    candidates = source._dyprol_candidates(tmp_path)
    records    = source._unique_records(candidates)
    keys       = {record.key for record in records}

    assert "DyProL:v1-2026-04-13:development:7S01_D" in keys
    assert "DyProL:v1-2026-04-13:development:7S01_d" in keys
    assert {value["protein_chain"] for value in candidates} == {"D", "d", "a"}
    assert len(source._unique_records((candidates[0], dict(candidates[0])))) == 1

    conflicting = dict(candidates[0], sequence="AAAA")
    with pytest.raises(RuntimeError, match="conflicting candidate key"):
        source._unique_records((candidates[0], conflicting))


def test_rcsb_metadata_uses_current_sequence_group_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    curator = DNACandidateCurator()
    payload = {
        "data": {
            "polymer_entity": {
                "rcsb_polymer_entity_group_membership": [
                    {
                        "aggregation_method": "sequence_identity",
                        "group_id": "group-30",
                        "similarity_cutoff": 30.0,
                    },
                    {
                        "aggregation_method": "sequence_identity",
                        "group_id": "group-90",
                        "similarity_cutoff": 90.0,
                    },
                ],
                    "entity_poly": {
                        "pdbx_seq_one_letter_code_can": "ACDE",
                        "rcsb_sample_sequence_length": 42,
                    },
                    "rcsb_polymer_entity_container_identifiers": {
                        "auth_asym_ids": ["A"],
                        "reference_sequence_identifiers": [
                            {"database_name": "UniProt", "database_accession": "P00001"}
                        ],
                    },
                "rcsb_polymer_entity": {"pdbx_description": "fixture protein"},
                "rcsb_entity_source_organism": [
                    {"ncbi_scientific_name": "Synthetic construct", "ncbi_taxonomy_id": 32630}
                ],
            },
            "entry": {
                "exptl": [{"method": "X-RAY DIFFRACTION"}],
                    "rcsb_entry_info": {
                        "resolution_combined": [2.1],
                        "polymer_entity_count_DNA": 0,
                    },
                "rcsb_accession_info": {"initial_release_date": "2026-01-01T00:00:00Z"},
            },
        }
    }
    monkeypatch.setattr(curator.client, "json", lambda *args: payload)

    metadata = curator._metadata("TEST", "1")

    assert metadata["sequence_cluster_id"] == "rcsb-mmseqs2-30:group-30"
    assert metadata["label_conflict_cluster_id"] == "rcsb-mmseqs2-90:group-90"
    assert metadata["reported_sequence_length"] == 42
    assert metadata["resolution_angstrom"] == 2.1


def test_explicit_negative_contact_conflict_is_quarantined() -> None:
    fixture = Path(__file__).parent / "data" / "dna_complex.pdb"
    curator = DNACandidateCurator(minimum_residues=1, minimum_interface_residues=1)
    result  = curator.transform(
        _candidate(fixture, [EvidenceKind.CURATED_NOT_DNA_BINDING]),
        _UnusedContext(),  # type: ignore[arg-type]
    )

    assert result.value["label_state"] == DNALabel.CONFLICT
    assert result.value["included"] is False


def test_negative_mapping_is_deterministic_and_rejects_public_conflicts(
    tmp_path : Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select the best exact experimental mapping and fail closed on DNA evidence."""
    structure = Path(__file__).parent / "data" / "tiny.pdb"
    biolip    = tmp_path / "biolip.txt"
    biolip.write_text("", encoding="utf-8")
    curator = DNACandidateCurator(minimum_residues=1)
    candidate = {
        "source_class": "negative",
        "sequence": "ACDE",
        "biolip_dna_uniprot_path": str(biolip),
    }

    def metadata(pdb_id: str, entity_id: str) -> dict[str, object]:
        """Return two equivalent mappings whose resolution determines selection."""
        return {
            "pdb_id": pdb_id,
            "entity_id": entity_id,
            "canonical_sequence": "ACDE",
            "entry_dna_polymer_count": 0,
            "uniprot_ids": ["P00001"],
            "protein_chains": ["A"],
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution_angstrom": 1.5 if pdb_id == "AAAA" else 2.5,
            "sequence_cluster_id": f"30:{pdb_id}",
            "label_conflict_cluster_id": f"90:{pdb_id}",
        }

    monkeypatch.setattr(curator, "_sequence_entities", lambda sequence: ("BBBB_1", "AAAA_1"))
    monkeypatch.setattr(curator, "_metadata", metadata)
    monkeypatch.setattr(curator, "_quickgo", lambda accession: (False, False))
    monkeypatch.setattr(curator, "_structure", lambda pdb_id, context: structure)
    monkeypatch.setattr(
        curator,
        "_protein_atoms",
        lambda chain: ([(0.0, 0.0, 0.0)] * 4, [0, 1, 2, 3], "ACDE"),
    )

    selected = curator._map_public_candidate(candidate, _UnusedContext())  # type: ignore[arg-type]

    assert selected["pdb_id"] == "AAAA"
    assert selected["negative_confidence"] == "high"
    assert selected["no_known_pdb_dna_complex"] is True

    def dna_metadata(pdb_id: str, entity_id: str) -> dict[str, object]:
        """Return the same exact sequence with an observed DNA polymer conflict."""
        value = metadata(pdb_id, entity_id)
        value["entry_dna_polymer_count"] = 1
        return value

    monkeypatch.setattr(curator, "_metadata", dna_metadata)
    with pytest.raises(RuntimeError, match="known PDB DNA complex"):
        curator._map_public_candidate(candidate, _UnusedContext())  # type: ignore[arg-type]

    monkeypatch.setattr(curator, "_metadata", metadata)
    monkeypatch.setattr(curator, "_quickgo", lambda accession: (True, False))
    with pytest.raises(RuntimeError, match="QuickGO"):
        curator._map_public_candidate(candidate, _UnusedContext())  # type: ignore[arg-type]


def test_missing_mapped_chain_is_rejected_without_inventing_an_example(
    tmp_path : Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an exact sequence hit whose declared polymer chain is absent from coordinates."""
    structure = Path(__file__).parent / "data" / "tiny.pdb"
    biolip    = tmp_path / "biolip.txt"
    biolip.write_text("", encoding="utf-8")
    curator = DNACandidateCurator(minimum_residues=1)
    monkeypatch.setattr(curator, "_sequence_entities", lambda sequence: ("AAAA_1",))
    monkeypatch.setattr(
        curator,
        "_metadata",
        lambda pdb_id, entity_id: {
            "pdb_id": pdb_id,
            "entity_id": entity_id,
            "canonical_sequence": "ACDE",
            "entry_dna_polymer_count": 0,
            "uniprot_ids": ["P00001"],
            "protein_chains": ["Z"],
            "experimental_method": "X-RAY DIFFRACTION",
            "resolution_angstrom": 1.5,
        },
    )
    monkeypatch.setattr(curator, "_quickgo", lambda accession: (False, False))
    monkeypatch.setattr(curator, "_structure", lambda pdb_id, context: structure)
    candidate = {
        "source_class": "negative",
        "sequence": "ACDE",
        "biolip_dna_uniprot_path": str(biolip),
    }

    with pytest.raises(RuntimeError, match="quality and coverage"):
        curator._map_public_candidate(candidate, _UnusedContext())  # type: ignore[arg-type]


def test_annotation_arrays_align_and_base_bytes_are_immutable(tmp_path: Path) -> None:
    fixture = Path(__file__).parent / "data" / "dna_complex.pdb"
    base     = tmp_path / "test_A.npz"
    metadata = {"coordinate_origin": [0.0, 0.0, 0.0]}
    np.savez_compressed(
        base,
        atom_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        residue_indices=np.asarray([0], dtype=np.int32),
        surface_positions=np.asarray([[3.2, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32),
        surface_area_weights=np.asarray([1.0, 1.0], dtype=np.float32),
        surface_atom_edge_index=np.asarray([[0, 1], [0, 0]], dtype=np.int32),
        surface_edge_index=np.asarray([[0], [1]], dtype=np.int32),
        metadata_json=np.asarray(__import__("json").dumps(metadata)),
    )
    before = hashlib.sha256(base.read_bytes()).hexdigest()
    source_hash = hashlib.sha256(fixture.read_bytes()).hexdigest()
    record = PreprocessingRecord(
        key="TEST_A",
        value={
            "base_npz": str(base),
            "label": 1,
            "structure_path": str(fixture),
            "structure_sha256": source_hash,
            "dna_chains": ["D"],
            "binding_residue_indices": [],
            "local_gt_expected": True,
            "local_gt_method": "dna_distance",
            "split": "test",
            "tier": "core",
        },
    )
    transformed = DNAAnnotationTransform().transform(
        record,
        _UnusedContext(),  # type: ignore[arg-type]
    )
    arrays = transformed.value["arrays"]

    assert arrays["surface_target_hard"].shape == (2,)
    assert arrays["surface_valid_mask"].shape == (2,)
    assert hashlib.sha256(base.read_bytes()).hexdigest() == before
    DNAAnnotationSink._validate(arrays, transformed.value["metadata"])

    context = TaskContext(
        name="annotation-fixture",
        run_dir=tmp_path / "annotation-run",
        source_dir=tmp_path,
        attempt_id="annotation-attempt",
        config_fingerprint="annotation-fingerprint",
        resume=True,
        outputs={"annotations": "annotations", "annotation-report": "report.json"},
    )
    sink = DNAAnnotationSink(report_output="annotation-report")
    sink.write(transformed, context)
    sink.finalize(context)
    manifest = (context.run_dir / "annotations" / "manifest.csv").read_text()
    index    = (context.run_dir / "annotations" / "members.jsonl").read_text()

    assert "base/" in manifest
    assert '"dna_binding":1' in index
    assert '"local_ground_truth":true' in index
    assert str(tmp_path) not in manifest
    assert hashlib.sha256(base.read_bytes()).hexdigest() == before


def test_metric_suite_preserves_undefined_and_point_cloud_channels(tmp_path: Path) -> None:
    metric = BinaryMetricSuite().compute(torch.tensor([0.1, 0.2]), torch.tensor([0, 0]))
    assert metric["accuracy"] == 1.0
    assert metric["auroc"] is None
    assert metric["recall"] is None

    probabilities = torch.tensor([0.9, 0.1, 0.2])
    targets       = torch.tensor([1, 1, 0])
    valid         = torch.tensor([True, False, True])
    masked = BinaryMetricSuite().compute(probabilities[valid], targets[valid])
    assert masked["accuracy"] == 1.0
    assert masked["precision"] == 1.0
    assert masked["recall"] == 1.0

    positions = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=np.float32)
    ply, companion = PointCloudExporter().export(
        tmp_path / "surface.ply",
        positions,
        {
            "surface_probability": np.asarray([0.1, 0.9]),
            "surface_embeddings": np.asarray([[1.0, 2.0], [3.0, 4.0]]),
        },
        latent_channels=(1,),
    )
    assert "latent_chemical_channel_1" in ply.read_text(encoding="ascii")
    with np.load(companion, allow_pickle=False) as archive:
        assert archive["positions"].shape == (2, 3)


def test_conflicts_duplicates_and_cluster_leakage_are_rejected() -> None:
    fixture = Path(__file__).parent / "data" / "dna_complex.pdb"
    positive = _catalog_row("AAAA_A", 1, fixture, "cluster-a", "seq-a", "structure-a")
    negative = _catalog_row("BBBB_A", 0, fixture, "cluster-b", "seq-b", "structure-b")
    negative["label_conflict_cluster_id"] = positive["label_conflict_cluster_id"]

    accepted, conflicts, _ = DNASelectionSink._resolve_identifiers([positive, negative])

    assert accepted == []
    assert len(conflicts) == 2
    assert all(row["label_state"] == DNALabel.CONFLICT for row in conflicts)
    assert all(row["positive_evidence"] and row["negative_evidence"] for row in conflicts)

    first  = _catalog_row("CCCC_A", 1, fixture, "cluster-c", "seq-c", "same-structure")
    second = _catalog_row("DDDD_A", 1, fixture, "cluster-d", "seq-d", "same-structure")
    accepted, _, exclusions = DNASelectionSink._resolve_identifiers([first, second])
    assert len(accepted) == 1
    assert exclusions[0]["exclusion_reason"] == "duplicate exact protein-chain coordinates"

    leaked = [
        _catalog_row("EEEE_A", 1, fixture, "leaked", "seq-e", "structure-e"),
        _catalog_row("FFFF_A", 1, fixture, "leaked", "seq-f", "structure-f"),
    ]
    leaked[0]["split"] = "train"
    leaked[1]["split"] = "test"
    with pytest.raises(RuntimeError, match="cluster_id leakage"):
        DNASelectionSink._validate_splits(leaked)


def test_dataset_sink_resumes_and_atomically_publishes_catalog(tmp_path: Path) -> None:
    positive_path = Path(__file__).parent / "data" / "dna_complex.pdb"
    negative_path = Path(__file__).parent / "data" / "tiny.pdb"
    context = TaskContext(
        name="dna-fixture",
        run_dir=tmp_path,
        source_dir=tmp_path,
        attempt_id="fixture-attempt",
        config_fingerprint="fixture-fingerprint",
        resume=True,
        outputs={
            "dataset": "dataset",
            "dataset-report": "report",
            "selection-checkpoints": "selection-checkpoints",
        },
    )
    sink = DNASelectionSink(reserve_fraction=0.0)
    rows = (
        _catalog_row("AAAA_A", 1, positive_path, "cluster-a", "seq-a", "structure-a"),
        _catalog_row("BBBB_A", 0, negative_path, "cluster-b", "seq-b", "structure-b"),
        _catalog_row("CCCC_A", 1, positive_path, "cluster-c", "seq-c", "structure-c"),
        _catalog_row("DDDD_A", 0, negative_path, "cluster-d", "seq-d", "structure-d"),
        _catalog_row("EEEE_A", 1, positive_path, "cluster-e", "seq-e", "structure-e"),
        _catalog_row("FFFF_A", 0, negative_path, "cluster-f", "seq-f", "structure-f"),
    )
    rows[4]["published_partition"] = "external_test"
    rows[5]["published_partition"] = "external_test"
    for row in rows:
        key = str(row["candidate_key"])
        sink.write(PreprocessingRecord(key=key, value=row), context)
        assert sink.is_complete(key, context)

    sink.finalize(context)

    assert (tmp_path / "dataset" / "catalog.csv").is_file()
    assert (tmp_path / "dataset" / "catalog.parquet").is_file()
    assert (tmp_path / "dataset" / "splits.csv").is_file()
    assert (tmp_path / "dataset" / "identifiers.json").is_file()
    assert (tmp_path / "dataset" / "validation.txt").is_file()
    assert (tmp_path / "report" / "distributions.png").is_file()
    assert json.loads((tmp_path / "report" / "summary.json").read_text())["verdict"] == "PASS"
    assert not tuple(tmp_path.rglob("*.tmp"))


def test_main_splits_are_exactly_balanced_by_deterministic_downsampling() -> None:
    """Retain every scarce class member and a stable equal-size majority subset."""
    fixture = Path(__file__).parent / "data" / "dna_complex.pdb"
    rows: list[dict[str, object]] = []
    for split in DNASelectionSink.MAIN_SPLITS:
        for index, label in enumerate((0, 1, 1, 1)):
            identifier = f"{split[:2].upper()}{index:02d}_{'A' if label else 'B'}"
            row = _catalog_row(
                identifier,
                label,
                fixture,
                f"{split}-cluster-{index}",
                f"{split}-sequence-{index}",
                f"{split}-structure-{index}",
            )
            row["split"] = split
            rows.append(row)

    sink                 = DNASelectionSink(reserve_fraction=0.0)
    retained, exclusions = sink._balance_main_splits(rows)  # type: ignore[arg-type]

    sink._validate_class_balance(retained)
    assert len(exclusions) == 6
    for split in DNASelectionSink.MAIN_SPLITS:
        split_rows = [row for row in retained if row["split"] == split]
        assert [row["label"] for row in split_rows].count(0) == 1
        assert [row["label"] for row in split_rows].count(1) == 1


def test_selection_dilutions_are_balanced_nested_and_cluster_diverse(tmp_path: Path) -> None:
    """Reduce every split by balanced prefixes that visit distinct clusters first."""
    fixture = Path(__file__).parent / "data" / "dna_complex.pdb"
    rows: list[dict[str, object]] = []
    for split in DNASelectionSink.MAIN_SPLITS:
        for label in (0, 1):
            for index in range(20):
                identifier = f"{split[:2].upper()}{label}{index:02d}_A"
                row = _catalog_row(
                    identifier,
                    label,
                    fixture,
                    f"{split}-{label}-cluster-{index // 2}",
                    f"{split}-{label}-sequence-{index}",
                    f"{split}-{label}-structure-{index}",
                )
                row["split"] = split
                rows.append(row)

    sink    = DNASelectionSink(dilutions=(0.10, 0.25, 0.50), reserve_fraction=0.0)
    summary = sink._write_dilutions(rows, tmp_path)  # type: ignore[arg-type]
    selected_ids: dict[str, set[str]] = {}
    for name, per_class in (("10pct", 2), ("25pct", 5), ("50pct", 10)):
        payload = json.loads((tmp_path / "subsets" / name / "identifiers.json").read_text())
        selected_ids[name] = {str(value["identifier"]) for value in payload["records"]}
        assert summary[name]["member_count"] == per_class * 2 * 3
        for split in DNASelectionSink.MAIN_SPLITS:
            counts = summary[name]["split_class_balance"][split]
            assert counts["positive"] == per_class
            assert counts["negative"] == per_class
            assert counts["sequence_clusters"] >= per_class
    assert selected_ids["10pct"] < selected_ids["25pct"] < selected_ids["50pct"]


def test_external_test_clusters_never_enter_development_and_assignment_is_deterministic() -> None:
    """Preserve official test provenance while keeping every 30%-identity cluster indivisible."""
    fixture = Path(__file__).parent / "data" / "dna_complex.pdb"
    rows = [
        _catalog_row("AAAA_A", 1, fixture, "development-positive", "seq-a", "structure-a"),
        _catalog_row("BBBB_A", 0, fixture, "development-negative", "seq-b", "structure-b"),
        _catalog_row("CCCC_A", 1, fixture, "external-positive", "seq-c", "structure-c"),
        _catalog_row("DDDD_A", 0, fixture, "external-negative", "seq-d", "structure-d"),
    ]
    rows[2]["published_partition"] = "external_test"
    rows[3]["published_partition"] = "external_test"
    sink = DNASelectionSink(reserve_fraction=0.0)

    first, _  = sink._assign_clusters(rows)
    second, _ = sink._assign_clusters(list(reversed(rows)))

    assert first == second
    assert first["external-positive"] == "test"
    assert first["external-negative"] == "test"
    assert first["development-positive"] in {"train", "val"}
    assert first["development-negative"] in {"train", "val"}


def test_zero_positive_projection_is_not_an_all_negative_local_target(tmp_path: Path) -> None:
    """Keep a reliable global positive while disabling an unusable local target."""
    base = tmp_path / "unmatched.npz"
    np.savez_compressed(
        base,
        atom_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        residue_indices=np.asarray([0], dtype=np.int32),
        surface_positions=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        surface_area_weights=np.asarray([1.0], dtype=np.float32),
        surface_atom_edge_index=np.asarray([[0], [0]], dtype=np.int32),
        surface_edge_index=np.empty((2, 0), dtype=np.int32),
        metadata_json=np.asarray(json.dumps({"coordinate_origin": [0.0, 0.0, 0.0]})),
    )
    record = PreprocessingRecord(
        key="TEST_A",
        value={
            "base_npz": str(base),
            "label": 1,
            "structure_sha256": "unused-for-residue-projection",
            "binding_residue_indices": [1],
            "local_gt_expected": True,
            "local_gt_method": "binding_residue_mask",
            "split": "test",
            "tier": "core",
        },
    )

    result = DNAAnnotationTransform().transform(record, _UnusedContext())  # type: ignore[arg-type]
    arrays = result.value["arrays"]

    assert result.value["metadata"]["protein_label"] == 1
    assert result.value["metadata"]["local_gt_available"] is False
    assert result.value["metadata"]["local_gt_reason"] == "zero_positive_surface_points"
    assert not arrays["surface_valid_mask"].any()


def test_training_positive_may_explicitly_lack_local_ground_truth(tmp_path: Path) -> None:
    """Separate weak global supervision from optional localization evaluation."""
    base = tmp_path / "global-only.npz"
    np.savez_compressed(
        base,
        atom_positions=np.asarray([[0.0, 0.0, 0.0]], dtype=np.float32),
        residue_indices=np.asarray([0], dtype=np.int32),
        surface_positions=np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        surface_area_weights=np.asarray([1.0], dtype=np.float32),
        surface_atom_edge_index=np.asarray([[0], [0]], dtype=np.int32),
        surface_edge_index=np.empty((2, 0), dtype=np.int32),
        metadata_json=np.asarray(json.dumps({"coordinate_origin": [0.0, 0.0, 0.0]})),
    )
    record = PreprocessingRecord(
        key="TEST_A",
        value={
            "base_npz": str(base),
            "label": 1,
            "structure_sha256": "unused-for-global-only",
            "binding_residue_indices": [],
            "local_gt_expected": False,
            "local_gt_method": "none",
            "split": "train",
            "tier": "core",
        },
    )

    result = DNAAnnotationTransform().transform(record, _UnusedContext())  # type: ignore[arg-type]

    assert result.value["metadata"]["protein_label"] == 1
    assert result.value["metadata"]["local_gt_available"] is False
    assert result.value["metadata"]["local_gt_reason"] == (
        "source_has_no_reliable_local_ground_truth"
    )
