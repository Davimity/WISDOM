"""Deterministic dataset-level curation without premature partition assignment."""

from __future__ import annotations

import csv
import json
import os
from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace


class DNACurationSink:
    """Collapse source candidates into one defensible record per logical protein.

    This sink deliberately does not create train, validation, or test partitions. It resolves
    exact duplicates and contradictory labels, chooses one experimental structure using a stable
    quality order, and publishes the complete candidate population needed by geometry. Sequence
    and structure similarity are computed only after those structures have been preprocessed.
    """

    def __init__(
        self,
        dataset_output: str = "curation",
        report_output : str = "curation-report",
    ) -> None:
        """Bind portable curation and report output directories.

        Args:
            dataset_output: Named directory receiving ``curated-catalog.csv`` and the protein list.
            report_output: Named directory receiving exclusions, conflicts, and summary evidence.

        Raises:
            ValueError: If any logical output name is empty.
        """
        if any(not value.strip() for value in (dataset_output, report_output)):
            raise ValueError("curation output names cannot be empty")
        self.dataset_output = dataset_output
        self.report_output  = report_output
        self.records: dict[str, dict[str, Any]] = {}

    def finalize(self, context: ProcessingWorkspace) -> tuple[Path, Path]:
        """Resolve logical proteins and publish a split-free canonical catalog.

        Args:
            context: Task context resolving curation and report roots.

        Returns:
            LambdaForge declarations for the portable curation and its audit report.

        Raises:
            RuntimeError: If no positive or no negative logical protein survives.
            OSError: If canonical tables cannot be published atomically.
        """
        curation_root = context.output(self.dataset_output)
        report_root = context.output(self.report_output)
        curation_root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)

        accepted, conflicts, exclusions = self.resolve_logical_proteins(
            [self.records[key] for key in sorted(self.records)]
        )
        counts = {label: sum(int(row["label"]) == label for row in accepted) for label in (0, 1)}
        if not all(counts.values()):
            raise RuntimeError(
                "curation must retain at least one positive and one negative protein"
            )

        # Discovery-cache paths are run-local evidence, not a portable data contract. Geometry
        # later downloads the exact PDB entry and annotation verifies its retained checksum.
        for row in accepted:
            row["structure_path"] = ""

        columns = self._columns(accepted)
        self._write_csv(curation_root / "curated-catalog.csv", accepted, columns)
        self._atomic_text(
            curation_root / "curated-proteins.txt",
            "".join(f"{row['base_identifier']}\n" for row in accepted),
        )
        self._write_csv(report_root / "conflicts.csv", conflicts, self._columns(conflicts))
        self._write_csv(report_root / "exclusions.csv", exclusions, self._columns(exclusions))

        summary = {
            "verdict": "PASS",
            "source_candidate_count": len(self.records),
            "logical_protein_count": len(accepted),
            "positive_count": counts[1],
            "negative_count": counts[0],
            "conflict_count": len(conflicts),
            "excluded_count": len(exclusions),
            "split_assignment": "deferred_until_after_geometry_and_annotation",
        }
        self._atomic_text(
            report_root / "summary.json",
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False),
        )
        return curation_root, report_root

    @staticmethod
    def resolve_logical_proteins(
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Choose one best structure per logical protein and quarantine contradictions.

        A logical protein is keyed first by a reviewed UniProt accession when available and then
        by exact canonical-sequence SHA-256. Competing structures are ranked by verified local
        evidence, sequence coverage, experimental resolution, experimental method, interface
        extent, and identifier. Alternatives are retained as compact provenance rather than model
        examples. Opposite labels for an identical sequence or logical accession are quarantined;
        merely similar sequences are retained for later leakage grouping.

        Args:
            rows: Accepted and rejected curator mappings from every public source record.

        Returns:
            Ordered accepted logical proteins, contradictory rows, and ordinary exclusions.
        """
        conflicts: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = [row for row in rows if row.get("included") is not True]
        eligible = [dict(row) for row in rows if row.get("included") is True]

        accepted: list[dict[str, Any]] = []
        for logical, alternatives in DNACurationSink._logical_groups(eligible):
            labels = {int(row["label"]) for row in alternatives}
            if len(labels) != 1:
                for row in alternatives:
                    row["included"] = False
                    row["exclusion_reason"] = "contradictory labels for one logical protein"
                conflicts.extend(alternatives)
                continue
            alternatives.sort(key=DNACurationSink._quality_rank)
            selected = alternatives[0]
            selected["logical_protein_id"] = logical
            selected["structure_alternatives_json"] = json.dumps(
                [
                    {
                        "identifier": str(row.get("base_identifier", "")),
                        "pdb_id": str(row.get("pdb_id", "")),
                        "resolution_angstrom": row.get("resolution_angstrom"),
                        "structure_sha256": str(row.get("structure_sha256", "")),
                    }
                    for row in alternatives
                ],
                sort_keys=True,
                separators=(",", ":"),
            )
            accepted.append(selected)
            for duplicate in alternatives[1:]:
                duplicate["included"] = False
                duplicate["exclusion_reason"] = "alternative structure for retained logical protein"
                exclusions.append(duplicate)

        # Exact coordinate duplicates cannot become separate examples even under different source
        # accessions. The same stable quality order selects one representative.
        by_structure: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in accepted:
            by_structure[str(row.get("protein_structure_sha256", row["structure_sha256"]))].append(
                row
            )
        deduplicated: list[dict[str, Any]] = []
        for alternatives in by_structure.values():
            alternatives.sort(key=DNACurationSink._quality_rank)
            deduplicated.append(alternatives[0])
            for duplicate in alternatives[1:]:
                duplicate["included"] = False
                duplicate["exclusion_reason"] = "duplicate exact protein-chain coordinates"
                exclusions.append(duplicate)
        deduplicated.sort(key=lambda row: str(row["base_identifier"]))
        return deduplicated, conflicts, exclusions

    @staticmethod
    def _logical_groups(
        rows: list[dict[str, Any]],
    ) -> list[tuple[str, list[dict[str, Any]]]]:
        """Merge candidates sharing any mapped UniProt accession or exact sequence.

        Logical identity is transitive: if one candidate shares an accession with a second and the
        second shares an exact sequence with a third, all three are alternative observations of one
        logical protein. This prevents separate source records, PDB choices, or source ensembles
        from becoming independent examples. The representative name uses the smallest available
        accession, otherwise the sequence SHA-256.

        Args:
            rows: Included curator rows with accession lists and exact sequence hashes.

        Returns:
            Deterministically named connected logical-protein groups.
        """
        parent = list(range(len(rows)))

        # Union-find closes identity transitively without constructing a dense candidate matrix.
        owners: dict[str, int] = {}
        for index, row in enumerate(rows):
            accessions = row.get("uniprot_ids", [])
            if isinstance(accessions, str):
                try:
                    accessions = json.loads(accessions)
                except json.JSONDecodeError:
                    accessions = [accessions]
            tokens = {
                f"uniprot:{str(value).strip()}" for value in accessions if str(value).strip()
            }
            tokens.add(f"sequence:{str(row.get('sequence_sha256', '')).strip()}")
            for token in sorted(value for value in tokens if not value.endswith(":")):
                if token not in owners:
                    owners[token] = index
                    continue
                first, second = index, owners[token]
                while parent[first] != first:
                    parent[first] = parent[parent[first]]
                    first = parent[first]
                while parent[second] != second:
                    parent[second] = parent[parent[second]]
                    second = parent[second]
                if first != second:
                    parent[max(first, second)] = min(first, second)

        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for index, row in enumerate(rows):
            while parent[index] != index:
                parent[index] = parent[parent[index]]
                index = parent[index]
            groups[index].append(row)

        result: list[tuple[str, list[dict[str, Any]]]] = []
        for alternatives in groups.values():
            group_accessions: set[str] = set()
            for row in alternatives:
                values = row.get("uniprot_ids", [])
                if isinstance(values, str):
                    try:
                        values = json.loads(values)
                    except json.JSONDecodeError:
                        values = [values]
                group_accessions.update(
                    str(value).strip() for value in values if str(value).strip()
                )
            logical = (
                f"uniprot:{min(group_accessions)}"
                if group_accessions
                else f"sequence:{min(str(row['sequence_sha256']) for row in alternatives)}"
            )
            result.append((logical, alternatives))
        return sorted(result, key=lambda value: value[0])

    @staticmethod
    def _quality_rank(row: Mapping[str, Any]) -> tuple[Any, ...]:
        """Return the deterministic best-structure order used during deduplication.

        Args:
            row: One accepted curator mapping.

        Returns:
            Sort key preferring local evidence, coverage, resolution, method, interface size, and
            finally the stable protein identifier.
        """
        method = str(row.get("experimental_method", "")).upper()
        method_rank = (
            0 if "X-RAY" in method else 1 if "ELECTRON" in method else 2 if "NMR" in method else 3
        )
        resolution = row.get("resolution_angstrom")
        resolution_value = (
            float(resolution) if resolution not in (None, "", "None") else float("inf")
        )
        return (
            not bool(row.get("local_gt_expected", False)),
            -float(row.get("sequence_coverage", 0.0)),
            resolution_value,
            method_rank,
            -int(row.get("interface_residue_count", 0)),
            str(row.get("base_identifier", "")),
        )

    @staticmethod
    def _columns(rows: list[dict[str, Any]]) -> tuple[str, ...]:
        """Return a stable union of table columns.

        Args:
            rows: Mappings that will be serialized as CSV rows.

        Returns:
            Alphabetically ordered column names with the identifier and label first.
        """
        names = sorted({str(name) for row in rows for name in row})
        leading = [
            name for name in ("base_identifier", "logical_protein_id", "label") if name in names
        ]
        return tuple(leading + [name for name in names if name not in leading])

    @staticmethod
    def _write_csv(path: Path, rows: list[dict[str, Any]], columns: tuple[str, ...]) -> None:
        """Atomically serialize JSON-compatible mappings as a portable CSV table.

        Args:
            path: Destination table path.
            rows: Ordered row mappings.
            columns: Stable complete column order.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            name: json.dumps(value, sort_keys=True, separators=(",", ":"))
                            if isinstance(value, (list, dict, tuple))
                            else value
                            for name, value in row.items()
                        }
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        """Write UTF-8 text by fsync followed by atomic replacement.

        Args:
            path: Final path below a task-owned output directory.
            content: Complete text payload; a final newline is added when absent.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="\n") as stream:
                stream.write(content.rstrip("\n") + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
