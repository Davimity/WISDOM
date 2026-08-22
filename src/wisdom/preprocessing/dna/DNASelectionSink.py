"""Atomic publication and hard validation of the curated DNA benchmark."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lambdaforge.preprocessing import PreprocessingRecord, PreprocessingSink
from lambdaforge.tasks import ArtifactDeclaration, ArtifactType, TaskContext

from wisdom.preprocessing.dna.DNALabel import DNALabel
from wisdom.preprocessing.dna.DNASelectionAudit import DNASelectionAudit
from wisdom.preprocessing.dna.EvidenceKind import EvidenceKind


class DNASelectionSink(PreprocessingSink):
    """Resolve evidence and publish a compact homology-safe protein selection."""

    MAIN_SPLITS = ("train", "val", "test")
    ALL_SPLITS  = (*MAIN_SPLITS, "validation_reserve", "test_reserve")

    def __init__(
        self,
        dataset_output      : str   = "dataset",
        report_output       : str   = "dataset-report",
        checkpoint_output   : str   = "selection-checkpoints",
        seed                : int   = 2026,
        validation_fraction : float = 0.20,
        reserve_fraction    : float = 0.10,
        dilutions           : Sequence[float] = (0.10, 0.25, 0.50, 0.75),
    ) -> None:
        """Configure deterministic development and reserve assignment.

        Published development records are divided by complete RCSB MMseqs2 30%-identity groups.
        The source's independent test partition remains test-only. A small deterministic fraction
        of positive development and external-test clusters is held outside the main partitions so
        later surface annotation can replace a locally unevaluable positive without moving it to
        training or consulting test outcomes.

        Args:
            dataset_output: Named directory receiving the portable benchmark.
            report_output: Named directory receiving exclusions and construction diagnostics.
            checkpoint_output: Named directory receiving per-candidate resume records. It remains
                outside the small portable selection artifact.
            seed: Integer used only for stable SHA-256 ordering within published partitions.
            validation_fraction: Fraction of development clusters assigned to validation.
            reserve_fraction: Fraction of positive clusters held as local-evaluation reserves.
            dilutions: Fractions of balanced training retained as nested, cluster-diverse views.
                Complete validation and test membership is fixed across every view.

        Raises:
            ValueError: If output names or fractions are invalid.
        """
        if not dataset_output.strip() or not report_output.strip() or not checkpoint_output.strip():
            raise ValueError("selection, report, and checkpoint output names cannot be empty")
        if not 0.0 < validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in (0,1)")
        if not 0.0 <= reserve_fraction < 0.5:
            raise ValueError("reserve_fraction must lie in [0,0.5)")
        fractions = tuple(sorted({float(value) for value in dilutions}))
        if any(not 0.0 < value < 1.0 for value in fractions):
            raise ValueError(
                "dilutions must contain unique fractions strictly between zero and one"
            )

        self.dataset_output      = dataset_output
        self.report_output       = report_output
        self.checkpoint_output   = checkpoint_output
        self.seed                = int(seed)
        self.validation_fraction = float(validation_fraction)
        self.reserve_fraction    = float(reserve_fraction)
        self.dilutions           = fractions
        self.records: dict[str, dict[str, Any]] = {}

    def write(self, record: PreprocessingRecord, context: TaskContext) -> None:
        """Checkpoint one curator result for deterministic final aggregation.

        Args:
            record: Curator output containing acceptance state, evidence, mapping, and clusters.
            context: LambdaForge task context locating the dataset staging directory.

        Raises:
            TypeError: If the curator output is not a mapping.
            ValueError: If a stable key is reused with different content.
        """
        if not isinstance(record.value, Mapping):
            raise TypeError("curated DNA row must be a mapping")
        value = dict(record.value)
        if record.key in self.records and self.records[record.key] != value:
            raise ValueError(f"conflicting duplicate candidate key: {record.key}")
        self.records[record.key] = value

        record_root = context.output(self.checkpoint_output)
        record_root.mkdir(parents=True, exist_ok=True)
        record_name = hashlib.sha256(record.key.encode()).hexdigest() + ".json"
        self._atomic_text(
            record_root / record_name,
            json.dumps({"key": record.key, "value": value}, indent=2, sort_keys=True) + "\n",
        )

    def is_complete(self, key: str, context: TaskContext) -> bool:
        """Revalidate a cached accepted or rejected curation result before resume.

        Rejected rows have no structure by design and are complete when their JSON checkpoint is
        readable. Accepted rows additionally require the exact cached structure checksum.

        Args:
            key: Stable public-source candidate key.
            context: LambdaForge task context locating record checkpoints.

        Returns:
            True when the checkpoint is internally consistent and any accepted structure is intact.
        """
        record_name = hashlib.sha256(key.encode()).hexdigest() + ".json"
        record_path = context.output(self.checkpoint_output) / record_name
        try:
            payload = json.loads(record_path.read_text(encoding="utf-8"))
            value   = payload.get("value") if isinstance(payload, dict) else None
            if not isinstance(payload, dict) or payload.get("key") != key:
                return False
            if not isinstance(value, dict):
                return False
            if value.get("included") is not True:
                self.records[key] = value
                return True

            structure_path = Path(str(value.get("structure_path", "")))
            expected_hash  = str(value.get("structure_sha256", ""))
            if not structure_path.is_file() or not expected_hash:
                return False
            if hashlib.sha256(structure_path.read_bytes()).hexdigest() != expected_hash:
                return False
            self.records[key] = value
            return True
        except (OSError, ValueError, TypeError, AttributeError, json.JSONDecodeError):
            return False

    def finalize(self, context: TaskContext) -> tuple[ArtifactDeclaration, ...]:
        """Resolve conflicts, assign immutable clusters, and publish the benchmark.

        Args:
            context: LambdaForge context resolving dataset and report outputs.

        Returns:
            Dataset, report, canonical table, and diagnostic figure declarations.

        Raises:
            RuntimeError: If no defensible rows survive or a provenance/leakage invariant fails.
            OSError: If publication cannot complete atomically.
        """
        dataset_root = context.output(self.dataset_output)
        report_root  = context.output(self.report_output)
        dataset_root.mkdir(parents=True, exist_ok=True)
        report_root.mkdir(parents=True, exist_ok=True)

        rows       = [self.records[key] for key in sorted(self.records)]
        accepted, conflicts, exclusions = self._resolve_identifiers(rows)
        if not accepted:
            raise RuntimeError("DNA curation produced no defensible positive or negative rows")
        self._validate_accepted(accepted)

        assignments, partition_exclusions = self._assign_clusters(accepted)
        exclusions.extend(partition_exclusions)
        accepted = [
            row
            for row in accepted
            if row.get("included") is True
            and str(row["sequence_cluster_id"]) in assignments
        ]
        for row in accepted:
            row["split"] = assignments[str(row["sequence_cluster_id"])]

        # Downsample only the majority class after homology-safe split assignment. This preserves
        # the external test boundary while making each train/validation/test partition exactly
        # balanced and leaves positive reserve pools available solely for local-GT replacement.
        accepted, balance_exclusions = self._balance_main_splits(accepted)
        exclusions.extend(balance_exclusions)
        self._validate_splits(accepted)
        self._validate_class_balance(accepted)
        # Keep only portable reconstruction data in the selection. The future relative path uses
        # the immutable source digest, while heavy bytes remain in the discovery cache until the
        # geometry/annotation recipe packages them under that exact name.
        for row in accepted:
            structure_hash = str(row["structure_sha256"])
            row["structure_path"] = f"structures/{structure_hash}.cif"

        columns = self._columns(accepted)
        self._write_csv(dataset_root / "catalog.csv", accepted, columns)
        self._write_csv(
            dataset_root / "labels.csv",
            accepted,
            ("base_identifier", "label", "split", "tier", "sequence_cluster_id"),
        )
        self._write_parquet(dataset_root / "catalog.parquet", accepted, columns)

        for split in self.ALL_SPLITS:
            identifiers = sorted(
                str(row["base_identifier"]) for row in accepted if row["split"] == split
            )
            split_text = "".join(f"{identifier}\n" for identifier in identifiers)
            self._atomic_text(dataset_root / f"{split}.txt", split_text)
        all_identifiers = sorted(str(row["base_identifier"]) for row in accepted)
        all_text        = "".join(f"{identifier}\n" for identifier in all_identifiers)
        self._atomic_text(dataset_root / "proteins.txt", all_text)

        # Keep one explicit machine-readable selection contract beside the convenient split TXT
        # files so another cluster can reproduce or inspect membership without querying sources.
        identifier_payload = {
            "schema_version": "1.1",
            "balance_policy": "exact_binary_balance_within_each_main_split",
            "leakage_unit": (
                "transitive_30pct_sequence_family_and_pdb_deposition_component"
            ),
            "split_seed": self.seed,
            "records": [
                {
                    "identifier": str(row["base_identifier"]),
                    "label": int(row["label"]),
                    "split": str(row["split"]),
                    "tier": str(row["tier"]),
                    "sequence_cluster_id": str(row["sequence_cluster_id"]),
                    "dataset_member": row["split"] in self.MAIN_SPLITS,
                }
                for row in sorted(accepted, key=lambda value: str(value["base_identifier"]))
            ],
        }
        self._atomic_text(
            dataset_root / "identifiers.json",
            json.dumps(identifier_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        dilution_summary = self._write_dilutions(accepted, dataset_root)
        quality_report   = DNASelectionAudit().audit(dataset_root)
        if quality_report["status"] == "FAIL":
            raise RuntimeError("published DNA selection failed its independent quality audit")

        self._write_csv(report_root / "conflicts.csv", conflicts, self._columns(conflicts))
        self._write_csv(report_root / "exclusions.csv", exclusions, self._columns(exclusions))
        summary = self._summary(rows, accepted, conflicts, exclusions, self.seed)
        summary["dilutions"] = dilution_summary
        self._atomic_text(
            report_root / "summary.json",
            json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )
        self._atomic_text(report_root / "summary.txt", self._verdict(summary))
        self._plot(accepted, report_root / "distributions.png")

        return (
            ArtifactDeclaration(
                path=dataset_root.relative_to(context.run_dir),
                kind=ArtifactType.DATASET,
            ),
            ArtifactDeclaration(
                path=report_root.relative_to(context.run_dir),
                kind=ArtifactType.REPORT,
            ),
            ArtifactDeclaration(
                path=(dataset_root / "catalog.csv").relative_to(context.run_dir),
                kind=ArtifactType.TABLE,
                media_type="text/csv",
            ),
            ArtifactDeclaration(
                path=(dataset_root / "proteins.txt").relative_to(context.run_dir),
                kind=ArtifactType.FILE,
                media_type="text/plain",
            ),
            ArtifactDeclaration(
                path=(report_root / "distributions.png").relative_to(context.run_dir),
                kind=ArtifactType.FIGURE,
                media_type="image/png",
            ),
        )

    @staticmethod
    def _resolve_identifiers(
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        """Resolve duplicates and quarantine label contradictions at 30% identity.

        Args:
            rows: Every accepted or rejected curator result.

        Returns:
            Accepted, conflicting, and ordinarily excluded rows.
        """
        conflicts : list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        eligible  = [row for row in rows if row.get("included") is True]
        exclusions.extend(row for row in rows if row.get("included") is not True)

        identifier_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in eligible:
            identifier_groups[str(row["base_identifier"])].append(row)
        provisional: list[dict[str, Any]] = []
        for identifier in sorted(identifier_groups):
            group  = identifier_groups[identifier]
            labels = {int(row["label"]) for row in group}
            if len(labels) > 1:
                DNASelectionSink._mark_conflict(group, "opposing labels for one protein identifier")
                conflicts.extend(group)
                continue
            group.sort(
                key=lambda row: (
                    int(row.get("structure_rank", 999999)),
                    -int(row.get("interface_residue_count", 0)),
                    str(row.get("candidate_key", "")),
                )
            )
            provisional.append(group[0])
            for duplicate in group[1:]:
                duplicate["included"]         = False
                duplicate["exclusion_reason"] = (
                    "duplicate source protein; better structure retained"
                )
                exclusions.append(duplicate)

        accepted = provisional
        for field, description in (
            ("sequence_sha256", "exact sequence"),
            ("label_conflict_cluster_id", "90%-identity cluster"),
            ("sequence_cluster_id", "30%-identity cluster"),
            ("pdb_id", "PDB deposition"),
        ):
            groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for row in accepted:
                groups[str(row[field])].append(row)
            contradictory = {
                identity
                for identity, group in groups.items()
                if len({int(row["label"]) for row in group}) > 1
            }
            retained: list[dict[str, Any]] = []
            for identity, group in groups.items():
                if identity not in contradictory:
                    retained.extend(group)
                    continue
                DNASelectionSink._mark_conflict(group, f"opposing labels within one {description}")
                conflicts.extend(group)
            accepted = retained

        structure_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in accepted:
            structure_groups[str(row["protein_structure_sha256"])].append(row)
        deduplicated: list[dict[str, Any]] = []
        for group in structure_groups.values():
            group.sort(key=lambda row: str(row["base_identifier"]))
            deduplicated.append(group[0])
            for duplicate in group[1:]:
                duplicate["included"]         = False
                duplicate["exclusion_reason"] = "duplicate exact protein-chain coordinates"
                exclusions.append(duplicate)
        return deduplicated, conflicts, exclusions

    def _assign_clusters(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[dict[str, str], list[dict[str, Any]]]:
        """Assign indivisible homology/deposition components to benchmark partitions.

        A leakage component connects two protein chains when they share either an external RCSB
        MMseqs2 30%-identity cluster or a PDB deposition. Connectivity is transitive: if chain A
        shares a family with B and B shares a deposition with C, all three remain in one split.
        This stricter unit prevents chains from the same experimental structure entering both
        training and evaluation even when those chains belong to different sequence families.

        Args:
            rows: Validated binary rows with official development/external-test provenance.

        Returns:
            Cluster-to-split assignments and development rows excluded because their connected
            component touches the source's protected external-test partition.
        """
        assignments : dict[str, str]       = {}
        excluded    : list[dict[str, Any]] = []
        class_groups: dict[tuple[str, int], list[tuple[str, list[dict[str, Any]]]]] = defaultdict(
            list
        )

        # Keep every connected family/deposition component intact. If official external-test and
        # development records touch, the external boundary wins and development rows are removed.
        for component in self._leakage_components(rows):
            partitions = {str(row["published_partition"]) for row in component}
            eligible   = component
            if "external_test" in partitions and "development" in partitions:
                eligible = [
                    row for row in component if row["published_partition"] == "external_test"
                ]
                for row in component:
                    if row in eligible:
                        continue
                    row["included"]         = False
                    row["exclusion_reason"] = (
                        "homology/deposition component overlaps external test"
                    )
                    excluded.append(row)
            partition = str(eligible[0]["published_partition"])
            label     = int(eligible[0]["label"])
            identities = sorted(
                {f"cluster:{row['sequence_cluster_id']}" for row in eligible}
                | {f"pdb:{row['pdb_id']}" for row in eligible}
            )
            component_id = "|".join(identities)
            class_groups[(partition, label)].append((component_id, eligible))

        for (partition, label), groups in sorted(class_groups.items()):
            ranked = sorted(groups, key=lambda value: self._rank(value[0]))
            reserve_count = 0
            if label == 1 and len(ranked) > 1:
                reserve_count = min(
                    len(ranked) - 1,
                    round(len(ranked) * self.reserve_fraction),
                )
            reserve_groups = ranked[-reserve_count:] if reserve_count else []
            available      = ranked[:-reserve_count] if reserve_count else ranked
            if partition == "external_test":
                self._assign_component_groups(assignments, available, "test")
                self._assign_component_groups(assignments, reserve_groups, "test_reserve")
                continue

            validation_count = 0
            if len(available) > 1:
                validation_count = min(
                    len(available) - 1,
                    max(1, round(len(available) * self.validation_fraction)),
                )
            self._assign_component_groups(assignments, available[:validation_count], "val")
            self._assign_component_groups(assignments, available[validation_count:], "train")
            self._assign_component_groups(
                assignments,
                reserve_groups,
                "validation_reserve",
            )
        return assignments, excluded

    @staticmethod
    def _leakage_components(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
        """Build transitive components linked by sequence family or PDB deposition.

        Args:
            rows: Validated candidate rows with ``sequence_cluster_id`` and ``pdb_id`` fields.

        Returns:
            Deterministically ordered connected components. Each input row occurs exactly once.

        Raises:
            RuntimeError: If a row lacks either leakage-control identity.
        """
        identity_members: dict[tuple[str, str], list[int]] = defaultdict(list)
        for index, row in enumerate(rows):
            cluster_id = str(row.get("sequence_cluster_id") or "")
            pdb_id     = str(row.get("pdb_id") or "")
            if not cluster_id or not pdb_id:
                raise RuntimeError("leakage control requires sequence_cluster_id and pdb_id")
            identity_members[("cluster", cluster_id)].append(index)
            identity_members[("pdb", pdb_id)].append(index)

        # Expanding identity buckets rather than all pairwise row edges keeps the traversal linear
        # in dataset size while still capturing transitive cluster/deposition connections.
        remaining  = set(range(len(rows)))
        components: list[list[dict[str, Any]]] = []
        while remaining:
            pending   = [min(remaining)]
            component: list[int] = []
            while pending:
                index = pending.pop()
                if index not in remaining:
                    continue
                remaining.remove(index)
                component.append(index)
                row = rows[index]
                for identity in (
                    ("cluster", str(row["sequence_cluster_id"])),
                    ("pdb", str(row["pdb_id"])),
                ):
                    pending.extend(identity_members.pop(identity, ()))
            components.append([rows[index] for index in sorted(component)])
        return components

    @staticmethod
    def _assign_component_groups(
        assignments: dict[str, str],
        groups     : Sequence[tuple[str, list[dict[str, Any]]]],
        split      : str,
    ) -> None:
        """Assign every sequence cluster in several connected components to one split.

        Args:
            assignments: Mutable cluster-to-split result mapping.
            groups: Ranked component identifiers and their member rows.
            split: Destination split shared by every component member.

        Raises:
            RuntimeError: If an earlier component assigned the same cluster to another split.
        """
        for _, rows in groups:
            for cluster_id in {str(row["sequence_cluster_id"]) for row in rows}:
                previous = assignments.get(cluster_id)
                if previous is not None and previous != split:
                    raise RuntimeError(
                        "one sequence cluster received conflicting split assignments"
                    )
                assignments[cluster_id] = split

    def _balance_main_splits(
        self,
        rows: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Downsample the majority label to exact parity in every main partition.

        Balancing happens only after complete 30%-identity clusters have been assigned to one
        partition. For a split ``s`` with ``n_0`` negatives and ``n_1`` positives, the retained
        count per class is ``m_s = min(n_0, n_1)``. Rows are ordered by
        ``SHA256(seed, split, label, cluster, identifier)`` before the first ``m_s`` are retained,
        so interrupted or remote builds choose the same proteins without relying on input order.
        Reserve rows are intentionally untouched because they are not training/evaluation dataset
        members; they can only replace locally unevaluable positives in localization reports.

        Args:
            rows: Accepted rows carrying main or reserve split assignments.

        Returns:
            Retained main/reserve rows and majority rows excluded by the balance policy.

        Raises:
            RuntimeError: If any train, validation, or test partition lacks either binary class.
        """
        retained   = [row for row in rows if row["split"] not in self.MAIN_SPLITS]
        exclusions: list[dict[str, Any]] = []

        for split in self.MAIN_SPLITS:
            by_label = {
                label: [row for row in rows if row["split"] == split and row["label"] == label]
                for label in (0, 1)
            }
            if not by_label[0] or not by_label[1]:
                raise RuntimeError(
                    f"cannot balance {split}: both positive and negative rows are required"
                )
            retained_per_class = min(len(by_label[0]), len(by_label[1]))
            for label in (0, 1):
                ranked = sorted(
                    by_label[label],
                    key=lambda row: hashlib.sha256(
                        (
                            f"{self.seed}:{split}:{label}:"
                            f"{row['sequence_cluster_id']}:{row['base_identifier']}"
                        ).encode()
                    ).hexdigest(),
                )
                retained.extend(ranked[:retained_per_class])
                for row in ranked[retained_per_class:]:
                    row["included"]         = False
                    row["exclusion_reason"] = "majority class removed for exact split balance"
                    exclusions.append(row)

        retained.sort(key=lambda row: str(row["base_identifier"]))
        return retained, exclusions

    def _rank(self, cluster_id: str) -> str:
        """Return the stable seeded rank used for cluster allocation.

        Args:
            cluster_id: External RCSB MMseqs2 group identifier.

        Returns:
            SHA-256 hexadecimal digest suitable for deterministic ordering.
        """
        return hashlib.sha256(f"{self.seed}:{cluster_id}".encode()).hexdigest()

    @staticmethod
    def _validate_accepted(rows: list[dict[str, Any]]) -> None:
        """Enforce source, evidence, mapping, and local-GT eligibility invariants.

        Args:
            rows: Candidate accepted rows.

        Raises:
            RuntimeError: If a row lacks defensible evidence, provenance, or structure.
        """
        mandatory = (
            "source_database",
            "source_version",
            "source_url",
            "source_checksum",
            "source_record",
            "pdb_id",
            "published_partition",
            "sequence_cluster_id",
            "label_conflict_cluster_id",
            "sequence_sha256",
            "protein_structure_sha256",
        )
        for row in rows:
            if row.get("label") not in {0, 1} or not row.get("evidence"):
                raise RuntimeError("accepted rows require a binary label and explicit evidence")
            if not Path(str(row.get("structure_path", ""))).is_file():
                raise RuntimeError(f"accepted source is missing: {row.get('candidate_key')}")
            missing = [name for name in mandatory if not row.get(name)]
            if missing:
                raise RuntimeError(f"accepted row lacks mandatory provenance: {missing}")
            if row["published_partition"] not in {"development", "external_test"}:
                raise RuntimeError("published_partition must be development or external_test")
            if row["label"] == 1 and not row.get("positive_assertion"):
                raise RuntimeError("a positive requires a curated DNA-binding assertion")
            if (
                row["label"] == 1
                and row.get("local_gt_expected")
                and row.get("local_gt_method") not in {"binding_residue_mask", "dna_distance"}
            ):
                raise RuntimeError("local GT requires binding residues or a DNA complex")
            if (
                row["label"] == 0
                and EvidenceKind.CURATED_NOT_DNA_BINDING.value not in row["evidence"]
            ):
                raise RuntimeError("every negative requires curated benchmark evidence")
            if row["label"] == 0 and not all(
                row.get(field) is True
                for field in (
                    "no_positive_uniprot_annotation",
                    "no_known_pdb_dna_complex",
                    "no_biolip_dna_binding",
                )
            ):
                raise RuntimeError("accepted negatives must pass every contradiction filter")

    @classmethod
    def _validate_splits(cls, rows: list[dict[str, Any]]) -> None:
        """Reject identifier, structure, exact-sequence, or homology leakage.

        Args:
            rows: Accepted rows with main/reserve split assignments.

        Raises:
            RuntimeError: If any scientific unit occurs in more than one split.
        """
        for field in ("base_identifier", "protein_structure_sha256"):
            values = [str(row[field]) for row in rows]
            if len(values) != len(set(values)):
                raise RuntimeError(f"duplicate {field} values are forbidden")
        for field in ("sequence_cluster_id", "sequence_sha256", "pdb_id"):
            owners: dict[str, set[str]] = defaultdict(set)
            for row in rows:
                owners[str(row[field])].add(str(row["split"]))
            if any(len(splits) != 1 for splits in owners.values()):
                raise RuntimeError(f"{field} leakage was detected across splits")
        for row in rows:
            if row["published_partition"] == "external_test" and row["split"] not in {
                "test",
                "test_reserve",
            }:
                raise RuntimeError("an external-test record entered development")

    @classmethod
    def _validate_class_balance(cls, rows: list[dict[str, Any]]) -> None:
        """Require exact positive/negative parity in all logical dataset partitions.

        Args:
            rows: Retained main and reserve rows after deterministic downsampling.

        Raises:
            RuntimeError: If a main split is empty or its positive and negative counts differ.
        """
        for split in cls.MAIN_SPLITS:
            negative_count = sum(row["split"] == split and row["label"] == 0 for row in rows)
            positive_count = sum(row["split"] == split and row["label"] == 1 for row in rows)
            if negative_count == 0 or negative_count != positive_count:
                raise RuntimeError(
                    f"{split} must be non-empty and exactly class-balanced; "
                    f"found {negative_count} negative and {positive_count} positive"
                )

    @staticmethod
    def _mark_conflict(rows: list[dict[str, Any]], reason: str) -> None:
        """Attach opposing evidence to every quarantined contradiction.

        Args:
            rows: Related rows carrying incompatible labels.
            reason: Human-readable identity relation causing the conflict.
        """
        positive_evidence = sorted(
            str(value)
            for row in rows
            if row.get("label") == 1
            for value in row.get("evidence", ())
        )
        negative_evidence = sorted(
            str(value)
            for row in rows
            if row.get("label") == 0
            for value in row.get("evidence", ())
        )
        for row in rows:
            row["label_state"]       = DNALabel.CONFLICT.value
            row["included"]          = False
            row["conflict_reason"]   = reason
            row["positive_evidence"] = positive_evidence
            row["negative_evidence"] = negative_evidence
            row["exclusion_reason"]  = reason

    def _write_dilutions(
        self,
        rows        : list[dict[str, Any]],
        dataset_root: Path,
    ) -> dict[str, Any]:
        """Publish nested training dilutions with fixed validation and test sets.

        Validation and test remain byte-for-byte identical in every view, so learning curves
        compare training-set size against one fixed evaluation problem. Within each training label,
        rows are ordered breadth-first across 30%-identity clusters: the first pass takes at most
        one member per family before a second member from any family. For full per-class training
        count ``N`` and fraction ``f``, each class retains ``max(1, floor(f N))`` records. Seeded
        prefixes make smaller training selections strict subsets of larger ones.

        Args:
            rows: Final balanced main and reserve catalog rows.
            dataset_root: Compact selection artifact receiving the ``subsets`` directory.

        Returns:
            Ordered summary of per-dilution split, class, and cluster counts.

        Raises:
            RuntimeError: If an upstream main split is not exactly class-balanced.
            OSError: If a TXT, CSV, or JSON selection cannot be atomically published.
        """
        main_rows = [row for row in rows if row["split"] in self.MAIN_SPLITS]
        for split in self.MAIN_SPLITS:
            counts = {
                label: sum(row["split"] == split and row["label"] == label for row in main_rows)
                for label in (0, 1)
            }
            if counts[0] == 0 or counts[0] != counts[1]:
                raise RuntimeError(f"cannot dilute an unbalanced {split} partition")

        # Only training membership changes. Evaluation rows are shared by every view and therefore
        # cannot introduce noise into comparisons between learning-curve points.
        evaluation_rows = [row for row in main_rows if row["split"] in {"val", "test"}]
        orderings = {
            label: self._cluster_diverse_order(
                [
                    row
                    for row in main_rows
                    if row["split"] == "train" and row["label"] == label
                ],
                "train",
                label,
            )
            for label in (0, 1)
        }

        summary: dict[str, Any] = {}
        for fraction in self.dilutions:
            name      = self._dilution_name(fraction)
            per_class = max(1, int(len(orderings[0]) * fraction))
            selected  = [
                *orderings[0][:per_class],
                *orderings[1][:per_class],
                *evaluation_rows,
            ]
            selected.sort(key=lambda row: str(row["base_identifier"]))

            subset_root = dataset_root / "subsets" / name
            subset_root.mkdir(parents=True, exist_ok=True)
            for split in self.MAIN_SPLITS:
                identifiers = sorted(
                    str(row["base_identifier"])
                    for row in selected
                    if row["split"] == split
                )
                self._atomic_text(
                    subset_root / f"{split}.txt",
                    "".join(f"{identifier}\n" for identifier in identifiers),
                )
            self._atomic_text(
                subset_root / "proteins.txt",
                "".join(f"{row['base_identifier']}\n" for row in selected),
            )
            self._write_csv(
                subset_root / "labels.csv",
                selected,
                ("base_identifier", "label", "split", "tier", "sequence_cluster_id"),
            )
            self._write_csv(
                subset_root / "catalog.csv",
                selected,
                self._columns(selected),
            )
            subset_payload = {
                "schema_version": "1.0",
                "fraction": fraction,
                "fraction_applies_to": "train",
                "selection_policy": "balanced_cluster_diverse_nested_training_prefix",
                "records": [
                    {
                        "identifier": str(row["base_identifier"]),
                        "label": int(row["label"]),
                        "split": str(row["split"]),
                        "tier": str(row["tier"]),
                        "sequence_cluster_id": str(row["sequence_cluster_id"]),
                    }
                    for row in selected
                ],
            }
            self._atomic_text(
                subset_root / "identifiers.json",
                json.dumps(subset_payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            )
            summary[name] = {
                "fraction": fraction,
                "fraction_applies_to": "train",
                "member_count": len(selected),
                "split_class_balance": {
                    split: {
                        "positive": sum(
                            row["split"] == split and row["label"] == 1 for row in selected
                        ),
                        "negative": sum(
                            row["split"] == split and row["label"] == 0 for row in selected
                        ),
                        "sequence_clusters": len(
                            {
                                str(row["sequence_cluster_id"])
                                for row in selected
                                if row["split"] == split
                            }
                        ),
                    }
                    for split in self.MAIN_SPLITS
                },
            }
        return summary

    def _cluster_diverse_order(
        self,
        rows : list[dict[str, Any]],
        split: str,
        label: int,
    ) -> list[dict[str, Any]]:
        """Order one class stratum by breadth-first traversal of sequence clusters.

        Args:
            rows: Records from one main split and one binary label.
            split: Main split name included in deterministic ranks.
            label: Binary class included in deterministic ranks.

        Returns:
            Stable row order that visits every cluster before taking repeated family members.
        """
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row["sequence_cluster_id"])].append(row)
        cluster_ids = sorted(
            groups,
            key=lambda cluster_id: hashlib.sha256(
                f"{self.seed}:{split}:{label}:cluster:{cluster_id}".encode()
            ).hexdigest(),
        )
        for cluster_id in cluster_ids:
            groups[cluster_id].sort(
                key=lambda row: hashlib.sha256(
                    f"{self.seed}:{split}:{label}:member:{row['base_identifier']}".encode()
                ).hexdigest()
            )

        ordered: list[dict[str, Any]] = []
        depth = 0
        while any(depth < len(groups[cluster_id]) for cluster_id in cluster_ids):
            ordered.extend(
                groups[cluster_id][depth]
                for cluster_id in cluster_ids
                if depth < len(groups[cluster_id])
            )
            depth += 1
        return ordered

    @staticmethod
    def _dilution_name(fraction: float) -> str:
        """Convert one unit fraction into a stable human-readable directory name.

        Args:
            fraction: Strictly interior unit fraction selected by configuration.

        Returns:
            Percentage name such as ``10pct`` or ``12p5pct`` without path separators.
        """
        percentage = f"{fraction * 100.0:.6f}".rstrip("0").rstrip(".")
        return percentage.replace(".", "p") + "pct"

    @staticmethod
    def _columns(rows: list[dict[str, Any]]) -> tuple[str, ...]:
        """Return a stable union of catalog fields.

        Args:
            rows: Catalog or report rows, possibly empty.

        Returns:
            Identity fields first, followed by alphabetically ordered metadata.
        """
        fields    = {key for row in rows for key in row}
        primary   = ("base_identifier", "label", "split", "label_state")
        initial   = tuple(value for value in primary if value in fields)
        remaining = tuple(sorted(fields - set(primary)))
        return initial + remaining

    @classmethod
    def _write_csv(
        cls,
        path   : Path,
        rows   : list[dict[str, Any]],
        columns: tuple[str, ...],
    ) -> None:
        """Publish deterministic CSV with nested fields encoded as JSON.

        Args:
            path: Final CSV path.
            rows: Ordered catalog mappings.
            columns: Stable output field order.

        Raises:
            OSError: If atomic publication fails.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            key: json.dumps(value, sort_keys=True)
                            if isinstance(value, (list, dict))
                            else value
                            for key, value in row.items()
                        }
                    )
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _write_parquet(
        path   : Path,
        rows   : list[dict[str, Any]],
        columns: tuple[str, ...],
    ) -> None:
        """Publish the typed analytical mirror of the canonical CSV.

        Args:
            path: Final Parquet path.
            rows: Accepted catalog records.
            columns: Canonical field order.

        Raises:
            OSError: If the temporary file cannot be synchronized or replaced.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            pd.DataFrame(rows, columns=columns).to_parquet(temporary, index=False)
            with temporary.open("rb") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _summary(
        all_rows  : list[dict[str, Any]],
        accepted  : list[dict[str, Any]],
        conflicts : list[dict[str, Any]],
        exclusions: list[dict[str, Any]],
        seed      : int,
    ) -> dict[str, Any]:
        """Create the machine-readable scientific construction audit.

        Args:
            all_rows: Every source candidate attempted.
            accepted: Final main and reserve examples.
            conflicts: Quarantined contradictory candidates.
            exclusions: Mapping, quality, duplication, and partition rejections.
            seed: Deterministic split seed recorded with the result.

        Returns:
            Counts by source, label, split, cluster, mapping state, and rejection reason.
        """
        split_class = {
            split: {
                "positive": sum(row["split"] == split and row["label"] == 1 for row in accepted),
                "negative": sum(row["split"] == split and row["label"] == 0 for row in accepted),
            }
            for split in DNASelectionSink.ALL_SPLITS
        }
        rejection_reasons = Counter(
            str(row.get("exclusion_reason") or row.get("label_reason") or "unspecified")
            for row in exclusions
        )
        main_rows    = [row for row in accepted if row["split"] in DNASelectionSink.MAIN_SPLITS]
        reserve_rows = [row for row in accepted if row["split"] not in DNASelectionSink.MAIN_SPLITS]
        return {
            "verdict": "PASS",
            "class_balance_enforced": True,
            "candidate_positives": sum(row.get("label") == 1 for row in all_rows),
            "candidate_negatives": sum(row.get("label") == 0 for row in all_rows),
            "positives_accepted": sum(row["label"] == 1 for row in accepted),
            "negatives_accepted": sum(row["label"] == 0 for row in accepted),
            "logical_member_count": len(main_rows),
            "logical_member_positives": sum(row["label"] == 1 for row in main_rows),
            "logical_member_negatives": sum(row["label"] == 0 for row in main_rows),
            "reserve_count": len(reserve_rows),
            "positives_rejected": sum(row.get("label") == 1 for row in exclusions + conflicts),
            "negatives_rejected": sum(row.get("label") == 0 for row in exclusions + conflicts),
            "mapped_to_pdb": sum(bool(row.get("pdb_id")) for row in accepted),
            "failed_pdb_mapping": sum(
                count for reason, count in rejection_reasons.items() if "mapping" in reason
            ),
            "conflict_count": len(conflicts),
            "exclusion_count": len(exclusions),
            "rejection_reasons": dict(sorted(rejection_reasons.items())),
            "split_class_balance": split_class,
            "sequence_cluster_count": len({str(row["sequence_cluster_id"]) for row in accepted}),
            "source_dataset_counts": dict(
                sorted(Counter(str(row["source_database"]) for row in accepted).items())
            ),
            "split_seed": seed,
            "sequence_identity_threshold": 0.30,
            "cluster_leakage_count": 0,
            "exact_sequence_leakage_count": 0,
            "feature_distributions_by_split": DNASelectionSink._distribution_statistics(accepted),
        }

    @staticmethod
    def _distribution_statistics(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize structural and biological covariates per main split.

        Args:
            rows: Accepted rows carrying structure, taxonomy, and interface descriptors.

        Returns:
            Counts and numeric ranges suitable for detecting accidental distribution shifts.
        """
        numeric_fields = (
            "sequence_length",
            "resolution_angstrom",
            "aspect_ratio",
            "interface_residue_fraction",
        )
        output: dict[str, Any] = {}
        for split in DNASelectionSink.MAIN_SPLITS:
            split_rows    = [row for row in rows if row["split"] == split]
            output[split] = {
                "taxonomy": dict(
                    sorted(
                        Counter(str(row.get("taxonomy") or "unknown") for row in split_rows).items()
                    )
                ),
                "structure_method": dict(
                    sorted(
                        Counter(
                            str(row.get("structure_method") or "unknown") for row in split_rows
                        ).items()
                    )
                ),
            }
            for field in numeric_fields:
                values = sorted(
                    float(row[field]) for row in split_rows if row.get(field) is not None
                )
                output[split][field] = {
                    "count": len(values),
                    "minimum": values[0] if values else None,
                    "mean": sum(values) / len(values) if values else None,
                    "maximum": values[-1] if values else None,
                }
        return output

    @staticmethod
    def _verdict(summary: Mapping[str, Any]) -> str:
        """Render the important construction result for terminal users.

        Args:
            summary: Machine-readable construction summary.

        Returns:
            Concise multi-line verdict including split class counts.
        """
        lines = [
            f"DNA DATASET: {summary['verdict']}",
            f"Logical members: {summary['logical_member_positives']} positive, "
            f"{summary['logical_member_negatives']} negative",
            f"Local-evaluation reserves: {summary['reserve_count']}",
            f"Conflicts: {summary['conflict_count']}; excluded: {summary['exclusion_count']}",
        ]
        for split, counts in summary["split_class_balance"].items():
            lines.append(f"{split}: {counts['positive']} positive, {counts['negative']} negative")
        lines.append("Sequence-cluster leakage: 0")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _plot(rows: list[dict[str, Any]], path: Path) -> None:
        """Plot class, length, resolution, and interface-residue diagnostics.

        Args:
            rows: Accepted main and reserve rows.
            path: Final PNG output path.
        """
        main = [row for row in rows if row["split"] in DNASelectionSink.MAIN_SPLITS]
        figure, axes = plt.subplots(2, 2, figsize=(11.0, 8.0))
        negatives = [
            sum(row["split"] == split and row["label"] == 0 for row in main)
            for split in DNASelectionSink.MAIN_SPLITS
        ]
        positives = [
            sum(row["split"] == split and row["label"] == 1 for row in main)
            for split in DNASelectionSink.MAIN_SPLITS
        ]
        axes[0, 0].bar(DNASelectionSink.MAIN_SPLITS, negatives, label="negative")
        axes[0, 0].bar(
            DNASelectionSink.MAIN_SPLITS,
            positives,
            bottom=negatives,
            label="positive",
        )
        axes[0, 0].set_title("Class distribution")
        axes[0, 0].legend()

        for axis, (field, title) in zip(
            axes.flat[1:],
            (
                ("sequence_length", "Sequence length"),
                ("resolution_angstrom", "Experimental resolution (Å)"),
                ("interface_residue_fraction", "Positive interface residue fraction"),
            ),
            strict=True,
        ):
            for split in DNASelectionSink.MAIN_SPLITS:
                values = [
                    float(row[field])
                    for row in main
                    if row["split"] == split and row.get(field) is not None
                ]
                if values:
                    axis.hist(values, bins=min(15, len(values)), alpha=0.45, label=split)
            axis.set_title(title)
            if axis.get_legend_handles_labels()[0]:
                axis.legend()
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    @staticmethod
    def _atomic_text(path: Path, content: str) -> None:
        """Publish complete UTF-8 content by synchronized atomic replacement.

        Args:
            path: Final output path.
            content: Complete UTF-8 text.

        Raises:
            OSError: If writing, syncing, or replacement fails.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
