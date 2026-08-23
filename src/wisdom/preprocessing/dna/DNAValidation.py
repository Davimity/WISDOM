"""Read-only scientific validation for a published WISDOM-DNA dataset placement."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lambdaforge as lf
import matplotlib
import numpy as np
from lambdaforge.data import DatasetIndex
from scipy.stats import wasserstein_distance
from sklearn.decomposition import PCA
from sklearn.preprocessing import RobustScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt


class DNAValidation(lf.Work):
    """Audit dataset integrity, leakage, balance, phenotype coverage, and train dilutions."""

    def run(self, dataset: Path, fail_on_error: bool = True) -> dict[str, Any]:
        """Run the scientific audit and register its reports with LambdaForge.

        Args:
            dataset: LambdaForge-resolved immutable dataset placement.
            fail_on_error: Raise after publishing reports when any hard check fails.

        Returns:
            Complete machine-readable validation payload.

        Raises:
            ValueError: If the dataset root or member index is absent.
            OSError: If dataset evidence or report outputs cannot be accessed.
            RuntimeError: After report registration, if any hard validation check fails.
        """
        output  = self.run_dir / "dna-validation"
        payload = self.audit(dataset, output)

        self.outputs.artifact(
            "dna-validation-report",
            output / "dna-validation-report.json",
            role="report",
            media_type="application/json",
        )
        self.outputs.artifact(
            "dna-validation-summary",
            output / "dna-validation-report.md",
            role="report",
            media_type="text/markdown",
        )
        self.outputs.artifact(
            "dna-validation-figures",
            output / "figures",
            role="figure",
        )
        failure_count = sum(not check["passed"] for check in payload["checks"])
        self.metrics.log("validation_failures", failure_count)
        if failure_count and fail_on_error:
            raise RuntimeError(
                f"WISDOM-DNA validation failed {failure_count} hard checks; inspect "
                f"{output / 'dna-validation-report.json'}"
            )
        return payload

    def audit(self, dataset: Path, output: Path) -> dict[str, Any]:
        """Validate an immutable dataset without changing its files or registry record.

        Args:
            dataset: Dataset placement containing canonical ``index.jsonl``
                (or the pre-publication ``members.jsonl``), catalogs, specialist-tool pair
                evidence, base NPZ files, and DNA sidecars.
            output: Destination directory for JSON, Markdown, and figure reports.

        Returns:
            Machine-readable verdict and counts also written as JSON, Markdown, and figures.

        Raises:
            ValueError: If the dataset root or its member index is absent.
            OSError: If an input cannot be read or a report artifact cannot be written.
        """
        root = dataset.resolve()
        index_path = root / "index.jsonl"
        if not index_path.is_file():
            index_path = root / "members.jsonl"
        if not index_path.is_file():
            raise ValueError(
                "WISDOM-DNA validation requires index.jsonl or pre-publication members.jsonl"
            )
        members = list(DatasetIndex(index_path))
        if not members:
            raise ValueError("WISDOM-DNA contains no logical members")
        evidence = self._evidence_root(root, members)

        checks: list[dict[str, Any]] = []
        details: list[dict[str, Any]] = []
        by_id = {member.member_id: member for member in members}
        self._check(
            checks,
            "unique_member_ids",
            len(by_id) == len(members),
            len(members) - len(by_id),
            "Each protein must occur exactly once.",
        )
        catalog, catalog_failures = self._catalog_audit(evidence / "catalog.csv", by_id)
        self._check(
            checks,
            "catalog_index_consistency",
            not catalog_failures,
            len(catalog_failures),
            "The canonical catalog and DatasetIndex must name identical members, labels, splits, "
            "and leakage groups.",
        )

        split_ids: dict[str, set[str]] = defaultdict(set)
        group_splits: dict[str, set[str]] = defaultdict(set)
        for member in members:
            split = str(member.partitions.get("split", ""))
            group = str(member.partitions.get("leakage_group", ""))
            split_ids[split].add(member.member_id)
            group_splits[group].add(split)
            try:
                self._validate_member(root, member)
            except (OSError, ValueError, KeyError, json.JSONDecodeError) as error:
                details.append({"identifier": member.member_id, "failure": str(error)})
        overlap = sum(
            len(split_ids[left] & split_ids[right])
            for index, left in enumerate(split_ids)
            for right in list(split_ids)[index + 1 :]
        )
        self._check(
            checks,
            "split_disjointness",
            overlap == 0,
            overlap,
            "No identifier may occur in two evaluation roles.",
        )
        leaked_groups = {
            group: sorted(splits)
            for group, splits in group_splits.items()
            if len(splits) != 1 or not group
        }
        self._check(
            checks,
            "leakage_group_disjointness",
            not leaked_groups,
            len(leaked_groups),
            "Every transitive sequence/structure identity component must remain in one split.",
        )
        self._check(
            checks,
            "npz_and_sidecar_integrity",
            not details,
            len(details),
            "Every archive must be checksum-correct, pickle-free, finite where required, "
            "and point-aligned.",
        )

        pair_failures: dict[str, int] = {}
        for name in ("sequence-pairs.csv", "structure-pairs.csv", "exact-identity-pairs.csv"):
            path = evidence / "clusters" / name
            failures = self._pair_leakage(path, by_id)
            pair_failures[name] = failures
            self._check(
                checks,
                name.removesuffix(".csv") + "_split_safety",
                failures == 0,
                failures,
                "Every retained specialist-tool pair must remain inside one split.",
            )

        local_failures = sum(
            member.partitions.get("split") in {"validation", "test"}
            and int(member.targets["dna_binding"]) == 1
            and not bool(member.targets["local_ground_truth"])
            for member in members
        )
        self._check(
            checks,
            "evaluation_positive_local_gt",
            local_failures == 0,
            local_failures,
            "Every positive validation/test protein must contain at least one valid positive "
            "surface point.",
        )

        class_counts = Counter(int(member.targets["dna_binding"]) for member in members)
        self._check(
            checks,
            "global_class_balance",
            class_counts[0] == class_counts[1] and class_counts[0] > 0,
            abs(class_counts[0] - class_counts[1]),
            "The published benchmark must contain equal positive and negative populations; "
            "group-aware split sizes may still deviate slightly.",
        )

        phenotype_failures = self._phenotype_coverage(members)
        self._check(
            checks,
            "phenotype_coverage_when_feasible",
            not phenotype_failures,
            len(phenotype_failures),
            "A stable physical phenotype represented by at least three independently movable "
            "leakage groups must occur in train, validation, and test.",
        )

        dilution = self._validate_dilutions(evidence, by_id)
        self._check(
            checks,
            "dilution_integrity",
            dilution["failure_count"] == 0,
            dilution["failure_count"],
            "Dilutions must be nested training-only subsets with no unknown IDs or split movement.",
        )
        statistics = self._statistics(members, catalog)
        cluster_audit = self._cluster_audit(evidence, by_id)
        raw_pair_failures = self._raw_pair_leakage(evidence, by_id)
        for name, failures in raw_pair_failures.items():
            self._check(
                checks,
                name + "_raw_threshold_safety",
                failures == 0,
                failures,
                "Reapplying the recorded identity/probability, bilateral-coverage, and E-value "
                "thresholds to raw specialist output must find no cross-split pair.",
            )
        verdict = "PASS" if all(check["passed"] for check in checks) else "FAIL"
        payload = {
            "verdict": verdict,
            "member_count": len(members),
            "checks": checks,
            "statistics": statistics,
            "cluster_audit": cluster_audit,
            "pair_failures": pair_failures,
            "raw_pair_failures": raw_pair_failures,
            "phenotype_coverage_failures": phenotype_failures,
            "dilutions": dilution,
            "member_failures": details,
            "catalog_failures": catalog_failures,
        }

        output.mkdir(parents=True, exist_ok=True)
        figures = output / "figures"
        figures.mkdir(exist_ok=True)
        report_json = output / "dna-validation-report.json"
        report_md   = output / "dna-validation-report.md"
        report_json.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8"
        )
        report_md.write_text(self._markdown(payload), encoding="utf-8")
        self._plot(payload, figures / "dataset-distributions.png")
        self._plot_phenotype_pca(evidence, figures / "phenotype-pca.png")
        return payload

    @staticmethod
    def _cluster_audit(evidence: Path, members: Mapping[str, Any]) -> dict[str, Any]:
        """Summarize leakage components, cross-split similarity, and HDBSCAN outcomes.

        Args:
            evidence: Dataset-wide evidence directory.
            members: Logical members keyed by identifier.

        Returns:
            Group-size, phenotype-by-split, specialist-pair maxima, and stability diagnostics.
        """
        group_sizes = Counter(
            str(member.partitions.get("leakage_group", "")) for member in members.values()
        )
        phenotype_by_split: dict[str, Counter[str]] = {
            split: Counter() for split in ("train", "validation", "test")
        }
        for member in members.values():
            split = str(member.partitions.get("split", ""))
            phenotype_by_split.setdefault(split, Counter())[
                str(member.partitions.get("phenotype", "unavailable"))
            ] += 1

        max_cross_sequence: float | None = None
        sequence_path = evidence / "clusters" / "sequence-pairs.tsv"
        if sequence_path.is_file():
            with sequence_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 7 or fields[0] not in members or fields[1] not in members:
                        continue
                    if members[fields[0]].partitions["split"] != members[fields[1]].partitions[
                        "split"
                    ]:
                        identity = float(fields[2])
                        max_cross_sequence = max(max_cross_sequence or 0.0, identity)

        max_cross_structure: float | None = None
        structure_path = evidence / "clusters" / "structure-pairs.tsv"
        if structure_path.is_file():
            with structure_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 7:
                        continue
                    left, right = Path(fields[0]).stem, Path(fields[1]).stem
                    if left not in members or right not in members:
                        continue
                    if members[left].partitions["split"] != members[right].partitions["split"]:
                        probability = float(fields[2])
                        if probability > 1.0:
                            probability /= 100.0
                        max_cross_structure = max(max_cross_structure or 0.0, probability)

        partition_report: dict[str, Any] = {}
        report_path = evidence / "partition-report.json"
        if report_path.is_file():
            partition_report = json.loads(report_path.read_text(encoding="utf-8"))
        return {
            "leakage_group_count": len(group_sizes),
            "largest_leakage_group": max(group_sizes.values(), default=0),
            "singleton_fraction": (
                sum(size == 1 for size in group_sizes.values()) / len(group_sizes)
                if group_sizes
                else None
            ),
            "group_size_histogram": dict(sorted(Counter(group_sizes.values()).items())),
            "maximum_cross_split_mmseqs2_identity": max_cross_sequence,
            "maximum_cross_split_foldseek_probability": max_cross_structure,
            "phenotype_by_split": {
                split: dict(sorted(counts.items()))
                for split, counts in phenotype_by_split.items()
            },
            "positive_hdbscan": partition_report.get("positive_phenotypes", {}),
            "negative_hdbscan": partition_report.get("negative_phenotypes", {}),
            "software": partition_report.get("software", {}),
            "thresholds": partition_report.get("parameters", {}),
        }

    @staticmethod
    def _catalog_audit(
        path: Path, members: Mapping[str, Any]
    ) -> tuple[dict[str, dict[str, str]], list[str]]:
        """Cross-check the canonical catalog against the logical member index.

        Args:
            path: Final ``catalog.csv`` path.
            members: Dataset members keyed by identifier.

        Returns:
            Unique catalog rows and concise invariant failures.
        """
        if not path.is_file():
            return {}, ["catalog.csv is missing"]
        rows: dict[str, dict[str, str]] = {}
        failures: list[str] = []
        with path.open("r", encoding="utf-8", newline="") as stream:
            for number, row in enumerate(csv.DictReader(stream), start=2):
                identifier = str(row.get("identifier", "")).strip()
                if not identifier or identifier in rows:
                    failures.append(f"catalog line {number} has an empty or duplicate identifier")
                    continue
                rows[identifier] = dict(row)
        missing = sorted(members.keys() - rows.keys())
        extra = sorted(rows.keys() - members.keys())
        if missing:
            failures.append(f"catalog misses {len(missing)} indexed members")
        if extra:
            failures.append(f"catalog contains {len(extra)} unindexed members")

        uniprot_splits: dict[str, set[str]] = defaultdict(set)
        for identifier in members.keys() & rows.keys():
            member = members[identifier]
            row = rows[identifier]
            try:
                label = int(row.get("label", ""))
            except ValueError:
                label = -1
            split = str(row.get("split", ""))
            group = str(row.get("leakage_group", ""))
            if label not in {0, 1} or label != int(member.targets.get("dna_binding", -1)):
                failures.append(f"{identifier}: catalog/index label mismatch")
            if split not in {"train", "validation", "test"} or split != str(
                member.partitions.get("split", "")
            ):
                failures.append(f"{identifier}: catalog/index split mismatch")
            if not group or group != str(member.partitions.get("leakage_group", "")):
                failures.append(f"{identifier}: catalog/index leakage group mismatch")
            logical = str(row.get("logical_protein_id", ""))
            if logical.startswith("uniprot:"):
                uniprot_splits[logical].add(split)
        for accession, splits in uniprot_splits.items():
            if len(splits) > 1:
                failures.append(f"{accession}: same UniProt crosses {sorted(splits)}")
        return rows, failures

    @staticmethod
    def _evidence_root(root: Path, members: list[Any]) -> Path:
        """Resolve run-owned or published dataset-wide audit evidence.

        Before publication, canonical tables live directly below the final run root. LambdaForge
        0.12 member publication has no global-assets argument, so the placement stores those same
        small files once as the first member's ``dataset_evidence`` directory asset.

        Args:
            root: Dataset or pre-publication root.
            members: Parsed logical members whose assets may expose the evidence directory.

        Returns:
            Directory containing ``catalog.csv``, ``clusters/``, and ``dilutions/``.

        Raises:
            ValueError: If neither representation contains the mandatory evidence contract.
        """
        if (root / "catalog.csv").is_file():
            return root
        candidates = [
            root / member.assets["dataset_evidence"].path
            for member in members
            if "dataset_evidence" in member.assets
        ]
        if len(candidates) != 1 or not (candidates[0] / "catalog.csv").is_file():
            raise ValueError("WISDOM-DNA lacks its unique dataset_evidence asset")
        return candidates[0]

    @staticmethod
    def _phenotype_coverage(members: list[Any]) -> dict[str, Any]:
        """Find stable phenotype clusters omitted from a split despite feasible group support.

        A phenotype is considered distributable when it occurs in at least three leakage groups
        and at least three of those groups may legally leave training. A group is train-only when
        it contains a positive protein without local ground truth. Noise and unavailable labels are
        descriptive outcomes, not claimed biological clusters, so they are excluded.

        Args:
            members: Complete logical dataset members.

        Returns:
            Mapping from offending phenotype to its available groups and observed splits.
        """
        groups: dict[str, list[Any]] = defaultdict(list)
        for member in members:
            groups[str(member.partitions.get("leakage_group", ""))].append(member)

        phenotype_groups: dict[str, set[str]] = defaultdict(set)
        phenotype_splits: dict[str, set[str]] = defaultdict(set)
        for member in members:
            phenotype = str(member.partitions.get("phenotype", "unavailable"))
            if phenotype.endswith("NOISE") or phenotype == "unavailable":
                continue
            group = str(member.partitions.get("leakage_group", ""))
            phenotype_groups[phenotype].add(group)
            phenotype_splits[phenotype].add(str(member.partitions.get("split", "")))

        failures: dict[str, Any] = {}
        required = {"train", "validation", "test"}
        for phenotype, candidate_groups in sorted(phenotype_groups.items()):
            movable = {
                group
                for group in candidate_groups
                if not any(
                    int(member.targets["dna_binding"]) == 1
                    and not bool(member.targets["local_ground_truth"])
                    for member in groups[group]
                )
            }
            observed = phenotype_splits[phenotype]
            if len(candidate_groups) >= 3 and len(movable) >= 3 and observed != required:
                failures[phenotype] = {
                    "group_count": len(candidate_groups),
                    "movable_group_count": len(movable),
                    "observed_splits": sorted(observed),
                    "missing_splits": sorted(required - observed),
                }
        return failures

    @staticmethod
    def _validate_member(root: Path, member: Any) -> None:
        """Validate checksums and cross-array alignment for one logical member.

        Args:
            root: Dataset placement root.
            member: LambdaForge ``DatasetMember`` with universal and DNA assets.

        Raises:
            ValueError: If checksums, labels, shapes, masks, values, or fingerprints disagree.
            OSError: If an asset cannot be read.
        """
        assets = member.assets
        for required in ("universal_npz", "dna_annotation", "source_structure"):
            if required not in assets:
                raise ValueError(f"missing required asset {required}")
            path = root / assets[required].path
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if assets[required].sha256 != f"sha256:{digest}":
                raise ValueError(f"checksum mismatch for {required}")
        base_path = root / assets["universal_npz"].path
        side_path = root / assets["dna_annotation"].path
        with (
            np.load(base_path, allow_pickle=False) as base,
            np.load(side_path, allow_pickle=False) as side,
        ):
            if any(base[name].dtype == object for name in base.files) or any(
                side[name].dtype == object for name in side.files
            ):
                raise ValueError("object arrays are forbidden")
            points = base["surface_positions"]
            hard = side["surface_target_hard"]
            valid = side["surface_valid_mask"]
            if (
                points.ndim != 2
                or points.shape[1] != 3
                or hard.shape != (len(points),)
                or valid.shape != hard.shape
            ):
                raise ValueError("surface and target arrays are not point-aligned")
            if not np.isfinite(points).all() or not np.isfinite(base["surface_normals"]).all():
                raise ValueError("base geometry contains non-finite values")
            base_hash = hashlib.sha256(base_path.read_bytes()).hexdigest()
            if str(side["base_npz_sha256"].item()) != base_hash:
                raise ValueError("sidecar points to different universal geometry bytes")
            label = int(member.targets["dna_binding"])
            if label == 1 and bool(member.targets["local_ground_truth"]) and not np.any(hard == 1):
                raise ValueError("positive local GT contains zero positive surface points")
            if label == 0 and np.any(hard != 0):
                raise ValueError("curated negative contains positive surface targets")

    @staticmethod
    def _pair_leakage(path: Path, members: Mapping[str, Any]) -> int:
        """Count thresholded similarity pairs crossing final splits.

        Args:
            path: Canonical two-column pair CSV.
            members: Dataset members keyed by identifier.

        Returns:
            Number of unknown or cross-split pairs.
        """
        if not path.is_file():
            return 1
        failures = 0
        with path.open("r", encoding="utf-8", newline="") as stream:
            for row in csv.DictReader(stream):
                left, right = str(row.get("left", "")), str(row.get("right", ""))
                if (
                    left not in members
                    or right not in members
                    or members[left].partitions["split"] != members[right].partitions["split"]
                ):
                    failures += 1
        return failures

    @staticmethod
    def _raw_pair_leakage(evidence: Path, members: Mapping[str, Any]) -> dict[str, int]:
        """Reapply recorded thresholds directly to raw MMseqs2 and Foldseek tables.

        Args:
            evidence: Dataset-wide evidence directory containing raw TSV and partition provenance.
            members: Dataset members keyed by exact logical identifier.

        Returns:
            Cross-split qualifying-pair counts for both specialist tools. A missing or malformed
            mandatory evidence file contributes one failure instead of being silently ignored.
        """
        report_path = evidence / "partition-report.json"
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            parameters = report["parameters"]
            sequence_identity = float(parameters["sequence_identity"])
            sequence_coverage = float(parameters["sequence_coverage"])
            sequence_evalue = float(parameters["sequence_evalue"])
            structure_probability = float(parameters["structure_probability"])
            structure_evalue = float(parameters["structure_evalue"])
        except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
            return {"mmseqs2": 1, "foldseek": 1}

        sequence_failures = 0
        sequence_path = evidence / "clusters" / "sequence-pairs.tsv"
        try:
            with sequence_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 7:
                        sequence_failures += 1
                        continue
                    left, right = fields[:2]
                    if left == right:
                        continue
                    if left not in members or right not in members:
                        sequence_failures += 1
                        continue
                    identity, query_coverage, target_coverage, evalue = map(float, fields[2:6])
                    retained = (
                        identity >= sequence_identity
                        and min(query_coverage, target_coverage) >= sequence_coverage
                        and evalue <= sequence_evalue
                    )
                    if retained and members[left].partitions["split"] != members[right].partitions[
                        "split"
                    ]:
                        sequence_failures += 1
        except (OSError, ValueError):
            sequence_failures += 1

        structure_failures = 0
        structure_path = evidence / "clusters" / "structure-pairs.tsv"
        try:
            with structure_path.open("r", encoding="utf-8") as stream:
                for line in stream:
                    fields = line.rstrip("\n").split("\t")
                    if len(fields) != 7:
                        structure_failures += 1
                        continue
                    left, right = Path(fields[0]).stem, Path(fields[1]).stem
                    if left == right:
                        continue
                    if left not in members or right not in members:
                        structure_failures += 1
                        continue
                    probability = float(fields[2])
                    if probability > 1.0:
                        probability /= 100.0
                    retained = (
                        probability >= structure_probability
                        and float(fields[3]) <= structure_evalue
                    )
                    if retained and members[left].partitions["split"] != members[right].partitions[
                        "split"
                    ]:
                        structure_failures += 1
        except (OSError, ValueError):
            structure_failures += 1
        return {"mmseqs2": sequence_failures, "foldseek": structure_failures}

    @staticmethod
    def _validate_dilutions(root: Path, members: Mapping[str, Any]) -> dict[str, Any]:
        """Audit canonical train dilution membership and nesting.

        Args:
            root: Dataset root containing ``dilutions/canonical/train-N.txt``.
            members: Final members keyed by identifier.

        Returns:
            Per-dilution sizes, class counts, nesting result, and total failure count.
        """
        paths = sorted(
            (root / "dilutions" / "canonical").glob("train-*.txt"),
            key=lambda path: int(path.stem.split("-")[1]),
        )
        previous: set[str] = set()
        result: dict[str, Any] = {}
        failures = int(not paths)
        warnings: list[str] = []
        train_ids = {
            identifier
            for identifier, member in members.items()
            if member.partitions.get("split") == "train"
        }
        train_positive = sum(
            int(members[identifier].targets["dna_binding"]) == 1 for identifier in train_ids
        )
        train_fraction = train_positive / len(train_ids) if train_ids else 0.0
        stable_phenotypes = {
            str(members[identifier].partitions.get("phenotype", "unavailable"))
            for identifier in train_ids
            if not str(members[identifier].partitions.get("phenotype", "unavailable")).endswith(
                ("NOISE", "unavailable")
            )
        }
        for path in paths:
            identifiers = {
                line.strip()
                for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            }
            unknown = identifiers - members.keys()
            non_train = {
                identifier
                for identifier in identifiers & members.keys()
                if members[identifier].partitions["split"] != "train"
            }
            nested = previous.issubset(identifiers)
            failures += len(unknown) + len(non_train) + int(not nested)
            counts = Counter(
                int(members[identifier].targets["dna_binding"])
                for identifier in identifiers & members.keys()
            )
            selected = identifiers & members.keys()
            phenotypes = Counter(
                str(members[identifier].partitions.get("phenotype", "unavailable"))
                for identifier in selected
            )
            leakage_groups = {
                str(members[identifier].partitions.get("leakage_group", ""))
                for identifier in selected
            }
            missing_phenotypes = sorted(stable_phenotypes - phenotypes.keys())
            phenotypes_in_previous = {
                str(members[identifier].partitions.get("phenotype", "unavailable"))
                for identifier in previous & members.keys()
            }
            additions_available = len(identifiers) - len(previous)
            missing_after_previous = stable_phenotypes - phenotypes_in_previous
            feasible_coverage = additions_available >= len(missing_after_previous)
            if missing_phenotypes and feasible_coverage:
                warnings.append(
                    f"{path.stem} omits stable phenotypes despite a nested alternative with the "
                    f"same size: "
                    f"{missing_phenotypes}"
                )
            positive_fraction = counts[1] / len(selected) if selected else None
            result[path.stem] = {
                "size": len(identifiers),
                "positive": counts[1],
                "negative": counts[0],
                "unknown": sorted(unknown),
                "non_train": sorted(non_train),
                "contains_previous": nested,
                "positive_fraction": positive_fraction,
                "positive_fraction_deviation_from_full_train": (
                    abs(positive_fraction - train_fraction)
                    if positive_fraction is not None
                    else None
                ),
                "phenotype_counts": dict(sorted(phenotypes.items())),
                "phenotype_coverage_fraction": (
                    len(stable_phenotypes & phenotypes.keys()) / len(stable_phenotypes)
                    if stable_phenotypes
                    else None
                ),
                "leakage_group_count": len(leakage_groups),
                "missing_stable_phenotypes": missing_phenotypes,
            }
            previous = identifiers
        return {"failure_count": failures, "warnings": warnings, "subsets": result}

    @staticmethod
    def _statistics(
        members: list[Any], catalog: Mapping[str, Mapping[str, str]]
    ) -> dict[str, Any]:
        """Summarize categorical and continuous scientific distributions by split.

        Args:
            members: Complete logical dataset members.
            catalog: Canonical scientific/provenance rows keyed by identifier.

        Returns:
            Class, source, phenotype, leakage, quality, and pairwise distribution summaries.
        """
        result: dict[str, Any] = {}
        continuous_fields = (
            "observed_residue_count",
            "surface_count",
            "atom_count",
            "resolution_angstrom",
            "sequence_coverage",
            "interface_fraction",
            "number_of_positive_regions",
        )
        values_by_split: dict[str, dict[str, list[float]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for split in ("train", "validation", "test"):
            selected = [member for member in members if member.partitions.get("split") == split]
            labels = Counter(int(member.targets["dna_binding"]) for member in selected)
            phenotypes = Counter(
                str(member.partitions.get("phenotype", "unavailable")) for member in selected
            )
            groups = Counter(str(member.partitions.get("leakage_group", "")) for member in selected)
            sources = Counter(
                str(catalog.get(member.member_id, {}).get("source_dataset", "unknown"))
                for member in selected
            )
            for member in selected:
                row = catalog.get(member.member_id, {})
                for field in continuous_fields:
                    try:
                        value = float(row.get(field, ""))
                    except (TypeError, ValueError):
                        continue
                    if np.isfinite(value):
                        values_by_split[split][field].append(value)
            result[split] = {
                "total": len(selected),
                "positive": labels[1],
                "negative": labels[0],
                "positive_fraction": labels[1] / len(selected) if selected else None,
                "leakage_groups": len(groups),
                "largest_leakage_group": max(groups.values(), default=0),
                "phenotypes": dict(sorted(phenotypes.items())),
                "source_datasets": dict(sorted(sources.items())),
                "continuous": {
                    field: DNAValidation._distribution(values)
                    for field, values in sorted(values_by_split[split].items())
                },
                "local_gt_available": sum(
                    bool(member.targets["local_ground_truth"]) for member in selected
                ),
            }
        comparisons: dict[str, Any] = {}
        for split in ("validation", "test"):
            comparisons[f"train_vs_{split}"] = {}
            fields = set(values_by_split["train"]) | set(values_by_split[split])
            for field in sorted(fields):
                train_values = values_by_split["train"].get(field, [])
                other_values = values_by_split[split].get(field, [])
                if not train_values or not other_values:
                    continue
                pooled = math.sqrt(
                    (
                        float(np.var(train_values)) + float(np.var(other_values))
                    )
                    / 2.0
                )
                difference = float(np.mean(other_values) - np.mean(train_values))
                comparisons[f"train_vs_{split}"][field] = {
                    "standardized_mean_difference": difference / pooled if pooled > 0.0 else None,
                    "wasserstein_distance": float(
                        wasserstein_distance(train_values, other_values)
                    ),
                }
        result["distribution_comparisons"] = comparisons
        return result

    @staticmethod
    def _distribution(values: list[float]) -> dict[str, float | int]:
        """Return robust and conventional summaries for one finite scalar measurement.

        Args:
            values: Non-empty finite observations in one scientific unit.

        Returns:
            Count, mean, standard deviation, median, and interquartile range.
        """
        array = np.asarray(values, dtype=np.float64)
        return {
            "count": len(array),
            "minimum": float(array.min()),
            "mean": float(array.mean()),
            "standard_deviation": float(array.std()),
            "median": float(np.median(array)),
            "q25": float(np.quantile(array, 0.25)),
            "q75": float(np.quantile(array, 0.75)),
            "maximum": float(array.max()),
        }

    @staticmethod
    def _check(
        checks: list[dict[str, Any]], name: str, passed: bool, failures: int, meaning: str
    ) -> None:
        """Append one consistently structured validation result.

        Args:
            checks: Mutable ordered result collection.
            name: Stable machine-readable check name.
            passed: Whether the invariant holds.
            failures: Number of offending records, groups, or pairs.
            meaning: Plain-language explanation of what the invariant protects.
        """
        checks.append(
            {"name": name, "passed": passed, "failure_count": failures, "meaning": meaning}
        )

    @staticmethod
    def _markdown(payload: Mapping[str, Any]) -> str:
        """Render a concise explanatory Markdown validation report.

        Args:
            payload: Complete JSON validation report.

        Returns:
            Markdown with verdict, explained hard checks, and split statistics.
        """
        lines = [
            "# WISDOM-DNA validation",
            "",
            f"**Verdict: {payload['verdict']}**",
            "",
            "A PASS means every byte/array, split boundary, leakage edge, local target, "
            "and dilution invariant was rechecked from the immutable placement.",
            "",
            "## Hard checks",
            "",
        ]
        for check in payload["checks"]:
            mark = "PASS" if check["passed"] else "FAIL"
            lines.append(
                f"- **{mark} — {check['name']}**: {check['meaning']} "
                f"Failures: {check['failure_count']}."
            )
        lines.extend(
            [
                "",
                "## Split interpretation",
                "",
                "The positive fraction measures label balance, while leakage-group counts measure "
                "independent sequence/structure families. Phenotype counts describe physical "
                "diversity; they are not identity groups and never define leakage.",
                "",
            ]
        )
        for split in ("train", "validation", "test"):
            values = payload["statistics"][split]
            fraction = values["positive_fraction"]
            shown = "unavailable" if fraction is None else f"{fraction:.3f}"
            lines.append(
                f"- **{split}**: {values['total']} proteins, positive fraction {shown}, "
                f"{values['leakage_groups']} leakage groups, largest group "
                f"{values['largest_leakage_group']}."
            )
        cluster = payload["cluster_audit"]
        lines.extend(
            [
                "",
                "## Similarity and phenotype interpretation",
                "",
                f"There are {cluster['leakage_group_count']} independent leakage groups; the "
                f"largest contains {cluster['largest_leakage_group']} retained proteins and the "
                f"singleton fraction is {cluster['singleton_fraction']}.",
                "HDBSCAN noise means that the current physical descriptors do not support a "
                "stable discrete phenotype for that protein. It is retained and is not renamed "
                "as a functional class.",
                "",
                "## Dilutions",
                "",
                "Only training membership changes. Positive-fraction deviation compares each "
                "nested subset with full train; phenotype coverage reports stable clusters still "
                "represented. Validation and test remain the same immutable members.",
            ]
        )
        for name, values in payload["dilutions"]["subsets"].items():
            lines.append(
                f"- **{name}**: {values['size']} proteins, positive fraction "
                f"{values['positive_fraction']}, {values['leakage_group_count']} leakage groups, "
                f"phenotype coverage {values['phenotype_coverage_fraction']}."
            )
        for warning in payload["dilutions"]["warnings"]:
            lines.append(f"- **WARNING**: {warning}")
        return "\n".join(lines) + "\n"

    @staticmethod
    def _plot(payload: Mapping[str, Any], path: Path) -> None:
        """Plot class, leakage, phenotype, size, surface, and dilution summaries.

        Args:
            payload: Complete validation result containing split and dilution statistics.
            path: PNG output path.
        """
        statistics = payload["statistics"]
        splits = ["train", "validation", "test"]
        positive = [statistics[split]["positive"] for split in splits]
        negative = [statistics[split]["negative"] for split in splits]
        figure, grid = plt.subplots(3, 3, figsize=(16, 13))
        axes = grid.ravel()
        positions = np.arange(len(splits))
        axes[0].bar(positions - 0.18, positive, width=0.36, label="positive")
        axes[0].bar(positions + 0.18, negative, width=0.36, label="negative")
        axes[0].set_xticks(positions, splits)
        axes[0].set_title("Class counts by split")
        axes[0].legend()
        group_histogram = payload["cluster_audit"]["group_size_histogram"]
        group_sizes = sorted(group_histogram)
        axes[1].bar(group_sizes, [group_histogram[size] for size in group_sizes])
        axes[1].set_xlabel("Proteins in leakage group")
        axes[1].set_ylabel("Group count")
        axes[1].set_title("Leakage-group size distribution")

        phenotype_table = payload["cluster_audit"]["phenotype_by_split"]
        for axis, prefix, title in (
            (axes[2], "P", "Positive local phenotypes by split"),
            (axes[3], "N", "Negative global phenotypes by split"),
        ):
            phenotypes = sorted(
                {
                    name
                    for values in phenotype_table.values()
                    for name in values
                    if name.startswith(prefix)
                }
            )
            bottom = np.zeros(len(splits))
            for phenotype in phenotypes:
                values = np.asarray(
                    [phenotype_table.get(split, {}).get(phenotype, 0) for split in splits]
                )
                axis.bar(splits, values, bottom=bottom, label=phenotype)
                bottom += values
            axis.set_title(title)
            if len(phenotypes) <= 12:
                axis.legend(fontsize=7)

        # Quartile boxes show within-split spread instead of presenting only a central value.
        for axis, field, title in (
            (axes[4], "observed_residue_count", "Observed sequence-length distribution"),
            (axes[5], "surface_count", "Surface point-count distribution"),
            (axes[6], "interface_fraction", "Positive interface-fraction distribution"),
        ):
            summaries = [statistics[split]["continuous"].get(field) for split in splits]
            available = [
                {
                    "label": split,
                    "whislo": values["minimum"],
                    "q1": values["q25"],
                    "med": values["median"],
                    "q3": values["q75"],
                    "whishi": values["maximum"],
                    "fliers": [],
                }
                for split, values in zip(splits, summaries, strict=True)
                if values
            ]
            if available:
                axis.bxp(available, showfliers=False)
            else:
                axis.text(0.5, 0.5, "Unavailable", ha="center", va="center")
                axis.set_axis_off()
            axis.set_title(title)

        dilutions = payload["dilutions"]["subsets"]
        names = list(dilutions)
        dilution_positive = [dilutions[name]["positive"] for name in names]
        dilution_negative = [dilutions[name]["negative"] for name in names]
        positions = np.arange(len(names))
        coverage = [dilutions[name]["phenotype_coverage_fraction"] for name in names]
        axes[7].plot(
            positions,
            [np.nan if value is None else value for value in coverage],
            marker="o",
        )
        axes[7].set_xticks(positions, names, rotation=45, ha="right")
        axes[7].set_ylim(0.0, 1.05)
        axes[7].set_title("Stable phenotype coverage in dilutions")
        axes[8].bar(positions, dilution_positive, label="positive")
        axes[8].bar(positions, dilution_negative, bottom=dilution_positive, label="negative")
        axes[8].set_xticks(positions, names, rotation=45, ha="right")
        axes[8].set_title("Nested training composition")
        axes[8].legend()
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)

    @staticmethod
    def _plot_phenotype_pca(evidence: Path, path: Path) -> None:
        """Project physical descriptors to two PCA axes for inspection only.

        PCA is fitted independently to the positive-interface and negative-global robust-scaled
        descriptor tables. The projection never feeds HDBSCAN, leakage grouping, splitting, or
        model training; it merely lets a reviewer see overlap, outliers, and claimed clusters.

        Args:
            evidence: Dataset-wide evidence root containing phenotype descriptor CSV files.
            path: Output PNG path.
        """
        figure, axes = plt.subplots(1, 2, figsize=(11, 5))
        for axis, name, title in (
            (axes[0], "positive-phenotypes.csv", "Positive local-interface PCA"),
            (axes[1], "negative-phenotypes.csv", "Negative global-morphology PCA"),
        ):
            source = evidence / "clusters" / name
            rows: list[dict[str, str]] = []
            if source.is_file():
                with source.open("r", encoding="utf-8", newline="") as stream:
                    rows = list(csv.DictReader(stream))
            feature_names = sorted(
                set(rows[0]) - {"identifier", "phenotype"} if rows else set()
            )
            if len(rows) < 2 or len(feature_names) < 2:
                axis.text(0.5, 0.5, "Insufficient descriptor support", ha="center", va="center")
                axis.set_axis_off()
                axis.set_title(title)
                continue
            matrix = np.asarray(
                [[float(row[feature]) for feature in feature_names] for row in rows],
                dtype=np.float64,
            )
            projected = PCA(n_components=2).fit_transform(RobustScaler().fit_transform(matrix))
            labels = sorted({row["phenotype"] for row in rows})
            for label in labels:
                mask = np.asarray([row["phenotype"] == label for row in rows])
                axis.scatter(
                    projected[mask, 0],
                    projected[mask, 1],
                    s=18,
                    alpha=0.75,
                    label=label,
                )
            axis.set_xlabel("PCA 1")
            axis.set_ylabel("PCA 2")
            axis.set_title(title)
            if len(labels) <= 12:
                axis.legend(fontsize=7)
        figure.tight_layout()
        figure.savefig(path, dpi=160)
        plt.close(figure)
