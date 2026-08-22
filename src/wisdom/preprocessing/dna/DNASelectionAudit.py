"""Human-readable and machine-readable quality control for DNA selections."""

from __future__ import annotations

import csv
import json
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


class DNASelectionAudit:
    """Validate split safety and explain the composition of every dataset view."""

    MAIN_SPLITS    = ("train", "val", "test")
    RESERVE_SPLITS = ("validation_reserve", "test_reserve")
    LEAKAGE_KEYS   = (
        ("base_identifier", "protein identifier"),
        ("sequence_sha256", "exact amino-acid sequence"),
        ("sequence_cluster_id", "30%-identity sequence family"),
        ("pdb_id", "PDB deposition"),
        ("protein_structure_sha256", "selected-chain coordinates"),
    )

    def audit(
        self,
        selection_root: Path,
        publish       : bool = True,
    ) -> dict[str, Any]:
        """Audit one complete selection and write reports beside each dataset view.

        The audit joins the compact membership files back to ``catalog.csv`` and checks exact file
        agreement, class balance, identifier uniqueness, external-test provenance, and five levels
        of cross-split leakage. It then measures sequence-family coverage, structural tier coverage,
        source/label coupling, and interpretable numeric distributions. Diluted views must retain
        full validation and test membership while their balanced training prefixes remain nested.

        Args:
            selection_root: Directory containing the complete selection and ``subsets`` children.
            publish: Whether to atomically write reports and figures. ``False`` performs the same
                checks read-only, which lets preprocessing reject unsafe external selections.

        Returns:
            JSON-compatible complete-selection report with summaries of every diluted view.

        Raises:
            FileNotFoundError: If a required catalog, label, identifier, or split file is absent.
            ValueError: If CSV/JSON content is malformed or references an unknown catalog member.
            OSError: If an audit JSON, Markdown, CSV, or PNG cannot be atomically published.
        """
        selection_root = selection_root.resolve()
        catalog        = pd.read_csv(selection_root / "catalog.csv")
        self._require_columns(catalog)

        view_roots = [("complete", selection_root)]
        subset_root = selection_root / "subsets"
        if subset_root.is_dir():
            view_roots.extend(
                (path.name, path)
                for path in sorted(subset_root.iterdir(), key=self._fraction_from_name)
                if path.is_dir()
            )

        reports: dict[str, dict[str, Any]] = {}
        frames : dict[str, pd.DataFrame]   = {}
        for name, root in view_roots:
            labels        = pd.read_csv(root / "labels.csv")
            reports[name] = self._audit_view(name, root, labels, catalog)
            frames[name]  = labels

        # Learning-curve views are comparable only when evaluation membership is invariant and
        # successively larger training views include every member of the smaller view.
        full = frames["complete"]
        full_evaluation = {
            split: set(full.loc[full["split"] == split, "base_identifier"].astype(str))
            for split in ("val", "test")
        }
        previous_training: set[str] = set()
        for name, _ in view_roots[1:]:
            frame    = frames[name]
            training = set(frame.loc[frame["split"] == "train", "base_identifier"].astype(str))
            for split in ("val", "test"):
                observed = set(
                    frame.loc[frame["split"] == split, "base_identifier"].astype(str)
                )
                self._add_check(
                    reports[name],
                    f"fixed_{split}_membership",
                    observed == full_evaluation[split],
                    f"Every learning-curve view uses the complete fixed {split} set.",
                    {"observed": len(observed), "expected": len(full_evaluation[split])},
                )
            self._add_check(
                reports[name],
                "nested_training_membership",
                previous_training <= training,
                "A smaller training view is a subset of every larger training view.",
                {"previous": len(previous_training), "current": len(training)},
            )
            previous_training = training

        for report in reports.values():
            report["status"] = self._status(report)

        complete = reports["complete"]
        complete["dataset_views"] = {
            name: {
                "status": report["status"],
                "members": report["member_count"],
                "training_members": report["split_statistics"]["train"]["members"],
            }
            for name, report in reports.items()
        }
        if publish:
            # Write only after cross-view checks and statuses are attached, so every report is
            # self-contained and the complete view can summarize all learning-curve views.
            for name, root in view_roots:
                report = reports[name]
                self._write_json(root / "audit.json", report)
                self._write_markdown(root / "audit.md", report)
                self._write_statistics(root / "statistics.csv", report)
                self._plot(root / "distributions.png", frames[name], catalog)
            self._write_guide(selection_root / "README.md", complete)
        return complete

    def _audit_view(
        self,
        name   : str,
        root   : Path,
        labels : pd.DataFrame,
        catalog: pd.DataFrame,
    ) -> dict[str, Any]:
        """Compute checks and descriptive statistics for one membership view.

        Args:
            name: Human-readable view name, either ``complete`` or a percentage directory name.
            root: Directory containing the view's compact membership files.
            labels: Parsed ``labels.csv`` table for this view.
            catalog: Complete rich catalog used to recover scientific metadata.

        Returns:
            JSON-compatible report without its final aggregate status.

        Raises:
            FileNotFoundError: If a required view file is missing.
            ValueError: If columns, identifiers, labels, or JSON records are inconsistent.
        """
        required_label_columns = {
            "base_identifier",
            "label",
            "split",
            "tier",
            "sequence_cluster_id",
        }
        missing_columns = required_label_columns - set(labels.columns)
        if missing_columns:
            raise ValueError(f"{root / 'labels.csv'} lacks columns {sorted(missing_columns)}")

        identifiers = labels["base_identifier"].astype(str)
        unknown     = sorted(set(identifiers) - set(catalog["base_identifier"].astype(str)))
        if unknown:
            raise ValueError(f"{name} references unknown catalog identifiers: {unknown[:5]}")
        selected = catalog.set_index("base_identifier", drop=False).loc[identifiers].reset_index(
            drop=True
        )
        selected["split"] = labels["split"].astype(str).to_numpy()
        selected["label"] = labels["label"].astype(int).to_numpy()

        report: dict[str, Any] = {
            "schema_version": "1.0",
            "view": name,
            "member_count": len(selected),
            "checks": [],
            "warnings": [],
            "limitations": [
                "A 30%-identity MMseqs2 cluster is a sequence-family proxy, not a standardized "
                "molecular-function category. The catalog has no ontology-complete function label, "
                "so functional-type coverage cannot be claimed from clustering alone."
            ],
        }

        self._add_check(
            report,
            "unique_identifiers",
            not identifiers.duplicated().any(),
            "A protein identifier occurs at most once in this view.",
            {"duplicates": int(identifiers.duplicated().sum())},
        )
        if name != "complete":
            view_catalog = pd.read_csv(root / "catalog.csv")
            catalog_ids  = set(view_catalog["base_identifier"].astype(str))
            self._add_check(
                report,
                "catalog_csv_agreement",
                catalog_ids == set(identifiers) and len(view_catalog) == len(labels),
                "The diluted catalog contains exactly the view members and no parent-only rows.",
                {"observed": len(catalog_ids), "expected": len(labels)},
            )
        self._check_membership_files(report, root, labels, name == "complete")
        self._check_balance(report, selected)
        self._check_leakage(report, selected)
        self._check_external_test(report, selected)

        report["split_statistics"] = self._split_statistics(selected)
        report["numeric_distributions"] = self._numeric_distributions(selected)
        report["source_by_label"] = self._source_by_label(selected)

        source_sets = {
            label: set(
                selected.loc[selected["label"] == label, "source_database"].fillna("unknown")
            )
            for label in (0, 1)
        }
        if source_sets[0].isdisjoint(source_sets[1]):
            report["warnings"].append(
                {
                    "name": "source_label_confounding",
                    "explanation": (
                        "Positive and negative proteins come from disjoint source datasets. Class "
                        "balance is exact, but a model could exploit source-specific structural or "
                        "curation biases instead of DNA-binding biology. This is a scientific "
                        "limitation, not split leakage."
                    ),
                    "observed": {
                        "negative_sources": sorted(source_sets[0]),
                        "positive_sources": sorted(source_sets[1]),
                    },
                }
            )
        return report

    @staticmethod
    def _require_columns(catalog: pd.DataFrame) -> None:
        """Require every field needed for safety and composition analysis.

        Args:
            catalog: Parsed complete selection catalog.

        Raises:
            ValueError: If one or more mandatory scientific fields are absent.
        """
        required = {
            "base_identifier",
            "label",
            "split",
            "tier",
            "pdb_id",
            "published_partition",
            "sequence_cluster_id",
            "sequence_sha256",
            "protein_structure_sha256",
            "source_database",
            "sequence_length",
            "resolution_angstrom",
            "aspect_ratio",
            "sequence_coverage",
        }
        missing = required - set(catalog.columns)
        if missing:
            raise ValueError(f"catalog.csv lacks mandatory audit columns: {sorted(missing)}")

    def _check_membership_files(
        self,
        report  : dict[str, Any],
        root    : Path,
        labels  : pd.DataFrame,
        complete: bool,
    ) -> None:
        """Check that TXT, CSV, and JSON views describe exactly the same members.

        Args:
            report: Mutable view report receiving individual checks.
            root: View directory containing membership files.
            labels: Authoritative compact label table for the view.
            complete: Whether reserve lists are required in addition to main split lists.

        Raises:
            FileNotFoundError: If a required text or JSON membership file is missing.
            ValueError: If ``identifiers.json`` has no records list.
        """
        expected_all = set(labels["base_identifier"].astype(str))
        proteins     = self._read_identifiers(root / "proteins.txt")
        self._add_check(
            report,
            "proteins_txt_agreement",
            proteins == expected_all,
            "proteins.txt is the exact union that this view sends to preprocessing.",
            {"observed": len(proteins), "expected": len(expected_all)},
        )

        payload = json.loads((root / "identifiers.json").read_text(encoding="utf-8"))
        records = payload.get("records") if isinstance(payload, Mapping) else None
        if not isinstance(records, list):
            raise ValueError(f"{root / 'identifiers.json'} has no records list")
        json_ids = {
            str(record["identifier"])
            for record in records
            if isinstance(record, Mapping) and record.get("identifier")
        }
        self._add_check(
            report,
            "identifiers_json_agreement",
            json_ids == expected_all and len(json_ids) == len(records),
            "identifiers.json contains one machine-readable record per labels.csv row.",
            {"observed": len(json_ids), "expected": len(expected_all)},
        )

        splits = (*self.MAIN_SPLITS, *self.RESERVE_SPLITS) if complete else self.MAIN_SPLITS
        for split in splits:
            observed = self._read_identifiers(root / f"{split}.txt")
            expected = set(
                labels.loc[labels["split"] == split, "base_identifier"].astype(str)
            )
            self._add_check(
                report,
                f"{split}_txt_agreement",
                observed == expected,
                f"{split}.txt agrees exactly with labels.csv membership.",
                {"observed": len(observed), "expected": len(expected)},
            )

    def _check_balance(self, report: dict[str, Any], selected: pd.DataFrame) -> None:
        """Require non-empty exact binary parity in train, validation, and test.

        Args:
            report: Mutable view report receiving one check per main split.
            selected: Rich catalog rows selected by this view.
        """
        for split in self.MAIN_SPLITS:
            counts = Counter(selected.loc[selected["split"] == split, "label"].astype(int))
            valid  = counts[0] > 0 and counts[0] == counts[1]
            self._add_check(
                report,
                f"{split}_class_balance",
                valid,
                (
                    "Exact 1:1 parity prevents the majority class from dominating accuracy or "
                    "the binary loss in this partition."
                ),
                {"negative": counts[0], "positive": counts[1], "ratio": self._ratio(counts)},
            )

    def _check_leakage(self, report: dict[str, Any], selected: pd.DataFrame) -> None:
        """Count identities owned by more than one split at five safety levels.

        Args:
            report: Mutable view report receiving leakage checks.
            selected: Rich catalog rows selected by this view.
        """
        for field, meaning in self.LEAKAGE_KEYS:
            owners = selected.groupby(field, dropna=False)["split"].nunique()
            leaked = owners[owners > 1]
            self._add_check(
                report,
                f"no_{field}_leakage",
                leaked.empty,
                (
                    f"No {meaning} may occur in more than one split; otherwise evaluation can "
                    "contain information already represented during training or model selection."
                ),
                {
                    "leaked_groups": len(leaked),
                    "examples": [str(x) for x in leaked.index[:5]],
                },
            )

    def _check_external_test(self, report: dict[str, Any], selected: pd.DataFrame) -> None:
        """Ensure source-protected external-test rows never enter development.

        Args:
            report: Mutable view report receiving the provenance check.
            selected: Rich catalog rows selected by this view.
        """
        invalid = selected[
            (selected["published_partition"] == "external_test")
            & (~selected["split"].isin(("test", "test_reserve")))
        ]
        self._add_check(
            report,
            "external_test_boundary",
            invalid.empty,
            "Proteins published as external test by a source never enter train or validation.",
            {"violations": len(invalid)},
        )

    def _split_statistics(self, selected: pd.DataFrame) -> dict[str, Any]:
        """Describe class, family, tier, taxonomy, and method coverage per split.

        Args:
            selected: Rich catalog rows selected by one view.

        Returns:
            Ordered per-split counts with an explicit interpretation of family coverage.
        """
        output: dict[str, Any] = {}
        for split in (*self.MAIN_SPLITS, *self.RESERVE_SPLITS):
            frame = selected[selected["split"] == split]
            if frame.empty:
                continue
            label_statistics: dict[str, Any] = {}
            for label, label_name in ((0, "negative"), (1, "positive")):
                stratum  = frame[frame["label"] == label]
                members  = len(stratum)
                clusters = stratum["sequence_cluster_id"].nunique()
                label_statistics[label_name] = {
                    "members": members,
                    "sequence_clusters": clusters,
                    "cluster_coverage_ratio": clusters / members if members else None,
                    "tiers": self._counts(stratum["tier"]),
                    "largest_cluster_members": int(
                        stratum.groupby("sequence_cluster_id").size().max()
                    )
                    if members
                    else 0,
                }
            output[split] = {
                "members": len(frame),
                "negative": int((frame["label"] == 0).sum()),
                "positive": int((frame["label"] == 1).sum()),
                "labels": label_statistics,
                "tiers": self._counts(frame["tier"]),
                "unique_taxa": int(frame["taxonomy_id"].nunique())
                if "taxonomy_id" in frame
                else None,
                "structure_methods": self._counts(frame["structure_method"])
                if "structure_method" in frame
                else {},
            }
        return output

    @staticmethod
    def _numeric_distributions(selected: pd.DataFrame) -> dict[str, Any]:
        """Summarize interpretable structural covariates with robust quantiles.

        Args:
            selected: Rich catalog rows selected by one view.

        Returns:
            Count, median, interquartile range, minimum, and maximum by split and label. Sequence
            length is measured in residues, resolution in ångströms, aspect ratio is dimensionless,
            and sequence coverage is a unit fraction.
        """
        fields = (
            "sequence_length",
            "resolution_angstrom",
            "aspect_ratio",
            "sequence_coverage",
        )
        output: dict[str, Any] = {}
        for split in DNASelectionAudit.MAIN_SPLITS:
            output[split] = {}
            for label, label_name in ((0, "negative"), (1, "positive")):
                frame = selected[(selected["split"] == split) & (selected["label"] == label)]
                output[split][label_name] = {}
                for field in fields:
                    values = pd.to_numeric(frame[field], errors="coerce").dropna()
                    output[split][label_name][field] = {
                        "count": len(values),
                        "minimum": DNASelectionAudit._finite(values.min()),
                        "q1": DNASelectionAudit._finite(values.quantile(0.25)),
                        "median": DNASelectionAudit._finite(values.median()),
                        "q3": DNASelectionAudit._finite(values.quantile(0.75)),
                        "maximum": DNASelectionAudit._finite(values.max()),
                    }
        return output

    @staticmethod
    def _source_by_label(selected: pd.DataFrame) -> dict[str, dict[str, int]]:
        """Count public source databases independently for each binary label.

        Args:
            selected: Rich catalog rows selected by one view.

        Returns:
            Source counts keyed by ``negative`` and ``positive``.
        """
        return {
            name: DNASelectionAudit._counts(
                selected.loc[selected["label"] == label, "source_database"]
            )
            for label, name in ((0, "negative"), (1, "positive"))
        }

    @staticmethod
    def _add_check(
        report     : dict[str, Any],
        name       : str,
        passed     : bool,
        explanation: str,
        observed   : Mapping[str, Any],
    ) -> None:
        """Append one explicit pass/fail invariant to a mutable report.

        Args:
            report: Mutable report containing a ``checks`` list.
            name: Stable machine-readable check name.
            passed: Whether the observed data satisfy the invariant.
            explanation: Plain-language meaning and scientific reason for the check.
            observed: JSON-compatible measurements supporting the verdict.
        """
        report["checks"].append(
            {
                "name": name,
                "status": "PASS" if passed else "FAIL",
                "explanation": explanation,
                "observed": dict(observed),
            }
        )

    @staticmethod
    def _status(report: Mapping[str, Any]) -> str:
        """Reduce detailed checks and warnings to one conservative verdict.

        Args:
            report: View report containing checks and scientific warnings.

        Returns:
            ``FAIL`` for any hard invariant failure, ``PASS_WITH_WARNINGS`` for valid data with a
            documented scientific limitation, or ``PASS`` otherwise.
        """
        if any(check["status"] == "FAIL" for check in report["checks"]):
            return "FAIL"
        return "PASS_WITH_WARNINGS" if report["warnings"] else "PASS"

    @staticmethod
    def _write_guide(path: Path, report: Mapping[str, Any]) -> None:
        """Write a concise map of selection files and their scientific roles.

        Args:
            path: Destination selection-level ``README.md`` path.
            report: Complete audit supplying current split sizes and verdict.

        Raises:
            OSError: If the guide cannot be atomically published.
        """
        splits = report["split_statistics"]
        lines  = [
            "# WISDOM-DNA selection",
            "",
            f"Quality verdict: **{report['status']}**. The complete portable selection contains "
            f"{report['member_count']} proteins including local-evaluation reserves.",
            "",
            "## Which files are needed?",
            "",
            "- `catalog.csv` is the authoritative human-readable scientific table. One row stores "
            "the label, split, external sequence family, structure choice, evidence, quality "
            "measurements, and provenance for one protein chain.",
            "- `catalog.parquet` contains the same table with typed columns for faster analysis. "
            "It is an analytical mirror, not a second source of truth.",
            "- `identifiers.json` is the compact machine-readable membership contract. It joins "
            "each identifier to its label, split, geometric tier, sequence family, and whether it "
            "is a normal dataset member or a reserve.",
            "- `labels.csv` is a smaller spreadsheet-friendly projection of the same membership. "
            "It is convenient for inspection and does not replace `catalog.csv`.",
            "- `train.txt`, `val.txt`, and `test.txt` contain only the identifiers in the three "
            "model partitions. Their current sizes are "
            f"{splits['train']['members']}, {splits['val']['members']}, and "
            f"{splits['test']['members']} respectively.",
            "- `validation_reserve.txt` and `test_reserve.txt` contain positive proteins held "
            "outside ordinary training and evaluation. They may replace a same-partition positive "
            "whose local surface annotation cannot be evaluated. Reserves are intentionally not "
            "class-balanced because they are spare localization examples, not model splits.",
            "- `proteins.txt` is the union of main and reserve identifiers. Structural "
            "preprocessing reads this union once so a reserve is ready if annotation needs it.",
            "- `audit.json`, `audit.md`, `statistics.csv`, and `distributions.png` are the machine "
            "verdict, explained report, tidy statistics, and diagnostic figure.",
            "",
            "## Diluted training views",
            "",
            "Each `subsets/<percentage>/` directory is a self-contained membership view with its "
            "own filtered `catalog.csv`, TXT, compact CSV, JSON, Markdown, and figure. The "
            "percentage applies only to balanced training membership. Validation and test are "
            "identical in every view, which makes learning-curve comparisons fair. Training "
            "selections are nested and visit distinct 30%-identity sequence families before "
            "taking repeated family members.",
            "",
            "Sequence clustering is a leakage barrier and a family-diversity mechanism. It does "
            "not establish coverage of every biochemical protein function; read `audit.md` for "
            "the measured breadth and the remaining source/label confounding warning.",
            "",
        ]
        DNASelectionAudit._atomic_text(path, "\n".join(lines))

    @staticmethod
    def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
        """Atomically write one deterministic, finite JSON report.

        Args:
            path: Destination ``audit.json`` path.
            payload: JSON-compatible audit mapping.

        Raises:
            OSError: If the temporary file cannot be synchronized or replaced.
            ValueError: If the payload contains non-finite JSON numbers.
        """
        DNASelectionAudit._atomic_text(
            path,
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    @staticmethod
    def _write_markdown(path: Path, report: Mapping[str, Any]) -> None:
        """Write a self-contained narrative explanation of one audit.

        Args:
            path: Destination ``audit.md`` path.
            report: Complete view report with checks, statistics, warnings, and limitations.

        Raises:
            OSError: If the report cannot be atomically published.
        """
        lines = [
            f"# WISDOM-DNA selection audit: {report['view']}",
            "",
            f"**Verdict:** `{report['status']}`. **Members:** {report['member_count']}.",
            "",
            "A failed check means this view must not be used for training or evaluation. A warning "
            "does not indicate file corruption, but it identifies a scientific bias that must be "
            "considered when interpreting model performance.",
            "",
            "## Safety checks",
            "",
            "| Check | Result | Meaning | Observed |",
            "|---|---:|---|---|",
        ]
        for check in report["checks"]:
            observed = json.dumps(check["observed"], sort_keys=True).replace("|", "\\|")
            lines.append(
                f"| `{check['name']}` | **{check['status']}** | {check['explanation']} | "
                f"`{observed}` |"
            )

        lines.extend(
            [
                "",
                "## Composition and sequence-family diversity",
                "",
                "`cluster coverage ratio = distinct 30%-identity clusters / proteins`. A value "
                "near 1 means nearly every protein belongs to a different broad sequence family, "
                "which is desirable for breadth and reduces domination by repeated homologues. It "
                "does not prove coverage of every biochemical DNA-binding mechanism.",
                "",
                "| Split | Negative | Positive | Negative families | Positive families | Tiers |",
                "|---|---:|---:|---:|---:|---|",
            ]
        )
        for split, values in report["split_statistics"].items():
            negative = values["labels"]["negative"]
            positive = values["labels"]["positive"]
            tiers    = json.dumps(values["tiers"], sort_keys=True).replace("|", "\\|")
            lines.append(
                f"| {split} | {values['negative']} | {values['positive']} | "
                f"{negative['sequence_clusters']}/{negative['members']} | "
                f"{positive['sequence_clusters']}/{positive['members']} | `{tiers}` |"
            )

        family_ratios = [
            values["labels"][label]["cluster_coverage_ratio"]
            for split, values in report["split_statistics"].items()
            if split in DNASelectionAudit.MAIN_SPLITS
            for label in ("negative", "positive")
            if values["labels"][label]["cluster_coverage_ratio"] is not None
        ]
        minimum_family_ratio = min(family_ratios) if family_ratios else 0.0
        tier_breadth = all(
            {"core", "challenge"} <= set(values["labels"][label]["tiers"])
            for split, values in report["split_statistics"].items()
            if split in DNASelectionAudit.MAIN_SPLITS
            for label in ("negative", "positive")
        )
        tier_interpretation = (
            "Both labels contain core and challenge geometries in every main split."
            if tier_breadth
            else "At least one split/label stratum does not contain both geometric tiers."
        )
        lines.extend(
            [
                "",
                f"The lowest main-split family coverage is {minimum_family_ratio:.1%}. This is "
                "high: repeated close-family membership is uncommon, and no one broad sequence "
                f"family dominates a class. {tier_interpretation} These are useful diversity "
                "signals, but they remain weaker than a curated functional ontology.",
            ]
        )

        lines.extend(["", "## Scientific warnings and limitations", ""])
        if report["warnings"]:
            for warning in report["warnings"]:
                lines.append(f"- **{warning['name']}:** {warning['explanation']}")
        else:
            lines.append("- No additional warning was detected by the implemented controls.")
        for limitation in report["limitations"]:
            lines.append(f"- **Scope limitation:** {limitation}")
        lines.extend(
            [
                "",
                "## Reading the numeric summaries",
                "",
                "`statistics.csv` and `audit.json` report the median (the middle observation), Q1 "
                "(25% of values are lower), and Q3 (75% are lower). The Q1-Q3 interval describes "
                "the central half without being dominated by a few extreme proteins. Sequence "
                "length is in residues; experimental resolution is in ångströms, where a smaller "
                "value generally means finer structural detail; aspect ratio measures elongation; "
                "and sequence coverage is the observed fraction of the source sequence.",
                "",
                "`distributions.png` visualizes class parity, sequence-family breadth, sequence "
                "length, and geometric-tier representation. These plots diagnose dataset "
                "composition; they do not by themselves demonstrate model quality.",
                "",
            ]
        )
        DNASelectionAudit._atomic_text(path, "\n".join(lines))

    @staticmethod
    def _write_statistics(path: Path, report: Mapping[str, Any]) -> None:
        """Write a tidy CSV mirror of split and numeric statistics.

        Args:
            path: Destination ``statistics.csv`` path.
            report: View report containing split and numeric statistics.

        Raises:
            OSError: If the temporary CSV cannot be synchronized or replaced.
        """
        rows: list[dict[str, Any]] = []
        for split, values in report["split_statistics"].items():
            for label_name in ("negative", "positive"):
                label = values["labels"][label_name]
                for metric in ("members", "sequence_clusters", "cluster_coverage_ratio"):
                    rows.append(
                        {
                            "split": split,
                            "label": label_name,
                            "variable": "sequence_family",
                            "statistic": metric,
                            "value": label[metric],
                        }
                    )
        for split, labels in report["numeric_distributions"].items():
            for label_name, fields in labels.items():
                for variable, statistics in fields.items():
                    for statistic, value in statistics.items():
                        rows.append(
                            {
                                "split": split,
                                "label": label_name,
                                "variable": variable,
                                "statistic": statistic,
                                "value": value,
                            }
                        )

        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8", newline="") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("split", "label", "variable", "statistic", "value"),
                )
                writer.writeheader()
                writer.writerows(rows)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _plot(path: Path, labels: pd.DataFrame, catalog: pd.DataFrame) -> None:
        """Plot four composition diagnostics for one complete or diluted view.

        Args:
            path: Destination PNG path.
            labels: Compact view membership and labels.
            catalog: Rich complete catalog used for numeric and tier values.

        Raises:
            OSError: If Matplotlib cannot publish the figure.
        """
        selected = catalog.set_index("base_identifier", drop=False).loc[
            labels["base_identifier"].astype(str)
        ].reset_index(drop=True)
        selected["split"] = labels["split"].astype(str).to_numpy()
        selected["label"] = labels["label"].astype(int).to_numpy()

        figure, axes = plt.subplots(2, 2, figsize=(12.0, 8.5), constrained_layout=True)
        positions = range(len(DNASelectionAudit.MAIN_SPLITS))
        negative  = [
            int(((selected["split"] == split) & (selected["label"] == 0)).sum())
            for split in DNASelectionAudit.MAIN_SPLITS
        ]
        positive = [
            int(((selected["split"] == split) & (selected["label"] == 1)).sum())
            for split in DNASelectionAudit.MAIN_SPLITS
        ]
        axes[0, 0].bar([x - 0.2 for x in positions], negative, width=0.4, label="negative")
        axes[0, 0].bar([x + 0.2 for x in positions], positive, width=0.4, label="positive")
        axes[0, 0].set_xticks(list(positions), DNASelectionAudit.MAIN_SPLITS)
        axes[0, 0].set_title("Exact class balance")
        axes[0, 0].set_ylabel("Proteins")
        axes[0, 0].legend()

        family_labels: list[str]   = []
        family_ratios: list[float] = []
        for split in DNASelectionAudit.MAIN_SPLITS:
            for label, short in ((0, "-"), (1, "+")):
                frame = selected[(selected["split"] == split) & (selected["label"] == label)]
                family_labels.append(f"{split}{short}")
                family_ratios.append(
                    frame["sequence_cluster_id"].nunique() / len(frame) if len(frame) else 0.0
                )
        axes[0, 1].bar(family_labels, family_ratios)
        axes[0, 1].set_ylim(0.0, 1.05)
        axes[0, 1].set_title("Sequence-family coverage")
        axes[0, 1].set_ylabel("Distinct clusters / proteins")

        box_values: list[pd.Series[Any]] = []
        box_labels: list[str]            = []
        for split in DNASelectionAudit.MAIN_SPLITS:
            for label, short in ((0, "-"), (1, "+")):
                values = pd.to_numeric(
                    selected.loc[
                        (selected["split"] == split) & (selected["label"] == label),
                        "sequence_length",
                    ],
                    errors="coerce",
                ).dropna()
                box_values.append(values)
                box_labels.append(f"{split}{short}")
        axes[1, 0].boxplot(box_values, tick_labels=box_labels, showfliers=False)
        axes[1, 0].set_title("Sequence-length distribution")
        axes[1, 0].set_ylabel("Residues")

        main = selected[selected["split"].isin(DNASelectionAudit.MAIN_SPLITS)]
        tier_table = pd.crosstab(main["split"], main["tier"]).reindex(
            DNASelectionAudit.MAIN_SPLITS,
            fill_value=0,
        )
        tier_table.plot.bar(stacked=True, ax=axes[1, 1], legend=True)
        axes[1, 1].set_title("Geometric tiers")
        axes[1, 1].set_xlabel("")
        axes[1, 1].set_ylabel("Proteins")
        axes[1, 1].tick_params(axis="x", rotation=0)

        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp.png")
        try:
            figure.savefig(temporary, dpi=160)
            os.replace(temporary, path)
        finally:
            plt.close(figure)
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _read_identifiers(path: Path) -> set[str]:
        """Read a strict non-empty unique identifier TXT file.

        Args:
            path: Text file containing one protein identifier per line.

        Returns:
            Unique non-empty identifiers.

        Raises:
            FileNotFoundError: If the membership file does not exist.
            ValueError: If an identifier occurs more than once.
        """
        values = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
        values = [value for value in values if value]
        if len(values) != len(set(values)):
            raise ValueError(f"duplicate identifiers in {path}")
        return set(values)

    @staticmethod
    def _counts(values: pd.Series[Any]) -> dict[str, int]:
        """Convert a pandas categorical count into deterministic JSON integers.

        Args:
            values: Possibly nullable categorical observations.

        Returns:
            Alphabetically ordered string-to-count mapping.
        """
        counts = values.fillna("unknown").astype(str).value_counts()
        return {str(key): int(counts[key]) for key in sorted(counts.index)}

    @staticmethod
    def _ratio(counts: Counter[int]) -> float | None:
        """Return positive-to-negative ratio when the denominator exists.

        Args:
            counts: Binary label counts keyed by zero and one.

        Returns:
            ``positive / negative`` or ``None`` when no negative exists.
        """
        return counts[1] / counts[0] if counts[0] else None

    @staticmethod
    def _finite(value: Any) -> float | None:
        """Convert one pandas scalar to finite JSON or ``None``.

        Args:
            value: Numeric pandas result that may be NaN.

        Returns:
            Native finite float or ``None`` for a missing/non-finite value.
        """
        return float(value) if pd.notna(value) else None

    @staticmethod
    def _fraction_from_name(path: Path) -> float:
        """Parse a deterministic percentage directory for numerical ordering.

        Args:
            path: Directory named like ``10pct`` or ``12p5pct``.

        Returns:
            Percentage value used only to sort views.

        Raises:
            ValueError: If the directory does not follow the percentage naming contract.
        """
        name = path.name
        if not name.endswith("pct"):
            raise ValueError(f"invalid dilution directory name: {name}")
        return float(name[:-3].replace("p", "."))

    @staticmethod
    def _atomic_text(path: Path, text: str) -> None:
        """Atomically publish synchronized UTF-8 text.

        Args:
            path: Final report path.
            text: Complete textual content.

        Raises:
            OSError: If writing, synchronization, or replacement fails.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("w", encoding="utf-8") as stream:
                stream.write(text if text.endswith("\n") else text + "\n")
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
