"""Readable evidence-to-design orchestration for the zinc-binding benchmark."""

import csv
import json
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections.abc import Sequence
from wisdom.preprocessing.common.audit import audit_dataset
from wisdom.preprocessing.zinc.evidence import load_evidence
from wisdom.preprocessing.common.splits import assign_splits
from wisdom.preprocessing.zinc.evidence import annotate_sites
from wisdom.preprocessing.common.shortcut import shortcut_baseline
from wisdom.preprocessing.zinc.audit import write_diversity_report
from wisdom.preprocessing.zinc.phenotypes import assign_phenotypes
from wisdom.preprocessing.common.dilutions import create_dilutions
from wisdom.preprocessing.zinc.discovery import discover_candidates
from wisdom.preprocessing.zinc.structures import analyse_structures
from wisdom.preprocessing.common.population import select_population
from wisdom.preprocessing.common.leakage import assign_leakage_groups
from wisdom.preprocessing.common.similarity import compute_similarity
from wisdom.preprocessing.common.snapshots import snapshot_structures
from wisdom.preprocessing.common.functional_metadata import annotate_functions


class ZincSelection(lf.Work):
    """Create immutable Zn design evidence using shared leakage and group-split algorithms."""

    def run(
        self,
        skip: bool = False,
        existing_design: Path | None = None,
        raw_path: Path | None = None,
        negative_evidence: Path | None = None,
        release_id: str = "zinc-pilot-1",
        maximum_entries: int = 0,
        raw_output_directory: str | None = None,
        negative_go_terms: Sequence[str] = ("GO:0008270", "GO:0046872"),
        workers: int = 8,
        coordination_cutoff: float = 3.0,
        minimum_occupancy: float = 0.5,
        minimum_donors: int = 2,
        minimum_residues: int = 2,
        maximum_resolution: float | None = 4.0,
        requests_per_second: float = 2.0,
        sequence_identity: float = 0.30,
        sequence_coverage: float = 0.80,
        sequence_evalue: float = 1e-3,
        foldseek_probability: float = 0.90,
        foldseek_tmscore: float = 0.75,
        foldseek_coverage: float = 0.80,
        foldseek_evalue: float = 1e-3,
        group_same_pdb: bool = True,
        phenotype_cluster_sizes: Sequence[int] = (5, 10, 15),
        phenotype_min_samples: int = 2,
        phenotype_stability: float = 0.7,
        positive_negative_ratio: float = 1.0,
        positive_policy: str = "diverse_quota",
        surface_resolution: float = 1.0,
        probe_radius: float = 1.4,
        positive_gap: float = 1.4,
        functional_metadata: Path | None = None,
        site_metadata: Path | None = None,
        evaluate_shortcuts: bool = False,
        train_fraction: float = 0.70,
        validation_fraction: float = 0.15,
        test_fraction: float = 0.15,
        dilution_fractions: Sequence[float] = (1.0, 0.75, 0.5, 0.25, 0.1),
        dilution_replicates: int = 1,
        seed: int = 2026,
        output_directory: str | None = "../data/zinc/design",
    ) -> dict[str, Any]:
        """Verify coordination, group the full RAW population, then select and split once.

        Args:
            skip: Forward existing_design without tools, downloads or scientific computation.
            existing_design: Complete portable design directory, required when skipped.
            raw_path: Optional frozen candidate JSONL; bypass acquisition when supplied.
            negative_evidence: Optional reviewed negative JSONL; null discovers experimental NOT.
            release_id: Frozen discovery query namespace, default zinc-pilot-1.
            maximum_entries: Sorted discovery pilot cap; zero inspects all query hits.
            raw_output_directory: Optional atomic acquisition copy; null keeps managed evidence.
            negative_go_terms: Exact denial terms when acquiring negatives, default Zn binding
                GO:0008270 and broader metal binding GO:0046872. Reviewed ion/cation ancestors
                GO:0043167/GO:0043169 are also accepted; never a narrower or unrelated term.
            workers: Bounded download and native tool threads, default 8.
            coordination_cutoff: Maximum donor-Zn center separation in ångströms, default 3.
            minimum_occupancy: Minimum occupied Zn/donor fraction, default 0.5.
            minimum_donors: At least this many selected-copy N/O/S atoms, default 2.
            minimum_residues: At least this many selected-copy coordinating residues, default 2.
            maximum_resolution: Worst accepted resolution in ångströms, default 4 or None.
            requests_per_second: Native shared RCSB ceiling, default 2.
            sequence_identity: Minimum MMseqs2 identity, default 0.30.
            sequence_coverage: Minimum query and target coverage, default 0.80.
            sequence_evalue: Maximum sequence E-value, default 1e-3.
            foldseek_probability: Minimum structural match probability, default 0.90.
            foldseek_tmscore: Minimum query and target TM-score, default 0.75.
            foldseek_coverage: Minimum query and target structure coverage, default 0.80.
            foldseek_evalue: Maximum structural E-value, default 1e-3.
            group_same_pdb: Join deposition identities, default true.
            phenotype_cluster_sizes: Distinct HDBSCAN size grid, default 5/10/15.
            phenotype_min_samples: HDBSCAN core-neighbor count, default 2.
            phenotype_stability: Minimum cross-grid ARI for >=2 clusters, default 0.7.
            positive_negative_ratio: Positive/negative canonical count ratio, default 1.
            positive_policy: diverse_quota (default) samples positives; all retains every
                eligible positive and reports the actual ratio instead of enforcing balance.
            surface_resolution: Early deterministic boundary spacing, default 1 Å.
            probe_radius: Early solvent expansion, default 1.4 Å.
            positive_gap: Early positive coordinating-atom surface gap, default 1.4 Å.
            functional_metadata: Optional frozen JSONL annotations bound to identifier and
                sequence_sha256, with source/version provenance. Used only for audit/selection.
            evaluate_shortcuts: Fit a small train-only covariate baseline; default false.
            site_metadata: Optional frozen metal-database JSONL bound to exact structure/site;
                enriches audit evidence, never coordination acceptance or labels. Default null.
            train_fraction: Group-wise training fraction, default 0.70.
            validation_fraction: Fixed validation fraction, default 0.15.
            test_fraction: Fixed test fraction, default 0.15; fractions must sum to one.
            dilution_fractions: Nested train-only fractions including 1, default 1/.75/.5/.25/.1.
            dilution_replicates: Deterministic dilution orderings, default 1.
            seed: SHA-based tie-breaking seed, default 2026.
            output_directory: Optional native published design copy, default ../data/zinc/design.

        Returns:
            Native design output path and full scientific audit. Structural positives alone do
            not establish physiological Zn specificity; confidence remains a separate field.

        Raises:
            RuntimeError: Missing specialist tool, insufficient explicit negatives, impossible
                group splits, or leakage audit failure. No fallback or invented negatives exist.
            ValueError: Invalid evidence, assembly/copy, sequence, or scientific thresholds.
        """
        # A frozen design already contains the scientific decision and exact coordinate snapshot.
        # Forward its native artifact without acquiring evidence or recomputing a single split.

        if skip:
            if existing_design is None:
                raise ValueError("existing_design is required when Zn Selection is skipped")
            self.outputs.artifact("zinc-design", existing_design, role="dataset-design")
            self.log("Zn selection skipped; forwarding the frozen design and structure snapshot")
            return {"skipped": True, "design": str(existing_design)}

        # 1. Resolve specialist binaries before expensive work. Then read explicit evidence:
        # an RCSB Zn hit is only a candidate; negative evidence is acquired or explicitly supplied.

        try:
            mmseqs = self.tools.require("mmseqs", version_args=("version",))
            foldseek = self.tools.require("foldseek", version_args=("version",))
        except Exception as error:
            raise RuntimeError("Zn selection needs mmseqs and foldseek; run ./install.sh locally "
                               "or lf clusters bootstrap <cluster> --project .") from error
        if raw_path is None:
            acquisition = discover_candidates(
                self, negative_evidence, release_id, workers, maximum_entries,
                requests_per_second, coordination_cutoff, minimum_occupancy,
                minimum_donors, minimum_residues, raw_output_directory, negative_go_terms,
            )
            raw_path = acquisition["raw_path"]
        rows = load_evidence(raw_path)
        if not any(int(row["label"]) == 0 for row in rows):
            raise RuntimeError("no explicit negative Zn evidence; benchmark construction stopped")

        # 2. Validate biological assemblies and occupied Zn coordination. Keep rejected but
        # parseable candidates in RAW so homologous bridges cannot disappear before grouping.

        self.log(f"Verifying {len(rows)} Zn candidates with {workers} workers")
        rows = analyse_structures(self, rows, workers, coordination_cutoff, minimum_occupancy,
                                  minimum_donors, minimum_residues, maximum_resolution,
                                  requests_per_second, surface_resolution, probe_radius,
                                  positive_gap)
        if functional_metadata is not None:
            rows = annotate_functions(rows, functional_metadata)
        if site_metadata is not None:
            rows = annotate_sites(rows, site_metadata)
        sequence_labels: dict[str, set[int]] = {}
        for row in rows:
            sequence_labels.setdefault(row["sequence_sha256"], set()).add(int(row["label"]))
        for row in rows:
            if len(sequence_labels[row["sequence_sha256"]]) > 1:
                row.update(quality_eligible=False,
                           quality_exclusion_reason="contradictory_exact_sequence_evidence")

        # 3. Compute full-RAW MMseqs2/Foldseek and transitive leakage components. Physical
        # phenotypes are descriptive strata only: they never create similarity/leakage edges.

        similarity = compute_similarity(self, rows, mmseqs, foldseek, workers, sequence_identity,
                                        sequence_coverage, sequence_evalue, foldseek_probability,
                                        foldseek_tmscore, foldseek_coverage, foldseek_evalue)
        rows, leakage = assign_leakage_groups(rows, similarity, group_same_pdb)
        rows, phenotypes = assign_phenotypes(rows, phenotype_cluster_sizes, phenotype_min_samples,
                                             phenotype_stability, workers)

        # 4. Keep all reliable negatives and sample diverse positives. Assign entire leakage
        # groups to fixed splits, and only afterwards derive nested group-wise train dilutions.

        if positive_policy not in {"diverse_quota", "all"}:
            raise ValueError("Zn positive_policy must be diverse_quota or all")
        for row in rows:
            row["coordination_signature"] = "|".join(sorted(
                "/".join(f"{name}{count}" for name, count in
                         site["selected_residue_counts"].items())
                for site in row["sites"] if site["accepted"]))
        selected, selection = select_population(rows, positive_negative_ratio, True, False, seed,
            keep_all_positives=positive_policy == "all",
            additional_strata=("method", "family", "coordination_signature"))
        global_only = [row["identifier"] for row in selected
                       if int(row["label"]) == 1 and not row["local_gt_expected"]]
        selected, split_audit = assign_splits(selected, train_fraction, validation_fraction,
                                              test_fraction, seed, training_only_ids=global_only)
        dilutions = create_dilutions(selected, dilution_fractions, dilution_replicates, seed)
        audit = audit_dataset(rows, selected, dilutions, selection["positive_negative_ratio"])
        audit.update(phenotypes=phenotypes, selection=selection, split_assignment=split_audit,
                     task="zinc_binding", positive_definition="verified deposited coordination",
                     physiological_specificity_claimed=False)
        if evaluate_shortcuts:
            audit["shortcut_baseline"] = shortcut_baseline(selected, seed)

        # 5. Preserve exact coordinates and evidence. Human labelled TXT is convenient but the
        # complete JSONL retains assembly/copy, coordination and grouping for preprocessing.

        root = Path(self.outputs.directory("zinc-design", role="dataset-design",
                                            publish_to=output_directory, overwrite=True))
        snapshot_structures(root / "structures", selected)
        audit["diversity"] = write_diversity_report(root, rows, selected)
        portable = [{key: value for key, value in row.items()
                     if key not in ("source_structure", "foldseek_structure")} for row in selected]
        design = root / "selection.jsonl"
        design.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in portable))
        (root / "raw-audit.jsonl").write_text("".join(
            json.dumps({k: v for k, v in row.items()
                        if k not in ("source_structure", "foldseek_structure")}, sort_keys=True)
            + "\n" for row in rows))
        (root / "dilutions.json").write_text(json.dumps(dilutions, indent=2))
        (root / "audit.json").write_text(json.dumps(audit, indent=2))
        selected_ids = {row["identifier"] for row in selected}
        (root / "zinc-sites.jsonl").write_text("".join(
            json.dumps({"identifier": row["identifier"], "pdb_id": row["pdb_id"],
                        "assembly_id": row["assembly_id"], "protein_copy": row["protein_copy"],
                        "selected": row["identifier"] in selected_ids,
                        "structure_sha256": row["structure_sha256"], **site}, sort_keys=True) + "\n"
            for row in rows for site in row["sites"] if site["accepted"]
        ), encoding="utf-8")
        for name, values in (("sequence-pairs.tsv", similarity["sequence_path"]),
                              ("structure-pairs.tsv", similarity["structure_path"])):
            (root / name).write_bytes(Path(values).read_bytes())
        (root / "leakage.json").write_text(json.dumps(leakage, indent=2))
        for split in ("train", "validation", "test"):
            (root / f"{split}-labelled.txt").write_text("".join(
                f"{row['identifier']} {row['label']}\n"
                for row in portable if row["split"] == split))
        for replicate, subsets in dilutions["replicates"].items():
            folder = root / "dilutions" / replicate
            folder.mkdir(parents=True)
            by_id = {row["identifier"]: row for row in portable}
            for name, subset in subsets.items():
                (folder / f"{name}-labelled.txt").write_text("".join(
                    f"{identifier} {by_id[identifier]['label']}\n"
                    for identifier in subset["identifiers"]))
        with (root / "split-counts.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(("split", "positive", "negative", "leakage_groups"))
            for split, summary in audit["splits"].items():
                writer.writerow((split, summary["positive"], summary["negative"],
                                 summary["leakage_groups"]))
        (root / "report.md").write_text(
            "# Zn design audit\n\nPositive means verified deposited coordination, not proven "
            "physiological specificity. Negatives require explicit sequence-bound evidence.\n\n"
            "## Fixed splits\n\n|Split|Positive|Negative|Leakage groups|\n|---|---:|---:|---:|\n"
            + "".join(f"|{split}|{s['positive']}|{s['negative']}|{s['leakage_groups']}|\n"
                      for split, s in audit["splits"].items())
            + "\nA leakage group is an indivisible transitive similarity/identity component. "
            "Zero crossing groups is a hard prerequisite, not proof that all dependencies "
            "are known. Exact balancing can be limited by group sizes.\n\n"
            "## Physical phenotypes\n\nMedian/IQR scaling prevents large-unit variables from "
            "dominating. ARI measures agreement of cluster assignments across a parameter grid "
            "(1=identical; approximately 0=chance agreement). All-noise agreement is not evidence "
            "of meaningful clusters: at least two clusters are also required. Noise means no "
            "robust group assignment, not bad data. Positive site and negative morphology "
            "clusters are separate and have no functional or leakage meaning.\n\n"
            + "\n".join(f"- {name}: {json.dumps(result)}" for name, result in phenotypes.items())
            + "\n\nSee [diversity.md](diversity.md) for the class-count/group-tail plot, "
            "source and chemistry coverage, covariate SMD interpretation and warnings. "
            "audit.json also records omission reasons.\n")
        self.metrics.log("selected_members", len(selected))
        self.metrics.log("cross_split_leakage_groups", audit["cross_split_leakage_groups"])
        return {"design": str(root), "members": len(selected),
                "evidence_sha256": hashlib.sha256(raw_path.read_bytes()).hexdigest()}
