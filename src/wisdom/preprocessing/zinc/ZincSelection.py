"""Readable evidence-to-design orchestration for the zinc-binding benchmark."""

import csv
import json
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections.abc import Sequence
from wisdom.preprocessing.zinc.evidence import load_evidence
from wisdom.preprocessing.zinc.audit import write_diversity_report
from wisdom.preprocessing.zinc.phenotypes import assign_phenotypes
from wisdom.preprocessing.dna.selection.audit import audit_dataset
from wisdom.preprocessing.zinc.structures import analyse_structures
from wisdom.preprocessing.dna.selection.splits import assign_splits
from wisdom.preprocessing.dna.selection.dilutions import create_dilutions
from wisdom.preprocessing.dna.selection.population import select_population
from wisdom.preprocessing.dna.selection.leakage import assign_leakage_groups
from wisdom.preprocessing.dna.selection.similarity import compute_similarity
from wisdom.preprocessing.dna.selection.structures import snapshot_structures


class ZincSelection(lf.Work):
    """Create immutable Zn design evidence using shared leakage and group-split algorithms."""

    def run(
        self,
        raw_path: Path,
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
            raw_path: Frozen JSONL with positive candidates and sequence-bound explicit negatives.
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
        # 1. Resolve specialist binaries before expensive work. Then read explicit evidence:
        # an RCSB Zn hit is only a candidate; negative evidence is mandatory and sequence-bound.

        try:
            mmseqs = self.tools.require("mmseqs", version_args=("version",))
            foldseek = self.tools.require("foldseek", version_args=("version",))
        except Exception as error:
            raise RuntimeError("Zn selection needs mmseqs and foldseek; run ./install.sh locally "
                               "or lf clusters bootstrap <cluster> --project .") from error
        rows = load_evidence(raw_path)
        if not any(int(row["label"]) == 0 for row in rows):
            raise RuntimeError("no explicit negative Zn evidence; benchmark construction stopped")

        # 2. Validate biological assemblies and occupied Zn coordination. Keep rejected but
        # parseable candidates in RAW so homologous bridges cannot disappear before grouping.

        self.log(f"Verifying {len(rows)} Zn candidates with {workers} workers")
        rows = analyse_structures(self, rows, workers, coordination_cutoff, minimum_occupancy,
                                  minimum_donors, minimum_residues, maximum_resolution,
                                  requests_per_second)
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

        selected, selection = select_population(rows, positive_negative_ratio, True, False, seed)
        selected, split_audit = assign_splits(selected, train_fraction, validation_fraction,
                                              test_fraction, seed)
        dilutions = create_dilutions(selected, dilution_fractions, dilution_replicates, seed)
        audit = audit_dataset(rows, selected, dilutions)
        audit.update(phenotypes=phenotypes, selection=selection, split_assignment=split_audit,
                     task="zinc_binding", positive_definition="verified deposited coordination",
                     physiological_specificity_claimed=False)

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
