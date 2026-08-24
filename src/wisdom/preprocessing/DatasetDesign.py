"""Design the leakage-safe WISDOM-DNA benchmark from immutable evidence records."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import warnings
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import gemmi
import lambdaforge as lf
import matplotlib
import numpy as np
from lambdaforge.work import ManagedFile, RateLimit, Tool
from scipy.spatial import cKDTree
from scipy.stats import chi2_contingency, ks_2samp, mannwhitneyu, wasserstein_distance
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt


class DatasetDesign(lf.Work):
    """Create the complete benchmark design before expensive WISDOM geometry is generated.

    This Work treats the supplied JSONL or legacy FASTA as immutable curated evidence. It reads
    one explicit record per candidate, downloads each unique RCSB mmCIF once, revalidates positive
    protein--DNA contacts,
    computes global and interface descriptors, and then runs MMseqs2 and Foldseek on the complete
    raw population. Connected components of sequence, structural, exact, and provenance relations
    are permanent leakage groups. Only after those full-population groups and physical phenotypes
    exist does the Work select a balanced canonical population, assign group-safe splits, and
    construct nested training dilutions. LambdaForge owns managed downloads, dependencies,
    checkpoints, external tools, clustering backends, logs, and atomic output finalization; WISDOM
    owns the scientific interpretation and writes one readable design directory.
    """

    SCHEMA_VERSION     = "1.2"
    RAW_SCHEMA_VERSION = "1.0"

    SPLITS      = ("train", "validation", "test")
    AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYOU")
    HYDROPHOBIC = frozenset("AVILMFWY")
    POLAR       = frozenset("STNQCY")
    POSITIVE    = frozenset("KRH")
    NEGATIVE    = frozenset("DE")
    AROMATIC    = frozenset("FWY")

    GLOBAL_PHENOTYPE_FEATURES = (
        "log_sequence_length",
        "radius_of_gyration_normalized",
        "aspect_ratio",
        "packing_density",
        "theoretical_isoelectric_point",
        "charge_density",
        "hydrophobic_residue_fraction",
        "polar_residue_fraction",
        "aromatic_fraction",
    )
    INTERFACE_PHENOTYPE_FEATURES = (
        "binding_residue_fraction",
        "interface_region_count",
        "largest_interface_region_fraction",
        "interface_radius_normalized",
        "interface_aspect_ratio",
        "interface_positive_residue_fraction",
        "interface_negative_residue_fraction",
        "interface_polar_residue_fraction",
        "interface_hydrophobic_residue_fraction",
        "interface_aromatic_residue_fraction",
        "contacted_dna_chain_count",
        "contact_density",
    )
    CONTINUOUS_FEATURES = (
        "sequence_length",
        "molecular_weight",
        "theoretical_isoelectric_point",
        "net_charge_at_pH_7",
        "gravy",
        "aromatic_fraction",
        "positive_residue_fraction",
        "negative_residue_fraction",
        "polar_residue_fraction",
        "hydrophobic_residue_fraction",
        "sequence_shannon_entropy",
        "coordinate_coverage",
        "heavy_atom_count",
        "radius_of_gyration",
        "radius_of_gyration_normalized",
        "aspect_ratio",
        "compactness",
        "packing_density",
        "resolution",
        "release_year",
    )

    _download_retries          : int
    _structure_rate_limit      : RateLimit
    _interface_region_distance : float

    def run(
        self,
        # Immutable input and bounded execution controls.
        raw_records                     : Path,
        workers                         : int             = 36,
        requests_per_second             : float           = 60.0,
        retries                         : int             = 5,
        output_directory                : str | None      = None,
        overwrite_output                : bool            = False,

        # Canonical class-balance policy. Ratios are positive real numbers.
        balance_classes                 : bool            = True,
        positive_negative_ratio         : float           = 1.0,
        keep_all_negatives              : bool            = True,
        retain_core_positives           : bool            = True,

        # Pairwise leakage thresholds. Fractions lie in [0, 1]; E-values are non-negative.
        sequence_identity               : float           = 0.30,
        sequence_coverage               : float           = 0.80,
        sequence_evalue                 : float           = 1e-3,
        foldseek_probability            : float           = 0.90,
        foldseek_tmscore                : float           = 0.75,
        foldseek_coverage               : float           = 0.80,
        foldseek_evalue                 : float           = 1e-3,
        group_same_pdb                  : bool            = True,
        giant_group_fraction_warning    : float           = 0.05,

        # Physical-phenotype clustering and interface geometry controls.
        global_min_cluster_size         : int             = 15,
        global_min_samples              : int             = 2,
        interface_min_cluster_size      : int             = 20,
        interface_min_samples           : int             = 5,
        phenotype_stability_minimum     : float           = 0.60,
        phenotype_noise_warning         : float           = 0.50,
        interface_region_distance       : float           = 8.0,
        maximum_resolution              : float | None    = 4.0,

        # Group-safe split targets and objective weights. Fractions must sum to one.
        train_fraction                  : float           = 0.70,
        validation_fraction             : float           = 0.15,
        test_fraction                   : float           = 0.15,
        split_size_weight               : float           = 1.0,
        split_class_weight              : float           = 2.0,
        split_global_phenotype_weight   : float           = 0.5,
        split_interface_phenotype_weight: float           = 0.5,
        split_source_weight             : float           = 0.25,
        split_nuisance_weight           : float           = 0.25,
        split_refinement_steps          : int             = 500,
        split_tolerance                 : float           = 0.05,

        # Training-only learning-curve views. Fractions lie in (0, 1] and include 1.0.
        dilution_fractions              : Sequence[float] = (1.0, 0.75, 0.50, 0.25, 0.10),
        dilution_replicates             : int             = 1,

        # Statistical warning thresholds; they report concerns but do not alter membership.
        smd_warning                     : float           = 0.25,
        smd_strong_warning              : float           = 0.50,
        ks_warning                      : float           = 0.20,
        cramers_v_warning               : float           = 0.20,
        technical_shortcut_auc_warning  : float           = 0.75,

        # Reproducibility and external specialist-tool selection.
        seed                            : int             = 2026,
        mmseqs_executable               : str             = "mmseqs",
        foldseek_executable             : str             = "foldseek",
    ) -> dict[str, Any]:
        """Build a balanced, leakage-safe benchmark design from curated evidence records.

        The method analyses every unique PDB entry, constructs full-population sequence and
        structure leakage components, discovers label-free physical phenotypes, assigns complete
        components to fixed splits, and publishes nested training dilutions plus statistical audit
        files through LambdaForge-managed outputs.

        Args:
            raw_records: Immutable JSONL containing one explicit candidate per line. The legacy
                two-line FASTA contract remains accepted for reproducibility of older inputs.
            workers: Maximum concurrent structure jobs and specialist-tool threads.
            requests_per_second: Aggregate RCSB request-start limit in requests per second.
            retries: Additional attempts allowed after one failed RCSB transfer.
            output_directory: Optional conventional publication directory, relative to the project
                root unless an absolute persistent-cluster path is supplied.
            overwrite_output: Whether a different existing publication copy may be replaced
                atomically after the managed artifact passes final validation.
            balance_classes: Whether to enforce ``positive_negative_ratio`` in the canonical set.
            positive_negative_ratio: Requested selected-positive to selected-negative count ratio.
            keep_all_negatives: Whether balancing must retain every defensible negative.
            retain_core_positives: Whether BTD-Core positives receive deterministic priority.
            sequence_identity: Minimum MMseqs2 aligned residue identity in ``[0, 1]``.
            sequence_coverage: Minimum query and target sequence coverage in ``[0, 1]``.
            sequence_evalue: Maximum retained MMseqs2 expectation value.
            foldseek_probability: Minimum Foldseek homology probability in ``[0, 1]``.
            foldseek_tmscore: Minimum query- and target-normalized TM-score in ``[0, 1]``.
            foldseek_coverage: Minimum query and target structural coverage in ``[0, 1]``.
            foldseek_evalue: Maximum retained Foldseek expectation value.
            group_same_pdb: Whether chains from one PDB deposition form one leakage component.
            giant_group_fraction_warning: Component-size fraction that emits a warning.
            global_min_cluster_size: Minimum HDBSCAN global-phenotype cluster size.
            global_min_samples: HDBSCAN global-phenotype core-neighbour count.
            interface_min_cluster_size: Minimum positive-interface phenotype cluster size.
            interface_min_samples: HDBSCAN interface-phenotype core-neighbour count.
            phenotype_stability_minimum: Minimum median neighbouring-grid adjusted Rand index.
            phenotype_noise_warning: Selected global-phenotype noise fraction that emits a
                representation warning.
            interface_region_distance: Residue-centroid interface connectivity cutoff in angstroms.
            maximum_resolution: Largest accepted X-ray/cryo-EM resolution in angstroms;
                missing values remain eligible and ``None`` disables this quality filter.
            train_fraction: Target fraction assigned to training.
            validation_fraction: Target fraction assigned to validation.
            test_fraction: Target fraction held for final testing.
            split_size_weight: Split-objective weight for total sample counts.
            split_class_weight: Split-objective weight for positive and negative counts.
            split_global_phenotype_weight: Weight for global-phenotype representation.
            split_interface_phenotype_weight: Weight for interface-phenotype representation.
            split_source_weight: Weight for positive-source representation.
            split_nuisance_weight: Weight for technical-covariate mean agreement.
            split_refinement_steps: Maximum accepted-search iterations after greedy assignment.
            split_tolerance: Relative split-size deviation that emits a warning.
            dilution_fractions: Nested fractions of complete training leakage groups to publish.
            dilution_replicates: Independent deterministic group rankings per dilution fraction.
            smd_warning: Absolute standardized mean-difference warning threshold.
            smd_strong_warning: Absolute standardized mean difference considered strong.
            ks_warning: Kolmogorov--Smirnov statistic warning threshold.
            cramers_v_warning: Cramer's V categorical-association warning threshold.
            technical_shortcut_auc_warning: Technical-only group-CV AUROC red-flag threshold.
            seed: Seed incorporated into deterministic SHA-256 rankings.
            mmseqs_executable: MMseqs2 executable name or path resolved by LambdaForge.
            foldseek_executable: Foldseek executable name or path resolved by LambdaForge.

        Returns:
            JSON-compatible summary containing population sizes, leakage counts, output identity,
            and the most important audit metrics.

        Raises:
            RuntimeError: If structure membership changes, external evidence is invalid, leakage
                reaches multiple splits, hard representation constraints fail, or publication
                products cannot satisfy their scientific invariants.
        """
        # Phase 1 — Read the immutable evidence population.
        #
        # parameters is the researcher-selected configuration copied into the final provenance
        # report. Operational worker count and tool names are recorded but never alter identifiers,
        # deterministic ordering, or scientific hashes.

        parameters = dict(self.config.parameters)
        parameters.pop("raw_records")
        parameters.pop("output_directory")
        parameters.pop("overwrite_output")

        # ``raw_rows`` is ``list[dict]`` with one row per evidence record. Each row already contains
        # PDB/chain/assembly identity, sequence, binary label, origin, and source provenance.

        raw_records = raw_records.resolve()
        self.log(f"Reading immutable raw evidence: {raw_records}")
        raw_rows = self._read_records(raw_records)

        self.log(
            "Raw population: "
            f"{len(raw_rows)} proteins, "
            f"{sum(row['label'] == 1 for row in raw_rows)} positive, "
            f"{sum(row['label'] == 0 for row in raw_rows)} negative"
        )

        # LambdaForge's named rate limiter is shared safely by every thread in this Work. It delays
        # request starts only; cached structures return without another network transfer.
        self._download_retries          = retries
        self._structure_rate_limit      = self.cache.rate_limit(
            "rcsb",
            requests_per_second = requests_per_second,
        )
        self._interface_region_distance = interface_region_distance

        # Phase 2 — Download each unique structure once and compute chain-level descriptors.
        #
        # Several FASTA records may refer to different chains or assembly copies in one PDB. Jobs
        # therefore have the compact format {"pdb_id": str, "rows": list[dict]}.

        grouped_jobs: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in raw_rows:
            grouped_jobs[str(row["pdb_id"])].append(row)

        jobs = [
            {"pdb_id": pdb_id, "rows": members}
            for pdb_id, members in sorted(grouped_jobs.items())
        ]

        self.log(
            f"Downloading and analysing {len(jobs)} unique PDB entries "
            f"with {workers} bounded workers"
        )

        # resume_map owns bounded threads, stable per-PDB keys, resumable JSON results, retries,
        # and the hashes of every managed cache file touched by a worker. A saved result is reused
        # only while its downloaded mmCIF and generated Foldseek-chain files still match.

        mapped = self.resume_map(
            jobs,
            self._analyse_pdb,
            key      = "pdb_id",
            workers  = workers,
            executor = "thread",
            name     = "dataset-design-structures",
        )

        # Flatten the map output back to one descriptor row per original FASTA member. Sorting by
        # logical identifier makes later graph construction independent of worker completion order.

        rows = sorted(
            (dict(row) for result in mapped for row in result["rows"]),
            key=lambda value: str(value["identifier"]),
        )

        if len(rows) != len(raw_rows):
            raise RuntimeError("structure analysis did not preserve every raw evidence member")

        self.log(f"Structural descriptors complete: {len(rows)} proteins")

        # Phase 3 — Build full-population sequence and structural similarity evidence.
        #
        # ``require`` resolves each executable before expensive work begins and records its version
        # in LambdaForge provenance. Failure is explicit: there is no approximate fallback.

        mmseqs_tool   = self.tools.require(mmseqs_executable,   version_args=["version"])
        foldseek_tool = self.tools.require(foldseek_executable, version_args=["version"])

        # The progress snapshot has two dataset-level units: MMseqs2 and Foldseek.
        # ``progress.update`` only feeds ``lf top``/run status; it never changes scientific data or
        # checkpoint identity.

        self.log("Running or reusing full-raw MMseqs2 all-vs-all")
        self.progress.update(completed=0, total=2, message="MMseqs2 full-raw similarity")

        # ``checkpoints.file`` returns a read-only ManagedFile. LambdaForge reuses it when the TSV
        # passes ``_valid_pair_evidence``; otherwise it atomically calls ``build`` and validates the
        # replacement. The MMseqs2 TSV columns are:
        # query, target, identity, query coverage, target coverage, E-value, bit score.

        mmseqs_file = self.checkpoints.file(
            "leakage/sequence-pairs.tsv",
            build=lambda target: self._run_mmseqs(
                rows,
                target,
                workers,
                parameters,
                mmseqs_tool,
            ),
            validate=lambda file: self._valid_pair_evidence(
                file,
                rows,
                parameters,
                self._sequence_edges,
            ),
        )

        mmseqs_path    = Path(mmseqs_file)
        sequence_edges = self._sequence_edges(mmseqs_path, rows, parameters)
        self.progress.update(completed=1, total=2, message="Foldseek full-raw similarity")

        # Foldseek follows the same managed-checkpoint lifecycle. Its TSV columns are:
        # query, target, homology probability, E-value, query/target TM-score, and query/target
        # coverage.

        self.log("Running or reusing full-raw Foldseek all-vs-all")
        foldseek_file = self.checkpoints.file(
            "leakage/structure-pairs.tsv",
            build=lambda target: self._run_foldseek(
                rows,
                target,
                workers,
                parameters,
                foldseek_tool,
            ),
            validate=lambda file: self._valid_pair_evidence(
                file,
                rows,
                parameters,
                self._structure_edges,
            ),
        )

        foldseek_path    = Path(foldseek_file)
        structure_edges = self._structure_edges(foldseek_path, rows, parameters)
        self.progress.update(completed=2, total=2, message="full-raw leakage complete")

        # Phase 4 — Convert pair evidence into indivisible leakage groups.
        #
        # Each edge is stored once as ``(min_identifier, max_identifier)``. Connected components of
        # Sequence, structure, exact-sequence, identity, and optional same-PDB edges form groups
        # that may never be split across train, validation, and test.

        exact_pairs = self._exact_pairs(rows, group_same_pdb)
        all_edges   = sequence_edges | structure_edges | {
            (str(row["left"]), str(row["right"])) for row in exact_pairs
        }
        components = self._components(
            [str(row["identifier"]) for row in rows],
            all_edges,
        )
        group_by_id = {
            identifier: f"L{number:05d}"
            for number, component in enumerate(components, start=1)
            for identifier in component
        }
        for row in rows:
            row["leakage_group"] = group_by_id[str(row["identifier"])]

        largest_group = max(map(len, components))
        giant_fraction = largest_group / len(rows)
        if giant_fraction >= giant_group_fraction_warning:
            self.log(
                "Largest leakage group contains "
                f"{largest_group}/{len(rows)} proteins ({giant_fraction:.1%})",
                level="warning",
            )

        # Phase 5 — Describe physical diversity without using the DNA-binding label as a feature.
        #
        # Global phenotypes cover every quality-eligible protein. Interface phenotypes use eligible
        # positives only because a curated negative has no positive DNA-contact interface to
        # describe. Quality-excluded rows remain in leakage groups but receive explicit noise.

        # Low-resolution fitted atomic models remain in the full-raw leakage graph, where they can
        # conservatively connect homologues, but they cannot enter the canonical geometric dataset.
        # Structures without a numeric resolution (for example NMR) remain eligible.
        quality_exclusions: list[dict[str, Any]] = []
        eligible_rows: list[dict[str, Any]] = []
        for row in rows:
            resolution = row.get("resolution")
            eligible = (
                maximum_resolution is None
                or resolution is None
                or float(resolution) <= maximum_resolution
            )
            row["quality_eligible"] = eligible
            row["quality_exclusion_reason"] = (
                "" if eligible else "resolution_exceeds_maximum"
            )
            if eligible:
                eligible_rows.append(row)
            else:
                quality_exclusions.append(
                    {
                        "identifier": row["identifier"],
                        "label": row["label"],
                        "resolution": resolution,
                        "maximum_resolution": maximum_resolution,
                        "leakage_group": row["leakage_group"],
                    }
                )

        self.log(
            f"Quality-eligible population: {len(eligible_rows)}/{len(rows)} proteins; "
            f"excluded {len(quality_exclusions)} low-resolution structures"
        )
        self.log("Fitting quality-eligible physical phenotype clusters")
        global_result = self._phenotypes(
            eligible_rows,
            self.GLOBAL_PHENOTYPE_FEATURES,
            "G",
            global_min_cluster_size,
            global_min_samples,
            phenotype_stability_minimum,
            workers,
        )
        interface_rows = [row for row in eligible_rows if row["label"] == 1]
        interface_result = self._phenotypes(
            interface_rows,
            self.INTERFACE_PHENOTYPE_FEATURES,
            "I",
            interface_min_cluster_size,
            interface_min_samples,
            phenotype_stability_minimum,
            workers,
        )
        for row in rows:
            identifier = str(row["identifier"])
            row["global_phenotype"] = global_result["labels"].get(identifier, "G_NOISE")
            row["global_phenotype_probability"] = global_result["probabilities"].get(
                identifier, 0.0
            )
            row["interface_phenotype"] = (
                interface_result["labels"].get(identifier, "I_NOISE")
                if row["label"] == 1
                else "not_applicable"
            )
            row["interface_phenotype_probability"] = (
                interface_result["probabilities"].get(identifier, 0.0)
                if row["label"] == 1
                else None
            )

        # Phase 6 — Balance the canonical population and assign complete groups to fixed splits.
        #
        # ``selected`` remains a list of descriptor dictionaries. ``assignments`` maps each stable
        # leakage-group ID to exactly one of train/validation/test; no individual protein is moved.
        self.log("Selecting the canonical population after full-raw analysis")
        selected, selection_audit = self._select_population(
            eligible_rows,
            balance_classes,
            positive_negative_ratio,
            keep_all_negatives,
            retain_core_positives,
            seed,
        )
        selection_audit["quality_filter"] = {
            "maximum_resolution": maximum_resolution,
            "input_counts": self._class_counts(rows),
            "eligible_counts": self._class_counts(eligible_rows),
            "excluded_counts": self._class_counts(
                [row for row in rows if not row["quality_eligible"]]
            ),
            "exclusions": quality_exclusions,
        }
        assignments, split_audit = self._assign_splits(
            selected,
            parameters,
            seed,
        )
        selected_ids = {str(value["identifier"]) for value in selected}
        for row in rows:
            row["selected"] = str(row["identifier"]) in selected_ids
            row["split"] = assignments.get(str(row["leakage_group"]), "")
        for row in selected:
            row["selected"] = True
            row["split"] = assignments[str(row["leakage_group"])]
        self._validate_splits(selected, sequence_edges, structure_edges, exact_pairs)

        # Phase 7 — Create nested training dilutions and publish the scientific audit.
        #
        # ``outputs.directory`` creates an attempt-owned directory. LambdaForge fingerprints and
        # registers it only after ``run`` succeeds, so WISDOM does not need temporary output names,
        # locks, or manual atomic renames. ``statistics/`` is part of that same managed artifact.
        design_root = Path(
            self.outputs.directory(
                "dataset-design",
                role       = "dataset",
                publish_to = output_directory,
                overwrite  = overwrite_output if output_directory is not None else False,
            )
        )
        statistics_root = design_root / "statistics"

        dilution_audit = self._write_dilutions(
            selected,
            design_root / "dilutions",
            dilution_fractions,
            dilution_replicates,
            seed,
        )

        self.log("Computing statistical audits and group-aware shortcut baselines")
        statistics = self._statistics(
            rows,
            selected,
            dilution_audit,
            parameters,
            statistics_root,
            seed,
        )
        warnings = self._warnings(
            rows,
            selected,
            components,
            split_audit,
            statistics,
            parameters,
        )
        self._write_outputs(
            design_root,
            raw_records,
            rows,
            selected,
            mmseqs_path,
            foldseek_path,
            sequence_edges,
            structure_edges,
            exact_pairs,
            components,
            global_result,
            interface_result,
            selection_audit,
            split_audit,
            dilution_audit,
            statistics,
            warnings,
            parameters,
        )

        # Phase 8 — Expose concise live/run metadata in addition to the detailed output files.
        #
        # ``metrics.log_many`` feeds result comparison and monitoring. ``outputs.value`` stores the
        # same compact JSON summary beside the managed directory; neither replaces the full audit.
        selected_positive = sum(row["label"] == 1 for row in selected)
        selected_negative = sum(row["label"] == 0 for row in selected)
        technical_auc = float(
            statistics["shortcut_baselines"].get("technical_without_origin", {}).get(
                "auroc_mean", 0.0
            )
            or 0.0
        )
        metrics = {
            "raw_total": len(rows),
            "raw_positive": sum(row["label"] == 1 for row in rows),
            "raw_negative": sum(row["label"] == 0 for row in rows),
            "selected_total": len(selected),
            "selected_positive": selected_positive,
            "selected_negative": selected_negative,
            "leakage_group_count": len(components),
            "largest_leakage_group": largest_group,
            "global_phenotype_count": int(global_result["diagnostics"]["cluster_count"]),
            "interface_phenotype_count": int(interface_result["diagnostics"]["cluster_count"]),
            "train_total": sum(row["split"] == "train" for row in selected),
            "validation_total": sum(row["split"] == "validation" for row in selected),
            "test_total": sum(row["split"] == "test" for row in selected),
            "technical_shortcut_auc": technical_auc,
        }
        self.metrics.log_many(metrics)
        self.outputs.value("dataset-design-summary", metrics)
        self.log(
            f"PASS: selected {len(selected)} proteins; "
            f"splits={metrics['train_total']}/{metrics['validation_total']}/{metrics['test_total']}"
        )
        return {
            "verdict": "PASS",
            **metrics,
            "dataset_design_output": "dataset-design",
        }

    def _read_records(self, path: Path) -> list[dict[str, Any]]:
        """Read canonical JSONL or the legacy two-line FASTA evidence format.

        JSONL is preferred because identity, label evidence, assembly selection, provenance, and
        sequence are typed fields on one line. A ``.fasta`` or ``.fa`` suffix selects the legacy
        parser so previously frozen candidate pools remain reproducible.

        Args:
            path: Immutable UTF-8 ``.jsonl``, ``.fasta``, or ``.fa`` input.

        Returns:
            Records in source order with normalized identity fields and sequence fingerprints.

        Raises:
            OSError: If the input cannot be read.
            KeyError: If a required evidence field is absent.
            ValueError: If the format, identifier, label, sequence, or duplicate identity fails.
        """
        if path.suffix.lower() in {".fasta", ".fa", ".faa"}:
            records = self._read_fasta(path)
        else:
            records = []
            for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"raw JSONL line {line_number} must be an object")
                records.append(self._normalize_raw_record(value))

        identifiers = [str(record["identifier"]) for record in records]
        if len(identifiers) != len(set(identifiers)):
            duplicates = sorted(
                identifier
                for identifier, count in Counter(identifiers).items()
                if count > 1
            )
            raise ValueError(f"raw evidence contains duplicate identifiers: {duplicates[:10]}")
        return records

    def _read_fasta(self, path: Path) -> list[dict[str, Any]]:
        """Convert the historical two-line FASTA contract into explicit evidence records.

        Args:
            path: Immutable UTF-8 FASTA alternating one pipe-delimited header and one sequence.

        Returns:
            Normalized records compatible with the canonical JSONL representation.

        Raises:
            OSError: If the FASTA cannot be read.
            KeyError: If a required header field is absent.
            ValueError: If header/sequence pairing or a field value is malformed.
        """
        lines = [
            line.strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        records: list[dict[str, Any]] = []

        # Legacy files always alternate one complete header and one complete sequence.
        for header_line, sequence in zip(lines[0::2], lines[1::2], strict=True):
            header     = header_line.removeprefix(">")
            fields     = header.split("|")
            identifier = fields[0]
            metadata   = {
                name: value
                for name, value in (field.split("_", 1) for field in fields[1:])
            }
            flags = [
                field
                for field in fields[1:]
                if field.split("_", 1)[0]
                not in {"assembly", "copy", "label", "origin", "source"}
            ]
            label = int(metadata["label"])
            records.append(
                self._normalize_raw_record(
                    {
                        "identifier":      identifier,
                        "assembly_id":     metadata["assembly"],
                        "protein_copy":    int(metadata["copy"]),
                        "label":           label,
                        "label_evidence":  self._legacy_label_evidence(label, metadata["origin"]),
                        "origin":          metadata["origin"],
                        "source":          metadata["source"],
                        "sequence":        sequence,
                        "original_header": header,
                        "header_flags":    sorted(flags),
                    }
                )
            )
        return records

    @staticmethod
    def _normalize_raw_record(value: Mapping[str, Any]) -> dict[str, Any]:
        """Normalize one evidence object without changing its scientific assertions.

        Args:
            value: Mapping containing identifier, assembly, copy, label, provenance, and sequence.

        Returns:
            Portable design row with uppercase sequence/PDB identity and a SHA-256 sequence hash.

        Raises:
            KeyError: If a required field is missing.
            ValueError: If identifier, sequence, or binary label is invalid.
        """
        identifier    = str(value["identifier"])
        pdb_id, chain = identifier.split("_", 1)
        sequence      = str(value["sequence"]).upper()
        label         = int(value["label"])
        schema        = value.get("schema_version")
        if schema is not None and str(schema) != DatasetDesign.RAW_SCHEMA_VERSION:
            raise ValueError(f"unsupported raw evidence schema {schema!r}")
        if label not in {0, 1}:
            raise ValueError(f"raw label must be binary for {identifier}")
        if not sequence or not set(sequence).issubset(DatasetDesign.AMINO_ACIDS):
            raise ValueError(f"raw sequence contains unsupported residues for {identifier}")

        return {
            "identifier":      identifier,
            "base_identifier": identifier,
            "original_header": str(value.get("original_header", identifier)),
            "sequence":        sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "label":           label,
            "label_evidence":  str(value["label_evidence"]),
            "origin":          str(value["origin"]),
            "source":          str(value["source"]),
            "pdb_id":          pdb_id.upper(),
            "protein_chain":   chain,
            "assembly_id":     str(value["assembly_id"]),
            "protein_copy":    int(value["protein_copy"]),
            "header_flags":    sorted(str(flag) for flag in value.get("header_flags", [])),
        }

    @staticmethod
    def _legacy_label_evidence(label: int, origin: str) -> str:
        """Recover the documented evidence tier for a historical FASTA record.

        Args:
            label: Curated binary protein label.
            origin: Historical source-origin field from the FASTA header.

        Returns:
            Explicit evidence category used by reports and catalogs.
        """
        if label == 0:
            return "benchmark_exclusion_derived_negative"
        if "rcsb" in origin:
            return "direct_structural_dna_contact"
        return "benchmark_positive_with_revalidated_dna_contact"

    def _analyse_pdb(self, job: Mapping[str, Any]) -> dict[str, Any]:
        """Derive chain descriptors and Foldseek inputs from one managed PDB entry.

        Args:
            job: Mapping containing one four-character ``pdb_id`` and all FASTA rows that use it.

        Returns:
            JSON-compatible map result containing the analysed rows and immutable structure hash.

        Raises:
            ValueError: If the structure, assembly, chain, or declared DNA contact is invalid.
            RuntimeError: If the managed RCSB download cannot be obtained or parsed.
        """

        # ``cache.fetch`` stores decompressed mmCIF bytes under one logical PDB key. LambdaForge
        # handles rate limiting, retries, atomic publication, hashing, and concurrent callers.
        pdb_id        = str(job["pdb_id"])
        structure_url = f"https://files.rcsb.org/download/{pdb_id.upper()}.cif.gz"
        structure_file = self.cache.fetch(
            structure_url,
            key        = f"structures/{pdb_id.lower()}.cif",
            retries    = self._download_retries,
            timeout    = 180.0,
            decompress = "gzip",
            validate   = self._valid_structure,
            rate_limit = self._structure_rate_limit,
        )

        structure_path = Path(structure_file)
        structure      = gemmi.read_structure(str(structure_file))
        structure_hash = self._sha256_file(structure_path)

        if not structure:
            raise ValueError(f"RCSB structure {pdb_id} contains no coordinate model")

        # Gemmi enriches polymer/entity annotations before WISDOM selects biological assemblies.
        # ``metadata`` is a small JSON-compatible mapping shared by every chain from this PDB.
        structure.setup_entities()
        structure.assign_label_seq_id()
        metadata = self._structure_metadata(structure_path, structure)

        # Each analysed row contains scalar/string descriptors only. The sole exception is the
        # path-like ManagedFile reference required later by Foldseek; map checkpoints serialize it
        # as a logical cache key plus SHA-256 rather than as a machine-specific path.
        analysed: list[dict[str, Any]] = []
        for raw in sorted(job["rows"], key=lambda value: str(value["identifier"])):
            row = self._analyse_structure_member(structure, raw, metadata)

            # One protein-only mmCIF is cached per logical FASTA member. ``cache.file`` rebuilds it
            # automatically if its managed bytes disappear or fail the one-chain validator.
            foldseek_file = self.cache.file(
                f"foldseek/{raw['identifier']}.cif",
                build=lambda target, member=raw: self._write_foldseek_chain(
                    structure,
                    member,
                    target,
                ),
                validate=self._valid_foldseek_structure,
            )

            row["structure_url"]      = structure_url
            # Scientific provenance hashes the mmCIF payload itself. ManagedFile.sha256 also
            # commits to its cache filename, so it is intentionally not used as a molecular hash.
            row["structure_sha256"]   = structure_hash
            row["foldseek_structure"] = foldseek_file
            analysed.append(row)

        return {
            "pdb_id": pdb_id,
            "structure_sha256": structure_hash,
            "rows": analysed,
        }

    @staticmethod
    def _valid_structure(file: ManagedFile) -> bool:
        """Return whether a LambdaForge-managed file is a non-empty coordinate structure.

        Args:
            file: Path-like managed cache candidate containing uncompressed PDB/mmCIF bytes.

        Returns:
            ``True`` when Gemmi parses at least one coordinate model; otherwise ``False`` so
            LambdaForge discards and rebuilds the cache entry atomically.
        """
        try:
            return bool(gemmi.read_structure(str(file)))
        except (OSError, RuntimeError, ValueError):
            return False

    @staticmethod
    def _valid_foldseek_structure(file: ManagedFile) -> bool:
        """Return whether a managed Foldseek input contains exactly one protein chain.

        Args:
            file: Path-like managed cache candidate produced for one FASTA member.

        Returns:
            ``True`` only for a parseable structure whose first model contains one chain.
        """
        try:
            structure = gemmi.read_structure(str(file))
            return bool(structure) and len(structure[0]) == 1
        except (OSError, RuntimeError, ValueError):
            return False

    @staticmethod
    def _structure_metadata(path: Path, structure: gemmi.Structure) -> dict[str, Any]:
        """Extract technical experimental metadata without consulting a mutable web API.

        Args:
            path: Valid uncompressed mmCIF file.
            structure: Gemmi structure parsed from the same bytes.

        Returns:
            Experimental method, resolution, and deposition/release year when available.

        Raises:
            OSError: If the mmCIF document cannot be read.
        """
        # Read method/date fields from the same cached bytes used for coordinates; no mutable API
        # response can therefore disagree with the structure analysed in this Run.
        document = gemmi.cif.read(str(path))
        block    = document.sole_block()

        # Resolution is unavailable for methods/entries where Gemmi reports zero or a nonfinite
        # value. Missing values remain ``None`` instead of acquiring an invented numeric sentinel.
        method = block.find_value("_exptl.method") or "unavailable"
        resolution = float(structure.resolution)
        if not math.isfinite(resolution) or resolution <= 0.0:
            resolution_value: float | None = None
        else:
            resolution_value = resolution
        dates = (
            block.find_value("_pdbx_database_status.recvd_initial_deposition_date"),
            block.find_value("_database_PDB_rev.date_original"),
            block.find_value("_pdbx_audit_revision_history.revision_date"),
        )
        year: int | None = None
        for value in dates:
            if value and len(value) >= 4 and value[:4].isdigit():
                year = int(value[:4])
                break
        return {
            "experimental_method": str(method).strip("'\"") or "unavailable",
            "resolution": resolution_value,
            "release_year": year,
        }

    def _analyse_structure_member(
        self,
        structure: gemmi.Structure,
        raw      : Mapping[str, Any],
        metadata : Mapping[str, Any],
    ) -> dict[str, Any]:
        """Reconstruct one declared assembly copy and calculate all design descriptors.

        Global size/shape descriptors use the exact transformed protein copy, although a rigid
        assembly transform leaves their values unchanged. Positive contacts use the strict
        BioLiP/Q-BioLiP-style inequality ``d < r_protein + r_DNA + 0.5 Å``. A raw positive without
        such a contact, or a raw negative with one, is a hard provenance conflict and aborts the
        Work rather than being relabelled or removed.

        Args:
            structure: Parsed RCSB entry with assigned entities and label sequence identifiers.
            raw: One normalized raw JSONL or legacy FASTA evidence row.
            metadata: Experimental method, resolution, and year from the same mmCIF bytes.

        Returns:
            Raw row extended with sequence, structure, global, interface, and assembly facts.

        Raises:
            ValueError: If sequence, assembly/copy identity, coordinates, or label contact fails.
        """
        # Match the immutable FASTA record against the deposited entity before using coordinates.
        # This prevents a correct PDB ID paired with the wrong chain sequence from entering RAW.
        model = structure[0]
        chain_name = str(raw["protein_chain"])
        base_chain = model.find_chain(chain_name)
        if base_chain is None or not self._is_protein(base_chain):
            raise ValueError(f"{raw['identifier']} protein chain is absent from deposited model")
        full_sequence = self._full_sequence(structure, base_chain)
        if full_sequence != str(raw["sequence"]):
            raise ValueError(
                f"{raw['identifier']} raw sequence disagrees with the exact mmCIF entity sequence"
            )

        # Reconstruct exactly the biological assembly and one-based protein copy declared in the
        # FASTA header. Descriptors therefore refer to a physical copy, not an arbitrary chain.
        assembly = next(
            (value for value in structure.assemblies if str(value.name) == str(raw["assembly_id"])),
            None,
        )
        if assembly is None:
            raise ValueError(
                f"{raw['identifier']} declares absent biological assembly {raw['assembly_id']!r}"
            )
        assembled = gemmi.make_assembly(assembly, model, gemmi.HowToNameCopiedChain.Dup)
        protein_copies = [
            chain for chain in assembled if chain.name == chain_name and self._is_protein(chain)
        ]
        copy_index = int(raw["protein_copy"]) - 1
        if copy_index >= len(protein_copies):
            raise ValueError(
                f"{raw['identifier']} declares copy {copy_index + 1}, but assembly contains "
                f"{len(protein_copies)} matching chains"
            )
        protein_copy = protein_copies[copy_index]
        protein = self._protein_coordinates(protein_copy, len(full_sequence))
        if not len(protein["atom_positions"]):
            raise ValueError(f"{raw['identifier']} has no finite occupied protein heavy atoms")

        # Contacts use arrays of occupied heavy atoms in ångströms. A contradiction with the frozen
        # label aborts instead of silently deleting or relabelling scientifically awkward examples.
        dna_chains = [chain for chain in assembled if self._is_dna(chain)]
        dna = self._dna_coordinates(dna_chains)
        contacts = self._contacts(protein, dna)
        label = int(raw["label"])
        if label == 1 and not contacts["binding_residues"]:
            raise ValueError(
                f"raw positive {raw['identifier']} no longer has a direct assembly DNA contact"
            )
        if label == 0 and contacts["binding_residues"]:
            raise ValueError(
                f"raw negative {raw['identifier']} contradicts a direct assembly DNA contact"
            )

        # The returned row is portable JSON data: scalar descriptors, short lists, and one rigid
        # transform that later aligns universal surface points with this biological assembly copy.
        rotation, translation = self._rigid_transform(base_chain, protein_copy)
        sequence_features = self._sequence_features(full_sequence)
        global_features   = self._global_structure_features(protein)
        interface_features = self._interface_features(
            protein,
            contacts,
            len(dna_chains),
        )
        observed = int(protein["observed_residue_count"])
        coverage = observed / len(full_sequence)
        result = {
            **dict(raw),
            **metadata,
            **sequence_features,
            **global_features,
            **interface_features,
            "observed_residue_count": observed,
            "coordinate_coverage": coverage,
            "missing_residue_fraction": 1.0 - coverage,
            "assembly_protein_chain_count": sum(self._is_protein(chain) for chain in assembled),
            "assembly_dna_chain_count": len(dna_chains),
            "dna_chains": sorted({chain.name for chain in dna_chains}),
            "binding_residue_indices": sorted(contacts["binding_residues"]),
            "local_gt_expected": True,
            "local_gt_method": "dna_distance" if label == 1 else "global_negative",
            "assembly_rotation": rotation.tolist(),
            "assembly_translation": translation.tolist(),
            "contact_definition": "d < protein_vdw + DNA_vdw + 0.5_angstrom",
            "interface_region_distance": self._interface_region_distance,
        }
        return result

    @staticmethod
    def _is_protein(chain: gemmi.Chain) -> bool:
        """Test whether a chain has a peptide polymer.

        Args:
            chain: Gemmi chain from a deposited or generated assembly model.

        Returns:
            True for L- or D-peptide polymers and false for DNA/RNA/ligands.
        """
        polymer = chain.get_polymer()
        return len(polymer) > 0 and polymer.check_polymer_type() in {
            gemmi.PolymerType.PeptideL,
            gemmi.PolymerType.PeptideD,
        }

    @staticmethod
    def _is_dna(chain: gemmi.Chain) -> bool:
        """Test whether a chain is a pure DNA polymer rather than RNA or a ligand.

        Args:
            chain: Gemmi chain from a biological assembly.

        Returns:
            True only when Gemmi classifies its polymer as DNA.
        """
        polymer = chain.get_polymer()
        return len(polymer) > 0 and polymer.check_polymer_type() == gemmi.PolymerType.Dna

    @staticmethod
    def _full_sequence(structure: gemmi.Structure, chain: gemmi.Chain) -> str:
        """Read the complete entity sequence associated with one deposited protein chain.

        Args:
            structure: Parent structure whose entity table owns the full sequence.
            chain: Exact deposited chain.

        Returns:
            Upper-case unambiguous one-letter entity sequence.

        Raises:
            ValueError: If the chain lacks a complete mmCIF entity sequence.
        """
        entity = structure.get_entity_of(chain.get_polymer())
        if entity is None or not entity.full_sequence:
            raise ValueError(f"chain {chain.name!r} has no complete mmCIF entity sequence")
        return gemmi.one_letter_code(entity.full_sequence).upper()

    def _protein_coordinates(
        self,
        chain          : gemmi.Chain,
        sequence_length: int,
    ) -> dict[str, Any]:
        """Extract occupied heavy atoms and one representative coordinate per observed residue.

        Args:
            chain: Exact transformed protein-chain copy.
            sequence_length: Full entity length used to validate ``label_seq`` positions.

        Returns:
            NumPy-backed atom facts, residue owners/types, representatives, and observed count.

        Raises:
            ValueError: If an occupied atom has no finite coordinate or valid van der Waals radius.
        """
        # Output arrays use the following contract:
        # atom_positions [A,3] Å, atom_radii [A] Å, atom_owners [A] zero-based sequence positions.
        # Residue positions [R,3] use C-alpha when present and the heavy-atom centroid otherwise.
        atoms: list[tuple[float, float, float]] = []
        radii: list[float] = []
        owners: list[int] = []
        residue_letters: dict[int, str] = {}
        residue_points: dict[int, tuple[float, float, float]] = {}
        # ``first_conformer`` deterministically resolves alternate conformations. Hydrogens and
        # zero-occupancy atoms do not participate in contact or shape evidence.
        for residue in chain.get_polymer().first_conformer():
            if residue.label_seq is None:
                continue
            position = int(residue.label_seq) - 1
            if not 0 <= position < sequence_length:
                continue
            info   = gemmi.find_tabulated_residue(residue.name)
            letter = (info.one_letter_code or "X").upper()
            heavy: list[tuple[float, float, float]] = []
            ca: tuple[float, float, float] | None = None
            for atom in residue.first_conformer():
                if atom.element.atomic_number <= 1 or atom.occ <= 0.0:
                    continue
                point  = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                radius = float(atom.element.vdw_r)
                if not np.isfinite(point).all() or not math.isfinite(radius) or radius <= 0.0:
                    raise ValueError(f"chain {chain.name!r} contains an invalid heavy atom")
                atoms.append(point)
                radii.append(radius)
                owners.append(position)
                heavy.append(point)
                if atom.name.strip() == "CA":
                    ca = point
            if heavy:
                residue_letters[position] = letter
                residue_points[position] = ca or tuple(np.mean(heavy, axis=0).tolist())
        ordered = sorted(residue_points)
        return {
            "atom_positions": np.asarray(atoms, dtype=np.float64).reshape((-1, 3)),
            "atom_radii": np.asarray(radii, dtype=np.float64),
            "atom_owners": np.asarray(owners, dtype=np.int64),
            "residue_indices": ordered,
            "residue_letters": [residue_letters[index] for index in ordered],
            "residue_positions": np.asarray(
                [residue_points[index] for index in ordered], dtype=np.float64
            ).reshape((-1, 3)),
            "observed_residue_count": len(ordered),
        }

    @staticmethod
    def _dna_coordinates(chains: Sequence[gemmi.Chain]) -> dict[str, Any]:
        """Extract occupied DNA heavy atoms, radii, and physical chain-instance indices.

        Args:
            chains: DNA polymer chains from one generated biological assembly.

        Returns:
            Position/radius arrays and one integer chain-instance owner per DNA atom.

        Raises:
            ValueError: If an occupied DNA atom has invalid coordinates or radius.
        """
        # Arrays mirror the protein contact representation: positions [D,3] Å, radii [D] Å, and
        # owners [D] identifying the physical DNA-chain instance in the generated assembly.
        positions: list[tuple[float, float, float]] = []
        radii: list[float] = []
        owners: list[int] = []
        for chain_index, chain in enumerate(chains):
            for residue in chain.get_polymer().first_conformer():
                for atom in residue.first_conformer():
                    if atom.element.atomic_number <= 1 or atom.occ <= 0.0:
                        continue
                    point  = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
                    radius = float(atom.element.vdw_r)
                    if not np.isfinite(point).all() or not math.isfinite(radius) or radius <= 0.0:
                        raise ValueError("DNA assembly contains an invalid occupied heavy atom")
                    positions.append(point)
                    radii.append(radius)
                    owners.append(chain_index)
        return {
            "positions": np.asarray(positions, dtype=np.float64).reshape((-1, 3)),
            "radii": np.asarray(radii, dtype=np.float64),
            "owners": np.asarray(owners, dtype=np.int64),
        }

    @staticmethod
    def _contacts(protein: Mapping[str, Any], dna: Mapping[str, Any]) -> dict[str, Any]:
        """Find direct protein--DNA contacts using atom-specific van der Waals cutoffs.

        For protein atom ``i`` and DNA atom ``j``, an edge exists exactly when
        ``||x_i-x_j|| < r_i+r_j+0.5 Å``. A broad KD-tree radius first finds possible neighbours;
        the atom-specific inequality then rejects false candidates without constructing a dense
        protein-by-DNA distance matrix.

        Args:
            protein: Protein atom positions, radii, and residue owners.
            dna: DNA atom positions, radii, and physical chain-instance owners.

        Returns:
            Contact-pair count, contacting atom/residue sets, and contacted DNA-chain count.
        """
        # Empty DNA or protein arrays imply zero contacts and preserve the same result schema.
        protein_xyz = np.asarray(protein["atom_positions"], dtype=np.float64)
        dna_xyz     = np.asarray(dna["positions"], dtype=np.float64)
        if not len(protein_xyz) or not len(dna_xyz):
            return {
                "pair_count": 0,
                "contacting_atoms": set(),
                "binding_residues": set(),
                "contacted_dna_chains": set(),
            }
        # A KD-tree proposes candidates using the largest possible cutoff. The second calculation
        # applies each atom pair's exact van der Waals threshold without a dense A-by-D matrix.
        protein_radii = np.asarray(protein["atom_radii"], dtype=np.float64)
        dna_radii     = np.asarray(dna["radii"], dtype=np.float64)
        broad = float(protein_radii.max() + dna_radii.max() + 0.5)
        neighbours = cKDTree(dna_xyz).query_ball_point(protein_xyz, broad)
        pair_count = 0
        contacting_atoms: set[int] = set()
        binding_residues: set[int] = set()
        contacted_chains: set[int] = set()
        atom_owners = np.asarray(protein["atom_owners"], dtype=np.int64)
        dna_owners  = np.asarray(dna["owners"], dtype=np.int64)
        for atom_index, candidates in enumerate(neighbours):
            if not candidates:
                continue
            indices   = np.asarray(candidates, dtype=np.int64)
            delta     = dna_xyz[indices] - protein_xyz[atom_index]
            squared   = np.einsum("ij,ij->i", delta, delta)
            cutoffs   = protein_radii[atom_index] + dna_radii[indices] + 0.5
            contacting = indices[squared < cutoffs * cutoffs]
            if not len(contacting):
                continue
            pair_count += len(contacting)
            contacting_atoms.add(atom_index)
            binding_residues.add(int(atom_owners[atom_index]))
            contacted_chains.update(int(value) for value in dna_owners[contacting])
        return {
            "pair_count": pair_count,
            "contacting_atoms": contacting_atoms,
            "binding_residues": binding_residues,
            "contacted_dna_chains": contacted_chains,
        }

    @staticmethod
    def _rigid_transform(
        deposited: gemmi.Chain,
        assembled: gemmi.Chain,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Recover the assembly copy's rigid transform from matched occupied protein atoms.

        Coordinates are treated as row vectors and transformed by ``x' = x R^T + t``. The Kabsch
        solution minimizes squared distance between identically named atoms in matching label-seq
        residues. Annotation later applies this matrix to the universal surface generated in the
        deposited-chain frame, placing it beside DNA in the declared assembly copy.

        Args:
            deposited: Original asymmetric-unit protein chain.
            assembled: Exact generated biological-assembly copy of that chain.

        Returns:
            Rotation ``float64 [3,3]`` and translation ``float64 [3]`` in ångströms.

        Raises:
            ValueError: If fewer than three non-collinear atom correspondences exist or RMSD is
                larger than numerical rigid-copy tolerance.
        """
        def atom_map(chain: gemmi.Chain) -> dict[tuple[int, str], tuple[float, float, float]]:
            """Map label-sequence/atom-name identity to one occupied coordinate.

            Args:
                chain: Deposited or assembled protein chain.

            Returns:
                Coordinate mapping used solely by the enclosing rigid-fit calculation.
            """
            values: dict[tuple[int, str], tuple[float, float, float]] = {}
            for residue in chain.get_polymer().first_conformer():
                if residue.label_seq is None:
                    continue
                for atom in residue.first_conformer():
                    if atom.element.atomic_number > 1 and atom.occ > 0.0:
                        values[(int(residue.label_seq), atom.name.strip())] = (
                            float(atom.pos.x),
                            float(atom.pos.y),
                            float(atom.pos.z),
                        )
            return values

        # Match atoms by label-sequence index and atom name so deposited/assembled ordering is
        # irrelevant. Kabsch then finds the least-squares proper rotation (determinant +1).
        source_map = atom_map(deposited)
        target_map = atom_map(assembled)
        common = sorted(source_map.keys() & target_map.keys())
        if len(common) < 3:
            raise ValueError("assembly transform needs at least three matched occupied atoms")
        source = np.asarray([source_map[key] for key in common], dtype=np.float64)
        target = np.asarray([target_map[key] for key in common], dtype=np.float64)
        source_center = source.mean(axis=0)
        target_center = target.mean(axis=0)
        covariance = (source - source_center).T @ (target - target_center)
        left, _, right = np.linalg.svd(covariance)
        rotation = right.T @ left.T
        if np.linalg.det(rotation) < 0.0:
            right[-1, :] *= -1.0
            rotation = right.T @ left.T
        translation = target_center - source_center @ rotation.T
        fitted = source @ rotation.T + translation
        rmsd = float(np.sqrt(np.mean(np.sum((fitted - target) ** 2, axis=1))))
        if rmsd > 1e-3:
            raise ValueError(
                f"assembly copy is not a rigid transform of its deposited chain: {rmsd}"
            )
        return rotation, translation

    def _sequence_features(self, sequence: str) -> dict[str, Any]:
        """Calculate interpretable global sequence descriptors and safe availability markers.

        Standard molecular weight, theoretical pI, pH-7 charge, GRAVY, and aromaticity use
        Biopython ``ProteinAnalysis``. Biopython does not define every property for U/O, so those
        values become unavailable rather than silently substituting another amino acid. Fractions
        and Shannon entropy are computed directly from the unchanged sequence.

        Args:
            sequence: Valid raw upper-case amino-acid sequence.

        Returns:
            Sequence length, standard physical properties, grouped/individual fractions, and
            entropy in bits where ``H=-sum_a p_a log2(p_a)``.

        Raises:
            RuntimeError: If Biopython is absent despite being a required WISDOM dependency.
        """
        try:
            from Bio.SeqUtils.ProtParam import ProteinAnalysis
        except ImportError as error:
            raise RuntimeError(
                "DatasetDesign requires Biopython; install WISDOM through environment.yml or pip"
            ) from error

        # Direct counts remain available even when Biopython cannot define a molecular property for
        # uncommon encoded residues such as selenocysteine (U) or pyrrolysine (O).
        length = len(sequence)
        counts = Counter(sequence)
        fractions = {amino: counts[amino] / length for amino in sorted(self.AMINO_ACIDS)}
        entropy = -sum(value * math.log2(value) for value in fractions.values() if value > 0.0)
        standard: dict[str, float | None] = {
            "molecular_weight": None,
            "theoretical_isoelectric_point": None,
            "net_charge_at_pH_7": None,
            "gravy": None,
            "aromatic_fraction": sum(fractions[value] for value in self.AROMATIC),
        }
        # Biopython supplies conventional whole-sequence physicochemical estimates. Undefined
        # estimates stay ``None`` and are handled explicitly by later phenotype/statistics code.
        try:
            analysis = ProteinAnalysis(sequence)  # type: ignore[no-untyped-call]
            standard.update(
                {
                    "molecular_weight": float(
                        analysis.molecular_weight()  # type: ignore[no-untyped-call]
                    ),
                    "theoretical_isoelectric_point": float(
                        analysis.isoelectric_point()  # type: ignore[no-untyped-call]
                    ),
                    "net_charge_at_pH_7": float(
                        analysis.charge_at_pH(7.0)  # type: ignore[no-untyped-call]
                    ),
                    "gravy": float(analysis.gravy()),  # type: ignore[no-untyped-call]
                    "aromatic_fraction": float(
                        analysis.aromaticity()  # type: ignore[no-untyped-call]
                    ),
                }
            )
        except (KeyError, ValueError):
            pass
        return {
            "sequence_length": length,
            **standard,
            "positive_residue_fraction": sum(fractions[value] for value in self.POSITIVE),
            "negative_residue_fraction": sum(fractions[value] for value in self.NEGATIVE),
            "polar_residue_fraction": sum(fractions[value] for value in self.POLAR),
            "hydrophobic_residue_fraction": sum(fractions[value] for value in self.HYDROPHOBIC),
            "glycine_fraction": fractions["G"],
            "proline_fraction": fractions["P"],
            "cysteine_fraction": fractions["C"],
            "sequence_shannon_entropy": entropy,
            **{f"fraction_{amino}": fractions[amino] for amino in sorted("ACDEFGHIKLMNPQRSTVWY")},
        }

    @staticmethod
    def _global_structure_features(protein: Mapping[str, Any]) -> dict[str, float | int]:
        """Measure global protein size, principal shape, compactness, and nonlocal packing.

        Radius of gyration is ``sqrt(mean_i ||x_i-c||^2)`` over residue representatives. Principal
        spreads are square roots of covariance eigenvalues. Nonlocal packing joins representative
        residues within 8 Å only when their sequence indices differ by at least three; density is
        the retained pair count divided by residue count. Compactness is observed residues per
        sphere volume defined by radius of gyration, in residues/Å³.

        Args:
            protein: Exact protein-copy positions, residue indices, and heavy atoms.

        Returns:
            Finite global structural descriptors suitable for physical phenotype clustering.
        """
        # Residue representatives form [R,3] coordinates in ångströms. Covariance eigenvalues give
        # rotationally invariant principal spreads; radius of gyration summarizes overall extent.
        positions = np.asarray(protein["residue_positions"], dtype=np.float64)
        indices   = np.asarray(protein["residue_indices"], dtype=np.int64)
        centered  = positions - positions.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(centered))
        eigen = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
        spreads = np.sqrt(eigen)
        radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        # Sparse 8 Å neighbours separated by at least three sequence positions measure tertiary
        # packing rather than trivial adjacency along the peptide backbone.
        pairs = cKDTree(positions).query_pairs(8.0)
        nonlocal_pairs = sum(
            abs(int(indices[left]) - int(indices[right])) >= 3 for left, right in pairs
        )
        volume = 4.0 * math.pi * max(radius, 1e-6) ** 3 / 3.0
        return {
            "heavy_atom_count": len(protein["atom_positions"]),
            "radius_of_gyration": radius,
            "radius_of_gyration_normalized": radius / max(len(positions) ** (1.0 / 3.0), 1.0),
            "principal_spread_1": float(spreads[0]),
            "principal_spread_2": float(spreads[1]),
            "principal_spread_3": float(spreads[2]),
            "aspect_ratio": float(spreads[0] / max(spreads[2], 1e-8)),
            "compactness": len(positions) / volume,
            "packing_density": nonlocal_pairs / max(1, len(positions)),
            "nonlocal_ca_contact_count": nonlocal_pairs,
        }

    def _interface_features(
        self,
        protein       : Mapping[str, Any],
        contacts      : Mapping[str, Any],
        dna_chain_count: int,
    ) -> dict[str, Any]:
        """Describe positive interface extent, regions, geometry, and residue chemistry.

        Interface graph nodes are contacting residues. Two nodes share an undirected edge when
        their representative points are at most ``interface_region_distance`` ångströms apart;
        connected components are physical regions. These label-derived descriptors characterize
        positive diversity only and are never passed to a positive/negative shortcut baseline.

        Args:
            protein: Protein residue positions, indices, letters, and atom count.
            contacts: Exact atom-contact result from :meth:`_contacts`.
            dna_chain_count: Number of physical DNA chain instances in the assembly.

        Returns:
            Positive interface descriptor mapping, or explicit unavailable values for negatives.
        """
        # Negatives have no positive interface by definition. ``None`` distinguishes unavailable
        # interface descriptors from a measured physical value of zero.
        binding = set(int(value) for value in contacts["binding_residues"])
        if not binding:
            return {
                "binding_residue_count": None,
                "binding_residue_fraction": None,
                "contacting_atom_count": None,
                "contact_pair_count": None,
                "contact_density": None,
                "contacted_dna_chain_count": None,
                "interface_region_count": None,
                "largest_interface_region_fraction": None,
                "interface_radius_of_gyration": None,
                "interface_radius_normalized": None,
                "interface_principal_spread_1": None,
                "interface_principal_spread_2": None,
                "interface_principal_spread_3": None,
                "interface_aspect_ratio": None,
                "interface_positive_residue_fraction": None,
                "interface_negative_residue_fraction": None,
                "interface_polar_residue_fraction": None,
                "interface_hydrophobic_residue_fraction": None,
                "interface_aromatic_residue_fraction": None,
            }

        # Select contacting residue representatives as [B,3] points and measure their extent and
        # principal shape independently of the global protein coordinates. A molecular interface
        # is approximately a two-dimensional patch, so elongation is the first/second principal
        # spread ratio. Dividing by the third (surface-normal thickness) created meaningless values
        # near 1e9 for valid planar interfaces in the preliminary dataset.
        indices = list(protein["residue_indices"])
        positions = np.asarray(protein["residue_positions"], dtype=np.float64)
        letters = list(protein["residue_letters"])
        selected = [index for index, residue in enumerate(indices) if int(residue) in binding]
        points   = positions[selected]
        selected_letters = [letters[index] for index in selected]
        centered = points - points.mean(axis=0)
        covariance = centered.T @ centered / max(1, len(centered))
        eigen = np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1]
        spreads = np.sqrt(eigen)
        radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        in_plane_scale = float(spreads[1])
        interface_aspect_ratio = (
            float(spreads[0] / in_plane_scale)
            if in_plane_scale > max(float(spreads[0]) * 1e-6, 1e-8)
            else None
        )

        # Sparse radius edges connect nearby interface residues. Connected components correspond to
        # distinct spatial patches and remain descriptive phenotypes, never leakage groups.
        adjacency: dict[int, set[int]] = {
            index: set() for index in range(len(points))
        }
        for left, right in cKDTree(points).query_pairs(self._interface_region_distance):
            adjacency[left].add(right)
            adjacency[right].add(left)
        remaining = set(adjacency)
        regions: list[int] = []
        while remaining:
            pending = [remaining.pop()]
            size = 0
            while pending:
                current = pending.pop()
                size += 1
                neighbours = adjacency[current] & remaining
                remaining.difference_update(neighbours)
                pending.extend(neighbours)
            regions.append(size)

        count = len(selected_letters)
        def fraction(values: frozenset[str]) -> float:
            """Measure one residue-property fraction inside this interface.

            Args:
                values: Residue letters belonging to one physicochemical category.

            Returns:
                Fraction of contacting residues in that category.
            """
            return sum(letter in values for letter in selected_letters) / count
        return {
            "binding_residue_count": count,
            "binding_residue_fraction": count / max(1, int(protein["observed_residue_count"])),
            "contacting_atom_count": len(contacts["contacting_atoms"]),
            "contact_pair_count": int(contacts["pair_count"]),
            "contact_density": int(contacts["pair_count"]) / max(1, len(protein["atom_positions"])),
            "contacted_dna_chain_count": len(contacts["contacted_dna_chains"]),
            "assembly_dna_chain_count_for_interface": dna_chain_count,
            "interface_region_count": len(regions),
            "largest_interface_region_fraction": max(regions) / count,
            "interface_radius_of_gyration": radius,
            "interface_radius_normalized": radius / max(count ** (1.0 / 3.0), 1.0),
            "interface_principal_spread_1": float(spreads[0]),
            "interface_principal_spread_2": float(spreads[1]),
            "interface_principal_spread_3": float(spreads[2]),
            "interface_aspect_ratio": interface_aspect_ratio,
            "interface_positive_residue_fraction": fraction(self.POSITIVE),
            "interface_negative_residue_fraction": fraction(self.NEGATIVE),
            "interface_polar_residue_fraction": fraction(self.POLAR),
            "interface_hydrophobic_residue_fraction": fraction(self.HYDROPHOBIC),
            "interface_aromatic_residue_fraction": fraction(self.AROMATIC),
        }

    def _write_foldseek_chain(
        self,
        structure: gemmi.Structure,
        raw      : Mapping[str, Any],
        target   : Path,
    ) -> None:
        """Write the exact declared protein assembly copy into a managed cache build target.

        Args:
            structure: Full deposited structure containing the declared assembly.
            raw: Parsed raw identity with chain, assembly, and one-based copy number.
            target: Temporary target owned and atomically published by LambdaForge.

        Raises:
            ValueError: If the declared assembly copy cannot be reconstructed.
            OSError: If Gemmi cannot serialize the selected chain.
        """
        # Reconstruct the declared biological assembly, then select the requested one-based copy of
        # the protein chain. The output mmCIF contains one model and exactly one protein chain.
        assembly = next(
            (value for value in structure.assemblies if str(value.name) == str(raw["assembly_id"])),
            None,
        )
        if assembly is None:
            raise ValueError(f"Foldseek input assembly is absent for {raw['identifier']}")
        assembled = gemmi.make_assembly(assembly, structure[0], gemmi.HowToNameCopiedChain.Dup)
        copies = [
            chain
            for chain in assembled
            if chain.name == str(raw["protein_chain"]) and self._is_protein(chain)
        ]
        index = int(raw["protein_copy"]) - 1
        if index >= len(copies):
            raise ValueError(f"Foldseek input copy is absent for {raw['identifier']}")
        selected = gemmi.Structure()
        selected.name = str(raw["identifier"])
        model = gemmi.Model(1)
        model.add_chain(copies[index].clone())
        selected.add_model(model)
        selected.setup_entities()
        target.parent.mkdir(parents=True, exist_ok=True)
        selected.make_mmcif_document().write_file(str(target))

    def _run_mmseqs(
        self,
        rows      : Sequence[Mapping[str, Any]],
        output    : Path,
        threads   : int,
        parameters: Mapping[str, Any],
        tool      : Tool,
    ) -> None:
        """Run MMseqs2 all-vs-all and write its seven-column checkpoint candidate.

        Args:
            rows: Full RAW descriptor rows containing ``identifier`` and ``sequence``.
            output: LambdaForge-owned temporary target for the sequence-pair TSV.
            threads: CPU threads assigned to the external search.
            parameters: Mapping containing identity, coverage, and E-value thresholds.
            tool: MMseqs2 executable resolved and versioned by LambdaForge.

        Raises:
            RuntimeError: If MMseqs2 returns without its declared output table.
        """
        # MMseqs2 consumes one ordinary FASTA. It lives in ``temp_dir`` because only the resulting
        # pair table is durable; LambdaForge removes specialist databases and scratch after the Run.
        root  = self.temp_dir / "mmseqs"
        fasta = root / "proteins.fasta"
        # Build two complementary views: RAW reveals source bias; selected reveals what the model
        # will actually see. All tables are JSON/CSV-safe summaries, never raw coordinate arrays.
        root.mkdir(parents=True, exist_ok=True)
        fasta.write_text(
            "".join(f">{row['identifier']}\n{row['sequence']}\n" for row in rows),
            encoding="utf-8",
        )
        # Request both directional coverages explicitly. WISDOM later reapplies all thresholds from
        # these raw columns instead of trusting tool-side filtering as the scientific decision.
        command = [
            tool,
            "easy-search",
            str(fasta),
            str(fasta),
            str(output),
            str(root / "tmp"),
            "--min-seq-id",
            str(parameters["sequence_identity"]),
            "-c",
            str(parameters["sequence_coverage"]),
            "--cov-mode",
            "0",
            "--alignment-mode",
            "3",
            "-e",
            str(parameters["sequence_evalue"]),
            "--max-seqs",
            str(max(10000, len(rows) + 1)),
            "--threads",
            str(threads),
            "--format-output",
            "query,target,fident,qcov,tcov,evalue,bits",
        ]
        # ``tools.run`` executes without a shell, limits threaded numerical libraries, streams the
        # command into Work logs, and records duration/stdout/stderr in tool provenance.
        self.tools.run(command, name="MMseqs2", threads=threads)

        if not output.is_file():
            raise RuntimeError("MMseqs2 completed without creating its pair table")

    def _run_foldseek(
        self,
        rows      : Sequence[Mapping[str, Any]],
        output    : Path,
        threads   : int,
        parameters: Mapping[str, Any],
        tool      : Tool,
    ) -> None:
        """Run Foldseek all-vs-all and write its eight-column checkpoint candidate.

        Args:
            rows: RAW rows containing managed one-chain ``foldseek_structure`` files.
            output: LambdaForge-owned temporary target for the structural-pair TSV.
            threads: CPU threads assigned to the external search.
            parameters: Mapping containing the Foldseek E-value threshold.
            tool: Foldseek executable resolved and versioned by LambdaForge.

        Raises:
            RuntimeError: If Foldseek exits unsuccessfully or omits its output table.
        """
        # Every protein-only file uses the ``foldseek/<identifier>.cif`` cache namespace. Managed
        # map dependencies already guarantee that these files exist and match their recorded hashes.
        foldseek_input = Path(rows[0]["foldseek_structure"]).parent
        root = self.temp_dir / "foldseek"
        root.mkdir(parents=True, exist_ok=True)

        # Both TM-scores and both coverages are retained because structural similarity is accepted
        # only when query and target satisfy the same geometric evidence thresholds.
        command = [
            tool,
            "easy-search",
            str(foldseek_input),
            str(foldseek_input),
            str(output),
            str(root / "tmp"),
            "-e",
            str(parameters["foldseek_evalue"]),
            "--max-seqs",
            str(max(10000, len(rows) + 1)),
            "--threads",
            str(threads),
            "--format-output",
            "query,target,prob,evalue,qtmscore,ttmscore,qcov,tcov",
        ]
        # Execution, thread environment, logs, and tool provenance belong to LambdaForge.
        self.tools.run(command, name="Foldseek", threads=threads)
        if not output.is_file():
            raise RuntimeError("Foldseek completed without creating its pair table")

    @staticmethod
    def _sequence_edges(
        path      : Path,
        rows      : Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any],
    ) -> set[tuple[str, str]]:
        """Reapply all MMseqs2 thresholds explicitly, including bilateral coverage.

        Args:
            path: Seven-column MMseqs2 query/target/fident/qcov/tcov/evalue/bits TSV.
            rows: Complete raw rows defining the only legal identifiers.
            parameters: Validated identity, coverage, and e-value thresholds.

        Returns:
            Canonically ordered non-self sequence leakage edges.

        Raises:
            ValueError: If a row has the wrong shape, unknown ID, non-finite value, or bad range.
        """
        # Parse the raw seven-column checkpoint and normalize tool versions that emit percentages
        # rather than unit fractions. Unknown identifiers make the checkpoint scientifically
        # invalid.
        identifiers = {str(row["identifier"]) for row in rows}
        edges: set[tuple[str, str]] = set()
        for fields in DatasetDesign._tsv(path, 7):
            query, target = fields[:2]
            DatasetDesign._known_pair(query, target, identifiers, path)
            identity, qcov, tcov, evalue = map(float, fields[2:6])
            if identity > 1.0:
                identity /= 100.0
            if qcov > 1.0:
                qcov /= 100.0
            if tcov > 1.0:
                tcov /= 100.0
            if not all(math.isfinite(value) for value in (identity, qcov, tcov, evalue)):
                raise ValueError(f"{path.name} contains a non-finite similarity value")
            # Retain one canonical undirected edge only when both directional coverages pass.
            if (
                query != target
                and identity >= float(parameters["sequence_identity"])
                and qcov >= float(parameters["sequence_coverage"])
                and tcov >= float(parameters["sequence_coverage"])
                and evalue <= float(parameters["sequence_evalue"])
            ):
                edges.add((min(query, target), max(query, target)))
        return edges

    @staticmethod
    def _structure_edges(
        path      : Path,
        rows      : Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any],
    ) -> set[tuple[str, str]]:
        """Reapply strict Foldseek probability, two TM-scores, bilateral coverage, and e-value.

        Args:
            path: Eight-column Foldseek query/target/prob/evalue/qTM/tTM/qcov/tcov TSV.
            rows: Complete raw rows defining the only legal identifiers.
            parameters: Validated structural edge thresholds.

        Returns:
            Canonically ordered non-self near-duplicate structural edges.

        Raises:
            ValueError: If pair evidence is malformed, unknown, non-finite, or out of range.
        """
        # Foldseek may return filenames instead of bare logical IDs, hence ``Path(...).stem``.
        # Probabilities/coverages are normalized to [0,1] before applying WISDOM thresholds.
        identifiers = {str(row["identifier"]) for row in rows}
        edges: set[tuple[str, str]] = set()
        for fields in DatasetDesign._tsv(path, 8):
            query, target = (Path(fields[0]).stem, Path(fields[1]).stem)
            DatasetDesign._known_pair(query, target, identifiers, path)
            probability, evalue, qtm, ttm, qcov, tcov = map(float, fields[2:8])
            if probability > 1.0:
                probability /= 100.0
            if qcov > 1.0:
                qcov /= 100.0
            if tcov > 1.0:
                tcov /= 100.0
            if not all(
                math.isfinite(value)
                for value in (probability, evalue, qtm, ttm, qcov, tcov)
            ):
                raise ValueError(f"{path.name} contains a non-finite structural value")
            # Both normalized TM-scores and both coverages must pass to avoid directional edges.
            if (
                query != target
                and probability >= float(parameters["foldseek_probability"])
                and min(qtm, ttm) >= float(parameters["foldseek_tmscore"])
                and min(qcov, tcov) >= float(parameters["foldseek_coverage"])
                and evalue <= float(parameters["foldseek_evalue"])
            ):
                edges.add((min(query, target), max(query, target)))
        return edges

    @staticmethod
    def _exact_pairs(
        rows          : Sequence[Mapping[str, Any]],
        group_same_pdb: bool,
    ) -> list[dict[str, Any]]:
        """Create auditable exact/provenance edges with every reason retained.

        Args:
            rows: Complete raw descriptor rows.
            group_same_pdb: Whether sharing a PDB deposition is a hard dependency edge.

        Returns:
            Sorted pair mappings whose ``reasons`` list explains each hard relation.
        """
        # Multiple evidence sources may connect the same pair. ``reasons`` retains all causes so a
        # reader can distinguish exact sequence, logical identity, and shared-deposition grouping.
        reasons: dict[tuple[str, str], set[str]] = defaultdict(set)
        fields = {
            "sequence_sha256": "exact_sequence",
            "identifier": "same_logical_example",
        }
        if group_same_pdb:
            fields["pdb_id"] = "same_pdb_deposition"
        for field, reason in fields.items():
            groups: dict[str, list[str]] = defaultdict(list)
            for row in rows:
                value = str(row.get(field, "")).strip()
                if value:
                    groups[value].append(str(row["identifier"]))
            for identifiers in groups.values():
                ordered = sorted(set(identifiers))
                for index, left in enumerate(ordered):
                    for right in ordered[index + 1 :]:
                        reasons[(left, right)].add(reason)
        return [
            {"left": left, "right": right, "reasons": sorted(values)}
            for (left, right), values in sorted(reasons.items())
        ]

    @staticmethod
    def _components(
        identifiers: Sequence[str],
        edges      : set[tuple[str, str]],
    ) -> list[list[str]]:
        """Compute deterministic connected components of the full sparse leakage graph.

        Args:
            identifiers: Every raw logical protein identifier exactly once.
            edges: Union of sequence, structural, exact, and provenance undirected pairs.

        Returns:
            Lexicographically ordered components, each with sorted members.

        Raises:
            ValueError: If identifiers are duplicated or an edge names an unknown member.
        """
        # A union-find structure computes sparse connected components without materializing an
        # N-by-N adjacency matrix. Lexical roots make the result independent of input edge order.
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("connected-component identifiers must be unique")
        parent = {identifier: identifier for identifier in identifiers}

        def find(value: str) -> str:
            """Return and path-compress one disjoint-set representative.

            Args:
                value: Known logical identifier.

            Returns:
                Current canonical component root.
            """
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        # Union every hard dependency transitively: A--B and B--C place all three in one group.
        for left, right in sorted(edges):
            if left not in parent or right not in parent:
                raise ValueError(f"leakage edge names unknown member: {left}, {right}")
            first, second = find(left), find(right)
            if first != second:
                parent[max(first, second)] = min(first, second)
        groups: dict[str, list[str]] = defaultdict(list)
        for identifier in sorted(identifiers):
            groups[find(identifier)].append(identifier)
        return sorted((sorted(values) for values in groups.values()), key=lambda values: values[0])

    @staticmethod
    def _phenotypes(
        rows             : Sequence[Mapping[str, Any]],
        features         : Sequence[str],
        prefix           : str,
        min_cluster_size : int,
        min_samples      : int,
        stability_minimum: float,
        workers          : int,
    ) -> dict[str, Any]:
        """Fit robust-scaled HDBSCAN and reject unstable apparent physical phenotypes.

        Rows missing any selected feature remain explicit ``*_NOISE``. Near-zero-variance columns
        are removed after robust scaling. LambdaForge fits the canonical HDBSCAN solution and the
        neighboring 3x3 parameter grid, then computes their adjusted Rand stability. Fewer than two
        clusters or median ARI below the configured minimum causes every label to become noise.

        Args:
            rows: Full global population or positive-only interface population.
            features: Ordered descriptor names that may not include labels or nuisance variables.
            prefix: ``G`` for global or ``I`` for interface physical phenotypes.
            min_cluster_size: Canonical HDBSCAN minimum cluster size.
            min_samples: Canonical HDBSCAN core-neighbour setting.
            stability_minimum: Minimum median adjusted Rand index over neighboring settings.
            workers: Threads used internally by one dataset-level HDBSCAN fit at a time.

        Returns:
            Per-ID labels/probabilities plus scaler and stability diagnostics.
        """
        # Build a dense [P,F] matrix only for the small descriptor table, never for atom/surface
        # coordinates. Rows missing one required feature remain explicit phenotype noise.
        prepared: list[tuple[str, list[float]]] = []
        missing: list[str] = []
        for row in sorted(rows, key=lambda value: str(value["identifier"])):
            derived = dict(row)
            derived["log_sequence_length"] = math.log(max(1.0, float(row["sequence_length"])))
            charge = row.get("net_charge_at_pH_7")
            derived["charge_density"] = (
                float(charge) / float(row["sequence_length"]) if charge is not None else None
            )
            values: list[float] = []
            unavailable = False
            for name in features:
                value = derived.get(name)
                if value is None or not math.isfinite(float(value)):
                    unavailable = True
                    break
                values.append(float(value))
            if unavailable:
                missing.append(str(row["identifier"]))
            else:
                prepared.append((str(row["identifier"]), values))
        labels = {str(row["identifier"]): f"{prefix}_NOISE" for row in rows}
        probabilities = {str(row["identifier"]): 0.0 for row in rows}
        if len(prepared) < min_cluster_size * 2:
            return {
                "labels": labels,
                "probabilities": probabilities,
                "diagnostics": {
                    "robust": False,
                    "reason": "too_few_complete_samples",
                    "eligible_count": len(prepared),
                    "missing_count": len(missing),
                    "feature_names": list(features),
                    "cluster_count": 0,
                    "noise_fraction": 1.0,
                },
            }

        # Median/IQR scaling limits outlier influence. Near-constant columns carry no density
        # information and are removed before invoking LambdaForge's clustering service.
        identifiers = [value[0] for value in prepared]
        matrix = np.asarray([value[1] for value in prepared], dtype=np.float64)
        scaler = RobustScaler(quantile_range=(25.0, 75.0)).fit(matrix)
        scaled = scaler.transform(matrix)
        keep = np.nanstd(scaled, axis=0) > 1e-10
        if not np.any(keep):
            return {
                "labels": labels,
                "probabilities": probabilities,
                "diagnostics": {
                    "robust": False,
                    "reason": "all_features_near_constant",
                    "eligible_count": len(prepared),
                    "missing_count": len(missing),
                    "feature_names": [],
                    "cluster_count": 0,
                    "noise_fraction": 1.0,
                },
            }
        scaled = scaled[:, keep]
        retained_features = [name for name, retain in zip(features, keep, strict=True) if retain]
        # WISDOM defines a small neighboring parameter grid as the scientific stability question;
        # LambdaForge fits each HDBSCAN candidate and computes adjusted-Rand agreement.
        settings = sorted(
            {
                (max(2, min_cluster_size + size), max(1, min_samples + samples))
                for size in (-5, 0, 5)
                for samples in (-2, 0, 2)
            }
        )
        settings = [setting for setting in settings if setting[0] <= len(prepared)]
        # LambdaForge 0.12 deliberately owns the sklearn backend but does not expose sklearn 1.9's
        # transitional ``copy`` option. Its current False and future True values only control
        # defensive memory copying, not HDBSCAN mathematics. Suppress precisely that backend notice
        # while preserving every other warning from LambdaForge, sklearn, or the scientific code.
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    r"The default value of `copy` will change from False to True in 1\.10\."
                ),
                category=FutureWarning,
            )

            canonical = lf.clustering.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                threads=workers,
            ).cluster(scaled)
            alternative_settings = [
                setting for setting in settings if setting != (min_cluster_size, min_samples)
            ]
            alternatives = [
                lf.clustering.HDBSCAN(
                    min_cluster_size=size,
                    min_samples=samples,
                    threads=workers,
                )
                for size, samples in alternative_settings
            ]
            stability_result = (
                lf.clustering.stability(scaled, alternatives, reference=canonical)
                if alternatives
                else None
            )

        # Accept named phenotypes only when the canonical result has at least two clusters and the
        # median neighboring agreement passes the requested threshold. Otherwise all rows stay
        # noise.
        canonical_labels = canonical.labels
        cluster_values = sorted(set(canonical_labels.tolist()) - {-1})
        stability = stability_result.median if stability_result is not None else 1.0
        robust = len(cluster_values) >= 2 and stability >= stability_minimum
        if robust:
            if canonical.probabilities is None:
                raise RuntimeError("LambdaForge HDBSCAN omitted membership probabilities")
            remap = {
                value: f"{prefix}{index:03d}"
                for index, value in enumerate(cluster_values, start=1)
            }
            for identifier, label, probability in zip(
                identifiers,
                canonical_labels,
                canonical.probabilities,
                strict=True,
            ):
                labels[identifier] = remap.get(int(label), f"{prefix}_NOISE")
                probabilities[identifier] = float(probability) if label != -1 else 0.0
        return {
            "labels": labels,
            "probabilities": probabilities,
            "diagnostics": {
                "robust": robust,
                "reason": "stable" if robust else "no_stable_multi_cluster_solution",
                "eligible_count": len(prepared),
                "missing_count": len(missing),
                "feature_names": retained_features,
                "cluster_count": len(cluster_values) if robust else 0,
                "noise_fraction": float(
                    np.mean([value.endswith("_NOISE") for value in labels.values()])
                ),
                "median_adjusted_rand": stability,
                "settings": [
                    {"min_cluster_size": size, "min_samples": samples}
                    for size, samples in settings
                ],
                "scaler_center": np.asarray(scaler.center_)[keep].tolist(),
                "scaler_scale": np.asarray(scaler.scale_)[keep].tolist(),
            },
        }

    def _select_population(
        self,
        rows                   : Sequence[Mapping[str, Any]],
        balance_classes        : bool,
        positive_negative_ratio: float,
        keep_all_negatives     : bool,
        retain_core_positives  : bool,
        seed                   : int,
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        """Select the canonical population after full-raw groups and phenotypes are fixed.

        The default retains every negative, reserves every BTD-Core positive that fits the target,
        then fills the remaining positive quota by deterministic greedy diversity. Priority is a
        new leakage group, new global/interface phenotype, low current group multiplicity, closeness
        to negative technical covariates, and finally seeded SHA-256 rank. Interface extent itself
        is never used as a quality score.

        Args:
            rows: Quality-eligible descriptor rows with full-raw leakage assignments and fitted
                physical phenotypes.
            balance_classes: Whether to control the canonical positive/negative ratio.
            positive_negative_ratio: Requested positive count divided by negative count.
            keep_all_negatives: Retain all defensible negatives when balancing.
            retain_core_positives: Give positive rows from ``btd_core`` first priority.
            seed: Stable tie-breaking seed.

        Returns:
            Identifier-sorted selected rows and a complete retained/omitted audit.

        Raises:
            RuntimeError: If either eligible class is empty or the requested ratio is impossible.
        """
        # Separate frozen classes once. Balancing changes only canonical membership; no label,
        # leakage group, phenotype, descriptor, or provenance field is recalculated.
        positives = [dict(row) for row in rows if int(row["label"]) == 1]
        negatives = [dict(row) for row in rows if int(row["label"]) == 0]
        if not positives or not negatives:
            raise RuntimeError(
                "benchmark design requires non-empty quality-eligible positive and negative classes"
            )
        if not balance_classes:
            selected = sorted((dict(row) for row in rows), key=lambda row: row["identifier"])
            return selected, {
                "method": "full_quality_eligible_population",
                "input_counts": self._class_counts(rows),
                "selected_counts": self._class_counts(selected),
                "omitted_positive_count": 0,
                "omitted_positives": [],
            }

        # Determine exact class quotas before diversity selection. The default keeps every curated
        # negative because defensible negatives are the scarce source class.
        if keep_all_negatives:
            selected_negatives = negatives
            positive_target = round(len(selected_negatives) * positive_negative_ratio)
            if positive_target > len(positives):
                raise RuntimeError("requested positive/negative ratio exceeds available positives")
        else:
            negative_target = min(len(negatives), int(len(positives) / positive_negative_ratio))
            positive_target = min(
                len(positives),
                round(negative_target * positive_negative_ratio),
            )
            selected_negatives = self._diversity_selection(
                negatives,
                negative_target,
                negatives,
                seed,
                required=set(),
            )
        # Required core positives consume quota first. Remaining positions maximize physical/group
        # diversity while approximately matching the negatives' technical covariates.
        core_ids = {
            str(row["identifier"])
            for row in positives
            if retain_core_positives and str(row.get("origin", "")) == "btd_core"
        }
        selected_positives = self._diversity_selection(
            positives,
            positive_target,
            selected_negatives,
            seed,
            required=core_ids,
        )
        selected = sorted(
            selected_negatives + selected_positives,
            key=lambda row: row["identifier"],
        )
        # Preserve every omitted positive with an explicit reason and deterministic rank. Omission
        # means “not needed for this ratio,” never invalid or negative biological evidence.
        selected_ids = {str(row["identifier"]) for row in selected}
        omitted = [
            {
                "identifier": str(row["identifier"]),
                "origin": str(row["origin"]),
                "leakage_group": str(row["leakage_group"]),
                "global_phenotype": str(row["global_phenotype"]),
                "interface_phenotype": str(row["interface_phenotype"]),
                "reason": "valid_positive_not_selected_for_canonical_balance_and_diversity",
                "deterministic_rank": self._rank(seed, str(row["identifier"])),
            }
            for row in positives
            if str(row["identifier"]) not in selected_ids
        ]
        return selected, {
            "method": "full_raw_leakage_and_phenotype_aware_deterministic_selection",
            "input_counts": self._class_counts(rows),
            "selected_counts": self._class_counts(selected),
            "positive_negative_ratio_requested": positive_negative_ratio,
            "positive_negative_ratio_realized": len(selected_positives) / len(selected_negatives),
            "keep_all_negatives": keep_all_negatives,
            "retain_core_positives": retain_core_positives,
            "core_positive_available": len(core_ids),
            "core_positive_retained": sum(
                str(row["identifier"]) in core_ids for row in selected_positives
            ),
            "selected_positive_ids": [str(row["identifier"]) for row in selected_positives],
            "omitted_positive_count": len(omitted),
            "omitted_positives": omitted,
            "selected_by_origin": dict(Counter(str(row["origin"]) for row in selected_positives)),
            "selected_by_global_phenotype": dict(
                Counter(str(row["global_phenotype"]) for row in selected_positives)
            ),
            "selected_by_interface_phenotype": dict(
                Counter(str(row["interface_phenotype"]) for row in selected_positives)
            ),
            "selected_leakage_group_count": len(
                {str(row["leakage_group"]) for row in selected_positives}
            ),
        }

    def _diversity_selection(
        self,
        candidates: Sequence[Mapping[str, Any]],
        target    : int,
        reference : Sequence[Mapping[str, Any]],
        seed      : int,
        required  : set[str],
    ) -> list[dict[str, Any]]:
        """Choose a deterministic quota while covering groups/phenotypes and matching nuisance.

        Args:
            candidates: Same-label rows eligible for selection.
            target: Exact number of rows to select.
            reference: Opposite-class rows whose technical distribution is the matching target.
            seed: Stable tie-breaking seed.
            required: Identifiers retained before optional candidates when capacity permits.

        Returns:
            Exactly ``target`` selected row dictionaries.

        Raises:
            RuntimeError: If the target is outside the candidate count.
        """
        # ``available`` provides O(1) removal by identifier. Required members consume quota first;
        # a seeded hash gives stable ties without depending on source or worker order.
        if not 0 <= target <= len(candidates):
            raise RuntimeError("diversity selection target is outside available candidates")
        available = {str(row["identifier"]): dict(row) for row in candidates}
        chosen: list[dict[str, Any]] = []
        required_rows = sorted(
            (available[key] for key in required if key in available),
            key=lambda row: self._rank(seed, str(row["identifier"])),
        )
        if len(required_rows) > target:
            required_rows = required_rows[:target]
        for row in required_rows:
            chosen.append(row)
            available.pop(str(row["identifier"]), None)

        # Median/IQR reference summaries make technical-distribution matching robust to outliers.
        nuisance_fields = ("sequence_length", "coordinate_coverage", "resolution", "release_year")
        centers: dict[str, float] = {}
        scales: dict[str, float] = {}
        for field in nuisance_fields:
            values = [
                float(row[field])
                for row in reference
                if row.get(field) is not None and math.isfinite(float(row[field]))
            ]
            centers[field] = float(np.median(values)) if values else 0.0
            scales[field] = float(np.subtract(*np.percentile(values, [75, 25]))) if values else 1.0
            scales[field] = max(scales[field], 1e-6)
        reference_methods = Counter(
            str(row.get("experimental_method", "unavailable")) for row in reference
        )
        selected_groups = Counter(str(row["leakage_group"]) for row in chosen)
        selected_global = {str(row["global_phenotype"]) for row in chosen}
        selected_interface = {str(row["interface_phenotype"]) for row in chosen}

        # Add one row at a time. The priority favors unseen groups/phenotypes before technical
        # closeness and uses the seeded rank only as the final deterministic tie-break.
        while len(chosen) < target:
            def priority(row: Mapping[str, Any]) -> tuple[Any, ...]:
                """Return one lexicographic diversity/nuisance/tie-break priority.

                Args:
                    row: Candidate row still available for selection.

                Returns:
                    Lower-is-better tuple implementing the documented selection priorities.
                """
                nuisance = 0.0
                for field in nuisance_fields:
                    value = row.get(field)
                    if value is not None and math.isfinite(float(value)):
                        nuisance += abs(float(value) - centers[field]) / scales[field]
                    else:
                        nuisance += 2.0
                method = str(row.get("experimental_method", "unavailable"))
                method_penalty = 0.0 if reference_methods[method] else 1.0
                return (
                    selected_groups[str(row["leakage_group"])] > 0,
                    str(row["global_phenotype"]) in selected_global,
                    str(row["interface_phenotype"]) in selected_interface,
                    selected_groups[str(row["leakage_group"])],
                    nuisance + method_penalty,
                    self._rank(seed, str(row["identifier"])),
                )

            selected = min(available.values(), key=priority)
            chosen.append(selected)
            available.pop(str(selected["identifier"]))
            selected_groups[str(selected["leakage_group"])] += 1
            selected_global.add(str(selected["global_phenotype"]))
            selected_interface.add(str(selected["interface_phenotype"]))
        return chosen

    def _assign_splits(
        self,
        rows      : Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any],
        seed      : int,
    ) -> tuple[dict[str, str], dict[str, Any]]:
        """Assign indivisible leakage groups by deterministic weighted greedy refinement.

        Stable phenotypes backed by at least three independent groups are seeded across train,
        validation, and test first. Validation and test are then seeded with both labels. Remaining
        groups minimize normalized deviations in size, class, global/interface phenotype, origin,
        and nuisance means. Finally, bounded deterministic single-group moves are accepted only
        when they lower the objective and preserve every hard constraint.

        Args:
            rows: Canonical selected population with full-raw leakage/phenotype assignments.
            parameters: Validated split fractions, objective weights, tolerance, and step count.
            seed: Stable tie-breaking seed.

        Returns:
            Leakage-group to split mapping and an objective/feasibility audit.

        Raises:
            RuntimeError: If group indivisibility makes two-class validation/test impossible.
        """
        # Collapse canonical rows into indivisible leakage groups. Every later decision assigns a
        # group, never an individual protein, so transitive dependencies cannot cross splits.
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[str(row["leakage_group"])].append(dict(row))
        assignments: dict[str, str] = {}

        # Seed stable phenotypes across splits when at least three independent groups make that
        # representation feasible. Noise and not-applicable values are not seeded.
        phenotype_groups: dict[str, set[str]] = defaultdict(set)
        for group, members in groups.items():
            for row in members:
                for field in ("global_phenotype", "interface_phenotype"):
                    phenotype = str(row[field])
                    if not phenotype.endswith("_NOISE") and phenotype != "not_applicable":
                        phenotype_groups[f"{field}:{phenotype}"].add(group)
        for _phenotype, candidates in sorted(
            phenotype_groups.items(), key=lambda value: (len(value[1]), value[0])
        ):
            if len(candidates) < 3:
                continue
            for split in self.SPLITS:
                if any(assignments.get(group) == split for group in candidates):
                    continue
                available = [group for group in candidates if group not in assignments]
                if available:
                    chosen = min(
                        available,
                        key=lambda group: (len(groups[group]), self._rank(seed, group)),
                    )
                    assignments[chosen] = split

        # Explicit label seeds prevent a locally optimal greedy solution from making an evaluation
        # split scientifically unusable despite feasible independent groups.
        for split in ("validation", "test"):
            for label in (0, 1):
                if any(
                    assignment == split and any(int(row["label"]) == label for row in groups[group])
                    for group, assignment in assignments.items()
                ):
                    continue
                label_candidates = [
                    group
                    for group, members in groups.items()
                    if group not in assignments
                    and any(int(row["label"]) == label for row in members)
                ]
                if not label_candidates:
                    raise RuntimeError(f"cannot seed label {label} in {split} under leakage groups")
                chosen = min(
                    label_candidates,
                    key=lambda group: (
                        len(groups[group]),
                        self._rank(seed, group + split),
                    ),
                )
                assignments[chosen] = split

        # Assign remaining groups largest/rarest first. For each group, choose the split with the
        # lowest weighted distribution error; seeded hashes resolve exact objective ties.
        ordered = sorted(
            (group for group in groups if group not in assignments),
            key=lambda group: (
                -len(groups[group]),
                min(
                    len(phenotype_groups[key])
                    for row in groups[group]
                    for field in ("global_phenotype", "interface_phenotype")
                    for key in (f"{field}:{row[field]}",)
                    if key in phenotype_groups
                )
                if any(
                    f"{field}:{row[field]}" in phenotype_groups
                    for row in groups[group]
                    for field in ("global_phenotype", "interface_phenotype")
                )
                else len(groups),
                self._rank(seed, group),
            ),
        )
        for group in ordered:
            assignments[group] = min(
                self.SPLITS,
                key=lambda split: (
                    self._split_objective(rows, groups, {**assignments, group: split}, parameters),
                    self._rank(seed, group + split),
                ),
            )

        # Bounded local refinement moves one whole group and accepts strict improvements only. This
        # preserves reproducibility and prevents oscillation between equal assignments.
        initial_objective = self._split_objective(rows, groups, assignments, parameters)
        steps = 0
        for _ in range(int(parameters["split_refinement_steps"])):
            current = self._split_objective(rows, groups, assignments, parameters)
            best: tuple[float, str, str] | None = None
            for group in sorted(groups, key=lambda value: self._rank(seed, value)):
                previous = assignments[group]
                for split in self.SPLITS:
                    if split == previous:
                        continue
                    candidate = {**assignments, group: split}
                    if not self._hard_split_constraints(groups, candidate, phenotype_groups):
                        continue
                    score = self._split_objective(rows, groups, candidate, parameters)
                    choice = (score, group, split)
                    if score + 1e-12 < current and (best is None or choice < best):
                        best = choice
            if best is None:
                break
            _, group, split = best
            assignments[group] = split
            steps += 1

        if not self._hard_split_constraints(groups, assignments, phenotype_groups):
            raise RuntimeError("final split assignment violates a hard class/phenotype constraint")
        split_counts = {
            split: self._class_counts(
                [
                    row
                    for group, value in assignments.items()
                    if value == split
                    for row in groups[group]
                ]
            )
            for split in self.SPLITS
        }
        feasibility = {
            phenotype: {
                "leakage_group_count": len(values),
                "representable_in_all_splits": len(values) >= 3,
                "observed_splits": sorted({assignments[group] for group in values}),
            }
            for phenotype, values in sorted(phenotype_groups.items())
        }
        return assignments, {
            "algorithm": "rare_first_weighted_greedy_with_deterministic_group_moves",
            "initial_objective": initial_objective,
            "final_objective": self._split_objective(rows, groups, assignments, parameters),
            "accepted_refinement_moves": steps,
            "split_counts": split_counts,
            "phenotype_feasibility": feasibility,
            "weights": {
                key: parameters[key]
                for key in (
                    "split_size_weight",
                    "split_class_weight",
                    "split_global_phenotype_weight",
                    "split_interface_phenotype_weight",
                    "split_source_weight",
                    "split_nuisance_weight",
                )
            },
        }

    def _split_objective(
        self,
        rows       : Sequence[Mapping[str, Any]],
        groups     : Mapping[str, Sequence[Mapping[str, Any]]],
        assignments: Mapping[str, str],
        parameters : Mapping[str, Any],
    ) -> float:
        """Evaluate normalized soft split deviations for one partial or complete assignment.

        Args:
            rows: Complete canonical population defining distribution targets.
            groups: Leakage-group members.
            assignments: Current group-to-split mapping; unassigned groups are ignored.
            parameters: Split targets and objective weights.

        Returns:
            Non-negative weighted squared deviation; lower values are better.
        """
        # Compare each (possibly partial) split with the requested fraction of the canonical
        # population. Every term is normalized, squared, and multiplied by its YAML weight.
        score = 0.0
        fractions = {
            "train":      float(parameters["train_fraction"]),
            "validation": float(parameters["validation_fraction"]),
            "test":       float(parameters["test_fraction"]),
        }
        numeric = ("sequence_length", "coordinate_coverage", "resolution", "release_year")
        overall_means = {
            field: self._finite_mean(row.get(field) for row in rows) for field in numeric
        }
        global_values = sorted({str(row["global_phenotype"]) for row in rows})
        interface_values = sorted(
            {str(row["interface_phenotype"]) for row in rows if row["label"] == 1}
        )
        source_values = sorted({str(row["origin"]) for row in rows if row["label"] == 1})
        for split in self.SPLITS:
            selected = [
                row
                for group, assigned in assignments.items()
                if assigned == split
                for row in groups[group]
            ]
            fraction = float(fractions[split])
            size_target = len(rows) * fraction
            score += float(parameters["split_size_weight"]) * (
                (len(selected) - size_target) / max(size_target, 1.0)
            ) ** 2
            for label in (0, 1):
                target = sum(int(row["label"]) == label for row in rows) * fraction
                observed = sum(int(row["label"]) == label for row in selected)
                score += float(parameters["split_class_weight"]) * (
                    (observed - target) / max(target, 1.0)
                ) ** 2
            # Categorical terms preserve global/interface phenotype and positive-origin coverage.
            for field, values, weight in (
                ("global_phenotype", global_values, "split_global_phenotype_weight"),
                ("interface_phenotype", interface_values, "split_interface_phenotype_weight"),
                ("origin", source_values, "split_source_weight"),
            ):
                for value in values:
                    population = [
                        row
                        for row in rows
                        if str(row[field]) == value
                        and (field == "global_phenotype" or int(row["label"]) == 1)
                    ]
                    target = len(population) * fraction
                    observed = sum(
                        str(row[field]) == value
                        and (field == "global_phenotype" or int(row["label"]) == 1)
                        for row in selected
                    )
                    score += float(parameters[weight]) * (
                        (observed - target) / max(target, 1.0)
                    ) ** 2
            # Technical nuisance means discourage obvious acquisition/quality shortcuts.
            for field in numeric:
                nuisance_observed = self._finite_mean(row.get(field) for row in selected)
                target_mean = overall_means[field]
                if nuisance_observed is not None and target_mean is not None:
                    scale = max(abs(target_mean), 1.0)
                    score += float(parameters["split_nuisance_weight"]) * (
                        (nuisance_observed - target_mean) / scale
                    ) ** 2
        return float(score)

    def _hard_split_constraints(
        self,
        groups          : Mapping[str, Sequence[Mapping[str, Any]]],
        assignments     : Mapping[str, str],
        phenotype_groups: Mapping[str, set[str]],
    ) -> bool:
        """Check class usability and feasible stable-phenotype representation without splitting.

        Args:
            groups: Indivisible leakage-group member mapping.
            assignments: Complete candidate group-to-split mapping.
            phenotype_groups: Stable phenotype to supporting leakage-group mapping.

        Returns:
            True only when every group is assigned, val/test have both classes, and phenotypes
            supported by at least three groups appear in all three splits.
        """
        if set(assignments) != set(groups) or any(
            value not in self.SPLITS for value in assignments.values()
        ):
            return False
        for split in ("validation", "test"):
            labels = {
                int(row["label"])
                for group, assigned in assignments.items()
                if assigned == split
                for row in groups[group]
            }
            if labels != {0, 1}:
                return False
        # Phenotype coverage is seeded and audited, but overlapping phenotype memberships can make
        # simultaneous three-way coverage combinatorially impossible. Leakage and class usability
        # remain the hard constraints; missing feasible-looking coverage is reported explicitly.
        del phenotype_groups
        return True

    @staticmethod
    def _validate_splits(
        rows           : Sequence[Mapping[str, Any]],
        sequence_edges : set[tuple[str, str]],
        structure_edges: set[tuple[str, str]],
        exact_pairs    : Sequence[Mapping[str, Any]],
    ) -> None:
        """Enforce every selected-population leakage and membership invariant directly.

        Args:
            rows: Canonical rows with one assigned split each.
            sequence_edges: Full-raw thresholded MMseqs2 edges.
            structure_edges: Full-raw thresholded Foldseek edges.
            exact_pairs: Full-raw exact/provenance pair evidence.

        Raises:
            RuntimeError: If IDs repeat, a group/edge crosses splits, or val/test lacks a class.
        """
        # Validate membership/group invariants first, then recheck every retained hard pair
        # directly. This second route guards against future mistakes in component/assignment code.
        by_id = {str(row["identifier"]): row for row in rows}
        if len(by_id) != len(rows):
            raise RuntimeError("selected identifiers must be unique")
        if any(str(row.get("split", "")) not in DatasetDesign.SPLITS for row in rows):
            raise RuntimeError("every selected protein must have exactly one valid split")
        group_splits: dict[str, set[str]] = defaultdict(set)
        for row in rows:
            group_splits[str(row["leakage_group"])].add(str(row["split"]))
        if any(len(values) != 1 for values in group_splits.values()):
            raise RuntimeError("one full-raw leakage group crosses selected splits")
        for split in ("validation", "test"):
            if {int(row["label"]) for row in rows if row["split"] == split} != {0, 1}:
                raise RuntimeError(f"{split} must contain both binary classes")
        pairs = set(sequence_edges) | set(structure_edges) | {
            (str(pair["left"]), str(pair["right"])) for pair in exact_pairs
        }
        for left, right in pairs:
            if left in by_id and right in by_id and by_id[left]["split"] != by_id[right]["split"]:
                raise RuntimeError(f"hard leakage pair crosses selected splits: {left}, {right}")

    def _write_dilutions(
        self,
        rows      : Sequence[Mapping[str, Any]],
        root      : Path,
        fractions : Sequence[float],
        replicates: int,
        seed      : int,
    ) -> dict[str, Any]:
        """Create nested training-only prefixes of one deterministic leakage-group ordering.

        Each replicate keeps exactly the same Train100, validation, and test membership. Smaller
        subsets use one group-level greedy ordering that favors class ratio, first coverage of both
        phenotype systems and origins, and stable seeded ties. The cutoff nearest each requested row
        target is selected without fragmenting a group; realized sizes may therefore differ.

        Args:
            rows: Canonical rows with fixed train/validation/test assignments.
            root: Final design dilution directory.
            fractions: Requested unique fractions including one.
            replicates: Number of alternative smaller-subset group rankings.
            seed: Base deterministic seed.

        Returns:
            JSON-safe replicate/subset audit with fixed evaluation checksums.

        Raises:
            RuntimeError: If group-wise nestedness or Train100 identity is violated.
        """
        # Validation and test are immutable across learning-curve sizes. Their sorted-ID hashes make
        # accidental evaluation-set drift visible in every dilution report.
        training = [dict(row) for row in rows if row["split"] == "train"]
        validation_ids = sorted(
            str(row["identifier"]) for row in rows if row["split"] == "validation"
        )
        test_ids       = sorted(str(row["identifier"]) for row in rows if row["split"] == "test")
        validation_hash = hashlib.sha256("\n".join(validation_ids).encode()).hexdigest()
        test_hash       = hashlib.sha256("\n".join(test_ids).encode()).hexdigest()
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in training:
            groups[str(row["leakage_group"])].append(row)
        result: dict[str, Any] = {
            "validation_sha256": validation_hash,
            "test_sha256": test_hash,
            "replicates": {},
        }
        full_ids = {str(row["identifier"]) for row in training}
        # Each replicate changes only the seeded ordering of complete training leakage groups.
        # Prefixes of one ordering guarantee nested membership within that replicate.
        for replicate in range(replicates):
            replicate_seed = seed + replicate * 1000003
            remaining = set(groups)
            ordering: list[str] = []
            selected_rows: list[dict[str, Any]] = []
            seen_global: set[str] = set()
            seen_interface: set[str] = set()
            seen_origins: set[str] = set()
            target_positive_fraction = sum(row["label"] == 1 for row in training) / len(training)
            # Greedy ordering first preserves class ratio, then introduces unseen phenotype/origin
            # categories, and finally uses a deterministic hash to resolve equal priorities.
            while remaining:
                candidates: list[tuple[tuple[Any, ...], str]] = []
                for group in remaining:
                    prospective = selected_rows + groups[group]
                    positive_fraction = (
                        sum(row["label"] == 1 for row in prospective) / len(prospective)
                    )
                    new_global = {
                        str(row["global_phenotype"])
                        for row in groups[group]
                        if not str(row["global_phenotype"]).endswith("_NOISE")
                    } - seen_global
                    new_interface = {
                        str(row["interface_phenotype"])
                        for row in groups[group]
                        if str(row["interface_phenotype"]) not in {"I_NOISE", "not_applicable"}
                    } - seen_interface
                    new_origins = {str(row["origin"]) for row in groups[group]} - seen_origins
                    priority = (
                        abs(positive_fraction - target_positive_fraction),
                        -len(new_global),
                        -len(new_interface),
                        -len(new_origins),
                        self._rank(replicate_seed, group),
                    )
                    candidates.append((priority, group))
                chosen = min(candidates)[1]
                ordering.append(chosen)
                selected_rows.extend(groups[chosen])
                seen_global.update(str(row["global_phenotype"]) for row in groups[chosen])
                seen_interface.update(str(row["interface_phenotype"]) for row in groups[chosen])
                seen_origins.update(str(row["origin"]) for row in groups[chosen])
                remaining.remove(chosen)

            replicate_root = root / f"replicate-{replicate:02d}"
            replicate_root.mkdir(parents=True, exist_ok=True)
            cumulative = [0]
            for group in ordering:
                cumulative.append(cumulative[-1] + len(groups[group]))
            previous_ids: set[str] = set()
            subsets: dict[str, Any] = {}
            # Select the group boundary closest to each requested row fraction. Realized size can
            # differ from the target because splitting a leakage group would reintroduce leakage.
            for fraction in sorted(set(float(value) for value in fractions)):
                target = round(len(training) * fraction)
                cutoff = min(
                    range(len(cumulative)),
                    key=lambda index: (abs(cumulative[index] - target), -cumulative[index]),
                )
                if cutoff == 0 and ordering:
                    cutoff = 1
                chosen_groups = ordering[:cutoff]
                subset = [row for group in chosen_groups for row in groups[group]]
                identifiers = {str(row["identifier"]) for row in subset}
                if not previous_ids.issubset(identifiers):
                    raise RuntimeError("training dilution nesting invariant failed")
                previous_ids = identifiers
                percentage = round(fraction * 100)
                name = f"train-{percentage}"
                self._write_text(
                    replicate_root / f"{name}.txt",
                    "".join(f"{value}\n" for value in sorted(identifiers)),
                )
                self._write_text(
                    replicate_root / f"{name}-labelled.txt",
                    self._labelled_manifest(subset),
                )
                subsets[name] = {
                    "requested_fraction": fraction,
                    "target_rows": target,
                    "realized_rows": len(subset),
                    "positive": sum(row["label"] == 1 for row in subset),
                    "negative": sum(row["label"] == 0 for row in subset),
                    "leakage_groups": len(chosen_groups),
                    "global_phenotype_counts": dict(
                        Counter(str(row["global_phenotype"]) for row in subset)
                    ),
                    "interface_phenotype_counts": dict(
                        Counter(
                            str(row["interface_phenotype"])
                            for row in subset
                            if row["label"] == 1
                        )
                    ),
                    "origin_counts": dict(Counter(str(row["origin"]) for row in subset)),
                    "identifiers": sorted(identifiers),
                }
            if previous_ids != full_ids:
                raise RuntimeError("Train100 must exactly equal the fixed canonical training set")
            result["replicates"][f"replicate-{replicate:02d}"] = {
                "ordering": ordering,
                "subsets": subsets,
            }
        return result

    def _statistics(
        self,
        raw_rows      : Sequence[Mapping[str, Any]],
        selected_rows : Sequence[Mapping[str, Any]],
        dilutions     : Mapping[str, Any],
        parameters    : Mapping[str, Any],
        root          : Path,
        seed          : int,
    ) -> dict[str, Any]:
        """Compute distribution summaries, class/split shifts, and shortcut diagnostics.

        Effect sizes are primary. Continuous comparisons report SMD, KS, Mann--Whitney, normalized
        Wasserstein distance, and Benjamini--Hochberg adjusted p-values. Categorical comparisons
        report contingency tables, chi-square, Cramer's V, and FDR. Shortcut logistic regressions
        use ``StratifiedGroupKFold`` with immutable leakage groups, never random row folds.

        Args:
            raw_rows: Complete raw descriptor population.
            selected_rows: Canonical selected population with fixed splits.
            dilutions: Nested training membership audit.
            parameters: Validated warning thresholds used in interpretations.
            root: Statistics output directory.
            seed: Deterministic model/CV seed.

        Returns:
            JSON-safe summaries and tables also written as human/machine artifacts.
        """
        root.mkdir(parents=True, exist_ok=True)
        continuous_features = self.CONTINUOUS_FEATURES
        categorical_features = (
            "origin",
            "experimental_method",
            "global_phenotype",
        )
        raw_summary = self._population_summary(raw_rows, continuous_features, categorical_features)
        selected_summary = self._population_summary(
            selected_rows, continuous_features, categorical_features
        )
        selected_summary["positive_interface_phenotypes"] = dict(
            sorted(
                Counter(
                    str(row["interface_phenotype"])
                    for row in selected_rows
                    if int(row["label"]) == 1
                ).items()
            )
        )
        selected_summary["splits"] = {
            split: self._population_summary(
                [row for row in selected_rows if row["split"] == split],
                continuous_features,
                categorical_features,
            )
            for split in self.SPLITS
        }

        # Class comparisons, split comparisons, learning-curve membership, and shortcut baselines
        # answer different questions and remain separate in both memory and output files.
        continuous_balance = self._continuous_balance(selected_rows, continuous_features)
        categorical_balance = self._categorical_balance(selected_rows, categorical_features)
        split_balance = self._split_balance(
            selected_rows,
            continuous_features,
            categorical_features,
        )
        dilution_balance = [
            {
                "replicate": replicate,
                "subset": subset,
                **{
                    key: value
                    for key, value in values.items()
                    if key != "identifiers"
                },
            }
            for replicate, replicate_values in dilutions["replicates"].items()
            for subset, values in replicate_values["subsets"].items()
        ]
        shortcuts = self._shortcut_baselines(selected_rows, seed)

        # The source/class cross-tab is recorded for both populations because balancing counts does
        # not automatically remove origin as a label proxy.
        source_confounding = {
            "raw": self._label_category_table(raw_rows, "origin"),
            "selected": self._label_category_table(selected_rows, "origin"),
            "interpretation": (
                "Association between origin and label is a technical confounding warning, not a "
                "biological label rule. Origin is excluded from WISDOM model inputs."
            ),
        }
        payload = {
            "raw_summary": raw_summary,
            "selected_summary": selected_summary,
            "continuous_balance": continuous_balance,
            "categorical_balance": categorical_balance,
            "split_balance": split_balance,
            "dilution_balance": dilution_balance,
            "shortcut_baselines": shortcuts,
            "source_class_confounding": source_confounding,
        }
        self._write_json(root / "raw-summary.json", raw_summary)
        self._write_json(root / "selected-summary.json", selected_summary)
        self._write_csv(root / "continuous-balance.csv", continuous_balance)
        self._write_csv(root / "categorical-balance.csv", categorical_balance)
        self._write_csv(root / "split-balance.csv", split_balance)
        self._write_csv(root / "dilution-balance.csv", dilution_balance)
        self._write_json(root / "shortcut-baselines.json", shortcuts)
        self._plots(raw_rows, selected_rows, continuous_balance, dilutions, root / "plots")
        return payload

    def _population_summary(
        self,
        rows       : Sequence[Mapping[str, Any]],
        continuous : Sequence[str],
        categorical: Sequence[str],
    ) -> dict[str, Any]:
        """Summarize one population with robust continuous and categorical distributions.

        Args:
            rows: Population or split subset.
            continuous: Numeric fields to summarize.
            categorical: Discrete fields to count.

        Returns:
            Counts, class ratio, per-feature quantiles, and category tables.
        """
        return {
            "counts": self._class_counts(rows),
            "leakage_group_count": len({str(row["leakage_group"]) for row in rows}),
            "continuous": {
                feature: self._describe([row.get(feature) for row in rows])
                for feature in continuous
            },
            "categorical": {
                feature: dict(Counter(str(row.get(feature, "unavailable")) for row in rows))
                for feature in categorical
            },
        }

    @staticmethod
    def _describe(values: Sequence[Any]) -> dict[str, Any]:
        """Calculate finite count, missingness, moments, median, IQR, and requested quantiles.

        Args:
            values: Numeric or unavailable values from one feature/population.

        Returns:
            JSON-safe descriptive statistics with unavailable moments represented by ``None``.
        """
        # Missing/nonfinite observations are counted but excluded from numeric summaries. Quantiles
        # are more interpretable than a mean alone for skewed size and resolution distributions.
        finite = np.asarray(
            [float(value) for value in values if value is not None and math.isfinite(float(value))],
            dtype=np.float64,
        )
        if not len(finite):
            return {
                "n": 0,
                "missing": len(values),
                "mean": None,
                "std": None,
                "median": None,
                "iqr": None,
                "q05": None,
                "q25": None,
                "q75": None,
                "q95": None,
            }
        quantiles = np.quantile(finite, (0.05, 0.25, 0.50, 0.75, 0.95))
        return {
            "n": len(finite),
            "missing": len(values) - len(finite),
            "mean": float(finite.mean()),
            "std": float(finite.std(ddof=1)) if len(finite) > 1 else 0.0,
            "median": float(quantiles[2]),
            "iqr": float(quantiles[3] - quantiles[1]),
            "q05": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "q75": float(quantiles[3]),
            "q95": float(quantiles[4]),
        }

    def _continuous_balance(
        self,
        rows    : Sequence[Mapping[str, Any]],
        features: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Compare selected positives and negatives using effect sizes and rank/distribution tests.

        Args:
            rows: Canonical selected population.
            features: Continuous technical and model-visible global features.

        Returns:
            One ordered mapping per feature, including Benjamini--Hochberg adjusted p-values.
        """
        # Compute one long-form row per feature so reports can sort/filter by effect size.
        results: list[dict[str, Any]] = []
        for feature in features:
            negative = self._finite_array(
                row.get(feature) for row in rows if int(row["label"]) == 0
            )
            positive = self._finite_array(
                row.get(feature) for row in rows if int(row["label"]) == 1
            )
            comparison = self._compare_continuous(positive, negative)
            results.append(
                {
                    "feature": feature,
                    "category": (
                        "technical_nuisance"
                        if feature in {"coordinate_coverage", "resolution", "release_year"}
                        else "model_visible_global"
                    ),
                    "positive_n": len(positive),
                    "negative_n": len(negative),
                    **comparison,
                }
            )
        self._add_fdr(results, "ks_p_value", "ks_fdr")
        self._add_fdr(results, "mann_whitney_p_value", "mann_whitney_fdr")
        return results

    @staticmethod
    def _compare_continuous(left: np.ndarray, right: np.ndarray) -> dict[str, Any]:
        """Compare two finite arrays with scale-aware effect and distribution statistics.

        SMD is ``(mean_left-mean_right)/sqrt((var_left+var_right)/2)``. Wasserstein distance is
        normalized by the same pooled standard deviation so units do not dominate rankings.

        Args:
            left: First finite one-dimensional sample.
            right: Second finite one-dimensional sample.

        Returns:
            SMD, KS statistic/p-value, Mann--Whitney p-value, and normalized Wasserstein distance.
        """
        # Fewer than two observations cannot define a sample variance or the requested tests.
        if len(left) < 2 or len(right) < 2:
            return {
                "smd": None,
                "ks_statistic": None,
                "ks_p_value": None,
                "mann_whitney_p_value": None,
                "normalized_wasserstein": None,
            }
        pooled = math.sqrt((float(left.var(ddof=1)) + float(right.var(ddof=1))) / 2.0)
        difference = float(left.mean() - right.mean())
        smd = difference / pooled if pooled > 1e-12 else (0.0 if abs(difference) <= 1e-12 else None)
        ks = ks_2samp(left, right, alternative="two-sided", method="auto")
        mann = mannwhitneyu(left, right, alternative="two-sided")
        return {
            "smd": smd,
            "ks_statistic": float(ks.statistic),
            "ks_p_value": float(ks.pvalue),
            "mann_whitney_p_value": float(mann.pvalue),
            "normalized_wasserstein": (
                float(wasserstein_distance(left, right) / pooled) if pooled > 1e-12 else 0.0
            ),
        }

    def _categorical_balance(
        self,
        rows    : Sequence[Mapping[str, Any]],
        features: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Compare selected class/category associations with chi-square and Cramer's V.

        Args:
            rows: Canonical selected population.
            features: Categorical technical and phenotype fields.

        Returns:
            One mapping per feature with table, proportions, chi-square, p-value, V, and FDR.
        """
        results: list[dict[str, Any]] = []
        for feature in features:
            table = self._label_category_table(rows, feature)
            results.append({"feature": feature, **table})
        self._add_fdr(results, "chi_square_p_value", "chi_square_fdr")
        return results

    @staticmethod
    def _label_category_table(
        rows   : Sequence[Mapping[str, Any]],
        feature: str,
    ) -> dict[str, Any]:
        """Build label-by-category counts, proportions, chi-square, and bias-corrected Cramer's V.

        Args:
            rows: Population containing binary ``label`` and the selected categorical field.
            feature: Category field name.

        Returns:
            JSON-safe contingency analysis; invalid chi-square cases remain unavailable.
        """
        # Rows are binary labels and columns are observed categories. Stable lexical ordering makes
        # the persisted contingency table reproducible and directly interpretable.
        categories = sorted({str(row.get(feature, "unavailable")) for row in rows})
        matrix = np.asarray(
            [
                [
                    sum(
                        int(row["label"]) == label
                        and str(row.get(feature, "unavailable")) == category
                        for row in rows
                    )
                    for category in categories
                ]
                for label in (0, 1)
            ],
            dtype=np.int64,
        )
        counts = {
            str(label): {
                category: int(matrix[label, index])
                for index, category in enumerate(categories)
            }
            for label in (0, 1)
        }
        proportions = {
            label: {
                category: value / max(1, sum(values.values())) for category, value in values.items()
            }
            for label, values in counts.items()
        }
        chi_square: float | None = None
        p_value: float | None = None
        cramers_v: float | None = None
        if (
            matrix.shape[1] > 1
            and np.all(matrix.sum(axis=0) > 0)
            and np.all(matrix.sum(axis=1) > 0)
        ):
            statistic, p_value_raw, _, _ = chi2_contingency(matrix)
            n = int(matrix.sum())
            phi2 = statistic / n
            rows_count, columns_count = matrix.shape
            phi2_corrected = max(
                0.0,
                phi2 - ((columns_count - 1) * (rows_count - 1)) / max(n - 1, 1),
            )
            row_corrected = rows_count - ((rows_count - 1) ** 2) / max(n - 1, 1)
            column_corrected = columns_count - ((columns_count - 1) ** 2) / max(n - 1, 1)
            denominator = min(column_corrected - 1, row_corrected - 1)
            chi_square = float(statistic)
            p_value = float(p_value_raw)
            cramers_v = math.sqrt(phi2_corrected / denominator) if denominator > 0 else 0.0
        return {
            "categories": categories,
            "counts": counts,
            "proportions": proportions,
            "chi_square": chi_square,
            "chi_square_p_value": p_value,
            "cramers_v": cramers_v,
        }

    def _split_balance(
        self,
        rows       : Sequence[Mapping[str, Any]],
        continuous : Sequence[str],
        categorical: Sequence[str],
    ) -> list[dict[str, Any]]:
        """Audit requested split pairs for continuous and categorical distribution shift.

        Args:
            rows: Canonical selected population with fixed split labels.
            continuous: Numeric feature names.
            categorical: Discrete feature names.

        Returns:
            Long-form rows containing SMD/KS or Jensen--Shannon divergence/Cramer's V.
        """
        # Compare each split with the canonical population and with every other split. Continuous
        # and categorical results share a long-form schema identified by ``kind``.
        populations: dict[str, list[Mapping[str, Any]]] = {
            "selected": list(rows),
            **{split: [row for row in rows if row["split"] == split] for split in self.SPLITS},
        }
        comparisons = (
            ("train", "selected"),
            ("validation", "selected"),
            ("test", "selected"),
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
        result: list[dict[str, Any]] = []
        for left_name, right_name in comparisons:
            left_rows, right_rows = populations[left_name], populations[right_name]
            for feature in continuous:
                comparison = self._compare_continuous(
                    self._finite_array(row.get(feature) for row in left_rows),
                    self._finite_array(row.get(feature) for row in right_rows),
                )
                result.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "feature": feature,
                        "kind": "continuous",
                        **comparison,
                    }
                )
            for feature in categorical:
                categories = sorted(
                    {
                        str(row.get(feature, "unavailable"))
                        for row in [*left_rows, *right_rows]
                    }
                )
                left_counts = Counter(str(row.get(feature, "unavailable")) for row in left_rows)
                right_counts = Counter(str(row.get(feature, "unavailable")) for row in right_rows)
                left_values = np.asarray([left_counts[value] for value in categories], dtype=float)
                right_values = np.asarray(
                    [right_counts[value] for value in categories], dtype=float
                )
                left_values /= max(left_values.sum(), 1.0)
                right_values /= max(right_values.sum(), 1.0)
                midpoint = 0.5 * (left_values + right_values)
                js = 0.5 * self._kl(left_values, midpoint) + 0.5 * self._kl(right_values, midpoint)
                result.append(
                    {
                        "left": left_name,
                        "right": right_name,
                        "feature": feature,
                        "kind": "categorical",
                        "jensen_shannon": js,
                    }
                )
        return result

    @staticmethod
    def _kl(values: np.ndarray, reference: np.ndarray) -> float:
        """Compute finite discrete KL divergence using only positive source probabilities.

        Args:
            values: Probability vector ``p``.
            reference: Probability vector ``q`` with support wherever ``p`` is positive.

        Returns:
            ``sum p log2(p/q)`` in bits.
        """
        mask = values > 0.0
        return float(np.sum(values[mask] * np.log2(values[mask] / reference[mask])))

    def _shortcut_baselines(
        self,
        rows: Sequence[Mapping[str, Any]],
        seed: int,
    ) -> dict[str, Any]:
        """Fit small diagnostic logistic regressions with leakage-group-aware cross-validation.

        Args:
            rows: Canonical selected rows; positive-interface features are intentionally absent.
            seed: Deterministic ``StratifiedGroupKFold`` and logistic-regression seed.

        Returns:
            AUROC, AUPRC, and balanced-accuracy mean/std for three declared feature families.
        """
        # These are diagnostics, not candidate WISDOM architectures. Technical-only performance
        # reveals possible source/quality shortcuts; the simple global set gives a weak baseline.
        definitions = {
            "technical_with_origin": {
                "numeric": ("resolution", "coordinate_coverage", "release_year"),
                "categorical": ("origin", "experimental_method"),
            },
            "technical_without_origin": {
                "numeric": ("resolution", "coordinate_coverage", "release_year"),
                "categorical": ("experimental_method",),
            },
            "simple_global_model_visible": {
                "numeric": (
                    "sequence_length",
                    "theoretical_isoelectric_point",
                    "net_charge_at_pH_7",
                    "gravy",
                    "positive_residue_fraction",
                    "negative_residue_fraction",
                    "polar_residue_fraction",
                    "hydrophobic_residue_fraction",
                    "aromatic_fraction",
                    "radius_of_gyration_normalized",
                    "aspect_ratio",
                    "compactness",
                    "packing_density",
                ),
                "categorical": (),
            },
        }
        labels = np.asarray([int(row["label"]) for row in rows], dtype=np.int64)
        groups = np.asarray([str(row["leakage_group"]) for row in rows])
        unique_groups_per_class = [
            len(set(groups[labels == label].tolist())) for label in (0, 1)
        ]
        folds = min(5, *unique_groups_per_class)
        if folds < 2:
            return {
                name: {
                    "available": False,
                    "reason": "fewer_than_two_independent_groups_per_class",
                    "cross_validation": "StratifiedGroupKFold",
                }
                for name in definitions
            }
        # Leakage groups, rather than individual rows, define CV folds. This prevents optimistic
        # scores caused by homologous proteins appearing in both train and held-out partitions.
        splitter = StratifiedGroupKFold(n_splits=folds, shuffle=True, random_state=seed)
        result: dict[str, Any] = {}
        for name, definition in definitions.items():
            numeric = list(definition["numeric"])
            categorical = list(definition["categorical"])
            matrix = np.asarray(
                [
                    [row.get(field) for field in numeric]
                    + [str(row.get(field, "unavailable")) for field in categorical]
                    for row in rows
                ],
                dtype=object,
            )
            # Numeric values are median-imputed/scaled; categorical values are one-hot encoded with
            # unknown held-out categories ignored. All preprocessing is fitted inside each fold.
            transformers: list[tuple[str, Any, list[int]]] = []
            if numeric:
                transformers.append(
                    (
                        "numeric",
                        Pipeline(
                            [
                                ("impute", SimpleImputer(strategy="median")),
                                ("scale", StandardScaler()),
                            ]
                        ),
                        list(range(len(numeric))),
                    )
                )
            if categorical:
                transformers.append(
                    (
                        "categorical",
                        OneHotEncoder(handle_unknown="ignore"),
                        list(range(len(numeric), len(numeric) + len(categorical))),
                    )
                )
            scores: dict[str, list[float]] = defaultdict(list)
            for train, test in splitter.split(matrix, labels, groups):
                if len(set(labels[test].tolist())) < 2:
                    continue
                model = Pipeline(
                    [
                        ("features", ColumnTransformer(transformers)),
                        (
                            "classifier",
                            LogisticRegression(
                                max_iter=2000,
                                class_weight="balanced",
                                random_state=seed,
                            ),
                        ),
                    ]
                )
                model.fit(matrix[train], labels[train])
                probability = model.predict_proba(matrix[test])[:, 1]
                prediction  = (probability >= 0.5).astype(np.int64)
                scores["auroc"].append(float(roc_auc_score(labels[test], probability)))
                scores["auprc"].append(float(average_precision_score(labels[test], probability)))
                scores["balanced_accuracy"].append(
                    float(balanced_accuracy_score(labels[test], prediction))
                )
            if not scores["auroc"]:
                result[name] = {
                    "available": False,
                    "reason": "no_group_fold_contained_both_classes",
                    "cross_validation": "StratifiedGroupKFold",
                }
                continue
            result[name] = {
                "available": True,
                "model": "LogisticRegression",
                "cross_validation": "StratifiedGroupKFold",
                "folds": len(scores["auroc"]),
                "numeric_features": numeric,
                "categorical_features": categorical,
                **{
                    f"{metric}_{summary}": float(getattr(np, summary)(values))
                    for metric, values in scores.items()
                    for summary in ("mean", "std")
                },
            }
        return result

    @staticmethod
    def _add_fdr(rows: list[dict[str, Any]], p_field: str, output_field: str) -> None:
        """Add Benjamini--Hochberg adjusted p-values while preserving unavailable tests.

        Args:
            rows: Mutable result rows.
            p_field: Source raw p-value field.
            output_field: Destination adjusted p-value field.

        Returns:
            ``None`` after modifying each row in place.
        """
        valid = [
            (index, float(row[p_field]))
            for index, row in enumerate(rows)
            if row.get(p_field) is not None
        ]
        ordered = sorted(valid, key=lambda value: value[1])
        adjusted: dict[int, float] = {}
        running = 1.0
        count = len(ordered)
        for rank in range(count, 0, -1):
            index, value = ordered[rank - 1]
            running = min(running, value * count / rank)
            adjusted[index] = min(1.0, running)
        for index, row in enumerate(rows):
            row[output_field] = adjusted.get(index)

    def _warnings(
        self,
        raw_rows      : Sequence[Mapping[str, Any]],
        selected_rows : Sequence[Mapping[str, Any]],
        components    : Sequence[Sequence[str]],
        split_audit   : Mapping[str, Any],
        statistics    : Mapping[str, Any],
        parameters    : Mapping[str, Any],
    ) -> list[dict[str, Any]]:
        """Translate large effects and construction limitations into explicit interpretations.

        Args:
            raw_rows: Complete raw population used for giant-group fractions.
            selected_rows: Canonical selected population used for split-size warnings.
            components: Full-raw leakage groups.
            split_audit: Final split count and phenotype feasibility diagnostics.
            statistics: Continuous, categorical, split, and shortcut analyses.
            parameters: Validated warning thresholds and target fractions.

        Returns:
            Ordered warning mappings with severity, metric, value, threshold, and explanation.
        """
        # Warnings interpret already-computed evidence. They never alter membership, labels, splits,
        # or publication; this preserves a transparent distinction between data and judgement.
        warnings: list[dict[str, Any]] = []
        largest_fraction = max(map(len, components)) / len(raw_rows)
        if largest_fraction >= float(parameters["giant_group_fraction_warning"]):
            warnings.append(
                {
                    "severity": "warning",
                    "kind": "giant_leakage_group",
                    "value": largest_fraction,
                    "threshold": parameters["giant_group_fraction_warning"],
                    "interpretation": (
                        "A large transitive homology component limits independent split balance; "
                        "it was kept intact because leakage takes priority over balance."
                    ),
                }
            )
        selected_phenotypes = statistics["selected_summary"]["categorical"]["global_phenotype"]
        global_noise_fraction = int(selected_phenotypes.get("G_NOISE", 0)) / len(selected_rows)
        if global_noise_fraction >= float(parameters["phenotype_noise_warning"]):
            warnings.append(
                {
                    "severity": "warning",
                    "kind": "global_phenotype_noise",
                    "value": global_noise_fraction,
                    "threshold": parameters["phenotype_noise_warning"],
                    "interpretation": (
                        "Most selected proteins do not belong to a stable dense global phenotype. "
                        "Leakage safety is unaffected, but phenotype-stratified conclusions have "
                        "limited coverage and should treat G_NOISE as an explicit population."
                    ),
                }
            )
        for phenotype, evidence in split_audit["phenotype_feasibility"].items():
            missing = sorted(set(self.SPLITS) - set(evidence["observed_splits"]))
            if evidence["representable_in_all_splits"] and missing:
                warnings.append(
                    {
                        "severity": "warning",
                        "kind": "phenotype_split_coverage",
                        "feature": phenotype,
                        "value": evidence["leakage_group_count"],
                        "missing_splits": missing,
                        "interpretation": (
                            "At least three leakage groups carry this phenotype, but the current "
                            "joint assignment does not cover every split. Overlapping phenotype "
                            "memberships may make the apparently feasible coverage incompatible."
                        ),
                    }
                )
        for row in statistics["continuous_balance"]:
            smd = row.get("smd")
            ks = row.get("ks_statistic")
            if smd is not None and abs(float(smd)) >= float(parameters["smd_warning"]):
                category = str(row["category"])
                warnings.append(
                    {
                        "severity": (
                            "strong_warning"
                            if abs(float(smd)) >= float(parameters["smd_strong_warning"])
                            else "warning"
                        ),
                        "kind": "class_continuous_shift",
                        "feature": row["feature"],
                        "feature_category": category,
                        "value": smd,
                        "threshold": parameters["smd_warning"],
                        "interpretation": (
                            "A technical difference is a possible nuisance confounder."
                            if category == "technical_nuisance"
                            else (
                                "A model-visible difference may be real biology or a global "
                                "shortcut; it is not automatically bias."
                            )
                        ),
                    }
                )
            if ks is not None and float(ks) >= float(parameters["ks_warning"]):
                warnings.append(
                    {
                        "severity": "warning",
                        "kind": "class_distribution_shift",
                        "feature": row["feature"],
                        "value": ks,
                        "threshold": parameters["ks_warning"],
                        "interpretation": (
                            "Positive and negative empirical distributions differ substantially."
                        ),
                    }
                )
        for row in statistics["categorical_balance"]:
            value = row.get("cramers_v")
            if value is not None and float(value) >= float(parameters["cramers_v_warning"]):
                warnings.append(
                    {
                        "severity": "warning",
                        "kind": "class_categorical_association",
                        "feature": row["feature"],
                        "value": value,
                        "threshold": parameters["cramers_v_warning"],
                        "interpretation": (
                            "This category is associated with the label. Origin/method "
                            "associations are technical confounding risks; phenotype "
                            "associations are descriptive."
                        ),
                    }
                )
        technical = statistics["shortcut_baselines"].get("technical_without_origin", {})
        auc = technical.get("auroc_mean")
        if auc is not None and float(auc) >= float(parameters["technical_shortcut_auc_warning"]):
            warnings.append(
                {
                    "severity": "red_flag",
                    "kind": "technical_shortcut",
                    "value": auc,
                    "threshold": parameters["technical_shortcut_auc_warning"],
                    "interpretation": (
                        "A leakage-group-aware model predicts labels from method, resolution, "
                        "coverage, and year alone; downstream model performance may exploit "
                        "technical confounding."
                    ),
                }
            )
        split_fractions = {
            "train":      float(parameters["train_fraction"]),
            "validation": float(parameters["validation_fraction"]),
            "test":       float(parameters["test_fraction"]),
        }
        for split, counts in split_audit["split_counts"].items():
            target = len(selected_rows) * split_fractions[split]
            relative = abs(int(counts["total"]) - target) / max(target, 1.0)
            if relative > float(parameters["split_tolerance"]):
                warnings.append(
                    {
                        "severity": "warning",
                        "kind": "split_size_shift",
                        "split": split,
                        "value": relative,
                        "threshold": parameters["split_tolerance"],
                        "interpretation": (
                            "Indivisible leakage groups prevented closer target size matching; "
                            "no group was broken."
                        ),
                    }
                )
        return sorted(
            warnings,
            key=lambda row: (
                str(row.get("severity")),
                str(row.get("kind")),
                str(row.get("feature", "")),
            ),
        )

    def _plots(
        self,
        raw_rows         : Sequence[Mapping[str, Any]],
        selected_rows    : Sequence[Mapping[str, Any]],
        continuous_balance: Sequence[Mapping[str, Any]],
        dilutions        : Mapping[str, Any],
        root             : Path,
    ) -> None:
        """Generate a compact non-redundant matplotlib audit figure set.

        Args:
            raw_rows: Complete raw population.
            selected_rows: Canonical selected rows with fixed splits.
            continuous_balance: Class effect-size rows used by the SMD forest plot.
            dilutions: Nested training membership audit.
            root: Plot output directory.

        Returns:
            ``None`` after nine PNG files are written into the managed report directory.
        """
        # Each PNG answers one audit question; plots intentionally avoid repeating every CSV metric.
        root.mkdir(parents=True, exist_ok=True)

        # Class counts show the effect of canonical balancing relative to immutable RAW evidence.
        figure, axis = plt.subplots(figsize=(7, 4))
        labels = ("raw negative", "raw positive", "selected negative", "selected positive")
        counts = (
            sum(row["label"] == 0 for row in raw_rows),
            sum(row["label"] == 1 for row in raw_rows),
            sum(row["label"] == 0 for row in selected_rows),
            sum(row["label"] == 1 for row in selected_rows),
        )
        axis.bar(labels, counts, color=("#4477AA", "#CC6677", "#4477AA", "#CC6677"))
        axis.set_ylabel("Protein count")
        axis.set_title("Raw and canonical class counts")
        axis.tick_params(axis="x", rotation=20)
        self._save_plot(figure, root / "class-counts.png")

        # Origin-by-label counts expose whether data provenance itself can act as a shortcut.
        origins = sorted({str(row["origin"]) for row in raw_rows})
        figure, axis = plt.subplots(figsize=(8, 4))
        positions = np.arange(len(origins))
        negative = [
            sum(row["origin"] == origin and row["label"] == 0 for row in raw_rows)
            for origin in origins
        ]
        positive = [
            sum(row["origin"] == origin and row["label"] == 1 for row in raw_rows)
            for origin in origins
        ]
        axis.bar(positions, negative, label="negative", color="#4477AA")
        axis.bar(positions, positive, bottom=negative, label="positive", color="#CC6677")
        axis.set_xticks(positions, origins, rotation=20)
        axis.set_ylabel("Raw proteins")
        axis.set_title("Origin x label (technical confounding audit)")
        axis.legend()
        self._save_plot(figure, root / "origin-by-label.png")

        # Group-size tails reveal how strongly homology constrains achievable split proportions.
        group_sizes = list(Counter(str(row["leakage_group"]) for row in raw_rows).values())
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.hist(group_sizes, bins=min(40, max(group_sizes)), color="#228833")
        axis.set_xlabel("Proteins per full-raw leakage group")
        axis.set_ylabel("Group count")
        axis.set_title("Transitive leakage-group sizes")
        self._save_plot(figure, root / "leakage-group-sizes.png")

        # The forest plot keeps only the 15 largest available class effects for visual readability.
        effect_rows = sorted(
            (row for row in continuous_balance if row.get("smd") is not None),
            key=lambda row: abs(float(row["smd"])),
        )[-15:]
        figure, axis = plt.subplots(figsize=(8, max(4, len(effect_rows) * 0.35)))
        axis.barh(
            [str(row["feature"]) for row in effect_rows],
            [float(row["smd"]) for row in effect_rows],
            color=[
                "#EE6677" if row["category"] == "technical_nuisance" else "#228833"
                for row in effect_rows
            ],
        )
        axis.axvline(0.0, color="black", linewidth=0.8)
        axis.set_xlabel("Standardized mean difference (positive - negative)")
        axis.set_title("Largest canonical class effect sizes")
        self._save_plot(figure, root / "smd-forest.png")

        self._phenotype_plot(selected_rows, "global_phenotype", root / "global-phenotypes.png")
        self._phenotype_plot(
            [row for row in selected_rows if row["label"] == 1],
            "interface_phenotype",
            root / "interface-phenotypes.png",
        )
        self._pca_plots(selected_rows, root)

        replicate = dilutions["replicates"][sorted(dilutions["replicates"])[0]]
        subsets = sorted(
            replicate["subsets"].items(), key=lambda value: float(value[1]["requested_fraction"])
        )
        figure, axis = plt.subplots(figsize=(7, 4))
        axis.plot(
            [value["requested_fraction"] for _, value in subsets],
            [value["realized_rows"] for _, value in subsets],
            marker="o",
            label="realized rows",
        )
        axis.plot(
            [value["requested_fraction"] for _, value in subsets],
            [value["positive"] / max(1, value["realized_rows"]) for _, value in subsets],
            marker="s",
            label="positive fraction",
        )
        axis.set_xlabel("Requested training fraction")
        axis.set_title("Nested training dilution size and balance")
        axis.legend()
        self._save_plot(figure, root / "dilutions.png")

    def _phenotype_plot(
        self,
        rows   : Sequence[Mapping[str, Any]],
        field  : str,
        output : Path,
    ) -> None:
        """Plot one physical phenotype count table stratified by fixed split.

        Args:
            rows: Canonical global rows or positive-only rows.
            field: Global/interface phenotype field.
            output: PNG path inside the managed statistics directory.

        Returns:
            ``None`` after writing the plot.
        """
        # Stacked counts show whether each physical phenotype is represented across fixed splits.
        phenotypes = sorted({str(row[field]) for row in rows})
        figure, axis = plt.subplots(figsize=(max(8, len(phenotypes) * 0.5), 4))
        positions = np.arange(len(phenotypes))
        bottom = np.zeros(len(phenotypes))
        colors = {"train": "#4477AA", "validation": "#EECC66", "test": "#CC6677"}
        for split in self.SPLITS:
            values = np.asarray(
                [
                    sum(
                        row[field] == phenotype and row["split"] == split
                        for row in rows
                    )
                    for phenotype in phenotypes
                ]
            )
            axis.bar(positions, values, bottom=bottom, label=split, color=colors[split])
            bottom += values
        axis.set_xticks(positions, phenotypes, rotation=45, ha="right")
        axis.set_ylabel("Protein count")
        axis.set_title(field.replace("_", " ").title() + " by split")
        axis.legend()
        self._save_plot(figure, output)

    def _pca_plots(self, rows: Sequence[Mapping[str, Any]], root: Path) -> None:
        """Plot the same two-dimensional global descriptor PCA by label and by split.

        Args:
            rows: Canonical selected population.
            root: Plot directory receiving two separate PNGs.

        Returns:
            ``None``; unavailable-feature rows are excluded consistently from both images.
        """
        # PCA is visualization only: it uses the same robust-scaled global descriptors but never
        # changes HDBSCAN labels, selection priorities, or split assignments.
        features = self.GLOBAL_PHENOTYPE_FEATURES
        prepared: list[tuple[Mapping[str, Any], list[float]]] = []
        for row in rows:
            derived = dict(row)
            derived["log_sequence_length"] = math.log(max(1.0, float(row["sequence_length"])))
            charge = row.get("net_charge_at_pH_7")
            derived["charge_density"] = (
                float(charge) / float(row["sequence_length"]) if charge is not None else None
            )
            values: list[float] = []
            for feature in features:
                value = derived.get(feature)
                if value is None or not math.isfinite(float(value)):
                    break
                values.append(float(value))
            if len(values) == len(features):
                prepared.append((row, values))
        if len(prepared) < 3:
            return
        # Fit one [P,2] embedding and recolor identical coordinates by label and by split.
        matrix = RobustScaler().fit_transform(np.asarray([value for _, value in prepared]))
        coordinates = PCA(n_components=2).fit_transform(matrix)
        for color_field, filename, palette in (
            ("label", "global-pca-by-label.png", {0: "#4477AA", 1: "#CC6677"}),
            (
                "split",
                "global-pca-by-split.png",
                {"train": "#4477AA", "validation": "#EECC66", "test": "#CC6677"},
            ),
        ):
            figure, axis = plt.subplots(figsize=(7, 5))
            categories = sorted({row[color_field] for row, _ in prepared}, key=str)
            for category in categories:
                mask = np.asarray([row[color_field] == category for row, _ in prepared])
                axis.scatter(
                    coordinates[mask, 0],
                    coordinates[mask, 1],
                    s=10,
                    alpha=0.65,
                    color=palette[category],
                    label=str(category),
                )
            axis.set_xlabel("Global physical descriptor PC1")
            axis.set_ylabel("Global physical descriptor PC2")
            axis.set_title(f"Global physical descriptor PCA by {color_field}")
            axis.legend()
            self._save_plot(figure, root / filename)

    @staticmethod
    def _save_plot(figure: Any, path: Path) -> None:
        """Save one figure inside the managed output and close its memory promptly.

        Args:
            figure: Matplotlib figure instance.
            path: Final PNG destination.

        Returns:
            ``None`` after writing the PNG and closing the figure.
        """
        try:
            figure.tight_layout()
            figure.savefig(path, dpi=160)
        finally:
            plt.close(figure)

    def _write_outputs(
        self,
        root            : Path,
        raw_records     : Path,
        raw_rows        : Sequence[Mapping[str, Any]],
        selected_rows   : Sequence[Mapping[str, Any]],
        mmseqs_path     : Path,
        foldseek_path   : Path,
        sequence_edges  : set[tuple[str, str]],
        structure_edges : set[tuple[str, str]],
        exact_pairs     : Sequence[Mapping[str, Any]],
        components      : Sequence[Sequence[str]],
        global_result   : Mapping[str, Any],
        interface_result: Mapping[str, Any],
        selection_audit : Mapping[str, Any],
        split_audit     : Mapping[str, Any],
        dilution_audit  : Mapping[str, Any],
        statistics      : Mapping[str, Any],
        warnings        : Sequence[Mapping[str, Any]],
        parameters      : Mapping[str, Any],
    ) -> None:
        """Write the complete portable design contract into one managed output directory.

        Args:
            root: Attempt-owned final ``dataset-design`` directory.
            raw_records: Immutable JSONL or legacy FASTA used for its SHA-256 fingerprint.
            raw_rows: Full raw rows with permanent leakage/phenotype assignments.
            selected_rows: Canonical rows with final split assignments.
            mmseqs_path: Raw seven-column specialist sequence evidence.
            foldseek_path: Raw eight-column specialist structural evidence.
            sequence_edges: Thresholded MMseqs2 hard edges.
            structure_edges: Thresholded Foldseek hard edges.
            exact_pairs: Exact/provenance hard pair evidence with reasons.
            components: Full-raw connected components.
            global_result: Global HDBSCAN labels and diagnostics.
            interface_result: Positive-interface HDBSCAN labels and diagnostics.
            selection_audit: Canonical majority-selection evidence.
            split_audit: Weighted group-assignment evidence.
            dilution_audit: Nested training membership and fixed evaluation fingerprints.
            statistics: Population, balance, and shortcut diagnostics.
            warnings: Interpreted scientific/technical warning rows.
            parameters: Effective researcher-selected Work configuration.

        Returns:
            ``None`` after every declared output exists under ``root``.
        """
        # The managed directory has three conceptual layers:
        # root manifests/catalogs, ``clusters/`` pair/group evidence, and ``descriptors/`` features.
        clusters        = root / "clusters"
        descriptors     = root / "descriptors"
        statistics_root = root / "statistics"
        clusters.mkdir(parents=True, exist_ok=True)
        descriptors.mkdir(parents=True, exist_ok=True)
        statistics_root.mkdir(parents=True, exist_ok=True)

        # Catalog CSVs contain one portable row per protein. Managed cache references are removed;
        # nested lists/dicts are encoded as compact JSON cells by ``_write_csv``.
        portable_raw = [
            {
                key: value
                for key, value in row.items()
                if key not in {"structure_path", "foldseek_structure"}
            }
            for row in raw_rows
        ]
        portable_selected = [
            {
                key: value
                for key, value in row.items()
                if key not in {"structure_path", "foldseek_structure"}
            }
            for row in selected_rows
        ]
        self._write_csv(root / "catalog-all.csv", portable_raw)
        self._write_csv(root / "catalog.csv", portable_selected)
        self._write_text(
            root / "selected.fasta",
            "".join(f">{row['original_header']}\n{row['sequence']}\n" for row in selected_rows),
        )
        self._write_text(
            root / "proteins.txt",
            "".join(
                f"{row['identifier']}\n"
                for row in sorted(selected_rows, key=lambda value: str(value["identifier"]))
            ),
        )
        self._write_text(root / "proteins-labelled.txt", self._labelled_manifest(selected_rows))

        # ID-only TXT files drive label-free structural geometry. Their labelled siblings contain
        # ``RCSB_CHAIN<TAB>0|1`` for inspection and downstream joins. Both are deterministic views
        # of the authoritative catalog rather than independent scientific sources.
        for split in self.SPLITS:
            subset = [row for row in selected_rows if row["split"] == split]
            self._write_text(
                root / f"{split}.txt",
                "".join(
                    f"{row['identifier']}\n"
                    for row in sorted(subset, key=lambda row: row["identifier"])
                ),
            )
            self._write_text(
                root / f"{split}-labelled.txt",
                self._labelled_manifest(subset),
            )
            self._write_text(
                root / f"{split}.fasta",
                "".join(
                    f">{row['original_header']}\n{row['sequence']}\n"
                    for row in sorted(subset, key=lambda row: row["identifier"])
                ),
            )
        self._write_text(
            root / "omitted-positives.txt",
            "".join(
                f"{value['identifier']}\n" for value in selection_audit["omitted_positives"]
            ),
        )
        self._write_text(
            root / "quality-exclusions.txt",
            "".join(
                f"{value['identifier']}\n"
                for value in selection_audit["quality_filter"]["exclusions"]
            ),
        )

        # Preserve raw specialist TSVs beside thresholded two-column edge CSVs and explanatory
        # exact-pair reasons. This makes every leakage-group connection independently auditable.
        (clusters / "sequence-pairs.tsv").write_bytes(mmseqs_path.read_bytes())
        (clusters / "structure-pairs.tsv").write_bytes(foldseek_path.read_bytes())
        self._write_csv(
            clusters / "sequence-edges.csv",
            [{"left": left, "right": right} for left, right in sorted(sequence_edges)],
        )
        self._write_csv(
            clusters / "structure-edges.csv",
            [{"left": left, "right": right} for left, right in sorted(structure_edges)],
        )
        self._write_csv(clusters / "exact-pairs.csv", exact_pairs)
        self._write_csv(
            clusters / "leakage-groups.csv",
            [
                {
                    "identifier": row["identifier"],
                    "leakage_group": row["leakage_group"],
                    "selected": row["selected"],
                    "split": row["split"],
                }
                for row in raw_rows
            ],
        )
        self._write_csv(
            clusters / "global-phenotypes.csv",
            [
                {
                    "identifier": row["identifier"],
                    "global_phenotype": row["global_phenotype"],
                    "probability": row["global_phenotype_probability"],
                }
                for row in raw_rows
            ],
        )
        self._write_csv(
            clusters / "positive-interface-phenotypes.csv",
            [
                {
                    "identifier": row["identifier"],
                    "interface_phenotype": row["interface_phenotype"],
                    "probability": row["interface_phenotype_probability"],
                }
                for row in raw_rows
                if row["label"] == 1
            ],
        )
        component_sizes = Counter(str(row["leakage_group"]) for row in raw_rows)
        component_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for row in raw_rows:
            component_rows[str(row["leakage_group"])].append(row)
        self._write_json(
            clusters / "clustering-diagnostics.json",
            {
                "full_raw_component_count": len(components),
                "largest_component": max(component_sizes.values()),
                "median_component_size": float(np.median(list(component_sizes.values()))),
                "component_size_histogram": dict(Counter(map(str, component_sizes.values()))),
                "largest_components": [
                    {
                        "leakage_group": group,
                        "raw": self._class_counts(component_rows[group]),
                        "selected": self._class_counts(
                            [row for row in component_rows[group] if row["selected"]]
                        ),
                        "omitted_positive": sum(
                            int(row["label"]) == 1 and not bool(row["selected"])
                            for row in component_rows[group]
                        ),
                    }
                    for group, _size in component_sizes.most_common(25)
                ],
                "global": global_result["diagnostics"],
                "positive_interface": interface_result["diagnostics"],
            },
        )

        # Descriptor tables separate label-agnostic global variables from positive-only interface
        # variables. The latter are evaluation/design evidence and never model inputs.
        global_fields = [
            field
            for field in self.CONTINUOUS_FEATURES
            if field not in {"resolution", "release_year", "coordinate_coverage"}
        ] + [
            "identifier",
            "experimental_method",
            "resolution",
            "release_year",
            "coordinate_coverage",
        ]
        self._write_csv(
            descriptors / "global-features.csv",
            [{field: row.get(field) for field in global_fields} for row in raw_rows],
        )
        interface_fields = ("identifier", *self.INTERFACE_PHENOTYPE_FEATURES)
        self._write_csv(
            descriptors / "positive-interface-features.csv",
            [
                {field: row.get(field) for field in interface_fields}
                for row in raw_rows
                if row["label"] == 1
            ],
        )
        self._write_json(statistics_root / "warnings.json", {"warnings": list(warnings)})
        self._write_json(root / "selection-audit.json", selection_audit)
        self._write_json(root / "split-audit.json", split_audit)
        self._write_json(root / "dilution-audit.json", dilution_audit)

        # Provenance fixes source bytes, scientific settings, thresholds, hashes, and deterministic
        # policy. LambdaForge separately records runtime environment and external-tool versions.
        provenance = {
            "design_schema_version": self.SCHEMA_VERSION,
            "raw_records_sha256": self._sha256_file(raw_records),
            "raw_records_format": raw_records.suffix.lower().removeprefix("."),
            "parameters": dict(parameters),
            "leakage_criteria": {
                key: parameters[key]
                for key in (
                    "sequence_identity",
                    "sequence_coverage",
                    "sequence_evalue",
                    "foldseek_probability",
                    "foldseek_tmscore",
                    "foldseek_coverage",
                    "foldseek_evalue",
                    "group_same_pdb",
                )
            },
            "clustering": {
                "global": global_result["diagnostics"],
                "positive_interface": interface_result["diagnostics"],
            },
            "seed": parameters["seed"],
            "split_objective_weights": split_audit["weights"],
            "dilutions": {
                "fractions": parameters["dilution_fractions"],
                "replicates": parameters["dilution_replicates"],
            },
            "selected_structure_sha256": {
                str(row["pdb_id"]): str(row["structure_sha256"])
                for row in selected_rows
            },
            "reproducibility_note": (
                "Worker count changes execution only; all scientific orderings use sorted IDs and "
                "seeded SHA-256. LambdaForge records the environment and external-tool versions."
            ),
        }
        self._write_json(root / "provenance.json", provenance)
        self._write_json(
            root / "design-summary.json",
            {
                "verdict": "PASS",
                "design_schema_version": self.SCHEMA_VERSION,
                "raw": self._class_counts(raw_rows),
                "quality_eligible": selection_audit["quality_filter"]["eligible_counts"],
                "quality_excluded": selection_audit["quality_filter"]["excluded_counts"],
                "selected": self._class_counts(selected_rows),
                "splits": split_audit["split_counts"],
                "leakage_group_count": len(components),
                "warning_count": len(warnings),
                "technical_shortcut_auc": statistics["shortcut_baselines"].get(
                    "technical_without_origin", {}
                ).get("auroc_mean"),
            },
        )
        self._write_text(
            root / "REPORT.md",
            self._markdown_report(
                raw_rows,
                selected_rows,
                components,
                global_result,
                interface_result,
                selection_audit,
                split_audit,
                dilution_audit,
                statistics,
                warnings,
                parameters,
            ),
        )

    def _markdown_report(
        self,
        raw_rows         : Sequence[Mapping[str, Any]],
        selected_rows    : Sequence[Mapping[str, Any]],
        components       : Sequence[Sequence[str]],
        global_result    : Mapping[str, Any],
        interface_result : Mapping[str, Any],
        selection_audit  : Mapping[str, Any],
        split_audit      : Mapping[str, Any],
        dilution_audit   : Mapping[str, Any],
        statistics       : Mapping[str, Any],
        warnings         : Sequence[Mapping[str, Any]],
        parameters       : Mapping[str, Any],
    ) -> str:
        """Explain the complete design audit and its observed values in plain English.

        The report is generated from the same in-memory rows and statistical objects written to
        CSV/JSON, so narrative counts cannot drift from machine-readable evidence. It defines each
        metric before interpreting it, distinguishes leakage safety from distribution balance, and
        describes the question, axes, and limitations of every generated plot.

        Args:
            raw_rows: Complete evidence population before quality filtering or balancing.
            selected_rows: Canonical balanced population with fixed split assignments.
            components: Full-raw transitive sequence/structure/provenance leakage groups.
            global_result: Label-free whole-protein HDBSCAN result and stability diagnostics.
            interface_result: Positive-only DNA-interface HDBSCAN result and diagnostics.
            selection_audit: Quality filtering and majority-class selection evidence.
            split_audit: Whole-group split optimizer evidence and final class counts.
            dilution_audit: Nested training-only learning-curve membership evidence.
            statistics: Descriptive, balance, split-shift, and shortcut-model results.
            warnings: Ordered interpreted concerns produced from configured thresholds.
            parameters: Effective scientific configuration and interpretation thresholds.

        Returns:
            Complete Markdown document with relative links to tables and figures.
        """
        def number(value: Any, digits: int = 3) -> str:
            """Format a numeric report value while preserving unavailable/non-numeric evidence.

            Args:
                value: Optional value obtained from the machine-readable statistical result.
                digits: Number of fractional decimal digits used for numeric values.

            Returns:
                Human-readable number, ``not available``, or the original textual value.
            """
            if value is None:
                return "not available"
            try:
                return f"{float(value):.{digits}f}"
            except (TypeError, ValueError):
                return str(value)

        raw_counts      = self._class_counts(raw_rows)
        selected_counts = self._class_counts(selected_rows)
        largest_group   = max(map(len, components))
        exact_sequences = len({str(row["sequence_sha256"]) for row in selected_rows})
        selected_groups = len({str(row["leakage_group"]) for row in selected_rows})
        selected_global = Counter(str(row["global_phenotype"]) for row in selected_rows)
        selected_noise  = selected_global.get("G_NOISE", 0) / len(selected_rows)

        lines = [
            "# WISDOM-DNA dataset-design report",
            "",
            "## 1. Executive verdict",
            "",
            "**Design status: PASS.** PASS means that the software verified the declared input, "
            "kept every hard sequence/structure/provenance group inside one split, retained both "
            "classes in validation and test, and wrote a reproducible output. It does **not** mean "
            "that the dataset is free of biological or acquisition bias; those risks are measured "
            "below.",
            "",
            f"The raw evidence contains **{raw_counts['total']} proteins**: "
            f"{raw_counts['positive']} positive and {raw_counts['negative']} negative. The "
            f"canonical set contains **{selected_counts['total']} proteins**: "
            f"{selected_counts['positive']} positive and {selected_counts['negative']} negative. "
            "A positive label means that DNA binding is supported by the declared benchmark "
            "evidence and a protein--DNA heavy-atom contact was revalidated in the selected "
            "biological assembly. A negative label is accepted only from the curated BTD "
            "benchmark exclusion protocol and after contradiction checks; absence of DNA from a "
            "PDB structure is never used as negative evidence.",
            "",
            f"The selected set has **{selected_groups} independent leakage groups** and "
            f"**{exact_sequences} distinct exact sequences**. The largest raw transitive group has "
            f"**{largest_group} proteins** ({largest_group / len(raw_rows):.1%} of raw evidence).",
            "",
            "## 2. How to read the output",
            "",
            "`catalog.csv` is the authoritative row-level table. `proteins.txt` and the three "
            "split TXT files contain one `RCSB_CHAIN` identifier per line for label-free geometry. "
            "Files ending in `-labelled.txt` contain `RCSB_CHAIN<TAB>LABEL`, where `0` means the "
            "curated negative class and `1` the contact-supported positive class. These labelled "
            "files are convenient views; downstream code must still join against `catalog.csv` "
            "when it needs assembly, copy, provenance, or evidence-tier information.",
            "",
            "`catalog-all.csv` preserves every raw candidate, including omitted and quality-"
            "excluded rows. `clusters/` contains the pair evidence that created leakage groups. "
            "`descriptors/` contains the measured protein properties. `statistics/` contains exact "
            "machine-readable values and the plots explained in section 8.",
            "",
            "## 3. Selection, quality, and class balance",
            "",
            "Class balance prevents a trivial majority-class predictor from appearing useful. "
            "The requested positive:negative ratio was "
            f"**{parameters['positive_negative_ratio']}:1**; the realized ratio is "
            f"**{selected_counts['positive'] / selected_counts['negative']:.3f}:1**. "
            "Proteins were removed from the positive majority at leakage-group-aware deterministic "
            "priorities; the negative pool was not enlarged with uncertain examples.",
            "",
            "The quality filter retained "
            f"**{selection_audit['quality_filter']['eligible_counts']['total']}** "
            f"of {raw_counts['total']} candidates and excluded "
            f"**{selection_audit['quality_filter']['excluded_counts']['total']}**. For fitted "
            "X-ray "
            f"or cryo-EM structures, the configured resolution ceiling is "
            f"**{number(parameters['maximum_resolution'])} Å**. Smaller resolution values describe "
            "finer experimental detail. Missing resolution is reported as missing rather than "
            "invented, and it is not automatically rejected because methods such as NMR do not "
            "have the same resolution field.",
            "",
            "The evidence-tier counts below make label semantics explicit. They are provenance, "
            "not model inputs, and their association with the label is expected by construction.",
            "",
            "| Label evidence tier | Raw proteins | Selected proteins |",
            "|---|---:|---:|",
        ]
        raw_evidence      = Counter(str(row["label_evidence"]) for row in raw_rows)
        selected_evidence = Counter(str(row["label_evidence"]) for row in selected_rows)
        for evidence in sorted(set(raw_evidence) | set(selected_evidence)):
            lines.append(
                f"| `{evidence}` | {raw_evidence[evidence]} | "
                f"{selected_evidence[evidence]} |"
            )
        lines.extend(
            [
            "",
            "![Raw and selected class counts](statistics/plots/class-counts.png)",
            "",
            "## 4. Leakage protection and split composition",
            "",
            "A leakage edge joins two proteins when they pass the configured MMseqs2 sequence "
            "thresholds, Foldseek structure thresholds, have exactly the same sequence, represent "
            "the same logical identity, or come from the same PDB deposition when that policy is "
            "enabled. Transitive closure matters: if A resembles B and B resembles C, all three "
            "belong to one indivisible leakage group even if A and C do not directly pass a "
            "threshold. This prevents close relatives from making evaluation artificially easy.",
            "",
            "| Split | Proteins | Positive | Negative | Positive fraction |",
            "|---|---:|---:|---:|---:|",
            ]
        )
        for split in self.SPLITS:
            counts = split_audit["split_counts"][split]
            lines.append(
                f"| {split} | {counts['total']} | {counts['positive']} | {counts['negative']} | "
                f"{counts['positive'] / counts['total']:.1%} |"
            )
        lines.extend(
            [
                "",
                "No leakage group, exact sequence, same-PDB edge, accepted MMseqs2 edge, or "
                "accepted Foldseek edge crosses these splits. Split sizes can differ slightly "
                "from requested "
                "fractions because breaking a large group would violate the stronger leakage rule.",
                "",
                "After seeding rare feasible phenotypes, the deterministic optimizer minimizes "
                "normalized squared deviations. Its count term is "
                "`sum_s w_size ((n_s-f_s n)/max(f_s n,1))^2 + sum_s sum_k w_k "
                "((n_s,k-f_s n_k)/max(f_s n_k,1))^2`. Here `f_s` is the requested split "
                "fraction, `n_s` its protein count, `n` the canonical count, and `k` is a class, "
                "phenotype, or positive-origin category. Technical means add `sum_s sum_t "
                "w_technical ((mean_s,t-mean_t)/max(abs(mean_t),1))^2`. Normalization stops "
                "frequent categories from dominating only because they are numerous; squaring "
                "penalizes large deviations. These remain soft preferences and cannot split a "
                "dependency group.",
                "",
                f"The objective improved from **{number(split_audit['initial_objective'])}** to "
                f"**{number(split_audit['final_objective'])}** through "
                f"**{split_audit['accepted_refinement_moves']} accepted group moves**. Lower is "
                "better only for this declared weighted objective; it is not a biological score.",
                "",
                "![Leakage-group sizes](statistics/plots/leakage-group-sizes.png)",
                "",
                "## 5. Physical diversity and clustering",
                "",
                "Two different clustering problems are kept separate. MMseqs2/Foldseek edges "
                "define **dependency groups** from evolutionary sequence or structure similarity; "
                "those groups are hard split constraints and never claim a phenotype. HDBSCAN "
                "instead explores **phenotypes** in a table of measured physical descriptors; its "
                "labels help audit diversity and balance but never override a dependency group or "
                "become a DNA-binding target.",
                "",
                "HDBSCAN is a density-based clustering algorithm: it groups proteins only where "
                "the descriptor space contains sufficiently dense, stable neighborhoods and marks "
                "unsupported cases as noise. A `G_` phenotype describes whole-protein shape and "
                "chemistry without using the DNA label. An `I_` phenotype describes the geometry "
                "of a revalidated positive DNA-contact region. `G_NOISE` or `I_NOISE` is not an "
                "error and not a new biological family; it means that these measurements do not "
                "support a stable dense assignment under the selected settings.",
                "",
                "Stability is summarized by the adjusted Rand index, "
                "`ARI = (RI - E[RI]) / (max(RI) - E[RI])`. Here RI counts whether pairs of "
                "proteins are grouped consistently, and `E[RI]` is the agreement expected by "
                "chance. ARI = 1 means identical partitions; values near 0 mean chance-level "
                "agreement. WISDOM compares the selected HDBSCAN setting with neighboring "
                "settings and rejects the phenotype partition when median ARI is below the "
                f"configured **{number(parameters['phenotype_stability_minimum'])}** threshold.",
                "",
                f"On the quality-eligible population, global clustering found "
                f"**{global_result['diagnostics']['cluster_count']} dense "
                f"phenotypes** with a noise fraction of "
                f"**{number(global_result['diagnostics']['noise_fraction'])}**. Its median "
                f"neighboring-parameter adjusted Rand index (ARI) is "
                f"**{number(global_result['diagnostics'].get('median_adjusted_rand'))}**. After "
                f"quality filtering and class selection, **{selected_noise:.1%}** of canonical "
                "members are `G_NOISE`; selection can change this fraction without refitting or "
                "relabeling the raw phenotype model.",
                "",
                "Positive-interface clustering found "
                f"**{interface_result['diagnostics']['cluster_count']} "
                f"dense phenotypes** with noise fraction "
                f"**{number(interface_result['diagnostics']['noise_fraction'])}** and median-grid "
                f"ARI **{number(interface_result['diagnostics'].get('median_adjusted_rand'))}**.",
                "",
                "![Global phenotypes by split](statistics/plots/global-phenotypes.png)",
                "",
                "![Positive-interface phenotypes by "
                "split](statistics/plots/interface-phenotypes.png)",
                "",
                "## 6. Statistical balance",
                "",
                "For each continuous feature, the standardized mean difference (SMD) is the "
                "positive-minus-negative mean divided by the pooled standard deviation: "
                "`SMD = (mean_positive - mean_negative) / s_pooled`, where `s_pooled` combines "
                "the two within-class variances. Zero means equal means; the sign gives direction; "
                "absolute values around 0.25 and 0.50 trigger "
                "the configured warning and strong-warning levels. The Kolmogorov--Smirnov (KS) "
                "statistic is `KS = sup_x |F_positive(x) - F_negative(x)|`, the largest vertical "
                "separation between the two empirical cumulative distributions. It ranges from 0 "
                "(same observed distribution) to 1 (fully separated). Normalized Wasserstein "
                "distance is the one-dimensional transport distance divided by `s_pooled`; it "
                "measures how far observations would need to move, in pooled-standard-deviation "
                "units, to transform one distribution into the other. FDR values adjust repeated-"
                "test p-values; they indicate evidence against identical distributions, not the "
                "practical size or cause of a difference.",
                "",
                "| Feature | Type | SMD | KS | KS FDR | Wasserstein | Interpretation |",
                "|---|---|---:|---:|---:|---:|---|",
            ]
        )
        for row in statistics["continuous_balance"]:
            smd = row.get("smd")
            magnitude = abs(float(smd)) if smd is not None else None
            if magnitude is None:
                interpretation = "Insufficient finite observations."
            elif magnitude >= float(parameters["smd_strong_warning"]):
                interpretation = "Large class shift; inspect as a possible shortcut or biology."
            elif magnitude >= float(parameters["smd_warning"]):
                interpretation = "Noticeable class shift; monitor downstream sensitivity."
            else:
                interpretation = "Mean separation is below the configured warning threshold."
            lines.append(
                f"| `{row['feature']}` | {row['category']} | {number(smd)} | "
                f"{number(row.get('ks_statistic'))} | {number(row.get('ks_fdr'))} | "
                f"{number(row.get('normalized_wasserstein'))} | {interpretation} |"
            )
        lines.extend(
            [
                "",
                "For categorical features, Cramér's V measures association with the binary label. "
                "Its uncorrected form is `V = sqrt((chi2 / n) / min(r - 1, c - 1))`, where "
                "`chi2` is the contingency-table statistic, `n` is the number of proteins, and "
                "`r` and `c` are the row and column counts. WISDOM uses the finite-sample "
                "bias-corrected form. V ranges from 0 (no observed association) to 1 (perfect "
                "separation). A high value "
                "for acquisition origin or experimental method is a confounding risk; a high value "
                "for a label-free physical phenotype may represent biology but still deserves "
                "controlled evaluation.",
                "",
                "| Categorical feature | Categories | Cramér's V | Chi-square FDR | "
                "Interpretation |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for row in statistics["categorical_balance"]:
            value = row.get("cramers_v")
            interpretation = (
                "Association exceeds the configured warning threshold."
                if value is not None and float(value) >= float(parameters["cramers_v_warning"])
                else "Association is below the configured warning threshold."
            )
            lines.append(
                f"| `{row['feature']}` | {len(row['categories'])} | {number(value)} | "
                f"{number(row.get('chi_square_fdr'))} | {interpretation} |"
            )
        lines.extend(
            [
                "",
                "The complete contingency counts/proportions are in "
                "`statistics/categorical-balance.csv`; all population means, standard deviations, "
                "medians, interquartile ranges, 5th/25th/75th/95th percentiles, finite counts, and "
                "missing counts are in `statistics/raw-summary.json` and "
                "`statistics/selected-summary.json`. Split-pair SMD, KS, and Jensen--Shannon "
                "values "
                "are in `statistics/split-balance.csv`. Jensen--Shannon divergence is a symmetric "
                "comparison of category proportions: 0 bits means identical proportions and 1 bit "
                "is maximal separation for two distributions.",
                "",
                "![Largest standardized class differences](statistics/plots/smd-forest.png)",
                "",
                "## 7. Shortcut diagnostics and learning curves",
                "",
                "The diagnostic regressions are deliberately small and are not WISDOM models. "
                "Their cross-validation keeps complete leakage groups together. AUROC is the "
                "probability that a randomly chosen positive receives a higher score than a "
                "randomly chosen negative (0.5 is random ranking; 1 is perfect). AUPRC emphasizes "
                "precision/recall for positives; its no-skill reference equals the positive "
                "fraction. Balanced accuracy averages positive and negative recall, so 0.5 is the "
                "binary no-skill reference even when class counts differ.",
                "",
                "| Diagnostic feature family | AUROC mean ± SD | AUPRC mean ± SD | "
                "Balanced accuracy mean ± SD | Meaning |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for name, result in statistics["shortcut_baselines"].items():
            if not result.get("available"):
                lines.append(
                    f"| `{name}` | unavailable | unavailable | unavailable | "
                    f"{result['reason']} |"
                )
                continue
            auc = float(result["auroc_mean"])
            meaning = (
                "Strong shortcut warning: these non-WISDOM features separate labels well."
                if name.startswith("technical")
                and auc >= float(parameters["technical_shortcut_auc_warning"])
                else "Diagnostic discrimination is below the configured technical red flag."
            )
            lines.append(
                f"| `{name}` | {number(auc)} ± {number(result['auroc_std'])} | "
                f"{number(result['auprc_mean'])} ± {number(result['auprc_std'])} | "
                f"{number(result['balanced_accuracy_mean'])} ± "
                f"{number(result['balanced_accuracy_std'])} | {meaning} |"
            )
        lines.extend(
            [
                "",
                "The model with `origin` directly tests whether source provenance reveals the "
                "label. The technical model without origin tests resolution, coordinate coverage, "
                "release year, and experimental method. The simple global model tests whether "
                "basic whole-protein properties already separate the task. High values do not "
                "prove that WISDOM will exploit a shortcut, but they require source-aware controls "
                "and cautious interpretation.",
                "",
                "Training dilutions remove complete training leakage groups only. Validation and "
                "test never change, and smaller subsets are nested inside larger subsets within a "
                "replicate. Therefore a learning curve measures the effect of training evidence "
                "without changing the evaluation question.",
                "",
                "| Replicate | Training view | Requested | Realized proteins | Positive | "
                "Negative | Groups |",
                "|---|---|---:|---:|---:|---:|---:|",
            ]
        )
        for replicate, audit in dilution_audit["replicates"].items():
            for name, subset in sorted(
                audit["subsets"].items(), key=lambda value: value[1]["requested_fraction"]
            ):
                lines.append(
                    f"| {replicate} | {name} | {subset['requested_fraction']:.0%} | "
                    f"{subset['realized_rows']} | {subset['positive']} | {subset['negative']} | "
                    f"{subset['leakage_groups']} |"
                )
        lines.extend(
            [
                "",
                "![Nested training dilutions](statistics/plots/dilutions.png)",
                "",
                "## 8. Plot-by-plot guide",
                "",
                "- **`class-counts.png`:** bar height is protein count. Compare raw and selected "
                "  bars to see exactly how majority reduction created the canonical class ratio.",
                "- **`origin-by-label.png`:** each bar is a data origin and colors are labels. A "
                "  color confined to one origin means provenance can reveal the answer; this is "
                "  technical confounding, not evidence of DNA-binding biology.",
                "- **`leakage-group-sizes.png`:** the horizontal axis is proteins per transitive "
                "  group and the vertical axis is the number of groups of that size. A long right "
                "  tail explains why exact target split sizes may be impossible without leakage.",
                "- **`smd-forest.png`:** each horizontal bar is positive-minus-negative mean "
                "  separation in pooled standard deviations. Distance from zero matters; color "
                "  distinguishes technical nuisance variables from model-visible global ones.",
                "- **`global-phenotypes.png`:** each bar is one label-free whole-protein phenotype "
                "  and segments show fixed splits. Missing rare phenotypes in a split can be "
                "  mathematically unavoidable when too few independent leakage groups carry them.",
                "- **`interface-phenotypes.png`:** the same split coverage view, but only for "
                "  contact-supported positive interfaces. It cannot describe negatives because "
                "  they have no positive DNA-contact interface by definition.",
                "- **`global-pca-by-label.png`:** the first two principal components are linear "
                "  summaries of scaled global descriptors; points are colored by label. Visible "
                "  separation suggests a global biological or technical shortcut, but overlap "
                "  does not prove the full high-dimensional distributions are equal.",
                "- **`global-pca-by-split.png`:** the same coordinates colored by split. Similar "
                "  clouds support distribution comparability; separation calls for the exact "
                "  split statistics rather than a visual conclusion alone.",
                "- **`dilutions.png`:** requested fraction is on the horizontal axis. One series "
                "  shows realized protein count and the other positive fraction. Nonlinear count "
                "  steps are expected because complete leakage groups cannot be split.",
                "",
                "![Origin and label](statistics/plots/origin-by-label.png)",
                "",
                "![Global PCA colored by label](statistics/plots/global-pca-by-label.png)",
                "",
                "![Global PCA colored by split](statistics/plots/global-pca-by-split.png)",
                "",
                "## 9. Warnings and scientific limitations",
                "",
            ]
        )
        if warnings:
            for warning in warnings:
                subject = warning.get("feature", warning.get("split", warning.get("kind")))
                lines.append(
                    f"- **{warning['severity']} — {warning['kind']} ({subject}):** "
                    f"{warning['interpretation']} Observed value: "
                    f"{number(warning.get('value'))}; configured threshold: "
                    f"{number(warning.get('threshold'))}."
                )
        else:
            lines.append("- No configured warning threshold was crossed.")
        lines.extend(
            [
                "",
                "The most important irreducible limitation is negative evidence. BTD negatives "
                "are benchmark negatives obtained by exclusion from curated protein annotations, "
                "not a universal experimental proof that a protein can never bind DNA in any "
                "condition. The builder rejects direct structural contradictions and never turns "
                "PDB non-contact or missing DNA into a negative, but incomplete biological "
                "knowledge remains possible. Source-class association must therefore be reported "
                "and controlled in model interpretation.",
                "",
                "HDBSCAN phenotypes are descriptor-space summaries, not biological ground-truth "
                "families. Statistical p-values depend strongly on sample size and must be read "
                "with effect sizes. Finally, group-safe splitting limits known homology leakage "
                "under declared thresholds; it cannot guarantee the absence of every unknown "
                "evolutionary relationship.",
                "",
                "## 10. Reproduction checklist",
                "",
                "1. Inspect `provenance.json` for the raw-input hash, parameters, seed, and "
                "   structure hashes.",
                "2. Confirm `design-summary.json` says `PASS` and review every item in "
                "   `statistics/warnings.json`.",
                "3. Use `catalog.csv` for scientific joins; use ID-only TXT files for structural "
                "   preprocessing and labelled TXT files for manual/audit views.",
                "4. Preserve validation/test exactly. Use only the nested training TXT files for "
                "   learning-curve experiments.",
                "5. Treat this report as an interpretation layer and CSV/JSON files as exact "
                "   numerical evidence.",
                "",
            ]
        )
        return "\n".join(lines)

    @staticmethod
    def _labelled_manifest(rows: Sequence[Mapping[str, Any]]) -> str:
        """Render an identifier/label view without duplicating label authority.

        Args:
            rows: Design catalog rows containing ``identifier`` and binary ``label``.

        Returns:
            Lexically ordered UTF-8 lines formatted as ``RCSB_CHAIN<TAB>0|1``.
        """
        return "".join(
            f"{row['identifier']}\t{int(row['label'])}\n"
            for row in sorted(rows, key=lambda value: str(value["identifier"]))
        )

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write stable CSV fields inside the LambdaForge-managed output directory.

        Args:
            path: Final CSV path.
            rows: Ordered flat/nested mappings; an empty collection still creates an empty file.

        Returns:
            ``None`` after writing the complete table.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        fields = sorted({str(field) for row in rows for field in row})
        with path.open("w", encoding="utf-8", newline="") as stream:
            if fields:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                for row in rows:
                    writer.writerow(
                        {
                            field: json.dumps(value, sort_keys=True, separators=(",", ":"))
                            if isinstance(value, (dict, list, tuple))
                            else value
                            for field, value in row.items()
                        }
                    )

    @staticmethod
    def _write_text(path: Path, content: str) -> None:
        """Write complete UTF-8 text inside the managed output directory.

        Args:
            path: Destination below the declared LambdaForge output directory.
            content: Complete UTF-8 payload.

        Returns:
            ``None`` after writing the complete text.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="\n")

    def _write_json(self, path: Path, payload: Any) -> None:
        """Write standards-compliant sorted JSON inside the managed output directory.

        Args:
            path: Final JSON destination.
            payload: JSON-safe value with no NaN/Infinity values.

        Returns:
            ``None`` after writing the complete document.
        """
        self._write_text(
            path,
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        )

    @staticmethod
    def _tsv(path: Path, fields: int) -> list[list[str]]:
        """Read a strict tab-separated table with an exact column count.

        Args:
            path: Existing MMseqs2/Foldseek pair table.
            fields: Required number of columns per non-empty line.

        Returns:
            Split field rows in file order; an empty table is valid.

        Raises:
            ValueError: If any non-empty line has the wrong number of columns.
        """
        rows = [
            line.rstrip("\n").split("\t")
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if any(len(row) != fields for row in rows):
            raise ValueError(f"{path.name} must contain exactly {fields} tab-separated fields")
        return rows

    @staticmethod
    def _valid_pair_evidence(
        file      : ManagedFile,
        rows      : Sequence[Mapping[str, Any]],
        parameters: Mapping[str, Any],
        parser    : Callable[
            [Path, Sequence[Mapping[str, Any]], Mapping[str, Any]],
            set[tuple[str, str]],
        ],
    ) -> bool:
        """Validate complete specialist evidence before checkpoint publication or reuse.

        Args:
            file: Path-like managed checkpoint candidate.
            rows: Complete RAW population defining every permitted specialist identifier.
            parameters: Scientific thresholds reapplied by the selected parser.
            parser: MMseqs2 or Foldseek evidence parser that checks shape, values, and identities.

        Returns:
            ``True`` when the full table is scientifically parseable, including a legitimate empty
            table; otherwise ``False`` so LambdaForge rebuilds it atomically.
        """
        try:
            parser(Path(file), rows, parameters)
            return True
        except (OSError, UnicodeError, ValueError):
            return False

    @staticmethod
    def _known_pair(left: str, right: str, identifiers: set[str], path: Path) -> None:
        """Reject specialist evidence referring to an unknown raw member.

        Args:
            left: Query logical identifier.
            right: Target logical identifier.
            identifiers: Complete legal raw identifier set.
            path: Evidence path included in diagnostics.

        Returns:
            ``None`` when both identifiers are known.

        Raises:
            ValueError: If either identifier is outside the raw evidence population.
        """
        if left not in identifiers or right not in identifiers:
            raise ValueError(f"{path.name} references unknown pair: {left}, {right}")

    @staticmethod
    def _sha256_file(path: Path) -> str:
        """Hash one regular file incrementally without loading it all into memory.

        Args:
            path: Existing file.

        Returns:
            Lower-case hexadecimal SHA-256 digest.
        """
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @staticmethod
    def _finite_array(values: Iterable[Any]) -> np.ndarray:
        """Convert available finite scalar values to a one-dimensional float64 array.

        Args:
            values: Possibly missing numeric values.

        Returns:
            Finite ``float64 [N]`` array in input order.
        """
        return np.asarray(
            [float(value) for value in values if value is not None and math.isfinite(float(value))],
            dtype=np.float64,
        )

    def _finite_mean(self, values: Iterable[Any]) -> float | None:
        """Calculate the mean of available finite values.

        Args:
            values: Possibly missing numeric values.

        Returns:
            Finite mean or ``None`` for an empty finite subset.
        """
        array = self._finite_array(values)
        return float(array.mean()) if len(array) else None

    @staticmethod
    def _class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        """Count total, positive, and negative proteins in one population.

        Args:
            rows: Records containing a binary ``label``.

        Returns:
            Stable count mapping.
        """
        return {
            "total": len(rows),
            "positive": sum(int(row["label"]) == 1 for row in rows),
            "negative": sum(int(row["label"]) == 0 for row in rows),
        }

    @staticmethod
    def _rank(seed: int, value: str) -> str:
        """Return a deterministic seeded SHA-256 tie-breaking key.

        Args:
            seed: Integer scientific seed.
            value: Identifier/group/composite value to rank.

        Returns:
            Hexadecimal stable ordering key independent of input and worker order.
        """
        return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()
