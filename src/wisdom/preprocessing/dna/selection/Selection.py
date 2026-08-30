"""Readable LambdaForge Work that designs the WISDOM-DNA benchmark."""

import lambdaforge as lf

from typing import Any
from pathlib import Path
from wisdom.preprocessing.dna.selection.audit import audit_dataset
from wisdom.preprocessing.dna.selection.report import write_design
from wisdom.preprocessing.dna.selection.splits import assign_splits
from wisdom.preprocessing.dna.selection.evidence import load_evidence
from wisdom.preprocessing.dna.selection.dilutions import create_dilutions
from wisdom.preprocessing.dna.selection.population import select_population
from wisdom.preprocessing.dna.selection.phenotypes import assign_phenotypes
from wisdom.preprocessing.dna.selection.leakage import assign_leakage_groups
from wisdom.preprocessing.dna.selection.similarity import compute_similarity
from wisdom.preprocessing.dna.selection.structures import analyse_structures, snapshot_structures


class Selection(lf.Work):
    """Turn frozen protein evidence into one balanced, leakage-safe dataset design."""

    def run(
        self,

        # A skipped Selection either remains disabled or forwards one prior audited design.

        skip            : bool        = False,
        existing_design : Path | None = None,

        # Immutable evidence and bounded execution.

        raw_path                    : Path | None       = None,
        workers                     : int               = 36,
        requests_per_second         : float             = 60.0,
        retries                     : int               = 5,
        output_directory            : str | None        = "../data/dna/design",
        overwrite_output            : bool              = True,

        # Sequence and structure relations that define potential information leakage.

        sequence_identity           : float             = 0.30,
        sequence_coverage           : float             = 0.80,
        sequence_evalue             : float             = 1e-3,
        foldseek_probability        : float             = 0.90,
        foldseek_tmscore            : float             = 0.75,
        foldseek_coverage           : float             = 0.80,
        foldseek_evalue             : float             = 1e-3,
        group_same_pdb              : bool              = True,

        # Structural quality and descriptive physical phenotypes.

        maximum_resolution          : float | None      = 4.0,
        interface_region_distance   : float             = 8.0,
        global_min_cluster_size     : int               = 15,
        global_min_samples          : int               = 2,
        interface_min_cluster_size  : int               = 20,
        interface_min_samples       : int               = 5,

        # Canonical balancing, fixed splits, and learning-curve views.

        positive_negative_ratio     : float             = 1.0,
        keep_all_negatives          : bool              = True,
        retain_core_positives       : bool              = True,
        train_fraction              : float             = 0.70,
        validation_fraction         : float             = 0.15,
        test_fraction               : float             = 0.15,
        dilution_fractions          : tuple[float, ...] = (1.0, 0.75, 0.50, 0.25, 0.10),
        dilution_replicates         : int               = 1,
        seed                        : int               = 2026,

        # Native specialist tools and optional detailed logging.

        mmseqs_executable           : str               = "mmseqs",
        foldseek_executable         : str               = "foldseek",
        verbose                     : bool              = False,
    ) -> dict[str, Any]:
        """Execute the complete selection as ten explicit scientific stages.

        Args:
            skip: Avoid every selection computation when true. If ``existing_design`` is supplied,
                its named outputs are forwarded to a subsequent preprocessing step.
            existing_design: Optional complete prior Selection directory used while skipping. When
                supplied, it must contain the labelled manifests, catalog, dilutions, and snapshot.
            raw_path: Frozen JSONL evidence with one protein chain per line.
            workers: Maximum concurrent PDB retrieval workers and specialist-tool threads.
            requests_per_second: Aggregate RCSB request-start ceiling.
            retries: Additional attempts for one failed RCSB request.
            output_directory: Optional convenient copy of the managed design directory.
            overwrite_output: Replace that convenient copy after a successful Work.
            sequence_identity: Minimum MMseqs2 aligned sequence identity.
            sequence_coverage: Minimum MMseqs2 coverage in both directions.
            sequence_evalue: Largest accepted MMseqs2 expectation value.
            foldseek_probability: Minimum Foldseek homolog probability.
            foldseek_tmscore: Minimum query- and target-normalized TM-score.
            foldseek_coverage: Minimum Foldseek coverage in both directions.
            foldseek_evalue: Largest accepted Foldseek expectation value.
            group_same_pdb: Join candidates from the same PDB deposition into one leakage group.
            maximum_resolution: Largest canonical experimental resolution in ångströms, or none.
            interface_region_distance: Contact-residue graph radius in ångströms.
            global_min_cluster_size: Smallest HDBSCAN global phenotype cluster.
            global_min_samples: HDBSCAN global core-neighbour count.
            interface_min_cluster_size: Smallest positive-interface phenotype cluster.
            interface_min_samples: HDBSCAN interface core-neighbour count.
            positive_negative_ratio: Selected positive count divided by negative count.
            keep_all_negatives: Retain every quality-eligible curated negative.
            retain_core_positives: Prefer BTD-Core positives while filling the positive quota.
            train_fraction: Target fraction assigned to training by complete leakage groups.
            validation_fraction: Target fraction assigned to validation.
            test_fraction: Target fraction assigned to final testing.
            dilution_fractions: Nested fractions of the fixed training population.
            dilution_replicates: Number of deterministic alternative group orderings.
            seed: Reproducible tie-breaking seed.
            mmseqs_executable: MMseqs2 binary name or path.
            foldseek_executable: Foldseek binary name or path.
            verbose: Log every evidence/PDB item instead of periodic summaries.

        Returns:
            Counts for the raw, selected, train, validation, and test populations.

        Raises:
            FileNotFoundError: If evidence or an external tool is absent.
            ValueError: If a supplied design, frozen evidence, or scientific stage is inconsistent.
            RuntimeError: If a specialist program or final audit fails.
        """
        if skip:
            if existing_design is None:
                self.log("Selection disabled; no design input or outputs were requested")
                return {"skipped": True, "forwarded": False}

            existing = {
                "train":      existing_design / "train-labelled.txt",
                "validation": existing_design / "validation-labelled.txt",
                "test":       existing_design / "test-labelled.txt",
                "catalog":    existing_design / "catalog.csv",
                "dilutions":  existing_design / "dilutions",
                "structures": existing_design / "structures",
            }
            missing = [name for name, path in existing.items() if not path.exists()]
            if missing:
                raise ValueError(f"existing design is missing required paths: {missing}")

            # No evidence, structure, similarity, clustering, balancing, or split operation runs.
            # These registrations only give the next LambdaForge step immutable named inputs.

            self.log("Selection skipped; forwarding the existing audited design")
            for name, path in existing.items():
                self.outputs.artifact(name, path, role="dataset-design-input")
            return {"skipped": True, "forwarded": True}

        if raw_path is None:
            raise ValueError("raw_path is required when Selection is not skipped")

        # ================================================================================
        # 0. Verify the native scientific tools.
        #
        # Input
        #   The executable names configured in the YAML.
        # Output
        #   LambdaForge Tool objects carrying the resolved paths and detected versions.
        # Why
        #   MMseqs2 and Foldseek are needed much later, after structure analysis. Resolving them
        #   now makes a missing cluster dependency fail immediately instead of wasting hours of
        #   downloads and descriptor computation before discovering an unusable environment.
        # ================================================================================

        self.log("Checking MMseqs2 and Foldseek before reading the dataset")

        mmseqs   = self.tools.require(mmseqs_executable, version_args=["version"])
        foldseek = self.tools.require(foldseek_executable, version_args=["version"])

        self.log(f"MMseqs2 ready: {mmseqs.version or 'version unavailable'}")
        self.log(f"Foldseek ready: {foldseek.version or 'version unavailable'}")

        # ================================================================================
        # 1. Load the immutable public evidence.
        #
        # Input
        #   One JSON object per line with an identifier, sequence, binary label, assembly, copy,
        #   source, and explicit label evidence.
        # Output
        #   An identifier-sorted list of normalized dictionaries. No scientific candidate is
        #   selected or discarded here.
        # Why
        #   Labels must come from frozen evidence rather than from filenames or from the absence
        #   of DNA in one deposited structure. Keeping this step separate makes that boundary
        #   visible before coordinates influence any decision.
        # ================================================================================

        rows = load_evidence(self, raw_path, verbose)

        # ================================================================================
        # 2. Verify structures and calculate physical descriptors.
        #
        # Input
        #   The evidence rows and their exact PDB, chain, biological assembly, and copy identity.
        # Output
        #   The same rows enriched with verified sequence/contact evidence, structure hashes,
        #   quality status, global descriptors, positive-interface descriptors, and one managed
        #   protein-only mmCIF reference for Foldseek.
        # Why
        #   A PDB identifier alone does not specify the physical object represented by a record.
        #   Reconstructing the declared biological assembly is necessary to verify that positives
        #   really contact DNA and that explicit negatives do not contradict their evidence.
        # ================================================================================

        rows = analyse_structures(
            self,
            rows,
            workers                   = workers,
            requests_per_second       = requests_per_second,
            retries                   = retries,
            maximum_resolution        = maximum_resolution,
            interface_region_distance = interface_region_distance,
            verbose                   = verbose,
        )

        # ================================================================================
        # 3. Calculate full-population sequence and structure similarity.
        #
        # Input
        #   Every RAW sequence and every verified protein-only assembly copy.
        # Output
        #   Auditable MMseqs2 and Foldseek tables plus the undirected pairs that satisfy the
        #   configured identity, coverage, probability, TM-score, and E-value thresholds.
        # Why
        #   Similar proteins are statistically dependent examples. Comparing all RAW candidates
        #   before balancing prevents an omitted positive from hiding a homology bridge between
        #   proteins that would otherwise be placed in different evaluation roles.
        # ================================================================================

        similarity = compute_similarity(
            self,
            rows,
            mmseqs                   = mmseqs,
            foldseek                 = foldseek,
            workers                  = workers,
            sequence_identity        = sequence_identity,
            sequence_coverage        = sequence_coverage,
            sequence_evalue          = sequence_evalue,
            foldseek_probability     = foldseek_probability,
            foldseek_tmscore         = foldseek_tmscore,
            foldseek_coverage        = foldseek_coverage,
            foldseek_evalue          = foldseek_evalue,
        )

        # ================================================================================
        # 4. Convert pairwise relations into indivisible leakage groups.
        #
        # Input
        #   Exact-sequence relations, accepted MMseqs2/Foldseek pairs, and optionally same-PDB
        #   provenance relations.
        # Output
        #   One transitive leakage-group identifier on every RAW row and complete edge/component
        #   evidence for later validation.
        # Why
        #   Pairwise checks are not sufficient: if A resembles B and B resembles C, A and C are
        #   indirectly dependent even without a direct edge. Connected components enforce this
        #   transitive constraint and become atomic units for split assignment and dilution.
        # ================================================================================

        rows, leakage = assign_leakage_groups(rows, similarity, group_same_pdb)

        # ================================================================================
        # 5. Describe the physical diversity represented by the candidates.
        #
        # Input
        #   Label-free whole-protein descriptors for all eligible rows and contact-interface
        #   descriptors for verified positives only.
        # Output
        #   Global and positive-interface HDBSCAN phenotype labels, membership probabilities,
        #   and concise diagnostics. Noise remains an explicit valid outcome.
        # Why
        #   Leakage groups protect independence but do not describe physical variety. Phenotypes
        #   provide a second, descriptive view that helps avoid selecting or splitting only one
        #   dominant shape while never being treated as a biological class or model input.
        # ================================================================================

        rows, phenotypes = assign_phenotypes(
            rows,
            global_min_cluster_size    = global_min_cluster_size,
            global_min_samples         = global_min_samples,
            interface_min_cluster_size = interface_min_cluster_size,
            interface_min_samples      = interface_min_samples,
            workers                    = workers,
        )

        # ================================================================================
        # 6. Select the balanced canonical population.
        #
        # Input
        #   Quality-eligible RAW rows with leakage groups, phenotypes, origins, and fixed labels.
        # Output
        #   A canonical positive/negative population and an audit of every omitted candidate.
        # Why
        #   Reliable curated negatives are scarcer than positives. The default therefore keeps
        #   every eligible negative and chooses the requested positive quota while spreading the
        #   choices over dependency groups, phenotypes, and evidence sources.
        # ================================================================================

        selected, selection_audit = select_population(
            rows,
            positive_negative_ratio = positive_negative_ratio,
            keep_all_negatives       = keep_all_negatives,
            retain_core_positives    = retain_core_positives,
            seed                     = seed,
        )

        # ================================================================================
        # 7. Assign the canonical population to train, validation, and test.
        #
        # Input
        #   The selected rows grouped by their full-RAW leakage components.
        # Output
        #   One fixed split on every selected row plus counts and group assignments.
        # Why
        #   The model must never be evaluated on a protein whose sequence, structure, exact
        #   sequence, or PDB provenance was available through another split. Whole groups are
        #   therefore assigned together, even when this prevents perfectly exact split sizes.
        # ================================================================================

        selected, split_audit = assign_splits(
            selected,
            train_fraction      = train_fraction,
            validation_fraction = validation_fraction,
            test_fraction       = test_fraction,
            seed                = seed,
        )

        # ================================================================================
        # 8. Create nested training dilutions for learning curves.
        #
        # Input
        #   Only the fixed training rows and their indivisible leakage groups.
        # Output
        #   Deterministic nested train subsets for each fraction and replicate, together with
        #   hashes proving that validation and test remain unchanged.
        # Why
        #   Learning curves must measure the effect of less training evidence, not a different
        #   evaluation problem. Reducing complete groups avoids introducing leakage inside the
        #   smaller views and preserves direct comparability between fractions.
        # ================================================================================

        dilutions = create_dilutions(
            selected,
            fractions  = dilution_fractions,
            replicates = dilution_replicates,
            seed       = seed,
        )

        # ================================================================================
        # 9. Enforce the final scientific invariants.
        #
        # Input
        #   RAW rows, canonical split rows, and every nested training view.
        # Output
        #   A machine-readable audit with class/group/phenotype/origin counts and warnings.
        # Why
        #   A successful program is not automatically a valid benchmark. Duplicate identities,
        #   crossed groups, missing evaluation classes, non-nested subsets, or fragmented groups
        #   are hard failures and must stop publication rather than appear only in a report.
        # ================================================================================

        audit = audit_dataset(rows, selected, dilutions)

        # ================================================================================
        # 10. Publish one complete and portable design artifact.
        #
        # Input
        #   Every scientific result and the exact thresholds that produced it.
        # Output
        #   A LambdaForge-managed directory containing catalogs, labelled/plain manifests,
        #   specialist evidence, leakage/phenotype tables, nested dilutions, JSON audits, the
        #   exact selected mmCIF snapshot, and a readable Markdown interpretation.
        # Why
        #   Downstream preprocessing must consume an immutable decision, not rerun selection or
        #   infer labels. LambdaForge publishes the directory atomically only after every file has
        #   been written successfully, so partial designs never replace a valid existing copy.
        # ================================================================================

        output = self.outputs.directory(
            "dataset-design",
            role       = "dataset-design",
            publish_to = output_directory,
            overwrite  = overwrite_output,
        )

        parameters = {
            "seed":                 seed,
            "group_same_pdb":       group_same_pdb,
            "sequence_evalue":      sequence_evalue,
            "foldseek_evalue":      foldseek_evalue,
            "sequence_identity":    sequence_identity,
            "sequence_coverage":    sequence_coverage,
            "foldseek_tmscore":     foldseek_tmscore,
            "foldseek_coverage":    foldseek_coverage,
            "maximum_resolution":   maximum_resolution,
            "foldseek_probability": foldseek_probability,
        }

        # The exact uncompressed files used above are deterministically compressed into the
        # portable design. Their hashes now protect stored evidence instead of assuming that a
        # mutable RCSB download URL will return identical bytes in a later Work.

        snapshot_structures(Path(output) / "structures", selected)

        write_design(
            Path(output),
            raw             = rows,
            selected        = selected,
            leakage         = leakage,
            phenotypes      = phenotypes,
            dilutions       = dilutions,
            selection_audit = selection_audit,
            split_audit     = split_audit,
            audit           = audit,
            similarity      = similarity,
            parameters      = parameters,
        )

        # Expose compact membership, scientific metadata, dilution views, and exact coordinates.
        # The same six-output contract is used when Selection is skipped.

        for name, filename in (
            ("train",      "train-labelled.txt"),
            ("validation", "validation-labelled.txt"),
            ("test",       "test-labelled.txt"),
            ("catalog",    "catalog.csv"),
            ("dilutions",  "dilutions"),
            ("structures", "structures"),
        ):
            self.outputs.artifact(
                name,
                Path(output) / filename,
                role       = "dataset-manifest",
            )

        summary = {
            "raw":        len(rows),
            "test":       sum(row["split"] == "test" for row in selected),
            "train":      sum(row["split"] == "train" for row in selected),
            "selected":   len(selected),
            "validation": sum(row["split"] == "validation" for row in selected),
        }

        return summary
