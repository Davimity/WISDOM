"""Readable LambdaForge Work that turns three fixed manifests into WISDOM NPZ data."""

import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections.abc import Sequence
from wisdom.preprocessing.dna.preprocessing.geometry import generate_geometry
from wisdom.preprocessing.dna.preprocessing.publication import publish_dataset
from wisdom.preprocessing.dna.preprocessing.annotations import generate_annotations
from wisdom.preprocessing.dna.preprocessing.DatasetManifests import DatasetManifests
from wisdom.preprocessing.dna.preprocessing.structures import validate_structure_snapshot


class Preprocessing(lf.Work):
    """Generate universal geometry, DNA sidecars, and one immutable DatasetVersion."""

    def run(
        self,

        # Skip the complete dataset build when only Selection should run.

        skip                : bool            = False,

        # The complete fixed population is represented by exactly three immutable files.

        train               : Path | None     = None,
        validation          : Path | None     = None,
        test                : Path | None     = None,
        catalog             : Path | None     = None,
        dilutions           : Path | None     = None,
        structures          : Path | None     = None,
        dataset_name        : str             = "wisdom-dna",
        dataset_version     : str             = "5",

        # Operational concurrency and researcher-facing progress.

        workers             : int             = 36,
        progress_log_seconds: float           = 120.0,
        verbose             : bool            = False,

        # Universal, label-free protein geometry.

        surface_resolution  : float           = 1.0,
        probe_radius        : float           = 1.4,
        atom_spatial_radius          : float           = 6.0,
        atom_spatial_k_max           : int             = 32,
        surface_atom_radius          : float           = 6.0,
        surface_atom_k_max           : int             = 32,
        diffusion_spectral_modes_max : int             = 128,
        surface_neighbor_k_max       : int             = 24,
        curvature_scales    : Sequence[float] = (2.5, 5.0),

        # Evaluation-only projection of DNA contacts onto the fixed surface.

        positive_gap        : float           = 1.4,
        negative_gap        : float           = 3.0,
        sensitivity_gaps    : Sequence[float] = (1.0, 1.4, 2.0),
    ) -> dict[str, Any]:
        """Execute preprocessing as five explicit, resumable scientific stages.

        Args:
            skip: Complete without reading manifests or publishing a dataset when true.
            train: Training JSONL or ``identifier<TAB>label`` TXT file.
            validation: Validation JSONL or ``identifier<TAB>label`` TXT file.
            test: Testing JSONL or ``identifier<TAB>label`` TXT file.
            catalog: Selection catalog required only when the three manifests are labelled TXT.
            dilutions: Optional labelled training views included as dataset subsets.
            structures: Immutable coordinate snapshot published by the same Selection.
            dataset_name: Stable LambdaForge Dataset Registry family.
            dataset_version: Immutable release identifier selected by the researcher.
            workers: Concurrent structure-validation threads and spawned protein processes.
            progress_log_seconds: Seconds between liveness messages during long maps.
            verbose: Emit per-protein debug messages in addition to normal phase summaries.
            surface_resolution: Target surface point spacing in ångströms.
            probe_radius: Solvent-probe radius in ångströms.
            atom_spatial_radius: Spatial atom-neighbor cutoff in ångströms.
            atom_spatial_k_max: Maximum ranked spatial candidates persisted per atom.
            surface_atom_radius: Surface-to-atom neighborhood cutoff in ångströms.
            surface_atom_k_max: Maximum nearest atoms persisted per surface point.
            diffusion_spectral_modes_max: Maximum low-frequency surface modes persisted.
            surface_neighbor_k_max: Maximum nearest surface points used by differential operators.
            curvature_scales: Positive curvature radii in surface-resolution units.
            positive_gap: Largest confidently DNA-contacting surface gap in ångströms.
            negative_gap: Smallest confidently non-contacting gap in ångströms.
            sensitivity_gaps: Additional evaluation-only positive cutoffs in ångströms.

        Returns:
            Published dataset identity, member count, leakage-group count, and verdict.

        Raises:
            ValueError: If required manifests, resources, labels, structures, or arrays disagree.
            RuntimeError: If parallel processing, validation, or publication fails.
            OSError: If immutable inputs or checkpoint-owned outputs cannot be accessed.
        """
        if skip:
            self.log("Preprocessing skipped; no structures, NPZ files, or dataset were produced")
            return {"skipped": True}

        if train is None or validation is None or test is None or structures is None:
            raise ValueError(
                "train, validation, test, and structures are required when preprocessing runs"
            )

        if workers < 1 or workers > int(self.resources.cpu):
            raise ValueError(
                f"workers must be between 1 and the allocated {self.resources.cpu} CPUs"
            )

        if self.resuming:
            self.log(
                "Compatible checkpoints were found; LambdaForge will restore verified "
                "structures, universal NPZ files, and DNA sidecars"
            )

        # ==============================================================================
        # 1. Read the fixed supervised population.
        #
        # Input
        #   Three fixed split files. Current JSONL records are already complete; existing labelled
        #   TXT views are joined to the explicit catalog and optional dilution directory.
        # Output
        #   One common identifier-sorted representation containing labels, exact
        #   chain/assembly/copy identity, contact evidence, groups, and dilution memberships.
        # Why
        #   Membership remains easy to inspect in labelled TXT, while the catalog provides the
        #   scientific fields that cannot fit into two columns. No `structure_path` is expected:
        #   portable coordinate paths are resolved from the Selection snapshot in the next phase.
        # ==============================================================================

        manifests = DatasetManifests(
            train,
            validation,
            test,
            catalog,
            dilutions,
        )
        rows = manifests.load(self, verbose)

        # ==============================================================================
        # 2. Validate the exact coordinate files published by Selection.
        #
        # Input
        #   The immutable `structures` directory plus catalogued uncompressed SHA-256 values.
        # Output
        #   The same validated `.cif.gz` snapshot directory.
        # Why
        #   Selection and preprocessing must read identical physical evidence. Carrying those
        #   bytes inside the design avoids a second RCSB download and makes later public revisions
        #   irrelevant, while exact hashes still detect corruption of the stored snapshot.
        # ==============================================================================

        structure_root = validate_structure_snapshot(
            self,
            rows,
            structures,
            workers              = workers,
            progress_log_seconds = progress_log_seconds,
            verbose              = verbose,
        )

        # ==============================================================================
        # 3. Convert every protein into one universal, label-free NPZ.
        #
        # Input
        #   A selected PDB chain plus geometry settings shared by every supervised split.
        # Output
        #   Atomic features/bounded edges, fixed surface geometry, compact nearest-atom tables,
        #   and intrinsic diffusion operators. No DNA label or learned feature enters this archive.
        # Why
        #   WISDOM needs the same structural representation for every future scientific task.
        #   `resume_map` checkpoints each protein, and its validator reopens the complete NPZ and
        #   checks source/configuration hashes before accepting an interrupted result.
        # ==============================================================================

        _geometry_root, geometry_report = generate_geometry(
            self,
            rows,
            structure_root,
            workers              = workers,
            progress_log_seconds = progress_log_seconds,
            surface_resolution   = surface_resolution,
            probe_radius         = probe_radius,
            atom_spatial_radius          = atom_spatial_radius,
            atom_spatial_k_max           = atom_spatial_k_max,
            surface_atom_radius          = surface_atom_radius,
            surface_atom_k_max           = surface_atom_k_max,
            diffusion_spectral_modes_max = diffusion_spectral_modes_max,
            surface_neighbor_k_max       = surface_neighbor_k_max,
            curvature_scales     = curvature_scales,
            verbose              = verbose,
        )

        # ==============================================================================
        # 4. Project DNA evidence into a separate point-aligned sidecar.
        #
        # Input
        #   The immutable base NPZ and the exact assembly transform/contact metadata carried by
        #   each JSONL line. `structure_path` is intentionally not an input: it is materialized
        #   here from the verified compressed structure and its content digest.
        # Output
        #   One small `.dna.npz` with hard/soft surface targets, validity masks, DNA gaps, cutoff
        #   sensitivity labels, and the SHA-256 of its corresponding universal NPZ.
        # Why
        #   Keeping task labels outside universal geometry prevents DNA supervision from changing
        #   reusable structural data and makes point ordering/fingerprint agreement auditable.
        # ==============================================================================

        annotation_root, annotation_report = generate_annotations(
            self,
            rows,
            geometry_report,
            structure_root,
            workers              = workers,
            progress_log_seconds = progress_log_seconds,
            positive_gap         = positive_gap,
            negative_gap         = negative_gap,
            sensitivity_gaps     = sensitivity_gaps,
            verbose              = verbose,
        )

        # ==============================================================================
        # 5. Assemble, revalidate, and publish the portable dataset.
        #
        # Input
        #   Universal NPZ files, DNA sidecars, source structures, and the three fixed manifests.
        # Output
        #   A machine-readable JSON audit, a plain-language Markdown verdict, diagnostic figures,
        #   and one content-addressed LambdaForge DatasetVersion. Publication stops on any
        #   checksum, schema, numerical, split, group, target, or dilution failure.
        # Why
        #   Successful worker completion proves executability, not scientific correctness. The
        #   final boundary independently reopens every archive with pickle disabled and checks
        #   that targets remain aligned to the exact surface points they describe. Models then
        #   consume a Registry identity such as `wisdom-dna@5`, never an attempt path; LambdaForge
        #   registers it atomically only after every member and asset has been hashed.
        # ==============================================================================

        return publish_dataset(
            self,
            rows,
            annotation_root,
            geometry_report,
            annotation_report,
            dataset_name,
            dataset_version,
        )
