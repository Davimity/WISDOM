"""Readable LambdaForge Work that turns three fixed manifests into WISDOM NPZ data."""

import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.clustering.geometry import generate_geometry
from wisdom.preprocessing.dna.preprocessing.publication import publish_dataset
from wisdom.preprocessing.common.structure.assets.external.StructureTokens import (
    StructureTokens,
)
from wisdom.preprocessing.dna.preprocessing.annotations import generate_annotations
from wisdom.preprocessing.dna.preprocessing.DatasetManifests import DatasetManifests
from wisdom.preprocessing.common.structure.assets.external.ProteinModelAssets import (
    ProteinModelAssets,
)
from wisdom.preprocessing.common.independence.snapshots import validate_structure_snapshot
from wisdom.preprocessing.common.structure.assets.external.AtomicProteinExtractor import (
    AtomicProteinExtractor,
)
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars
from wisdom.preprocessing.common.structure.assets.external.ExternalProteinRepresentation import (
    ExternalProteinRepresentation,
)
from wisdom.preprocessing.common.structure.assets.external.ProteinRepresentationExtractor import (
    ProteinRepresentationExtractor,
)


class Preprocessing(lf.Work):
    """Generate universal geometry, DNA sidecars, and one immutable DatasetVersion."""

    def run(
        self,

        # Skip the complete dataset build when only Selection should run.

        skip                         : bool            = False,

        # The complete fixed population is represented by exactly three immutable files.

        train                        : Path | None     = None,
        validation                   : Path | None     = None,
        test                         : Path | None     = None,
        catalog                      : Path | None     = None,
        dilutions                    : Path | None     = None,
        structures                   : Path | None     = None,
        dataset_name                 : str             = "wisdom-dna",
        dataset_version              : str             = "5",
        include_full_train           : bool            = True,
        train_dilutions              : Sequence[str]   = (),
        include_validation           : bool            = True,
        include_test                 : bool            = True,

        # Operational concurrency and researcher-facing progress.

        workers                      : int             = 36,
        progress_log_seconds         : float           = 120.0,
        verbose                      : bool            = False,

        # Universal, label-free protein geometry.

        surface_resolution           : float           = 1.0,
        probe_radius                 : float           = 1.4,
        atom_spatial_radius          : float           = 6.0,
        atom_spatial_k_max           : int             = 32,
        surface_atom_radius          : float           = 6.0,
        surface_atom_k_max           : int             = 32,
        diffusion_spectral_modes_max : int             = 128,
        surface_neighbor_k_max       : int             = 24,
        curvature_scales             : Sequence[float] = (2.5, 5.0),

        # Evaluation-only projection of DNA contacts onto the fixed surface.

        positive_gap                 : float           = 1.4,
        negative_gap                 : float           = 3.0,
        sensitivity_gaps             : Sequence[float] = (1.0, 1.4, 2.0),

        # Optional universal assets never change the base NPZ configuration or task targets.

        atomic_context               : bool            = False,
        source_quality               : bool            = True,
        extended_surface_atom        : bool            = False,
        extended_radius              : float           = 12.0,
        extended_k_max               : int             = 96,
        sequence_context             : bool            = False,
        covalent_environment         : bool            = False,
        atomic_sasa                  : bool            = False,
        sasa_probe                   : float           = 1.4,
        sasa_samples                 : int             = 960,
        residue_graph                : bool            = False,
        residue_contact_radius       : float           = 8.0,
        residue_surface_radius       : float           = 12.0,
        surface_local_graph          : bool            = False,
        surface_graph_resolution     : float           = 1.0,
        spectral_modes               : int | None      = None,
        surface_view                 : bool            = False,
        view_id                      : int             = 1,
        view_resolution              : float | None    = None,
        structure_tokens             : bool            = False,

        # Frozen external representations need explicit staged weights, never auto-downloads.

        external_model               : str | None      = None,
        external_checkpoint          : Path | None     = None,
        external_identifier          : str             = "",
        external_revision            : str             = "",
        external_layer               : int             = -1,
        external_maximum_length      : int             = 1022,
        external_device              : str             = "cpu",
        external_sequence_policy     : str             = "fragments",
        static_embedding_table       : bool            = False,
        protein_models               : Path | None     = None,
        residue_ca_radius            : float | None    = None,
        residue_ca_k_max              : int             = 32,
        surface_environment          : bool            = False,
        environment_options          : Mapping[str, Any] | None = None,
        atomic_checkpoint            : Path | None = None,
        atomic_device                : str = "cpu",
        protein_model_kinds          : Sequence[str] | None = None,
        protein_static_tables        : bool = True,
    ) -> dict[str, Any]:
        """Execute preprocessing as five explicit, resumable scientific stages.

        Args:
            skip: Complete without reading manifests or publishing a dataset when true.
            residue_ca_radius: Independent CA radius in Å; None retains historical residue assets.
            residue_ca_k_max: Directed stored CA neighbor maximum, default 32.
            surface_environment: Generate masked depth/occlusion/void sidecars, default false.
            environment_options: SurfaceEnvironment parameters; omitted uses documented defaults.
            atomic_checkpoint: Optional official SLAE base checkpoint; never auto-downloads or
                installs native dependencies. Missing optional runtime fails before geometry.
            atomic_device: Logical extraction device, cpu by default; never broadens GPU grants.
            train: Training JSONL or ``identifier<TAB>label`` TXT file.
            validation: Validation JSONL or ``identifier<TAB>label`` TXT file.
            test: Testing JSONL or ``identifier<TAB>label`` TXT file.
            catalog: Selection catalog required only when the three manifests are labelled TXT.
            dilutions: Optional labelled training views included as dataset subsets.
            structures: Immutable coordinate snapshot published by the same Selection.
            dataset_name: Stable LambdaForge Dataset Registry family.
            dataset_version: Immutable release identifier selected by the researcher.
            include_full_train: Geometrically preprocess the complete canonical train split.
            train_dilutions: Nested training views to preprocess when full train is disabled.
            include_validation: Preprocess the complete fixed validation population.
            include_test: Preprocess the complete held-out test population. Disable during HPO
                dataset construction when test must remain unopened.
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
            atomic_context: Attach independently versioned atom/residue context, default false.
            source_quality: Include observed source quality when available, default true.
            extended_surface_atom: Attach expanded CSR candidates, default false.
            extended_radius: Positive optional candidate radius in Å, default 12.
            extended_k_max: Positive optional per-point storage cap, default 96.
            sequence_context: Include physical residue/token correspondence, default false.
            covalent_environment: Include covalent neighbour element/order counts, default false.
            atomic_sasa: Include absolute per-atom solvent exposure, default false.
            sasa_probe: Positive solvent-probe radius for SASA in Å, default 1.4.
            sasa_samples: Deterministic directions per atom, default 960, at least 24.
            residue_graph: Include minimum-heavy-atom contacts and surface CSR, default false.
            residue_contact_radius: Positive residue contact cutoff in Å, default 8.
            residue_surface_radius: Positive residue-to-surface cutoff in Å, default 12.
            surface_local_graph: Include normal/chord-filtered surface topology, default false.
            surface_graph_resolution: Positive coherence-rule scale in Å, default 1.
            spectral_modes: Optional independent fixed eigenmode budget, default None.
            surface_view: Build one alternate cloud with all its own operators, default false.
            view_id: Nonnegative deterministic sampling ID, default 1; zero preserves history.
            view_resolution: Optional positive view spacing in Å; None preserves base spacing.
            structure_tokens: Include official Foldseek 3Di, default false; requires the tool.
            external_model: None (default), esm2, esmc or saprot; weights stay outside the base.
            external_checkpoint: Typed complete local checkpoint directory; required for a model.
            external_identifier: Original external repository identity; required for a model.
            external_revision: Immutable staged model revision; required for a model.
            external_layer: Hidden-state layer index, default -1 (last).
            external_maximum_length: Disjoint residue chunk limit, default 1022, excluding CLS/EOS.
            external_device: Explicit extraction device, default cpu; no automatic GPU allocation.
            external_sequence_policy: fragments (default) or verified complete chains.
            static_embedding_table: Export input weights too; requires a model, default false.
            protein_models: Typed directory containing checkpoints.json and selected pinned weights.
                None preserves the single-model interface; otherwise extract one model/member at
                a time. It cannot be combined with external_model.
            protein_model_kinds: esm2/esmc/saprot subset; None selects all three.
            protein_static_tables: Export selected input tables, default true for compatibility.

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

        # Reject an occupied immutable release before spending time on geometry or enrichment.

        self.outputs.dataset_preflight(name=dataset_name, version=dataset_version)

        if protein_models is not None and external_model is not None:
            raise ValueError("protein_models cannot be combined with the single external_model")
        model_assets = (ProteinModelAssets(protein_models, kinds=protein_model_kinds,
            static_tables=protein_static_tables) if protein_models is not None else None)
        need_tokens = structure_tokens or external_model == "saprot" or (
            model_assets is not None and model_assets.requires_structure_tokens)
        if need_tokens:
            StructureTokens(self)
        if model_assets is not None:
            model_assets.preflight()

        # Resolve specialist tools and optional model dependencies before expensive geometry.
        # Missing requested inputs fail here, never after hours of preprocessing or by omission.

        external = None
        if external_model is not None:
            if external_checkpoint is None:
                raise ValueError("external_model requires an explicit local external_checkpoint")
            external = ExternalProteinRepresentation(
                ProteinRepresentationExtractor(
                    external_model, external_checkpoint, external_identifier, external_revision,
                    external_layer, external_maximum_length, external_device,
                ),
                external_sequence_policy,
            )

        atomic_external = (AtomicProteinExtractor(atomic_checkpoint, atomic_device, work=self)
                           if atomic_checkpoint is not None else None)
        if atomic_external is not None and workers != 1:
            raise ValueError("frozen atomic extraction requires workers=1")
        sidecars = StructuralSidecars(
            atomic_context, source_quality, extended_surface_atom, extended_radius, extended_k_max,
            sequence_context         = sequence_context or model_assets is not None,
            covalent_environment     = covalent_environment,
            atomic_sasa              = atomic_sasa,
            sasa_probe               = sasa_probe,
            sasa_samples             = sasa_samples,
            residue_graph            = residue_graph,
            residue_contact_radius   = residue_contact_radius,
            residue_surface_radius   = residue_surface_radius,
            residue_ca_radius        = residue_ca_radius,
            residue_ca_k_max          = residue_ca_k_max,
            surface_environment      = surface_environment,
            environment_options      = environment_options,
            atomic_external          = atomic_external,
            surface_local_graph      = surface_local_graph,
            surface_graph_resolution = surface_graph_resolution,
            spectral_modes           = spectral_modes,
            surface_view             = surface_view,
            view_id                  = view_id,
            view_resolution          = view_resolution,
            structure_tokens         = need_tokens,
            external                 = external,
            static_embedding_table   = static_embedding_table,
        )
        required_assets = sidecars.requested_families | (
            model_assets.requested_families if model_assets is not None else frozenset())
        self.log(f"Required dataset assets: {', '.join(sorted(required_assets))}")

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
        rows = manifests.load(
            self,
            verbose,
            include_full_train = include_full_train,
            train_dilutions    = train_dilutions,
            include_validation = include_validation,
            include_test       = include_test,
        )

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
            atom_spatial_radius   = atom_spatial_radius,
            atom_spatial_k_max    = atom_spatial_k_max,
            surface_atom_radius   = surface_atom_radius,
            surface_atom_k_max    = surface_atom_k_max,
            diffusion_spectral_modes_max = diffusion_spectral_modes_max,
            surface_neighbor_k_max       = surface_neighbor_k_max,
            curvature_scales             = curvature_scales,
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
        #   and one content-addressed LambdaForge DatasetVersion with every requested structural
        #   family. Publication stops on missing assets, checksum, schema, numerical, split,
        #   group, target, or dilution failure. Alternative surfaces never reuse base GT by index.
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
            structural_sidecars = sidecars,
            workers = workers,
            protein_models = model_assets,
        )
