"""Publish optional structural context for an existing dataset, without rebuilding geometry."""

import lambdaforge as lf

from typing import Any
from pathlib import Path
from typing import ClassVar
from collections.abc import Mapping
from lambdaforge.data import DatasetIndex
from lambdaforge.work import RecoveryAllocationGuard
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars
from wisdom.preprocessing.common.structure.assets.physics.PhysicalFields import PhysicalFields
from wisdom.preprocessing.common.structure.io.VerifiedStructureSource import VerifiedStructureSource
from wisdom.preprocessing.common.structure.assets.external.StructureTokens import (
    StructureTokens,
)
from wisdom.preprocessing.common.structure.assets.external.AtomicProteinExtractor import (
    AtomicProteinExtractor,
)
from wisdom.preprocessing.common.structure.assets.external.ExternalProteinRepresentation import (
    ExternalProteinRepresentation,
)
from wisdom.preprocessing.common.structure.assets.external.ProteinRepresentationExtractor import (
    ProteinRepresentationExtractor,
)


class StructuralEnrichment(lf.Work):
    """Reuse immutable base and task assets and attach common independently validated sidecars."""

    recovery_allocation_guards: ClassVar[Mapping[str, RecoveryAllocationGuard]] = {
        "physics_maximum_voxels": RecoveryAllocationGuard(
        reason="Raise only the APBS allocation refusal ceiling; physical grids, equations and "
               "checkpoint validation remain unchanged.")}

    def run(
        self, dataset: Path, dataset_name: str, dataset_version: str,
        atomic_context: bool = True, source_quality: bool = True,
        extended_surface_atom: bool = True, extended_radius: float = 12.0,
        extended_k_max: int = 96, workers: int = 4,
        sequence_context: bool = False, atomic_sasa: bool = False,
        sasa_probe: float = 1.4, sasa_samples: int = 960, residue_graph: bool = False,
        residue_contact_radius: float = 8.0, residue_surface_radius: float = 12.0,
        surface_local_graph: bool = False, surface_graph_resolution: float = 1.0,
        surface_view: bool = False, view_id: int = 1, view_resolution: float | None = None,
        external_model: str | None = None, external_checkpoint: Path | None = None,
        external_identifier: str = "", external_revision: str = "", external_layer: int = -1,
        external_maximum_length: int = 1022, external_device: str = "cpu",
        external_sequence_policy: str = "fragments", structure_tokens: bool = False,
        static_embedding_table: bool = False,
        covalent_environment: bool = False, spectral_modes: int | None = None,
        residue_ca_radius: float | None = None, residue_ca_k_max: int = 32,
        surface_environment: bool = False, environment_options: Mapping[str, Any] | None = None,
        atomic_checkpoint: Path | None = None,
        atomic_device: str = "cpu",
        physics_options: Mapping[str, Any] | None = None,
        physics_maximum_voxels: int = 2000000,
    ) -> dict[str, Any]:
        """Enrich a typed Registry placement without changing membership or base geometry.

        Explicit alternate views recompute their own surface/operators, and a larger spectral
        budget can require an eigensolve; neither operation rewrites the historical base archive.

        Args:
            dataset: Native immutable dataset root containing index.jsonl and base assets.
            residue_ca_radius: Independent CA graph radius in Å; None preserves old assets.
            residue_ca_k_max: Directed CA neighbor budget, default 32.
            surface_environment: Generate masked protein-only free-space features, default false.
            environment_options: SurfaceEnvironment parameters; None uses its physical defaults.
            atomic_checkpoint: Optional exact reviewed SLAE base checkpoint; None adds no atom
                representations and never installs an extractor or downloads weights.
            atomic_device: Explicit logical extraction device, default cpu; CUDA requires grant.
            physics_options: Optional PhysicsConfig mapping; None preserves every inherited VF
                asset but computes none. Adds separate protein-only reference/query sidecars,
                never overwrites geometry, labels, SLAE or other pretrained representations.
            physics_maximum_voxels: Operational APBS allocation ceiling, default two million
                points; an oversized grid fails explicitly, never becomes a coarser target.
            dataset_name: Destination Registry family (existing or new).
            dataset_version: New immutable version, never overwrite the source version.
            atomic_context: Include context, default true.
            source_quality: Include observed metadata if exact original source remains available.
            extended_surface_atom: Include expanded CSR candidates, default true.
            extended_radius: Positive physical radius in Å, default 12.
            extended_k_max: Positive stored per-point cap, default 96; truncation audited.
            workers: Bounded native file-map concurrency, default 4; does not affect content.
            sequence_context: Attach physical residue/token correspondence, default false.
            atomic_sasa: Attach optional per-atom solvent exposure, default false.
            sasa_probe: Positive solvent probe radius, default 1.4 Å.
            sasa_samples: Deterministic samples per atom, default 960 (at least 24).
            residue_graph: Attach residue contacts/surface transfer, default false.
            residue_contact_radius: Positive residue contact radius, default 8 Å.
            residue_surface_radius: Positive point-to-residue radius, default 12 Å.
            surface_local_graph: Attach existing filtered coherence graph, default false.
            surface_graph_resolution: Positive coherence rule scale, default 1 Å.
            surface_view: Build one alternate surface and all matching operators, default false.
            view_id: Nonnegative deterministic view ID, default 1.
            view_resolution: Optional positive spacing in Å; None reuses base spacing.
            external_model: None (default), esm2, esmc or saprot; never inferred or downloaded.
            external_checkpoint: Explicit local typed checkpoint directory; required when enabled.
            external_identifier: Original model repository identity, required when enabled.
            external_revision: Immutable staged source revision, required when enabled.
            external_layer: Hidden-state layer index, default -1 (last layer).
            external_maximum_length: Positive disjoint chunk limit, default 1022 residues.
            external_device: Explicit extraction device, default cpu; excluded from cache identity.
            external_sequence_policy: fragments (default) or complete verified chains.
            structure_tokens: Generate official Foldseek 3Di independently, default false.
            static_embedding_table: Also export the extractor's input weights, default false.
            covalent_environment: Count neighbouring elements/orders, default false.
            spectral_modes: Optional fixed spectral extension/prefix budget, default None.

        Returns:
            Native DatasetVersion record. Member IDs, targets, partitions, metadata and all
            existing task/base assets are preserved, with optional structural assets added.

        Raises:
            ValueError: Invalid base or sidecar; already published destination; bad member index.
        """
        self.outputs.dataset_preflight(name=dataset_name, version=dataset_version)
        if structure_tokens or external_model == "saprot":
            StructureTokens(self)
        external = None
        if external_model is not None:
            if external_checkpoint is None:
                raise ValueError("external_model requires an explicit local external_checkpoint")
            external = ExternalProteinRepresentation(ProteinRepresentationExtractor(
                external_model, external_checkpoint, external_identifier, external_revision,
                external_layer, external_maximum_length, external_device), external_sequence_policy)
        atomic_external = (AtomicProteinExtractor(atomic_checkpoint, atomic_device, work=self)
                           if atomic_checkpoint is not None else None)
        if atomic_external is not None and workers != 1:
            raise ValueError("frozen atomic extraction requires workers=1 with native file resume")
        physics = (PhysicalFields(self, physics_options, physics_maximum_voxels)
                   if physics_options is not None else None)
        members = list(DatasetIndex(dataset / "index.jsonl"))
        sidecars = StructuralSidecars(atomic_context, source_quality, extended_surface_atom,
            extended_radius, extended_k_max, sequence_context=sequence_context,
            atomic_sasa=atomic_sasa, sasa_probe=sasa_probe, sasa_samples=sasa_samples,
            residue_graph=residue_graph, residue_contact_radius=residue_contact_radius,
            residue_surface_radius=residue_surface_radius, surface_local_graph=surface_local_graph,
            surface_graph_resolution=surface_graph_resolution, surface_view=surface_view,
            view_id=view_id, view_resolution=view_resolution, external=external,
            structure_tokens=structure_tokens, static_embedding_table=static_embedding_table,
            covalent_environment=covalent_environment, spectral_modes=spectral_modes,
            residue_ca_radius=residue_ca_radius, residue_ca_k_max=residue_ca_k_max,
            surface_environment=surface_environment, environment_options=environment_options,
            atomic_external=atomic_external)
        bases = {m.member_id: dataset / m.assets["universal_npz"].path for m in members}
        existing = {member.member_id: {name: dataset / asset.path for name, asset
                    in member.assets.items()} for member in members}
        source_metadata = {}
        for member in members:
            selection = VerifiedStructureSource.selection(member.metadata)
            if selection is not None:
                source_metadata[member.member_id] = selection
        report = Path(self.outputs.directory("structural-enrichment-report", role="report"))
        try:
            additions = sidecars.generate(self, bases, workers, existing=existing,
                source_metadata=source_metadata,
                source_hashes={member.member_id: str(member.assets["source_structure"].sha256)
                    for member in members if "source_structure" in member.assets
                    and member.assets["source_structure"].sha256 is not None})
        finally:
            ProteinArchive.write_report(report / "report.json", sidecars.report)

        if physics is not None:
            try:
                physical = physics.generate(bases, workers, existing=existing)
                for identifier, files in physical.items():
                    additions[identifier].update(files)
            finally:
                # Missing chemistry/field support is part of the scientific audit, not a reason
                # to remove a member or repartition the benchmark. Report the denominators so a
                # small or class-dependent VF subset cannot be mistaken for full coverage.

                partitions = {member.member_id: member.partitions for member in members}
                split_support: dict[str, Any] = {}
                for row in physics.report.get("records", []):
                    partition = partitions[row["id"]]
                    split = str(partition.get("split", "unspecified"))
                    support = split_support.setdefault(split, {"proteins": 0, "unavailable": 0,
                        "failed": 0, "potential": 0, "lj1": 0, "lj2": 0})
                    support["proteins"] += 1
                    support["unavailable"] += row.get("availability") == "unavailable"
                    support["failed"] += "error" in row
                    for i, field in enumerate(("potential", "lj1", "lj2")):
                        support[field] += row.get("valid_queries", [0, 0, 0])[i] > 0
                physics.report["split_support"] = split_support
                self.checkpoints.save_json("physics/last-report.json", physics.report)
                ProteinArchive.write_report(report / "physical-fields.json", physics.report)

        # Import original assets once through the public ownership API. Publication retains byte
        # identity, not necessarily a zero-copy physical placement. No Registry shim is introduced.

        source = self.outputs.artifact("source-dataset", dataset)
        published = []
        present: set[str] = set()
        for member in members:
            extra = additions[member.member_id]
            assets = {name: source / asset.path for name, asset in member.assets.items()}
            assets.update({name: self.outputs.artifact(f"{name}-{member.member_id}", path)
                           for name, path in extra.items()})
            # Each generated/reused addition has already passed its complete scientific validator.
            # Native artifact import preserves those exact bytes. Audit only untouched families
            # here instead of recomputing every expensive derivation a second time.

            StructuralSidecars.validate_assets(assets["universal_npz"],
                {name: path for name, path in assets.items() if name not in extra})
            present.update(name for name in assets if name in StructuralSidecars.FAMILIES)
            published.append({"id": member.member_id, "assets": assets,
                "targets": dict(member.targets), "partitions": dict(member.partitions),
                "metadata": dict(member.metadata), "display": dict(member.display)})
        return dict(self.outputs.dataset(name=dataset_name, version=dataset_version,
            members=published, metadata={"structural_schema": "3.0",
                "atomic_context_schema": "1.0" if "atomic_context" in present else None,
                "extended_surface_atom_schema": (
                    "1.0" if "extended_surface_atom" in present else None),
                "optional_universal_assets": sorted(present),
                "base_geometry_recomputed": False}))
