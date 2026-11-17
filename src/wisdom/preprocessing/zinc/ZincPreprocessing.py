"""Generate universal geometry and Zn evaluation sidecars from one immutable design."""

import json
import numpy as np
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.zinc.ZincAnnotation import ZincAnnotation
from wisdom.preprocessing.zinc.ZincValidation import ZincValidation
from wisdom.preprocessing.zinc.stages.legacy import prepare_legacy_design
from wisdom.preprocessing.common.ProgressHeartbeat import ProgressHeartbeat
from wisdom.preprocessing.common.clustering.geometry import generate_geometry
from wisdom.preprocessing.common.structure.assets.external.ProteinModelAssets import (
    ProteinModelAssets,
)
from wisdom.preprocessing.common.independence.snapshots import validate_structure_snapshot
from wisdom.preprocessing.common.structure.assets.external.AtomicProteinExtractor import (
    AtomicProteinExtractor,
)
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars
from wisdom.preprocessing.common.structure.assets.external.StructureTokens import StructureTokens
from wisdom.preprocessing.common.structure.assets.external.ExternalProteinRepresentation import (
    ExternalProteinRepresentation,
)
from wisdom.preprocessing.common.structure.assets.external.ProteinRepresentationExtractor import (
    ProteinRepresentationExtractor,
)


class ZincPreprocessing(lf.Work):
    """Reuse the universal NPZ algorithm without rediscovery, relabelling or resplitting."""

    def run(
        self, skip: bool = False, design: Path | None = None,
        dataset_name: str = "wisdom-zinc", dataset_version: str = "1",
        training_subset: str = "full", workers: int = 8,
        surface_resolution: float = 1.0, probe_radius: float = 1.4,
        atom_spatial_radius: float = 6.0, atom_spatial_k_max: int = 32,
        surface_atom_radius: float = 6.0, surface_atom_k_max: int = 32,
        diffusion_spectral_modes_max: int = 128, surface_neighbor_k_max: int = 24,
        curvature_scales: Sequence[float] = (2.5, 5.0), positive_gap: float = 1.4,
        negative_gap: float = 3.0, sensitivity_gaps: Sequence[float] = (1.0, 1.4, 2.0),
        progress_log_seconds: float = 120.0, verbose: bool = False,
        atomic_context: bool = False, source_quality: bool = True,
        extended_surface_atom: bool = False, extended_radius: float = 12.0,
        extended_k_max: int = 96,
        residue_graph: bool = False, residue_ca_radius: float | None = None,
        residue_ca_k_max: int = 32, surface_environment: bool = False,
        environment_options: Mapping[str, Any] | None = None,
        sequence_context: bool = False, covalent_environment: bool = False,
        atomic_sasa: bool = False, sasa_probe: float = 1.4, sasa_samples: int = 960,
        residue_contact_radius: float = 8.0, residue_surface_radius: float = 12.0,
        surface_local_graph: bool = False, surface_graph_resolution: float = 1.0,
        spectral_modes: int | None = None, surface_view: bool = False,
        view_id: int = 1, view_resolution: float | None = None,
        structure_tokens: bool = False, protein_models: Path | None = None,
        external_model: str | None = None, external_checkpoint: Path | None = None,
        external_identifier: str = "", external_revision: str = "", external_layer: int = -1,
        external_maximum_length: int = 1022, external_device: str = "cpu",
        external_sequence_policy: str = "fragments", static_embedding_table: bool = False,
        atomic_checkpoint: Path | None = None, atomic_device: str = "cpu",
        protein_model_kinds: Sequence[str] | None = None, protein_static_tables: bool = True,
        legacy_train: Path | None = None, legacy_validation: Path | None = None,
        legacy_test: Path | None = None, legacy_positives: Path | None = None,
        legacy_negatives: Path | None = None, legacy_all: Path | None = None,
    ) -> dict[str, Any]:
        """Verify exact snapshot, resume universal geometry, annotate and atomically publish.

        Args:
            skip: Return without computation/publication when true, default false.
            design: Exact ZincSelection directory containing selection.jsonl and structures.
                Mutually exclusive with historical TXT inputs.
            legacy_train: Historical training IDs; enables the explicit TFG reproduction protocol.
            legacy_validation: Historical validation IDs, required with legacy_train.
            legacy_test: Historical test IDs, required with legacy_train; never dropped.
            legacy_positives: Historical positive IDs, required with legacy_train.
            legacy_negatives: Historical negative IDs, required with legacy_train; not re-curated.
            legacy_all: Optional historical full inventory, checked against splits and labels.
            dataset_name: Registry family, default wisdom-zinc.
            dataset_version: New immutable version, default 1.
            training_subset: full or a saved replicate-XX/train-NN dilution; evaluation fixed.
            workers: Bounded geometry processes and annotation threads, default 8.
            surface_resolution: Surface spacing in ångströms, default 1.
            probe_radius: Solvent probe radius in ångströms, default 1.4.
            atom_spatial_radius: Spatial atom cutoff in ångströms, default 6.
            atom_spatial_k_max: Maximum persisted spatial atom neighbors, default 32.
            surface_atom_radius: Protein atom-to-point cutoff in ångströms, default 6.
            surface_atom_k_max: Maximum persisted atom candidates per point, default 32.
            diffusion_spectral_modes_max: Maximum stored intrinsic spectral modes, default 128.
            surface_neighbor_k_max: Maximum stored intrinsic point neighbors, default 24.
            curvature_scales: Positive local fit radii in surface-resolution units, default 2.5/5.
            positive_gap: Positive gap to coordinating protein atoms in ångströms, default 1.4.
            negative_gap: Negative gap cutoff in ångströms, default 3.
            sensitivity_gaps: Alternative local positive cutoffs in ångströms, default 1/1.4/2.
            progress_log_seconds: Parent liveness log interval in seconds, default 120.
            verbose: Enable per-protein diagnostic logs, default false.
            atomic_context: Attach optional common atom/residue context, default false.
            source_quality: Preserve observed source quality when available, default true.
            extended_surface_atom: Attach optional expanded CSR candidates, default false.
            extended_radius: Positive expanded candidate cutoff in Å, default 12.
            extended_k_max: Positive per-point expanded storage cap, default 96.
            residue_graph: Optional physical heavy-contact graph, default false.
            residue_ca_radius: Independent positive CA radius in Å; None retains old assets.
            residue_ca_k_max: Stored directed CA budget, default 32.
            surface_environment: Optional protein-only depth/occlusion/void asset, default false.
            environment_options: SurfaceEnvironment scientific parameters, None uses defaults.
            sequence_context: Physical residue/sequence correspondence, default false.
            covalent_environment: Fixed neighboring element/bond-order counts, default false.
            atomic_sasa: Protein-only per-atom solvent area, default false.
            sasa_probe: Positive solvent radius in Å, default 1.4.
            sasa_samples: Deterministic sphere quadrature size, at least 24, default 960.
            residue_contact_radius: Heavy-atom residue contact cutoff in Å, default 8.
            residue_surface_radius: Direct residue/surface candidate cutoff in Å, default 12.
            surface_local_graph: Common filtered surface topology, default false.
            surface_graph_resolution: Surface coherence scale in Å, default 1.
            spectral_modes: Optional fixed spectral extension budget, default None.
            surface_view: Separate deterministic surface/operators, default false.
            view_id: Nonnegative alternative sampling identifier, default 1.
            view_resolution: Alternative spacing in Å; None reuses the base spacing.
            structure_tokens: Official protein-only Foldseek 3Di asset, default false.
            protein_models: Typed reviewed model bundle, default None. Enables sequence mapping
                and selected model-qualified assets; 3Di is required only for selected SaProt.
            protein_model_kinds: esm2/esmc/saprot subset; None selects all three.
            protein_static_tables: Export selected input tables, default true for compatibility.
            external_model: Optional single esm2/esmc/saprot extractor, default None.
            external_checkpoint: Exact local checkpoint directory for that extractor, default None.
            external_identifier: Original checkpoint repository identifier, default empty.
            external_revision: Immutable upstream revision, default empty.
            external_layer: Hidden-state layer, default -1 (last).
            external_maximum_length: Disjoint inference chunk size in residues, default 1022.
            external_device: Explicit logical inference device, default cpu; no automatic GPU use.
            external_sequence_policy: fragments (default) or complete verified chains.
            static_embedding_table: Export single extractor input weights, default false.
            atomic_checkpoint: Optional reviewed SLAE base weights, default None. Extraction uses
                one native file worker after parallel structural generation, regardless of workers.
            atomic_device: Explicit logical SLAE device matching its isolated runtime, default cpu.

        Returns:
            Native DatasetVersion publication record; no new scientific membership is selected.

        Raises:
            ValueError: Missing design, changed snapshot or invalid annotation/geometry.
            RuntimeError: Scientific audit failure; successful managed checkpoints are retained.
        """
        if skip:
            self.log("Zn preprocessing skipped; no dataset published")
            return {"skipped": True}
        legacy_files = {"train": legacy_train, "validation": legacy_validation,
                        "test": legacy_test, "positives": legacy_positives,
                        "negatives": legacy_negatives}
        legacy = any(path is not None for path in (*legacy_files.values(), legacy_all))
        if legacy:
            if design is not None or any(path is None for path in legacy_files.values()):
                raise ValueError("Legacy Zn requires all five split/label TXT inputs and no design")
            if training_subset != "full":
                raise ValueError("Legacy TFG reproduction must preserve the full original train")
        elif design is None:
            raise ValueError("Zn preprocessing requires the exact ZincSelection design")
        self.outputs.dataset_preflight(name=dataset_name, version=dataset_version)

        # Resolve every requested tool/model before snapshot inspection or expensive geometry.
        # The same common producers used by DNA see only filtered protein atoms, never Zn sites.

        if protein_models is not None and external_model is not None:
            raise ValueError("protein_models cannot be combined with the single external_model")
        models = (ProteinModelAssets(protein_models, external_maximum_length, external_layer,
            external_device, kinds=protein_model_kinds, static_tables=protein_static_tables)
            if protein_models is not None else None)
        need_tokens = structure_tokens or external_model == "saprot" or (
            models is not None and models.requires_structure_tokens)
        if need_tokens:
            StructureTokens(self)
        if models is not None:
            models.preflight()
        external = None
        if external_model is not None:
            if external_checkpoint is None:
                raise ValueError("external_model requires an explicit local external_checkpoint")
            external = ExternalProteinRepresentation(ProteinRepresentationExtractor(
                external_model, external_checkpoint, external_identifier, external_revision,
                external_layer, external_maximum_length, external_device), external_sequence_policy)
        atomic = (AtomicProteinExtractor(atomic_checkpoint, atomic_device, work=self)
                  if atomic_checkpoint is not None else None)
        atomic_sidecars = StructuralSidecars(atomic_external=atomic) if atomic is not None else None
        sidecars = StructuralSidecars(
            atomic_context, source_quality, extended_surface_atom, extended_radius, extended_k_max,
            sequence_context=sequence_context or models is not None,
            covalent_environment=covalent_environment, atomic_sasa=atomic_sasa,
            sasa_probe=sasa_probe, sasa_samples=sasa_samples,
            residue_graph=residue_graph, residue_contact_radius=residue_contact_radius,
            residue_surface_radius=residue_surface_radius,
            residue_ca_radius=residue_ca_radius, residue_ca_k_max=residue_ca_k_max,
            surface_environment=surface_environment, environment_options=environment_options,
            surface_local_graph=surface_local_graph,
            surface_graph_resolution=surface_graph_resolution,
            spectral_modes=spectral_modes, surface_view=surface_view,
            view_id=view_id, view_resolution=view_resolution,
            structure_tokens=need_tokens, external=external,
            static_embedding_table=static_embedding_table,
        )
        required = sidecars.requested_families | (
            models.requested_families if models is not None else frozenset()) | (
            atomic_sidecars.requested_families if atomic_sidecars is not None else frozenset())
        self.log(f"Required Zn dataset assets: {', '.join(sorted(required))}")

        # The explicit historical route preserves labels and partitions rather than claiming
        # to repeat modern Selection. Only deposited bytes and optional local GT are recovered.

        if legacy:
            files = {name: path for name, path in legacy_files.items() if path is not None}
            if legacy_all is not None:
                files["all"] = legacy_all
            design = prepare_legacy_design(self, files, workers, progress_log_seconds)
        assert design is not None
        if (design / "legacy-inputs.json").is_file():
            legacy = True
            if training_subset != "full":
                raise ValueError("A frozen legacy design must preserve the full original train")

        # 1. Read fixed scientific membership, including optional saved train-only views.
        # A labelled TXT alone cannot reconstruct assembly/copy or coordinating-site evidence.

        rows = [json.loads(line) for line in (design / "selection.jsonl").read_text().splitlines()]
        for row in rows:
            if "local_phenotype" not in row and "interface_phenotype" in row:
                row["local_phenotype"] = row.pop("interface_phenotype")
        dilutions = json.loads((design / "dilutions.json").read_text())
        if training_subset != "full":
            replicate, subset = training_subset.split("/", 1)
            ids = set(dilutions["replicates"][replicate][subset]["identifiers"])
            rows = [row for row in rows if row["split"] != "train" or row["identifier"] in ids]
        snapshot = validate_structure_snapshot(self, rows, design / "structures", workers,
                                                progress_log_seconds, verbose)

        # 2. Reuse the identical protein-only universal geometry algorithm as the DNA workflow.
        # Per-protein validated checkpoints survive interruption; Zn never enters that archive.

        self.log(f"Generating or restoring universal geometry for {len(rows)} Zn proteins")
        processed, report = generate_geometry(
            self, rows, snapshot, workers, progress_log_seconds, surface_resolution, probe_radius,
            atom_spatial_radius, atom_spatial_k_max, surface_atom_radius, surface_atom_k_max,
            diffusion_spectral_modes_max, surface_neighbor_k_max, curvature_scales, verbose,
        )
        mapping = {row["identifier"]: row for row in json.loads(report.read_text())["records"]}
        annotation = ZincAnnotation(positive_gap, negative_gap, sensitivity_gaps)

        # 3. Add only task-specific evaluation sidecars. Their keys bind the exact universal NPZ
        # and fixed design. No target is appended to the geometry or explicit input features.

        def annotate(row: dict[str, Any]) -> str:
            """Restore a validated per-protein zinc sidecar through the native checkpoint API.

            Args:
                row: Exact selected assembly/copy and coordination evidence.

            Returns:
                Managed checkpoint path with point-count and base-hash verification.
            """
            base = processed / mapping[row["identifier"]]["output"]
            return str(self.checkpoints.file(
                f"zinc-annotation/{row['identifier']}.npz",
                build=partial(annotation.write, base, row=row),
                validate=partial(ZincAnnotation.validate, base=base),
            ))

        paths = self.resume_map(rows, annotate, key="identifier", workers=workers,
                                executor="thread", name="zinc-annotation")

        # Common optional context sees protein-only NPZs, never Zn coordinates or labels.

        bases = {row["identifier"]: processed / mapping[row["identifier"]]["output"]
                 for row in rows}
        sources = {row["identifier"]: {"source_structure": snapshot / f"{row['pdb_id']}.cif.gz"}
                   for row in rows}
        selections = {row["identifier"]: {key: row[key] for key in (
            "assembly_id", "protein_chain", "protein_copy") if key in row} for row in rows}
        structural_report = Path(self.outputs.directory(
            "structural-enrichment-report", role="report"))
        try:
            additions = sidecars.generate(self, bases, workers,
                existing=sources, source_metadata=selections)
        finally:
            (structural_report / "report.json").write_text(json.dumps(sidecars.report, indent=2))
        if models is not None:
            neural = models.generate(self, bases, additions)
            for identifier, assets in neural.items():
                additions[identifier].update(assets)

        # Frozen atom inference owns one isolated tool process at a time. Keep CPU geometry and
        # structural sidecars parallel, then use a separate native file-checkpoint pass for SLAE.
        # It receives only protein NPZ arrays; Zn evidence and local targets cannot reach it.

        if atomic_sidecars is not None:
            self.log(f"Generating or restoring SLAE atom/residue embeddings for {len(bases)} "
                     "Zn proteins with one granted-device extraction at a time")
            try:
                with ProgressHeartbeat(self, "Zn frozen atomic embeddings", progress_log_seconds):
                    neural = atomic_sidecars.generate(self, bases, 1)
            finally:
                (structural_report / "atomic-report.json").write_text(
                    json.dumps(atomic_sidecars.report, indent=2))
            for identifier, assets in neural.items():
                additions[identifier].update(assets)

        for identifier, assets in additions.items():
            missing = required - assets.keys()
            if missing:
                raise RuntimeError(f"member {identifier!r} lacks required structural assets: "
                                   f"{sorted(missing)}")

        # 4. Audit logical members before publication. Native LF publication then checks the
        # bytes and installs a permanent placement; no project registry/caching shim is used.

        root = Path(self.outputs.directory("zinc-publication-audit", role="report"))
        members = []
        for row, sidecar in zip(rows, paths, strict=True):
            base = processed / mapping[row["identifier"]]["output"]
            with np.load(sidecar, allow_pickle=False) as archive:
                local = bool(archive["local_gt_available"].item())
            views = [f"{replicate}/{name}" for replicate, subsets in dilutions["replicates"].items()
                     for name, subset in subsets.items()
                     if row["identifier"] in subset["identifiers"]]
            members.append({
                "id": row["identifier"], "targets": {"zinc_binding": int(row["label"]),
                                                     "local_ground_truth": local},
                "partitions": {key: str(row[key]) for key in (
                    "split", "leakage_group", "global_phenotype", "local_phenotype")},
                "metadata": {"task_specification": {"task_name": "zinc_binding",
                                                 "global_target_key": "zinc_binding",
                                                 "local_annotation_asset": "zinc_annotation"},
                          "zinc_evidence": row, "dilutions": views,
                          "required_structural_assets": sorted(required)},
                "assets": {"universal_npz": base, "zinc_annotation": Path(sidecar),
                           "source_structure": snapshot / f"{row['pdb_id']}.cif.gz",
                           **additions[row["identifier"]]},
            })
        members[0]["assets"]["selection_evidence"] = design
        self.log(f"Auditing {len(members)} complete Zn members before dataset publication")
        with ProgressHeartbeat(self, "Zn final scientific validation", progress_log_seconds):
            verdict = ZincValidation().audit_members(members, root / "validation")
        if verdict["verdict"] != "PASS":
            raise RuntimeError("Zn scientific validation failed; inspect zinc-publication-audit")

        # Import external inputs and durable checkpoints through LF before publication. Each
        # shared geometry/snapshot directory is imported once, not once per protein.

        self.log("Zn scientific audit passed; importing immutable assets for publication")
        owned_geometry = self.outputs.artifact("universal-archives", processed)
        owned_snapshot = self.outputs.artifact("structure-snapshot", snapshot)
        for member in members:
            assets = member["assets"]
            relative_base = assets["universal_npz"].relative_to(processed)
            relative_source = assets["source_structure"].relative_to(snapshot)
            assets["universal_npz"] = owned_geometry / relative_base
            assets["source_structure"] = owned_snapshot / relative_source
            assets["zinc_annotation"] = self.outputs.artifact(
                f"zinc-sidecar-{member['id']}", assets["zinc_annotation"],
            )
            for name in additions[member["id"]]:
                assets[name] = self.outputs.artifact(f"{name}-{member['id']}", assets[name])
        members[0]["assets"].update(
            selection_evidence=self.outputs.artifact("selection-evidence", design),
        )
        return dict(self.outputs.dataset(
            name=dataset_name, version=dataset_version, members=members,
            metadata={"task": "zinc_binding", "structural_schema": "3.0",
                      "annotation_schema": "2.0", "supervision": "protein-level-only",
                      "required_structural_assets": sorted(required),
                      "optional_universal_assets": sorted(required),
                      "label_protocol": "legacy_tfg" if legacy else "verified_design",
                      "homology_independence_verified": not legacy,
                      "physiological_specificity_claimed": False},
            target_schema={"type": "object", "properties": {
                "zinc_binding": {"type": "integer", "enum": [0, 1]},
                "local_ground_truth": {"type": "boolean"}},
                "required": ["zinc_binding", "local_ground_truth"], "additionalProperties": False},
        ))
