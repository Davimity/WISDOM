"""Generate universal geometry and Zn evaluation sidecars from one immutable design."""

import json
import numpy as np
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Sequence
from wisdom.preprocessing.zinc.ZincAnnotation import ZincAnnotation
from wisdom.preprocessing.zinc.ZincValidation import ZincValidation
from wisdom.preprocessing.dna.preprocessing.geometry import generate_geometry
from wisdom.preprocessing.dna.preprocessing.structures import validate_structure_snapshot


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
    ) -> dict[str, Any]:
        """Verify exact snapshot, resume universal geometry, annotate and atomically publish.

        Args:
            skip: Return without computation/publication when true, default false.
            design: Exact ZincSelection directory containing selection.jsonl and structures.
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

        Returns:
            Native DatasetVersion publication record; no new scientific membership is selected.

        Raises:
            ValueError: Missing design, changed snapshot or invalid annotation/geometry.
            RuntimeError: Scientific audit failure; successful managed checkpoints are retained.
        """
        if skip:
            self.log("Zn preprocessing skipped; no dataset published")
            return {"skipped": True}
        if design is None:
            raise ValueError("Zn preprocessing requires the exact ZincSelection design")
        self.outputs.dataset_preflight(name=dataset_name, version=dataset_version)

        # 1. Read fixed scientific membership, including optional saved train-only views.
        # A labelled TXT alone cannot reconstruct assembly/copy or coordinating-site evidence.

        rows = [json.loads(line) for line in (design / "selection.jsonl").read_text().splitlines()]
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
                    "split", "leakage_group", "global_phenotype", "interface_phenotype")},
                "metadata": {"task_specification": {"task_name": "zinc_binding",
                                                 "global_target_key": "zinc_binding",
                                                 "local_annotation_asset": "zinc_annotation"},
                          "zinc_evidence": row, "dilutions": views},
                "assets": {"universal_npz": base, "zinc_annotation": Path(sidecar),
                           "source_structure": snapshot / f"{row['pdb_id']}.cif.gz"},
            })
        verdict = ZincValidation().audit_members(members, root / "validation")
        if verdict["verdict"] != "PASS":
            raise RuntimeError("Zn scientific validation failed; inspect zinc-publication-audit")

        # Import external inputs and durable checkpoints through LF before publication. Each
        # shared geometry/snapshot directory is imported once, not once per protein.

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
        members[0]["assets"].update(
            selection_evidence=self.outputs.artifact("selection-evidence", design),
        )
        return dict(self.outputs.dataset(
            name=dataset_name, version=dataset_version, members=members,
            metadata={"task": "zinc_binding", "structural_schema": "3.0",
                      "annotation_schema": "1.0", "supervision": "protein-level-only",
                      "physiological_specificity_claimed": False},
            target_schema={"type": "object", "properties": {
                "zinc_binding": {"type": "integer", "enum": [0, 1]},
                "local_ground_truth": {"type": "boolean"}},
                "required": ["zinc_binding", "local_ground_truth"], "additionalProperties": False},
        ))
