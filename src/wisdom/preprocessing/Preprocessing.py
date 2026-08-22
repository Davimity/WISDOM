"""One function-first LambdaForge Work for heavy WISDOM-DNA preprocessing."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import lambdaforge as lf
from lambdaforge.data import DatasetIndex
from lambdaforge.preprocessing import PreprocessingTask
from lambdaforge.tasks import TaskContext

from wisdom.preprocessing.dna.DNAAnnotationSink import DNAAnnotationSink
from wisdom.preprocessing.dna.DNAAnnotationSource import DNAAnnotationSource
from wisdom.preprocessing.dna.DNAAnnotationTransform import DNAAnnotationTransform
from wisdom.preprocessing.dna.DNASelectionAudit import DNASelectionAudit
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessPipeline import PreprocessPipeline
from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource


def preprocess_dna(
    selection             : Path,
    dataset_name          : str = "wisdom-dna",
    dataset_version       : str = "2",
    workers               : int = 36,
    surface_resolution    : float = 1.0,
    probe_radius          : float = 1.4,
    atom_radius           : float = 6.0,
    atom_surface_radius   : float = 6.0,
    curvature_scales      : Sequence[float] = (2.5, 5.0),
    positive_gap          : float = 1.4,
    negative_gap          : float = 3.0,
    sensitivity_gaps      : Sequence[float] = (1.0, 1.4, 2.0),
) -> dict[str, Any]:
    """Build structural arrays, project DNA annotations, and publish one managed dataset.

    ``selection`` is the complete directory fetched from the preceding selection Work. Geometry
    preprocesses its union exactly once, including local-evaluation reserves. Annotation consumes
    the exact geometry report and its already downloaded coordinate cache, so it performs no second
    RCSB download. LambdaForge's record pipeline owns process workers, checkpoints, resume, logs,
    and failures inside both stages; LambdaForge 0.11's runtime API owns the final immutable,
    streaming DatasetVersion publication.

    The published DatasetIndex is the model-ingestion contract. Each member contains explicit split
    and tier partitions, global DNA targets, local-ground-truth availability, structural metadata,
    and checksummed universal-NPZ, annotation-sidecar, and source-structure assets. Dilution
    membership is stored as member metadata so model runs can select a view without duplicating any
    heavy array.

    Args:
        selection: Fetched selection artifact directory containing catalog and membership files.
        dataset_name: Non-empty immutable DatasetRegistry name.
        dataset_version: Non-empty immutable version label; changed content needs a new value.
        workers: Positive spawned-process count for both CPU-bound stages.
        surface_resolution: Target surface point spacing in ångströms.
        probe_radius: Solvent probe radius in ångströms.
        atom_radius: Sparse atom-graph spatial cutoff in ångströms.
        atom_surface_radius: Sparse surface-to-atom communication cutoff in ångströms.
        curvature_scales: Strictly increasing positive geodesic fit radii in ångströms.
        positive_gap: Largest DNA-to-surface gap considered confidently positive in ångströms.
        negative_gap: Smallest DNA-to-surface gap considered confidently negative in ångströms.
        sensitivity_gaps: Positive evaluation-only alternative interface cutoffs in ångströms.

    Returns:
        LambdaForge dataset registry record augmented with preprocessing counts and artifact names.

    Raises:
        ValueError: If the selection contract or a scientific/operational parameter is invalid.
        RuntimeError: If any selected member fails geometry, annotation, validation, or publication.
        OSError: If an input, checkpoint, archive, or atomic output cannot be read or written.
    """
    selection = selection.resolve()
    required  = ("catalog.csv", "identifiers.json", "proteins.txt")
    missing   = [name for name in required if not (selection / name).is_file()]
    if missing:
        raise ValueError(f"selection artifact is incomplete; missing {missing}")
    if not dataset_name.strip() or not dataset_version.strip() or workers < 1:
        raise ValueError("dataset identity must be non-empty and workers must be positive")

    # Recompute every selection invariant from source tables before any expensive surface work.
    quality = DNASelectionAudit().audit(selection, publish=False)
    if quality["status"] == "FAIL":
        raise ValueError("selection failed leakage, balance, or membership quality controls")

    runtime = lf.current()

    # Stage one runs every selected protein through the deterministic universal geometry pipeline.
    geometry_root = runtime.run_dir / "geometry"
    geometry_context = TaskContext(
        name=f"{runtime.name}-geometry",
        run_dir=geometry_root,
        source_dir=selection,
        attempt_id=runtime.attempt_id,
        config_fingerprint=f"{runtime.config_fingerprint}:geometry-v1",
        resume=True,
        inputs=(
            {
                "name": "protein_identifiers",
                "path": str(selection / "proteins.txt"),
                "resolved_path": str(selection / "proteins.txt"),
            },
        ),
        outputs={
            "downloads": "raw",
            "processed": "processed",
            "report": "preprocessing-report.json",
        },
    )
    geometry_task = PreprocessingTask(
        source=ProteinSource(input_name="protein_identifiers"),
        transforms=(
            PreprocessPipeline(
                config=PreprocessConfig(
                    atom_radius=atom_radius,
                    surface_resolution=surface_resolution,
                    probe_radius=probe_radius,
                    atom_surface_radius=atom_surface_radius,
                    curvature_scales=tuple(curvature_scales),
                ),
                identifier_input="protein_identifiers",
                download_output="downloads",
                download=True,
            ),
        ),
        sink=ProteinSink(
            identifier_input="protein_identifiers",
            dataset_output="processed",
            report_output="report",
            download_output="downloads",
        ),
        workers=workers,
        workload="cpu",
        on_error="fail",
        checkpoint_interval=1,
        progress_interval_seconds=10.0,
    )
    geometry_task.run(geometry_context)

    # Stage two joins labels to the exact geometry and reuses its coordinate bytes for local GT.
    annotation_root = runtime.run_dir / "annotation"
    annotation_context = TaskContext(
        name=f"{runtime.name}-annotation",
        run_dir=annotation_root,
        source_dir=selection,
        attempt_id=runtime.attempt_id,
        config_fingerprint=f"{runtime.config_fingerprint}:annotation-v1",
        resume=True,
        inputs=(
            {"name": "selection", "path": str(selection), "resolved_path": str(selection)},
            {
                "name": "dataset_catalog",
                "path": str(selection / "catalog.csv"),
                "resolved_path": str(selection / "catalog.csv"),
            },
            {
                "name": "base_preprocessing_report",
                "path": str(geometry_root / "preprocessing-report.json"),
                "resolved_path": str(geometry_root / "preprocessing-report.json"),
            },
            {
                "name": "structure_cache",
                "path": str(geometry_root / "raw"),
                "resolved_path": str(geometry_root / "raw"),
            },
        ),
        outputs={
            "annotations": "annotations",
            "annotation-report": "annotations/annotation-report.json",
            "resolved-structures": "resolved-structures",
        },
    )
    annotation_task = PreprocessingTask(
        source=DNAAnnotationSource(
            catalog_input="dataset_catalog",
            base_report_input="base_preprocessing_report",
            structure_cache_input="structure_cache",
        ),
        transforms=(
            DNAAnnotationTransform(
                positive_gap=positive_gap,
                negative_gap=negative_gap,
                sensitivity_gaps=tuple(sensitivity_gaps),
                structure_output="resolved-structures",
            ),
        ),
        sink=DNAAnnotationSink(
            annotation_output="annotations",
            report_output="annotation-report",
            selection_input="selection",
        ),
        workers=workers,
        workload="cpu",
        on_error="fail",
        checkpoint_interval=1,
        progress_interval_seconds=10.0,
    )
    annotation_task.run(annotation_context)

    final_root = annotation_root / "annotations"
    report     = json.loads((final_root / "annotation-report.json").read_text(encoding="utf-8"))
    members    = tuple(_published_members(final_root, selection))
    target_schema = {
        "type": "object",
        "properties": {
            "dna_binding": {"type": "integer", "enum": [0, 1]},
            "local_ground_truth": {"type": "boolean"},
        },
        "required": ["dna_binding", "local_ground_truth"],
        "additionalProperties": False,
    }
    record = lf.publish_dataset(
        dataset_name,
        dataset_version,
        members,
        metadata={
            "description": "Balanced DNA-binding proteins with universal WISDOM surfaces",
            "structural_schema": "2.1",
            "annotation_schema": "1.1",
            "selection_schema": "1.0",
            "supervision": "protein-level-only",
        },
        target_schema=target_schema,
    )

    lf.artifact("preprocessing-report", geometry_root / "preprocessing-report.json", role="report")
    lf.artifact("annotation-report", final_root / "annotation-report.json", role="report")
    lf.metric("published_members", len(members))
    lf.metric("local_gt_available", int(report["local_gt_available_count"]))

    return {
        **record,
        "published_members": len(members),
        "local_gt_available": report["local_gt_available_count"],
        "preprocessing_report": "preprocessing-report",
        "annotation_report": "annotation-report",
    }


def _published_members(final_root: Path, selection: Path) -> Iterable[Mapping[str, Any]]:
    """Translate the validated intermediate index into LambdaForge 0.11 member mappings.

    Args:
        final_root: Run-owned annotation root containing the validated intermediate index/assets.
        selection: Selection artifact containing deterministic dilution membership.

    Yields:
        Streaming mappings accepted by :func:`lambdaforge.publish_dataset`.

    Raises:
        ValueError: If dilution identifiers are malformed or reference an unknown member.
        OSError: If an index or selection file cannot be read.
    """
    dilution_members: dict[str, set[str]] = {}
    subset_root = selection / "subsets"
    if subset_root.is_dir():
        for path in sorted(subset_root.glob("*/identifiers.json")):
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows    = payload.get("records") if isinstance(payload, Mapping) else None
            if not isinstance(rows, list):
                raise ValueError(f"dilution {path.parent.name} has no records list")
            dilution_members[path.parent.name] = {
                str(row["identifier"])
                for row in rows
                if isinstance(row, Mapping) and row.get("identifier")
            }

    known = {member.member_id for member in DatasetIndex(final_root / "members.jsonl")}
    unknown = {
        identifier
        for identifiers in dilution_members.values()
        for identifier in identifiers
        if identifier not in known
    }
    if unknown:
        raise ValueError(f"dilutions reference unknown dataset members: {sorted(unknown)[:5]}")

    for member in DatasetIndex(final_root / "members.jsonl"):
        metadata = dict(member.metadata)
        metadata["dilutions"] = sorted(
            name
            for name, identifiers in dilution_members.items()
            if member.member_id in identifiers
        )
        yield {
            "id": member.member_id,
            "partitions": dict(member.partitions),
            "targets": dict(member.targets),
            "metadata": metadata,
            "display": dict(member.display),
            "assets": {
                name: final_root / asset.path
                for name, asset in member.assets.items()
            },
        }
