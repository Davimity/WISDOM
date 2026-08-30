"""Project DNA evidence onto the immutable universal surface points."""

import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping, Sequence

from wisdom.preprocessing.dna.DNAAnnotationSink import DNAAnnotationSink
from wisdom.preprocessing.dna.preprocessing.manifests import annotation_records
from wisdom.preprocessing.dna.DNAAnnotationTransform import DNAAnnotationTransform
from wisdom.preprocessing.dna.preprocessing.ProgressHeartbeat import ProgressHeartbeat


def generate_annotations(
    work                : lf.Work,
    rows                : Sequence[Mapping[str, Any]],
    geometry_report     : Path,
    structure_root      : Path,
    workers             : int,
    progress_log_seconds: float,
    positive_gap        : float,
    negative_gap        : float,
    sensitivity_gaps    : Sequence[float],
    verbose             : bool,
) -> tuple[Path, Path]:
    """Create one point-aligned DNA sidecar for every universal NPZ.

    Args:
        work: Active Work providing checkpoints, process maps, progress, and logs.
        rows: Self-contained split records with exact assembly/contact metadata.
        geometry_report: Identifier-to-universal-NPZ report from the prior phase.
        structure_root: Directory containing exact compressed Selection snapshots.
        workers: Spawned CPU annotation processes.
        progress_log_seconds: Seconds between parent-process liveness messages.
        positive_gap: Largest confidently contacting DNA-to-surface gap in ångströms.
        negative_gap: Smallest confidently non-contacting gap in ångströms.
        sensitivity_gaps: Additional positive cutoffs retained for evaluation.
        verbose: Print one worker line for every started and completed protein.

    Returns:
        Self-contained annotation root and its ordered JSON report.
    """
    root                    = work.checkpoints.path("annotation")
    output_root             = root / "annotations"
    report_path             = output_root / "annotation-report.json"
    resolved_structure_root = root / "resolved-structures"
    records                 = annotation_records(rows, geometry_report, structure_root)
    transform = DNAAnnotationTransform(
        positive_gap     = positive_gap,
        negative_gap     = negative_gap,
        sensitivity_gaps = tuple(sensitivity_gaps),
    )
    process = partial(
        _process_annotation,
        transform               = transform,
        output_root             = output_root,
        resolved_structure_root = resolved_structure_root,
        verbose                 = verbose,
    )
    by_identifier = {str(record["key"]): record for record in records}
    validate      = partial(
        _valid_annotation_result,
        records          = by_identifier,
        output_root      = output_root,
        positive_gap     = positive_gap,
        negative_gap     = negative_gap,
        sensitivity_gaps = tuple(sensitivity_gaps),
    )

    work.log(f"Projecting DNA targets for {len(records)} proteins with {workers} workers")
    with ProgressHeartbeat(work, "DNA annotation", progress_log_seconds):
        results = work.resume_map(
            records,
            process,
            key      = "key",
            workers  = workers,
            executor = "process",
            name     = "dna-annotations",
            validate = validate,
        )

    sink = DNAAnnotationSink()
    sink.records = {
        str(value["key"]): dict(value["value"])
        for value in results
    }
    sink.finalize(output_root, report_path)
    work.log(f"DNA annotation complete: {len(results)} validated sidecars are ready")
    return output_root, report_path


def _process_annotation(
    record                 : Mapping[str, Any],
    transform              : DNAAnnotationTransform,
    output_root            : Path,
    resolved_structure_root: Path,
    verbose                : bool,
) -> dict[str, Any]:
    """Run one DNA projection inside a spawned LambdaForge worker.

    Args:
        record: Joined universal-geometry and exact Selection metadata.
        transform: Configured DNA surface projection.
        output_root: Directory receiving sidecars.
        resolved_structure_root: Directory receiving verified uncompressed structures.
        verbose: Print detailed worker diagnostics when true.

    Returns:
        Compact JSON-compatible sidecar report.
    """
    key = str(record["key"])
    if verbose:
        print(f"[Preprocessing:annotation] starting {key}", flush=True)
    result = transform.process(record, output_root, resolved_structure_root)
    if verbose:
        print(f"[Preprocessing:annotation] completed {key}", flush=True)
    return result


def _valid_annotation_result(
    result          : Mapping[str, Any],
    records         : Mapping[str, Mapping[str, Any]],
    output_root     : Path,
    positive_gap    : float,
    negative_gap    : float,
    sensitivity_gaps: tuple[float, ...],
) -> bool:
    """Revalidate a restored annotation checkpoint and its base fingerprint.

    Args:
        result: JSON result restored by LambdaForge ``resume_map``.
        records: Current annotation inputs keyed by exact protein identifier.
        output_root: Directory containing sidecar NPZ files.
        positive_gap: Current positive target cutoff in ångströms.
        negative_gap: Current negative target cutoff in ångströms.
        sensitivity_gaps: Current evaluation cutoffs in ångströms.

    Returns:
        True only when the current base NPZ, source structure, thresholds, and sidecar agree.
    """
    record = records.get(str(result.get("key", "")))
    if record is None:
        return False
    sink = DNAAnnotationSink()
    return sink.resume(
        record,
        output_root,
        positive_gap,
        negative_gap,
        sensitivity_gaps,
    ) is not None
