"""Generate universal protein geometry with per-protein scientific resume."""

import os
import time
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping, Sequence

from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource
from wisdom.preprocessing.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.ProteinPreprocessor import ProteinPreprocessor
from wisdom.preprocessing.dna.preprocessing.ProgressHeartbeat import ProgressHeartbeat


def generate_geometry(
    work                : lf.Work,
    rows                : Sequence[Mapping[str, Any]],
    structure_root      : Path,
    workers             : int,
    progress_log_seconds: float,
    surface_resolution  : float,
    probe_radius        : float,
    atom_spatial_radius          : float,
    atom_spatial_k_max           : int,
    surface_atom_radius          : float,
    surface_atom_k_max           : int,
    diffusion_spectral_modes_max : int,
    surface_neighbor_k_max       : int,
    curvature_scales    : Sequence[float],
    verbose             : bool,
) -> tuple[Path, Path]:
    """Create one validated universal NPZ for every selected protein.

    Args:
        work: Active Work providing checkpoints, resumable process maps, progress, and logs.
        rows: Fixed manifest records whose identifiers select exact PDB chains.
        structure_root: Directory containing all checksum-verified source structures.
        workers: Spawned CPU processes used for independent proteins.
        progress_log_seconds: Seconds between parent-process liveness messages.
        surface_resolution: Target solvent-surface point spacing in ångströms.
        probe_radius: Solvent-probe radius in ångströms.
        atom_spatial_radius: Spatial atom-neighbour cutoff in ångströms.
        atom_spatial_k_max: Maximum ranked spatial candidates persisted per atom.
        surface_atom_radius: Surface-to-atom cutoff in ångströms.
        surface_atom_k_max: Maximum nearest atoms persisted per surface point.
        diffusion_spectral_modes_max: Maximum low-frequency surface modes persisted.
        surface_neighbor_k_max: Maximum nearest surface points used by intrinsic operators.
        curvature_scales: Curvature fit radii in surface-resolution units.
        verbose: Print one worker line for every started and completed protein.

    Returns:
        Checkpoint-owned NPZ directory and its ordered geometry report.
    """
    identifiers = [str(row["identifier"]) for row in rows]
    manifest    = work.checkpoints.file(
        "geometry/protein-identifiers.txt",
        build=partial(_write_text, content="".join(f"{value}\n" for value in identifiers)),
    )
    root            = work.checkpoints.path("geometry")
    output_root     = root / "processed"
    report_path     = root / "preprocessing-report.json"
    records         = list(ProteinSource().records(Path(manifest)))
    estimated_atoms = {
        str(row["identifier"]): int(float(row.get("heavy_atom_count") or 0))
        for row in rows
    }

    # When Selection provides atom counts, launch the largest proteins first. Process-pool workers
    # still pull jobs dynamically, but this longest-processing-time order prevents one giant
    # protein from becoming an avoidable serial tail after every small protein has finished.

    records.sort(
        key=lambda record: (
            -estimated_atoms.get(str(record["identifier"]), 0),
            str(record["key"]),
        )
    )
    config = PreprocessConfig(
        atom_spatial_radius          = atom_spatial_radius,
        atom_spatial_k_max           = atom_spatial_k_max,
        surface_resolution           = surface_resolution,
        probe_radius                 = probe_radius,
        surface_atom_radius          = surface_atom_radius,
        surface_atom_k_max           = surface_atom_k_max,
        diffusion_spectral_modes_max = diffusion_spectral_modes_max,
        surface_neighbor_k_max       = surface_neighbor_k_max,
        curvature_scales             = tuple(curvature_scales),
    )
    pipeline = ProteinPreprocessor(config)

    # Spawned workers inherit these limits before importing numerical kernels. This prevents each
    # protein process from creating another BLAS/OpenMP pool and oversubscribing the reserved CPUs.

    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[variable] = "1"

    scheduling = (
        "largest estimated atom counts first"
        if any(estimated_atoms.values())
        else "stable identifier order"
    )
    work.log(
        f"Generating universal geometry for {len(records)} proteins with {workers} workers; "
        f"scheduling {scheduling}"
    )
    process = partial(
        _process_geometry,
        pipeline       = pipeline,
        manifest       = Path(manifest),
        structure_root = structure_root,
        output_root    = output_root,
        verbose        = verbose,
    )
    validate = partial(
        _valid_geometry_result,
        root   = root / "processed",
        config = config,
    )
    with ProgressHeartbeat(work, "protein geometry", progress_log_seconds):
        results = work.resume_map(
            records,
            process,
            key      = "key",
            workers  = workers,
            executor = "process",
            name     = "protein-geometry",
            validate = validate,
        )

    sink         = ProteinSink()
    sink.records = {
        str(value["key"]): dict(value["value"])
        for value in results
    }
    sink.finalize(Path(manifest), output_root, report_path)

    # A failed record is deliberately not a valid resume result. Successful NPZs remain validated
    # checkpoints, while a compatible retry recomputes only these reported identifiers.

    failures = [
        dict(value["value"])
        for value in results
        if value["value"]["status"] == "failed"
    ]
    for failure in failures[:20]:
        work.log(
            f"Geometry failure for {failure['identifier']}: "
            f"{failure['error_type']}: {failure['error']}"
        )
    if len(failures) > 20:
        work.log(f"Geometry has {len(failures) - 20} additional failed proteins in its report")
    if failures:
        raise RuntimeError(
            f"geometry failed for {len(failures)} of {len(results)} proteins; "
            "successful NPZ checkpoints were retained and failed identifiers are listed in "
            "preprocessing-report.json"
        )

    statuses = [str(value["value"]["status"]) for value in results]
    work.log(
        f"Geometry complete: {statuses.count('processed')} generated and "
        f"{statuses.count('skipped')} restored after complete NPZ validation"
    )
    return output_root, report_path


def _process_geometry(
    record        : Mapping[str, Any],
    pipeline      : ProteinPreprocessor,
    manifest      : Path,
    structure_root: Path,
    output_root   : Path,
    verbose       : bool,
) -> dict[str, Any]:
    """Run one geometry transform inside a spawned LambdaForge worker.

    Args:
        record: Stable protein record created from the identifier manifest.
        pipeline: Configured structural transform.
        manifest: Identifier manifest used to resolve relative paths.
        structure_root: Directory containing verified source structures.
        output_root: Directory receiving universal NPZ files.
        verbose: Print detailed worker diagnostics when true.

    Returns:
        Compact JSON-compatible processed, restored, or failed result. Failed results preserve the
        identifier and error for the parent report but deliberately fail checkpoint validation so
        a compatible retry recomputes them.
    """
    key     = str(record["key"])
    started = time.perf_counter()
    if verbose:
        print(f"[Preprocessing:geometry] starting {key}", flush=True)

    try:
        result = pipeline.process(record, manifest, structure_root, output_root)
    except Exception as error:
        if verbose:
            print(
                f"[Preprocessing:geometry] {key}: {type(error).__name__}: {error}",
                flush=True,
            )
        return {
            "key": key,
            "value": {
                "identifier": key,
                "status":     "failed",
                "output":     None,
                "error_type": type(error).__name__,
                "error":      str(error),
                "seconds":    time.perf_counter() - started,
                "warnings":   [],
            },
        }

    if verbose:
        print(
            f"[Preprocessing:geometry] {key}: {result['value']['status']}",
            flush=True,
        )
    return result


def _valid_geometry_result(
    result: Mapping[str, Any],
    root  : Path,
    config: PreprocessConfig,
) -> bool:
    """Revalidate a restored map checkpoint against its exact NPZ bytes.

    Args:
        result: JSON result restored by LambdaForge ``resume_map``.
        root: Checkpoint-owned universal NPZ directory.
        config: Current scientific configuration.

    Returns:
        True only when source hash, config hash, schema, and numerical arrays still agree.
    """
    value = result.get("value")
    if not isinstance(value, Mapping):
        return False
    output = value.get("output")
    if not isinstance(output, str):
        return False
    path     = root / output
    archive  = ProteinArchive(config)
    metadata = archive.read_metadata(path)
    if metadata is None:
        return False
    source_hash = str(metadata.get("source_hash", ""))
    return bool(source_hash) and archive.can_resume(path, source_hash)


def _write_text(path: Path, content: str) -> None:
    """Build one LambdaForge-managed text checkpoint.

    Args:
        path: Temporary path supplied by ``checkpoints.file``.
        content: Complete UTF-8 manifest payload.
    """
    path.write_text(content, encoding="utf-8")
