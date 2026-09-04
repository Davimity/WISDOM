"""Validate the immutable coordinate snapshot published by Selection."""

import gzip
import json
import gemmi
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping, Sequence

from wisdom.preprocessing.dna.preprocessing.ProgressHeartbeat import ProgressHeartbeat


def validate_structure_snapshot(
    work                : lf.Work,
    rows                : Sequence[Mapping[str, Any]],
    structures          : Path,
    workers             : int,
    progress_log_seconds: float,
    verbose             : bool,
) -> Path:
    """Validate every selected structure against the portable Selection snapshot.

    The snapshot contains deterministic gzip archives plus ``index.json``. Exact hashes now verify
    that stored coordinates have not been corrupted; they do not compare the design with whatever
    a mutable RCSB endpoint happens to serve when preprocessing runs.

    Args:
        work: Active Work providing resumable parallel validation, progress, and logs.
        rows: Fixed selected records containing one uncompressed structure digest per PDB.
        structures: Selection snapshot directory with ``index.json`` and ``*.cif.gz`` files.
        workers: Concurrent local archive-validation threads.
        progress_log_seconds: Seconds between parent-process liveness messages.
        verbose: Emit one debug line for every validated PDB when true.

    Returns:
        The unchanged snapshot directory after every structure required by the selected subset has
        passed exact index, digest, gzip, and coordinate validation. Unselected design structures
        remain available but are not read.

    Raises:
        ValueError: If a selected PDB is absent or its index, compressed bytes, uncompressed bytes,
            or mmCIF contents disagree with the selected catalog.
        OSError: If snapshot files cannot be read.
    """
    index_path = structures / "index.json"
    if not index_path.is_file():
        raise ValueError("Selection structure snapshot is missing index.json")

    index   = json.loads(index_path.read_text(encoding="utf-8"))
    entries = index.get("structures")
    if index.get("schema_version") != "1.0" or not isinstance(entries, list):
        raise ValueError("Selection structure snapshot has an unsupported index schema")

    catalog_hashes: dict[str, set[str]] = {}
    for row in rows:
        pdb_id = str(row["pdb_id"]).lower()
        catalog_hashes.setdefault(pdb_id, set()).add(str(row["structure_sha256"]).lower())
    conflicts = sorted(pdb_id for pdb_id, values in catalog_hashes.items() if len(values) != 1)
    if conflicts:
        raise ValueError(f"manifests assign conflicting structure hashes to PDBs: {conflicts}")

    by_pdb: dict[str, Mapping[str, Any]] = {}
    for value in entries:
        if not isinstance(value, Mapping):
            raise ValueError("Selection structure index entries must be objects")
        pdb_id = str(value.get("pdb_id", "")).lower()
        if not pdb_id or pdb_id in by_pdb:
            raise ValueError(f"Selection structure index has duplicate or empty PDB ID {pdb_id!r}")
        by_pdb[pdb_id] = value

    selected_pdbs = set(catalog_hashes)
    snapshot_pdbs = set(by_pdb)
    missing       = sorted(selected_pdbs - snapshot_pdbs)
    if missing:
        raise ValueError(f"Selection structure snapshot is missing selected PDBs: {missing}")

    # A selective build intentionally consumes a train dilution and/or selected fixed splits from
    # the complete audited design. Its structure snapshot is therefore a valid superset. Ignoring
    # unused entries prevents needless hashing and parsing without weakening any published member.

    unused = snapshot_pdbs - selected_pdbs
    if unused:
        work.log(
            f"Selection snapshot contains {len(unused)} unused PDB archives; validating only the "
            f"{len(selected_pdbs)} required by this dataset view"
        )

    jobs: list[dict[str, Any]] = []
    for pdb_id, expected_values in sorted(catalog_hashes.items()):
        entry                 = by_pdb[pdb_id]
        expected_uncompressed = next(iter(expected_values))
        if str(entry.get("uncompressed_sha256", "")).lower() != expected_uncompressed:
            raise ValueError(
                f"Selection structure index and catalog disagree for {pdb_id.upper()}"
            )
        jobs.append(
            {
                "pdb_id":              pdb_id,
                "file":                str(entry.get("file", "")),
                "compressed_sha256":   str(entry.get("compressed_sha256", "")).lower(),
                "uncompressed_sha256": expected_uncompressed,
            }
        )

    validate = partial(
        _validate_structure,
        root    = structures,
        verbose = verbose,
    )
    work.log(
        f"Validating {len(jobs)} immutable Selection structure snapshots with {workers} workers"
    )
    with ProgressHeartbeat(work, "structure snapshot validation", progress_log_seconds):
        work.resume_map(
            jobs,
            validate,
            key      = "pdb_id",
            workers  = workers,
            executor = "thread",
            name     = "preprocessing-structures",
        )

    work.log(f"Structure snapshot validation complete: {len(jobs)} archives are ready")
    return structures


def _validate_structure(
    job    : Mapping[str, Any],
    root   : Path,
    verbose: bool,
) -> dict[str, str]:
    """Validate one deterministic gzip archive and its parsed coordinate model.

    Args:
        job: PDB ID, safe snapshot filename, and expected compressed/uncompressed SHA-256 values.
        root: Selection snapshot directory.
        verbose: Print one completed validation line when true.

    Returns:
        Small JSON-compatible result identifying the verified PDB and file.

    Raises:
        ValueError: If the path is unsafe, either digest differs, gzip is invalid, or Gemmi finds
            no valid coordinate model.
        OSError: If the archive cannot be read.
    """
    pdb_id   = str(job["pdb_id"]).lower()
    filename = str(job["file"])
    if not filename or Path(filename).name != filename:
        raise ValueError(f"unsafe structure snapshot filename for {pdb_id.upper()}: {filename!r}")
    path = root / filename
    if not path.is_file():
        raise ValueError(f"structure snapshot is missing {filename}")

    compressed = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            compressed.update(chunk)
    observed_compressed = compressed.hexdigest()
    expected_compressed = str(job["compressed_sha256"])
    if observed_compressed != expected_compressed:
        raise ValueError(
            f"compressed structure snapshot changed for {pdb_id.upper()}: "
            f"expected {expected_compressed}, observed {observed_compressed}"
        )

    uncompressed = hashlib.sha256()
    try:
        with gzip.open(path, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                uncompressed.update(chunk)
    except (OSError, EOFError) as error:
        raise ValueError(
            f"structure snapshot for {pdb_id.upper()} is not a readable gzip archive: {error}"
        ) from error
    observed_uncompressed = uncompressed.hexdigest()
    expected_uncompressed = str(job["uncompressed_sha256"])
    if observed_uncompressed != expected_uncompressed:
        raise ValueError(
            f"uncompressed structure snapshot changed for {pdb_id.upper()}: "
            f"expected {expected_uncompressed}, observed {observed_uncompressed}"
        )

    try:
        structure = gemmi.read_structure(str(path))
    except (OSError, RuntimeError, ValueError) as error:
        raise ValueError(
            f"structure snapshot for {pdb_id.upper()} is not valid PDBx/mmCIF: {error}"
        ) from error
    if not structure:
        raise ValueError(f"structure snapshot for {pdb_id.upper()} has no coordinate model")

    if verbose:
        print(f"[Preprocessing:structures] validated {pdb_id.upper()}", flush=True)
    return {"pdb_id": pdb_id, "file": filename}
