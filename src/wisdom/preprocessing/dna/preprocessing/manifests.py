"""Join fixed dataset records to their generated universal NPZ files."""

import json

from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence


def annotation_records(
    rows          : Sequence[Mapping[str, Any]],
    report_path   : Path,
    structure_root: Path,
) -> list[dict[str, Any]]:
    """Join manifest metadata to exact universal NPZ outputs.

    Args:
        rows: Self-contained protein records loaded from the three manifests.
        report_path: Geometry report mapping identifiers to generated NPZ filenames.
        structure_root: Directory containing verified Selection ``.cif.gz`` snapshots.

    Returns:
        One annotation record per successful universal NPZ.

    Raises:
        ValueError: If geometry coverage differs from manifest membership.
    """
    report  = json.loads(report_path.read_text(encoding="utf-8"))
    outputs = {
        str(value["identifier"]): report_path.parent / "processed" / str(value["output"])
        for value in report["records"]
        if value["status"] in {"processed", "skipped"}
    }
    if set(outputs) != {str(row["identifier"]) for row in rows}:
        raise ValueError("geometry report and preprocessing manifests have different members")

    records: list[dict[str, Any]] = []
    for row in rows:
        identifier = str(row["identifier"])
        value       = dict(row)
        value["base_npz"]               = str(outputs[identifier].resolve())
        value["structure_archive_path"] = str(
            (structure_root / f"{str(row['pdb_id']).lower()}.cif.gz").resolve()
        )
        records.append({"key": identifier, "value": value})
    return records
