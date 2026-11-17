"""Join frozen functional annotations without inventing task labels or leakage edges."""

import json
import hashlib

from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence


def annotate_functions(
    rows: Sequence[Mapping[str, Any]], path: Path,
) -> list[dict[str, Any]]:
    """Attach sequence-bound public functional metadata only for audit and diversity.

    Args:
        rows: Structurally verified candidates with identifier and exact sequence digest.
        path: Frozen UTF-8 JSONL entries with identifier, sequence_sha256, source, version and
            optional uniprot/pfam/interpro/cath/ec/family. Live service queries are not hidden here.

    Returns:
        Copied candidates with supported functional fields and exact input provenance. Missing
        annotations remain missing, not invented families. Labels, contacts and groups remain
        untouched.

    Raises:
        ValueError: Duplicate annotation identities or sequence/release mismatches.
        OSError: Frozen input cannot be read.
    """
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    annotations = {entry["identifier"]: entry for entry in entries}
    if len(annotations) != len(entries):
        raise ValueError("duplicate functional metadata identities")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output = []
    for original in rows:
        row   = dict(original)
        entry = annotations.get(row["identifier"])
        if entry is not None:
            if (entry["sequence_sha256"] != row["sequence_sha256"]
                or not entry["source"] or not entry["version"]):
                raise ValueError("functional metadata must identify exact sequence and release")
            row.update({field: entry[field] for field in
                ("uniprot", "pfam", "interpro", "cath", "ec", "family") if field in entry})
            row["functional_metadata_provenance"] = {
                "source": entry["source"], "version": entry["version"],
                "input_sha256": digest,
            }
        output.append(row)
    return output
