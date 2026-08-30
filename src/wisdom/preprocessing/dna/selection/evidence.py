"""Load the immutable JSONL evidence used by DNA benchmark selection."""

import json
import hashlib

from typing import Any
from pathlib import Path

AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYOU")


def load_evidence(work: Any, raw_path: Path, verbose: bool) -> list[dict[str, Any]]:
    """Read and normalize one explicit protein-evidence object per JSONL line.

    Args:
        work: Active LambdaForge Work used only for researcher-facing logs.
        raw_path: UTF-8 JSONL produced by the independent public-evidence collection step.
        verbose: Log every record when true; otherwise report every 500 physical lines.

    Returns:
        Identifier-sorted dictionaries retaining all provenance fields and normalized core values.

    Raises:
        ValueError: If a core field is missing, malformed, contradictory, or duplicated.
        OSError: If the evidence file cannot be read.
    """
    work.log(f"Reading frozen evidence from {raw_path}")
    rows: list[dict[str, Any]] = []

    with raw_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            raw        = json.loads(line)
            identifier = str(raw["identifier"])
            sequence   = str(raw["sequence"]).upper()
            label      = int(raw["label"])
            pdb_id, protein_chain = identifier.split("_", 1)

            if label not in {0, 1}:
                raise ValueError(f"line {line_number} has a non-binary label")
            if not sequence or not set(sequence).issubset(AMINO_ACIDS):
                raise ValueError(f"line {line_number} has an unsupported protein sequence")
            if int(raw["protein_copy"]) < 1:
                raise ValueError(f"line {line_number} has protein_copy < 1")
            if raw.get("schema_version") not in {None, "1.0"}:
                raise ValueError(f"line {line_number} uses an unsupported evidence schema")

            row = dict(raw)
            row.update(
                {
                    "label":           label,
                    "origin":          str(raw["origin"]),
                    "pdb_id":          pdb_id.upper(),
                    "source":          str(raw["source"]),
                    "sequence":        sequence,
                    "assembly_id":     str(raw["assembly_id"]),
                    "header_flags":    sorted(
                        str(value) for value in raw.get("header_flags", [])
                    ),
                    "identifier":      identifier,
                    "protein_copy":    int(raw["protein_copy"]),
                    "protein_chain":   protein_chain,
                    "base_identifier": identifier,
                    "label_evidence":  str(raw["label_evidence"]),
                    "sequence_sha256": hashlib.sha256(
                        sequence.encode("ascii")
                    ).hexdigest(),
                }
            )
            rows.append(row)

            if verbose:
                work.log(f"Evidence {line_number}: {identifier}")
            elif line_number % 500 == 0:
                work.log(f"Read {line_number} evidence lines")

    identifiers = [str(row["identifier"]) for row in rows]
    if not rows or len(identifiers) != len(set(identifiers)):
        raise ValueError("frozen evidence must contain unique non-empty identifiers")

    rows.sort(key=lambda row: str(row["identifier"]))
    positives = sum(int(row["label"]) == 1 for row in rows)
    work.log(
        f"Evidence ready: {len(rows)} proteins, {positives} positive and "
        f"{len(rows) - positives} negative"
    )
    return rows
