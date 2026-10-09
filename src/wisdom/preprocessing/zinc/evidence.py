"""Frozen zinc evidence contracts, with explicit negative support and contradiction quarantine."""

import json

from typing import Any
from pathlib import Path


def load_evidence(path: Path) -> list[dict[str, Any]]:
    """Read typed JSONL candidates without inferring any negative from coordinate absence.

    Args:
        path: Frozen JSONL with identifier=PDB_CHAIN, sequence, label, assembly_id,
            protein_copy, origin, and label_evidence. Negatives require reference, scope,
            sequence_sha256 and either experimental_non_binding/assay or an experimental
            curated_not_annotation for GO:0008270 with qualifier NOT.

    Returns:
        Exact records with explicit PDB/chain identity. Conflicting duplicates are retained as
        quality-ineligible records for full-population leakage, never silently relabelled.

    Raises:
        ValueError: Non-binary labels, missing reliable negative support or inconsistent identity.
    """
    rows: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        label = int(row["label"])
        pdb, chain = row["identifier"].split("_", 1)
        if label not in (0, 1) or "_" in chain:
            raise ValueError("Zn candidates require one complete chain and a binary label")
        evidence = row["label_evidence"]
        if label == 0:
            explicit = evidence.get("kind") == "experimental_non_binding" and evidence.get("assay")
            curated = (evidence.get("kind") == "curated_not_annotation"
                       and evidence.get("term") == "GO:0008270"
                       and evidence.get("qualifier") == "NOT"
                       and evidence.get("evidence_code") in (
                           "EXP", "IDA", "IPI", "IMP", "IGI", "IEP"))
            if not (explicit or curated) or not evidence.get("reference") or (
                evidence.get("scope") != "zinc_binding" or not evidence.get("sequence_sha256")
            ):
                raise ValueError(f"{row['identifier']}: explicit sequence-bound negatives required")
        row.update(pdb_id=pdb.lower(), protein_chain=chain,
                   protein_copy=int(row["protein_copy"]), assembly_id=str(row["assembly_id"]))
        previous = rows.get(row["identifier"])
        if previous is not None and previous["sequence"] != row["sequence"]:
            raise ValueError("one Zn identity has conflicting sequences; revise frozen evidence")
        if previous is not None and previous != row:
            previous["evidence_conflict"] = True
            previous.setdefault("conflicting_evidence", []).append(evidence)
        else:
            rows[row["identifier"]] = row
    return list(rows.values())
