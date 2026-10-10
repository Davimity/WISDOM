"""Frozen zinc evidence contracts, with explicit negative support and contradiction quarantine."""

import json
import hashlib

from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence


# Negative annotations propagate to children, never from a child to its parent.
# These reviewed is-a ancestors therefore deny Zn binding; unrelated metals do not.

ZINC_NEGATIVE_TERMS = ("GO:0008270", "GO:0046872", "GO:0043169", "GO:0043167")
EXPERIMENTAL_CODES  = ("EXP", "IDA", "IPI", "IMP", "IGI", "IEP")


def load_evidence(path: Path) -> list[dict[str, Any]]:
    """Read typed JSONL candidates without inferring any negative from coordinate absence.

    Args:
        path: Frozen JSONL with identifier=PDB_CHAIN, sequence, label, assembly_id,
            protein_copy, origin, and label_evidence. Negatives require reference, scope,
            sequence_sha256 and either experimental_non_binding/assay or an experimental
            curated_not_annotation for Zn binding or a reviewed broader ion-binding term
            in ZINC_NEGATIVE_TERMS, with qualifier NOT.

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
                       and evidence.get("term") in ZINC_NEGATIVE_TERMS
                       and evidence.get("qualifier") == "NOT"
                       and evidence.get("evidence_code") in EXPERIMENTAL_CODES)
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


def annotate_sites(rows: Sequence[Mapping[str, Any]], path: Path) -> list[dict[str, Any]]:
    """Attach a reviewed frozen metal-database record to an exact WISDOM coordination site.

    Args:
        rows: Verified candidate records with identifier, structure_sha256 and site dictionaries.
        path: UTF-8 JSONL with identifier, site_id, structure_sha256, source, version,
            external_id and optional family/reference. The researcher must map external site
            identities to this design's assembly/copy; no live API or approximate match is used.

    Returns:
        Candidate copies carrying site-level external_metadata and exact input provenance.
        Coordination, positivity, global/local targets and leakage groups are never altered.

    Raises:
        ValueError: Duplicate/unknown sites, changed coordinates or missing source/release identity.
        OSError: The frozen annotation input cannot be read.
    """
    entries = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    annotations = {(entry["identifier"], entry["site_id"]): entry for entry in entries}
    if len(entries) != len(annotations):
        raise ValueError("duplicate external metal-site metadata identities")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    output = []
    matched = set()
    for original in rows:
        row = dict(original, sites=[dict(site) for site in original["sites"]])
        for site in row["sites"]:
            key = (row["identifier"], site["site_id"])
            entry = annotations.get(key)
            if entry is None:
                continue
            if (entry["structure_sha256"] != row["structure_sha256"]
                or not entry["source"] or not entry["version"] or not entry["external_id"]):
                raise ValueError("metal-site metadata requires exact coordinates "
                                 "and source release")
            site["external_metadata"] = {
                field: entry[field] for field in
                ("source", "version", "external_id", "family", "reference") if field in entry}
            site["external_metadata"]["input_sha256"] = digest
            matched.add(key)
        output.append(row)
    if matched != set(annotations):
        raise ValueError("external metal-site metadata refers to an unknown design site")
    return output
