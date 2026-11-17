"""Preserve the TFG benchmark while reusing the current task-independent representation."""

import json
import hashlib

from typing import Any
from pathlib import Path
from functools import partial
from collections import Counter
from collections.abc import Mapping
from wisdom.preprocessing.zinc.stages.sources import freeze_source
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination
from wisdom.preprocessing.common.ProgressHeartbeat import ProgressHeartbeat
from wisdom.preprocessing.common.clustering.descriptors import rigid_transform
from wisdom.preprocessing.common.independence.snapshots import snapshot_structures


def read_membership(files: Mapping[str, Path]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Join historical split and label files without selecting, sampling or repairing members.

    Args:
        files: train, validation, test, positives and negatives TXT paths; optional all path.
            Each nonempty line contains one PDB_CHAIN ID with a complete chain name.

    Returns:
        Original split-ordered records with inherited binary labels, plus exact file hashes,
        counts and an explicit warning that original negative-label acquisition is not reviewed.

    Raises:
        ValueError: Duplicate IDs, overlapping labels/splits, inconsistent coverage or invalid IDs.
        OSError: An original input cannot be read. No missing or conflicting member is discarded.
    """
    identifiers = {}
    hashes = {}
    for name, path in files.items():
        values = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        if len(values) != len(set(values)):
            raise ValueError(f"Legacy {name} contains repeated identifiers")
        for identifier in values:
            pdb, chain = identifier.split("_", 1)
            if not pdb or not chain or "_" in chain:
                raise ValueError("Legacy Zn membership requires one complete PDB chain per line")
        identifiers[name] = values
        hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    positives = set(identifiers["positives"])
    negatives = set(identifiers["negatives"])
    if positives & negatives:
        raise ValueError("Legacy positive and negative labels overlap")
    splits = [set(identifiers[name]) for name in ("train", "validation", "test")]
    if any(splits[a] & splits[b] for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError("Legacy split identifiers overlap; refusing to change historical splits")
    population = set.union(*splits)
    if population != positives | negatives or (
            "all" in identifiers and set(identifiers["all"]) != population):
        raise ValueError("Legacy split, label and all.txt coverage disagree")
    rows = [{"identifier": identifier, "label": int(identifier in positives), "split": split}
            for split in ("train", "validation", "test") for identifier in identifiers[split]]
    report = {"protocol": "legacy_tfg", "membership_sha256": hashes,
              "members": rows, "member_count": len(rows),
              "split_counts": {split: dict(Counter(str(row["label"]) for row in rows
                  if row["split"] == split)) for split in ("train", "validation", "test")},
              "selection_performed": False, "homology_independence_verified": False,
              "experimental_negative_confirmation": False,
              "warning": "Labels and partitions are inherited from the TFG, not re-curated. "
                         "Current RCSB bytes do not establish identity to historical coordinates."}
    return rows, report


def prepare_legacy_design(
    work: Any, files: Mapping[str, Path], workers: int, progress_log_seconds: float,
    coordination_cutoff: float = 3.0, minimum_occupancy: float = 0.5,
    minimum_donors: int = 2, minimum_residues: int = 2,
) -> Path:
    """Freeze original membership and observed coordinates, without executing scientific selection.

    Args:
        work: Existing ZincPreprocessing Work; LF owns downloads, durable checkpoints and maps.
        files: Original split/label/all TXT paths accepted by read_membership.
        workers: Maximum concurrent native source downloads and independent record inspection.
        progress_log_seconds: Parent phase-liveness interval in seconds.
        coordination_cutoff: Zn-donor distance in Å used only for optional local reference recovery.
        minimum_occupancy: Atom/metal occupancy threshold in [0,1] for reference recovery.
        minimum_donors: Minimum selected-copy donor count for an accepted reference site.
        minimum_residues: Minimum selected-copy residue count for an accepted reference site.

    Returns:
        Portable legacy-design artifact with exact original files, source hashes, selection.jsonl
        and deterministic coordinate snapshot. Missing structures fail, never shrink membership.
        No homology clustering, balancing, split assignment or dilution is performed.

    Raises:
        ValueError: Historical file membership is inconsistent or an exact chain is unavailable.
        OSError: Native source acquisition or portable publication fails.
    """
    rows, report = read_membership(files)
    root = Path(work.outputs.directory("legacy-design", role="dataset-evidence"))
    inputs = root / "inputs"
    inputs.mkdir(parents=True, exist_ok=True)
    for name, path in files.items():
        (inputs / f"{name}.txt").write_bytes(path.read_bytes())
    (root / "legacy-inputs.json").write_text(json.dumps(report, indent=2))
    work.log(f"TFG reproduction: preserving all {len(rows)} proteins, labels and splits; "
             "no selection, homology independence claim or train dilution")

    # The first observed bytes become durable inputs. Compatible recovery reuses them even
    # if the reconstructible HTTP cache was collected; fresh Runs observe current RCSB bytes.

    pdbs = sorted({row["identifier"].split("_", 1)[0].lower() for row in rows})
    with ProgressHeartbeat(work, "Legacy Zn coordinate acquisition", progress_log_seconds):
        downloaded = work.resume_map(pdbs, partial(download_source, work=work),
            key=lambda pdb: pdb, workers=workers, executor="thread", name="legacy-zinc-downloads")
    sources = {entry["pdb_id"]: entry for entry in downloaded}
    jobs = [{**row, **sources[row["identifier"].split("_", 1)[0].lower()]}
            for row in rows]
    membership_digest = hashlib.sha256(json.dumps(report["membership_sha256"],
                                                 sort_keys=True).encode()).hexdigest()
    with ProgressHeartbeat(work, "Legacy Zn reference recovery", progress_log_seconds):
        records = work.resume_map(jobs, partial(recover_record,
            membership_digest=membership_digest, coordination_cutoff=coordination_cutoff,
            minimum_occupancy=minimum_occupancy, minimum_donors=minimum_donors,
            minimum_residues=minimum_residues), key=lambda row: row["identifier"],
            workers=workers, executor="thread", name="legacy-zinc-references")
    snapshot_structures(root / "structures", records)
    portable = [{key: value for key, value in row.items() if key != "source_structure"}
                for row in records]
    (root / "selection.jsonl").write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                                                for row in portable))
    (root / "dilutions.json").write_text(json.dumps({"replicates": {}}))
    report["reference_disagreements"] = [row["identifier"] for row in portable
        if (not row["label"] and any(site["selected_donor_count"] for site in row["sites"]))
        or (row["label"] and not any(site["accepted"] for site in row["sites"]))]
    (root / "legacy-inputs.json").write_text(json.dumps(report, indent=2))
    return root


def download_source(pdb: str, work: Any) -> dict[str, Any]:
    """Acquire one exact deposited source through LF's managed retry/checkpoint services.

    Args:
        pdb: Lowercase RCSB deposition ID from historical membership.
        work: Native Work owning rate limits and durable acquisition checkpoints.

    Returns:
        PDB identity and spawn-safe native source checkpoint. No protein is dropped on error.

    Raises:
        OSError: Retrieval fails after native retries; completed checkpoints remain reusable.
    """
    source = freeze_source(work, f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
                           f"legacy-zinc/{pdb}.cif", 2.0, decompress="gzip")
    return {"pdb_id": pdb, "source_structure": source}


def recover_record(
    row: Mapping[str, Any], membership_digest: str, coordination_cutoff: float,
    minimum_occupancy: float, minimum_donors: int, minimum_residues: int,
) -> dict[str, Any]:
    """Recover local evaluation evidence while treating the historical global label as fixed.

    Args:
        row: Inherited identifier/label/split and exact LF source path with lowercase pdb_id.
        membership_digest: SHA-256 of the original input-file digest mapping.
        coordination_cutoff: Maximum donor-Zn center distance in Å.
        minimum_occupancy: Minimum metal/donor occupancy in [0,1].
        minimum_donors: Minimum donor count for an accepted coordination reference.
        minimum_residues: Minimum distinct coordinating residue count.

    Returns:
        Existing ZnPreprocessing-compatible record. The first declared assembly containing the
        exact chain and its first copy are chosen deterministically, never by label/site score.
        Positives without sites and negatives with contacts retain their global label; their
        local target will be unavailable. Group/phenotype placeholders explicitly mean unchecked.

    Raises:
        ValueError: The requested chain has no complete sequence or declared assembly copy.
        OSError: Gemmi cannot read the frozen source. No replacement chain is guessed.
    """
    structure = ProteinStructure(Path(row["source_structure"]))
    chain_name = str(row["identifier"]).split("_", 1)[1]
    for descriptor in sorted(structure.structure.assemblies, key=lambda item: item.name):
        assembly = structure.assembly(str(descriptor.name))
        if any(chain.name == chain_name for chain in assembly.protein_chains()):
            break
    else:
        raise ValueError(f"{row['identifier']}: no declared assembly contains the historical chain")
    deposited, chain = assembly.protein_copy(chain_name, 1)
    sequence = structure.sequence(deposited)
    # A zero occupancy is a quality warning, not an absent stored coordinate in the historical
    # benchmark. Fit its geometric pose without treating those atoms as occupied Zn donors.

    try:
        rotation, translation = rigid_transform(deposited, chain, include_unoccupied=True)
    except ValueError as error:
        raise ValueError(f"{row['identifier']}: cannot align historical assembly copy: "
                         f"{error}") from error
    heavy_atoms = [atom for residue in deposited.get_polymer().first_conformer()
                   for atom in residue.first_conformer() if atom.element.atomic_number > 1]
    quality = {"occupied_heavy_atoms": sum(atom.occ > 0 for atom in heavy_atoms),
               "zero_occupancy_heavy_atoms": sum(atom.occ == 0 for atom in heavy_atoms)}
    parameters: dict[str, Any] = {
        "cutoff": coordination_cutoff, "minimum_occupancy": minimum_occupancy,
        "minimum_donors": minimum_donors, "minimum_residues": minimum_residues}
    sites = ZincCoordination(**parameters).analyse(assembly, chain_name, 1)
    return {**dict(row), "source_structure": str(row["source_structure"]),
        "sequence": sequence, "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
        "assembly_id": str(descriptor.name), "protein_chain": chain_name, "protein_copy": 1,
        "assembly_rotation": rotation.tolist(), "assembly_translation": translation.tolist(),
        "structure_sha256": structure.sha256(), "sites": sites,
        "coordination_parameters": parameters, "origin": "legacy_tfg",
        "legacy_coordinate_quality": quality,
        "leakage_group": f"legacy_unchecked:{row['identifier']}",
        "global_phenotype": "NOT_COMPUTED", "local_phenotype": "NOT_COMPUTED",
        "label_evidence": {"kind": "legacy_tfg_manifest", "reference": "inputs/",
            "membership_sha256": membership_digest, "scope": "zinc_binding",
            "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
            "experimental_confirmation": False},
    }
