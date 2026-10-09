"""Revalidate zinc candidates and build full-population protein-only structural evidence."""

import json
import math
import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping, Sequence
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination
from wisdom.preprocessing.dna.selection.structures import (
    _global_features, _protein_coordinates, _rigid_transform, _sequence_features,
)


def analyse_structures(
    work: Any, rows: Sequence[Mapping[str, Any]], workers: int, cutoff: float,
    minimum_occupancy: float, minimum_donors: int, minimum_residues: int,
    maximum_resolution: float | None, requests_per_second: float,
) -> list[dict[str, Any]]:
    """Download once per deposition, then verify every selected assembly/copy.

    Args:
        work: Native Work cache, checkpoints and resume_map services.
        rows: Complete frozen positive/explicit-negative evidence population.
        workers: Concurrent I/O workers, not a scientific setting.
        cutoff: Maximum Zn-protein N/O/S center distance in ångströms.
        minimum_occupancy: Minimum occupancy for metal and protein donor atoms.
        minimum_donors: Minimum selected-copy N/O/S donors per accepted site.
        minimum_residues: Minimum selected-copy coordinating residues per accepted site.
        maximum_resolution: Worst accepted deposited resolution in ångströms; None disables.
        requests_per_second: Shared native public-service request ceiling.

    Returns:
        Analysed full RAW rows, including rejected contacts and contradictory evidence. All
        parseable candidates still enter homology grouping before canonical filtering.
    """
    return work.resume_map(rows, partial(
        _analyse, work=work, cutoff=cutoff, minimum_occupancy=minimum_occupancy,
        minimum_donors=minimum_donors, minimum_residues=minimum_residues,
        maximum_resolution=maximum_resolution, requests_per_second=requests_per_second,
    ), key=lambda row: str(row["identifier"]), workers=workers, executor="thread",
        name="zinc-structures")


def _analyse(
    raw: Mapping[str, Any], work: Any, cutoff: float, minimum_occupancy: float,
    minimum_donors: int, minimum_residues: int, maximum_resolution: float | None,
    requests_per_second: float,
) -> dict[str, Any]:
    """Verify one deposited-chain identity and retain explicit exclusion reasons.

    Args:
        raw: Frozen candidate including assembly, copy, label and sequence evidence.
        work: Native managed file services.
        cutoff: Coordination center distance cutoff in ångströms.
        minimum_occupancy: Deposited occupancy cutoff.
        minimum_donors: Selected-copy donor atom minimum.
        minimum_residues: Selected-copy donor residue minimum.
        maximum_resolution: Maximum resolution in ångströms or None.
        requests_per_second: Shared native download ceiling.

    Returns:
        JSON-compatible row with source digest, rigid transform, sites and physical descriptors.

    Raises:
        ValueError: Missing chain/copy, sequence mismatch or invalid coordinates.
    """
    pdb = str(raw["pdb_id"])
    source = work.cache.fetch(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
                              key=f"structures/{pdb}.cif", decompress="gzip",
                              retries=5, timeout=180, rate_limit=requests_per_second)
    structure = ProteinStructure(Path(source))
    if raw.get("structure_sha256") and raw["structure_sha256"] != structure.sha256():
        raise ValueError(f"{raw['identifier']}: deposition revision differs from frozen discovery")
    assembly = structure.assembly(str(raw["assembly_id"]))
    deposited, chain = assembly.protein_copy(str(raw["protein_chain"]), int(raw["protein_copy"]))
    sequence = structure.sequence(deposited)
    if sequence != str(raw["sequence"]).upper():
        raise ValueError(f"{raw['identifier']}: source sequence disagrees with frozen evidence")
    digest = hashlib.sha256(sequence.encode()).hexdigest()
    evidence = raw["label_evidence"]
    if int(raw["label"]) == 0 and evidence["sequence_sha256"] != digest:
        raise ValueError(f"{raw['identifier']}: negative evidence belongs to another sequence")
    sites = ZincCoordination(cutoff, minimum_occupancy, minimum_donors,
                             minimum_residues).analyse(assembly, str(raw["protein_chain"]),
                                                       int(raw["protein_copy"]))
    accepted = [site for site in sites if site["accepted"]]
    local_donors = [(site, atom) for site in accepted for atom in site["partners"]
                    if atom["chain_index"] == site["selected_chain_index"]]
    lengths = [float(np.linalg.norm(np.asarray(atom["position"]) - site["position"]))
               for site, atom in local_donors]
    reasons = []
    if raw.get("evidence_conflict"):
        reasons.append("contradictory_identity_evidence")
    if bool(accepted) != bool(int(raw["label"])):
        reasons.append("no_verified_site" if int(raw["label"]) else "negative_has_Zn_coordination")
    if int(raw["label"]) == 0 and any(site["selected_donor_count"] for site in sites):
        reasons.append("negative_has_ambiguous_Zn_contact")
    if (maximum_resolution is not None and structure.resolution is not None
        and structure.resolution > maximum_resolution):
        reasons.append("resolution_exceeds_maximum")
    protein = _protein_coordinates(chain, len(sequence))
    sequence_features = _sequence_features(sequence)
    rotation, translation = _rigid_transform(deposited, chain)
    foldseek = work.checkpoints.file(
        f"zinc-foldseek/{raw['identifier']}.cif",
        build=partial(assembly.write_protein_copy, chain_name=str(raw["protein_chain"]),
                      copy_index=int(raw["protein_copy"]), identifier=str(raw["identifier"])),
    )
    return {
        **dict(raw), **sequence_features, **_global_features(protein),
        "sequence_sha256": digest, "source_structure": str(source),
        "log_sequence_length": math.log(len(sequence)),
        "charge_density": sequence_features["net_charge_at_pH_7"] / len(sequence),
        "structure_sha256": structure.sha256(), "foldseek_structure": str(foldseek),
        "assembly_rotation": rotation.tolist(), "assembly_translation": translation.tolist(),
        "quality_eligible": not reasons, "quality_exclusion_reason": ";".join(reasons),
        "resolution": structure.resolution, "method": structure.experimental_method,
        "release_year": structure.release_year, "sites": sites, "zn_site_count": len(accepted),
        "zn_donor_count": sum(site["selected_donor_count"] for site in accepted),
        "zn_coordination_distance_mean": float(np.mean(lengths)) if lengths else 0.0,
        "zn_coordination_distance_std": float(np.std(lengths)) if lengths else 0.0,
        **{f"zn_{element}_fraction": sum(atom["element"] == element for _, atom in local_donors)
           / len(local_donors) if local_donors else 0.0 for element in ("N", "O", "S")},
        "zn_residue_count": len({r for site in accepted for r in site["selected_residues"]}),
        "zn_interchain_fraction": (sum(site["interchain"] for site in accepted) / len(accepted)
                                   if accepted else 0.0),
        "local_gt_expected": bool(accepted) or int(raw["label"]) == 0,
        "positive_confidence": (evidence.get("confidence", "structural_coordination_only")
                                if int(raw["label"]) else "not_applicable"),
        "negative_confidence": evidence["kind"] if not int(raw["label"]) else "not_applicable",
        "coordination_parameters": {"cutoff": cutoff, "minimum_occupancy": minimum_occupancy,
                                    "minimum_donors": minimum_donors,
                                    "minimum_residues": minimum_residues},
        "evidence_record_json": json.dumps(dict(raw), sort_keys=True),
    }
