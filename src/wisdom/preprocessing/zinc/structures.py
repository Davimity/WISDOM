"""Revalidate zinc candidates and build full-population protein-only structural evidence."""

import json
import math
import gemmi
import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.descriptors import (
    global_features, protein_coordinates, rigid_transform, sequence_features,
)
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination
from wisdom.preprocessing.common.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.SurfaceBuilder import SurfaceBuilder
from wisdom.preprocessing.common.structure.StructureSource import StructureSource
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig


def analyse_structures(
    work: Any, rows: Sequence[Mapping[str, Any]], workers: int, cutoff: float,
    minimum_occupancy: float, minimum_donors: int, minimum_residues: int,
    maximum_resolution: float | None, requests_per_second: float,
    surface_resolution: float = 1.0, probe_radius: float = 1.4, positive_gap: float = 1.4,
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
        surface_resolution: Early surface sampling spacing in ångströms, default 1.
        probe_radius: Solvent expansion radius in ångströms, default 1.4.
        positive_gap: Maximum coordinating-atom gap supporting local GT, default 1.4 Å.

    Returns:
        Analysed full RAW rows, including rejected contacts and contradictory evidence. All
        parseable candidates still enter homology grouping before canonical filtering.
    """
    return work.resume_map(rows, partial(
        _analyse, work=work, cutoff=cutoff, minimum_occupancy=minimum_occupancy,
        minimum_donors=minimum_donors, minimum_residues=minimum_residues,
        maximum_resolution=maximum_resolution, requests_per_second=requests_per_second,
        surface_resolution=surface_resolution, probe_radius=probe_radius, positive_gap=positive_gap,
    ), key=lambda row: str(row["identifier"]), workers=workers, executor="thread",
        name="zinc-structures")


def _analyse(
    raw: Mapping[str, Any], work: Any, cutoff: float, minimum_occupancy: float,
    minimum_donors: int, minimum_residues: int, maximum_resolution: float | None,
    requests_per_second: float,
    surface_resolution: float, probe_radius: float, positive_gap: float,
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
        surface_resolution: Early surface spacing in ångströms.
        probe_radius: Early solvent expansion radius in ångströms.
        positive_gap: Early positive surface threshold in ångströms.

    Returns:
        JSON-compatible row with source digest, rigid transform, sites and physical descriptors.

    Raises:
        ValueError: Missing chain/copy, sequence mismatch or invalid coordinates.
    """
    pdb = str(raw["pdb_id"])
    rate = work.cache.rate_limit("files.rcsb.org", requests_per_second=requests_per_second)
    source = work.cache.fetch(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
                              key=f"structures/{pdb}.cif", decompress="gzip",
                              retries=5, timeout=180, rate_limit=rate)
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
    protein = protein_coordinates(chain, len(sequence))
    rotation, translation = rigid_transform(deposited, chain)
    # Local evaluability is audited before balancing/splits, without fitting curvature or
    # spectral operators. Buried positive sites remain valid protein-level evidence.

    exposed = False
    if int(raw["label"]) == 1 and accepted:
        points = _surface_points(Path(source), raw, structure.sha256(), surface_resolution,
                                 probe_radius, rotation, translation)
        for site in accepted:
            local = [atom for atom in site["partners"]
                     if atom["chain_index"] == site["selected_chain_index"]]
            gaps = ZincCoordination.surface_gaps(points,
                np.asarray([a["position"] for a in local]),
                np.asarray([a["vdw_radius"] for a in local]))
            support = int((gaps <= positive_gap).sum())
            site["surface_support"] = {
                "positive_point_count": support, "point_count": len(points),
                "classification": "surface_evaluable" if support else "buried_global_only",
                "resolution_angstrom": surface_resolution, "probe_radius_angstrom": probe_radius,
                "positive_gap_angstrom": positive_gap,
            }
            exposed |= support > 0
    sequence_descriptors = sequence_features(sequence)
    foldseek = work.checkpoints.file(
        f"zinc-foldseek/{raw['identifier']}.cif",
        build=partial(assembly.write_protein_copy, chain_name=str(raw["protein_chain"]),
                      copy_index=int(raw["protein_copy"]), identifier=str(raw["identifier"])),
    )
    return {
        **dict(raw), **sequence_descriptors, **global_features(protein),
        "sequence_sha256": digest, "source_structure": str(source),
        "log_sequence_length": math.log(len(str(raw["sequence"]))),
        "charge_density": sequence_descriptors["net_charge_at_pH_7"] / len(sequence),
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
        "local_gt_expected": exposed or int(raw["label"]) == 0,
        "surface_evaluability": "surface_evaluable" if exposed else (
            "explicit_negative" if int(raw["label"]) == 0 else "buried_global_only"),
        "positive_confidence": (evidence.get("confidence", "structural_coordination_only")
                                if int(raw["label"]) else "not_applicable"),
        "negative_confidence": evidence["kind"] if not int(raw["label"]) else "not_applicable",
        "coordination_parameters": {"cutoff": cutoff, "minimum_occupancy": minimum_occupancy,
                                    "minimum_donors": minimum_donors,
                                    "minimum_residues": minimum_residues},
        "evidence_record_json": json.dumps(dict(raw), sort_keys=True),
    }


def _surface_points(
    path: Path, row: Mapping[str, Any], digest: str, resolution: float, probe_radius: float,
    rotation: np.ndarray, translation: np.ndarray,
) -> np.ndarray:
    """Reuse preprocessing's filtered, centered sampling frame before an assembly transform.

    Sampling a rotated assembly directly changes the voxel grid and alternate-conformer choices.
    Instead use the universal reader and its float32 atom arrays, sample its centered deposition,
    cast exactly as the published surface does, then restore and transform the points. No graphs,
    curvature, labels or spectral operators are computed.

    Args:
        path: Exact cached deposited mmCIF bytes.
        row: Candidate identifier, PDB ID and complete selected chain name.
        digest: SHA-256 of the exact deposited bytes.
        resolution: Surface spacing in ångströms, equal to intended preprocessing settings.
        probe_radius: Solvent expansion in ångströms, equal to intended preprocessing settings.
        rotation: Rigid deposition-to-assembly matrix [3,3].
        translation: Deposition-to-assembly translation [3], ångströms.

    Returns:
        Float64 [M,3] surface positions in assembly coordinates. With the same preprocessing
        settings these are the points that final annotation evaluates, not a rotated resampling.

    Raises:
        ValueError: The selected chain is invalid or no protein surface can be sampled.
    """
    source = StructureSource(str(row["identifier"]), str(row["pdb_id"]),
                             (str(row["protein_chain"]),), path, digest, "mmcif", False)
    protein, provenance = ProteinReader(PreprocessConfig()).read(source)
    atoms = [atom for chain in protein.chains for residue in chain.residues
             for atom in residue.atoms]
    positions = np.asarray([atom.position for atom in atoms], dtype=np.float32)
    radii = np.asarray([gemmi.Element(atom.atomic_number).vdw_r for atom in atoms],
                       dtype=np.float32)
    points, _ = SurfaceBuilder(resolution, probe_radius).sample(positions, radii)
    restored = points.astype(np.float32).astype(np.float64) + provenance.coordinate_origin
    return restored @ rotation.T + translation
