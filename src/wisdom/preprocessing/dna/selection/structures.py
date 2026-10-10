"""Download, verify, snapshot, and describe structures referenced by frozen evidence."""

import numpy as np

from typing import Any
from pathlib import Path
from functools import partial
from scipy.spatial import cKDTree
from collections import defaultdict
from lambdaforge.work import RateLimit
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.descriptors import (
    HYDROPHOBIC, POLAR, POSITIVE, NEGATIVE, AROMATIC,
    protein_coordinates, sequence_features, global_features, rigid_transform,
    valid_structure, valid_foldseek_structure,
)
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.utils.structure.BiologicalAssembly import BiologicalAssembly


def analyse_structures(
    work                     : Any,
    rows                     : Sequence[Mapping[str, Any]],
    workers                  : int,
    requests_per_second      : float,
    retries                  : int,
    maximum_resolution       : float | None,
    interface_region_distance: float,
    verbose                  : bool,
) -> list[dict[str, Any]]:
    """Analyse every evidence row while downloading each PDB entry only once.

    Args:
        work: Active LambdaForge Work providing cache, resume-map, and logs.
        rows: Normalized frozen evidence rows.
        workers: Concurrent I/O-heavy unique-PDB jobs.
        requests_per_second: Aggregate RCSB request-start ceiling.
        retries: Additional attempts after one failed download.
        maximum_resolution: Largest canonical resolution in ångströms, or none.
        interface_region_distance: Radius joining contacted residues into patches in ångströms.
        verbose: Log each unique PDB when true.

    Returns:
        Identifier-sorted rows extended with verified assembly, contact, and physical descriptors.

    Raises:
        ValueError: If sequence, assembly, copy, coordinates, or label evidence disagrees.
        RuntimeError: If RCSB retrieval cannot produce a parseable structure.
    """
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["pdb_id"]).upper()].append(dict(row))
    jobs: list[dict[str, Any]] = []
    for pdb_id, members in sorted(grouped.items()):
        ordered = sorted(members, key=lambda row: str(row["identifier"]))
        jobs.append({"pdb_id": pdb_id, "rows": ordered})

    work.log(
        f"Analysing {len(rows)} proteins from {len(jobs)} unique PDB entries with "
        f"{workers} workers; lf overview and the Research Console show exact aggregate progress"
    )
    rate_limit = work.cache.rate_limit(
        "rcsb-dna-selection",
        requests_per_second=requests_per_second,
    )
    results = work.resume_map(
        jobs,
        partial(
            _analyse_pdb,
            work,
            retries=retries,
            rate_limit=rate_limit,
            maximum_resolution=maximum_resolution,
            interface_region_distance=interface_region_distance,
            verbose=verbose,
        ),
        key      = "pdb_id",
        workers  = workers,
        executor = "thread",
        name     = "dna-selection-structures",
    )
    analysed = sorted(
        (dict(row) for result in results for row in result["rows"]),
        key=lambda row: str(row["identifier"]),
    )
    if len(analysed) != len(rows):
        raise RuntimeError("structural analysis did not preserve every evidence row")
    work.log(f"Structural analysis complete: {len(analysed)} proteins")
    return analysed



def _analyse_pdb(
    work                     : Any,
    job                      : Mapping[str, Any],
    *,
    retries                  : int,
    rate_limit               : RateLimit,
    maximum_resolution       : float | None,
    interface_region_distance: float,
    verbose                  : bool,
) -> dict[str, Any]:
    """Download and analyse all evidence members belonging to one PDB deposition.

    Args:
        work: Active LambdaForge Work.
        job: Unique PDB ID and all evidence rows that refer to it.
        retries: Additional RCSB download attempts.
        rate_limit: Shared LambdaForge request limiter.
        maximum_resolution: Canonical resolution ceiling in ångströms, or none.
        interface_region_distance: Contact-patch radius in ångströms.
        verbose: Log this PDB when true.

    Returns:
        JSON-compatible PDB result containing every analysed member.
    """
    pdb_id = str(job["pdb_id"])
    if verbose:
        work.log(f"Analysing PDB {pdb_id} ({len(job['rows'])} evidence members)")

    # LambdaForge owns transfer retries, rate limiting, atomic cache publication, and dependency
    # tracking. Decompression makes the molecular SHA-256 independent from gzip metadata.

    url = f"https://files.rcsb.org/download/{pdb_id}.cif.gz"
    structure_file = work.cache.fetch(
        url,
        key        = f"structures/{pdb_id.lower()}.cif",
        retries    = retries,
        timeout    = 180.0,
        decompress = "gzip",
        validate   = valid_structure,
        rate_limit = rate_limit,
    )
    protein_structure = ProteinStructure(Path(structure_file))
    metadata          = {
        "resolution":          protein_structure.resolution,
        "release_year":        protein_structure.release_year,
        "experimental_method": protein_structure.experimental_method,
    }
    structure_hash    = protein_structure.sha256()

    # Biological-assembly generation copies the complete coordinate model and is substantially
    # more expensive than selecting one chain. Several evidence rows commonly share the same PDB
    # and assembly, so each distinct assembly is generated once and reused for every member.

    assemblies: dict[str, BiologicalAssembly] = {}
    for raw in job["rows"]:
        assembly_id = str(raw["assembly_id"])
        if assembly_id in assemblies:
            continue

        assemblies[assembly_id] = protein_structure.assembly(assembly_id)

    analysed: list[dict[str, Any]] = []
    for raw in job["rows"]:
        assembled = assemblies[str(raw["assembly_id"])]

        row = _analyse_member(
            protein_structure,
            assembled,
            raw,
            metadata,
            maximum_resolution,
            interface_region_distance,
        )

        # Foldseek consumes one assembly copy per file. The managed cache makes this deterministic
        # product reusable and records it as a resume-map dependency.

        foldseek_file = work.cache.file(
            f"foldseek/{raw['identifier']}.cif",
            build=lambda target, member=raw, assembly=assembled: assembly.write_protein_copy(
                target,
                str(member["protein_chain"]),
                int(member["protein_copy"]),
                str(member["identifier"]),
            ),
            validate=valid_foldseek_structure,
        )
        row.update(
            {
                "structure_url":      url,
                "structure_sha256":   structure_hash,
                "source_structure":   structure_file,
                "foldseek_structure": foldseek_file,
            }
        )
        analysed.append(row)
    return {"pdb_id": pdb_id, "rows": analysed}


def _analyse_member(
    structure                : ProteinStructure,
    assembled                : BiologicalAssembly,
    raw                      : Mapping[str, Any],
    metadata                 : Mapping[str, Any],
    maximum_resolution       : float | None,
    interface_region_distance: float,
) -> dict[str, Any]:
    """Verify one protein assembly copy and calculate design descriptors.

    Args:
        structure: Parsed structure and shared biological-assembly operations.
        assembled: Reused generated biological assembly selected by the evidence row.
        raw: Frozen evidence row.
        metadata: Experimental method, resolution, and release year.
        maximum_resolution: Canonical resolution ceiling in ångströms, or none.
        interface_region_distance: Contact-patch radius in ångströms.

    Returns:
        Portable row containing provenance, contacts, geometry, and quality status.

    Raises:
        ValueError: If immutable evidence contradicts the selected structure or assembly.
    """
    base_chain, protein_copy = assembled.protein_copy(
        str(raw["protein_chain"]),
        int(raw["protein_copy"]),
    )

    sequence = structure.sequence(base_chain)
    if sequence != str(raw["sequence"]):
        raise ValueError(f"{raw['identifier']} sequence disagrees with the mmCIF entity")

    protein                              = protein_coordinates(protein_copy, len(sequence))
    dna_chains                           = assembled.dna_chains()
    dna_positions, dna_radii, dna_owners = assembled.dna_atoms()
    dna                                  = {
        "radii":     dna_radii,
        "owners":    dna_owners,
        "positions": dna_positions,
    }
    contacts = _contacts(protein, dna)
    label    = int(raw["label"])
    if label == 1 and not contacts["binding_residues"]:
        raise ValueError(f"positive {raw['identifier']} has no direct assembly DNA contact")
    if label == 0 and contacts["binding_residues"]:
        raise ValueError(f"negative {raw['identifier']} has a direct assembly DNA contact")

    rotation, translation = rigid_transform(base_chain, protein_copy)

    observed   = int(protein["observed_residue_count"])
    coverage   = observed / len(sequence)
    resolution = metadata.get("resolution")
    eligible   = (
        maximum_resolution is None
        or resolution is None
        or float(resolution) <= maximum_resolution
    )
    return {
        **dict(raw),
        **dict(metadata),
        **sequence_features(sequence),
        **global_features(protein),
        **_interface_features(protein, contacts, interface_region_distance),

        "coordinate_coverage":        coverage,
        "missing_residue_fraction": 1.0 - coverage,
        "observed_residue_count":     observed,

        "assembly_dna_chain_count":               len(dna_chains),
        "assembly_protein_chain_count":           len(assembled.protein_chains()),
        "assembly_dna_chain_count_for_interface": (
            len(dna_chains) if label == 1 else None
        ),

        "dna_chains":              sorted({chain.name for chain in dna_chains}),
        "binding_residue_indices": sorted(contacts["binding_residues"]),

        "local_gt_method":      "dna_distance" if label == 1 else "global_negative",
        "local_gt_expected":    True,
        "assembly_rotation":    rotation.tolist(),
        "assembly_translation": translation.tolist(),

        "contact_definition":        "distance < protein_vdw + DNA_vdw + 0.5_angstrom",
        "interface_region_distance": interface_region_distance,

        "quality_eligible":         eligible,
        "quality_exclusion_reason": "" if eligible else "resolution_exceeds_maximum",
    }



def _contacts(protein: Mapping[str, Any], dna: Mapping[str, Any]) -> dict[str, Any]:
    """Find direct heavy-atom contacts without a dense protein-by-DNA distance matrix.

    Args:
        protein: Protein positions, van der Waals radii, and residue owners.
        dna: DNA positions, radii, and chain owners.

    Returns:
        Pair count and sets of contacting protein atoms/residues and DNA chain instances.
    """
    protein_xyz = np.asarray(protein["atom_positions"], dtype=np.float64)
    dna_xyz     = np.asarray(dna["positions"], dtype=np.float64)
    if not len(protein_xyz) or not len(dna_xyz):
        return {
            "pair_count":           0,
            "contacting_atoms":     set(),
            "binding_residues":     set(),
            "contacted_dna_chains": set(),
        }
    protein_radii = np.asarray(protein["atom_radii"], dtype=np.float64)
    dna_radii     = np.asarray(dna["radii"], dtype=np.float64)
    neighbours = cKDTree(dna_xyz).query_ball_point(
        protein_xyz,
        float(protein_radii.max() + dna_radii.max() + 0.5),
    )
    pair_count = 0
    contacting_atoms: set[int] = set()
    binding_residues: set[int] = set()
    contacted_chains: set[int] = set()
    for atom_index, candidates in enumerate(neighbours):
        indices    = np.asarray(candidates, dtype=np.int64)
        if not len(indices):
            continue
        delta      = dna_xyz[indices] - protein_xyz[atom_index]
        cutoffs    = protein_radii[atom_index] + dna_radii[indices] + 0.5
        contacting = indices[np.einsum("ij,ij->i", delta, delta) < cutoffs * cutoffs]
        if len(contacting):
            pair_count += len(contacting)
            contacting_atoms.add(atom_index)
            binding_residues.add(int(protein["atom_owners"][atom_index]))
            contacted_chains.update(int(dna["owners"][index]) for index in contacting)
    return {
        "pair_count":           pair_count,
        "contacting_atoms":     contacting_atoms,
        "binding_residues":     binding_residues,
        "contacted_dna_chains": contacted_chains,
    }




def _interface_features(
    protein       : Mapping[str, Any],
    contacts      : Mapping[str, Any],
    region_distance: float,
) -> dict[str, Any]:
    """Describe the geometry and chemistry of one positive DNA-contact interface.

    Args:
        protein: Residue representatives and letters.
        contacts: Direct protein-DNA contact result.
        region_distance: Radius joining contacted residues into regions in ångströms.

    Returns:
        Interface descriptors, or explicit unavailable values for a negative protein.
    """
    binding = set(int(value) for value in contacts["binding_residues"])
    fields = (
        "binding_residue_count",
        "binding_residue_fraction",
        "contacting_atom_count",
        "contact_pair_count",
        "contact_density",
        "contacted_dna_chain_count",
        "interface_region_count",
        "largest_interface_region_fraction",
        "interface_radius_of_gyration",
        "interface_radius_normalized",
        "interface_principal_spread_1",
        "interface_principal_spread_2",
        "interface_principal_spread_3",
        "interface_aspect_ratio",
        "interface_positive_residue_fraction",
        "interface_negative_residue_fraction",
        "interface_polar_residue_fraction",
        "interface_hydrophobic_residue_fraction",
        "interface_aromatic_residue_fraction",
    )
    if not binding:
        return {field: None for field in fields}

    residue_indices = list(protein["residue_indices"])
    selected         = [i for i, value in enumerate(residue_indices) if int(value) in binding]
    points           = np.asarray(protein["residue_positions"], dtype=np.float64)[selected]
    letters          = [protein["residue_letters"][index] for index in selected]
    centered         = points - points.mean(axis=0)
    covariance       = centered.T @ centered / max(1, len(centered))
    spreads          = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1])
    radius           = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))

    adjacency: dict[int, set[int]] = {index: set() for index in range(len(points))}
    for left, right in cKDTree(points).query_pairs(region_distance):
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(adjacency)
    regions: list[int] = []
    while remaining:
        pending = [remaining.pop()]
        size    = 0
        while pending:
            current = pending.pop()
            size += 1
            neighbours = adjacency[current] & remaining
            remaining.difference_update(neighbours)
            pending.extend(neighbours)
        regions.append(size)

    def fraction(group: frozenset[str]) -> float:
        """Return the interface-residue fraction belonging to ``group``."""
        return sum(letter in group for letter in letters) / len(letters)

    in_plane = float(spreads[1])

    return {
        "contact_density":            int(contacts["pair_count"])
        / len(protein["atom_positions"]),
        "contact_pair_count":         int(contacts["pair_count"]),
        "binding_residue_count":      len(letters),
        "contacting_atom_count":      len(contacts["contacting_atoms"]),
        "interface_region_count":     len(regions),
        "binding_residue_fraction":   len(letters)
        / int(protein["observed_residue_count"]),
        "interface_aspect_ratio":     (
            float(spreads[0] / in_plane) if in_plane > 1e-8 else None
        ),
        "contacted_dna_chain_count":  len(contacts["contacted_dna_chains"]),
        "interface_radius_normalized": radius
        / max(len(letters) ** (1.0 / 3.0), 1.0),
        "interface_radius_of_gyration": radius,
        "interface_principal_spread_1": float(spreads[0]),
        "interface_principal_spread_2": float(spreads[1]),
        "interface_principal_spread_3": float(spreads[2]),
        "interface_polar_residue_fraction":       fraction(POLAR),
        "interface_aromatic_residue_fraction":    fraction(AROMATIC),
        "interface_negative_residue_fraction":    fraction(NEGATIVE),
        "interface_positive_residue_fraction":    fraction(POSITIVE),
        "largest_interface_region_fraction":      max(regions) / len(letters),
        "interface_hydrophobic_residue_fraction": fraction(HYDROPHOBIC),
    }
