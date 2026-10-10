"""Describe protein sequences and coordinates independently of benchmark labels."""

import math
import gemmi
import numpy as np

from typing import Any
from collections import Counter
from scipy.spatial import cKDTree
from collections.abc import Mapping
from lambdaforge.work import ManagedFile
from Bio.SeqUtils.ProtParam import ProteinAnalysis

HYDROPHOBIC = frozenset("AVILMFWY")
POLAR       = frozenset("STNQCY")
POSITIVE    = frozenset("KRH")
NEGATIVE    = frozenset("DE")
AROMATIC    = frozenset("FWY")
AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYOU")


def protein_coordinates(chain: gemmi.Chain, sequence_length: int) -> dict[str, Any]:
    """Extract occupied heavy atoms and one representative point per observed residue.

    Args:
        chain: Selected biological-assembly protein copy.
        sequence_length: Complete entity sequence length.

    Returns:
        NumPy atom/residue arrays used only for sparse contacts and compact descriptors.
    """
    atoms: list[tuple[float, float, float]] = []
    radii: list[float] = []
    owners: list[int] = []
    residue_letters: dict[int, str] = {}
    residue_points: dict[int, tuple[float, float, float]] = {}
    for residue in chain.get_polymer().first_conformer():
        if residue.label_seq is None:
            continue
        position = int(residue.label_seq) - 1
        if not 0 <= position < sequence_length:
            continue
        letter = (gemmi.find_tabulated_residue(residue.name).one_letter_code or "X").upper()
        heavy: list[tuple[float, float, float]] = []
        alpha_carbon: tuple[float, float, float] | None = None
        for atom in residue.first_conformer():
            if atom.element.atomic_number <= 1 or atom.occ <= 0.0:
                continue
            point  = (float(atom.pos.x), float(atom.pos.y), float(atom.pos.z))
            radius = float(atom.element.vdw_r)
            if not np.isfinite(point).all() or not math.isfinite(radius) or radius <= 0.0:
                raise ValueError(f"chain {chain.name!r} contains an invalid heavy atom")
            atoms.append(point)
            radii.append(radius)
            owners.append(position)
            heavy.append(point)
            if atom.name.strip() == "CA":
                alpha_carbon = point
        if heavy:
            residue_letters[position] = letter
            residue_points[position]  = alpha_carbon or tuple(np.mean(heavy, axis=0).tolist())
    ordered = sorted(residue_points)
    if not ordered:
        raise ValueError(f"protein chain {chain.name!r} has no occupied heavy atoms")
    return {
        "atom_radii":     np.asarray(radii, dtype=np.float64),
        "atom_owners":    np.asarray(owners, dtype=np.int64),
        "atom_positions": np.asarray(atoms, dtype=np.float64).reshape((-1, 3)),

        "residue_indices":   ordered,
        "residue_letters":   [residue_letters[index] for index in ordered],
        "residue_positions": np.asarray(
            [residue_points[index] for index in ordered], dtype=np.float64
        ).reshape((-1, 3)),

        "observed_residue_count": len(ordered),
    }

def sequence_features(sequence: str) -> dict[str, Any]:
    """Calculate interpretable whole-sequence physicochemical descriptors.

    Args:
        sequence: Complete uppercase protein sequence.

    Returns:
        Length, fractions, entropy, and conventional Biopython physical estimates.
    """
    length    = len(sequence)
    counts    = Counter(sequence)
    fractions = {amino: counts[amino] / length for amino in AMINO_ACIDS}
    entropy   = -sum(value * math.log2(value) for value in fractions.values() if value)
    physical: dict[str, float | None] = {
        "gravy":                         None,
        "molecular_weight":              None,
        "net_charge_at_pH_7":            None,
        "theoretical_isoelectric_point": None,
        "aromatic_fraction":             sum(fractions[value] for value in AROMATIC),
    }
    try:
        analysis: Any = ProteinAnalysis(sequence)  # type: ignore[no-untyped-call]
        physical.update(
            {
                "gravy":                         float(analysis.gravy()),
                "molecular_weight":              float(analysis.molecular_weight()),
                "net_charge_at_pH_7":            float(analysis.charge_at_pH(7.0)),
                "aromatic_fraction":             float(analysis.aromaticity()),
                "theoretical_isoelectric_point": float(analysis.isoelectric_point()),
            }
        )
    except (KeyError, ValueError):
        pass
    return {
        "sequence_length":           length,
        **physical,
        "glycine_fraction":          fractions["G"],
        "proline_fraction":          fractions["P"],
        "cysteine_fraction":         fractions["C"],
        "polar_residue_fraction":    sum(fractions[value] for value in POLAR),
        "negative_residue_fraction": sum(fractions[value] for value in NEGATIVE),
        "positive_residue_fraction": sum(fractions[value] for value in POSITIVE),
        "hydrophobic_residue_fraction": sum(fractions[value] for value in HYDROPHOBIC),
        "sequence_shannon_entropy":    entropy,
        **{f"fraction_{amino}": fractions[amino] for amino in sorted("ACDEFGHIKLMNPQRSTVWY")},
    }

def global_features(protein: Mapping[str, Any]) -> dict[str, float | int]:
    """Measure global protein size, shape, and non-local residue packing.

    Args:
        protein: Residue representatives, sequence owners, and heavy-atom positions.

    Returns:
        Rotation-invariant global physical descriptors.
    """
    positions  = np.asarray(protein["residue_positions"], dtype=np.float64)
    indices    = np.asarray(protein["residue_indices"], dtype=np.int64)
    centered   = positions - positions.mean(axis=0)
    covariance = centered.T @ centered / max(1, len(centered))
    spreads    = np.sqrt(np.maximum(np.linalg.eigvalsh(covariance), 0.0)[::-1])
    radius     = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    pairs      = cKDTree(positions).query_pairs(8.0)
    nonlocal_pairs = sum(
        abs(int(indices[left]) - int(indices[right])) >= 3 for left, right in pairs
    )
    volume = 4.0 * math.pi * max(radius, 1e-6) ** 3 / 3.0

    return {
        "compactness":              len(positions) / volume,
        "aspect_ratio":             float(spreads[0] / max(spreads[2], 1e-8)),
        "packing_density":          nonlocal_pairs / max(1, len(positions)),
        "heavy_atom_count":         len(protein["atom_positions"]),
        "principal_spread_1":       float(spreads[0]),
        "principal_spread_2":       float(spreads[1]),
        "principal_spread_3":       float(spreads[2]),
        "radius_of_gyration":       radius,
        "nonlocal_ca_contact_count": nonlocal_pairs,
        "radius_of_gyration_normalized": (
            radius / max(len(positions) ** (1.0 / 3.0), 1.0)
        ),
    }

def rigid_transform(
    deposited: gemmi.Chain,
    assembled: gemmi.Chain,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit the proper rigid transform ``x' = x R^T + t`` by Kabsch alignment.

    Args:
        deposited: Original asymmetric-unit chain.
        assembled: Selected generated biological-assembly copy.

    Returns:
        Rotation ``[3,3]`` and translation ``[3]`` in ångströms.
    """
    source_map = _atom_map(deposited)
    target_map = _atom_map(assembled)
    common     = sorted(source_map.keys() & target_map.keys())
    if len(common) < 3:
        raise ValueError("assembly transform needs at least three matching heavy atoms")
    source        = np.asarray([source_map[key] for key in common], dtype=np.float64)
    target        = np.asarray([target_map[key] for key in common], dtype=np.float64)
    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    left, _, right = np.linalg.svd(
        (source - source_center).T @ (target - target_center)
    )
    rotation = right.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right[-1] *= -1.0
        rotation = right.T @ left.T
    translation = target_center - source_center @ rotation.T
    return rotation, translation

def _atom_map(chain: gemmi.Chain) -> dict[tuple[int, str], tuple[float, float, float]]:
    """Map residue/atom identity to occupied heavy-atom coordinates.

    Args:
        chain: Deposited or assembled protein chain.

    Returns:
        Coordinate mapping used by rigid alignment.
    """
    atoms: dict[tuple[int, str], tuple[float, float, float]] = {}
    for residue in chain.get_polymer().first_conformer():
        if residue.label_seq is None:
            continue
        for atom in residue.first_conformer():
            if atom.element.atomic_number > 1 and atom.occ > 0.0:
                atoms[(int(residue.label_seq), atom.name.strip())] = (
                    float(atom.pos.x),
                    float(atom.pos.y),
                    float(atom.pos.z),
                )
    return atoms

def valid_structure(file: ManagedFile) -> bool:
    """Check whether one downloaded coordinate file contains a parseable model.

    Args:
        file: LambdaForge-managed, decompressed PDBx/mmCIF candidate.

    Returns:
        True when Gemmi parses at least one coordinate model; false for I/O, syntax, or empty-model
        failures. LambdaForge uses this result before publishing the cache entry.
    """
    try:
        return bool(gemmi.read_structure(str(file)))
    except (OSError, RuntimeError, ValueError):
        return False

def valid_foldseek_structure(file: ManagedFile) -> bool:
    """Check whether one generated Foldseek input contains exactly one protein chain.

    Args:
        file: LambdaForge-managed PDBx/mmCIF file generated for one selected chain copy.

    Returns:
        True when Gemmi parses a nonempty first model containing exactly one chain; false for
        malformed, unreadable, empty, or multi-chain files.
    """
    try:
        structure = gemmi.read_structure(str(file))
        return bool(structure) and len(structure[0]) == 1
    except (OSError, RuntimeError, ValueError):
        return False
