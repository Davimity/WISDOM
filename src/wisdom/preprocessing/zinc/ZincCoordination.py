"""Assembly-aware verification of occupied zinc and protein coordination partners."""

import numpy as np

from typing import Any
from collections import Counter
from scipy.spatial import cKDTree
from wisdom.utils.structure.BiologicalAssembly import BiologicalAssembly


class ZincCoordination:
    """Measure deposited coordination, without claiming physiological metal specificity."""

    def __init__(
        self, cutoff: float = 3.0, minimum_occupancy: float = 0.5,
        minimum_donors: int = 2, minimum_residues: int = 2,
    ) -> None:
        """Configure structural acceptance thresholds, not a learned chemistry model.

        Args:
            cutoff: Maximum Zn-protein N/O/S center distance in ångströms.
            minimum_occupancy: Minimum deposited Zn and donor occupancy in [0,1].
            minimum_donors: Minimum distinct coordinating protein N/O/S atoms.
            minimum_residues: Minimum distinct protein residues contributing those atoms.
        """
        self.cutoff = cutoff
        self.minimum_occupancy = minimum_occupancy
        self.minimum_donors = minimum_donors
        self.minimum_residues = minimum_residues

    def analyse(
        self, assembly: BiologicalAssembly, chain_name: str, copy_index: int,
    ) -> list[dict[str, Any]]:
        """Return every assembly zinc site and its selected-copy coordination evidence.

        A sparse KD-tree queries only N/O/S atoms within cutoff. Water, unrelated protein
        copies and non-protein ligands are not selected-copy donors. Alternate conformers use
        Gemmi's deterministic first conformer. This geometric proxy does not resolve valence,
        protonation, affinity, or biological relevance; these require independent evidence.

        Args:
            assembly: Exact named biological assembly, not only the asymmetric unit.
            chain_name: Complete deposited protein chain name.
            copy_index: One-based assembly copy selected by the design.

        Returns:
            JSON sites including occupancy, metal coordinates, donor/residue counts, partners,
            shared/inter-chain status, coordinating residues and threshold acceptance.

        Raises:
            ValueError: Missing requested protein copy or non-finite occupied atom coordinates.
        """
        _, selected = assembly.protein_copy(chain_name, copy_index)
        # Each chain occurrence is a separate assembly entity even when names are repeated.

        selected_index = next(i for i, chain in enumerate(assembly.model) if chain == selected)
        donors: list[dict[str, Any]] = []
        metals: list[dict[str, Any]] = []
        for chain_index, chain in enumerate(assembly.model):
            peptide = assembly.is_protein(chain)
            for residue in chain.first_conformer():
                for atom in residue.first_conformer():
                    if atom.occ < self.minimum_occupancy:
                        continue
                    xyz = [float(atom.pos.x), float(atom.pos.y), float(atom.pos.z)]
                    if not np.isfinite(xyz).all():
                        raise ValueError("occupied coordination atom has non-finite coordinates")
                    record = {"chain_index": chain_index, "chain": chain.name,
                              "residue": str(residue.seqid), "residue_name": residue.name,
                              "atom": atom.name, "position": xyz, "occupancy": float(atom.occ),
                              "element": atom.element.name,
                              "vdw_radius": float(atom.element.vdw_r)}
                    if atom.element.atomic_number == 30:
                        metals.append(record)
                    elif peptide and atom.element.atomic_number in (7, 8, 16):
                        donors.append(record)
        tree = cKDTree(np.asarray([atom["position"] for atom in donors]).reshape(-1, 3))
        sites = []
        for index, metal in enumerate(metals):
            partners = [{**donors[i], "distance_angstrom": float(np.linalg.norm(
                np.asarray(donors[i]["position"]) - metal["position"]))}
                for i in sorted(tree.query_ball_point(metal["position"], self.cutoff))]
            local = [atom for atom in partners if atom["chain_index"] == selected_index]
            residues = sorted({atom["residue"] for atom in local})
            # Count distinct residues, not donor atoms or unique residue types. Two Cys and
            # two His must remain distinguishable from one Cys and three His.

            distinct = {atom["residue"]: atom["residue_name"] for atom in local}
            residue_counts = dict(sorted(Counter(distinct.values()).items()))
            element_counts = dict(sorted(Counter(a["element"] for a in local).items()))
            distances = [atom["distance_angstrom"] for atom in local]
            directions = np.asarray([np.asarray(a["position"]) - metal["position"]
                                     for a in local]).reshape(-1, 3)
            lengths = np.linalg.norm(directions, axis=1)
            directions = directions[lengths > 0] / lengths[lengths > 0, None]
            angles = [float(np.degrees(np.arccos(np.clip(np.dot(left, right), -1, 1))))
                      for i, left in enumerate(directions) for right in directions[i + 1:]]
            sites.append({**metal, "site_id": f"ZN{index:04d}", "partners": partners,
                          "coordination_schema": "2.0",
                          "assembly_id": assembly.assembly_id,
                          "protein_chain": chain_name, "protein_copy": copy_index,
                          "selected_chain_index": selected_index,
                          "selected_donor_count": len(local),
                          "selected_residues": residues,
                          "selected_residue_names": sorted({a["residue_name"] for a in local}),
                          "selected_residue_counts": residue_counts,
                          "selected_element_counts": element_counts,
                          "selected_atom_type_counts": dict(sorted(Counter(
                              f"{a['residue_name']}:{a['atom']}" for a in local).items())),
                          "coordination_distance_mean": float(np.mean(distances))
                          if distances else None,
                          "coordination_distance_std": float(np.std(distances))
                          if distances else None,
                          "donor_angles_degrees": angles,
                          "interchain": len({a["chain_index"] for a in partners}) > 1,
                          "accepted": len(local) >= self.minimum_donors
                          and len(residues) >= self.minimum_residues,
                          "definition": f"occupied protein N/O/S centers within {self.cutoff} Å"})
        return sites

    @staticmethod
    def surface_gaps(
        points: np.ndarray, coordinates: np.ndarray, radii: np.ndarray,
    ) -> np.ndarray:
        """Measure minimum signed gaps to coordinating spheres without dense point pairs.

        Args:
            points: Surface positions [M,3] in assembly coordinates, ångströms.
            coordinates: Coordinating atom centers [N,3] in the same frame.
            radii: Positive donor van der Waals radii [N], ångströms.

        Returns:
            Float32 [M] minimum(||point-atom||-radius). A nearest-center upper bound plus
            maximum radius includes every atom that could improve the minimum.

        Raises:
            ValueError: No coordinating atoms are provided.
        """
        if len(coordinates) == 0:
            raise ValueError("coordinating surface gaps require at least one donor")
        tree       = cKDTree(coordinates)
        nearest    = tree.query(points)[0]
        candidates = tree.query_ball_point(points, nearest + radii.max())
        return np.asarray([
            np.min(np.linalg.norm(coordinates[ids] - point, axis=1) - radii[ids])
            for point, ids in zip(points, candidates, strict=True)
        ], dtype=np.float32)

    @staticmethod
    def matches(observed: list[dict[str, Any]], expected: list[dict[str, Any]]) -> bool:
        """Compare immutable site evidence with a controlled historical-schema boundary.

        Args:
            observed: Newly reproduced coordination evidence from exact deposited bytes.
            expected: Published site records, historical schema or current 2.0 records.

        Returns:
            True only when coordination fields agree. Schema-2 excludes separately derived
            surface-support/phenotype fields; all remaining fields require exact equality;
            historical records compare their complete original vocabulary, not new derived fields.
        """
        if len(observed) != len(expected):
            return False
        for actual, prior in zip(observed, expected, strict=True):
            if prior.get("coordination_schema") == "2.0":
                coordination = {key: value for key, value in prior.items()
                                if key not in {
                                    "surface_support", "local_phenotype", "external_metadata"}}
                if actual != coordination:
                    return False
                continue
            for key, value in prior.items():
                if key == "partners":
                    partners = actual[key]
                    remaining = list(partners)
                    for partner in value:
                        match = next((index for index, candidate in enumerate(remaining)
                                      if all(candidate.get(field) == item
                                             for field, item in partner.items())), None)
                        if match is None:
                            return False
                        remaining.pop(match)
                    if remaining:
                        return False
                elif actual.get(key) != value:
                    return False
        return True
