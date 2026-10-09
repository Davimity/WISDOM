"""Assembly-aware verification of occupied zinc and protein coordination partners."""

import numpy as np

from typing import Any
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
            partners = [donors[i] for i in tree.query_ball_point(metal["position"], self.cutoff)]
            local = [atom for atom in partners if atom["chain_index"] == selected_index]
            residues = sorted({atom["residue"] for atom in local})
            sites.append({**metal, "site_id": f"ZN{index:04d}", "partners": partners,
                          "selected_chain_index": selected_index,
                          "selected_donor_count": len(local),
                          "selected_residues": residues,
                          "selected_residue_names": sorted({a["residue_name"] for a in local}),
                          "interchain": len({a["chain_index"] for a in partners}) > 1,
                          "accepted": len(local) >= self.minimum_donors
                          and len(residues) >= self.minimum_residues,
                          "definition": f"occupied protein N/O/S centers within {self.cutoff} Å"})
        return sites
