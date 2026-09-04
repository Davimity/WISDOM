"""Task-independent atom and residue descriptors derived from structural records."""

from __future__ import annotations

import numpy as np

from collections.abc import Mapping


class AtomicDescriptors:
    """Derive conservative generic chemistry from persisted atom and bond identities."""

    ARRAY_NAMES = (
        "atom_hybridization_ids",
        "atom_aromaticity",
        "atom_hbond_donor",
        "atom_hbond_acceptor",
        "residue_hydropathy",
        "residue_polarity",
    )
    HYDROPATHY: Mapping[str, float] = {
        "ALA": 1.8, "ARG": -4.5, "ASN": -3.5, "ASP": -3.5, "CYS": 2.5,
        "GLN": -3.5, "GLU": -3.5, "GLY": -0.4, "HIS": -3.2, "ILE": 4.5,
        "LEU": 3.8, "LYS": -3.9, "MET": 1.9, "PHE": 2.8, "PRO": -1.6,
        "SER": -0.8, "THR": -0.7, "TRP": -0.9, "TYR": -1.3, "VAL": 4.2,
    }
    POLAR_RESIDUES = frozenset(
        {"ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "HIS", "LYS", "SER", "THR", "TYR"}
    )
    DONORS = frozenset({
        ("ARG", "NE"), ("ARG", "NH1"), ("ARG", "NH2"), ("ASN", "ND2"),
        ("CYS", "SG"), ("GLN", "NE2"), ("HIS", "ND1"), ("HIS", "NE2"),
        ("LYS", "NZ"), ("SER", "OG"), ("THR", "OG1"), ("TRP", "NE1"),
        ("TYR", "OH"),
    })
    ACCEPTORS = frozenset({
        ("ASN", "OD1"), ("ASP", "OD1"), ("ASP", "OD2"), ("CYS", "SG"),
        ("GLN", "OE1"), ("GLU", "OE1"), ("GLU", "OE2"), ("HIS", "ND1"),
        ("HIS", "NE2"), ("MET", "SD"), ("SER", "OG"), ("THR", "OG1"),
        ("TYR", "OH"),
    })

    @classmethod
    def derive(
        cls,
        atomic_numbers : np.ndarray,
        atom_names     : np.ndarray,
        residue_names  : np.ndarray,
        formal_charges : np.ndarray,
        edge_index     : np.ndarray,
        bond_orders    : np.ndarray,
        is_covalent    : np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Build generic per-atom descriptors without using a task label.

        Aromaticity and hybridization are inferred from the conservative covalent graph already
        reconstructed by preprocessing. Donor/acceptor flags use standard heavy-atom protein
        chemistry; unknown/non-canonical residues remain unassigned rather than receiving a
        speculative type. Kyte--Doolittle hydropathy is divided by 4.5 to give an approximately
        unit interval scale.

        Args:
            atomic_numbers: Atomic numbers with shape ``[N]``.
            atom_names: PDB atom names with shape ``[N]``.
            residue_names: Residue names with shape ``[N]``.
            formal_charges: Integer formal charges with shape ``[N]``.
            edge_index: Undirected covalent/spatial pairs with shape ``[2,E]``.
            bond_orders: Bond order for each stored pair with shape ``[E]``.
            is_covalent: Boolean covalent flag for each stored pair with shape ``[E]``.

        Returns:
            Boolean aromatic/donor/acceptor arrays, categorical hybridization IDs, normalized
            hydropathy, residue-polarity flags, and formal charge as floating-point values.
        """
        count       = len(atomic_numbers)
        aromatic    = np.zeros(count, dtype=np.bool_)
        hybrid      = np.zeros(count, dtype=np.uint8)
        maximum_bond = np.zeros(count, dtype=np.float32)

        covalent_edges  = edge_index[:, is_covalent]
        covalent_orders = bond_orders[is_covalent]
        for endpoint in (0, 1):
            np.maximum.at(maximum_bond, covalent_edges[endpoint], covalent_orders)

        aromatic_edges = covalent_edges[:, np.isclose(covalent_orders, 1.5)]
        if aromatic_edges.size:
            aromatic[np.unique(aromatic_edges)] = True

        hybrid[maximum_bond > 0.0]  = 3
        hybrid[maximum_bond >= 1.5] = 2
        hybrid[maximum_bond >= 2.5] = 1

        names    = [str(value).strip().upper() for value in atom_names]
        residues = [str(value).strip().upper() for value in residue_names]
        elements = np.asarray(atomic_numbers)

        donor   = np.asarray(
            [
                (residue, name) in cls.DONORS
                or (name == "N" and element == 7 and residue != "PRO")
                for residue, name, element in zip(residues, names, elements, strict=True)
            ],
            dtype=np.bool_,
        )
        acceptor = np.asarray(
            [
                name in {"O", "OXT"} or (residue, name) in cls.ACCEPTORS
                for residue, name in zip(residues, names, strict=True)
            ],
            dtype=np.bool_,
        )
        hydropathy = np.asarray(
            [cls.HYDROPATHY.get(residue, 0.0) / 4.5 for residue in residues],
            dtype=np.float32,
        )
        polarity = np.asarray(
            [residue in cls.POLAR_RESIDUES for residue in residues],
            dtype=np.bool_,
        )
        return {
            "atom_aromaticity":       aromatic,
            "atom_hbond_donor":       donor,
            "atom_hbond_acceptor":    acceptor,
            "atom_hybridization_ids": hybrid,
            "residue_hydropathy":     hydropathy,
            "residue_polarity":       polarity,
            "formal_charges":         np.asarray(formal_charges, dtype=np.float32),
        }
