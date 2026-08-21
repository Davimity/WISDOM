"""Atomic features and multirelational graph construction."""

from __future__ import annotations

from itertools import pairwise

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from preprocess.chemical_data import (
    AMINO_ACID_TO_ID,
    BACKBONE_ATOM_NAMES,
    BACKBONE_BONDS,
    STANDARD_RESIDUE_BONDS,
    WATER_RESIDUES,
)
from preprocess.dataclasses.Atom import Atom
from preprocess.dataclasses.Protein import Protein
from preprocess.dataclasses.Residue import Residue
from preprocess.enums.AtomRole import AtomRole
from preprocess.enums.BondSource import BondSource
from preprocess.enums.BondType import BondType
from preprocess.enums.ConnectionType import ConnectionType
from preprocess.enums.Relation import Relation

Bond = tuple[BondType, BondSource, float]


class AtomicStructureBuilder:
    """Encode normalized atoms and construct one sparse multirelational topology."""

    def __init__(self, radius: float) -> None:
        """Set the Euclidean cutoff used for spatial atomic neighbors.

        Args:
            radius: Positive spatial graph radius in ångströms. Covalent pairs are retained even
                when their Euclidean distance exceeds this value.

        Raises:
            ValueError: If ``radius`` is not strictly positive.
        """
        if radius <= 0:
            raise ValueError("atom radius must be positive")

        self.radius = radius

    def build(self, protein: Protein) -> dict[str, np.ndarray]:
        """Encode atom features and the union of spatial and reconstructed covalent edges.

        Spatial pairs satisfy ``||x_i-x_j||_2 <= radius``. The final set is the sorted union
        ``E_spatial union E_covalent`` with each undirected pair stored once as ``i < j``. Relation
        bit flags preserve whether a pair is spatial, covalent, or both.

        Args:
            protein: Parser-independent ownership hierarchy with consecutive atom indices and
                optional source-declared connections.

        Returns:
            Compact NumPy atom features and edge arrays. Coordinates, distances, and radii are
            ``float32``; indices and closed categories use compact integer dtypes.

        Raises:
            ValueError: If hierarchy traversal does not produce consecutive atom indices.
        """
        # Residue indices are global, while the hierarchy itself remains free of storage metadata.
        entries       : list[tuple[Atom, Residue, int, int]] = []
        residue_index : int                                  = 0
        for chain_index, chain in enumerate(protein.chains):
            for residue in chain.residues:
                entries.extend(
                    (atom, residue, residue_index, chain_index) for atom in residue.atoms
                )
                residue_index += 1

        # Validate the critical invariant that atom indices address every persisted atom array.
        atoms = [entry[0] for entry in entries]
        if [atom.index for atom in atoms] != list(range(len(atoms))):
            raise ValueError("atom indices must be consecutive and follow hierarchy order")

        # Build working arrays in float64 before compact output conversion.
        positions       = np.asarray([atom.position for atom in atoms], dtype=np.float64)
        residue_indices = np.asarray([entry[2] for entry in entries], dtype=np.int32)
        chain_indices   = np.asarray([entry[3] for entry in entries], dtype=np.int16)
        elements        = [gemmi.Element(atom.atomic_number) for atom in atoms]

        # Assign exactly one role by chemical/entity precedence, never by learned atom typing.
        roles: list[AtomRole] = []
        for atom, residue, _, _ in entries:
            element = gemmi.Element(atom.atomic_number)
            if element.is_hydrogen:
                role = AtomRole.HYDROGEN
            elif element.is_metal:
                role = AtomRole.METAL
            elif residue.name in WATER_RESIDUES:
                role = AtomRole.WATER
            elif not residue.is_polymer:
                role = AtomRole.NONPOLYMER
            elif atom.name in BACKBONE_ATOM_NAMES:
                role = AtomRole.BACKBONE
            else:
                role = AtomRole.SIDECHAIN
            roles.append(role)

        # A KD-tree enumerates spatial pairs without constructing a dense distance matrix.
        bonds         = self._bonds(protein, positions)
        spatial_pairs = cKDTree(positions).query_pairs(self.radius, output_type="ndarray")
        spatial       = {
            tuple(map(int, pair))
            for pair in np.asarray(spatial_pairs, dtype=np.int32).reshape(-1, 2)
        }
        pairs      = sorted(spatial | set(bonds))
        edge_index = (
            np.asarray(pairs, dtype=np.int32).T if pairs else np.empty((2, 0), dtype=np.int32)
        )
        distances  = (
            np.linalg.norm(positions[edge_index[0]] - positions[edge_index[1]], axis=1)
            if pairs
            else np.empty(0, dtype=np.float64)
        )

        # Allocate one compact feature vector per union edge.
        relation     = np.zeros(len(pairs), dtype=np.uint8)
        bond_types   = np.full(len(pairs), BondType.NONE, dtype=np.uint8)
        bond_orders  = np.zeros(len(pairs), dtype=np.float32)
        bond_sources = np.full(len(pairs), BondSource.NONE, dtype=np.uint8)
        confidence   = np.zeros(len(pairs), dtype=np.float32)

        same_residue = np.zeros(len(pairs), dtype=np.uint8)
        same_chain   = np.zeros(len(pairs), dtype=np.uint8)
        separation   = np.full(len(pairs), np.iinfo(np.int16).max, dtype=np.int16)

        # Fill relation semantics and topological context in deterministic pair order.
        for index, pair in enumerate(pairs):
            if pair in spatial:
                relation[index] |= Relation.SPATIAL
            if pair in bonds:
                relation[index] |= Relation.COVALENT
                bond_type, source, bond_confidence = bonds[pair]
                bond_types[index]   = bond_type
                bond_orders[index]  = bond_type.order
                bond_sources[index] = source
                confidence[index]   = bond_confidence

            same_residue[index] = residue_indices[pair[0]] == residue_indices[pair[1]]
            same_chain[index]   = chain_indices[pair[0]] == chain_indices[pair[1]]
            if same_chain[index]:
                separation[index] = min(
                    abs(int(residue_indices[pair[0]]) - int(residue_indices[pair[1]])),
                    np.iinfo(np.int16).max,
                )

        return {
            "atom_positions": positions.astype(np.float32),
            "atomic_numbers": np.asarray([atom.atomic_number for atom in atoms], dtype=np.uint8),
            "residue_type_ids": np.asarray(
                [AMINO_ACID_TO_ID.get(entry[1].name, 0) for entry in entries],
                dtype=np.uint8,
            ),
            "atom_role_ids": np.asarray(roles, dtype=np.uint8),
            "residue_indices": residue_indices,
            "chain_indices": chain_indices,
            "formal_charges": np.asarray([atom.formal_charge for atom in atoms], dtype=np.int8),
            "vdw_radii": np.asarray([element.vdw_r for element in elements], dtype=np.float32),
            "covalent_radii": np.asarray(
                [element.covalent_r for element in elements], dtype=np.float32
            ),
            "atom_names": np.asarray([atom.name for atom in atoms], dtype="U8"),
            "residue_names": np.asarray([entry[1].name for entry in entries], dtype="U8"),
            "atom_edge_index": edge_index,
            "atom_edge_distance": distances.astype(np.float32),
            "atom_edge_relation_mask": relation,
            "atom_edge_bond_type": bond_types,
            "atom_edge_bond_order": bond_orders,
            "atom_edge_bond_source": bond_sources,
            "atom_edge_bond_confidence": confidence,
            "atom_edge_same_residue": same_residue,
            "atom_edge_same_chain": same_chain,
            "atom_edge_residue_separation": separation,
        }

    def _bonds(
        self,
        protein   : Protein,
        positions : np.ndarray,
    ) -> dict[tuple[int, int], Bond]:
        """Reconstruct conservative covalent connectivity with explicit precedence.

        Evidence is applied as explicit source records, standard residue templates, peptide bonds,
        disulfides, then an intra-residue geometric fallback for non-canonical components. The
        fallback accepts ``0.4 <= d_ij <= 1.15(r_i+r_j)`` ångströms. Existing higher-priority pairs
        are never overwritten except by explicit records.

        Args:
            protein: Normalized hierarchy and source-declared typed connections.
            positions: ``float64 [N,3]`` atom coordinates in ångströms indexed by ``Atom.index``.

        Returns:
            Undirected ``(min_index, max_index)`` pairs mapped to bond type, evidence source, and
            deterministic heuristic confidence.
        """
        bonds: dict[tuple[int, int], Bond] = {}

        def add(
            left     : int,
            right    : int,
            value    : Bond,
            *,
            replace  : bool = False,
        ) -> None:
            """Insert one normalized non-self bond unless stronger evidence already exists.

            Args:
                left: First consecutive atom index.
                right: Second consecutive atom index.
                value: Bond type, evidence source, and heuristic confidence.
                replace: Whether explicit evidence may overwrite an existing pair.
            """
            if left == right:
                return

            pair = (left, right) if left < right else (right, left)
            if replace or pair not in bonds:
                bonds[pair] = value

        # Explicit covalent-like records have absolute precedence; hydrogen bonds are excluded.
        covalent_connections = {
            ConnectionType.COVALENT,
            ConnectionType.DISULFIDE,
            ConnectionType.METAL_COORDINATION,
        }
        for left, right, connection_type, bond_type in protein.explicit_connections:
            if connection_type in covalent_connections:
                add(
                    left.index,
                    right.index,
                    (bond_type, BondSource.EXPLICIT, 1.0),
                    replace=True,
                )

        # Canonical atom-name templates establish known intra-residue chemistry.
        residues    = [residue for chain in protein.chains for residue in chain.residues]
        named_atoms = [{atom.name: atom.index for atom in residue.atoms} for residue in residues]
        for residue, named in zip(residues, named_atoms, strict=True):
            if residue.name not in STANDARD_RESIDUE_BONDS:
                continue
            for left_name, right_name, bond_type, _ in (
                *BACKBONE_BONDS,
                *STANDARD_RESIDUE_BONDS[residue.name],
            ):
                if left_name in named and right_name in named:
                    add(
                        named[left_name],
                        named[right_name],
                        (BondType(bond_type), BondSource.TEMPLATE, 0.98),
                    )

        # Adjacent polymer residues form a peptide bond only under a conservative C-N cutoff.
        residue_offset = 0
        for chain in protein.chains:
            for local_left, local_right in pairwise(range(len(chain.residues))):
                left_residue  = chain.residues[local_left]
                right_residue = chain.residues[local_right]
                if not (left_residue.is_polymer and right_residue.is_polymer):
                    continue
                left_index  = named_atoms[residue_offset + local_left].get("C")
                right_index = named_atoms[residue_offset + local_right].get("N")
                if (
                    left_index is not None
                    and right_index is not None
                    and np.linalg.norm(positions[left_index] - positions[right_index]) <= 1.9
                ):
                    add(
                        left_index,
                        right_index,
                        (BondType.PEPTIDE, BondSource.PEPTIDE, 0.99),
                    )
            residue_offset += len(chain.residues)

        # A KD-tree finds CYS SG pairs compatible with disulfide geometry.
        sulfurs = [
            atom.index
            for residue in residues
            if residue.name == "CYS"
            for atom in residue.atoms
            if atom.name == "SG"
        ]
        if len(sulfurs) > 1:
            sulfur_indices = np.asarray(sulfurs, dtype=np.int32)
            sulfur_pairs   = cKDTree(positions[sulfur_indices]).query_pairs(
                2.3, output_type="ndarray"
            )
            for local_left, local_right in np.asarray(sulfur_pairs, dtype=np.int32).reshape(-1, 2):
                add(
                    int(sulfur_indices[local_left]),
                    int(sulfur_indices[local_right]),
                    (BondType.DISULFIDE, BondSource.DISULFIDE, 0.95),
                )

        # Geometry is deliberately a final fallback restricted to one non-canonical residue.
        for residue in residues:
            if residue.name in STANDARD_RESIDUE_BONDS or len(residue.atoms) < 2:
                continue
            indices = np.asarray([atom.index for atom in residue.atoms], dtype=np.int32)
            radii   = np.asarray(
                [gemmi.Element(atom.atomic_number).covalent_r for atom in residue.atoms]
            )

            candidate_pairs = cKDTree(positions[indices]).query_pairs(
                2.3 * float(radii.max()), output_type="ndarray"
            )
            for local_left, local_right in np.asarray(candidate_pairs, dtype=np.int32).reshape(
                -1, 2
            ):
                source_index = int(indices[local_left])
                target_index = int(indices[local_right])
                distance     = float(
                    np.linalg.norm(positions[source_index] - positions[target_index])
                )
                cutoff       = 1.15 * (radii[local_left] + radii[local_right])
                if 0.4 <= distance <= cutoff:
                    add(
                        source_index,
                        target_index,
                        (BondType.SINGLE, BondSource.GEOMETRIC, 0.55),
                    )

        return bonds
