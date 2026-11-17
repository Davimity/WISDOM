"""Bound modeled chemical completion without modifying the observed molecular graph."""

import numpy as np

from typing import Any
from scipy.spatial import cKDTree
from collections import Counter
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.structure.assets.physics.PhysicsConfig import PhysicsConfig
from wisdom.preprocessing.common.structure.assets.physics.HeavyAtomPolicy import HeavyAtomPolicy
from wisdom.preprocessing.common.structure.assets.physics.PhysicalReferenceUnavailable import (
    PhysicalReferenceUnavailable,
)


class PreparationAdmission:
    """Apply explicit pilot limits; admission never establishes experimental physical accuracy."""

    BACKBONE = frozenset({"N", "CA", "C", "O"})

    def __init__(self, config: PhysicsConfig) -> None:
        """Bind scientific completion thresholds, leaving numerical convergence independent.

        Args:
            config: Strict historical policy or bounded completion and its count/geometry limits.
        """
        self.config = config
        self.policy = HeavyAtomPolicy(config.heavy_atom_policy)

    def validate_observed_termini(self, arrays: Mapping[str, np.ndarray]) -> None:
        """Reject incompatible observed terminal oxygen geometry before chemical preparation.

        Args:
            arrays: Immutable heavy-atom positions [N,3] in Å, atom names [N] and integer
                residue/chain ordinals [N]. No source coordinate or atom identity is changed.

        Raises:
            PhysicalReferenceUnavailable: An observed OXT lacks its carbonyl C, its C--OXT
                distance is outside the generous 0.9--2.2 Å bond guard, or the same carbon
                continues a peptide to the next observed residue at 0.9--1.9 Å. These are
                incompatible terminal assignments, not small missing-side-chain repairs.
                The structured audit records the original atom identity and measured distance.

        An oxygen named OXT denotes a peptide C terminus. PDB2PQR may split and rename chains
        at an internal OXT even with --keep-chain. Do not normalize that rename and silently
        introduce new charged termini. Atom-wise alternative-conformation selection can leave
        such conflicts in an immutable parent; retain the parent and make VF unavailable only.
        These checks identify contradictions, not a general validation of observed chemistry.
        """
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        for index in np.flatnonzero(arrays["atom_names"] == "OXT"):
            chain   = int(arrays["chain_indices"][index])
            residue = int(arrays["residue_indices"][index])
            same_chain = arrays["chain_indices"] == chain
            carbon = np.flatnonzero(same_chain & (arrays["residue_indices"] == residue) &
                                    (arrays["atom_names"] == "C"))
            audit: dict[str, Any] = {"policy": self.policy.value,
                "category": "incompatible_observed_terminus",
                "terminal_atom": [alphabet[chain], residue + 1, "OXT"]}
            if not len(carbon):
                raise PhysicalReferenceUnavailable("VF observed OXT lacks its carbonyl C", audit)
            distance = float(np.linalg.norm(arrays["atom_positions"][index] -
                                             arrays["atom_positions"][carbon[0]]))
            audit["carbon_oxygen_distance_angstrom"] = distance
            if not .9 <= distance <= 2.2:
                raise PhysicalReferenceUnavailable("VF observed C--OXT bond is incompatible "
                    f"with a physical reference ({distance:.3f} Å)", audit)

            # A genuine disconnected segment may end before the last observed residue. Reject
            # only an OXT whose carbon is also at peptide-bond distance from the next segment.

            following = arrays["residue_indices"][same_chain &
                                                  (arrays["residue_indices"] > residue)]
            if not len(following):
                continue
            next_residue = int(following.min())
            nitrogen = np.flatnonzero(same_chain &
                (arrays["residue_indices"] == next_residue) & (arrays["atom_names"] == "N"))
            if not len(nitrogen):
                continue
            peptide = float(np.linalg.norm(arrays["atom_positions"][carbon[0]] -
                                            arrays["atom_positions"][nitrogen[0]]))
            if .9 <= peptide <= 1.9:
                audit["following_residue"] = [alphabet[chain], next_residue + 1]
                audit["peptide_distance_angstrom"] = peptide
                raise PhysicalReferenceUnavailable("VF observed OXT conflicts with a continuing "
                                                    "peptide bond", audit)

    def assess(
        self, arrays: Mapping[str, np.ndarray], prepared_keys: Sequence[tuple[str, int, str]],
        positions: np.ndarray, elements: np.ndarray,
        sequence: Mapping[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Check exact observed correspondence and admit only small supported completions.

        Args:
            arrays: Original positions [N,3] Å, names/elements and residue/chain ordinals.
            prepared_keys: Prepared atom identities in canonical temporary PDB chain/residue space.
            positions: Prepared coordinates [P,3] Å, including hydrogen and modeled heavy atoms.
            elements: Prepared atomic numbers [P].
            sequence: Optional base-bound SequenceContext arrays. Newly modeled OXT requires an
                exact match to the last token of the deposited complete chain, not its last atom.

        Returns:
            JSON-compatible counts, rebuilt atom identities/indices and original-heavy mapping.
            No new atom becomes an encoder input. Numeric limits are pilot heuristics, not a
            theorem that a bounded reconstruction negligibly changes every physical field.

        Raises:
            PhysicalReferenceUnavailable: Unsupported reconstruction, backbone gap or count limit.
            ValueError: Lost/moved/element-changed observed atoms or duplicate prepared identities.
        """
        alphabet = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"
        original = [(alphabet[int(c)], int(r) + 1, str(n)) for c, r, n in zip(
            arrays["chain_indices"], arrays["residue_indices"], arrays["atom_names"], strict=True)]
        lookup = {key: i for i, key in enumerate(prepared_keys)}
        heavy = {key for key, element in zip(prepared_keys, elements, strict=True) if element > 1}
        if len(lookup) != len(prepared_keys):
            raise ValueError("VF prepared atom identities are duplicated")
        missing = sorted(set(original) - heavy)
        rebuilt = sorted(heavy - set(original))
        category = "terminal_oxygen_only" if rebuilt and all(k[-1] == "OXT" for k in rebuilt) \
                   else "sidechain_completion" if rebuilt else "observed_only"
        affected = Counter((c, r) for c, r, _ in rebuilt)
        residues = set((c, r) for c, r, _ in original)
        audit: dict[str, Any] = {"policy": self.policy.value, "category": category,
            "missing_heavy": missing, "reconstructed_heavy": rebuilt,
            "original_heavy_count": len(original), "reconstructed_heavy_count": len(rebuilt),
            "prepared_heavy_count": len(heavy), "affected_residues": len(affected),
            "reconstructed_fraction": len(rebuilt) / len(original),
            "affected_residue_fraction": len(affected) / len(residues),
            "rebuilt_indices": [lookup[key] for key in rebuilt]}
        if missing:
            raise ValueError(f"VF preparation removed observed heavy atoms: {missing[:8]}")
        if not np.isfinite(positions).all() or positions.shape != (len(prepared_keys), 3):
            raise ValueError("VF prepared coordinates are nonfinite or lose their atom domain")
        mapping = np.asarray([lookup[key] for key in original], dtype=np.int32)
        if not np.array_equal(elements[mapping], arrays["atomic_numbers"]) or not np.allclose(
            positions[mapping], arrays["atom_positions"], rtol=0, atol=.002
        ):
            raise ValueError("VF preparation changed observed elements or coordinates "
                             "beyond rounding")
        audit["heavy_mapping"] = mapping.tolist()
        if self.policy == HeavyAtomPolicy.STRICT:
            if rebuilt:
                raise PhysicalReferenceUnavailable("VF strict policy rejects reconstructed heavy "
                    f"atoms: {rebuilt[:8]}", audit)
            return audit

        # A missing peptide segment must not become a long fictitious covalent bond just because
        # temporary PDB residue ordinals are consecutive. Never reconstruct N/CA/C/O or residues.

        for chain in np.unique(arrays["chain_indices"]):
            rows = np.flatnonzero(arrays["chain_indices"] == chain)
            previous = None
            for residue in np.unique(arrays["residue_indices"][rows]):
                ids = rows[arrays["residue_indices"][rows] == residue]
                atoms = {str(arrays["atom_names"][i]): int(i) for i in ids}
                if not atoms.keys() >= self.BACKBONE:
                    raise PhysicalReferenceUnavailable("VF bounded repair requires observed "
                        "N/CA/C/O in every residue; no backbone rebuilding is admitted", audit)
                if previous is not None:
                    gap = np.linalg.norm(arrays["atom_positions"][previous] -
                                         arrays["atom_positions"][atoms["N"]])
                    if gap > 1.9 or gap < .9:
                        raise PhysicalReferenceUnavailable("VF observed peptide backbone has a "
                            f"gap or invalid C--N distance ({gap:.3f} Å)", audit)
                previous = atoms["C"]
        limits = (
            len(rebuilt) <= self.config.repair_maximum_atoms,
            audit["reconstructed_fraction"] <= self.config.repair_maximum_fraction,
            max(affected.values(), default=0) <= self.config.repair_maximum_per_residue,
            audit["affected_residue_fraction"] <= self.config.repair_maximum_residue_fraction,
        )
        if not all(limits):
            raise PhysicalReferenceUnavailable("VF completion exceeds its declared atom/residue "
                                               "count or fraction limits", audit)
        verified_termini = []
        for chain, residue, name in rebuilt:
            if (chain, residue) not in residues or name in self.BACKBONE:
                raise PhysicalReferenceUnavailable("VF cannot add residues or backbone atoms",
                                                   audit)
            if name != "OXT":
                continue
            row = residue - 1
            if sequence is None or not sequence["complete_valid"][row]:
                raise PhysicalReferenceUnavailable("VF modeled OXT lacks verified deposited "
                                                   "C-terminal sequence evidence", audit)
            c = int(sequence["residue_chain"][row])
            if sequence["complete_index"][row] != sequence["complete_ptr"][c + 1] - 1:
                raise PhysicalReferenceUnavailable("VF refuses to cap an unresolved sequence "
                                                   "tail with a modeled OXT", audit)
            verified_termini.append([chain, residue])
        audit["verified_modeled_c_termini"] = verified_termini
        return audit

    def geometry(
        self, positions: np.ndarray, elements: np.ndarray, sigma: np.ndarray,
        bonds: Sequence[tuple[int, int]], rebuilt: Sequence[int],
    ) -> dict[str, Any]:
        """Reject collapsed modeled bonds and severe nonbonded heavy-atom overlaps.

        Args:
            positions: Prepared coordinates [P,3] in Å; observed coordinates stay fixed.
            elements: Prepared atomic numbers [P].
            sigma: Exact force-field LJ diameters [P] in nm, not heuristic atom radii.
            bonds: OpenMM standard/disulfide topology pairs in prepared atom order.
            rebuilt: Indices of newly modeled heavy atoms. Hydrogen addition is not a count limit.

        Returns:
            Accepted geometric checks. Bonded and angle-neighbor pairs are excluded from the
            nonbonded test; all other modeled-heavy pairs require r >= fraction * mixed sigma.

        Raises:
            PhysicalReferenceUnavailable: A repaired heavy bond is outside 0.9--2.2 Å or a severe
                nonbonded overlap remains. This is a clash guard, not a conformer validation.
        """
        if not rebuilt:
            return {"performed": False, "reason": "no modeled heavy atoms"}
        adjacency: dict[int, set[int]] = {i: set() for i in range(len(positions))}
        for a, b in bonds:
            adjacency[a].add(b)
            adjacency[b].add(a)
        tree = cKDTree(positions)
        for index in rebuilt:
            if not adjacency[index]:
                raise PhysicalReferenceUnavailable("VF modeled atom has no force-field bond")
            excluded = {index, *adjacency[index]}
            for neighbor in adjacency[index]:
                excluded.update(adjacency[neighbor])
                if elements[neighbor] > 1:
                    length = float(np.linalg.norm(positions[index] - positions[neighbor]))
                    if not .9 <= length <= 2.2:
                        raise PhysicalReferenceUnavailable("VF reconstructed heavy bond is "
                                                           f"implausible ({length:.3f} Å)")
            radius = self.config.repair_minimum_nonbonded_fraction * 5 * (
                sigma[index] + float(sigma.max()))
            for other in tree.query_ball_point(positions[index], radius):
                if other in excluded or elements[other] <= 1:
                    continue
                threshold = self.config.repair_minimum_nonbonded_fraction * 5 * (
                    sigma[index] + sigma[other])
                if np.linalg.norm(positions[index] - positions[other]) < threshold:
                    raise PhysicalReferenceUnavailable("VF reconstructed heavy atom has a severe "
                                                       "nonbonded steric overlap")
        return {"performed": True, "passed": True, "modeled_heavy_atoms": len(rebuilt)}
