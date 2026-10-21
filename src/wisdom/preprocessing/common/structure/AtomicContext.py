"""Compact optional chemistry and conformation derived from immutable universal geometry."""

import json
import numpy as np

from pathlib import Path
from itertools import pairwise, combinations
from collections.abc import Mapping
from wisdom.utils.structure.models.Protein import Protein
from wisdom.preprocessing.common.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.context_chemistry import (
    CHI_PATHS, NAME_IDS, POSITIONS, ATOM_NAMES, EXACT_IDS, EXACT_ATOMS, POSITION_IDS,
    FUNCTIONAL_ATOMS, FUNCTIONAL_GROUPS, SIDECHAIN_POSITION_IDS,
)


class AtomicContext:
    """Own the independently versioned atom/residue context contract, never task labels."""

    VERSION = "1.0"

    def __init__(self, source_quality: bool = True) -> None:
        """Select whether verified deposited quality fields should be retained.

        Args:
            source_quality: Include observed occupancy/B/selected-altloc when available; these
                are metadata, not automatically activated neural features.
        """
        self.source_quality = source_quality

    @staticmethod
    def torsion(points: np.ndarray) -> tuple[float, bool]:
        """Compute the signed IUPAC-style dihedral about the middle bond.

        Args:
            points: Four ordered float coordinates [4,3], in one Cartesian frame (Å).

        Returns:
            (angle radians in [-pi,pi], valid). Project b0=x0-x1 and b2=x3-x2 onto
            the plane normal to b1=x2-x1; atan2((u cross v) dot w, v dot w), u=b1/|b1|.
            Zero-length bonds or collinear defining triples return (0,false), never valid zero.
        """
        a, b, c, d = np.asarray(points, dtype=np.float64)
        axis = c - b
        length = np.linalg.norm(axis)
        if length < 1e-10:
            return 0.0, False
        axis /= length
        left  = (a - b) - np.dot(a - b, axis) * axis
        right = (d - c) - np.dot(d - c, axis) * axis
        if min(np.linalg.norm(left), np.linalg.norm(right)) < 1e-10:
            return 0.0, False
        return float(np.arctan2(np.dot(np.cross(axis, left), right), np.dot(left, right))), True

    def build(
        self, arrays: Mapping[str, np.ndarray], protein: Protein | None = None,
    ) -> dict[str, np.ndarray]:
        """Derive stable categories, masked torsions and represented heavy-atom chemistry.

        Args:
            arrays: Validated schema-3 base arrays. Atom indices/order are authoritative.
            protein: Optional exact-source, identically filtered and coordinate-aligned hierarchy.
                The caller must already verify its source fingerprint before providing it.

        Returns:
            Compact atom categories, residue table [R], masked torsions [R,7] (phi,psi,
            omega,chi1..4), heavy degree, multilabel motif bits, and optional source quality.
            Original atom names/residue names/atom→residue arrays are not duplicated.

        Raises:
            ValueError: Nonconsecutive residue indices or mismatched source atom identity/order.
        """
        names     = arrays["atom_names"].astype(str)
        residues  = arrays["residue_names"].astype(str)
        positions = arrays["atom_positions"].astype(np.float64)
        indices   = arrays["residue_indices"]
        chains    = arrays["chain_indices"]
        n         = len(names)
        r         = int(indices.max()) + 1
        if not np.array_equal(np.unique(indices), np.arange(r)):
            raise ValueError("atomic context requires consecutive residue indices")
        grouped = [np.flatnonzero(indices == i) for i in range(r)]
        first   = np.asarray([group[0] for group in grouped])
        named   = [{names[j]: int(j) for j in group} for group in grouped]
        standard = np.asarray([any((residues[j], names[j]) in EXACT_IDS for j in group)
                               for group in grouped], dtype=np.bool_)

        covalent = arrays["atom_edge_is_covalent"] & (arrays["atom_edge_bond_order"] > 0)
        pairs    = arrays["atom_edge_index"][:, covalent].T
        bonds    = {tuple(map(int, pair)) for pair in pairs}
        adjacency: list[set[int]] = [set() for _ in range(n)]
        for a, b in pairs:
            adjacency[a].add(int(b))
            adjacency[b].add(int(a))

        # Peptide neighbors come from actual named C-N covalent edges, never ordinal residue
        # IDs or author-number arithmetic. Ambiguous branching leaves the torsion unavailable.

        previous: dict[int, list[int]] = {}
        following: dict[int, list[int]] = {}
        for a, b in pairs:
            if names[a] == "N" and names[b] == "C":
                a, b = b, a
            ra, rb = int(indices[a]), int(indices[b])
            if names[a] == "C" and names[b] == "N" and ra != rb and chains[a] == chains[b]:
                previous.setdefault(rb, []).append(ra)
                following.setdefault(ra, []).append(rb)

        angles = np.zeros((r, 7), dtype=np.float32)
        valid  = np.zeros((r, 7), dtype=np.bool_)
        applicable = np.zeros((r, 7), dtype=np.bool_)
        for i, lookup in enumerate(named):
            residue = residues[first[i]]
            if not standard[i]:
                continue
            applicable[i, :3] = True
            chi = CHI_PATHS.get(residue, ())
            applicable[i, 3:3 + len(chi)] = True
            paths: list[tuple[int, list[int | None]]] = [
                (j + 3, [lookup.get(name) for name in path.split()])
                for j, path in enumerate(chi)]
            before, after = previous.get(i, []), following.get(i, [])
            if len(before) == 1 and standard[before[0]]:
                paths.append((0, [named[before[0]].get("C"), lookup.get("N"),
                                  lookup.get("CA"), lookup.get("C")]))
            if len(after) == 1 and standard[after[0]]:
                other = named[after[0]]
                paths.extend([(1, [lookup.get("N"), lookup.get("CA"), lookup.get("C"),
                                   other.get("N")]),
                              (2, [lookup.get("CA"), lookup.get("C"), other.get("N"),
                                   other.get("CA")])])
            for column, path in paths:
                if any(j is None for j in path):
                    continue
                ids = [int(j) for j in path if j is not None]
                if not all(tuple(sorted((a, b))) in bonds for a, b in pairwise(ids)):
                    continue
                angles[i, column], valid[i, column] = self.torsion(positions[ids])

        # Ring membership is topology-derived: an edge lies on a cycle iff its endpoints remain
        # connected after removing that edge. Restrict searches to canonical residue templates;
        # unknown components keep an explicit unknown mask, not a guessed aromatic assignment.

        ring = np.zeros(n, dtype=np.bool_)
        for group in grouped:
            allowed = set(map(int, group))
            if not standard[int(indices[group[0]])]:
                continue
            for a in group:
                for b in adjacency[a] & allowed:
                    if a >= b:
                        continue
                    seen, queue = {int(a)}, [int(a)]
                    while queue:
                        current = queue.pop()
                        for target in adjacency[current] & allowed:
                            if {current, target} == {a, b} or target in seen:
                                continue
                            seen.add(target)
                            queue.append(target)
                    if b in seen:
                        ring[a] = ring[b] = True

        motif = np.zeros(n, dtype=np.uint16)
        disulfide_atoms = {int(atom) for a, b in pairs
            if arrays["atomic_numbers"][a] == arrays["atomic_numbers"][b] == 16
            and names[a] == names[b] == "SG" and residues[a] == residues[b] == "CYS"
            for atom in (a, b)}
        known = np.asarray([(res, name) in EXACT_IDS for res, name in
                            zip(residues, names, strict=True)], dtype=np.bool_)
        for i, (residue, name) in enumerate(zip(residues, names, strict=True)):
            if not known[i]:
                continue
            groups = {key for key, table in FUNCTIONAL_ATOMS.items()
                      if name in table.get(residue, "").split()}
            peptide_carbon = i if name == "C" else named[int(indices[i])].get("C")
            peptide_amide = (name == "N" and any(names[j] == "C" and indices[j] != indices[i]
                for j in adjacency[i])) or (name in {"C", "O"} and peptide_carbon is not None
                and any(names[j] == "N" and indices[j] != indices[i]
                        for j in adjacency[peptide_carbon]))
            if peptide_amide:
                groups.add("backbone_amide")
            if name in {"C", "O"} or (residue, name) in {
                ("ASN", "CG"), ("ASN", "OD1"), ("GLN", "CD"), ("GLN", "OE1")}:
                groups.add("carbonyl")
            if i in disulfide_atoms:
                groups.add("disulfide")
                groups.discard("thiol")
            for motif_name in groups:
                motif[i] |= np.uint16(1 << FUNCTIONAL_GROUPS.index(motif_name))

        values = {
            "atom_name_id": np.asarray([NAME_IDS.get(name, 0) for name in names], np.uint16),
            "atom_identity_id": np.asarray([EXACT_IDS.get((res, name), 0) for res, name in
                                            zip(residues, names, strict=True)], np.uint16),
            "atom_position_id": np.asarray([POSITION_IDS.get(name,
                SIDECHAIN_POSITION_IDS.get(name[1:2], 0)) if known[i] else 0
                for i, name in enumerate(names)], np.uint8),
            "residue_chain_index": chains[first].astype(np.int16),
            "torsion": angles, "torsion_valid": valid, "torsion_applicable": applicable,
            "heavy_degree": np.asarray([sum(arrays["atomic_numbers"][j] > 1 for j in neighbors)
                                        for neighbors in adjacency], np.uint16),
            "ring_member": ring, "chemistry_known": known, "functional_group_bits": motif,
            "author_number": np.zeros(r, np.int32),
            "insertion_code": np.full(r, "", dtype="U4"),
            "is_polymer": np.zeros(r, np.bool_), "author_valid": np.zeros(r, np.bool_),
            "chain_author_id": np.full(int(chains.max()) + 1, "", dtype="U32"),
        }
        if self.source_quality:
            values.update(occupancy=np.zeros(n, np.float32), occupancy_valid=np.zeros(n, np.bool_),
                          b_iso=np.zeros(n, np.float32), b_iso_valid=np.zeros(n, np.bool_),
                          selected_altloc=np.full(n, "", dtype="U4"),
                          alternate_count=np.zeros(n, np.uint16))
        if protein is not None:
            self.source_fields(values, arrays, protein)
        return values

    def source_fields(
        self, values: dict[str, np.ndarray], arrays: Mapping[str, np.ndarray], protein: Protein,
    ) -> None:
        """Attach source-only fields only after verifying all normalized atom correspondences.

        Args:
            values: Derived context arrays to update in place.
            arrays: Exact base positions/names/elements and parent indices.
            protein: Hierarchy read from the digest-verified source using identical filters.

        Raises:
            ValueError: Any difference in atom names/elements/positions or parent ownership.
        """
        atoms = [a for chain in protein.chains for residue in chain.residues for a in residue.atoms]
        residues = [residue for chain in protein.chains for residue in chain.residues]
        parent = [i for i, residue in enumerate(residues) for _ in residue.atoms]
        chain_indices = [i for i, chain in enumerate(protein.chains)
                         for residue in chain.residues for _ in residue.atoms]
        checks = (
            np.array_equal([a.index for a in atoms], np.arange(len(atoms))),
            np.array_equal([a.name for a in atoms], arrays["atom_names"]),
            np.array_equal([a.atomic_number for a in atoms], arrays["atomic_numbers"]),
            np.array_equal([r.name for r in residues for _ in r.atoms], arrays["residue_names"]),
            np.array_equal(parent, arrays["residue_indices"]),
            np.array_equal(chain_indices, arrays["chain_indices"]),
            np.array_equal(np.asarray([a.position for a in atoms], np.float32),
                           arrays["atom_positions"]),
        )
        if not all(checks):
            raise ValueError("atomic context source does not match base atom order/geometry")
        if any(len(r.insertion_code) > 4 for r in residues) or any(
            len(chain.id) > 32 for chain in protein.chains):
            raise ValueError("source identifiers exceed the atomic context string schema")
        values["author_number"][:] = [r.number for r in residues]
        values["insertion_code"][:] = [r.insertion_code for r in residues]
        values["is_polymer"][:] = [r.is_polymer for r in residues]
        values["author_valid"][:] = True
        values["chain_author_id"][:] = [chain.id for chain in protein.chains]
        if self.source_quality:
            for i, atom in enumerate(atoms):
                for name, value in (("occupancy", atom.occupancy), ("b_iso", atom.b_iso)):
                    if value is not None and np.isfinite(value) and value >= 0 and (
                        name != "occupancy" or value <= 1):
                        values[name][i] = value
                        values[f"{name}_valid"][i] = True
                values["selected_altloc"][i] = atom.altloc
                values["alternate_count"][i] = atom.alternate_count

    def write(self, base: Path, output: Path, protein: Protein | None = None) -> None:
        """Serialize context linked to one exact base, with explicit conventions and vocabularies.

        Args:
            base: Immutable validated schema-3 universal NPZ.
            output: Managed temporary destination; no base bytes are rewritten.
            protein: Optional source-verified hierarchy; otherwise source fields are unavailable.
        """
        with np.load(base, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
            original = json.loads(str(archive["metadata_json"].item()))
        if original["preprocessing_schema_version"] != "3.0":
            raise ValueError("atomic context requires structural schema 3.0")
        values = self.build(arrays, protein)
        metadata = {
            "schema": self.VERSION, "algorithm": "canonical-context-v1",
            "source_quality": self.source_quality, "source_aligned": protein is not None,
            "source_hash": original.get("source_hash"), "config_hash": original.get("config_hash"),
            "angle_unit": "radian",
            "torsions": ["phi", "psi", "omega", "chi1", "chi2", "chi3", "chi4"],
            "chi_templates": CHI_PATHS, "name_vocabulary": ["UNK", *ATOM_NAMES],
            "identity_vocabulary": ["UNK", *[f"{r}:{a}" for r, a in EXACT_ATOMS]],
            "position_vocabulary": POSITIONS,
            "functional_groups": FUNCTIONAL_GROUPS,
            "missing": "zero placeholder plus false valid mask; not observed zero",
            "b_iso_unit": "angstrom^2", "numpy_version": np.__version__,
            "atom_count": len(arrays["atom_positions"]),
            "residue_count": len(values["residue_chain_index"]),
            "chain_count": len(values["chain_author_id"]),
        }
        with output.open("wb") as stream:
            np.savez_compressed(stream, **values,  # type: ignore[arg-type]
                base_npz_sha256=np.asarray(ProteinArchive.sha256(base)),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)))

    def validate(self, path: Path, base: Path, protein: Protein | None = None) -> bool:
        """Verify base identity, every derived field and source-field validity semantics.

        Args:
            path: Context NPZ loaded without pickle.
            base: Exact universal NPZ whose atom/residue indices are authoritative.
            protein: Optional verified source hierarchy to additionally check observed fields.

        Returns:
            True after deterministic scientific reconstruction and source checks.

        Raises:
            ValueError: Wrong schema/digest, unexpected fields, invalid masks or numerical data.
        """
        with np.load(path, allow_pickle=False) as archive:
            values = {name: archive[name] for name in archive.files}
        digest = str(values.pop("base_npz_sha256").item())
        metadata = json.loads(str(values.pop("metadata_json").item()))
        if digest != ProteinArchive.sha256(base):
            raise ValueError("atomic context base NPZ hash mismatch")
        if metadata["schema"] != self.VERSION or metadata["source_quality"] != self.source_quality:
            raise ValueError("atomic context schema/configuration mismatch")
        vocabulary = {
            "name_vocabulary": ["UNK", *ATOM_NAMES],
            "identity_vocabulary": ["UNK", *[f"{r}:{a}" for r, a in EXACT_ATOMS]],
            "position_vocabulary": list(POSITIONS), "functional_groups": list(FUNCTIONAL_GROUPS),
            "algorithm": "canonical-context-v1", "angle_unit": "radian",
        }
        if any(metadata.get(key) != value for key, value in vocabulary.items()):
            raise ValueError("atomic context vocabulary/convention mismatch")
        with np.load(base, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
            original = json.loads(str(archive["metadata_json"].item()))
        if any(metadata.get(key) != original.get(key) for key in ("source_hash", "config_hash")):
            raise ValueError("atomic context source/config provenance mismatch")
        if metadata.get("chi_templates") != {key: list(paths) for key, paths in CHI_PATHS.items()}:
            raise ValueError("atomic context chi convention mismatch")
        expected = self.build(arrays, protein)
        if metadata.get("atom_count") != len(arrays["atom_positions"]) or metadata.get(
            "residue_count") != len(expected["residue_chain_index"]) or metadata.get(
                "chain_count") != len(expected["chain_author_id"]):
            raise ValueError("atomic context support counts mismatch")
        if set(values) != set(expected):
            raise ValueError("atomic context array schema mismatch")
        source = {"author_number", "insertion_code", "is_polymer", "author_valid",
                  "chain_author_id",
                  "occupancy", "occupancy_valid", "b_iso", "b_iso_valid", "selected_altloc",
                  "alternate_count"}
        for name, actual in values.items():
            wanted = expected[name]
            if actual.dtype != wanted.dtype or actual.shape != wanted.shape:
                raise ValueError(f"invalid atomic context dtype/shape: {name}")
            if actual.dtype.kind == "f" and not np.isfinite(actual).all():
                raise ValueError(f"non-finite atomic context: {name}")
            if (name not in source or protein is not None or not metadata["source_aligned"]) and (
                not np.array_equal(actual, wanted)):
                raise ValueError(f"invalid atomic context array: {name}")
        for name in ("occupancy", "b_iso"):
            if name in values and np.any(values[name][~values[f"{name}_valid"]] != 0):
                raise ValueError(f"unavailable {name} must have zero placeholder")
            if name in values and np.any(values[name] < 0):
                raise ValueError(f"invalid negative {name}")
        if "occupancy" in values and np.any(values["occupancy"] > 1):
            raise ValueError("occupancy exceeds one")
        if np.any(values["author_valid"] != bool(metadata["source_aligned"])):
            raise ValueError("source availability and author identity masks disagree")
        if self.source_quality and metadata["source_aligned"] and np.any(
            values["alternate_count"] < 1):
            raise ValueError("source-aligned atoms must retain at least one source alternative")
        return True

    @staticmethod
    def bond_angles(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Derive deterministic i-j-k covalent angles without storing another graph.

        Args:
            arrays: Base positions [N,3], undirected edges [2,E], covalent flags and bond orders.

        Returns:
            int32 triplets [T,3] with endpoint i<k and central j, float32 radians [T].
            Degenerate zero-length arms are omitted; spatial-only relations never participate.
        """
        xyz = arrays["atom_positions"].astype(np.float64)
        adjacent: list[list[int]] = [[] for _ in xyz]
        mask = arrays["atom_edge_is_covalent"] & (arrays["atom_edge_bond_order"] > 0)
        for a, b in arrays["atom_edge_index"][:, mask].T:
            adjacent[a].append(int(b))
            adjacent[b].append(int(a))
        triplets, angles = [], []
        for j, neighbors in enumerate(adjacent):
            for i, k in combinations(sorted(neighbors), 2):
                left, right = xyz[i] - xyz[j], xyz[k] - xyz[j]
                scale = np.linalg.norm(left) * np.linalg.norm(right)
                if scale > 1e-12:
                    triplets.append((i, j, k))
                    angles.append(np.arccos(np.clip(np.dot(left, right) / scale, -1, 1)))
        return np.asarray(triplets, np.int32).reshape(-1, 3), np.asarray(angles, np.float32)

    @staticmethod
    def frames(arrays: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        """Construct optional N-CA-C right-handed local frames at runtime only.

        Args:
            arrays: Base atom names/positions and consecutive residue indices.

        Returns:
            float32 frames [R,3,3] with row axes e1=(C-CA)/norm,
            e3=normalize(e1 cross (N-CA)), e2=e3 cross e1, plus bool valid [R].
            Frames rotate with the protein; e3 is a pseudovector under reflection.
        """
        residue = arrays["residue_indices"]
        frames = np.zeros((int(residue.max()) + 1, 3, 3), np.float32)
        valid = np.zeros(len(frames), np.bool_)
        covalent = {tuple(edge) for edge in arrays["atom_edge_index"][:,
            arrays["atom_edge_is_covalent"] & (arrays["atom_edge_bond_order"] > 0)].T}
        for i in range(len(frames)):
            named = {str(arrays["atom_names"][j]): j for j in np.flatnonzero(residue == i)}
            if not {"N", "CA", "C"} <= named.keys():
                continue
            if not all(tuple(sorted((named[a], named[b]))) in covalent
                       for a, b in (("N", "CA"), ("CA", "C"))):
                continue
            n, ca, c = arrays["atom_positions"][[named[k] for k in ("N", "CA", "C")]].astype(
                np.float64)
            e1 = c - ca
            e3 = np.cross(e1, n - ca)
            if min(np.linalg.norm(e1), np.linalg.norm(e3)) < 1e-10:
                continue
            e1 /= np.linalg.norm(e1)
            e3 /= np.linalg.norm(e3)
            frames[i] = (e1, np.cross(e3, e1), e3)
            valid[i] = True
        return frames, valid
