"""Physical residue geometry, exact minimum atom contacts and residue-to-surface candidates."""

import numpy as np

from pathlib import Path
from scipy.spatial import cKDTree
from collections.abc import Mapping
from wisdom.utils.structure.enums.AtomRole import AtomRole
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.graphs.ResidueRelation import ResidueRelation


class ResidueGraph:
    """Derive task-independent residue topology without new atom or surface geometry."""

    FAMILY = "residue_graph"

    def __init__(self, contact_radius: float = 8.0, surface_radius: float = 12.0,
                 ca_radius: float | None = None, ca_k_max: int = 32) -> None:
        """Choose positive minimum-heavy-atom contact cutoffs.

        Args:
            contact_radius: Residue-pair minimum atom distance threshold in Å.
            surface_radius: Surface-to-residue minimum atom threshold in Å; no atom K truncation.
            ca_radius: Optional independent C-alpha contact radius in Å. None retains the exact
                historical heavy-contact asset; it never interprets heavy distances as CA ones.
            ca_k_max: Maximum directed CA neighbors before union symmetrization, default 32.
        """
        if contact_radius <= 0 or surface_radius <= 0:
            raise ValueError("residue graph radii must be positive")
        self.parameters = {"contact_radius": contact_radius, "surface_radius": surface_radius}
        if ca_radius is not None:
            if ca_radius <= 0 or ca_k_max < 1:
                raise ValueError("CA graph radius and stored neighbor budget must be positive")
            self.parameters.update(ca_radius=ca_radius, ca_k_max=ca_k_max)

    def build(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Construct one node per physical residue and exact untruncated sparse relations.

        Args:
            arrays: Schema-3 atoms, parents, covalent topology and surface positions.

        Returns:
            Residue coordinates/masks, undirected edges, closest atom pairs and relation bits
            (1=peptide, 2=spatial, 4=disulfide, 8=other inter-residue covalent), plus surface CSR
            with one entry per residue and its closest real heavy atom. Distance ties use atom ID.
            CA and side-chain centroid remain masked when unavailable, never silently substituted.
        """
        xyz = arrays["atom_positions"].astype(np.float64)
        parent = arrays["residue_indices"]
        _, first = np.unique(parent, return_index=True)
        count = len(first)
        names = arrays["atom_names"].astype(str)
        heavy = arrays["atomic_numbers"] > 1
        chain = arrays["chain_indices"][first]
        backbone = np.zeros((count, 4, 3), np.float32)
        backbone_valid = np.zeros((count, 4), np.bool_)
        centroids = np.zeros((count, 2, 3), np.float32)
        centroid_valid = np.zeros((count, 2), np.bool_)
        polymers = np.zeros(count, np.bool_)
        for residue in range(count):
            ids = np.flatnonzero((parent == residue) & heavy)
            polymers[residue] = bool(np.any(np.isin(arrays["atom_role_ids"][ids],
                                                   [AtomRole.BACKBONE, AtomRole.SIDECHAIN])))
            for j, name in enumerate(("N", "CA", "C", "O")):
                match = ids[names[ids] == name]
                if len(match) == 1 and polymers[residue]:
                    backbone[residue, j] = xyz[match[0]]
                    backbone_valid[residue, j] = True
            sidechain = ids[~np.isin(names[ids], ["N", "CA", "C", "O", "OXT"])]
            if not polymers[residue]:
                sidechain = np.empty(0, np.int64)
            for j, subset in enumerate((ids, sidechain)):
                if len(subset):
                    centroids[residue, j] = xyz[subset].mean(axis=0)
                    centroid_valid[residue, j] = True

        # Every spatial pair comes from actual heavy atoms, not CA proxies or truncated atom
        # neighbourhoods. Retain all inter-residue covalent pairs even outside the spatial radius.

        heavy_ids = np.flatnonzero(heavy)
        tree = cKDTree(xyz[heavy_ids])
        candidates = tree.query_pairs(self.parameters["contact_radius"], output_type="ndarray")
        spatial = heavy_ids[candidates].reshape(-1, 2)
        covalent = arrays["atom_edge_index"][:, arrays["atom_edge_is_covalent"].astype(bool)].T
        grouped: dict[tuple[int, int], tuple[float, int, int, int]] = {}
        for source, pairs in ((ResidueRelation.SPATIAL, spatial),
                              (ResidueRelation.OTHER_COVALENT, covalent)):
            for a, b in pairs:
                ra, rb = int(parent[a]), int(parent[b])
                if ra == rb or not heavy[a] or not heavy[b]:
                    continue
                if ra > rb:
                    ra, rb, a, b = rb, ra, b, a
                bit = source
                if source == ResidueRelation.OTHER_COVALENT and polymers[ra] and polymers[rb] \
                        and chain[ra] == chain[rb] \
                        and {names[a], names[b]} == {"C", "N"}:
                    bit = ResidueRelation.PEPTIDE
                if source == ResidueRelation.OTHER_COVALENT and names[a] == names[b] == "SG" and \
                        arrays["atomic_numbers"][a] == arrays["atomic_numbers"][b] == 16:
                    bit = ResidueRelation.DISULFIDE
                distance = float(np.linalg.norm(xyz[a] - xyz[b]))
                previous = grouped.get((ra, rb), (float("inf"), 0, 0, 0))
                closest = min((distance, int(a), int(b)), previous[:3])
                grouped[(ra, rb)] = (*closest, previous[3] | bit)
        for key, (distance, _a, _b, relation) in grouped.items():
            if distance > self.parameters["contact_radius"]:
                left = np.flatnonzero((parent == key[0]) & heavy)
                right = np.flatnonzero((parent == key[1]) & heavy)
                distances, nearest_atom = cKDTree(xyz[right]).query(xyz[left], k=1)
                closest = min((float(d), int(i), int(right[j])) for d, i, j in
                              zip(distances, left, nearest_atom, strict=True))
                grouped[key] = (*closest, relation)
        keys = sorted(grouped)
        edge = np.asarray(keys, np.int32).reshape(-1, 2).T
        contact_atoms = np.asarray([grouped[key][1:3] for key in keys], np.int32).reshape(-1, 2).T

        # Direct radius queries enumerate all real heavy atoms for each point before deduplicating
        # physical residue IDs. A large residue cannot receive extra weight simply by having atoms.

        pointer = [0]
        residues, closest_atoms, distances = [], [], []
        for point in arrays["surface_positions"].astype(np.float64):
            ids = heavy_ids[tree.query_ball_point(point, self.parameters["surface_radius"])]
            nearest: dict[int, tuple[float, int]] = {}
            for atom in ids:
                residue = int(parent[atom])
                value = (float(np.linalg.norm(xyz[atom] - point)), int(atom))
                nearest[residue] = min(value, nearest.get(residue, (float("inf"), -1)))
            ordered = sorted(nearest.items(), key=lambda row: (*row[1], row[0]))
            for residue, (distance, atom) in ordered:
                residues.append(residue)
                closest_atoms.append(atom)
                distances.append(distance)
            pointer.append(len(residues))
        result = {
            "residue_chain": chain.astype(np.int32),
            "residue_type": arrays["residue_type_ids"][first].astype(np.int16),
            "atom_residue": parent.astype(np.int32),
            "backbone_positions": backbone, "backbone_valid": backbone_valid,
            "centroid_positions": centroids, "centroid_valid": centroid_valid,
            "edge_index": edge, "edge_atom_index": contact_atoms,
            "edge_distance": np.asarray([grouped[key][0] for key in keys], np.float32),
            "edge_relation": np.asarray([grouped[key][3] for key in keys], np.uint8),
            "surface_ptr": np.asarray(pointer, np.int64),
            "surface_residue": np.asarray(residues, np.int32),
            "surface_atom": np.asarray(closest_atoms, np.int32),
            "surface_distance": np.asarray(distances, np.float32),
        }
        if "ca_radius" in self.parameters:
            result.update(self.ca_geometry(backbone, backbone_valid, edge, result["edge_relation"]))
        return result

    def ca_geometry(self, backbone: np.ndarray, valid: np.ndarray,
                    chemical_edges: np.ndarray, relations: np.ndarray) -> dict[str, np.ndarray]:
        """Build independent bounded CA queries, preserving chemically verified continuity.

        Args:
            backbone: N/CA/C/O coordinates [R,4,3], in the base protein's centered Å frame.
            valid: Observed-atom masks [R,4]; absent CA nodes never receive invented coordinates.
            chemical_edges: Historical heavy-contact/covalent pairs [2,E], src < dst.
            relations: Peptide/spatial/disulfide/other-covalent bit masks [E].

        Returns:
            CA pairs once, distances, endpoint-specific spatial ranks and chemical relation bits;
            backbone frames [R,3,3] and explicit validity. Union KNN does not cap incoming degree.
            Peptide bonds survive the spatial cutoff, while invalid CA nodes have no new edges.
        """
        xyz = backbone[:, 1].astype(np.float64)
        nodes = np.flatnonzero(valid[:, 1])
        pairs: dict[tuple[int, int], list[int]] = {}
        if len(nodes):
            tree = cKDTree(xyz[nodes])
            radius = float(self.parameters["ca_radius"])
            maximum = int(self.parameters["ca_k_max"])
            for local, neighbors in enumerate(tree.query_ball_point(xyz[nodes], radius)):
                source = int(nodes[local])
                ranked = sorted((float(np.linalg.norm(xyz[source] - xyz[nodes[j]])),
                                 int(nodes[j])) for j in neighbors if j != local)[:maximum]
                for rank, (_, destination) in enumerate(ranked, 1):
                    key = (min(source, destination), max(source, destination))
                    row = pairs.setdefault(key, [0, 0, 0])
                    row[0 if source < destination else 1] = rank
        for (a, b), relation in zip(chemical_edges.T, relations, strict=True):
            chemical = int(relation) & ~int(ResidueRelation.SPATIAL)
            if chemical and valid[a, 1] and valid[b, 1]:
                pairs.setdefault((int(a), int(b)), [0, 0, 0])[2] |= chemical
        keys = sorted(pairs)
        edges = np.asarray(keys, np.int32).reshape(-1, 2).T
        delta = xyz[edges[0]] - xyz[edges[1]]

        # A local frame is observed only when N/CA/C exist and Gram-Schmidt is well conditioned.
        # Its axes rotate with the molecule; no global XYZ coordinate becomes a scalar feature.

        first = backbone[:, 2].astype(float) - xyz
        second = backbone[:, 0].astype(float) - xyz
        lengths = np.linalg.norm(first, axis=1)
        first /= np.maximum(lengths[:, None], 1e-12)
        second -= np.einsum("ij,ij->i", first, second)[:, None] * first
        orthogonal = np.linalg.norm(second, axis=1)
        frame_valid = valid[:, :3].all(axis=1) & (lengths > 1e-6) & (
            orthogonal > 1e-6 * np.maximum(lengths, 1))
        second /= np.maximum(orthogonal[:, None], 1e-12)
        frames = np.stack((first, second, np.cross(first, second)), axis=1)
        frames[~frame_valid] = 0
        return {"ca_edge_index": edges,
                "ca_edge_distance": np.linalg.norm(delta, axis=1).astype(np.float32),
                "ca_edge_rank": np.asarray([pairs[key][:2] for key in keys],
                                           np.int16).T.reshape(2, -1),
                "ca_edge_relation": np.asarray([pairs[key][2] for key in keys], np.uint8),
                "backbone_frame": frames.astype(np.float32), "frame_valid": frame_valid}

    def write(self, base: Path, output: Path) -> None:
        """Persist physical geometry and index correspondences in an independent asset.

        Args:
            base: Immutable schema-3 archive.
            output: LF-managed temporary path.
        """
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files if key != "metadata_json"}
        StructuralAsset.write(output, base, self.build(arrays), self.FAMILY, self.parameters,
            {"algorithm": "minimum-heavy-atom-contact-v1", "units": "angstrom",
             "backbone_order": ["N", "CA", "C", "O"],
             "centroid_order": ["unweighted-heavy", "unweighted-sidechain-heavy"],
             "relations": {"peptide": 1, "spatial": 2, "disulfide": 4, "other_covalent": 8}})

    def validate(self, path: Path, base: Path) -> bool:
        """Verify exact contacts, masks, parent indices and untruncated residue candidates.

        Args:
            path: Persisted residue graph asset.
            base: Exact structural input.

        Returns:
            True after deterministic physical reconstruction.

        Raises:
            ValueError: Configuration, correspondence or numerical mismatch.
        """
        actual, header = StructuralAsset.read(path, base, self.FAMILY)
        if header["parameters"] != self.parameters or header.get("units") != "angstrom" or \
                header.get("algorithm") != "minimum-heavy-atom-contact-v1":
            raise ValueError("residue graph configuration mismatch")
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files if key != "metadata_json"}
        StructuralAsset.compare(actual, self.build(arrays))
        return True
