"""Bounded voxel paths through the complement of expanded protein spheres."""

import numpy as np

from scipy import sparse
from itertools import product
from scipy.spatial import cKDTree
from scipy.ndimage import maximum_filter
from scipy.sparse.csgraph import dijkstra


class AccessibleSpace:
    """Own one task-independent physical free-space domain, never ligand coordinates."""

    def __init__(self, atoms: np.ndarray, radii: np.ndarray, probe: float = 1.4,
                 spacing: float = 1.0, maximum_voxels: int = 2000000,
                 hull_directions: int = 96, grid_offset: tuple[float, ...] = (0., 0., 0.)) -> None:
        """Sample weighted-sphere clearance and a finite outer convex reference.

        Args:
            atoms: Selected immutable heavy/all protein atom centers [N,3], in Å.
            radii: Physical van der Waals radii [N], in Å; unequal radii are respected.
            probe: Excluded solvent-center radius in Å, default 1.4.
            spacing: Voxel side length h in Å, default one.
            maximum_voxels: Hard allocation guard, never silently changes resolution.
            hull_directions: Number of supporting half-spaces of the expanded spheres.
            grid_offset: Three fractions of h shifting only the voxel origin.

        Raises:
            MemoryError: Bounding box exceeds the explicit voxel budget.
            ValueError: Empty atoms or invalid physical lengths.
        """
        if not len(atoms) or spacing <= 0 or probe <= 0 or maximum_voxels < 1:
            raise ValueError("free-space geometry requires atoms and positive physical budgets")
        self.atoms = np.asarray(atoms, float)
        self.radii = np.asarray(radii, float)
        self.probe, self.spacing = probe, spacing
        self.tree = cKDTree(self.atoms)
        self.origin = (self.atoms - self.radii[:, None] - probe).min(0) - 2 * spacing
        self.origin += np.asarray(grid_offset) * spacing
        upper = (self.atoms + self.radii[:, None] + probe).max(0) + 2 * spacing
        self.shape = tuple((np.ceil((upper - self.origin) / spacing).astype(int) + 1).tolist())
        count = int(np.prod(self.shape))
        if count > maximum_voxels:
            raise MemoryError(f"free-space box needs {count} voxels, limit={maximum_voxels}; "
                              "increase the declared budget or choose a coarser explicit spacing")
        self.points = self.origin + np.indices(self.shape).reshape(3, -1).T * spacing
        self.clearance = self.gaps(self.points)
        self.free = self.clearance > probe

        # Intersect supporting half-spaces of ALL expanded atomic balls. This is a finite outer
        # approximation of their convex hull, independent of the historical surface sampling.

        index = np.arange(hull_directions)
        z = 1 - 2 * (index + .5) / hull_directions
        phi = index * np.pi * (3 - np.sqrt(5))
        self.directions = np.column_stack((np.sqrt(1 - z*z) * np.cos(phi),
            np.sqrt(1 - z*z) * np.sin(phi), z))
        self.support = np.max(self.atoms @ self.directions.T +
                              (self.radii + probe)[:, None], axis=0)
        self.outside = self.outside_hull(self.points)
        self.targets = self.free & self.outside
        self.distance: np.ndarray | None = None

    def gaps(self, points: np.ndarray) -> np.ndarray:
        """Evaluate min_a(||z-x_a||-r_a) exactly using a radius-aware KD-tree bound.

        Args:
            points: Query centers [Q,3], Å, in the unchanged protein coordinate frame.

        Returns:
            Signed clearance c0 [Q], Å. Nearest atom CENTER alone is not sufficient for
            unequal radii; all centers capable of improving the nearest-center bound are tested.
        """
        result = np.empty(len(points), float)
        for start in range(0, len(points), 4096):
            query = points[start:start + 4096]
            distance, nearest = self.tree.query(query)
            bound = distance - self.radii[nearest] + self.radii.max() + 1e-10
            neighbors = self.tree.query_ball_point(query, np.maximum(bound, 0))
            lengths = np.asarray([len(row) for row in neighbors])
            owner = np.repeat(np.arange(len(query)), lengths)
            ids = np.concatenate(neighbors).astype(int)
            gap = np.linalg.norm(query[owner] - self.atoms[ids], axis=1) - self.radii[ids]
            values = np.full(len(query), np.inf)
            np.minimum.at(values, owner, gap)
            result[start:start + len(query)] = values
        return result

    def outside_hull(self, points: np.ndarray) -> np.ndarray:
        """Classify query centers outside at least one supporting half-space.

        Args:
            points: Physical query coordinates [Q,3], Å.

        Returns:
            Boolean exterior-reference membership [Q], using a 1e-9 Å numerical margin.
        """
        return np.concatenate([np.any(points[i:i+4096] @ self.directions.T >
                                     self.support + 1e-9, axis=1)
                               for i in range(0, len(points), 4096)])

    def visible(self, start: np.ndarray, end: np.ndarray) -> np.ndarray:
        """Check whole line segments against expanded spheres, including diagonal corner cuts.

        Args:
            start: Free segment origins [Q,3], Å.
            end: Corresponding destination centers [Q,3], Å.

        Returns:
            True only when every point of each closed segment stays outside expanded spheres.
            No dense point-by-atom matrix is materialized.
        """
        result = np.ones(len(start), bool)
        expanded = self.radii + self.probe
        for i in range(0, len(start), 4096):
            a, b = start[i:i+4096], end[i:i+4096]
            direction = b-a
            half = np.linalg.norm(direction, axis=1)/2
            candidates = self.tree.query_ball_point((a+b)/2, half + expanded.max())
            lengths = np.asarray([len(row) for row in candidates])
            if not lengths.sum():
                continue
            owner = np.repeat(np.arange(len(a)), lengths)
            ids = np.concatenate(candidates).astype(int)
            offset = self.atoms[ids] - a[owner]
            t = (offset*direction[owner]).sum(1) / np.maximum(
                (direction[owner]**2).sum(1), 1e-20)
            closest = a[owner] + np.clip(t, 0, 1)[:, None] * direction[owner]
            blocked = np.linalg.norm(closest-self.atoms[ids], axis=1) < expanded[ids] - 1e-9
            result[i + owner[blocked]] = False
        return result

    def paths(self) -> np.ndarray:
        """Solve multi-source Dijkstra on a symmetric 26-neighbor free-center graph.

        Returns:
            Distance to the exterior reference [V], Å; infinity means no represented path.
            Each edge is an actual unobstructed segment, not just two unoccupied endpoints.
            The finite stencil is anisotropic; reducing h does not make it exactly Euclidean.
        """
        nodes = np.arange(len(self.points)).reshape(self.shape)
        source_parts, target_parts, weight_parts = [], [], []
        for delta in product((-1, 0, 1), repeat=3):
            if delta <= (0, 0, 0):
                continue
            left = tuple(slice(max(0, -d), n-max(0, d))
                         for d, n in zip(delta, self.shape, strict=True))
            right = tuple(slice(max(0, d), n-max(0, -d))
                          for d, n in zip(delta, self.shape, strict=True))
            a, b = nodes[left].ravel(), nodes[right].ravel()
            mask = self.free[a] & self.free[b]
            a, b = a[mask], b[mask]
            length = float(np.linalg.norm(delta) * self.spacing)
            # Clearance is 1-Lipschitz. Only segments near the wall need exact sphere tests;
            # both endpoints above length/2 guarantee the entire segment is free.

            near = np.minimum(self.clearance[a], self.clearance[b]) <= self.probe + length/2
            keep = np.ones(len(a), bool)
            keep[near] = self.visible(self.points[a[near]], self.points[b[near]])
            a, b = a[keep], b[keep]
            source_parts.append(a)
            target_parts.append(b)
            weight_parts.append(np.full(len(a), length))
        a, b, weights = map(np.concatenate, (source_parts, target_parts, weight_parts))
        graph = sparse.coo_matrix((weights, (a, b)), shape=(len(nodes.ravel()),)*2).tocsr()
        graph = graph + graph.T
        if not self.targets.any():
            raise ValueError("free-space box has no exterior-reference cells")
        self.distance = dijkstra(graph, directed=False,
                                 indices=np.flatnonzero(self.targets), min_only=True)
        return self.distance

    def centers(self) -> np.ndarray:
        """Find discrete clearance maxima strictly inside the bounded convex reference.

        Returns:
            Grid indices of local maxima in the probe-free domain. Exterior bulk solvent is
            excluded, and later association must still verify wall distance and line of sight.
        """
        local = maximum_filter(self.clearance.reshape(self.shape), size=3, mode="nearest").ravel()
        return np.flatnonzero(self.free & ~self.outside & (self.clearance >= local - 1e-10))
