"""Physical query domains and deterministic stratified sampling independent of functional labels."""

import numpy as np

from scipy import ndimage
from scipy.spatial import cKDTree


class FieldQueries:
    """Classify a finite solver volume and retain an offline bank larger than an optimizer batch."""

    @staticmethod
    def domains(points: np.ndarray, atoms: np.ndarray, radii: np.ndarray,
                shape: tuple[int, ...], probe: float, connectivity: int = 6,
                boundary_width: float = 2.0, exterior_width: float = 6.0) -> dict[str, np.ndarray]:
        """Compute min_i(||x-c_i||-r_i), exterior connectivity and disjoint sampling strata.

        Args:
            points: Flattened Cartesian grid centers [Q,3], Å, in C-order of shape.
            atoms: Retained heavy-atom centers [N,3], Å; never ligand coordinates.
            radii: Positive WISDOM vdW radii [N], Å, not force-field probe diameters.
            shape: Three grid dimensions whose product is Q.
            probe: Solvent probe radius in Å for accessibility, not the LJ exclusion mask.
            connectivity: 6 face neighbors or 26 face/edge/corner neighbors.
            boundary_width: Nonnegative d_vdW band for stratum 0, Å.
            exterior_width: Exterior band for stratum 1, Å; remaining points enter stratum 3.

        Returns:
            Signed distance [Q], probe-free and externally accessible masks [Q], and int8
            region IDs: 0 boundary, 1 near exterior, 2 closed void, 3 other exterior, -1 interior.
            Cavities are geometric voids; no biological function or observed water is inferred.
        """
        tree = cKDTree(atoms)
        nearest, nearest_id = tree.query(points)
        distance = np.empty(len(points), dtype=np.float64)
        maximum_radius = float(np.max(radii))

        # The nearest-center gap is an upper bound. Any sphere improving it must have its center
        # within nearest-center distance + max radius; query all such candidates, not top-K.

        for start in range(0, len(points), 2048):
            batch = points[start:start + 2048]
            upper_gap = nearest[start:start + len(batch)] - radii[
                nearest_id[start:start + len(batch)]]
            nearby = tree.query_ball_point(
                batch, upper_gap + maximum_radius + 1e-10)
            counts = np.fromiter(map(len, nearby), dtype=np.int64)
            indices = np.concatenate(nearby).astype(np.intp)
            owners = np.repeat(np.arange(len(batch)), counts)
            gaps = np.linalg.norm(batch[owners] - atoms[indices], axis=1) - radii[indices]
            distance[start:start + len(batch)] = np.minimum.reduceat(gaps,
                np.r_[0, np.cumsum(counts[:-1])])
        free = (distance >= probe).reshape(shape)
        if connectivity not in {6, 26}:
            raise ValueError("physical accessibility connectivity must be 6 or 26")
        stencil = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
        boundary = np.zeros(shape, dtype=bool)
        boundary[[0, -1], :, :] = True
        boundary[:, [0, -1], :] = True
        boundary[:, :, [0, -1]] = True
        outside = ndimage.binary_propagation(boundary & free, structure=stencil, mask=free).ravel()
        closed = free.ravel() & ~outside
        region = np.full(len(points), -1, dtype=np.int8)
        region[distance >= 0] = 3
        region[(distance >= 0) & (distance <= boundary_width)] = 0
        region[outside & (distance > boundary_width) & (distance <= exterior_width)] = 1
        region[closed] = 2
        return {"distance": distance.astype(np.float32), "free": free.ravel(),
                "accessible": outside, "region": region}

    @staticmethod
    def sample(regions: np.ndarray, valid: np.ndarray, maximum: int, seed: int,
               policy: str = "stratified") -> np.ndarray:
        """Select reproducible bank rows and redistribute quotas without inventing cavity rows.

        Args:
            regions: Disjoint int region IDs [Q], -1 denotes invalid interior.
            valid: Boolean field masks [Q,3], or any-field validity [Q].
            maximum: Maximum requested rows, at least one; a smaller domain yields fewer rows.
            seed: Independent query-sampling seed, never a functional label.
            policy: stratified (40/25/20/15 percent), uniform or boundary. Boundary samples its
                stratum first and fills only from remaining valid rows when it is insufficient.

        Returns:
            Unique sorted original row IDs. Sampling is without replacement and field-specific
            masks remain authoritative; a valid LJ1 row is not automatically a valid LJ2 row.
        """
        if maximum < 1 or policy not in {"uniform", "boundary", "stratified"}:
            raise ValueError("physical sampling needs a positive count and a known policy")
        rng = np.random.default_rng(seed)
        available = np.flatnonzero((valid.any(1) if valid.ndim == 2 else valid) & (regions >= 0))
        count = min(maximum, len(available))
        if policy == "uniform":
            return np.sort(rng.choice(available, count, replace=False))
        quotas = (1.0, 0.0, 0.0, 0.0) if policy == "boundary" else (.4, .25, .2, .15)
        selected: list[int] = []
        for region, quota in enumerate(quotas):
            pool = available[regions[available] == region]
            selected.extend(rng.choice(pool, min(len(pool), int(count * quota)), replace=False))
        remaining = np.setdiff1d(available, selected)
        selected.extend(rng.choice(remaining, count - len(selected), replace=False))
        return np.sort(np.asarray(selected, dtype=np.int64))
