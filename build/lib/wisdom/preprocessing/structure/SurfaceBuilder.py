"""Deterministic molecular surface point-cloud geometry."""

from __future__ import annotations

import math
from itertools import pairwise

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree


class SurfaceBuilder:
    """Generate and characterize a deterministic expanded-sphere molecular surface.

    The builder samples the solvent-accessible boundary of van der Waals spheres expanded by a
    probe radius, reduces density by deterministic voxels, estimates outward normals and two-scale
    multi-scale curvature, area weights, and temporary connectivity diagnostics.
    """

    def __init__(
        self,
        resolution       : float,
        probe_radius     : float                    = 1.4,
        curvature_scales : tuple[float, ...] | list[float] = (2.5, 5.0),
    ) -> None:
        """Set the physical/geometric length scales used by all surface operations.

        Args:
            resolution: Target point spacing ``h`` in ångströms. Candidate density, voxel side,
                curvature radii, and surface graph radius are derived from this value.
            probe_radius: Solvent probe radius added to each atomic van der Waals radius, in
                ångströms.
            curvature_scales: Positive curvature-neighborhood radius multipliers. Scale ``s`` fits
                one local quadratic inside radius ``s * resolution``.

        Raises:
            ValueError: If any length scale is not strictly positive, or curvature scales are empty,
                non-positive, or duplicated.
        """
        if resolution <= 0 or probe_radius <= 0:
            raise ValueError("surface resolution and probe radius must be positive")
        scales = tuple(float(value) for value in curvature_scales)
        if not scales or any(value <= 0 for value in scales) or len(set(scales)) != len(scales):
            raise ValueError("curvature scales must be non-empty, positive and unique")

        self.resolution       = resolution
        self.probe_radius     = probe_radius
        self.curvature_scales = scales

    def build(
        self,
        atom_positions : np.ndarray,
        vdw_radii      : np.ndarray,
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Build the complete fixed surface representation for one atom cloud.

        Atom ``i`` is expanded to ``R_i = r_i + probe_radius``. Fibonacci candidates on each sphere
        are removed when another expanded sphere contains them, reduced to one point per resolution
        voxel, assigned soft-min sphere-gradient normals, fitted with multi-scale quadratic
        curvature, weighted by local spacing, and audited through a temporary sparse radius graph.

        Args:
            atom_positions: Finite Cartesian atom coordinates with shape ``[N,3]`` in ångströms.
            vdw_radii: Van der Waals radius for each atom with shape ``[N]`` in ångströms.

        Returns:
            A dictionary of ``float32`` geometry/edge distances and compact integer topology arrays,
            plus diagnostic surface-connectivity warnings.

        Raises:
            ValueError: If no exposed candidates survive.
        """
        # Expand van der Waals spheres by the solvent probe to define the sampled SAS boundary.
        positions = np.asarray(atom_positions, dtype=np.float64)
        expanded  = np.asarray(vdw_radii, dtype=np.float64) + self.probe_radius

        # Sample, expose, and deterministically reduce the union-of-spheres boundary.
        candidates, owners = self._surface_candidates(positions, expanded)
        exposed, owners    = self._remove_buried(candidates, owners, positions, expanded)
        if len(exposed) == 0:
            raise ValueError("surface generation produced no exposed candidates")

        points, owners = self._voxel_select(exposed, owners)

        # Compute fixed local geometry and the filtered surface topology.
        normals         = self._envelope_normals(points, owners, positions, expanded)
        curvatures      = self.estimate_curvatures(points, normals)
        weights         = self.area_weights(points)
        _, warnings     = self.build_graph(points, normals)

        # Only immutable molecular geometry is published here. Trainable neighborhoods and
        # intrinsic operators are built by their dedicated stages without changing point order.
        arrays = {
            "surface_positions": points.astype(np.float32),
            "surface_normals": normals.astype(np.float32),
            "surface_curvatures": curvatures.astype(np.float32),
            "surface_area_weights": weights,
        }
        return arrays, warnings

    def fibonacci_sphere(self, count: int, phase: float = 0.0) -> np.ndarray:
        """Generate deterministic approximately uniform unit-sphere directions.

        For ``k=0,...,count-1``, the method uses
        ``z_k=1-2(k+1/2)/count``, ``rho_k=sqrt(1-z_k²)``, and
        ``theta_k=k*pi*(3-sqrt(5))+phase``. The half-step avoids poles and the golden-angle advance
        avoids latitude clustering without optimization or randomness.

        Args:
            count: Positive number of unit directions to produce.
            phase: Deterministic azimuthal offset in radians.

        Returns:
            A ``float64 [count,3]`` array of approximately uniform unit vectors.

        Raises:
            ValueError: If ``count`` is smaller than one.
        """
        if count < 1:
            raise ValueError("count must be positive")

        indices = np.arange(count, dtype=np.float64)
        z       = 1.0 - 2.0 * (indices + 0.5) / count
        radial  = np.sqrt(np.maximum(0.0, 1.0 - z * z))

        golden_angle = math.pi * (3.0 - math.sqrt(5.0))
        theta        = indices * golden_angle + phase
        return np.column_stack((radial * np.cos(theta), radial * np.sin(theta), z))

    def estimate_normals(
        self,
        points            : np.ndarray,
        outward_reference : np.ndarray | None = None,
    ) -> np.ndarray:
        """Estimate oriented PCA normals for an externally sampled point cloud.

        The neighborhood covariance surrogate is ``X.T @ X`` after local mean subtraction. Its
        smallest-eigenvalue eigenvector minimizes squared orthogonal distance to the local tangent
        plane. Neighborhoods use radius ``3h`` and fall back to up to eight nearest points.

        Args:
            points: Cartesian point cloud with shape ``[N,3]`` and ``N >= 3``.
            outward_reference: Either one reference point ``[3]`` or one orientation vector or
                reference per point ``[N,3]``. When absent, the largest-magnitude normal component
                is made positive for deterministic, but not necessarily outward, orientation.

        Returns:
            Unit ``float32 [N,3]`` PCA normals.

        Raises:
            ValueError: If the input does not have shape ``[N,3]`` with at least three points.
        """
        values = np.asarray(points, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 3 or len(values) < 3:
            raise ValueError("points must have shape [N,3] with N >= 3")
        tree    = cKDTree(values)
        normals = np.empty_like(values)

        # Solve one local symmetric eigensystem and orient its least-variance eigenvector.
        for index, point in enumerate(values):
            neighbors = tree.query_ball_point(point, 3.0 * self.resolution)
            if len(neighbors) < 6:
                _, nearest = tree.query(point, k=min(8, len(values)))
                neighbors = np.atleast_1d(nearest).tolist()
            local      = values[neighbors] - values[neighbors].mean(axis=0)
            _, vectors = np.linalg.eigh(local.T @ local)
            normal     = vectors[:, 0]
            if outward_reference is None:
                pivot = int(np.argmax(np.abs(normal)))
                if normal[pivot] < 0:
                    normal = -normal
            else:
                reference = np.asarray(outward_reference)
                direction = point - reference if reference.shape == (3,) else reference[index]

                if np.dot(normal, direction) < 0:
                    normal = -normal
            normals[index] = normal
        return self._unit(normals).astype(np.float32)

    def estimate_curvatures(
        self,
        points  : np.ndarray,
        normals : np.ndarray,
    ) -> np.ndarray:
        """Estimate mean curvature, Gaussian curvature, and curvedness at configured scales.

        At radius ``r=s*h`` for each configured multiplier ``s``, neighbor offsets are expressed in
        an orthonormal tangent frame. Coordinates ``u,v,z`` are divided by ``r`` before a
        Gaussian-weighted least-squares fit to
        ``z = 0.5*a*u² + b*u*v + 0.5*c*v² + d*u + e*v + f``. In the small-slope Monge
        approximation, the dimensional outward shape matrix is ``-[[a,b],[b,c]]/r``. Singular
        directions smaller than five percent of the largest design singular value are discarded;
        this prevents nearly collinear or duplicate samples from creating unbounded curvature.
        Its eigenvalues ``k1,k2`` produce ``H=(k1+k2)/2``, ``K=k1*k2``, and
        ``C=sqrt((k1²+k2²)/2)``.

        Args:
            points: Cartesian surface samples with shape ``[M,3]`` in ångströms.
            normals: Corresponding finite normals with shape ``[M,3]``; they are renormalized.

        Returns:
            ``float32 [M,S,3]`` values, where ``S=len(curvature_scales)``. Axis one follows the
            configured scale order and axis two is ``(H,K,C)``. ``H/C`` have units Å⁻¹ and ``K``
            has units Å⁻². Non-finite fit results are replaced with zero.
        """
        values       = np.asarray(points, dtype=np.float64)
        unit_normals = self._unit(np.asarray(normals, dtype=np.float64))
        tree         = cKDTree(values)
        output       = np.zeros((len(values), len(self.curvature_scales), 3), dtype=np.float32)

        # Tangent frames and fallback nearest-neighbor rows do not depend on curvature scale. Build
        # them once instead of repeating the same geometry for every configured radius.

        bases           = [self._tangent_basis(normal) for normal in unit_normals]
        first_tangents  = np.asarray([basis[0] for basis in bases])
        second_tangents = np.asarray([basis[1] for basis in bases])
        _, nearest      = tree.query(values, k=min(12, len(values)), workers=1)
        nearest         = np.asarray(nearest).reshape(len(values), -1)

        # Query all radius neighborhoods once per scale. The remaining loop performs the genuinely
        # independent weighted quadratic fits in dimensionless local coordinates.

        for scale_index, scale in enumerate(self.curvature_scales):
            radius        = scale * self.resolution
            neighborhoods = tree.query_ball_point(
                values,
                radius,
                workers       = 1,
                return_sorted = True,
            )

            for index, point in enumerate(values):
                neighbors = neighborhoods[index]
                if len(neighbors) < 7:
                    neighbors = nearest[index]

                offsets       = values[neighbors] - point
                u             = offsets @ first_tangents[index] / radius
                v             = offsets @ second_tangents[index] / radius
                z             = offsets @ unit_normals[index] / radius

                design = np.column_stack(
                    (0.5 * u * u, u * v, 0.5 * v * v, u, v, np.ones(len(u)))
                )
                weights = np.exp(
                    -np.sum(offsets * offsets, axis=1) / max(radius * radius, 1.0e-12)
                )
                coefficients, *_ = np.linalg.lstsq(
                    design * np.sqrt(weights[:, None]),
                    z * np.sqrt(weights),
                    rcond=5.0e-2,
                )
                # Negating the height Hessian makes outward-oriented convex spheres positive.
                shape = -np.array(
                    [
                        [coefficients[0], coefficients[1]],
                        [coefficients[1], coefficients[2]],
                    ]
                ) / radius
                principal = np.linalg.eigvalsh(shape)
                output[index, scale_index] = (
                    float(principal.mean()),
                    float(principal.prod()),
                    float(np.sqrt(np.mean(principal * principal))),
                )
        return np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

    def area_weights(self, points: np.ndarray) -> np.ndarray:
        """Approximate normalized represented area from local point spacing.

        For point ``i``, ``ell_i`` is the median distance to up to six non-self nearest neighbors.
        The raw proxy is ``max(ell_i², eps_float32)`` and final weights divide every proxy by their
        sum. This corrects density variation for future integration but is not exact Voronoi area or
        solvent-accessible area in square ångströms.

        Args:
            points: Surface point coordinates with shape ``[M,3]``.

        Returns:
            Positive finite ``float32 [M]`` dimensionless weights that sum to one.
        """
        count = len(points)
        if count == 1:
            return np.ones(1, dtype=np.float32)

        # Squared local spacing has area units before normalization.
        distances, _ = cKDTree(points).query(points, k=min(7, count))
        distances    = np.atleast_2d(distances)
        spacing      = np.median(distances[:, 1:], axis=1)
        raw          = np.maximum(spacing * spacing, np.finfo(np.float32).eps)
        return (raw / raw.sum()).astype(np.float32)

    def build_graph(
        self,
        points  : np.ndarray,
        normals : np.ndarray,
    ) -> tuple[dict[str, np.ndarray], list[str]]:
        """Build a sparse local surface graph while rejecting opposite-wall shortcuts.

        Candidate pairs lie within ``2.5h``. A pair is discarded if ``n_i·n_j < -0.25`` or if
        ``max(|delta·n_i|, |delta·n_j|) > 0.8||delta||``. The second rule rejects displacements
        dominated by a surface normal rather than a tangent direction. Connected components are
        computed on a symmetric sparse COO adjacency.

        Args:
            points: Surface coordinates with shape ``[M,3]`` in ångströms.
            normals: Corresponding outward unit normals with shape ``[M,3]``.

        Returns:
            Undirected ``int32`` edge indices, ``float32`` distances, component IDs, and warnings
            when component fragmentation exceeds deterministic thresholds.
        """
        # Radius search creates undirected candidates without an M-by-M distance matrix.
        candidates = cKDTree(points).query_pairs(2.5 * self.resolution, output_type="ndarray")
        kept: list[tuple[int, int]] = []

        # Reject opposed normals and normal-dominated chords before graph publication.
        for src, dst in np.asarray(candidates, dtype=np.int32).reshape(-1, 2):
            delta    = points[dst] - points[src]
            distance = float(np.linalg.norm(delta))
            if float(np.dot(normals[src], normals[dst])) < -0.25:
                continue
            normal_offset = max(
                abs(float(np.dot(delta, normals[src]))),
                abs(float(np.dot(delta, normals[dst]))),
            )
            if normal_offset > 0.8 * distance:
                continue
            kept.append((int(src), int(dst)))
        edge_index = (
            np.asarray(kept, dtype=np.int32).T if kept else np.empty((2, 0), dtype=np.int32)
        )
        if kept:
            distances = np.linalg.norm(
                points[edge_index[0]] - points[edge_index[1]], axis=1
            ).astype(np.float32)
            row       = np.r_[edge_index[0], edge_index[1]]
            column    = np.r_[edge_index[1], edge_index[0]]
            adjacency = coo_matrix(
                (np.ones(len(row)), (row, column)), shape=(len(points), len(points))
            )
            component_count, component_ids = connected_components(adjacency, directed=False)
        else:
            distances       = np.empty(0, dtype=np.float32)
            component_count = len(points)
            component_ids   = np.arange(len(points), dtype=np.int32)

        # Diagnose excessive fragmentation without assigning pocket/cavity semantics.
        component_ids = np.asarray(component_ids, dtype=np.int32)
        sizes         = np.bincount(component_ids, minlength=component_count)
        tiny          = int(np.count_nonzero(sizes < 5))

        warnings: list[str] = []
        if component_count > max(3, len(points) // 100) or tiny > max(2, component_count // 2):
            warnings.append(
                f"surface graph has {component_count} components ({tiny} with fewer than 5 points)"
            )
        return {
            "surface_edge_index": edge_index,
            "surface_edge_distance": distances,
            "surface_component_ids": component_ids,
        }, warnings

    def _surface_candidates(
        self,
        atom_positions : np.ndarray,
        expanded_radii : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample deterministic Fibonacci candidates on every expanded atomic sphere.

        Sphere ``i`` receives ``max(24, ceil(4*pi*R_i²/(0.55*h²)))`` directions. Its phase is
        ``2*pi*frac(0.7548776662466927*i)`` so neighboring atoms do not reuse identical azimuthal
        patterns. Candidate coordinates are ``c_i + R_i*u_k``.

        Args:
            atom_positions: ``float64 [N,3]`` atom centers in ångströms.
            expanded_radii: ``float64 [N]`` values ``vdw_radius + probe_radius`` in ångströms.

        Returns:
            Concatenated candidate coordinates and the ``int32`` owner atom for every candidate.
        """
        points: list[np.ndarray] = []
        owners: list[np.ndarray] = []

        # Allocate approximately 0.55*h² of expanded-sphere area to each raw direction.
        for atom_index, (center, radius) in enumerate(
            zip(atom_positions, expanded_radii, strict=True)
        ):
            count = max(
                24,
                math.ceil(4.0 * math.pi * radius * radius / (0.55 * self.resolution**2)),
            )
            phase = (atom_index * 0.7548776662466927 % 1.0) * 2.0 * math.pi

            points.append(center + radius * self.fibonacci_sphere(count, phase))
            owners.append(np.full(count, atom_index, dtype=np.int32))
        return np.concatenate(points), np.concatenate(owners)

    def _remove_buried(
        self,
        candidates     : np.ndarray,
        owners         : np.ndarray,
        atom_positions : np.ndarray,
        expanded_radii : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove candidates that lie strictly inside another expanded atomic sphere.

        Candidate ``p`` owned by ``i`` is buried when another atom ``j`` satisfies
        ``||p-c_j|| < R_j-tau``, with ``tau=max(1e-5,0.02h)`` ångströms. This is the signed implicit
        test ``g_j(p)<-tau`` for ``g_j(x)=||x-c_j||-R_j``. A center KD-tree limits exact tests to
        atoms capable of containing the point.

        Args:
            candidates: Raw expanded-sphere samples with shape ``[P,3]``.
            owners: Owner atom indices with shape ``[P]``.
            atom_positions: Atom centers with shape ``[N,3]``.
            expanded_radii: Expanded radii with shape ``[N]``.

        Returns:
            Candidate and owner arrays masked by solvent exposure.
        """
        tree          = cKDTree(atom_positions)
        neighborhoods = tree.query_ball_point(
            candidates, float(expanded_radii.max()) + self.resolution * 0.05
        )
        keep      = np.ones(len(candidates), dtype=bool)
        tolerance = max(1.0e-5, self.resolution * 0.02)

        # The owner sphere contains the candidate on its boundary and cannot bury itself.
        for index, nearby in enumerate(neighborhoods):
            owner = int(owners[index])
            for atom_index in nearby:
                if atom_index == owner:
                    continue
                distance = float(np.linalg.norm(candidates[index] - atom_positions[atom_index]))
                if distance < float(expanded_radii[atom_index]) - tolerance:
                    keep[index] = False
                    break
        return candidates[keep], owners[keep]

    def _voxel_select(
        self,
        points : np.ndarray,
        owners : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Reduce exposed candidates to one deterministic representative per resolution voxel.

        With componentwise minimum ``o``, the cell coordinate is ``q=floor((p-o)/h)``. Cells are
        lexicographically sorted and the original candidate closest to ``o+(q+1/2)h`` is retained;
        original index resolves exact ties. Points are selected, never moved.

        Args:
            points: Exposed candidate coordinates with shape ``[P,3]``.
            owners: Corresponding owner atom indices with shape ``[P]``.

        Returns:
            Reduced point/owner arrays containing at most one sample per occupied voxel.
        """
        origin = points.min(axis=0)
        cells  = np.floor((points - origin) / self.resolution).astype(np.int64)
        order  = np.lexsort((np.arange(len(points)), cells[:, 2], cells[:, 1], cells[:, 0]))

        # Adjacent runs in lexicographic order identify every occupied cell exactly once.
        ordered_cells = cells[order]
        boundaries    = np.r_[
            0,
            np.flatnonzero(np.any(np.diff(ordered_cells, axis=0), axis=1)) + 1,
            len(order),
        ]
        selected: list[int] = []
        for start, end in pairwise(boundaries):
            group     = order[start:end]
            center    = origin + (cells[group[0]] + 0.5) * self.resolution
            distances = np.sum((points[group] - center) ** 2, axis=1)
            selected.append(int(group[int(np.argmin(distances))]))
        chosen = np.asarray(selected, dtype=np.int64)
        return points[chosen], owners[chosen]

    def _envelope_normals(
        self,
        points          : np.ndarray,
        owners          : np.ndarray,
        atom_positions  : np.ndarray,
        expanded_radii  : np.ndarray,
    ) -> np.ndarray:
        """Estimate outward normals by blending gradients of signed sphere gaps.

        For ``g_j(p)=||p-c_j||-R_j``, atoms satisfying
        ``g_j <= min(g)+2.5*sigma`` are active, where ``sigma=max(0.25h,1e-3)``. Their unit radial
        gradients are weighted by ``exp(-(g_j-min(g))/sigma)``, summed, and normalized. The owner
        radial direction is a deterministic fallback when the weighted sum nearly cancels.

        Args:
            points: Reduced exposed surface coordinates with shape ``[M,3]``.
            owners: Owner atom index for each surface point.
            atom_positions: Atom centers with shape ``[N,3]``.
            expanded_radii: Expanded sphere radii with shape ``[N]``.

        Returns:
            Outward unit ``float64 [M,3]`` normals.
        """
        # Only centers close enough to influence the soft minimum enter each local blend.
        neighborhoods = cKDTree(atom_positions).query_ball_point(
            points, float(expanded_radii.max()) + self.resolution
        )
        normals    = np.empty_like(points)
        smoothness = max(0.25 * self.resolution, 1.0e-3)

        for point_index, nearby in enumerate(neighborhoods):
            indices = np.asarray(nearby, dtype=np.int32)
            offsets = points[point_index] - atom_positions[indices]
            lengths = np.linalg.norm(offsets, axis=1)
            gaps    = lengths - expanded_radii[indices]

            minimum   = float(gaps.min())
            active    = gaps <= minimum + 2.5 * smoothness
            weights   = np.exp(-(gaps[active] - minimum) / smoothness)
            direction = np.sum(weights[:, None] * offsets[active] / lengths[active, None], axis=0)

            if float(np.linalg.norm(direction)) < 1.0e-10:
                owner = int(owners[point_index])
                direction = points[point_index] - atom_positions[owner]
            normals[point_index] = direction
        return self._unit(normals)

    @staticmethod
    def _unit(vectors: np.ndarray) -> np.ndarray:
        """Normalize a batch of vectors with a safe lower norm bound.

        Args:
            vectors: Numeric vectors with shape ``[N,3]``.

        Returns:
            Input vectors divided row-wise by ``max(||v||_2, 1e-12)``.
        """
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / np.maximum(norms, 1.0e-12)

    @staticmethod
    def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construct a stable right-handed orthonormal basis perpendicular to a normal.

        Args:
            normal: Unit surface normal with shape ``[3]``.

        Returns:
            Tangents ``(t1,t2)`` where ``t1=normalize(normal cross axis)`` and
            ``t2=normal cross t1``. The reference axis switches from x to y near x parallelism.
        """
        axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        first = np.cross(normal, axis)
        first /= np.linalg.norm(first)
        return first, np.cross(normal, first)
