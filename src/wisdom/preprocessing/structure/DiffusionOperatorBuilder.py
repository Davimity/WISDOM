"""Sparse intrinsic operators for DiffusionNet and controlled surface encoders."""

from __future__ import annotations

import numpy as np
import robust_laplacian
from scipy.sparse import coo_matrix, diags
from scipy.sparse.linalg import eigsh
from scipy.spatial import cKDTree


class DiffusionOperatorBuilder:
    """Construct bounded point-cloud Laplacian, spectral, and tangent-gradient operators."""

    def __init__(
        self,
        resolution    : float,
        spectral_modes: int = 128,
        max_neighbors : int = 24,
    ) -> None:
        """Set operator resolution, spectral budget, and bounded surface neighborhood width.

        Args:
            resolution: Physical surface sampling scale ``h`` in ångströms.
            spectral_modes: Maximum number ``Qmax`` of generalized Laplacian modes persisted.
            max_neighbors: Maximum nearest surface neighbors retained per point for differential
                operators and V3 encoders.

        Raises:
            ValueError: If a setting is non-positive.
        """
        if resolution <= 0.0 or spectral_modes < 1 or max_neighbors < 1:
            raise ValueError("diffusion resolution, modes, and neighbors must be positive")

        self.resolution     = resolution
        self.spectral_modes = spectral_modes
        self.max_neighbors  = max_neighbors

    def build(
        self,
        positions : np.ndarray,
        normals   : np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Approximate intrinsic operators directly on an oriented molecular point cloud.

        A symmetric bounded KNN graph supplies a positive-semidefinite kernel stiffness ``L``.
        Local squared spacing supplies a positive lumped mass ``M`` in Å², so the generalized
        eigenvalues in ``L phi = lambda M phi`` have units Å⁻². Sparse shift-invert ``eigsh``
        computes only the lowest ``min(Qmax,M-1)`` modes; no dense pair matrix or full
        eigendecomposition is formed.

        Tangent derivatives fit ``df ~= gx*u + gy*v`` by weighted local least squares. Each row of
        the resulting sparse ``Gx`` and ``Gy`` includes the center coefficient needed to express
        neighbor differences, hence both operators annihilate a constant field.

        Args:
            positions: Finite surface coordinates ``float [M,3]`` in ångströms.
            normals: Corresponding outward unit normals ``float [M,3]``.

        Returns:
            Lumped mass, low eigenpairs, sparse tangent gradients, and a padded bounded surface
            neighborhood table. Spectral point order is exactly the input point order.

        Raises:
            ValueError: If fewer than two points exist or eigensolving produces invalid modes.
            RuntimeError: If SciPy cannot solve the sparse symmetric eigenproblem.
        """
        # Intrinsic operators are translation invariant. Subtracting the point-cloud centroid
        # before neighbor and Laplacian arithmetic prevents a large absolute origin from consuming
        # floating-point precision without changing any physical displacement.

        points       = np.asarray(positions, dtype=np.float64)
        points       = points - points.mean(axis=0, keepdims=True)
        unit_normals = np.asarray(normals, dtype=np.float64)
        point_count = len(points)
        if point_count < 2:
            raise ValueError("diffusion operators require at least two surface points")

        neighbors, distances, mask = self._neighborhoods(points)
        stiffness, mass            = self._laplacian(points)
        eigenvalues, eigenvectors  = self._eigenpairs(stiffness, mass)
        gradient_index, gradient_x, gradient_y = self._gradients(
            points,
            unit_normals,
            neighbors,
            distances,
            mask,
        )

        return {
            "surface_neighbors": neighbors,
            "surface_neighbor_distances": distances,
            "surface_neighbor_mask": mask,
            "diffusion_mass": mass.astype(np.float32),
            "diffusion_eigenvalues": eigenvalues.astype(np.float32),
            "diffusion_eigenvectors": eigenvectors.astype(np.float32),
            "diffusion_gradient_index": gradient_index,
            "diffusion_gradient_x": gradient_x.astype(np.float32),
            "diffusion_gradient_y": gradient_y.astype(np.float32),
        }

    def _neighborhoods(
        self,
        points: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Build a deterministic padded nearest-surface table excluding self references.

        Args:
            points: Surface coordinates ``float64 [M,3]``.

        Returns:
            Neighbor IDs, distances, and validity mask with shape ``[M,Ksmax]``. Rows use
            ``(distance, point_id)`` ordering and invalid columns use ID ``-1``.
        """
        point_count = len(points)
        width = min(point_count, self.max_neighbors + 1)
        tree = cKDTree(points)
        raw_distances, _ = tree.query(
            points,
            k=width,
            workers=1,
        )
        raw_distances = np.asarray(raw_distances).reshape(point_count, width)

        neighbors = np.full((point_count, self.max_neighbors), -1, dtype=np.int32)
        distances = np.zeros((point_count, self.max_neighbors), dtype=np.float32)
        mask      = np.zeros((point_count, self.max_neighbors), dtype=np.bool_)

        for source in range(point_count):
            finite_distances = raw_distances[source][np.isfinite(raw_distances[source])]
            cutoff = float(finite_distances[-1])
            candidates_with_ties = tree.query_ball_point(
                points[source],
                np.nextafter(cutoff, np.inf),
            )
            candidates = [
                (float(np.linalg.norm(points[source] - points[target])), int(target))
                for target in candidates_with_ties
                if target != source
            ]
            candidates.sort(key=lambda value: (value[0], value[1]))
            for column, (distance, target) in enumerate(candidates[: self.max_neighbors]):
                neighbors[source, column] = target
                distances[source, column] = distance
                mask[source, column]      = True

        if np.any(mask.sum(axis=1) == 0):
            raise ValueError("every surface point requires at least one diffusion neighbor")
        return neighbors, distances, mask

    def _laplacian(self, points: np.ndarray) -> tuple[coo_matrix, np.ndarray]:
        """Construct the official robust point-cloud stiffness and physical lumped mass.

        ``robust_laplacian`` implements Sharp and Crane's intrinsic-Delaunay construction used by
        the official DiffusionNet point-cloud pipeline. It returns a sparse positive-semidefinite
        weak Laplacian and diagonal area mass without forming dense point-pair matrices. Coordinates
        remain in ångströms, hence mass has units Å² and generalized eigenvalues have units Å⁻².

        Args:
            points: Surface coordinates ``float64 [M,3]`` in physical ångströms.

        Returns:
            Sparse stiffness ``L`` and strictly positive diagonal mass ``float64 [M]``.

        Raises:
            RuntimeError: If robust point-cloud operator construction fails.
            ValueError: If the returned matrix shapes or masses are invalid.
        """
        neighbor_count = min(len(points) - 1, max(6, self.max_neighbors))
        try:
            stiffness, mass_matrix = robust_laplacian.point_cloud_laplacian(
                points,
                n_neighbors=neighbor_count,
            )
        except Exception as error:
            message = f"robust point-cloud Laplacian construction failed: {error}"
            raise RuntimeError(message) from error

        stiffness = stiffness.tocoo().astype(np.float64)
        mass       = np.asarray(mass_matrix.diagonal(), dtype=np.float64)
        if stiffness.shape != (len(points), len(points)) or mass.shape != (len(points),):
            raise ValueError("robust Laplacian returned incompatible operator dimensions")
        if (
            not np.isfinite(stiffness.data).all()
            or not np.isfinite(mass).all()
            or np.any(mass <= 0)
        ):
            raise ValueError("robust Laplacian returned non-finite or non-positive operators")
        return stiffness, mass

    def _eigenpairs(
        self,
        stiffness: coo_matrix,
        mass     : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Solve the lowest generalized modes with deterministic normalization and signs.

        After symmetric mass normalization, the positive-semidefinite problem is
        ``A y = lambda y``. ARPACK shift-invert applies ``(A - sigma I)^-1`` with the small
        negative shift ``sigma=-1e-4 Å^-2``. The desired eigenvalues nearest zero then become the
        largest transformed magnitudes, which converge much faster than asking ARPACK for the
        smallest magnitudes directly. A negative shift lies outside the non-negative spectrum and
        avoids singular factorization at the constant zero mode; it does not change the recovered
        eigenpairs.

        Args:
            stiffness: Sparse symmetric positive-semidefinite ``[M,M]`` matrix.
            mass: Positive diagonal mass entries ``[M]``.

        Returns:
            Sorted eigenvalues ``[Q]`` and mass-orthonormal eigenvectors ``[M,Q]``.

        Raises:
            RuntimeError: If sparse symmetric eigensolving fails.
            ValueError: If the returned spectrum is non-finite or materially negative.
        """
        point_count = len(mass)
        mode_count  = min(self.spectral_modes, point_count - 1)

        inverse_sqrt_mass = 1.0 / np.sqrt(mass)
        normalized = (
            diags(inverse_sqrt_mass) @ stiffness.tocsr() @ diags(inverse_sqrt_mass)
        ).tocsr()
        try:
            eigenvalues, normalized_vectors = eigsh(
                normalized,
                k       = mode_count,
                sigma   = -1.0e-4,
                which   = "LM",
                v0      = np.ones(point_count, dtype=np.float64),
                tol     = 1.0e-9,
                maxiter=max(1000, point_count * 20),
            )
        except Exception as error:
            raise RuntimeError(f"sparse diffusion eigensolver failed: {error}") from error

        order = np.argsort(eigenvalues, kind="stable")
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = inverse_sqrt_mass[:, None] * normalized_vectors[:, order]

        # A canonical largest-entry sign makes persisted bytes stable without changing diffusion.
        for mode in range(mode_count):
            pivot = int(np.argmax(np.abs(eigenvectors[:, mode])))
            if eigenvectors[pivot, mode] < 0.0:
                eigenvectors[:, mode] *= -1.0

        gram = eigenvectors.T @ (mass[:, None] * eigenvectors)
        if (
            not np.isfinite(eigenvalues).all()
            or not np.isfinite(eigenvectors).all()
            or float(eigenvalues.min()) < -1.0e-7
            or not np.allclose(gram, np.eye(mode_count), rtol=2.0e-4, atol=2.0e-4)
        ):
            raise ValueError("diffusion eigenpairs are not finite and mass-orthonormal")
        return eigenvalues, eigenvectors

    def _gradients(
        self,
        points    : np.ndarray,
        normals   : np.ndarray,
        neighbors : np.ndarray,
        distances : np.ndarray,
        mask      : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fit sparse tangent-frame derivatives by regularized weighted least squares.

        Args:
            points: Surface coordinates ``float64 [M,3]``.
            normals: Unit normals ``float64 [M,3]``.
            neighbors: Neighbor IDs ``int32 [M,Ksmax]``.
            distances: Neighbor distances ``float32 [M,Ksmax]`` in ångströms.
            mask: Validity mask for neighborhood columns.

        Returns:
            Shared COO indices ``int32 [2,G]`` and ``Gx/Gy`` values ``float64 [G]``.
        """
        rows: list[int]       = []
        columns: list[int]    = []
        values_x: list[float] = []
        values_y: list[float] = []

        for source in range(len(points)):
            targets = neighbors[source, mask[source]]
            offsets = points[targets] - points[source]
            tangent_x, tangent_y = self._tangent_basis(normals[source])
            coordinates = np.column_stack((offsets @ tangent_x, offsets @ tangent_y))

            scale   = max(float(np.median(distances[source, mask[source]])), 1.0e-6)
            weights = np.exp(-0.5 * (distances[source, mask[source]] / scale) ** 2)
            normal  = coordinates.T @ (weights[:, None] * coordinates)
            normal += np.eye(2) * max(float(np.trace(normal)) * 1.0e-6, 1.0e-10)
            coefficients = np.linalg.solve(normal, coordinates.T * weights[None, :])

            for target, coefficient_x, coefficient_y in zip(
                targets,
                coefficients[0],
                coefficients[1],
                strict=True,
            ):
                rows.append(source)
                columns.append(int(target))
                values_x.append(float(coefficient_x))
                values_y.append(float(coefficient_y))

            rows.append(source)
            columns.append(source)
            values_x.append(float(-coefficients[0].sum()))
            values_y.append(float(-coefficients[1].sum()))

        return (
            np.asarray((rows, columns), dtype=np.int32),
            np.asarray(values_x, dtype=np.float64),
            np.asarray(values_y, dtype=np.float64),
        )

    @staticmethod
    def _tangent_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Construct one stable orthonormal tangent frame around a unit normal.

        Args:
            normal: Unit normal ``float64 [3]``.

        Returns:
            Right-handed tangent vectors ``(tx,ty)``.
        """
        axis = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.8 else np.array([0.0, 1.0, 0.0])
        tangent_x = np.cross(normal, axis)
        tangent_x /= np.linalg.norm(tangent_x)
        return tangent_x, np.cross(normal, tangent_x)
