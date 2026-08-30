"""Bounded deterministic atom neighborhoods for molecular-surface points."""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


class SurfaceAtomNeighborhoodBuilder:
    """Store a compact nearest-atom table and invariant local geometry per surface point."""

    def __init__(
        self,
        radius        : float = 6.0,
        max_neighbors : int = 32,
    ) -> None:
        """Set the physical cutoff and maximum table width.

        Args:
            radius: Largest accepted atom-center distance in ångströms.
            max_neighbors: Persisted table width ``Jmax`` and maximum valid atoms per point.

        Raises:
            ValueError: If the radius or table width is not positive.
        """
        if radius <= 0.0 or max_neighbors < 1:
            raise ValueError("surface-atom radius and maximum neighbor count must be positive")

        self.radius        = radius
        self.max_neighbors = max_neighbors

    def build(
        self,
        atom_positions    : np.ndarray,
        surface_positions : np.ndarray,
        surface_normals   : np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Find the nearest valid atoms and encode rotation/translation-invariant offsets.

        For surface point ``p`` with normal ``n`` and atom center ``a``, the persisted geometry is
        distance ``d=||a-p||``, signed normal offset ``z=(a-p)·n``, and tangential magnitude
        ``rho=sqrt(max(d²-z²,0))``. Distances are persisted as ``float32``, so rows are ordered by
        ``(float32(d), atom_id)``. Using the stored precision for ordering makes exact ties
        deterministic after serialization. Padding uses atom ID ``-1`` plus a false mask, so
        slicing the first ``J`` columns yields nested neighborhoods.

        Args:
            atom_positions: Finite ``float [N,3]`` atom centers in ångströms.
            surface_positions: Finite ``float [M,3]`` surface points in the same frame.
            surface_normals: Outward unit normals with shape ``[M,3]``.

        Returns:
            ``surface_atom_neighbors``, distance, normal-offset, tangential-distance, and mask
            arrays with common shape ``[M,Jmax]``.

        Raises:
            ValueError: If a surface point has no atom within the configured physical cutoff.
        """
        atoms   = np.asarray(atom_positions, dtype=np.float64)
        points  = np.asarray(surface_positions, dtype=np.float64)
        normals = np.asarray(surface_normals, dtype=np.float64)

        tree      = cKDTree(atoms)
        neighbors = tree.query_ball_point(points, self.radius, workers=1)

        output_neighbors = np.full((len(points), self.max_neighbors), -1, dtype=np.int32)
        output_distances = np.zeros((len(points), self.max_neighbors), dtype=np.float32)
        output_normal    = np.zeros((len(points), self.max_neighbors), dtype=np.float32)
        output_tangent   = np.zeros((len(points), self.max_neighbors), dtype=np.float32)
        output_mask      = np.zeros((len(points), self.max_neighbors), dtype=np.bool_)

        for point_index in range(len(points)):
            atom_ids = np.asarray(neighbors[point_index], dtype=np.int32)
            if len(atom_ids) == 0:
                raise ValueError(
                    f"surface point {point_index} has no atom within "
                    f"surface_atom_radius={self.radius}"
                )

            # Compute one vectorized row, round distances to their persisted precision, and use
            # atom IDs to resolve only the ties that will still exist after the NPZ is reopened.

            offsets = atoms[atom_ids] - points[point_index]
            exact_distances, normal_offsets, tangential_distances = self.invariant_geometry(
                offsets,
                normals[point_index],
            )
            stored_distances = exact_distances.astype(np.float32)
            order            = np.lexsort((atom_ids, stored_distances))[: self.max_neighbors]

            selected_ids     = atom_ids[order]
            selected_stored  = stored_distances[order]
            selected_normal  = normal_offsets[order]
            selected_tangent = tangential_distances[order]
            width = len(selected_ids)

            output_neighbors[point_index, :width] = selected_ids
            output_distances[point_index, :width] = selected_stored
            output_normal[point_index, :width]    = selected_normal
            output_tangent[point_index, :width]   = selected_tangent
            output_mask[point_index, :width]      = True

        return {
            "surface_atom_neighbors": output_neighbors,
            "surface_atom_distances": output_distances,
            "surface_atom_normal_offsets": output_normal,
            "surface_atom_tangential_distances": output_tangent,
            "surface_atom_mask": output_mask,
        }

    @staticmethod
    def invariant_geometry(
        offsets : np.ndarray,
        normals : np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculate stable distance, normal, and tangential offset components.

        For offset ``o=a-p`` and unit surface normal ``n``, the signed normal component is
        ``z=o·n`` and the tangential vector is ``t=o-z*n``. Its magnitude
        ``rho=||t||`` is mathematically equal to ``sqrt(||o||²-z²)`` but avoids subtracting nearly
        equal squared values when an atom lies almost exactly along the normal. Inputs are promoted
        to ``float64`` before arithmetic, and normals are normalized so generation and archive
        validation share one numerical definition.

        Args:
            offsets: One or more atom-minus-surface vectors with shape ``[...,3]`` in ångströms.
            normals: One broadcast-compatible surface normal with shape ``[3]`` or ``[...,3]``.

        Returns:
            ``float64`` distance, signed normal-offset, and tangential-magnitude arrays with shape
            ``offsets.shape[:-1]``.

        Raises:
            ValueError: If any supplied normal has effectively zero length.
        """
        values     = np.asarray(offsets, dtype=np.float64)
        directions = np.asarray(normals, dtype=np.float64)
        lengths    = np.linalg.norm(directions, axis=-1, keepdims=True)
        if np.any(lengths <= 1.0e-12):
            raise ValueError("surface normals must have non-zero length")

        unit              = directions / lengths
        distances         = np.linalg.norm(values, axis=-1)
        normal_offsets    = np.sum(values * unit, axis=-1)
        tangent_vectors   = values - normal_offsets[..., None] * unit
        tangent_distances = np.linalg.norm(tangent_vectors, axis=-1)
        return distances, normal_offsets, tangent_distances
