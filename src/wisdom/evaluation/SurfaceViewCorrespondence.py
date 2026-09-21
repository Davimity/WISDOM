"""Approximate physical correspondence between two surface discretizations."""

import torch
import numpy as np

from torch import Tensor
from scipy.spatial import cKDTree


class SurfaceViewCorrespondence:
    """Match physically compatible points without assuming equal point indices."""

    def align(
        self,
        reference_positions : Tensor,
        reference_normals   : Tensor,
        view_positions      : Tensor,
        view_normals        : Tensor,
        maximum_distance    : float = 1.5,
        minimum_normal_cosine: float = 0.5,
        mutual              : bool  = True,
    ) -> dict[str, Tensor]:
        """Align one surface view to a reference surface through bounded nearest neighbours.

        A reference point ``p_i`` first receives the nearest point ``q_j`` in the other view. The
        pair is retained only when ``||p_i-q_j|| <= r`` and the cosine between their normals is at
        least ``c``. With mutual matching, the reverse nearest-neighbour query must also map
        ``q_j`` back to ``p_i``. Mutuality prevents a dense region in one discretization from
        contributing the same coarse point many times.

        Args:
            reference_positions: Finite Cartesian coordinates ``[M,3]`` in ångströms.
            reference_normals: Finite reference unit-normal directions ``[M,3]``.
            view_positions: Finite Cartesian coordinates ``[N,3]`` in the same coordinate frame.
            view_normals: Finite view unit-normal directions ``[N,3]``.
            maximum_distance: Largest accepted Euclidean point separation ``r`` in ångströms.
            minimum_normal_cosine: Smallest accepted normal cosine ``c`` in ``[-1,1]``.
            mutual: Require both directed nearest-neighbour queries to select the same pair.

        Returns:
            CPU tensors ``reference_indices``, ``view_indices``, ``distances``, and
            ``normal_cosines`` ordered by increasing reference index. Indices use ``torch.long``;
            geometric values use ``torch.float32``.

        Raises:
            ValueError: If arrays are empty, non-finite, mis-shaped, contain zero normals, use an
                invalid threshold, or yield no physically compatible match.
        """
        reference = reference_positions.detach().cpu().to(torch.float64)
        candidate = view_positions.detach().cpu().to(torch.float64)
        normals_a = reference_normals.detach().cpu().to(torch.float64)
        normals_b = view_normals.detach().cpu().to(torch.float64)

        arrays = (reference, candidate, normals_a, normals_b)
        if any(values.ndim != 2 or values.shape[1] != 3 or not len(values) for values in arrays):
            raise ValueError("surface correspondence requires non-empty [M,3] arrays")
        if reference.shape != normals_a.shape or candidate.shape != normals_b.shape:
            raise ValueError("surface positions and normals must be aligned within each view")
        if any(not torch.isfinite(values).all() for values in arrays):
            raise ValueError("surface correspondence arrays must be finite")
        if maximum_distance <= 0.0:
            raise ValueError("maximum correspondence distance must be positive")
        if not -1.0 <= minimum_normal_cosine <= 1.0:
            raise ValueError("minimum normal cosine must lie in [-1,1]")

        normal_norm_a = torch.linalg.vector_norm(normals_a, dim=1)
        normal_norm_b = torch.linalg.vector_norm(normals_b, dim=1)
        if torch.any(normal_norm_a == 0.0) or torch.any(normal_norm_b == 0.0):
            raise ValueError("surface correspondence normals must be non-zero")

        # KD-trees keep the correspondence sparse: only one bounded neighbour is queried for each
        # point instead of materializing the dense M-by-N distance matrix.

        reference_numpy = np.asarray(reference)
        candidate_numpy = np.asarray(candidate)
        distances, nearest = cKDTree(candidate_numpy).query(
            reference_numpy,
            k=1,
            distance_upper_bound=maximum_distance,
        )

        reference_indices = np.arange(len(reference_numpy), dtype=np.int64)
        valid             = np.isfinite(distances) & (nearest < len(candidate_numpy))

        if mutual and np.any(valid):
            reverse = cKDTree(reference_numpy).query(candidate_numpy, k=1)[1]
            valid_indices = reference_indices[valid]
            valid_nearest = nearest[valid].astype(np.int64, copy=False)
            mutual_valid  = reverse[valid_nearest] == valid_indices
            valid[valid_indices] = mutual_valid

        reference_indices = reference_indices[valid]
        view_indices      = nearest[valid].astype(np.int64, copy=False)
        matched_distances = distances[valid]

        if not len(reference_indices):
            raise ValueError("surface views have no bounded nearest-neighbour correspondence")

        reference_tensor = torch.from_numpy(reference_indices)
        view_tensor      = torch.from_numpy(view_indices)
        cosine = torch.sum(
            normals_a[reference_tensor] * normals_b[view_tensor],
            dim=1,
        ) / (normal_norm_a[reference_tensor] * normal_norm_b[view_tensor])
        compatible = cosine >= minimum_normal_cosine
        if not torch.any(compatible):
            raise ValueError("surface views have no normal-compatible correspondence")

        return {
            "reference_indices": reference_tensor[compatible].to(torch.long),
            "view_indices":      view_tensor[compatible].to(torch.long),
            "distances":         torch.from_numpy(matched_distances).to(torch.float32)[compatible],
            "normal_cosines":    cosine.to(torch.float32)[compatible],
        }
