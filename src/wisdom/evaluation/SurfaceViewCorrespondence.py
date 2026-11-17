"""Approximate physical correspondence between two surface discretizations."""

import torch
import numpy as np

from pathlib import Path
from torch import Tensor
from scipy.spatial import cKDTree
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.surface.SurfaceView import SurfaceView


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

    def align_assets(
        self, base: Path, view: Path, maximum_distance: float = 1.5,
        minimum_normal_cosine: float = 0.5,
    ) -> dict[str, Tensor]:
        """Verify shared physical identity before matching a base cloud to an alternate view.

        Args:
            base: Immutable universal archive defining physical atoms and reference surface.
            view: Base-bound surface_view asset; a different protein's view is rejected.
            maximum_distance: Bounded nearest-neighbour threshold in Å.
            minimum_normal_cosine: Accepted normal cosine in [-1,1].

        Returns:
            Mutual sparse alignment plus reference/view coverage masks and area-weighted coverage.

        Raises:
            ValueError: Incompatible base identity or no geometrically compatible pairs.
        """
        arrays, header = StructuralAsset.read(view, base, "surface_view")
        SurfaceView(**header["parameters"]).validate(view, base)
        with np.load(base, allow_pickle=False) as stored:
            reference = {name: torch.from_numpy(stored[name]) for name in (
                "surface_positions", "surface_normals", "surface_area_weights")}
        result = self.align(reference["surface_positions"], reference["surface_normals"],
            torch.from_numpy(arrays["surface_positions"]),
            torch.from_numpy(arrays["surface_normals"]), maximum_distance, minimum_normal_cosine)
        for prefix, weights, indices in (
            ("reference", reference["surface_area_weights"], result["reference_indices"]),
            ("view", torch.from_numpy(arrays["surface_area_weights"]), result["view_indices"]),
        ):
            mask = torch.zeros(len(weights), dtype=torch.bool)
            mask[indices] = True
            result[prefix + "_covered"] = mask
            result[prefix + "_area_coverage"] = weights[mask].sum() / weights.sum()
        return result

    @staticmethod
    def transport(
        source_values: Tensor, correspondence: dict[str, Tensor], target_count: int,
    ) -> tuple[Tensor, Tensor]:
        """Transport a scalar field by matched nearest-neighbour interpolation, with coverage.

        Args:
            source_values: Finite scalar reference field [M], independent of any functional label.
            correspondence: Mutual alignment from align/align_assets, with unique target indices.
            target_count: Number of points in the other discretization, not the reference count.

        Returns:
            CPU target field [N] and valid [N]. Uncovered entries are zero placeholders, never
            observed zero. Constant fields transport exactly on coverage; smooth/localized fields
            incur sampling error that must be measured separately from neural prediction error.

        Raises:
            ValueError: Non-finite fields, out-of-range indices or non-mutual many-to-one mapping.
        """
        values = source_values.detach().cpu()
        source = correspondence["reference_indices"].long().cpu()
        target = correspondence["view_indices"].long().cpu()
        if values.ndim != 1 or not torch.isfinite(values).all() or len(source) != len(target) \
                or target_count < 1 or len(torch.unique(target)) != len(target):
            raise ValueError("scalar transport requires a finite field and one-to-one coverage")
        if len(source) and (source.min() < 0 or source.max() >= len(values) or target.min() < 0
                            or target.max() >= target_count):
            raise ValueError("surface transport index out of range")
        result = torch.zeros(target_count, dtype=values.dtype)
        valid = torch.zeros(target_count, dtype=torch.bool)
        result[target] = values[source]
        valid[target] = True
        return result, valid
