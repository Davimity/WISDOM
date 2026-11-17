"""Masked, point-aligned travel depth, ray occlusion and associated empty-space radii."""

import json
import numpy as np

from typing import Any
from pathlib import Path
from scipy.spatial import cKDTree
from collections.abc import Mapping
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.surface.AccessibleSpace import AccessibleSpace


class SurfaceEnvironment:
    """Describe protein-only free space, never a ligand-selected pocket or a task label."""

    FAMILY = "surface_environment"
    FEATURES = ("travel_depth", "euclidean_depth", "occlusion_near", "occlusion_far",
                "empty_radius")

    def __init__(self, spacing: float = 1.0, probe: float = 1.4, rays: int = 64,
                 horizons: tuple[float, ...] = (4.0, 15.0), offset: float = .2,
                 association_radius: float = 8.0, maximum_voxels: int = 18000000,
                 hull_directions: int = 96) -> None:
        """Bind physical scales and finite approximation budgets, not a fitted geometry model.

        Args:
            spacing: Free-space voxel side length in Å, default one.
            probe: Expanded-sphere solvent probe in Å, default 1.4; must match base geometry.
            rays: Deterministic outward hemisphere directions, default 64.
            horizons: Exactly two increasing occlusion horizons in Å, default 4 and 15.
            offset: Initial outward displacement from SAS in Å, default 0.2.
            association_radius: Largest point-to-empty-center distance in Å, default eight.
            maximum_voxels: Hard allocation guard, default eighteen million cells. This changes
                feasibility, not numerical geometry; resolution is never silently changed.
            hull_directions: Supporting-plane angular resolution, default 96.

        Raises:
            ValueError: Invalid scales or angular budgets.
        """
        if len(horizons) != 2 or not 0 < horizons[0] < horizons[1] or rays < 8 or (
            min(spacing, probe, offset, association_radius) <= 0
        ) or hull_directions < 12:
            raise ValueError("invalid surface environment scales or angular budgets")
        self.parameters: dict[str, Any] = dict(spacing=spacing, probe=probe, rays=rays,
            horizons=list(horizons), offset=offset, association_radius=association_radius,
            maximum_voxels=maximum_voxels, hull_directions=hull_directions)

    def scientific_parameters(self) -> dict[str, Any]:
        """Return physical/discretization settings independently of the allocation ceiling.

        Returns:
            Parameters defining computed values, excluding maximum_voxels. That ceiling rejects
            oversized domains before allocation; it does not truncate cells or change results.
            Full parameters, including the actual ceiling, remain in each asset's provenance.
        """
        return {key: value for key, value in self.parameters.items() if key != "maximum_voxels"}

    def build(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Compute finite features/masks in exactly the immutable parent point order.

        Args:
            arrays: Protein atoms/radii and SAS positions/normals; no labels are consumed.

        Returns:
            Scalar values [M], valid masks [M], explicit closed-cavity/bulk flags, and the
            actually used outward offset. Invalid values are zero only together with false masks.
            Distances are Å; occlusion is a fraction, not SASA or binding probability.
        """
        p = self.parameters
        domain = AccessibleSpace(arrays["atom_positions"], arrays["vdw_radii"],
            float(p["probe"]), float(p["spacing"]), int(p["maximum_voxels"]),
            int(p["hull_directions"]))
        points = arrays["surface_positions"].astype(float)
        normals = arrays["surface_normals"].astype(float)
        length = np.linalg.norm(normals, axis=1)
        normals = normals / np.maximum(length[:, None], 1e-12)
        valid = np.isfinite(normals).all(1) & (length > 1e-8)
        query = points.copy()
        offsets = np.zeros(len(points))
        unresolved = valid.copy()
        for factor in (1., 2., 4., 8.):
            ids = np.flatnonzero(unresolved)
            if not len(ids):
                break
            candidate = points[ids] + factor * float(p["offset"]) * normals[ids]
            free = domain.gaps(candidate) > float(p["probe"]) + 1e-8
            chosen = ids[free]
            query[chosen] = candidate[free]
            offsets[chosen] = factor * float(p["offset"])
            unresolved[chosen] = False
        valid &= ~unresolved
        result: dict[str, np.ndarray] = {name: np.zeros(len(points), np.float32)
                                       for name in self.FEATURES}
        result.update({name + "_valid": valid.copy() for name in self.FEATURES})
        result["query_offset"] = offsets.astype(np.float32)
        result["closed_cavity"] = np.zeros(len(points), bool)
        result["bulk_solvent"] = np.zeros(len(points), bool)
        result["association_distance"] = np.zeros(len(points), np.float32)
        distances = domain.paths()
        free_ids = np.flatnonzero(domain.free)
        free_tree = cKDTree(domain.points[free_ids])
        exterior_tree = cKDTree(domain.points[domain.targets])
        center_ids = domain.centers()
        center_tree = cKDTree(domain.points[center_ids]) if len(center_ids) else None

        # Depth follows a free segment to a voxel and then unobstructed graph edges. There is
        # no interpolation across occupied cells; inaccessible cavities do not receive depth zero.

        for i in np.flatnonzero(valid):
            outside = bool(domain.outside_hull(query[i:i+1])[0])
            result["bulk_solvent"][i] = outside
            local_distance, nearest = free_tree.query(query[i], k=min(32, len(free_ids)),
                                                      distance_upper_bound=2*domain.spacing)
            local_distance, nearest = np.atleast_1d(local_distance), np.atleast_1d(nearest)
            supported = np.isfinite(local_distance) & (nearest < len(free_ids))
            cells = free_ids[nearest[supported]]
            visible = domain.visible(np.repeat(query[i:i+1], len(cells), axis=0),
                                     domain.points[cells])
            cells, local_distance = cells[visible], local_distance[supported][visible]
            connected = np.isfinite(distances[cells])
            result["travel_depth_valid"][i] = outside or bool(connected.any())
            result["closed_cavity"][i] = bool(len(cells) and not connected.any())
            if not outside and connected.any():
                result["travel_depth"][i] = np.min(distances[cells[connected]] +
                                                   local_distance[connected])
            if not outside:
                result["euclidean_depth"][i] = exterior_tree.query(query[i])[0]
            if center_tree is not None and not outside:
                candidates = center_ids[center_tree.query_ball_point(query[i],
                    float(p["association_radius"]))]
                separation = np.linalg.norm(domain.points[candidates] - points[i], axis=1)
                wall_error = np.abs(separation - (domain.clearance[candidates] - domain.probe))
                compatible = wall_error <= 2 * domain.spacing
                candidates = candidates[compatible]
                visible = domain.visible(np.repeat(query[i:i+1], len(candidates), axis=0),
                                         domain.points[candidates])
                candidates = candidates[visible]
                if len(candidates):
                    chosen = sorted(candidates, key=lambda j: (-domain.clearance[j],
                        np.linalg.norm(domain.points[j]-query[i]), int(j)))[0]
                    result["empty_radius"][i] = domain.clearance[chosen]
                    result["association_distance"][i] = np.linalg.norm(
                        domain.points[chosen]-query[i])
                else:
                    result["empty_radius_valid"][i] = False
            else:
                result["empty_radius_valid"][i] = False
        near, far = self.occlusion(domain, query[valid], normals[valid])
        result["occlusion_near"][valid], result["occlusion_far"][valid] = near, far
        return result

    def occlusion(self, domain: AccessibleSpace, points: np.ndarray,
                  normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Integrate first ray/sphere intersections over a finite outward hemisphere.

        Args:
            domain: Protein-only weighted-sphere domain; no surface triangulation is required.
            points: Free outward query positions [Q,3], Å.
            normals: Outward unit normals [Q,3].

        Returns:
            Fractions [Q] blocked before the near and far horizons. Global-axis azimuth is a
            declared finite-quadrature approximation, not an exact rotation-invariance claim.
        """
        number = int(self.parameters["rays"])
        near, far = map(float, self.parameters["horizons"])
        index = np.arange(number)
        z = (index + .5) / number
        phi = index*np.pi*(3-np.sqrt(5))
        disk = np.column_stack((np.sqrt(1-z*z)*np.cos(phi), np.sqrt(1-z*z)*np.sin(phi)))
        results = np.zeros((len(points), 2), np.float32)
        expanded = domain.radii + domain.probe
        for i, (point, normal) in enumerate(zip(points, normals, strict=True)):
            axis = np.eye(3)[np.argmin(np.abs(normal))]
            first = np.cross(normal, axis)
            first /= np.linalg.norm(first)
            second = np.cross(normal, first)
            directions = disk[:, :1]*first + disk[:, 1:]*second + z[:, None]*normal
            ids = np.asarray(domain.tree.query_ball_point(point, far+expanded.max()), int)
            delta = domain.atoms[ids]-point
            projection = directions @ delta.T
            discriminant = projection**2 - ((delta**2).sum(1)-expanded[ids]**2)[None, :]
            root = projection - np.sqrt(np.maximum(discriminant, 0))
            hit = np.where((discriminant >= 0) & (root > 1e-8), root, np.inf)
            first_hit = hit.min(1) if hit.shape[1] else np.full(number, np.inf)
            results[i] = ((first_hit <= near).mean(), (first_hit <= far).mean())
        return results[:, 0], results[:, 1]

    def write(self, base: Path, output: Path) -> None:
        """Write a new parent-bound sidecar without altering historical surface or task arrays.

        Args:
            base: Immutable universal NPZ.
            output: Temporary managed file owned and atomically published by LambdaForge.
        """
        with np.load(base, allow_pickle=False) as stored:
            metadata = json.loads(str(stored["metadata_json"].item()))
            if metadata["config"]["probe_radius"] != self.parameters["probe"]:
                raise ValueError("environment probe must match the immutable SAS definition")
            arrays = {name: stored[name] for name in (
                "atom_positions", "vdw_radii", "surface_positions", "surface_normals")}
        StructuralAsset.write(output, base, self.build(arrays), self.FAMILY, self.parameters,
            {"algorithm": "free-segment-26-stencil-outer-hull-v1", "units": "angstrom",
             "occlusion_units": "fraction", "quadrature_frame": "global-axis-finite",
             "depth_reference": "finite supporting-halfspace outer convex envelope",
             "limitations": ["finite stencil anisotropy", "voxel connectivity error",
                             "finite angular quadrature", "bounded medial-center association"]})

    def validate(self, path: Path, base: Path) -> bool:
        """Audit exact binding, fields, masks and physical bounds without a second voxel solve.

        Args:
            path: Managed output to validate or resume.
            base: Exact immutable parent archive defining point order and probe geometry.

        Returns:
            True after complete storage/schema, coverage and numerical checks. LF checks exact
            sidecar bytes separately. Algorithmic correctness is tested on analytic geometries;
            publication does not claim a second identical expensive solve is independent evidence.
            A different maximum_voxels ceiling is compatible because it affects admission only;
            the saved physical discretization, units, masks and base binding must still match.
        """
        values, header = StructuralAsset.read(path, base, self.FAMILY)
        physical = {key: value for key, value in header["parameters"].items()
                    if key != "maximum_voxels"}
        if physical != self.scientific_parameters():
            raise ValueError("surface environment configuration mismatch")
        if header.get("algorithm") != "free-segment-26-stencil-outer-hull-v1":
            raise ValueError("surface environment algorithm mismatch")
        with np.load(base, allow_pickle=False) as stored:
            count = len(stored["surface_positions"])
            metadata = json.loads(str(stored["metadata_json"].item()))
        if metadata["config"]["probe_radius"] != self.parameters["probe"]:
            raise ValueError("environment source probe mismatch")
        expected = {*self.FEATURES, *(f + "_valid" for f in self.FEATURES),
                    "query_offset", "closed_cavity", "bulk_solvent", "association_distance"}
        if set(values) != expected or any(v.shape != (count,) for v in values.values()):
            raise ValueError("surface environment fields do not match the parent point domain")
        for name in self.FEATURES:
            valid = values[name + "_valid"]
            if valid.dtype != np.bool_ or np.any(values[name] < 0) or np.any(values[name][~valid]):
                raise ValueError("invalid environmental masks, units or masked values")
        if np.any(values["occlusion_near"] > values["occlusion_far"] + 1e-7) or np.any(
            values["occlusion_far"] > 1
        ):
            raise ValueError("ray occlusion fractions violate horizon monotonicity")
        if np.any(values["closed_cavity"] & values["travel_depth_valid"]):
            raise ValueError("a closed cavity cannot have a finite exterior travel depth")
        finite = values["travel_depth_valid"] & values["euclidean_depth_valid"]
        if np.any(values["travel_depth"][finite] + 1e-4 < values["euclidean_depth"][finite]):
            raise ValueError("obstacle-respecting depth is shorter than its Euclidean reference")
        return True
