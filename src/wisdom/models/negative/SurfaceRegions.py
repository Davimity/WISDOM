"""Fixed, label-free surface geometry used only by the negative-region selector."""

import json
import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from wisdom.features.SurfaceFeatureSchema import SurfaceFeatureSchema
from wisdom.features.SurfaceFeatureProjector import SurfaceFeatureProjector
from wisdom.utils.structure.AtomicDescriptors import AtomicDescriptors
from wisdom.preprocessing.common.structure.geometry.SurfaceBuilder import SurfaceBuilder


class SurfaceRegions:
    """Own one immutable point order, proxy area measure and filtered surface graph."""

    def __init__(self, path: Path, chemistry: bool = False, sigma: float = 2.0) -> None:
        """Read structural arrays only and reconstruct the existing normal/chord graph.

        Args:
            path: Validated universal NPZ, never its task-annotation sidecar.
            chemistry: Project generic chemistry for N2 only; never feed it to the predictor.
            sigma: Existing Gaussian atom-to-point projection width, in Å.

        Raises:
            ValueError: Geometry, proxy weights or chemical support are unusable.
            KeyError: Required structural fields or the original spacing are absent.
        """
        names = {"metadata_json", "surface_positions", "surface_normals", "surface_area_weights"}
        if chemistry:
            names.update({"atomic_numbers", "atom_names", "residue_names", "formal_charges",
                "atom_edge_index", "atom_edge_bond_order", "atom_edge_is_covalent",
                "surface_atom_mask", "surface_atom_neighbors", "surface_atom_distances",
                "surface_curvatures"})
        with np.load(path, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in names}
        metadata = json.loads(str(arrays["metadata_json"].item()))
        self.points  = arrays["surface_positions"].astype(np.float64)
        self.weights = arrays["surface_area_weights"].astype(np.float64)
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0):
            raise ValueError("negative mining requires finite positive surface proxy weights")
        self.weights /= self.weights.sum()
        resolution = float(metadata["config"]["surface_resolution"])
        graph, self.warnings = SurfaceBuilder(resolution).build_graph(
            self.points, arrays["surface_normals"])
        self.resolution = resolution
        edges = graph["surface_edge_index"]
        lengths = np.linalg.norm(self.points[edges[0]] - self.points[edges[1]], axis=1)
        self.graph = csr_matrix((np.r_[lengths, lengths],
            (np.r_[edges[0], edges[1]], np.r_[edges[1], edges[0]])),
            shape=(len(self.points), len(self.points)))
        self.identity = hashlib.sha256(b"".join(value.tobytes() for value in
            (self.points, arrays["surface_normals"], self.weights, edges))).hexdigest()
        self.features: np.ndarray | None = None
        self.feature_valid: np.ndarray | None = None
        self.projection: dict[str, Any] | None = None
        if chemistry:
            # Missing neighborhood/CA support is missing evidence, not a chemical zero.

            mask = arrays["surface_atom_mask"]
            neighbors = arrays["surface_atom_neighbors"].clip(min=0)
            self.projection = {"sigma_angstrom": sigma,
                "stored_radius_angstrom": metadata["config"]["surface_atom_radius"],
                "stored_candidate_width": neighbors.shape[1], "normalized_by_volume": False}
            ca = arrays["atom_names"].astype(str) == "CA"
            supported = mask.any(axis=1)
            known = np.isin(arrays["residue_names"].astype(str), list(AtomicDescriptors.HYDROPATHY))
            all_known = np.all(~mask | known[neighbors], axis=1)
            known_ca = (np.any(mask & ca[neighbors], axis=1) &
                        np.all(~mask | ~ca[neighbors] | known[neighbors], axis=1))
            chemical = SurfaceFeatureProjector(sigma).project(
                arrays, SurfaceFeatureSchema.GENERIC[:6])
            curvature = arrays["surface_curvatures"][:, :, :2].reshape(len(self.points), -1)
            self.features = np.column_stack((curvature, chemical)).astype(np.float64)
            chemical_valid = np.column_stack((supported, *([supported & all_known] * 3),
                                               known_ca, known_ca))
            self.feature_valid = np.column_stack((np.ones_like(curvature, dtype=bool),
                                                   chemical_valid))
            if not np.isfinite(self.features).all():
                raise ValueError("N2 structural attributes must be finite")
            self.identity = hashlib.sha256(
                self.identity.encode() + self.features.tobytes() +
                self.feature_valid.tobytes()).hexdigest()

    def diagnostics(self) -> dict[str, Any]:
        """Describe actual selector connectivity/coverage without inventing biological clusters.

        Returns:
            Immutable identity, spacing in Å, node/edge/component/isolation counts, degree
            extrema, finite attribute coverage and structural graph warnings. Point area is a
            normalized proxy, never exact physical SASA.
        """
        components, _ = connected_components(self.graph, directed=False)
        degree = np.diff(self.graph.indptr)
        return {"identity": self.identity, "resolution_angstrom": self.resolution,
            "points": len(self.points), "edges": self.graph.nnz // 2,
            "components": int(components), "isolated": int((degree == 0).sum()),
            "degree_min": int(degree.min()), "degree_max": int(degree.max()),
            "degree_mean": float(degree.mean()), "warnings": self.warnings,
            "chemical_projection": self.projection,
            "attributes": ("mean/Gaussian curvature per stored scale; charge/donor/acceptor/"
                           "aromatic Gaussian densities; CA-weighted hydropathy/polarity")
                          if self.features is not None else None,
            "attribute_valid_fraction": self.feature_valid.mean(axis=0).tolist()
                if self.feature_valid is not None else None,
            "chemical_support": "explicit_per_channel_mask" if self.features is not None
                else "not_requested"}

    def components(self, probabilities: np.ndarray, threshold: float) -> list[dict[str, Any]]:
        """Find all current superlevel components, including isolated selected points.

        Args:
            probabilities: Finite operational probabilities [M] in immutable point order.
            threshold: Lower superlevel probability in (0,1).

        Returns:
            Disjoint supports with deterministic peak anchors, maxima and proxy area fractions.

        Raises:
            ValueError: A prediction has changed point count or contains invalid probabilities.
        """
        if (probabilities.shape != self.weights.shape or not np.isfinite(probabilities).all()
                or np.any((probabilities < 0) | (probabilities > 1))):
            raise ValueError("negative review predictions disagree with immutable surface geometry")
        indices = np.flatnonzero(probabilities >= threshold)
        if not len(indices):
            return []
        _, labels = connected_components(self.graph[indices][:, indices], directed=False)
        regions = []
        for label in np.unique(labels):
            support = indices[labels == label]
            anchor = int(support[np.argmax(probabilities[support])])
            regions.append({"support": support.tolist(), "anchor": anchor,
                "score": float(probabilities[anchor]),
                "area": float(self.weights[support].sum()),
                "quantiles": np.quantile(probabilities[support], [.25, .5, .75]).tolist()})
        return regions
