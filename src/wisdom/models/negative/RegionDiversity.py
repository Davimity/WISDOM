"""Current-epoch FGW families for selection only; no prototypes enter model gradients."""

import time
import numpy as np

from typing import Any
from importlib.util import find_spec
from importlib.metadata import version
from scipy.sparse.csgraph import dijkstra
from lambdaforge.clustering import HDBSCAN
from collections.abc import Mapping, Sequence
from lambdaforge.distances import StructuredMeasure
from lambdaforge.distances import FusedGromovWassersteinDistance
from wisdom.models.negative.SurfaceRegions import SurfaceRegions


class RegionDiversity:
    """Freeze common train-only attribute scales and rebuild FGW families each review."""

    def __init__(self, geometry: Mapping[str, SurfaceRegions], options: Mapping[str, Any]) -> None:
        """Fit median/IQR on negative-training structural attributes, not predicted regions.

        Args:
            geometry: All selected training-negative surfaces with explicit chemical attributes.
            options: representatives=64, distance_scale=10 Å, alpha=.5, max_iter=200,
                min_cluster_size=5, min_samples=3, maximum_bytes=268435456 and noise_quota=1.

        Raises:
            ImportError: The native LF optimal-transport extra is not installed.
            ValueError: Invalid support or physical scaling configuration.
        """
        if find_spec("ot") is None:
            raise ImportError("N2 requires lambdaforge[optimal-transport]; reinstall WISDOM and "
                              "bootstrap the selected cluster before training")
        self.options = dict(options)
        self.maximum = int(options.get("representatives", 64))
        self.scale   = float(options.get("distance_scale", 10.0))
        self.budget  = int(options.get("maximum_bytes", 268435456))
        if self.maximum < 1 or self.scale <= 0 or self.budget < 1:
            raise ValueError("N2 representative count, Å scale and memory budget must be positive")
        if int(options.get("noise_quota", 1)) < 0:
            raise ValueError("N2 first-pass noise quota must be nonnegative")
        # Constructor errors are configuration failures, not numerical top-K fallbacks.
        # Resolve public solver/backend constraints before any training epoch is spent.

        FusedGromovWassersteinDistance(alpha=float(options.get("alpha", .5)),
            max_iter=int(options.get("max_iter", 200)),
            tol_rel=float(options.get("tol_rel", 1e-9)),
            tol_abs=float(options.get("tol_abs", 1e-9)), max_memory_bytes=self.budget)
        HDBSCAN(distance="precomputed", min_cluster_size=int(options.get("min_cluster_size", 5)),
            min_samples=int(options.get("min_samples", 3)), threads=1,
            max_pairwise_bytes=self.budget)
        features = np.concatenate([g.features for g in geometry.values()])
        validity = np.concatenate([g.feature_valid for g in geometry.values()])
        if not validity.any(axis=0).all():
            raise ValueError("N2 cannot fit a requested attribute with no valid train observations")
        self.center = np.asarray([np.median(column[valid])
            for column, valid in zip(features.T, validity.T, strict=True)])
        spread = np.asarray([np.quantile(column[valid], .75) - np.quantile(column[valid], .25)
            for column, valid in zip(features.T, validity.T, strict=True)])
        self.spread = np.where(spread > 1e-8, spread, 1.0)

    def measure(self, region: Mapping[str, Any], geometry: SurfaceRegions) -> StructuredMeasure:
        """Compress one connected support by geodesic farthest-point coverage and mass transfer.

        Args:
            region: Current component with sorted local point indices, not an old memory mask.
            geometry: Owning immutable surface and its sparse normal/chord-filtered graph.

        Returns:
            At most representatives nodes with common-scaled attributes, conserved normalized
            proxy mass and Å shortest paths divided by distance_scale. No per-region rescaling.
            Missing attributes contribute centered zero plus an explicit coverage coordinate;
            representative features preserve weighted means of both values and coverage masks.

        Raises:
            MemoryError: Representative-to-support workspace exceeds the declared byte guard.
            ValueError: A supposedly connected support has disconnected internal geometry.
        """
        support = np.asarray(region["support"], dtype=np.int64)
        count = len(support)
        limit = min(count, self.maximum)
        if 8 * (limit + 4) * count > self.budget:
            raise MemoryError("N2 geodesic representative workspace exceeds maximum_bytes")
        graph = geometry.graph[support][:, support]
        representatives: list[int] = []
        rows: list[np.ndarray] = []
        nearest = np.full(count, np.inf)
        for _ in range(limit):
            chosen = 0 if not representatives else int(np.argmax(nearest))
            representatives.append(chosen)
            distances = np.asarray(dijkstra(graph, directed=False, indices=chosen))
            if not np.isfinite(distances).all():
                raise ValueError("N2 region graph is disconnected")
            rows.append(distances)
            nearest = np.minimum(nearest, distances)
            # Distinct co-located nodes remain eligible, with point-index tie breaking.

            nearest[representatives] = -1
        distances = np.stack(rows)
        # Equal-distance assignment uses the original point index, not insertion order.

        ranks = np.argsort(representatives)
        assignment = ranks[np.argmin(distances[ranks], axis=0)]
        weights = geometry.weights[support]
        masses = np.bincount(assignment, weights=weights, minlength=limit)
        assert geometry.features is not None
        assert geometry.feature_valid is not None
        valid = geometry.feature_valid[support]
        attributes = np.where(valid, (geometry.features[support] - self.center) / self.spread, 0)
        # Centered-zero imputation is accompanied by explicit coverage coordinates. Thus a
        # missing chemical measurement is distinguishable from a measured physical zero.

        attributed = np.column_stack((attributes, valid.astype(np.float64)))
        compressed = np.zeros((limit, attributed.shape[1]))
        np.add.at(compressed, assignment, attributed * weights[:, None])
        if np.any(masses <= 0):
            raise ValueError("N2 cannot form positive representative masses for coincident points")
        compressed /= masses[:, None]
        structure = distances[:, representatives] / self.scale
        return StructuredMeasure(structure, features=compressed, weights=masses)

    def cluster(self, regions: Sequence[Mapping[str, Any]],
                geometry: Mapping[str, SurfaceRegions]) -> tuple[list[int], dict[str, Any]]:
        """Recompute both comparison directions and precomputed-distance HDBSCAN each epoch.

        Args:
            regions: All currently eligible candidates plus budget/age-valid remembered supports.
            geometry: Negative-training-only structural geometry indexed by protein identifier.

        Returns:
            Current family IDs (-1 denotes noise) and solver, medoid, fallback and cost evidence.
            A numerical/resource failure explicitly falls back to all-noise/top-K; never zero costs.
        """
        began = time.perf_counter()
        audit: dict[str, Any] = {"regions": len(regions), "fallback": False, "comparisons": 0,
            "solver_calls": 0, "pairs": [], "self_costs": [], "POT_version": version("POT")}
        try:
            if 8 * len(regions) ** 2 > self.budget:
                raise MemoryError("N2 region dissimilarity matrix exceeds maximum_bytes")
            measures = [self.measure(r, geometry[r["protein"]]) for r in regions]
            measure_bytes = sum(m.structure.nbytes + m.features.nbytes + m.weights.nbytes
                                for m in measures)
            if measure_bytes + 8 * len(regions) ** 2 > self.budget:
                raise MemoryError("N2 current measures plus pair matrix exceed maximum_bytes")
            solver = FusedGromovWassersteinDistance(
                alpha=float(self.options.get("alpha", .5)),
                max_iter=int(self.options.get("max_iter", 200)),
                tol_rel=float(self.options.get("tol_rel", 1e-9)),
                tol_abs=float(self.options.get("tol_abs", 1e-9)),
                max_memory_bytes=self.budget - measure_bytes - 8 * len(regions) ** 2)
            matrix = np.zeros((len(regions), len(regions)))
            for i, left in enumerate(measures):
                audit["solver_calls"] += 1
                self_cost = solver.solve(left, left)
                audit["self_costs"].append(self_cost.value)
                if self_cost.value > 1e-7:
                    raise RuntimeError("N2 self-comparison did not recover zero discrepancy")
                for j in range(i):
                    audit["solver_calls"] += 2
                    solves = [solver.solve(left, measures[j]), solver.solve(measures[j], left)]
                    # Retry nonconvergence once with a predefined larger iteration budget.

                    if not all(s.converged for s in solves):
                        audit["solver_calls"] += 2
                        recovery = FusedGromovWassersteinDistance(
                            alpha=solver.alpha, max_iter=4 * solver.max_iter,
                            tol_rel=solver.tol_rel, tol_abs=solver.tol_abs,
                            max_memory_bytes=solver.max_memory_bytes)
                        solves = [recovery.solve(left, measures[j]),
                                  recovery.solve(measures[j], left)]
                    if not all(s.converged for s in solves):
                        raise RuntimeError("N2 FGW failed to converge after the declared retry")
                    matrix[i, j] = matrix[j, i] = sum(s.value for s in solves) / 2
                    audit["pairs"].append({"pair": [i, j], "values": [s.value for s in solves],
                        "marginal_errors": [s.marginal_error for s in solves],
                        "iterations": [s.iterations for s in solves]})
                    audit["comparisons"] += 1
            audit["symmetrization"] = "mean_of_independent_directions"
            minimum = int(self.options.get("min_cluster_size", 5))
            samples = int(self.options.get("min_samples", 3))
            labels = [-1] * len(regions)
            if len(regions) >= max(minimum, samples):
                clustered = HDBSCAN(distance="precomputed", min_cluster_size=minimum,
                    min_samples=samples, threads=1, max_pairwise_bytes=self.budget).cluster(matrix)
                labels = clustered.labels.tolist()
                audit["clustering"] = dict(clustered.diagnostics)
            audit["family_labels"] = labels
            audit["families"] = [{"label": label,
                "members": [i for i, value in enumerate(labels) if value == label],
                "medoid": min((i for i, value in enumerate(labels) if value == label),
                    key=lambda i: (sum(matrix[i, j] for j, v in enumerate(labels)
                                       if v == label), i))}
                for label in sorted(set(labels) - {-1})]
        except (RuntimeError, ValueError, MemoryError, FloatingPointError) as error:
            labels = [-1] * len(regions)
            audit.update(fallback=True, reason=f"{type(error).__name__}: {error}")
        audit["seconds"] = time.perf_counter() - began
        return labels, audit

    @staticmethod
    def order(regions: Sequence[dict[str, Any]], labels: Sequence[int],
              noise_quota: int = 1) -> list[dict[str, Any]]:
        """Prioritize one hardest member per current family, then fill by difficulty.

        Args:
            regions: Eligible regions already sorted by score/protein/minimum point index.
            labels: Corresponding current cluster IDs; negative IDs represent individual noise.
            noise_quota: Maximum noise representatives competing in the first pass.

        Returns:
            Deterministic priority order; all-noise recovers the exact pure top-K order.
        """
        first, rest = [], []
        seen: set[int] = set()
        noise = 0
        for region, label in zip(regions, labels, strict=True):
            if (label >= 0 and label not in seen) or (label < 0 and noise < noise_quota):
                first.append(region)
                seen.add(label)
                noise += int(label < 0)
            else:
                rest.append(region)
        return first + rest
