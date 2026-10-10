"""Pickle-free feature alignment and train-only statistical normalization."""

import json
import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence
from wisdom.features.SurfaceFeatureSchema import SurfaceFeatureSchema
from wisdom.features.SurfaceFeatureProjector import SurfaceFeatureProjector


class SurfaceFeatureSidecar:
    """Bind a fixed field matrix to exact universal geometry and training statistics."""

    @staticmethod
    def write(base: Path, output: Path, names: Sequence[str], sigma: float = 2.0) -> None:
        """Build a label-free sidecar without changing any universal archive byte.

        Args:
            base: Immutable protein-only NPZ.
            output: Framework-managed build destination.
            names: Ordered requested channel names.
            sigma: Gaussian kernel width in ångströms.

        Raises:
            ValueError: If scientific projection fails.
            OSError: If a file cannot be read or written.
        """
        with np.load(base, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in (
                "atomic_numbers", "atom_names", "residue_names", "formal_charges",
                "atom_edge_index", "atom_edge_bond_order", "atom_edge_is_covalent",
                "surface_atom_neighbors", "surface_atom_distances", "surface_atom_mask",
                "atom_positions",
            )}
            config = json.loads(str(archive["metadata_json"].item()))["config"]
        names  = SurfaceFeatureSchema.resolve(names)
        values = SurfaceFeatureProjector(sigma).project(arrays, names)
        metadata = {
            "schema_version": SurfaceFeatureSchema.VERSION,
            "point_count": len(values),
            "projection": "gaussian_stored_atom_neighborhood",
            "kernel": {"sigma_angstrom": sigma, "radius_angstrom": config["surface_atom_radius"],
                       "neighbor_width": arrays[
                "surface_atom_neighbors"].shape[1]},
            "channels": [SurfaceFeatureSchema.describe(name) for name in names],
        }
        with output.open("wb") as stream:
            np.savez_compressed(
                stream, feature_values=values, feature_names=np.asarray(names),
                base_npz_sha256=np.asarray(hashlib.sha256(base.read_bytes()).hexdigest()),
                metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
            )

    @staticmethod
    def read(path: Path, base: Path, names: Sequence[str], verify_hash: bool = True) -> np.ndarray:
        """Validate alignment and select channels in the requested order.

        Args:
            path: Feature sidecar NPZ, loaded with pickle disabled.
            base: Corresponding immutable universal NPZ.
            names: Ordered channel subset.
            verify_hash: Rehash base bytes; publication uses true. Managed epoch ingestion may
                use false because Registry asset verification already established byte identity.

        Returns:
            Float32 [M,K] selected raw values.

        Raises:
            ValueError: If schema, shapes, channels, finite values, or fingerprint disagree.
        """
        with np.load(path, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            names_in = archive["feature_names"].tolist()
            values   = archive["feature_values"]
            digest   = str(archive["base_npz_sha256"].item())
        with np.load(base, allow_pickle=False) as archive:
            count = len(archive["surface_area_weights"])
        SurfaceFeatureSchema.validate_stored(names_in, metadata["schema_version"])
        if values.shape != (count, len(names_in)) or metadata["point_count"] != count:
            raise ValueError("surface feature sidecar point count/shape mismatch")
        if len(set(names_in)) != len(names_in) or not np.isfinite(values).all():
            raise ValueError("surface feature channels must be unique and finite")
        metadata_names = [channel["name"] for channel in metadata["channels"]]
        if values.dtype != np.float32 or metadata_names != names_in:
            raise ValueError("surface feature values/metadata violate the float32 channel contract")
        if verify_hash and hashlib.sha256(base.read_bytes()).hexdigest() != digest:
            raise ValueError("surface feature base NPZ hash mismatch")
        if not set(names) <= set(names_in):
            raise ValueError(f"surface feature sidecar lacks {sorted(set(names) - set(names_in))}")
        return values[:, [names_in.index(name) for name in names]].astype(np.float32)

    @staticmethod
    def validate(path: Path, base: Path, names: Sequence[str]) -> bool:
        """Check a managed projection before cache reuse or publication.

        Args:
            path: Candidate sidecar path.
            base: Exact immutable universal archive.
            names: Expected channel subset in researcher-authored order.

        Returns:
            True after complete shape/channel/hash checks.

        Raises:
            ValueError: If read detects a scientific alignment failure.
        """
        SurfaceFeatureSidecar.read(path, base, names)
        return True

    @staticmethod
    def fit_statistics(
        records: Sequence[tuple[str, Path, Path]], names: Sequence[str],
        weighting: str = "pooled_points", population: str = "full",
    ) -> dict[str, object]:
        """Fit one frozen training population using stable weighted covariance merging.

        For point weights w, mu=sum(w*x)/sum(w) and covariance is the population
        weighted second moment about mu. Each protein's sufficient statistics are merged
        without retaining all surfaces. Audit zero counts remain unweighted point counts.

        Args:
            records: Exact training ID/base/sidecar triples, never validation or test.
            names: Ordered normalization channels.
            weighting: pooled_points gives every point weight one; pooled_area uses positive
                surface areas; equal_protein gives every protein total weight one.
            population: Exact training view name, full or a published dilution.

        Returns:
            JSON mean/std, sources and a descriptive support/correlation audit. Nothing is
            deleted automatically. Constant channels normalize to zero through epsilon.

        Raises:
            ValueError: Empty population, invalid weights or non-finite/misaligned sidecars.
        """
        if weighting not in {"pooled_points", "pooled_area", "equal_protein"}:
            raise ValueError("unknown surface feature normalization weighting")
        width       = len(names)
        count       = 0
        weight_sum  = 0.0
        mean        = np.zeros(width, dtype=np.float64)
        moment      = np.zeros((width, width), dtype=np.float64)
        minimum     = np.full(width, np.inf)
        maximum     = np.full(width, -np.inf)
        zero_count  = np.zeros(width, dtype=np.int64)
        sources: list[dict[str, object]] = []

        for identifier, base, path in records:
            values = SurfaceFeatureSidecar.read(path, base, names).astype(np.float64)
            n      = len(values)
            if n == 0:
                raise ValueError("training surface contains no points")
            weights = np.ones(n, dtype=np.float64)
            if weighting == "pooled_area":
                with np.load(base, allow_pickle=False) as archive:
                    weights = archive["surface_area_weights"].astype(np.float64)
            elif weighting == "equal_protein":
                weights /= n
            if weights.shape != (n,) or not np.isfinite(weights).all() or np.any(weights <= 0):
                raise ValueError("normalization requires finite positive point weights")

            # Merge centered second moments rather than subtracting two large raw moments.

            local_weight = float(weights.sum())
            local_mean   = (values * weights[:, None]).sum(axis=0) / local_weight
            centered     = values - local_mean
            delta        = local_mean - mean
            following    = weight_sum + local_weight
            moment += (centered * weights[:, None]).T @ centered
            moment += np.outer(delta, delta) * weight_sum * local_weight / following
            mean += delta * local_weight / following
            weight_sum = following
            count += n
            minimum = np.minimum(minimum, values.min(axis=0))
            maximum = np.maximum(maximum, values.max(axis=0))
            zero_count += (values == 0).sum(axis=0)
            sources.append({
                "id": identifier,
                "base_npz_sha256": hashlib.sha256(base.read_bytes()).hexdigest(),
                "feature_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "point_count": n,
            })
        if count == 0:
            raise ValueError("feature statistics require non-empty train points")
        covariance = moment / weight_sum
        std        = np.sqrt(np.maximum(np.diag(covariance), 0))
        denominator = std[:, None] * std[None, :]
        correlation = np.divide(covariance, denominator, out=np.zeros_like(covariance),
                                where=denominator > 0)
        constant = [name for name, deviation, low, high in
                    zip(names, std, minimum, maximum, strict=True)
                    if deviation <= 1e-8 * max(1.0, abs(low), abs(high))]
        unsupported = [name for name, zeros in zip(names, zero_count, strict=True)
                       if zeros == count]
        related = [
            {"left": names[i], "right": names[j], "correlation": float(correlation[i, j]),
             "reason": "possible_duplicate_or_linear_dependency"}
            for i in range(width) for j in range(i + 1, width)
            if std[i] > 0 and std[j] > 0 and abs(correlation[i, j]) >= 0.999
        ]
        return {
            "schema_version": "1.0", "split": "train", "population": population,
            "feature_names": list(names), "count": count, "mean": mean.tolist(),
            "std": std.tolist(), "epsilon": 1e-8, "weighting": weighting,
            "weight_sum": weight_sum, "sources": sources,
            "audit": {
                "minimum": minimum.tolist(), "maximum": maximum.tolist(),
                "zero_fraction": (zero_count / count).tolist(),
                "nonzero_fraction": (1 - zero_count / count).tolist(),
                "finite_fraction": [1.0] * width,
                "correlation": [[float(correlation[i, j]) if denominator[i, j] > 0 else None
                                 for j in range(width)] for i in range(width)],
                "constant_channels": constant,
                "unsupported_channels": unsupported, "highly_correlated_pairs": related,
                "interpretation": (
                    "Train-only descriptive audit. Zero support is missing local chemical "
                    "support, not proof of absence in the whole protein. High correlation "
                    "flags redundancy, not a reason to silently remove a channel."
                ),
            },
        }

    @staticmethod
    def validate_statistics(
        statistics: Mapping[str, Any], names: Sequence[str],
        populations: Mapping[str, Mapping[str, str]],
    ) -> bool:
        """Audit every frozen training fit independently of model-side normalization.

        Args:
            statistics: Historical single-full fit or schema-2 population collection.
            names: Expected ordered field vocabulary.
            populations: Exact training IDs/base digests for full and each published view.

        Returns:
            True when all declared fits have exact population coverage and valid scalars.
            Historical full-only files remain valid for full, never for a requested dilution.

        Raises:
            ValueError: Unknown population, duplicate sources, foreign geometry or malformed fit.
        """
        if statistics.get("schema_version") == "2.0":
            fits = statistics["populations"]
            if set(fits) != set(populations):
                raise ValueError("normalization populations differ from published training views")
        else:
            fits = {"full": statistics}
        for population, fitted in fits.items():
            if population not in populations:
                raise ValueError("normalization declares an unknown training population")
            sources = fitted["sources"]
            observed = {source["id"]: source["base_npz_sha256"] for source in sources}
            if len(observed) != len(sources) or observed != populations[population]:
                raise ValueError("normalization includes non-train or changed geometry")
            SurfaceFeatureSidecar.normalize(np.zeros((1, len(names))), names, fitted)
        return True

    @staticmethod
    def normalize(
        values: np.ndarray, names: Sequence[str], stats: Mapping[str, Any],
    ) -> np.ndarray:
        """Apply one frozen train transform identically to every split.

        Args:
            values: Raw float32 [M,K] fields.
            names: Column order in values.
            stats: Training-only statistics from ``fit_statistics``.

        Returns:
            Float32 (x-mu_train)/(sigma_train+epsilon), never protein-wise rescaled.

        Raises:
            ValueError: If statistics declare any non-training fit or incompatible channels.
        """
        if stats["split"] != "train":
            raise ValueError("surface feature normalization must be fitted on train only")
        order = list(stats["feature_names"])
        if (len(set(order)) != len(order) or
            np.asarray(stats["mean"]).shape != (len(order),) or
            np.asarray(stats["std"]).shape != (len(order),)):
            raise ValueError("surface feature normalization has misaligned field scalars")
        columns = [order.index(name) for name in names]
        mean = np.asarray(stats["mean"])[columns]
        std  = np.asarray(stats["std"])[columns]
        if (not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0)
            or not np.isfinite(float(stats["epsilon"])) or float(stats["epsilon"]) <= 0):
            raise ValueError("surface feature normalization statistics are invalid")
        return ((values - mean) / (std + float(stats["epsilon"]))).astype(np.float32)
