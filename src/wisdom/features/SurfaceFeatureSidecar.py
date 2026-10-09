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
        if metadata["schema_version"] != SurfaceFeatureSchema.VERSION:
            raise ValueError("unsupported surface feature schema")
        if values.shape != (count, len(names_in)) or metadata["point_count"] != count:
            raise ValueError("surface feature sidecar point count/shape mismatch")
        if len(set(names_in)) != len(names_in) or not np.isfinite(values).all():
            raise ValueError("surface feature channels must be unique and finite")
        metadata_names = [channel["name"] for channel in metadata["channels"]]
        if values.dtype != np.float32 or metadata_names != names_in:
            raise ValueError("surface feature values/metadata violate the float32 channel contract")
        SurfaceFeatureSchema.resolve(names_in)
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
    ) -> dict[str, object]:
        """Fit pooled-point mean/std from the explicitly supplied training population only.

        Args:
            records: Training member ID, base NPZ and raw feature sidecar triples. The caller
                must pass train only; IDs and base digests are persisted for an independent audit.
            names: Ordered normalization channels.

        Returns:
            JSON statistics with sample counts, population std, epsilon and exact train sources.

        Raises:
            ValueError: If no training points exist or an input sidecar is invalid.
        """
        count = 0
        total = np.zeros(len(names), dtype=np.float64)
        mean  = total.copy()
        m2    = total.copy()
        sources = []
        for identifier, base, path in records:
            values = SurfaceFeatureSidecar.read(path, base, names).astype(np.float64)
            n      = len(values)
            delta  = values.mean(axis=0) - mean
            following = count + n
            m2 += ((values - values.mean(axis=0))**2).sum(axis=0)
            m2 += delta**2 * count * n / following
            mean += delta * n / following
            count = following
            sources.append({"id": identifier, "base_npz_sha256": hashlib.sha256(
                base.read_bytes()).hexdigest()})
        if count == 0:
            raise ValueError("feature statistics require non-empty train points")
        return {
            "schema_version": "1.0", "split": "train", "feature_names": list(names),
            "count": count, "mean": mean.tolist(), "std": np.sqrt(m2 / count).tolist(),
            "epsilon": 1e-8, "weighting": "pooled_surface_points", "sources": sources,
        }

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
        columns = [order.index(name) for name in names]
        mean = np.asarray(stats["mean"])[columns]
        std  = np.asarray(stats["std"])[columns]
        if (not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std < 0)
            or float(stats["epsilon"]) <= 0):
            raise ValueError("surface feature normalization statistics are invalid")
        return ((values - mean) / (std + float(stats["epsilon"]))).astype(np.float32)
