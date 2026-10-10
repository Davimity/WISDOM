"""Assign descriptive physical phenotypes with LambdaForge HDBSCAN."""

import math
import warnings
import numpy as np
import lambdaforge as lf

from typing import Any
from collections.abc import Mapping, Sequence
from sklearn.preprocessing import RobustScaler
from lambdaforge.clustering import HDBSCAN, stability

GLOBAL_FEATURES = (
    "log_sequence_length",
    "radius_of_gyration_normalized",
    "aspect_ratio",
    "packing_density",
    "theoretical_isoelectric_point",
    "charge_density",
    "hydrophobic_residue_fraction",
    "polar_residue_fraction",
    "aromatic_fraction",
)


def stable_phenotypes(
    rows: Sequence[Mapping[str, Any]], features: Sequence[str],
    cluster_sizes: Sequence[int], min_samples: int, stability_threshold: float,
    workers: int, prefix: str,
) -> tuple[dict[str, str], dict[str, Any]]:
    """Keep a descriptive HDBSCAN partition only with supported cross-grid stability.

    Args:
        rows: Complete finite descriptor records with unique identifier fields.
        features: Task-selected physical scalar names, never labels or external family IDs.
        cluster_sizes: Alternative minimum cluster sizes, each at least two.
        min_samples: Density core-neighbor count.
        stability_threshold: Minimum adjusted Rand agreement required across the grid.
        workers: Native clustering threads, not part of scientific data.
        prefix: Distinct descriptive naming namespace, unrelated to leakage components.

    Returns:
        Identifier labels and a scaling/support/noise/grid audit. A constant, small, unstable,
        or single-cluster population is explicit noise, not an invented functional family.
    """
    labels = {str(row["identifier"]): f"{prefix}_NOISE" for row in rows}
    sizes  = sorted({size for size in cluster_sizes if 2 <= size <= len(rows)})
    report: dict[str, Any] = {
        "features": list(features), "count": len(rows), "scaling": "median/IQR",
        "grid": sizes, "robust_multicluster": False, "noise_fraction": 1.0,
        "minimum_ARI": None,
    }
    if len(sizes) < 2 or not rows:
        report["reason"] = "insufficient_grid_support"
        return labels, report
    raw = np.asarray([[float(row[field]) for field in features] for row in rows])
    if not np.isfinite(raw).all():
        raise ValueError("phenotype descriptors must be finite")
    scaled = RobustScaler().fit_transform(raw)
    keep   = scaled.std(axis=0) > 1e-10
    if not keep.any():
        report["reason"] = "constant_descriptors"
        return labels, report
    values = scaled[:, keep]
    models = [HDBSCAN(min_cluster_size=size, min_samples=min_samples, threads=workers)
              for size in sizes]
    reference = models[0].cluster(values)
    evidence  = stability(values, models[1:], reference=reference)
    count     = len(set(reference.labels.tolist()) - {-1})
    robust    = count >= 2 and evidence.minimum >= stability_threshold
    report.update(
        minimum_ARI=evidence.minimum, cluster_count=count, robust_multicluster=robust,
        raw_noise_fraction=float(np.mean(reference.labels == -1)),
        center=np.median(raw, axis=0).tolist(),
        IQR=(np.percentile(raw, 75, axis=0) - np.percentile(raw, 25, axis=0)).tolist(),
        retained_features=[name for name, present in zip(features, keep, strict=True) if present],
    )
    if robust:
        for row, cluster in zip(rows, reference.labels, strict=True):
            labels[str(row["identifier"])] = (
                f"{prefix}_{int(cluster)}" if cluster >= 0 else f"{prefix}_NOISE"
            )
        report["noise_fraction"] = float(np.mean(reference.labels == -1))
    report["reason"] = "stable_multicluster" if robust else "no_robust_multicluster_solution"
    return labels, report


def cluster_phenotypes(
    rows            : Sequence[Mapping[str, Any]],
    features        : Sequence[str],
    prefix          : str,
    min_cluster_size: int,
    min_samples     : int,
    workers         : int,
) -> dict[str, Any]:
    """Robust-scale complete descriptor rows and apply one HDBSCAN model.

    Args:
        rows: Population eligible for this phenotype system.
        features: Label-free physical descriptor names.
        prefix: Human-readable cluster prefix.
        min_cluster_size: HDBSCAN minimum cluster size.
        min_samples: HDBSCAN core-neighbour setting.
        workers: Backend threads.

    Returns:
        Per-identifier labels/probabilities and model diagnostics.
    """
    prepared: list[tuple[str, list[float]]] = []
    for row in rows:
        derived = dict(row)
        derived["log_sequence_length"] = math.log(float(row["sequence_length"]))
        charge = row.get("net_charge_at_pH_7")
        derived["charge_density"] = (
            float(charge) / float(row["sequence_length"]) if charge is not None else None
        )
        values = [derived.get(feature) for feature in features]
        complete = [float(value) for value in values if value is not None]
        if len(complete) == len(values) and all(math.isfinite(value) for value in complete):
            prepared.append((str(row["identifier"]), complete))

    labels = {str(row["identifier"]): f"{prefix}_NOISE" for row in rows}
    probabilities = {str(row["identifier"]): 0.0 for row in rows}
    if len(prepared) < min_cluster_size * 2:
        return {
            "labels":        labels,
            "probabilities": probabilities,
            "diagnostics":   {
                "clusters":       0,
                "eligible":       len(prepared),
                "features":       list(features),
                "noise_fraction": 1.0,
                "reason":         "too_few_complete_rows",
            },
        }

    identifiers = [identifier for identifier, _ in prepared]
    matrix      = np.asarray([values for _, values in prepared], dtype=np.float64)
    scaled      = RobustScaler().fit_transform(matrix)
    keep        = np.std(scaled, axis=0) > 1e-10
    scaled      = scaled[:, keep]
    retained    = [feature for feature, selected in zip(features, keep, strict=True) if selected]
    if not retained:
        return {
            "labels":        labels,
            "probabilities": probabilities,
            "diagnostics":   {
                "clusters":       0,
                "eligible":       len(prepared),
                "features":       [],
                "noise_fraction": 1.0,
                "reason":         "constant_features",
            },
        }

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"The default value of `copy` will change from False to True in 1\.10\.",
            category=FutureWarning,
        )
        result = lf.clustering.HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            threads=workers,
        ).cluster(scaled)
    clusters = sorted(set(result.labels.tolist()) - {-1})
    if len(clusters) >= 2:
        names = {cluster: f"{prefix}{index:03d}" for index, cluster in enumerate(clusters, 1)}
        member_probabilities = result.probabilities
        if member_probabilities is None:
            member_probabilities = np.ones(len(identifiers), dtype=np.float64)
        for identifier, cluster, probability in zip(
            identifiers,
            result.labels,
            member_probabilities,
            strict=True,
        ):
            labels[identifier] = names.get(int(cluster), f"{prefix}_NOISE")
            probabilities[identifier] = float(probability) if cluster != -1 else 0.0
    return {
        "labels":        labels,
        "probabilities": probabilities,
        "diagnostics":   {
            "clusters":       len(clusters) if len(clusters) >= 2 else 0,
            "eligible":       len(prepared),
            "features":       retained,
            "noise_fraction": (
                sum(value.endswith("NOISE") for value in labels.values()) / len(labels)
            ),
            "reason":         (
                "clustered" if len(clusters) >= 2 else "no_multi_cluster_solution"
            ),
        },
    }
