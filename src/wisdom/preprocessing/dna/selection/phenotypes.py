"""Assign descriptive physical phenotypes with LambdaForge HDBSCAN."""

import math
import warnings
import numpy as np
import lambdaforge as lf

from typing import Any
from collections.abc import Mapping, Sequence
from sklearn.preprocessing import RobustScaler

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
INTERFACE_FEATURES = (
    "binding_residue_fraction",
    "interface_region_count",
    "largest_interface_region_fraction",
    "interface_radius_normalized",
    "interface_aspect_ratio",
    "interface_positive_residue_fraction",
    "interface_negative_residue_fraction",
    "interface_polar_residue_fraction",
    "interface_hydrophobic_residue_fraction",
    "interface_aromatic_residue_fraction",
    "contacted_dna_chain_count",
    "contact_density",
)


def assign_phenotypes(
    rows                      : Sequence[Mapping[str, Any]],
    global_min_cluster_size   : int,
    global_min_samples        : int,
    interface_min_cluster_size: int,
    interface_min_samples     : int,
    workers                   : int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Cluster label-free global shape and positive-only interface descriptors.

    Args:
        rows: Full RAW rows with structural descriptors and quality status.
        global_min_cluster_size: Smallest accepted global cluster.
        global_min_samples: Global HDBSCAN core-neighbour setting.
        interface_min_cluster_size: Smallest accepted positive-interface cluster.
        interface_min_samples: Interface HDBSCAN core-neighbour setting.
        workers: Threads used by LambdaForge's clustering backend.

    Returns:
        Rows with phenotype labels/probabilities and compact clustering diagnostics.
    """
    eligible = [row for row in rows if bool(row["quality_eligible"])]
    positives = [row for row in eligible if int(row["label"]) == 1]
    global_result = _cluster(
        eligible,
        GLOBAL_FEATURES,
        "G",
        global_min_cluster_size,
        global_min_samples,
        workers,
    )
    interface_result = _cluster(
        positives,
        INTERFACE_FEATURES,
        "I",
        interface_min_cluster_size,
        interface_min_samples,
        workers,
    )
    assigned: list[dict[str, Any]] = []
    for original in rows:
        identifier = str(original["identifier"])
        row = dict(original)
        row["global_phenotype"] = global_result["labels"].get(identifier, "G_NOISE")
        row["global_phenotype_probability"] = global_result["probabilities"].get(
            identifier, 0.0
        )
        if int(row["label"]) == 1:
            row["interface_phenotype"] = interface_result["labels"].get(
                identifier, "I_NOISE"
            )
            row["interface_phenotype_probability"] = interface_result["probabilities"].get(
                identifier, 0.0
            )
        else:
            row["interface_phenotype"] = "not_applicable"
            row["interface_phenotype_probability"] = 0.0
        assigned.append(row)
    return assigned, {
        "global":    global_result["diagnostics"],
        "interface": interface_result["diagnostics"],
    }


def _cluster(
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
