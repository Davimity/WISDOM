"""Separate physical zinc-site and negative-morphology clustering, never leakage edges."""

import numpy as np

from typing import Any
from collections.abc import Mapping, Sequence
from sklearn.preprocessing import RobustScaler
from lambdaforge.clustering import HDBSCAN, stability
from wisdom.preprocessing.dna.selection.phenotypes import GLOBAL_FEATURES


def assign_phenotypes(
    rows: Sequence[Mapping[str, Any]], cluster_sizes: Sequence[int], min_samples: int,
    stability_threshold: float, workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit robust-scaled HDBSCAN independently for positive sites and negative shape.

    Args:
        rows: Complete verified RAW population; only quality-eligible rows enter phenotypes.
        cluster_sizes: At least two distinct min_cluster_size hypotheses, each >=2.
        min_samples: Positive density-core neighborhood count.
        stability_threshold: Minimum ARI across alternative fits required to retain clusters.
        workers: Native clustering threads, excluded from scientific descriptors.

    Returns:
        Rows with descriptive phenotype names and grid/stability/noise diagnostics. All-noise,
        single-cluster, small populations or unstable fits remain explicit noise, not an invented
        functional classification. These descriptors do not determine leakage membership.
    """
    assigned = [dict(row, global_phenotype="not_applicable",
                     interface_phenotype="not_applicable") for row in rows]
    reports = {}
    for label, name, features in (
        (0, "global_phenotype", GLOBAL_FEATURES),
        (1, "interface_phenotype", ("zn_site_count", "zn_donor_count", "zn_residue_count",
                                    "zn_interchain_fraction", "zn_coordination_distance_mean",
                                    "zn_coordination_distance_std", "zn_N_fraction",
                                    "zn_O_fraction", "zn_S_fraction")),
    ):
        selected = [row for row in assigned
                    if row["quality_eligible"] and int(row["label"]) == label]
        sizes = sorted({size for size in cluster_sizes if 2 <= size <= len(selected)})
        report: dict[str, Any] = {"features": list(features), "count": len(selected),
                                  "scaling": "median/IQR", "grid": sizes,
                                  "robust_multicluster": False, "noise_fraction": 1.0,
                                  "minimum_ARI": None}
        for row in selected:
            row[name] = "ZN_NOISE"
        if len(sizes) >= 2:
            raw = np.asarray([[float(row[field]) for field in features] for row in selected])
            values = RobustScaler().fit_transform(raw)
            models = [HDBSCAN(min_cluster_size=size, min_samples=min_samples, threads=workers)
                      for size in sizes]
            reference = models[0].cluster(values)
            evidence = stability(values, models[1:], reference=reference)
            count = len(set(reference.labels.tolist()) - {-1})
            robust = count >= 2 and evidence.minimum >= stability_threshold
            report.update(minimum_ARI=evidence.minimum, cluster_count=count,
                          robust_multicluster=robust,
                          noise_fraction=float(np.mean(reference.labels == -1)),
                          center=np.median(raw, axis=0).tolist(),
                          IQR=(np.percentile(raw, 75, axis=0)
                               - np.percentile(raw, 25, axis=0)).tolist())
            if robust:
                for row, cluster in zip(selected, reference.labels, strict=True):
                    row[name] = f"ZN_{int(cluster)}" if cluster >= 0 else "ZN_NOISE"
        reports[name] = report
    return assigned, reports
