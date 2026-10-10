"""Describe label-free global morphology and individual positive zinc sites separately."""

from typing import Any
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.phenotypes import GLOBAL_FEATURES, stable_phenotypes

SITE_FEATURES = (
    "donor_count", "residue_count", "interchain", "distance_mean", "distance_std",
    "N_fraction", "O_fraction", "S_fraction", "CYS_count", "HIS_count", "GLU_count", "ASP_count",
)


def assign_phenotypes(
    rows: Sequence[Mapping[str, Any]], cluster_sizes: Sequence[int], min_samples: int,
    stability_threshold: float, workers: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Fit global morphology on both classes and local chemistry on individual accepted sites.

    Args:
        rows: Complete verified RAW population; only quality-eligible records enter clustering.
        cluster_sizes: At least two supported HDBSCAN size hypotheses, each >=2.
        min_samples: Positive density-core neighborhood count.
        stability_threshold: Minimum adjusted Rand agreement across alternative fits.
        workers: Native clustering threads, not a scientific descriptor.

    Returns:
        Copies with global/local labels, site-level local assignments, and separate support and
        stability reports. A protein with several sites retains all site labels as a sorted
        composition, never an average site. No phenotype creates a leakage edge.
    """
    assigned = [dict(row, sites=[dict(site) for site in row["sites"]],
                     global_phenotype="not_applicable", local_phenotype="not_applicable")
                for row in rows]
    eligible = [row for row in assigned if row["quality_eligible"]]
    global_labels, global_report = stable_phenotypes(
        eligible, GLOBAL_FEATURES, cluster_sizes, min_samples, stability_threshold, workers, "ZG"
    )
    site_rows = []
    for row in eligible:
        if int(row["label"]) != 1:
            continue
        for site in row["sites"]:
            if not site["accepted"]:
                continue
            donors = site["selected_donor_count"]
            elements = site["selected_element_counts"]
            counts = site["selected_residue_counts"]
            site_rows.append({
                "identifier": f"{row['identifier']}/{site['site_id']}",
                "donor_count": donors, "residue_count": len(site["selected_residues"]),
                "interchain": float(site["interchain"]),
                "distance_mean": site["coordination_distance_mean"],
                "distance_std": site["coordination_distance_std"],
                **{f"{element}_fraction": elements.get(element, 0) / donors
                   for element in ("N", "O", "S")},
                **{f"{residue}_count": counts.get(residue, 0)
                   for residue in ("CYS", "HIS", "GLU", "ASP")},
            })
    local_labels, local_report = stable_phenotypes(
        site_rows, SITE_FEATURES, cluster_sizes, min_samples, stability_threshold, workers, "ZL"
    )
    for row in assigned:
        row["global_phenotype"] = global_labels.get(row["identifier"], "ZG_NOISE")
        if int(row["label"]) == 1:
            labels = []
            for site in row["sites"]:
                if site["accepted"]:
                    label = local_labels.get(f"{row['identifier']}/{site['site_id']}", "ZL_NOISE")
                    site["local_phenotype"] = label
                    labels.append(label)
            row["local_phenotype"] = "|".join(sorted(labels)) if labels else "ZL_NO_SITE"
    return assigned, {"global_phenotype": global_report, "local_phenotype": local_report}
