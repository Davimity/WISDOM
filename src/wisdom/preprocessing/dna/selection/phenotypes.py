"""Assign descriptive physical phenotypes with LambdaForge HDBSCAN."""


from typing import Any
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.phenotypes import GLOBAL_FEATURES, cluster_phenotypes

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
    global_result = cluster_phenotypes(
        eligible,
        GLOBAL_FEATURES,
        "G",
        global_min_cluster_size,
        global_min_samples,
        workers,
    )
    interface_result = cluster_phenotypes(
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
            row["local_phenotype"] = interface_result["labels"].get(
                identifier, "I_NOISE"
            )
            row["local_phenotype_probability"] = interface_result["probabilities"].get(
                identifier, 0.0
            )
        else:
            row["local_phenotype"] = "not_applicable"
            row["local_phenotype_probability"] = 0.0
        assigned.append(row)
    return assigned, {
        "global":    global_result["diagnostics"],
        "interface": interface_result["diagnostics"],
    }
