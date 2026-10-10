"""Describe Zn diversity and shortcut risks without changing labels or independent groups."""

import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from pathlib import Path
from collections import Counter
from collections.abc import Mapping, Sequence


def write_diversity_report(
    root: Path, raw: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write interpretable class/source/chemistry coverage and covariate diagnostics.

    Args:
        root: Managed Selection output directory receiving JSON, SVG and Markdown.
        raw: Full similarity population, including scientific exclusions.
        selected: Canonical balanced records with fixed splits and physical descriptors.

    Returns:
        JSON-compatible coverage, supported standardized mean differences and warnings.
        No matching, relabelling, resplitting, significance testing or model fitting occurs.

    Raises:
        OSError: Unwritable report destination.
        ValueError: Malformed scientific scalar or missing mandatory row field.
    """
    groups = Counter(str(row["leakage_group"]) for row in raw)
    retained = Counter(str(row["leakage_group"]) for row in selected)
    coverage: dict[str, Any] = {}
    warnings: list[str] = []

    # Sequence/structure groups measure independence, not biological families. External family
    # annotations are counted only when evidence supplies them; unknown is not a new family.

    for field in ("origin", "method", "global_phenotype", "local_phenotype",
                  "positive_confidence", "negative_confidence", "family", "pfam", "interpro",
                  "cath", "ec"):
        distribution: dict[str, Any] = {}
        for label in (0, 1):
            values = [row.get(field) for row in selected if int(row["label"]) == label]
            counts = Counter(str(value) for value in values if value not in (None, "", "unknown"))
            distribution[str(label)] = {"counts": dict(sorted(counts.items())),
                                        "missing": sum(value in (None, "", "unknown")
                                                       for value in values)}
        coverage[field] = distribution
    if all(coverage["family"][str(label)]["missing"] == sum(
        int(row["label"]) == label for row in selected) for label in (0, 1)):
        warnings.append("No functional-family annotations: leakage groups are not families.")

    covariates: dict[str, Any] = {}
    for field in ("sequence_length", "resolution", "radius_of_gyration_normalized",
                  "aspect_ratio", "packing_density", "charge_density"):
        arrays = [np.asarray([float(row[field]) for row in selected
                              if int(row["label"]) == label and row.get(field) is not None
                              and np.isfinite(float(row[field]))]) for label in (0, 1)]
        entry: dict[str, Any] = {"support": [len(values) for values in arrays]}
        entry["class_summaries"] = [{"mean": float(values.mean()),
                                     "median": float(np.median(values)),
                                     "min": float(values.min()), "max": float(values.max())}
                                    if len(values) else None for values in arrays]
        smd = None
        if all(len(values) >= 2 for values in arrays):
            deviation = float(np.sqrt(sum(values.var(ddof=1) for values in arrays) / 2))
            if deviation > 0:
                smd = float((arrays[1].mean() - arrays[0].mean()) / deviation)
        entry["smd"] = smd
        covariates[field] = entry
        if smd is not None and abs(smd) >= 0.5:
            warnings.append(f"{field}: |SMD|={abs(smd):.2f}; potential class shortcut, not proof.")

    chemistry = Counter(("/".join(f"{name}{count}" for name, count in
                        sorted(site["selected_residue_counts"].items()))
                        if "selected_residue_counts" in site else
                        "types_only:" + "/".join(site["selected_residue_names"]))
                        for row in selected if int(row["label"]) == 1
                        for site in row["sites"] if site["accepted"])
    rare = sorted(name for name, count in chemistry.items() if count < 5)
    if rare:
        warnings.append(f"{len(rare)} coordination residue classes have fewer than five sites.")
    result = {"raw_leakage_groups": len(groups), "selected_leakage_groups": len(retained),
              "raw_group_sizes": dict(sorted(groups.items())),
              "selected_group_sizes": dict(sorted(retained.items())),
              "selected_singleton_groups": sum(count == 1 for count in retained.values()),
              "coverage": coverage, "covariates": covariates,
              "site_residue_classes": dict(sorted(chemistry.items())), "warnings": warnings}
    result["site_topology"] = dict(Counter(
        "interchain" if site["interchain"] else "intrachain"
        for row in selected for site in row["sites"] if site["accepted"]))
    result["surface_evaluability"] = dict(Counter(
        str(row.get("surface_evaluability", "historical_unavailable")) for row in selected))
    (root / "diversity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # One small figure makes imbalance visible. It is not a clustering embedding or evidence
    # of independence: the transitive grouping and split audit establish those constraints.

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    splits = ("train", "validation", "test")
    positions = np.arange(3)
    for label, offset in ((0, -0.18), (1, 0.18)):
        split_counts = [sum(row["split"] == split and int(row["label"]) == label
                      for row in selected) for split in splits]
        axes[0].bar(positions + offset, split_counts, width=0.36,
                    label="Explicit negative" if label == 0 else "Verified coordination")
    axes[0].set_xticks(positions, splits)
    axes[0].set_ylabel("Proteins")
    axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(1, len(groups) + 1), sorted(groups.values(), reverse=True))
    axes[1].set_xlabel("RAW group rank, largest first")
    axes[1].set_ylabel("Proteins per leakage group")
    figure.tight_layout()
    figure.savefig(root / "diversity.svg")
    plt.close(figure)
    (root / "diversity.md").write_text(
        "# Zn diversity and shortcut audit\n\n"
        "![Split counts and RAW group tail](diversity.svg)\n\n"
        "The left panel compares protein counts, not independent sample counts. "
        "Large imbalance merits review but groups must never be broken to hide it. "
        "The right panel ranks full-RAW leakage groups: a long high tail indicates repeated "
        "sequence/structure evidence. These groups are not functional families.\n\n"
        "## Covariates\n\nSMD=(positive mean-negative mean)/sqrt((positive variance+negative "
        "variance)/2), using sample variances. Zero means similar means, not equal distributions. "
        "Positive values mean larger positives; |SMD|>=0.5 is a review flag, not a significance "
        "test or a biological rule. Fewer than two values per class or zero pooled variance "
        "makes SMD unavailable; it is never replaced by zero. Missing support remains explicit.\n\n"
        "|Variable|Negative/positive support|SMD|\n|---|---|---:|\n"
        + "".join(f"|{name}|{entry['support']}|"
                  + (f"{entry['smd']:.3f}" if entry["smd"] is not None else "unavailable")
                  + "|\n" for name, entry in covariates.items())
        + "\n## Coverage and warnings\n\nThe JSON records sources, experimental methods, "
        "physical phenotypes, confidence, optional functional annotations, and residue classes "
        "of accepted sites. Rare chemistry is reported, never removed because it is rare. "
        "Source-confounded labels remain a limitation even when class counts match. "
        "No hypothesis test treating homologous proteins as independent is reported.\n\n"
        + "\n".join(f"- {warning}" for warning in warnings) + "\n", encoding="utf-8",
    )
    return result
