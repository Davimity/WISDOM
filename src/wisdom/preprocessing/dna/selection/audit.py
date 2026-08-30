"""Audit the final WISDOM-DNA selection before publication."""

from typing import Any
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


def audit_dataset(
    raw      : Sequence[Mapping[str, Any]],
    selected : Sequence[Mapping[str, Any]],
    dilutions: Mapping[str, Any],
) -> dict[str, Any]:
    """Check hard leakage invariants and summarize dataset composition.

    Args:
        raw: Full evidence population after structural analysis.
        selected: Balanced canonical population with fixed splits.
        dilutions: Nested training views returned by ``create_dilutions``.

    Returns:
        JSON-compatible counts, distributions, warnings, and a passing verdict.

    Raises:
        RuntimeError: If identity, class, group, or dilution invariants fail.
    """
    identifiers = [str(row["identifier"]) for row in selected]
    failures: list[str] = []
    if len(identifiers) != len(set(identifiers)):
        failures.append("the canonical population contains duplicate identifiers")

    group_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        group_splits[str(row["leakage_group"])].add(str(row["split"]))
    crossing = sorted(group for group, splits in group_splits.items() if len(splits) > 1)
    if crossing:
        failures.append(f"{len(crossing)} leakage groups cross supervised splits")

    split_rows = {
        split: [row for row in selected if row["split"] == split]
        for split in ("train", "validation", "test")
    }
    for split, rows in split_rows.items():
        if {int(row["label"]) for row in rows} != {0, 1}:
            failures.append(f"{split} does not contain both binary classes")

    train_ids = {str(row["identifier"]) for row in split_rows["train"]}
    train_groups: dict[str, set[str]] = defaultdict(set)
    for row in split_rows["train"]:
        train_groups[str(row["leakage_group"])].add(str(row["identifier"]))
    for replicate, subsets in dilutions["replicates"].items():
        previous: set[str] = set()
        for name, subset in sorted(subsets.items(), key=lambda item: item[1]["fraction"]):
            members = set(subset["identifiers"])
            if not members <= train_ids:
                failures.append(f"{replicate}/{name} contains non-training proteins")
            if not previous <= members:
                failures.append(f"{replicate}/{name} is not nested")
            fragmented = [
                group
                for group, values in train_groups.items()
                if values & members and not values <= members
            ]
            if fragmented:
                failures.append(f"{replicate}/{name} fragments {len(fragmented)} leakage groups")
            previous = members

    if failures:
        raise RuntimeError("dataset selection audit failed: " + "; ".join(failures))

    warnings: list[str] = []
    for split, rows in split_rows.items():
        counts = _class_counts(rows)
        imbalance = abs(counts["positive"] - counts["negative"]) / counts["total"]
        if imbalance > 0.10:
            warnings.append(
                f"{split} class imbalance is {imbalance:.1%}; indivisible leakage groups "
                "limit exact balance"
            )
    largest_group = max(Counter(str(row["leakage_group"]) for row in raw).values())
    if largest_group / len(raw) >= 0.05:
        warnings.append(
            f"the largest RAW leakage group contains {largest_group / len(raw):.1%} of candidates"
        )

    # Phenotypes guide the greedy split objective but never override indivisible leakage groups.
    # Missing one split is therefore recorded for interpretation instead of invalidating an
    # otherwise independent benchmark.

    phenotype_groups: dict[str, set[str]] = defaultdict(set)
    phenotype_splits: dict[str, set[str]] = defaultdict(set)
    for row in selected:
        for field in ("global_phenotype", "interface_phenotype"):
            phenotype = str(row[field])
            if phenotype.endswith("NOISE") or phenotype in {"unavailable", "not_applicable"}:
                continue

            key = f"{field}:{phenotype}"
            phenotype_groups[key].add(str(row["leakage_group"]))
            phenotype_splits[key].add(str(row["split"]))

    required_splits = {"train", "validation", "test"}
    for phenotype, groups in sorted(phenotype_groups.items()):
        observed = phenotype_splits[phenotype]
        if len(groups) >= 3 and observed != required_splits:
            warnings.append(
                f"{phenotype} occurs in {len(groups)} leakage groups but is absent from "
                f"{', '.join(sorted(required_splits - observed))}; phenotype matching is a "
                "soft split objective"
            )

    return {
        "valid":                   True,
        "failures":                [],
        "warnings":                warnings,
        "raw_counts":              _class_counts(raw),
        "selected_counts":         _class_counts(selected),
        "cross_split_leakage_groups": 0,
        "largest_raw_leakage_group":  largest_group,
        "quality_eligible_counts": _class_counts(
            [row for row in raw if bool(row["quality_eligible"])]
        ),
        "splits":                  {
            split: {
                **_class_counts(rows),
                "origins":              dict(
                    sorted(Counter(str(row["origin"]) for row in rows).items())
                ),
                "leakage_groups":       len(
                    {str(row["leakage_group"]) for row in rows}
                ),
                "global_phenotypes":    dict(
                    sorted(Counter(str(row["global_phenotype"]) for row in rows).items())
                ),
                "interface_phenotypes": dict(
                    sorted(Counter(str(row["interface_phenotype"]) for row in rows).items())
                ),
            }
            for split, rows in split_rows.items()
        },
    }


def _class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count total, positive, and negative proteins.

    Args:
        rows: Protein rows carrying binary labels.

    Returns:
        Three integer counts.
    """
    positives = sum(int(row["label"]) == 1 for row in rows)
    return {"total": len(rows), "positive": positives, "negative": len(rows) - positives}
