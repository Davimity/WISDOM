"""Assign complete leakage groups to train, validation, and test."""

import hashlib

from typing import Any
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence

SPLITS = ("train", "validation", "test")


def assign_splits(
    rows               : Sequence[Mapping[str, Any]],
    train_fraction     : float,
    validation_fraction: float,
    test_fraction      : float,
    seed               : int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance indivisible leakage groups across the three supervised roles.

    Args:
        rows: Canonical proteins carrying leakage and phenotype labels.
        train_fraction: Target training fraction.
        validation_fraction: Target validation fraction.
        test_fraction: Target final-test fraction.
        seed: Deterministic tie-breaker.

    Returns:
        Identifier-sorted rows carrying ``split`` and a compact assignment audit.

    Raises:
        ValueError: If the fractions do not sum to one.
        RuntimeError: If validation or test cannot contain both classes without leakage.
    """
    fractions = {
        "test":       test_fraction,
        "train":      train_fraction,
        "validation": validation_fraction,
    }
    if abs(sum(fractions.values()) - 1.0) > 1e-9:
        raise ValueError("train, validation, and test fractions must sum to one")

    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row["leakage_group"])].append(dict(row))

    # These strata are descriptive rather than predictive labels. Matching their distributions
    # keeps each split physically diverse without splitting homologous proteins apart.

    strata = ("label", "global_phenotype", "interface_phenotype", "origin")
    totals = {field: Counter(str(row[field]) for row in rows) for field in strata}
    counts: dict[str, dict[str, Counter[str]]] = {
        split: {field: Counter() for field in strata}
        for split in SPLITS
    }
    sizes       = Counter({split: 0 for split in SPLITS})
    assignments: dict[str, str] = {}

    # Large and mixed-label groups are hardest to place, so assigning them first leaves smaller
    # groups available to correct the remaining size and composition deficits.

    ordered = sorted(
        groups,
        key=lambda group: (
            -len(groups[group]),
            -len({int(row["label"]) for row in groups[group]}),
            _rank(seed, group),
        ),
    )
    for group in ordered:
        group_counts = {
            field: Counter(str(row[field]) for row in groups[group])
            for field in strata
        }
        split = min(
            SPLITS,
            key=lambda candidate: (
                _placement_cost(
                    candidate,
                    len(groups[group]),
                    group_counts,
                    len(rows),
                    totals,
                    sizes,
                    counts,
                    fractions,
                ),
                _rank(seed, f"{group}:{candidate}"),
            ),
        )
        assignments[group] = split
        sizes[split] += len(groups[group])
        for field in strata:
            counts[split][field].update(group_counts[field])

    selected = sorted(
        (
            {**row, "split": assignments[str(row["leakage_group"])]}
            for row in rows
        ),
        key=lambda row: str(row["identifier"]),
    )
    for split in ("validation", "test"):
        labels = {int(row["label"]) for row in selected if row["split"] == split}
        if labels != {0, 1}:
            raise RuntimeError(
                f"{split} cannot contain both classes under the current indivisible leakage groups"
            )

    split_counts = {
        split: _class_counts([row for row in selected if row["split"] == split])
        for split in SPLITS
    }

    return selected, {
        "method":           "greedy_whole_group_stratification",
        "counts":           split_counts,
        "target_fractions": fractions,
        "assignments":      dict(sorted(assignments.items())),
        "group_counts":     dict(Counter(assignments.values())),
    }


def _placement_cost(
    split       : str,
    group_size  : int,
    group_counts: Mapping[str, Counter[str]],
    total_size  : int,
    totals      : Mapping[str, Counter[str]],
    sizes       : Counter[str],
    counts      : Mapping[str, Mapping[str, Counter[str]]],
    fractions   : Mapping[str, float],
) -> float:
    """Score one prospective group placement using incremental counters.

    Args:
        split: Prospective destination role.
        group_size: Number of proteins in the indivisible group.
        group_counts: Group category counts by stratum.
        total_size: Canonical population size.
        totals: Canonical category counts by stratum.
        sizes: Current split sizes.
        counts: Current split category counts.
        fractions: Target role fractions.

    Returns:
        Non-negative normalized deviation; lower values indicate a better placement.
    """
    fraction = fractions[split]
    target   = total_size * fraction
    current_size = sizes[split]
    next_size    = current_size + group_size
    before       = 2.0 * ((current_size - target) / max(target, 1.0)) ** 2
    after        = 2.0 * ((next_size - target) / max(target, 1.0)) ** 2
    for field, total_values in totals.items():
        for value, total in total_values.items():
            expected = total * fraction
            current  = counts[split][field][value]
            following = current + group_counts[field][value]
            weight    = 2.0 if field == "label" else 0.25
            before   += weight * ((current - expected) / max(expected, 1.0)) ** 2
            after    += weight * ((following - expected) / max(expected, 1.0)) ** 2
    return after - before


def _rank(seed: int, value: str) -> str:
    """Return one stable seeded lexical rank.

    Args:
        seed: Reproducibility seed.
        value: Group or candidate identity.

    Returns:
        SHA-256 hexadecimal rank independent from Python hash randomization.
    """
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count total, positive, and negative proteins.

    Args:
        rows: Protein rows carrying binary labels.

    Returns:
        Three integer population counts.
    """
    positives = sum(int(row["label"]) == 1 for row in rows)
    return {"total": len(rows), "positive": positives, "negative": len(rows) - positives}
