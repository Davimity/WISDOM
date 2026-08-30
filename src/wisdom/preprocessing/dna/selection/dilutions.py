"""Create nested group-wise training dilutions for learning curves."""

import hashlib

from typing import Any
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence


def create_dilutions(
    rows      : Sequence[Mapping[str, Any]],
    fractions : Sequence[float],
    replicates: int,
    seed      : int,
) -> dict[str, Any]:
    """Build nested training subsets without changing validation or test.

    Args:
        rows: Canonical rows with fixed split and leakage group.
        fractions: Requested fractions in ``(0, 1]``; order is irrelevant.
        replicates: Number of deterministic alternative group orderings.
        seed: Base reproducibility seed.

    Returns:
        Replicate/subset membership and fixed evaluation-set digests.

    Raises:
        ValueError: If fractions, replicates, or the training population are invalid.
    """
    requested = sorted(set(float(value) for value in fractions))
    if (
        not requested
        or requested[-1] != 1.0
        or any(value <= 0.0 or value > 1.0 for value in requested)
    ):
        raise ValueError("dilution fractions must be unique values in (0, 1] including 1.0")
    if replicates < 1:
        raise ValueError("dilution_replicates must be positive")

    train = [dict(row) for row in rows if row["split"] == "train"]
    if not train:
        raise ValueError("training dilution requires a non-empty train split")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train:
        groups[str(row["leakage_group"])].append(row)

    result: dict[str, Any] = {
        "replicates":        {},
        "test_sha256":       _identifier_hash(rows, "test"),
        "validation_sha256": _identifier_hash(rows, "validation"),
    }
    for replicate in range(replicates):
        ordering = _group_order(groups, seed + replicate)
        subsets: dict[str, Any] = {}
        for fraction in requested:
            target = round(len(train) * fraction)
            chosen: list[str] = []
            size = 0
            for group in ordering:
                if size >= target and fraction < 1.0:
                    break
                chosen.extend(str(row["identifier"]) for row in groups[group])
                size += len(groups[group])
            members    = sorted(chosen)
            member_set = set(members)
            key        = f"train-{round(100 * fraction)}"
            selected   = [row for row in train if str(row["identifier"]) in member_set]

            subsets[key] = {
                "counts":         _class_counts(selected),
                "fraction":       fraction,
                "identifiers":    members,
                "leakage_groups": sorted({str(row["leakage_group"]) for row in selected}),
            }
        result["replicates"][f"replicate-{replicate:02d}"] = subsets
    return result


def _group_order(
    groups: Mapping[str, Sequence[Mapping[str, Any]]],
    seed  : int,
) -> list[str]:
    """Choose a deterministic nested order that preserves class and phenotype coverage.

    Args:
        groups: Complete train leakage groups.
        seed: Replicate-specific tie-breaker seed.

    Returns:
        Every group exactly once in nested-selection order.
    """
    remaining = set(groups)
    ordering: list[str] = []
    labels: Counter[str] = Counter()
    global_phenotypes: Counter[str] = Counter()
    interface_phenotypes: Counter[str] = Counter()
    origins: Counter[str] = Counter()
    while remaining:
        chosen = min(
            remaining,
            key=lambda group: (
                sum(labels[str(row["label"])] for row in groups[group]),
                sum(global_phenotypes[str(row["global_phenotype"])] for row in groups[group]),
                sum(interface_phenotypes[str(row["interface_phenotype"])] for row in groups[group]),
                sum(origins[str(row["origin"])] for row in groups[group]),
                _rank(seed, group),
            ),
        )
        ordering.append(chosen)
        remaining.remove(chosen)
        for row in groups[chosen]:
            labels[str(row["label"])] += 1
            global_phenotypes[str(row["global_phenotype"])] += 1
            interface_phenotypes[str(row["interface_phenotype"])] += 1
            origins[str(row["origin"])] += 1
    return ordering


def _identifier_hash(rows: Sequence[Mapping[str, Any]], split: str) -> str:
    """Hash one fixed evaluation membership.

    Args:
        rows: Canonical split rows.
        split: Evaluation role to select.

    Returns:
        SHA-256 over newline-delimited sorted identifiers.
    """
    identifiers = sorted(str(row["identifier"]) for row in rows if row["split"] == split)
    return hashlib.sha256(("\n".join(identifiers) + "\n").encode()).hexdigest()


def _rank(seed: int, value: str) -> str:
    """Return one stable seeded lexical rank.

    Args:
        seed: Replicate-specific seed.
        value: Leakage-group identity.

    Returns:
        SHA-256 hexadecimal rank.
    """
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _class_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Count total, positive, and negative proteins.

    Args:
        rows: Protein rows carrying binary labels.

    Returns:
        Three integer counts.
    """
    positives = sum(int(row["label"]) == 1 for row in rows)
    return {"total": len(rows), "positive": positives, "negative": len(rows) - positives}
