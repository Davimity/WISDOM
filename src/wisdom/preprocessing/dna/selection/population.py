"""Choose the balanced canonical population from quality-eligible RAW proteins."""

import hashlib

from typing import Any
from collections import Counter
from collections.abc import Mapping, Sequence


def select_population(
    rows                   : Sequence[Mapping[str, Any]],
    positive_negative_ratio: float,
    keep_all_negatives     : bool,
    retain_core_positives  : bool,
    seed                   : int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance classes while avoiding concentration in one homologous/phenotype family.

    Args:
        rows: RAW rows with quality, leakage, and phenotype assignments.
        positive_negative_ratio: Requested positive count divided by negative count.
        keep_all_negatives: Keep every eligible curated negative when true.
        retain_core_positives: Prefer BTD-Core positives while capacity remains.
        seed: Deterministic final tie-breaker.

    Returns:
        Identifier-sorted selected population and an omitted-member audit.
    """
    eligible  = [dict(row) for row in rows if bool(row["quality_eligible"])]
    positives = [row for row in eligible if int(row["label"]) == 1]
    negatives = [row for row in eligible if int(row["label"]) == 0]
    if not positives or not negatives:
        raise RuntimeError("canonical selection requires both quality-eligible classes")

    if keep_all_negatives:
        selected_negatives = negatives
        positive_target    = round(len(negatives) * positive_negative_ratio)
    else:
        negative_target    = min(len(negatives), int(len(positives) / positive_negative_ratio))
        selected_negatives = _diverse(negatives, negative_target, seed, False)
        positive_target    = round(len(selected_negatives) * positive_negative_ratio)
    if positive_target < 1 or positive_target > len(positives):
        raise RuntimeError("requested positive/negative ratio is impossible for eligible evidence")

    selected_positives = _diverse(
        positives,
        positive_target,
        seed,
        retain_core_positives,
    )
    selected = sorted(selected_negatives + selected_positives, key=lambda row: row["identifier"])
    selected_ids = {str(row["identifier"]) for row in selected}
    omitted = [
        {
            "label":               row["label"],
            "origin":              row["origin"],
            "identifier":          row["identifier"],
            "leakage_group":       row["leakage_group"],
            "global_phenotype":    row["global_phenotype"],
            "interface_phenotype": row["interface_phenotype"],
        }
        for row in eligible
        if str(row["identifier"]) not in selected_ids
    ]

    return selected, {
        "omitted":                 omitted,
        "eligible":                _counts(eligible),
        "selected":                _counts(selected),
        "kept_all_negatives":      keep_all_negatives,
        "positive_negative_ratio": len(selected_positives) / len(selected_negatives),
        "retained_core_positives": sum(
            str(row["origin"]).startswith("btd_core") for row in selected_positives
        ),
    }


def _diverse(
    candidates : Sequence[Mapping[str, Any]],
    target     : int,
    seed       : int,
    prefer_core: bool,
) -> list[dict[str, Any]]:
    """Greedily spread a fixed quota across groups, phenotypes, and origins.

    Args:
        candidates: One-class candidate population.
        target: Exact number of rows to retain.
        seed: Deterministic tie-breaker.
        prefer_core: Place BTD-Core candidates before other origins.

    Returns:
        Selected rows in deterministic choice order.
    """
    remaining = {str(row["identifier"]): dict(row) for row in candidates}
    selected: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    interface_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    while len(selected) < target:
        chosen = min(
            remaining.values(),
            key=lambda row: (
                0 if prefer_core and str(row["origin"]).startswith("btd_core") else 1,
                group_counts[str(row["leakage_group"])],
                global_counts[str(row["global_phenotype"])],
                interface_counts[str(row["interface_phenotype"])],
                origin_counts[str(row["origin"])],
                _rank(seed, str(row["identifier"])),
            ),
        )
        selected.append(chosen)
        remaining.pop(str(chosen["identifier"]))
        group_counts[str(chosen["leakage_group"])] += 1
        global_counts[str(chosen["global_phenotype"])] += 1
        interface_counts[str(chosen["interface_phenotype"])] += 1
        origin_counts[str(chosen["origin"])] += 1
    return selected


def _rank(seed: int, value: str) -> str:
    """Return one stable lexical rank for ``value`` under ``seed``."""
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return total, positive, and negative population counts for ``rows``."""
    positives = sum(int(row["label"]) == 1 for row in rows)
    return {"total": len(rows), "positive": positives, "negative": len(rows) - positives}
