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
    preferred_origin_prefix: str = "",
    keep_all_positives      : bool = False,
    additional_strata      : Sequence[str] = (),
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Balance classes while avoiding concentration in one homologous/phenotype family.

    Args:
        rows: RAW rows with quality, leakage, and phenotype assignments.
        positive_negative_ratio: Requested positive count divided by negative count.
        keep_all_negatives: Keep every eligible curated negative when true.
        retain_core_positives: Prefer the task's explicitly supplied origin prefix.
        seed: Deterministic final tie-breaker.
        preferred_origin_prefix: Task-specific preferred provenance; empty disables preference.
        keep_all_positives: Retain all eligible positives instead of imposing a quota.
        additional_strata: Optional task-specific audit fields whose diversity breaks ties.

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
    if (positive_target < 1 or positive_target > len(positives)) and not keep_all_positives:
        raise RuntimeError("requested positive/negative ratio is impossible for eligible evidence")
    if keep_all_positives:
        positive_target = len(positives)

    selected_positives = _diverse(
        positives,
        positive_target,
        seed,
        retain_core_positives,
        preferred_origin_prefix,
        additional_strata,
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
            "local_phenotype": row["local_phenotype"],
        }
        for row in eligible
        if str(row["identifier"]) not in selected_ids
    ]

    return selected, {
        "omitted":                 omitted,
        "eligible":                _counts(eligible),
        "selected":                _counts(selected),
        "kept_all_negatives":      keep_all_negatives,
        "kept_all_positives":      keep_all_positives,
        "diversity_strata":        list(additional_strata),
        "positive_negative_ratio": len(selected_positives) / len(selected_negatives),
        "retained_core_positives": sum(
            bool(preferred_origin_prefix) and str(row["origin"]).startswith(preferred_origin_prefix)
            for row in selected_positives
        ),
    }


def _diverse(
    candidates : Sequence[Mapping[str, Any]],
    target     : int,
    seed       : int,
    prefer_core: bool,
    preferred_origin_prefix: str = "",
    additional_strata: Sequence[str] = (),
) -> list[dict[str, Any]]:
    """Greedily spread a fixed quota across groups, phenotypes, and origins.

    Args:
        candidates: One-class candidate population.
        target: Exact number of rows to retain.
        seed: Deterministic tie-breaker.
        prefer_core: Prefer the supplied provenance prefix when true.
        preferred_origin_prefix: Task-owned provenance preference, empty for none.
        additional_strata: Extra descriptive fields used only after group/phenotype diversity.

    Returns:
        Selected rows in deterministic choice order.
    """
    remaining = {str(row["identifier"]): dict(row) for row in candidates}
    selected: list[dict[str, Any]] = []
    group_counts: Counter[str] = Counter()
    global_counts: Counter[str] = Counter()
    local_counts: Counter[str] = Counter()
    origin_counts: Counter[str] = Counter()
    extra_counts = {field: Counter[str]() for field in additional_strata}
    while len(selected) < target:
        chosen = min(
            remaining.values(),
            key=lambda row: (
                0 if prefer_core and preferred_origin_prefix
                and str(row["origin"]).startswith(preferred_origin_prefix) else 1,
                group_counts[str(row["leakage_group"])],
                global_counts[str(row["global_phenotype"])],
                local_counts[str(row["local_phenotype"])],
                origin_counts[str(row["origin"])],
                tuple(extra_counts[field][str(row.get(field, "unknown"))]
                      for field in additional_strata),
                _rank(seed, str(row["identifier"])),
            ),
        )
        selected.append(chosen)
        remaining.pop(str(chosen["identifier"]))
        group_counts[str(chosen["leakage_group"])] += 1
        global_counts[str(chosen["global_phenotype"])] += 1
        local_counts[str(chosen["local_phenotype"])] += 1
        origin_counts[str(chosen["origin"])] += 1
        for field in additional_strata:
            extra_counts[field][str(chosen.get(field, "unknown"))] += 1
    return selected


def _rank(seed: int, value: str) -> str:
    """Return one stable lexical rank for ``value`` under ``seed``."""
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def _counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    """Return total, positive, and negative population counts for ``rows``."""
    positives = sum(int(row["label"]) == 1 for row in rows)
    return {"total": len(rows), "positive": positives, "negative": len(rows) - positives}
