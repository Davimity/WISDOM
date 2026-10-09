"""Explicit post-hoc policies over native checkpoint-bound evidence, never training state."""

from __future__ import annotations

import math
import json
import statistics

from typing import Any
from collections.abc import Mapping, Sequence


class ReviewSelection:
    """Keep Trial aggregation and representative-seed selection separate and reproducible."""

    @staticmethod
    def eligible(runs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
        """Retain the latest completed, unpruned, full-fidelity Attempt per native Run.

        Args:
            runs: ResultStore Run records, including native identities and attempt numbers.

        Returns:
            Copied records with a native trial_index. A newer failed Attempt never silently
            falls back to older successful weights. Missing checkpoints remain eligible here
            so the inference boundary can audit or fail according to strict_checkpoints.
        """
        latest: dict[tuple[Any, ...], Mapping[str, Any]] = {}
        for row in runs:
            key = (row["run_id"], row.get("study_phase"), json.dumps(row.get("fidelity")))
            if key not in latest or row.get("attempt_number", 1) > latest[key].get(
                "attempt_number", 1
            ):
                latest[key] = row
        return [
            {**row, "trial_index": (row.get("trial") or {}).get("index", 0)}
            for row in latest.values()
            if row.get("status") == "succeeded"
            and not row.get("pruned", False)
            and row.get("termination_type", "completed") == "completed"
            and (not row.get("fidelity") or row["fidelity"]["target"] >= row["fidelity"]["maximum"])
        ]

    @staticmethod
    def value(row: Mapping[str, Any], metric: str) -> float:
        """Read a finite metric on best-model, never the latest Run metric.

        Args:
            row: Native Run record containing one best-model artifact.
            metric: Exact artifact metadata.metrics key; no implicit alias conversion.

        Returns:
            Finite checkpoint-bound scalar.

        Raises:
            ValueError: Missing/undefined evidence, including historical artifacts. Use
                explicit/all Trial and seed policies to review such checkpoints honestly.
        """
        artifacts = [a for a in row.get("artifacts", ()) if a["name"] == "best-model"]
        value = (
            artifacts[0].get("metadata", {}).get("metrics", {}).get(metric)
            if len(artifacts) == 1
            else None
        )
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(value)
        ):
            raise ValueError(
                f"Run {row['run_id']} lacks checkpoint-bound {metric!r}; "
                "choose explicit/all Trial and seed policies for historical evidence"
            )
        return float(value)

    @classmethod
    def trials(
        cls,
        rows: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Select configurations explicitly, aggregating only artifact-bound seed metrics.

        Args:
            rows: Eligible native Runs with trial_index and parameters.
            policy: mode=all/explicit/best_by_metric/top_k_by_metric/parameter_filter/pareto.
                Ranked policies use metric, direction=max|min, k, aggregation=median|mean.
                Pareto uses metrics={name: direction} and maximum_trials. Filters use where
                with scalar/list values: absent (inactive) parameters never match explicit null.

        Returns:
            All eligible seed records of selected native Trials, in stable native-index order.

        Raises:
            ValueError: Multiple Trials without a policy, unavailable ranking evidence, empty
                selection, unknown Trial indices, or unsupported policy/aggregation/direction.
        """
        groups: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row["trial_index"], []).append(row)
        if not groups:
            raise ValueError("source has no completed, unpruned, full-fidelity Runs")
        if policy is None:
            if len(groups) != 1:
                raise ValueError("multiple Trials require an explicit trial_selection policy")
            policy = {"mode": "all"}
        mode = policy.get("mode", "all")
        chosen = sorted(groups)
        if mode == "explicit":
            chosen = list(policy["trial_indices"])
            if set(chosen) - set(groups):
                raise ValueError("explicit Trial indices are absent or ineligible")
        elif mode == "parameter_filter":
            where = policy["where"]
            chosen = [
                index
                for index in chosen
                if all(
                    key in groups[index][0]["parameters"]
                    and groups[index][0]["parameters"][key]
                    in (value if isinstance(value, list) else [value])
                    for key, value in where.items()
                )
            ]
        elif mode in ("best_by_metric", "top_k_by_metric", "pareto"):
            aggregate = {"median": statistics.median, "mean": statistics.mean}[
                policy.get("aggregation", "median")
            ]
            directions = (
                policy["metrics"]
                if mode == "pareto"
                else {policy.get("metric", "wisdom_score"): policy.get("direction", "max")}
            )
            if any(direction not in ("max", "min") for direction in directions.values()):
                raise ValueError("metric direction must be max or min")
            scores = {
                index: tuple(
                    aggregate(cls.value(row, metric) for row in groups[index])
                    * (1 if direction == "max" else -1)
                    for metric, direction in directions.items()
                )
                for index in chosen
            }
            if mode == "pareto":
                # Dominance needs no scalarization: retain G/S trade-offs instead of substituting W.

                chosen = [
                    index
                    for index in chosen
                    if not any(
                        all(a >= b for a, b in zip(scores[other], scores[index], strict=True))
                        and any(a > b for a, b in zip(scores[other], scores[index], strict=True))
                        for other in chosen
                        if other != index
                    )
                ][: policy.get("maximum_trials", 8)]
            else:
                chosen.sort(key=lambda index: (-scores[index][0], index))
                chosen = chosen[: 1 if mode == "best_by_metric" else policy.get("k", 3)]
        elif mode != "all":
            raise ValueError(f"unsupported Trial policy: {mode}")
        if not chosen:
            raise ValueError("Trial policy selected no eligible configurations")
        return [dict(row) for index in chosen for row in groups[index]]

    @classmethod
    def seeds(
        cls,
        rows: Sequence[Mapping[str, Any]],
        policy: Mapping[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Select representative replications independently inside each selected Trial.

        Args:
            rows: Selected-Trial records; no test metrics are ever consulted.
            policy: mode=all/explicit/best/worst/median/top_k/bottom_k/representative;
                metric defaults wisdom_score, direction max, k=3. Explicit uses seeds.

        Returns:
            Deterministic copied records. Median is the lower central observed replication
            in best-to-worst order, not an averaged synthetic model. Representative retains
            best, median and worst once each. All/explicit need no ranking metadata.

        Raises:
            ValueError: Unsupported direction/mode, missing requested seeds, or absent exact
                snapshot metrics. A sole seed is not automatically called median without evidence.
        """
        policy = dict(policy or {"mode": "median", "metric": "wisdom_score"})
        groups: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            groups.setdefault(row["trial_index"], []).append(row)
        result: list[dict[str, Any]] = []
        for group in groups.values():
            mode = policy.get("mode", "median")
            ordered = sorted(group, key=lambda row: (str(row.get("seed")), row["run_id"]))
            if mode == "explicit":
                requested = set(policy["seeds"])
                if requested - {row.get("seed") for row in group}:
                    raise ValueError("requested seeds are absent or ineligible in a selected Trial")
                ordered = [row for row in ordered if row.get("seed") in requested]
            elif mode != "all":
                direction = policy.get("direction", "max")
                if direction not in ("max", "min"):
                    raise ValueError("metric direction must be max or min")
                sign = -1 if direction == "max" else 1
                ordered.sort(
                    key=lambda row: sign * cls.value(row, policy.get("metric", "wisdom_score"))
                )
                median = (len(ordered) - 1) // 2
                indices = {
                    "best": [0],
                    "worst": [-1],
                    "median": [median],
                    "representative": [0, median, len(ordered) - 1],
                    "top_k": list(range(min(policy.get("k", 3), len(ordered)))),
                    "bottom_k": list(
                        range(max(0, len(ordered) - policy.get("k", 3)), len(ordered))
                    ),
                }
                if mode not in indices:
                    raise ValueError(f"unsupported seed policy: {mode}")
                ordered = [ordered[index] for index in dict.fromkeys(indices[mode])]
            result.extend(dict(row) for row in ordered)
        return result
