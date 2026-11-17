"""Predeclared V9f contrasts on native checkpoint metadata, never training/HPO decisions."""

import csv
import json
import math

from typing import Any
from pathlib import Path
from statistics import mean, stdev
from matplotlib.figure import Figure
from collections.abc import Mapping, Sequence
from wisdom.visualization.ReviewSelection import ReviewSelection


class SchNetCampaignAudit:
    """Render fixed radial/width/metadata questions without discovering or selecting a winner."""

    SEEDS = (173, 179, 181, 191)
    CELLS = (("baseline", "gaussian", 16, 128), *(
        (kind, mode, count, width)
        for kind in ("schnet", "schnet_radial")
        for width in (128, 64)
        for mode, count in (("direct", 1), ("gaussian", 8), ("gaussian", 16), ("gaussian", 32))
        if kind == "schnet" or count in (1, 16)
    ))
    QUALITY = ("surface_score", "surface_raw_positive_macro_auprc", "global_score",
               "surface_positive_macro_top_5_enrichment", "surface_negative_positive_mass",
               "selection_regret", "global_surface_spearman", "coupling", "wisdom_score")
    COST = ("trainable_parameter_count", "atomic_encoder_trainable_parameter_count",
            "atomic_filter_parameter_count", "peak_cuda_allocated_gib", "peak_cuda_reserved_gib",
            "median_complete_epoch_seconds_excluding_first", "resource.duration_seconds",
            "resource.gpu_seconds")

    def __init__(self, execution: Mapping[str, Any]) -> None:
        """Bind one researcher-selected native Execution, never merge historical seed cohorts.

        Args:
            execution: ResultStore.select output, including Runs, Attempts and artifact metadata.

        Raises:
            ValueError: Unexpected candidates/seeds or multiple completed records for one cell/seed.
        """
        self.execution_id = execution.get("execution_id")
        self.records: dict[tuple[int, int], dict[str, Any]] = {}
        self.unavailable = []
        for row in ReviewSelection.eligible(execution["runs"]):
            parameters = row["parameters"]
            kind = parameters["atomic_encoder_type"]
            key = (kind, parameters.get("atomic_radial_encoding", "gaussian"),
                   parameters.get("atomic_radial_count", 16),
                   parameters.get("atomic_hidden_dim", 128))
            if key not in self.CELLS or row.get("seed") not in self.SEEDS:
                raise ValueError("Execution does not match the predeclared V9f SchNet matrix/seeds")
            cell, seed = self.CELLS.index(key) + 1, int(row["seed"])
            if (cell, seed) in self.records:
                raise ValueError("duplicate V9f cell/seed; do not merge independent Executions")
            artifacts = [a for a in row.get("artifacts", ()) if a["name"] == "best-model"]
            if len(artifacts) != 1:
                self.unavailable.append(
                    {"cell": cell, "seed": seed, "reason": "no unique best-model"})
                continue
            # Quality comes only from the G-selected checkpoint. Latest scalar snapshots are
            # accepted ONLY for static capacity or whole-Run costs, never as final surface quality.

            quality = artifacts[0].get("metadata", {}).get("metrics", {})
            metrics = {metric: quality.get(metric) for metric in self.QUALITY}
            metrics.update({metric: row.get("metrics", {}).get(metric) for metric in self.COST})
            self.records[cell, seed] = {
                "run_id": row["run_id"], "attempt_number": row.get("attempt_number"),
                "checkpoint_sha256": artifacts[0].get("sha256"),
                "checkpoint_epoch": artifacts[0].get("metadata", {}).get("epoch"),
                "cell": cell, "seed": seed, "metrics": metrics,
            }

    @staticmethod
    def summary(values: Sequence[float]) -> dict[str, Any]:
        """Describe available matched observations without filling gaps or inferential claims.

        Args:
            values: Finite scalar samples, already matched by the caller.

        Returns:
            Count, mean and sample SD; SD is unavailable below two samples, mean below one.
        """
        return {"n": len(values), "mean": mean(values) if values else None,
                "sd": stdev(values) if len(values) > 1 else None}

    def values(self, cell: int, metric: str) -> dict[int, float]:
        """Return finite values keyed by exact seed for one cell and one declared metric.

        Args:
            cell: One-based predeclared cell ID.
            metric: Checkpoint quality or separately declared whole-Run cost.

        Returns:
            Available observations; undefined/censored/missing evidence never becomes zero.
        """
        return {seed: float(value) for (index, seed), row in self.records.items()
                if index == cell and isinstance(value := row["metrics"].get(metric), (int, float))
                and not isinstance(value, bool) and math.isfinite(value)}

    def contrast(self, terms: Mapping[int, float], metric: str) -> dict[str, Any]:
        """Compute a predeclared within-seed linear contrast with complete matched support.

        Args:
            terms: Cell IDs and signed coefficients; e.g. {9:1,6:-1,5:-1,2:1} for interaction.
            metric: Exact checkpoint-quality or cost key.

        Returns:
            Seed deltas, mean/SD and positive/negative/tied signs. No p-value or scientific winner.
        """
        columns = {cell: self.values(cell, metric) for cell in terms}
        seeds = sorted(set.intersection(*(set(v) for v in columns.values())))
        deltas = {seed: sum(terms[cell] * columns[cell][seed] for cell in terms) for seed in seeds}
        return {**self.summary(list(deltas.values())), "by_seed": deltas,
                "positive_seeds": sum(value > 0 for value in deltas.values()),
                "negative_seeds": sum(value < 0 for value in deltas.values()),
                "tied_seeds": sum(value == 0 for value in deltas.values()),
                "complete_four_seed_contrast": len(seeds) == 4}

    def write(self, output: Path) -> dict[str, Any]:
        """Write the fixed campaign audit and descriptive plots without changing LF evidence.

        Args:
            output: New standalone report directory, not a Run/Attempt/artifact path.

        Returns:
            JSON-compatible coverage, paired comparisons, factorial contrasts and descriptive
            quality/cost dominance. Native LF remains responsible for curves, Study decisions,
            general parameter discovery and its standard report. No training is launched here.

        Raises:
            FileExistsError: Output already exists; evidence is never overwritten.
            ValueError: Output targets retained Run evidence.
        """
        if any(part in {"runs", "attempts", "artifacts"} for part in output.parts):
            raise ValueError("write audits outside retained Run evidence")
        output.mkdir(parents=True, exist_ok=False)
        coverage = {str(cell): {"completed_seeds": sorted(seed for (index, seed) in self.records
                                                        if index == cell),
                                "required_seeds": list(self.SEEDS)} for cell in range(1, 14)}
        metrics = (*self.QUALITY, *self.COST)
        summaries = {str(cell): {metric: self.summary(list(self.values(cell, metric).values()))
                                 for metric in metrics} for cell in range(1, 14)}
        paired = {str(cell): {metric: self.contrast({cell: 1, 4: -1}, metric)
                              for metric in metrics} for cell in range(1, 14) if cell != 4}
        width = {str(n): self.contrast({n + 4: 1, n: -1}, "surface_score") for n in range(2, 6)}
        metadata = {str(n): self.contrast({restricted: 1, n: -1}, "surface_score")
                    for n, restricted in ((2, 10), (4, 11), (6, 12), (8, 13))}
        interactions = {
            f"{mode}-versus-direct_width-interaction": self.contrast(
                {large + 4: 1, 6: -1, large: -1, 2: 1}, "surface_score")
            for mode, large in (("RBF8", 3), ("RBF16", 4), ("RBF32", 5))}
        # Pairwise descriptive dominance is not a scalar utility or winner selection. Require
        # all four same seeds for both coordinates, separately for each declared cost measure.

        dominance = {cost: self._dominance(summaries, cost) for cost in self.COST}
        report = {"execution_id": self.execution_id, "coverage": coverage,
                  "observations": list(self.records.values()),
                  "unavailable": self.unavailable, "cells": summaries, "paired_vs_cell04": paired,
                  "width64_minus128": width, "restricted_minus_full": metadata,
                  "descriptive_interactions": interactions,
                  "descriptively_dominated_by_cell_for_each_cost": dominance,
                  "interpretation": "Within-seed descriptive differences, not causal proof or a "
                  "StudyDecision. Costs can reflect concurrent hardware contention. Missing values "
                  "stay unavailable. Only complete four-seed cells enter dominance summaries."}
        (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
        with (output / "paired-vs-cell04.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["cell", "metric", "n", "mean_delta", "sample_sd", "positive_seeds"])
            for cell, row in paired.items():
                for metric, values in row.items():
                    writer.writerow([cell, metric, values["n"], values["mean"], values["sd"],
                                     values["positive_seeds"]])
        with (output / "relative-cost.csv").open("w", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(["cell", "reference_cell", "cost", "paired_seeds", "mean_ratio", "sd"])
            for reference in (1, 4):
                for cell_id in range(1, 14):
                    for cost in self.COST:
                        first = self.values(cell_id, cost)
                        second = self.values(reference, cost)
                        ratios = [first[seed] / second[seed]
                                  for seed in sorted(first.keys() & second)
                                  if second[seed] > 0]
                        values = self.summary(ratios)
                        writer.writerow([cell_id, reference, cost, values["n"],
                                         values["mean"], values["sd"]])
        figure = Figure(figsize=(14, 10), layout="constrained")
        axes = figure.subplots(2, 3)
        for width_index, label in ((0, "128 channels"), (4, "64 channels")):
            for axis, metric in ((axes[0, 0], "surface_score"), (axes[0, 1], "global_score")):
                radial_values = [summaries[str(n + width_index)][metric]["mean"]
                                 for n in range(2, 6)]
                axis.plot(["Direct", "RBF8", "RBF16", "RBF32"], radial_values,
                          marker="o", label=label)
                axis.set(title=metric, xlabel="Radial representation")
                axis.legend()
        for axis, metric in ((axes[0, 2], "peak_cuda_allocated_gib"),
                             (axes[1, 0], "trainable_parameter_count")):
            for cell, row in summaries.items():
                x, y = row[metric], row["surface_score"]
                if x["mean"] is not None and y["mean"] is not None:
                    axis.scatter(x["mean"], y["mean"])
                    axis.annotate(cell, (x["mean"], y["mean"]))
            axis.set(xlabel=metric, ylabel="S at G-selected checkpoint")
        for axis, contrast_values, title in ((axes[1, 1], width, "64 minus 128 channels"),
                                    (axes[1, 2], metadata, "Restricted minus full metadata")):
            axis.bar(list(contrast_values), [v["mean"] if v["mean"] is not None else float("nan")
                                             for v in contrast_values.values()])
            axis.set(title=title, xlabel="Full-width/full-metadata cell", ylabel="Paired delta S")
        figure.savefig(output / "radial-width-cost.png", dpi=140)
        return report

    def _dominance(self, summaries: Mapping[str, Any], cost: str) -> dict[str, list[int]]:
        """Compute two-coordinate descriptive dominance only on complete four-seed means.

        Args:
            summaries: Per-cell per-metric support/mean/SD computed from this Execution.
            cost: Explicit cost to minimize alongside maximizing checkpoint surface quality.

        Returns:
            Eligible cell IDs mapped to cells no worse on both coordinates and strictly better
            on at least one. This is descriptive, not a test, recommendation or StudyDecision.
        """
        coordinates = {
            cell: (row["surface_score"]["mean"], row[cost]["mean"])
            for cell, row in summaries.items()
            if row["surface_score"]["n"] == row[cost]["n"] == 4}
        return {cell: [int(other) for other, (other_s, other_c) in coordinates.items()
                       if other != cell and other_s >= quality and other_c <= resource
                       and (other_s > quality or other_c < resource)]
                for cell, (quality, resource) in coordinates.items()}
