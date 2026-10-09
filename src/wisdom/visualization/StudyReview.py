"""Inference-only review of locally registered or natively imported LambdaForge Studies."""

from __future__ import annotations

import lambdaforge as lf

from typing import Any
from pathlib import Path
from wisdom.Training import Training
from collections.abc import Mapping, Sequence
from lambdaforge.work.managed import fingerprint
from lambdaforge.work.ResultStore import ResultStore
from wisdom.visualization.ReviewInference import ReviewInference
from wisdom.visualization.ReviewSelection import ReviewSelection


class StudyReview(lf.Work):
    """Use native ResultStore evidence for flexible, explicitly described post-hoc selection."""

    analysis_profile = Training.analysis_profile

    def run(
        self,
        source_execution      : str,
        dataset               : Path,
        results_root          : Path | None = None,
        trial_selection       : Mapping[str, Any] | None = None,
        seed_selection        : Mapping[str, Any] | None = None,
        evaluation_scope      : str = "full",
        protein_selection     : Mapping[str, Any] | None = None,
        splits                : Sequence[str] = ("val",),
        allow_test            : bool = False,
        content               : str = "predictions",
        maximum_surface_points: int = 2000,
        maximum_proteins      : int = 12,
        prediction_threshold  : float = 0.5,
        report_maximum_mib    : float = 14.0,
        save_predictions      : bool = False,
        batch_size            : int = 4,
        data_workers          : int = 0,
        strict_checkpoints    : bool = True,
    ) -> dict[str, Any]:
        """Select existing Trials/seeds and evaluate their frozen best-model snapshots.

        Args:
            source_execution: Exact registered Execution ID; import portable Studies with lf import.
            dataset: Typed immutable dataset placement matching the original training identity.
            results_root: Optional typed local ResultStore root; None uses LF's project default.
            trial_selection: all, explicit trial_indices, best_by_metric, top_k_by_metric,
                parameter_filter where, or pareto metrics. Rankings aggregate seed checkpoint
                metrics by median (default) or mean; directions max/min. None requires one Trial.
            seed_selection: all, explicit seeds, best, worst, median, top_k, bottom_k, or
                representative. Default median by wisdom_score, direction max; k defaults three.
            evaluation_scope: full evaluates complete splits; explicit forwards requested IDs only.
            protein_selection: Default automatic/diagnostic/two cases per category; explicit
                identifiers or all. Automatic also supports best, worst and mixed strategies.
            splits: Explicit train/val/test partition names; validation only by default.
            allow_test: Permit descriptive test inspection, never test-driven model selection.
            content: predictions for prediction/logit/GT channels; full for structural channels too.
            maximum_surface_points: Positive display-only cap; numerical metrics use every point.
            maximum_proteins: Display cap per checkpoint/split; zero is explicitly unbounded.
            prediction_threshold: Hard prediction and protein error cutoff in [0,1].
            report_maximum_mib: Embedded document budget in (0,16] MiB; omissions are reported.
            save_predictions: Persist prediction NPZ for evaluated proteins; false saves space.
            batch_size: Positive proteins per inference batch.
            data_workers: Nonnegative decoding processes.
            strict_checkpoints: Missing selected checkpoints fail instead of being audited/skipped.

        Returns:
            Native source identity, reviewed model count, missing evidence and HTML byte summary.

        Raises:
            KeyError: Source Execution is not registered in the selected local ResultStore.
            ValueError: Ambiguous Trial policy, missing ranking metadata, dataset/weight mismatch,
                unsafe artifact path, checksum failure or unauthorized test access.
            FileNotFoundError: Missing checkpoint in strict mode. Native recovery is required;
                no review path fits a model or changes a StudyDecision.
        """
        if "test" in splits and not allow_test:
            raise ValueError("test inspection requires allow_test=true")
        if not callable(getattr(ResultStore, "execution_directory", None)):
            raise RuntimeError("StudyReview requires current LF ResultStore import/review APIs")
        store = ResultStore(results_root)
        execution = store.select(source_execution)
        root = store.execution_directory(source_execution).resolve()
        eligible = ReviewSelection.eligible(execution["runs"])
        selected = ReviewSelection.seeds(
            ReviewSelection.trials(eligible, trial_selection), seed_selection
        )
        self.outputs.file("candidate-inventory", filename="candidate-inventory.json").write_json(
            [
                {
                    "run_id": row["run_id"],
                    "trial": row["trial_index"],
                    "seed": row.get("seed"),
                    "parameters": row["parameters"],
                }
                for row in eligible
            ]
        )

        # ResultStore relocates imported Attempts. Resolve only its authoritative run_dir/path,
        # never the original remote host, private manifest internals, or a directory glob.

        models = []
        for row in selected:
            artifacts = [a for a in row.get("artifacts", ()) if a["name"] == "best-model"]
            artifact = artifacts[0] if len(artifacts) == 1 else {}
            checkpoint = None
            if artifact:
                attempt = Path(row["run_dir"]).absolute()
                relative = Path(artifact["path"])
                checkpoint = attempt / relative
                if (
                    relative.is_absolute()
                    or ".." in relative.parts
                    or checkpoint.resolve() != checkpoint
                    or not checkpoint.is_relative_to(root)
                ):
                    raise ValueError("registered checkpoint escapes its owned Execution")
                if artifact.get("role") not in ("checkpoint", "model"):
                    raise ValueError("best-model must have native checkpoint/model artifact role")
                if checkpoint.is_file() and fingerprint(checkpoint) != (
                    artifact["sha256"],
                    artifact["size_bytes"],
                ):
                    raise ValueError(
                        "registered checkpoint bytes differ from native artifact evidence"
                    )
            metadata = artifact.get("metadata", {})
            models.append(
                {
                    "label": f"Trial {row['trial_index']} / seed {row.get('seed')}",
                    "run_id": row["run_id"],
                    "seed": row.get("seed"),
                    "trial_index": row["trial_index"],
                    "parameters": row["parameters"],
                    "inputs": row.get("inputs", ()),
                    "checkpoint": checkpoint,
                    "sha256": artifact.get("sha256"),
                    "step": metadata.get("step"),
                    "metrics": metadata.get("metrics", {}),
                }
            )
        return ReviewInference().run(
            self,
            dataset,
            models,
            {
                "source_execution": execution["execution_id"],
                "source_kind": "ResultStore",
                "trial_selection": trial_selection or {"mode": "single Trial"},
                "seed_selection": seed_selection or {"mode": "median", "metric": "wisdom_score"},
            },
            splits,
            allow_test,
            evaluation_scope,
            protein_selection,
            content,
            maximum_surface_points,
            maximum_proteins,
            prediction_threshold,
            report_maximum_mib,
            save_predictions,
            batch_size,
            data_workers,
            strict_checkpoints,
        )
