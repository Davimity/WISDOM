"""Review durable native ModelSets without consulting or recovering their producer Study."""

from __future__ import annotations

import lambdaforge as lf

from typing import Any
from pathlib import Path
from wisdom.Training import Training
from lambdaforge.products import ProductInput
from collections.abc import Mapping, Sequence
from wisdom.visualization.ReviewInference import ReviewInference


class ModelSetReview(lf.Work):
    """Consume a typed wisdom/review-models:v1 product using only verified promoted artifacts."""

    analysis_profile = Training.analysis_profile

    def run(
        self,
        models                : ProductInput,
        dataset               : Path,
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
        """Forward every deliberately promoted model; never reselect seeds inside a ModelSet.

        Args:
            models: Typed native ProductInput with ModelSet kind and wisdom/review-models:v1
                contract. ProductInput.artifact verifies durable bytes; producer may be absent.
            dataset: Exact training dataset resolved independently through LF's typed inputs.
            evaluation_scope: full for complete requested splits, explicit for requested IDs only.
            protein_selection: Default automatic/diagnostic/two cases per category; automatic
                best/worst/mixed, explicit identifiers, or all are supported.
            splits: train/val/test names; validation only by default.
            allow_test: Explicit permission for descriptive test views, never model selection.
            content: predictions includes prediction/logit/GT; full adds structural channels.
            maximum_surface_points: Positive display point ceiling, not a metric subsample.
            maximum_proteins: Viewer ceiling per model/split; zero permits all with a warning.
            prediction_threshold: Hard prediction and diagnostic decision cutoff in [0,1].
            report_maximum_mib: Document budget in (0,16] MiB, omissions explicit.
            save_predictions: Persist evaluated full-precision prediction NPZ; false by default.
            batch_size: Positive proteins per inference batch.
            data_workers: Nonnegative decoding process count.
            strict_checkpoints: Missing files reported as FileNotFoundError can be audited
                instead of failing. Native product integrity errors, including missing files
                reported as ValueError, always propagate: this flag cannot bypass LF verification.

        Returns:
            Product identity, reviewed model count and HTML presentation summary.

        Raises:
            ValueError: Incompatible contract/kind, corrupt promoted artifact, dataset mismatch,
                or unauthorized test inspection. No missing producer triggers hidden training.
            FileNotFoundError: Missing durable model bytes in strict mode.
        """
        if "test" in splits and not allow_test:
            raise ValueError("test inspection requires allow_test=true")
        if (
            models.product.kind != "ModelSet"
            or models.product.contract.identifier != "wisdom/review-models:v1"
        ):
            raise ValueError("models requires a ModelSet with wisdom/review-models:v1 contract")
        inputs = models.product.scientific_meaning["inputs"]
        ReviewInference.matching_dataset(inputs, self.inputs["dataset"])
        descriptors = []
        for index, row in enumerate(models.payload["models"]):
            # The artifact API verifies promoted files, not paths under a former Run directory.

            try:
                checkpoint = models.artifact(row["artifact"])
            except FileNotFoundError:
                if strict_checkpoints:
                    raise
                checkpoint = None
            descriptors.append(
                {
                    **row,
                    "checkpoint": checkpoint,
                    "inputs": inputs,
                    "label": f"Model {index + 1}: {row.get('group', {})} / seed {row.get('seed')}",
                    "sha256": next(
                        a.sha256 for a in models.product.artifacts if a.name == row["artifact"]
                    ),
                }
            )
        return ReviewInference().run(
            self,
            dataset,
            descriptors,
            {
                "product": models.content_id,
                "source_kind": "ModelSet",
                "product_name": models.product.name,
                "selection": models.product.scientific_meaning["selection"],
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
