"""DNA-binding evaluation invoked by LambdaForge's public post-run lifecycle."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from lambdaforge.experiments import (
    ObjectFactory,
    PostRunAction,
    PostRunContext,
    PostRunResult,
)
from torch import Tensor, nn
from torch.utils.data import DataLoader

from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite
from wisdom.evaluation.PointCloudExporter import PointCloudExporter


class WisdomDNAPostRunEvaluator(PostRunAction):
    """Evaluate global labels, unseen surface labels, and aligned 3D maps."""

    def __init__(
        self,
        threshold       : float = 0.5,
        batch_size      : int = 1,
        num_workers     : int = 0,
        device          : str = "auto",
        latent_channels : Sequence[int] = (),
        export_limit    : int = 0,
    ) -> None:
        """Configure deterministic test evaluation and controlled 3D channel export.

        Args:
            threshold: Explicit probability cutoff for all discrete predictions.
            batch_size: Number of disjoint protein graphs evaluated together.
            num_workers: Local DataLoader worker count.
            device: PyTorch device or ``auto`` for CUDA when available and CPU otherwise.
            latent_channels: Explicit learned surface embedding indices to include in PLY files.
            export_limit: Maximum proteins exported in 3D; zero exports every evaluated protein.

        Raises:
            ValueError: If worker/batch/export counts are invalid or device cannot be resolved.
        """
        if batch_size < 1 or num_workers < 0 or export_limit < 0:
            raise ValueError("post-run evaluation counts must be non-negative and batch positive")
        self.metrics         = BinaryMetricSuite(threshold)
        self.threshold       = float(threshold)
        self.batch_size      = batch_size
        self.num_workers     = num_workers
        resolved_device = (
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else "cpu"
            if device == "auto"
            else device
        )
        self.device          = torch.device(resolved_device)
        self.latent_channels = tuple(int(value) for value in latent_channels)
        self.export_limit    = export_limit

    def run(self, context: PostRunContext) -> PostRunResult:
        """Evaluate the exact selected checkpoint on the explicit test split.

        The model is rebuilt from the materialized experiment specification and receives only the
        declared ``task.params.model_input_keys``. Surface ground truth remains in the batch for
        diagnostics but is never routed into the model, loss, optimizer, checkpoint selection, or
        HPO objective. All local logits are sigmoid-transformed before probability metrics.

        Args:
            context: Immutable LambdaForge post-run context containing the materialized
                configuration, completed training result, explicit checkpoint selection, action
                identity, and safe run-relative artifact resolver.

        Returns:
            Structured LambdaForge result containing JSON outputs, scalar headline metrics, and
            provenance metadata. Declared artifacts are materialized by LambdaForge from the YAML
            contract after this method returns.

        Raises:
            ValueError: If the experiment has no test dataset, model routing, or surface annotation.
            TypeError: If public object specifications do not construct the expected objects.
            RuntimeError: If checkpoint weights or prediction shapes disagree with the model.
            FileNotFoundError: If the explicitly selected checkpoint is unavailable.
            KeyboardInterrupt: If LambdaForge requests cooperative cancellation between batches.
        """
        config     = context.config
        checkpoint = context.selected_checkpoint
        if checkpoint is None or context.selected_checkpoint_sha256 is None:
            raise FileNotFoundError("DNA post-run evaluation requires its selected checkpoint")

        data = config.get("data")
        if not isinstance(data, Mapping) or not isinstance(data.get("test"), Mapping):
            raise ValueError("post-run DNA evaluation requires data.test")
        dataset = ObjectFactory.build(data["test"])
        datamodule = data.get("datamodule")
        if not isinstance(datamodule, Mapping):
            raise ValueError("post-run DNA evaluation requires data.datamodule")
        data_params = datamodule.get("params", {})
        if not isinstance(data_params, Mapping):
            raise TypeError("data.datamodule.params must be a mapping")
        collate_spec = data_params.get("collate_fn")
        collator = (
            ObjectFactory.build(collate_spec) if isinstance(collate_spec, Mapping) else None
        )
        loader       = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collator,
        )

        model = ObjectFactory.build(config["model"])
        if not isinstance(model, nn.Module):
            raise TypeError("post-run model specification must construct torch.nn.Module")
        checkpoint_payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = (
            checkpoint_payload.get("state_dict", checkpoint_payload)
            if isinstance(checkpoint_payload, Mapping)
            else checkpoint_payload
        )
        if not isinstance(state, Mapping):
            raise TypeError("post-run checkpoint does not contain a state mapping")
        model_state = {
            str(key).removeprefix("model."): value
            for key, value in state.items()
            if str(key).startswith("model.")
        }
        model.load_state_dict(model_state or state)
        model.to(self.device).eval()

        task_spec = config.get("task", {})
        task_params = task_spec.get("params", {}) if isinstance(task_spec, Mapping) else {}
        routes = task_params.get("model_input_keys") if isinstance(task_params, Mapping) else None
        if not isinstance(routes, Mapping) or not routes:
            raise ValueError("post-run DNA evaluation requires mapped model_input_keys")

        output_root = context.artifact_path("evaluation")
        metrics_root = output_root / "metrics"
        predictions_root = output_root / "predictions"
        visualization_root = output_root / "visualizations"
        report_root = output_root / "report"
        for directory in (metrics_root, predictions_root, visualization_root, report_root):
            directory.mkdir(parents=True, exist_ok=True)

        protein_scores: list[Tensor] = []
        protein_targets: list[Tensor] = []
        surface_scores: list[Tensor] = []
        surface_targets: list[Tensor] = []
        positive_surface_scores: list[Tensor] = []
        positive_surface_targets: list[Tensor] = []
        sensitivity_scores: list[list[Tensor]] = []
        sensitivity_targets: list[list[Tensor]] = []
        sensitivity_gaps: Tensor | None = None
        per_protein: list[dict[str, Any]] = []
        global_rows: list[dict[str, Any]] = []
        exported = 0
        with torch.inference_mode():
            for batch in loader:
                # Cancellation is checked between complete batches so partially written reports are
                # never advertised as successful LambdaForge artifacts.
                if context.stop_requested:
                    raise KeyboardInterrupt
                if not isinstance(batch, Mapping):
                    raise TypeError("post-run DNA evaluation expects mapping batches")
                required_annotation = {"surface_target_hard", "surface_valid_mask"}
                if not required_annotation.issubset(batch):
                    raise ValueError("test dataset lacks DNA surface annotation sidecars")
                moved = self._move(batch, self.device)
                arguments = {
                    str(argument): moved[str(batch_key)]
                    for argument, batch_key in routes.items()
                }
                outputs = model(**arguments)
                if not isinstance(outputs, Mapping):
                    raise TypeError("WISDOM model must return a mapping")
                protein_probability = torch.sigmoid(outputs["logits"]).detach().cpu()
                local_probability   = torch.sigmoid(outputs["surface_logits"]).detach().cpu()
                local_logits        = outputs["surface_logits"].detach().cpu()
                local_embeddings    = outputs.get("surface_embeddings")
                if isinstance(local_embeddings, Tensor):
                    local_embeddings = local_embeddings.detach().cpu()

                target        = self._tensor(batch, "target").detach().cpu().long()
                surface_batch = self._tensor(batch, "surface_batch").detach().cpu().long()
                valid         = self._tensor(batch, "surface_valid_mask").detach().cpu().bool()
                hard          = self._tensor(batch, "surface_target_hard").detach().cpu().long()
                batch_sensitivity = self._tensor(
                    batch,
                    "surface_target_hard_sensitivity",
                ).detach().cpu().long()
                batch_gaps = self._tensor(batch, "sensitivity_gaps").detach().cpu().float()
                if sensitivity_gaps is None:
                    sensitivity_gaps    = batch_gaps
                    sensitivity_scores  = [[] for _ in range(len(batch_gaps))]
                    sensitivity_targets = [[] for _ in range(len(batch_gaps))]
                elif not torch.equal(sensitivity_gaps, batch_gaps):
                    raise ValueError("test sidecars disagree on sensitivity gap definitions")
                protein_scores.append(protein_probability)
                protein_targets.append(target)
                surface_scores.append(local_probability[valid])
                surface_targets.append(hard[valid])

                identifiers = batch.get("identifier")
                tiers       = batch.get("tier")
                if not isinstance(identifiers, list) or not isinstance(tiers, list):
                    raise ValueError("DNA test batches require identifier and tier lists")
                for protein_index, identifier in enumerate(identifiers):
                    point_mask  = surface_batch == protein_index
                    point_valid = valid[point_mask]
                    point_score = local_probability[point_mask]
                    point_hard  = hard[point_mask]
                    if int(target[protein_index]) == 1:
                        positive_surface_scores.append(point_score[point_valid])
                        positive_surface_targets.append(point_hard[point_valid])
                        point_sensitivity = batch_sensitivity[point_mask]
                        for index in range(len(batch_gaps)):
                            sensitivity_scores[index].append(point_score[point_valid])
                            sensitivity_targets[index].append(
                                point_sensitivity[point_valid, index]
                            )
                    row = self._protein_metrics(
                        str(identifier),
                        str(tiers[protein_index]),
                        int(target[protein_index]),
                        float(protein_probability[protein_index]),
                        point_score,
                        point_hard,
                        point_valid,
                        self._tensor(batch, "surface_area_weights").detach().cpu()[point_mask],
                        self._protein_edges(
                            self._tensor(batch, "surface_edge_index").detach().cpu(),
                            point_mask,
                        ),
                    )
                    per_protein.append(row)
                    global_rows.append(
                        {
                            "identifier": identifier,
                            "tier": tiers[protein_index],
                            "target": int(target[protein_index]),
                            "probability": float(protein_probability[protein_index]),
                            "prediction": int(protein_probability[protein_index] >= self.threshold),
                        }
                    )
                    if self.export_limit == 0 or exported < self.export_limit:
                        self._export_protein(
                            visualization_root / f"{identifier}.ply",
                            batch,
                            point_mask,
                            local_logits[point_mask],
                            point_score,
                            (
                                local_embeddings[point_mask]
                                if isinstance(local_embeddings, Tensor)
                                else None
                            ),
                        )
                        exported += 1

        global_metrics = self.metrics.compute(
            torch.cat(protein_scores),
            torch.cat(protein_targets),
        )
        surface_micro = self._compute_or_undefined(surface_scores, surface_targets)
        positive_micro = self._compute_or_undefined(
            positive_surface_scores,
            positive_surface_targets,
        )
        evaluated_gaps = sensitivity_gaps if sensitivity_gaps is not None else torch.empty(0)
        sensitivity = {
            f"{float(gap):g}_angstrom": self._compute_or_undefined(
                sensitivity_scores[index],
                sensitivity_targets[index],
            )
            for index, gap in enumerate(evaluated_gaps)
        }
        positive_rows = [row for row in per_protein if row["protein_target"] == 1]
        surface_metrics = {
            "threshold": self.threshold,
            "micro": surface_micro,
            "macro": self._macro(per_protein),
            "positive_localization": {
                "micro": positive_micro,
                "macro": self._macro(positive_rows),
                "regional": self._aggregate_rows(positive_rows),
            },
            "negative_diagnostics": self._aggregate_rows(
                [row for row in per_protein if row["protein_target"] == 0]
            ),
            "ground_truth_sensitivity_on_positives": sensitivity,
        }
        self._json(metrics_root / "global_metrics.json", global_metrics)
        self._json(metrics_root / "surface_metrics.json", surface_metrics)
        self._csv(predictions_root / "global_predictions.csv", global_rows)
        self._csv(metrics_root / "per_protein_metrics.csv", per_protein)
        training_outcome = (
            "early_stopped"
            if bool(context.result.get("trainer_stopped_early", False))
            else "completed"
        )
        summary = {
            "training_outcome": training_outcome,
            "checkpoint_role": context.selected_checkpoint_role,
            "selected_checkpoint_sha256": context.selected_checkpoint_sha256,
            "action_name": context.action_name,
            "action_identity": context.action_identity,
            "protein_count": len(per_protein),
            "local_evaluation_protein_count": sum(
                int(row["valid_surface_points"] > 0) for row in per_protein
            ),
            "local_evaluation_positive_count": sum(
                int(row["protein_target"] == 1 and row["valid_surface_points"] > 0)
                for row in per_protein
            ),
            "local_unavailable_positive_count": sum(
                int(row["protein_target"] == 1 and row["valid_surface_points"] == 0)
                for row in per_protein
            ),
            "surface_valid_point_count": sum(
                int(row["valid_surface_points"]) for row in per_protein
            ),
            "visualization_count": exported,
            "global_metrics": global_metrics,
            "surface_metrics": surface_metrics,
        }
        self._json(report_root / "evaluation_summary.json", summary)
        (report_root / "evaluation_summary.md").write_text(
            self._markdown(summary),
            encoding="utf-8",
        )
        headline_metrics = {
            f"test_{name}": float(value)
            for name, value in global_metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        return PostRunResult(
            outputs={
                "protein_count": len(per_protein),
                "visualization_count": exported,
                "global_metrics": global_metrics,
                "surface_micro": surface_micro,
                "artifacts_root": str(output_root.relative_to(context.run_dir)),
            },
            metrics=headline_metrics,
            metadata={
                "training_outcome": training_outcome,
                "checkpoint_role": context.selected_checkpoint_role,
                "selected_checkpoint_sha256": context.selected_checkpoint_sha256,
            },
        )

    def _protein_metrics(
        self,
        identifier     : str,
        tier           : str,
        protein_target : int,
        protein_score  : float,
        scores         : Tensor,
        targets        : Tensor,
        valid          : Tensor,
        areas          : Tensor,
        edges          : Tensor,
    ) -> dict[str, Any]:
        """Compute one protein's localization metrics and negative-map diagnostics.

        Args:
            identifier: Stable benchmark identifier.
            tier: Core/challenge tier.
            protein_target: Global binary label.
            protein_score: Global positive probability.
            scores: Surface probabilities ``[M]``.
            targets: Hard interface targets ``[M]``.
            valid: Ambiguity-excluding mask ``[M]``.
            areas: Positive represented-area weights ``[M]``.
            edges: Protein-local directed surface graph ``[2,E]``.

        Returns:
            Flat CSV-compatible row with nullable local metrics and region diagnostics.
        """
        local = self.metrics.compute(scores[valid], targets[valid])
        predicted = (scores >= self.threshold) & valid
        truth     = (targets == 1) & valid
        intersection = float(areas[predicted & truth].sum())
        predicted_area = float(areas[predicted].sum())
        true_area      = float(areas[truth].sum())
        union_area     = predicted_area + true_area - intersection
        components     = self._components(predicted, edges)
        row: dict[str, Any] = {
            "identifier": identifier,
            "tier": tier,
            "protein_target": protein_target,
            "protein_probability": protein_score,
            "valid_surface_points": int(valid.sum()),
            "local_gt_available": bool(valid.any()),
            **{f"surface_{name}": value for name, value in local.items()},
            "surface_dice": (
                2.0 * intersection / (predicted_area + true_area)
                if predicted_area + true_area > 0.0
                else None
            ),
            "surface_iou": intersection / union_area if union_area > 0.0 else None,
            "predicted_positive_area_fraction": predicted_area / float(areas.sum()),
            "maximum_surface_probability": float(scores.max()),
            "predicted_positive_components": len(components),
            "largest_predicted_component_points": max(
                (len(value) for value in components),
                default=0,
            ),
        }
        return row

    def _compute_or_undefined(
        self,
        scores : list[Tensor],
        targets: list[Tensor],
    ) -> dict[str, float | None]:
        """Compute metrics for a non-empty subset or preserve an explicit undefined schema.

        Args:
            scores: Probability vectors from one or more proteins.
            targets: Aligned hard-target vectors.

        Returns:
            Metric mapping, with every value ``None`` if the selected subset contains no points.
        """
        if not scores or sum(value.numel() for value in scores) == 0:
            return self.metrics.undefined()
        return self.metrics.compute(torch.cat(scores), torch.cat(targets))

    def _export_protein(
        self,
        path       : Path,
        batch      : Mapping[str, Any],
        point_mask : Tensor,
        logits     : Tensor,
        scores     : Tensor,
        embeddings : Tensor | None,
    ) -> None:
        """Export aligned geometry, ground truth, predictions, and selected latent channels.

        Args:
            path: PLY output path.
            batch: Original CPU batch.
            point_mask: Surface ownership mask for one protein.
            logits: Local model logits for selected points.
            scores: Sigmoid probabilities for selected points.
            embeddings: Optional learned surface representation ``[M,H]``.
        """
        curvatures = self._tensor(batch, "surface_curvatures")[point_mask].numpy()
        normals    = self._tensor(batch, "surface_normals")[point_mask].numpy()
        channels: dict[str, np.ndarray] = {
            "normal_x": normals[:, 0],
            "normal_y": normals[:, 1],
            "normal_z": normals[:, 2],
            "area_weight": self._tensor(batch, "surface_area_weights")[point_mask].numpy(),
            "surface_target_hard": self._tensor(batch, "surface_target_hard")[point_mask].numpy(),
            "surface_target_soft": self._tensor(batch, "surface_target_soft")[point_mask].numpy(),
            "surface_valid": self._tensor(batch, "surface_valid_mask")[point_mask].numpy(),
            "distance_to_dna": self._tensor(batch, "surface_distance_to_dna")[point_mask].numpy(),
            "surface_logit": logits.numpy(),
            "surface_probability": scores.numpy(),
        }
        symbols = ("mean", "gaussian", "curvedness")
        for scale in range(curvatures.shape[1]):
            for feature, name in enumerate(symbols):
                channels[f"curvature_scale_{scale}_{name}"] = curvatures[:, scale, feature]
        if embeddings is not None:
            channels["surface_embeddings"] = embeddings.numpy()
        PointCloudExporter().export(
            path,
            self._tensor(batch, "surface_positions")[point_mask].numpy(),
            channels,
            self.latent_channels,
        )

    @staticmethod
    def _protein_edges(edges: Tensor, point_mask: Tensor) -> Tensor:
        """Remap a batched directed graph to one protein's local point indices.

        Args:
            edges: Batched directed surface edges ``[2,E]``.
            point_mask: Boolean owner mask over all batched surface points.

        Returns:
            Local edge index ``[2,E_b]``.
        """
        selected = point_mask[edges[0]] & point_mask[edges[1]]
        remap    = torch.full((len(point_mask),), -1, dtype=torch.long)
        remap[point_mask] = torch.arange(int(point_mask.sum()))
        return remap[edges[:, selected]]

    @staticmethod
    def _components(mask: Tensor, edges: Tensor) -> list[set[int]]:
        """Find connected predicted-positive regions in a sparse surface graph.

        Args:
            mask: Boolean predicted-positive mask ``[M]``.
            edges: Directed local edges ``[2,E]``.

        Returns:
            Connected components as sets of local point indices.
        """
        selected = {int(value) for value in torch.nonzero(mask, as_tuple=False).flatten()}
        adjacency: dict[int, set[int]] = defaultdict(set)
        for source, destination in edges.T.tolist():
            if source in selected and destination in selected:
                adjacency[source].add(destination)
                adjacency[destination].add(source)
        output: list[set[int]] = []
        while selected:
            seed      = selected.pop()
            component = {seed}
            frontier  = [seed]
            while frontier:
                current = frontier.pop()
                neighbors = adjacency[current] & selected
                selected.difference_update(neighbors)
                component.update(neighbors)
                frontier.extend(neighbors)
            output.append(component)
        return output

    @staticmethod
    def _macro(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Aggregate each surface metric equally across proteins where it is defined.

        Args:
            rows: Per-protein metric rows.

        Returns:
            Per-metric means, evaluated protein counts, and undefined counts.
        """
        names = sorted(
            {
                key
                for row in rows
                for key in row
                if key.startswith("surface_")
                and key not in {"surface_dice", "surface_iou"}
            }
        )
        output: dict[str, Any] = {}
        for name in names:
            values = [float(row[name]) for row in rows if row.get(name) is not None]
            output[name.removeprefix("surface_")] = {
                "mean": sum(values) / len(values) if values else None,
                "evaluated_proteins": len(values),
                "undefined_count": len(rows) - len(values),
            }
        return output

    @staticmethod
    def _aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
        """Summarize positive-localization or negative-diagnostic protein rows.

        Args:
            rows: Selected per-protein mappings.

        Returns:
            Protein count and means for numeric diagnostic fields.
        """
        keys = (
            "predicted_positive_area_fraction",
            "maximum_surface_probability",
            "predicted_positive_components",
            "largest_predicted_component_points",
            "surface_dice",
            "surface_iou",
        )
        output: dict[str, Any] = {"protein_count": len(rows)}
        for key in keys:
            values = [float(row[key]) for row in rows if row.get(key) is not None]
            output[key] = sum(values) / len(values) if values else None
            output[f"{key}_undefined_count"] = len(rows) - len(values)
        return output

    @classmethod
    def _move(cls, value: Any, device: torch.device) -> Any:
        """Move tensors recursively while preserving identifiers and metadata.

        Args:
            value: Nested batch value.
            device: Target torch device.

        Returns:
            Structure-equivalent value with tensors on ``device``.
        """
        if isinstance(value, Tensor):
            return value.to(device)
        if isinstance(value, Mapping):
            return {key: cls._move(item, device) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(cls._move(item, device) for item in value)
        return value

    @staticmethod
    def _tensor(mapping: Mapping[str, Any], name: str) -> Tensor:
        """Return a required tensor field.

        Args:
            mapping: Batch mapping.
            name: Required tensor key.

        Returns:
            Tensor value.

        Raises:
            ValueError: If the key does not contain a tensor.
        """
        value = mapping.get(name)
        if not isinstance(value, Tensor):
            raise ValueError(f"evaluation field {name!r} must be a tensor")
        return value

    @staticmethod
    def _json(path: Path, payload: Mapping[str, Any]) -> None:
        """Write deterministic strict JSON.

        Args:
            path: Destination path.
            payload: JSON-compatible mapping without non-finite values.
        """
        path.write_text(
            json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )

    @staticmethod
    def _csv(path: Path, rows: list[dict[str, Any]]) -> None:
        """Write an ordered CSV whose empty fields represent undefined values.

        Args:
            path: Destination path.
            rows: Flat row mappings.
        """
        columns = sorted({key for row in rows for key in row})
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _markdown(summary: Mapping[str, Any]) -> str:
        """Render a concise human-readable post-run evaluation report.

        Args:
            summary: Complete evaluation summary.

        Returns:
            Markdown report text.
        """
        global_metrics = summary["global_metrics"]
        lines = [
            "# WISDOM DNA post-run evaluation",
            "",
            f"- Training outcome: `{summary['training_outcome']}`",
            f"- Checkpoint role: `{summary['checkpoint_role']}`",
            f"- Evaluated proteins: {summary['protein_count']}",
            f"- Valid surface points: {summary['surface_valid_point_count']}",
            f"- Interactive-data 3D point clouds: {summary['visualization_count']}",
            "",
            "## Global metrics",
            "",
        ]
        lines.extend(
            f"- {name}: {value if value is not None else 'undefined'}"
            for name, value in global_metrics.items()
        )
        return "\n".join(lines) + "\n"
