"""Inference-only review of explicitly selected checkpoints from portable LF evidence."""

from __future__ import annotations

import csv
import json
import torch
import hashlib
import numpy as np
import lambdaforge as lf

from typing import Any
from pathlib import Path
from torch.utils.data import DataLoader
from collections.abc import Mapping, Sequence
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.Training import Training, _create_model, _evaluate
from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite
from wisdom.visualization.ProteinReportPage import ProteinReportPage
from wisdom.visualization.ValidationSelection import ValidationSelection
from wisdom.evaluation.SurfacePredictionReport import SurfacePredictionReport


class ModelValidation(lf.Work):
    """Audit frozen predictions without an optimizer, candidate ranking or test-driven selection."""

    analysis_profile = Training.analysis_profile

    def run(
        self,
        source_study         : Path,
        dataset              : Path,
        selected_trials      : Mapping[str, int] | None = None,
        source_seeds         : Sequence[int]            = (),
        content              : str                      = "predictions",
        maximum_surface_points: int                     = 2000,
        splits               : Sequence[str]            = ("val",),
        allow_test           : bool                     = False,
        selection            : str                      = "diagnostic",
        cases_per_category   : int                      = 2,
        identifiers          : Sequence[str]            = (),
        prediction_threshold : float                    = 0.5,
        report_maximum_mib   : float                    = 14.0,
        save_predictions     : bool                     = False,
        batch_size           : int                      = 4,
        data_workers         : int                      = 0,
        strict_checkpoints   : bool                     = True,
    ) -> dict[str, Any]:
        """Evaluate reviewed Trial/seed checkpoints and explain which proteins are displayed.

        Every eligible member of each requested split receives one forward pass per checkpoint.
        Only presentation is sampled. No gradients, optimizer, fit, HPO objective, checkpoint
        selection or hidden retraining occurs. Empty selected_trials publishes the available
        Trial inventory for manual review rather than guessing a winner.

        Args:
            source_study: Native LF portable export root containing manifest.json and execution/.
            dataset: Exact immutable dataset used by the source Runs, resolved by LambdaForge.
            selected_trials: Reviewed display labels (e.g. pooling names) mapped to native Trial
                indices. All completed, unpruned seeds of each selected Trial remain available.
            source_seeds: Optional exact seed filter; empty retains all recorded seeds.
            content: predictions displays logits/probabilities and GT only; full includes atoms,
                structural channels and mesh. Neither changes inference or numerical metrics.
            maximum_surface_points: Positive deterministic display-only point ceiling; metrics
                still use all points. Four-decimal display rounding never changes saved NPZ arrays.
            splits: Explicit train/val/test partitions; validation alone by default.
            allow_test: Explicit permission for descriptive test inspection, never selection.
            selection: diagnostic samples contrasting cases; all renders every member; explicit
                uses identifiers without affecting metric coverage.
            cases_per_category: Maximum cases per diagnostic category; positive integer.
            identifiers: Exact displayed member IDs when selection=explicit.
            prediction_threshold: Global and interactive local decision cutoff in [0,1].
            report_maximum_mib: Embedded HTML ceiling in (0,16] MiB; omissions are explicit.
            save_predictions: Also persist point-aligned prediction NPZ files for all members.
            batch_size: Positive number of proteins per inference batch.
            data_workers: Non-negative NPZ-decoding subprocess count.
            strict_checkpoints: Fail on missing/ineligible selected Runs instead of auditing and
                skipping them. Neither policy retrains a model.

        Returns:
            Source identity, evaluated checkpoint count, missing evidence and viewer-byte summary.

        Raises:
            ValueError: If test access is unauthorized, evidence bytes/identity differ, the source
                is not a native export, or a selected Trial/split is invalid.
            FileNotFoundError: If strict checkpoint recovery requires researcher intervention.
            OSError: If source assets or managed reports cannot be read/written.
        """
        if not callable(getattr(self.outputs, "html_section", None)):
            raise RuntimeError("native HTML sections require an updated LambdaForge installation")
        selected_trials = dict(selected_trials or {})
        if "test" in splits and not allow_test:
            raise ValueError(
                "test inspection requires allow_test=true; never use it to choose Trials"
            )
        selection_mode = ValidationSelection(selection)
        if cases_per_category < 1 or batch_size < 1 or data_workers < 0:
            raise ValueError(
                "case/batch counts must be positive; data_workers must be non-negative"
            )

        # Consume only the documented portable evidence contract. Native Run/Attempt identities
        # reconnect exported checkpoints; stale remote paths are never followed or globbed.

        root = source_study.resolve()
        manifest = json.loads((root / "manifest.json").read_text())
        if manifest.get("lambdaforge_export_version") != 2:
            raise ValueError("source_study must be a native LF version-2 portable export")
        inventory = {entry["path"]: entry for entry in manifest["inventory"]}
        execution = json.loads(self._verified(root, "execution/result.json", inventory).read_text())
        if execution["execution_id"] != manifest["execution_id"]:
            raise ValueError("export manifest and result identify different executions")
        runs = execution["runs"]
        available = [
            {
                "trial":  (run.get("trial") or {}).get("index", 0),
                "seed":   run.get("seed"),
                "run_id": run["run_id"],
                "status": run["status"],
                "pruned": run.get("pruned", False),
                "parameters": run["parameters"],
            }
            for run in runs
        ]
        self.outputs.file("candidate-inventory", filename="candidate-inventory.json").write_json(
            available
        )

        # An unfilled review ledger is useful, not a reason to render hundreds of HPO candidates.
        # WISDOM does not rank hyperparameters or replace LF's paired scientific conclusions.

        viewer_documents: list[dict[str, Any]] = []
        audits: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        builder = ProteinReportPage(report_maximum_mib)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            torch.set_float32_matmul_precision("medium")
        evaluated = 0
        report_root = Path(self.outputs.directory("model-validation", role="report"))
        for label, trial in selected_trials.items():
            matches = [
                run
                for run in runs
                if (run.get("trial") or {}).get("index", 0) == trial
                and (not source_seeds or run.get("seed") in source_seeds)
            ]
            if not matches:
                raise ValueError(f"selected Trial {trial} has no recorded Runs for requested seeds")
            for run in matches:
                key = f"{label} / Trial {trial} / seed {run.get('seed')}"
                artifacts = [a for a in run.get("artifacts", ()) if a["name"] == "best-model"]
                if run["status"] != "succeeded" or run.get("pruned") or len(artifacts) != 1:
                    missing.append(
                        {"source": key, "reason": "No completed unpruned best-model artifact"}
                    )
                    if strict_checkpoints:
                        raise FileNotFoundError(
                            f"{key}: recover the original LF Run; validation never retrains"
                        )
                    continue

                # Dataset identity must match even when placements differ between clusters.
                # Presentation cannot silently compare a checkpoint against another release.

                recorded = next(value for value in run["inputs"] if value["name"] == "dataset")
                current = self.inputs["dataset"]
                same = (
                    recorded.get("content_id") == current.content_id
                    if current.content_id
                    else recorded.get("sha256") == current.sha256
                )
                if not same:
                    raise ValueError(f"{key}: source and validation dataset identities differ")
                relative = (
                    f"execution/runs/{run['run_id']}/attempts/{run['attempt_id']}/"
                    f"{artifacts[0]['path']}"
                )
                if relative not in inventory or not (root / relative).is_file():
                    missing.append(
                        {"source": key, "reason": "Checkpoint absent from export inventory/files"}
                    )
                    if strict_checkpoints:
                        raise FileNotFoundError(
                            f"{key}: recover/export best-model; validation never retrains"
                        )
                    continue
                path = self._verified(root, relative, inventory)
                state = torch.load(path, map_location="cpu", weights_only=True)
                if int(state["model_version"]) != int(run["parameters"].get("model_version", 1)):
                    raise ValueError(f"{key}: checkpoint and source model versions differ")
                model, _ = _create_model(int(state["model_version"]), state["model_parameters"])
                model.load_state_dict(state["state_dict"])
                model.to(device).eval().requires_grad_(False)

                # Match the original dense-layer precision where the review device supports it.
                # Shared evaluation keeps sparse diffusion derivatives in their safe FP32 path.

                precision = run["parameters"].get("precision", "auto")
                supports_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
                if precision == "auto":
                    precision = "bfloat16" if supports_bf16 else "float32"
                if device.type != "cuda" or (precision == "bfloat16" and not supports_bf16):
                    precision = "float32"
                autocast_dtype = {
                    "bfloat16": torch.bfloat16,
                    "float16":  torch.float16,
                }.get(precision)

                # Gate warm-up is runtime state, not a weight. New checkpoints record it; old
                # checkpoints reconstruct it from the saved epoch and authored warm-up schedule.

                warmup = state.get("initialization", {}).get("gate_warmup_fraction", 0.0)
                epochs = run["parameters"].get("epochs", 100)
                override = "all_on" if (state["epoch"] - 1) / epochs < warmup else "learned"
                override = state.get("inference_state", {}).get("gate_override", override)
                model.semantic_gates.set_override(override)
                for split in splits:
                    data = WisdomDataset(
                        dataset,
                        split,
                        include_surface_targets=True,
                        include_surface_geometry=int(state["model_version"]) == 3,
                        include_atom_geometry=state["model_parameters"].get(
                            "vector_atomic_channels", 0
                        )
                        > 0,
                    )
                    loader = DataLoader(
                        data,
                        batch_size=batch_size,
                        shuffle=False,
                        num_workers=data_workers,
                        collate_fn=WisdomCollator(**state["data_parameters"]),
                    )
                    output = report_root / str(len(audits))
                    collector = SurfacePredictionReport(
                        data,
                        output,
                        split,
                        prediction_threshold,
                        report_only=True,
                        save_predictions=save_predictions,
                        content=content,
                        maximum_surface_points=maximum_surface_points,
                    )
                    self.log(f"Inference only: {key}, split={split}, proteins={len(data)}")
                    global_metrics, surface_metrics = _evaluate(
                        model, loader, device, autocast_dtype, collector
                    )
                    statistics = self._statistics(collector, prediction_threshold)
                    chosen, reasons = self._select(
                        statistics, selection_mode, cases_per_category, identifiers
                    )
                    # An empty explicit selection must not fall back to automatic sampling.
                    collector.identifiers = (
                        tuple(chosen) if chosen else ("__no_selected_protein__",)
                    )
                    collector.publish(surface_metrics, int(state["epoch"]))
                    for document in collector.report_documents:
                        document.update(
                            variant=label,
                            seed=run.get("seed"),
                            details={
                                **statistics[document["identifier"]],
                                "selection_reasons": reasons[document["identifier"]],
                            },
                        )
                    viewer_documents.extend(collector.report_documents)
                    audits.append(
                        {
                            "variant": label,
                            "trial":   trial,
                            "seed":    run.get("seed"),
                            "run_id":  run["run_id"],

                            "checkpoint_epoch":  state["epoch"],
                            "checkpoint_sha256": inventory[relative]["sha256"],

                            "model_version":    state["model_version"],
                            "model_parameters": state["model_parameters"],
                            "data_parameters":  state["data_parameters"],

                            "inference_precision": precision,
                            "gate_override":       override,

                            "split":   split,
                            "global":  global_metrics,
                            "surface": surface_metrics,

                            "proteins":          statistics,
                            "selection_reasons": reasons,
                        }
                    )
                    for name, value in {**global_metrics, **surface_metrics}.items():
                        if value is not None:
                            self.metrics.log(f"review_{len(audits)}_{name}", value)
                    with (output / "proteins.csv").open("w", newline="") as stream:
                        writer = csv.DictWriter(
                            stream,
                            fieldnames=("identifier", *next(iter(statistics.values())).keys()),
                        )
                        writer.writeheader()
                        writer.writerows(
                            {"identifier": name, **row} for name, row in statistics.items()
                        )
                evaluated += 1
                del model

        # One compressed library serves every reviewed pooling and seed. Byte omission is
        # presentation-only; the detailed report retains metrics for every evaluated protein.

        page, summary = builder.render(
            viewer_documents,
            {
                "source_execution": execution["execution_id"],
                "selected_trials": dict(selected_trials),
                "selection": selection,
                "content": content,
                "maximum_surface_points": maximum_surface_points,
                "test_opened": "test" in splits,
                "missing": missing,
                "policy": "Frozen checkpoints; no optimizer, training or hyperparameter ranking.",
            },
            reason=("No viewers. Fill selected_trials using candidate-inventory.json, "
                    "or inspect missing checkpoints."),
        )
        self.outputs.html_section(
            "protein-report", section="wisdom-proteins", title="WISDOM proteins"
        ).write_text(page)
        self.outputs.file("validation-audit", filename="validation-audit.json").write_json(audits)
        self.outputs.file("interpretation", filename="README.md", role="report").write_text(
            "# Frozen-model visual validation\n\n"
            "The Trial ledger is researcher-selected, not a new ranking. All eligible seeds are "
            "retained unless explicitly filtered. Each checkpoint performs inference on every "
            "requested-split protein; only the viewer list is sampled.\n\n"
            "Protein probability is the global DNA-binding confidence. A binary error is a "
            "wrong decision at the configured cutoff. Surface AUPRC measures ranking of valid "
            "interface points on positive proteins: 1 is perfect; random ranking has baseline "
            "interface prevalence pi. Normalized AP=(AP-pi)/(1-pi) is 0 at that baseline and 1 "
            "at perfect ranking. Undefined local targets remain null, not zero.\n\n"
            "Positive mass is area-weighted mean local probability; peak is its maximum. "
            "For a curated negative, high values reveal broad false regions or false hotspots. "
            "Disagreement ranks positive maps by |global probability-max(0,normalized AP)|; "
            "this is a descriptive mismatch, not a calibrated probability or training target.\n\n"
            "Diagnostic galleries combine wrong global decisions, worst/best evaluable positive "
            "maps, mismatch cases, negative false mass and confidently correct binary examples. "
            "They deliberately show extremes: do not estimate average quality from these pictures. "
            "Use validation-audit.json and model-validation/*/proteins.csv for full coverage, "
            "and LF's original paired-seed reports for hyperparameter comparisons.\n\n"
            "No gradients, optimizer, model selection or retraining occurs here. Test is opt-in "
            "and must remain excluded from parameter selection. Missing checkpoints require "
            "native source-Study recovery. HTML byte omissions never remove numerical metrics.\n"
        )
        self.metrics.log("reviewed_models", evaluated)
        return {
            "source_execution": execution["execution_id"],
            "reviewed_models":  evaluated,
            "missing":          missing,
            "report_summary":   summary,
            "retrained_models": 0,
        }

    @staticmethod
    def _verified(root: Path, relative: str, inventory: Mapping[str, Any]) -> Path:
        """Verify one explicitly referenced exported scientific file against its LF inventory.

        Args:
            root: Typed portable export root.
            relative: Documented export-relative evidence path, never a remote absolute path.
            inventory: LF's authoritative raw-file checksum inventory.

        Returns:
            Verified local regular-file path.

        Raises:
            ValueError: If traversal, links, byte size or SHA-256 contradict the export evidence.
        """
        path = root / relative
        if not path.resolve().is_relative_to(root) or any(
            p.is_symlink() for p in (path, *path.parents)
        ):
            raise ValueError("checkpoint evidence escapes its portable export ownership")
        entry = inventory[relative]
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
        if path.stat().st_size != entry["size_bytes"] or digest.hexdigest() != entry["sha256"]:
            raise ValueError(f"export evidence changed: {relative}")
        return path

    @staticmethod
    def _statistics(report: SurfacePredictionReport, threshold: float) -> dict[str, dict[str, Any]]:
        """Describe full-split binary errors and localization before selecting viewer cases.

        Args:
            report: Complete point-aligned inference collector with global sigmoid probabilities.
            threshold: Protein decision cutoff in [0,1].

        Returns:
            Per-ID rows with confidence, binary error, positive-only AP/nAP and negative false mass.
            nAP=(AP-pi)/(1-pi), where pi is the valid-point interface prevalence; unavailable local
            targets or one-class masks produce None, never a fabricated perfect/zero AP.
        """
        rows: dict[str, dict[str, Any]] = {}
        for identifier, (base, annotation, label) in report.records.items():
            scores = report.predictions[identifier]
            global_score = report.protein_probabilities[identifier]
            ap: float | None = None
            prevalence: float | None = None
            normalized: float | None = None
            if annotation is not None:
                with np.load(annotation, allow_pickle=False) as data:
                    valid = data["surface_valid_mask"].astype(bool)
                    hard = data["surface_target_hard"][valid]
                if label == 1 and valid.any():
                    ap = BinaryMetricSuite().compute(
                        torch.from_numpy(scores[valid]), torch.from_numpy(hard.copy()), ("auprc",)
                    )["auprc"]
                    prevalence = float(hard.mean())
                    if ap is not None and prevalence < 1:
                        normalized = (ap - prevalence) / (1 - prevalence)
            with np.load(base, allow_pickle=False) as data:
                area = data["surface_area_weights"]
                mass = float(np.dot(scores, area) / area.sum())
            rows[identifier] = {
                "label":               label,
                "protein_probability": global_score,
                "binary_error": int(global_score >= threshold) != label,

                "surface_auprc":            ap,
                "interface_prevalence":     prevalence,
                "normalized_surface_auprc": normalized,

                "positive_mass": mass,
                "positive_peak": float(scores.max()),
            }
        return rows

    @staticmethod
    def _select(
        rows: Mapping[str, Mapping[str, Any]],
        mode: ValidationSelection,
        quota: int,
        identifiers: Sequence[str],
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Select contrasting descriptive cases, not models or hyperparameters.

        Args:
            rows: Full-split statistics keyed by immutable protein ID.
            mode: all, explicit or diagnostic.
            quota: Maximum displayed members from each diagnostic category.
            identifiers: Exact requested IDs for explicit mode; absent IDs stay absent.

        Returns:
            Unique IDs and all reasons for each inclusion. Diagnostic categories overlap; reasons
            retain that overlap. Best/worst localization applies only to positive, evaluable maps.
        """
        if mode is not ValidationSelection.DIAGNOSTIC:
            chosen = (
                sorted(rows)
                if mode is ValidationSelection.ALL
                else list(dict.fromkeys(i for i in identifiers if i in rows))
            )
            return chosen, {i: [mode.value] for i in chosen}
        positive = [i for i, row in rows.items() if row["normalized_surface_auprc"] is not None]
        errors = [i for i, row in rows.items() if row["binary_error"]]
        negative = [i for i, row in rows.items() if row["label"] == 0]
        correct = [i for i, row in rows.items() if not row["binary_error"]]
        categories = {
            "binary_error": sorted(
                errors, key=lambda i: (-abs(rows[i]["protein_probability"] - rows[i]["label"]), i)
            ),
            "worst_surface": sorted(positive, key=lambda i: (rows[i]["surface_auprc"], i)),
            "best_surface": sorted(positive, key=lambda i: (-rows[i]["surface_auprc"], i)),
            "global_surface_disagreement": sorted(
                positive,
                key=lambda i: (
                    -abs(
                        rows[i]["protein_probability"] - max(0, rows[i]["normalized_surface_auprc"])
                    ),
                    i,
                ),
            ),
            "negative_false_mass": sorted(negative, key=lambda i: (-rows[i]["positive_mass"], i)),
            "best_binary": sorted(
                correct, key=lambda i: (abs(rows[i]["protein_probability"] - rows[i]["label"]), i)
            ),
        }
        reasons: dict[str, list[str]] = {}
        for reason, candidates in categories.items():
            for identifier in candidates[:quota]:
                reasons.setdefault(identifier, []).append(reason)
        return list(reasons), reasons
