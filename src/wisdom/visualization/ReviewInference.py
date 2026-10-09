"""Shared frozen-checkpoint inference and bounded presentation for both native review sources."""

from __future__ import annotations

import csv
import torch
import lambdaforge as lf

from typing import Any
from pathlib import Path
from torch.utils.data import DataLoader
from collections.abc import Mapping, Sequence
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.Training import _create_model, _evaluate
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.visualization.ModelValidation import ModelValidation
from wisdom.visualization.ProteinReportPage import ProteinReportPage
from wisdom.evaluation.SurfacePredictionReport import SurfacePredictionReport


class ReviewInference:
    """Restore existing predictors without fitting, backward, optimizer, or hidden recovery."""

    @staticmethod
    def matching_dataset(
        recorded: Sequence[Mapping[str, Any]],
        current : Any,
    ) -> None:
        """Require the exact training dataset even when physical placements have changed.

        Args:
            recorded: Native Run inputs or ModelSet scientific_meaning.inputs records.
            current: LambdaForge's resolved dataset input identity.

        Raises:
            ValueError: No authoritative dataset identity, or any recorded dataset differs.
                A ModelSet spanning different datasets must be split into separate reviews.
        """
        datasets = [row for row in recorded if row["name"] == "dataset"]
        if not datasets or any(
            row.get("content_id") != current.content_id
            if current.content_id
            else row.get("sha256") != current.sha256
            for row in datasets
        ):
            raise ValueError("checkpoint and review dataset identities differ or are unavailable")

    @staticmethod
    def proteins(
        rows     : Mapping[str, Mapping[str, Any]],
        policy   : Mapping[str, Any],
        maximum  : int,
        threshold: float,
    ) -> tuple[list[str], dict[str, list[str]]]:
        """Choose contrasting display cases without changing numerical evaluation coverage.

        Args:
            rows: Full-resolution per-protein diagnostic statistics from frozen inference.
            policy: mode=automatic|explicit|all; automatic strategy=diagnostic|best|worst|mixed
                and cases_per_category=2; explicit identifiers are exact member IDs.
            maximum: Positive display ceiling; zero permits all selected proteins.
            threshold: Protein decision probability cutoff in [0,1].

        Returns:
            Stable IDs and explanatory category lists. Positive localization ranks use nAP;
            unavailable local GT does not become a zero. False positives/negatives and negative
            peaks are separate categories. Diagnostic extremes are not a population average.

        Raises:
            ValueError: Unknown mode or automatic strategy.
        """
        mode = policy.get("mode", "automatic")
        reasons: dict[str, list[str]] = {}
        if mode in ("all", "explicit"):
            identifiers = sorted(rows) if mode == "all" else policy.get("identifiers", ())
            reasons = {identifier: [mode] for identifier in identifiers if identifier in rows}
        elif mode == "automatic":
            strategy = policy.get("strategy", "diagnostic")
            quota = policy.get("cases_per_category", 2)
            local = [key for key in rows if rows[key]["normalized_surface_auprc"] is not None]
            local.sort(key=lambda key: (rows[key]["normalized_surface_auprc"], key))
            categories = {"worst_localization": local, "best_localization": list(reversed(local))}
            if strategy == "diagnostic":
                categories.update(
                    {
                        "false_positive": sorted(
                            (
                                key
                                for key in rows
                                if rows[key]["label"] == 0
                                and rows[key]["protein_probability"] >= threshold
                            ),
                            key=lambda key: (-rows[key]["protein_probability"], key),
                        ),
                        "false_negative": sorted(
                            (
                                key
                                for key in rows
                                if rows[key]["label"] == 1
                                and rows[key]["protein_probability"] < threshold
                            ),
                            key=lambda key: (rows[key]["protein_probability"], key),
                        ),
                        "negative_peak": sorted(
                            (key for key in rows if rows[key]["label"] == 0),
                            key=lambda key: (-rows[key]["positive_peak"], key),
                        ),
                        "global_surface_disagreement": sorted(
                            local,
                            key=lambda key: (
                                -(
                                    rows[key]["protein_probability"]
                                    - max(0.0, rows[key]["normalized_surface_auprc"])
                                ),
                                key,
                            ),
                        ),
                    }
                )
            elif strategy == "best":
                categories.pop("worst_localization")
            elif strategy == "worst":
                categories.pop("best_localization")
            elif strategy != "mixed":
                raise ValueError(f"unsupported automatic protein strategy: {strategy}")
            # Round-robin prevents the display ceiling from removing every later category.

            for index in range(quota):
                for category, candidates in categories.items():
                    if index < len(candidates):
                        reasons.setdefault(candidates[index], []).append(category)
        else:
            raise ValueError(f"unsupported protein policy: {mode}")
        chosen = list(reasons)[:maximum] if maximum else list(reasons)
        return chosen, {key: reasons[key] for key in chosen}

    def run(
        self,
        work                  : lf.Work,
        dataset               : Path,
        models                : Sequence[Mapping[str, Any]],
        source                : Mapping[str, Any],
        splits                : Sequence[str],
        allow_test            : bool,
        evaluation_scope      : str,
        protein_selection     : Mapping[str, Any] | None,
        content               : str,
        maximum_surface_points: int,
        maximum_proteins      : int,
        prediction_threshold  : float,
        report_maximum_mib    : float,
        save_predictions      : bool,
        batch_size            : int,
        data_workers          : int,
        strict_checkpoints    : bool,
    ) -> dict[str, Any]:
        """Evaluate a source-independent list of exact checkpoint descriptors and publish HTML.

        Args:
            work: Owning LambdaForge Work; manages logging, metrics, and report outputs.
            dataset: Resolved immutable dataset placement, never a source-Study directory.
            models: Descriptors with checkpoint Path, parameters, epoch/metrics metadata,
                source Run/seed identity, and original dataset inputs.
            source: JSON-compatible source identity and explicit post-hoc selection policies.
            splits: Requested train/val/test partitions; default wrappers choose val only.
            allow_test: Explicit opt-in to descriptive test inspection; no selection uses test.
            evaluation_scope: full evaluates every requested-split member; explicit evaluates
                only protein_selection.identifiers and labels metrics as partial coverage.
            protein_selection: automatic diagnostic, explicit identifiers, or all display policy.
            content: predictions includes prediction/logit/GT; full adds structural channels.
            maximum_surface_points: Display-only point cap; scientific metrics retain all points.
            maximum_proteins: Viewer ceiling per model/split; zero removes this bound.
            prediction_threshold: Hard prediction and protein diagnostic cutoff in [0,1].
            report_maximum_mib: Positive HTML document budget at most 16 MiB.
            save_predictions: Persist full-precision prediction NPZ for evaluated proteins only.
            batch_size: Positive protein count per forward batch, independent of training batch.
            data_workers: Nonnegative decoding processes; no training workers or optimizer.
            strict_checkpoints: Missing checkpoint fails if true, otherwise recorded and skipped.

        Returns:
            Source, model count, missing checkpoints and presentation-byte summary.

        Raises:
            ValueError: Unauthorized test, mismatched dataset/checkpoint metadata, invalid scope
                or unknown explicit IDs. Corrupted bytes are never skipped as missing evidence.
            FileNotFoundError: Missing selected checkpoint with strict_checkpoints enabled.
        """
        if "test" in splits and not allow_test:
            raise ValueError("test inspection requires allow_test=true; never select using test")
        policy = dict(protein_selection or {"mode": "automatic", "strategy": "diagnostic"})
        identifiers = set(policy.get("identifiers", ()))
        if evaluation_scope not in ("full", "explicit"):
            raise ValueError("evaluation_scope must be full or explicit")
        if evaluation_scope == "explicit" and (policy.get("mode") != "explicit" or not identifiers):
            raise ValueError("explicit evaluation requires nonempty explicit protein identifiers")
        if policy.get("mode") == "all" or maximum_proteins == 0:
            work.log("Unbounded protein presentation requested; report byte budget still applies")
        if not callable(getattr(work.outputs, "html_section", None)):
            raise RuntimeError("Review requires LambdaForge native HTML-section APIs; update LF")

        # Selection has already used source checkpoint evidence, never the forthcoming test
        # results. Each selected model remains frozen, including saved semantic gate overrides.

        root = Path(work.outputs.directory("model-review", role="report"))
        audits: list[dict[str, Any]] = []
        documents: list[dict[str, Any]] = []
        missing: list[dict[str, Any]] = []
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        evaluated = 0
        for descriptor in models:
            self.matching_dataset(descriptor["inputs"], work.inputs["dataset"])
            checkpoint = descriptor.get("checkpoint")
            if checkpoint is None or not Path(checkpoint).is_file():
                missing.append({"run_id": descriptor["run_id"], "reason": "missing best-model"})
                if strict_checkpoints:
                    raise FileNotFoundError(
                        "recover best-model with native LF; review never retrains"
                    )
                continue
            state = torch.load(checkpoint, map_location="cpu", weights_only=True)
            parameters = descriptor["parameters"]
            if int(state["model_version"]) != int(parameters.get("model_version", 1)):
                raise ValueError("checkpoint and recorded model versions differ")
            if descriptor.get("step") is not None and state["epoch"] != descriptor["step"]:
                raise ValueError("artifact metric step and checkpoint epoch differ")
            if not descriptor.get("metrics"):
                work.log("Historical checkpoint has no ranking metadata; explicitly selected only")
            model, _ = _create_model(int(state["model_version"]), state["model_parameters"])
            model.load_state_dict(state["state_dict"])
            model.to(device).eval().requires_grad_(False)
            warmup = state.get("initialization", {}).get("gate_warmup_fraction", 0.0)
            override = (
                "all_on"
                if (state["epoch"] - 1) / parameters.get("epochs", 100) < warmup
                else "learned"
            )
            override = state.get("inference_state", {}).get("gate_override", override)
            model.semantic_gates.set_override(override)
            precision = parameters.get("precision", "auto")
            bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
            if precision == "auto":
                precision = "bfloat16" if bf16 else "float32"
            if device.type != "cuda" or (precision == "bfloat16" and not bf16):
                precision = "float32"
            autocast_dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}.get(precision)
            found: set[str] = set()
            for split in splits:
                data = WisdomDataset(
                    dataset,
                    split,
                    surface_features=tuple(state.get("surface_features", ())),
                    subset=parameters.get("subset", "full"),
                    include_surface_targets=True,
                    include_surface_geometry=(
                        int(state["model_version"]) == 3
                        or state["model_parameters"].get("surface_refiner_type", "none")
                        not in {"none", "heat", "learned_heat"}
                    ),
                    include_atom_geometry=state["model_parameters"].get("vector_atomic_channels", 0)
                    > 0,
                )
                total = len(data)
                found.update(str(record[3]) for record in data.records if record[3] in identifiers)

                # Explicit scope filters records before DataLoader and before any neural forward.
                # Full-scope diagnostics use every point even when only two viewers are shown.

                if evaluation_scope == "explicit":
                    data.records = tuple(
                        record for record in data.records if record[3] in identifiers
                    )
                if not len(data):
                    continue
                output = root / str(len(audits))
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
                loader = DataLoader(
                    data,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=data_workers,
                    collate_fn=WisdomCollator(**state["data_parameters"]),
                )
                work.log(
                    f"Frozen inference: {descriptor['label']}, {split}: "
                    f"{len(data)}/{total} proteins"
                )
                with torch.inference_mode():
                    protein, surface = _evaluate(model, loader, device, autocast_dtype, collector)
                statistics = ModelValidation._statistics(collector, prediction_threshold)
                chosen, reasons = self.proteins(
                    statistics, policy, maximum_proteins, prediction_threshold
                )
                collector.identifiers = tuple(chosen) or ("__no_selected_protein__",)
                collector.publish(surface, int(state["epoch"]))
                for document in collector.report_documents:
                    document.update(
                        variant=descriptor["label"],
                        seed=descriptor.get("seed"),
                        details={
                            **statistics[document["identifier"]],
                            "selection_reasons": reasons[document["identifier"]],
                        },
                    )
                documents.extend(collector.report_documents)
                audits.append(
                    {
                        "run_id": descriptor["run_id"],
                        "seed": descriptor.get("seed"),
                        "trial": descriptor.get("trial_index"),
                        "parameters": parameters,
                        "checkpoint_epoch": state["epoch"],
                        "checkpoint_metrics": descriptor.get("metrics", {}),
                        "checkpoint_sha256": descriptor.get("sha256"),
                        "split": split,
                        "evaluation_scope": evaluation_scope,
                        "evaluated_proteins": len(data),
                        "split_proteins": total,
                        "coverage": "complete"
                        if evaluation_scope == "full"
                        else "requested subset only",
                        "global": protein,
                        "surface": surface,
                        "proteins": statistics,
                        "selection_reasons": reasons,
                        "inference_precision": precision,
                        "gate_override": override,
                    }
                )
                for name, value in {**protein, **surface}.items():
                    if value is not None:
                        work.metrics.log(f"review_{len(audits)}_{name}", value)
                with (output / "proteins.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(
                        stream, fieldnames=("identifier", *next(iter(statistics.values())).keys())
                    )
                    writer.writeheader()
                    writer.writerows({"identifier": key, **row} for key, row in statistics.items())
            if identifiers - found:
                raise ValueError(
                    "requested proteins absent from evaluated splits: "
                    f"{sorted(identifiers - found)}"
                )
            evaluated += 1
            del model

        # The same inspector and compressed gallery are used by preprocessing and Training.
        # Report omissions affect presentation only, not the per-protein CSV or numerical audit.

        context = {
            **source,
            "policy": "post-hoc inference on frozen checkpoint",
            "evaluation_scope": evaluation_scope,
            "test_opened": "test" in splits,
            "missing": missing,
            "protein_selection": policy,
        }
        page, summary = ProteinReportPage(report_maximum_mib).render(
            documents,
            context,
            reason="No selected checkpoint/protein viewers; inspect review-audit.json",
        )
        work.outputs.html_section(
            "protein-report", section="wisdom-proteins", title="WISDOM proteins"
        ).write_text(page)
        work.outputs.file("review-audit", filename="review-audit.json").write_json(audits)
        work.outputs.file("interpretation", filename="README.md", role="report").write_text(
            "# Post-hoc inference on frozen checkpoint\n\n"
            "No fitting, optimizer, backward, HPO changes or new checkpoint selection occurs. "
            "Model policies use original checkpoint-bound validation evidence, never review test "
            "metrics. Explicit scope reports only the requested subset, not population metrics. "
            "Display caps do not subsample scientific points.\n\n"
            "Surface AP ranks valid interface points on positive proteins; random ranking has "
            "baseline interface prevalence pi. Normalized AP=(AP-pi)/(1-pi) is zero at that "
            "baseline and one for perfect ranking. Undefined local metrics remain null. "
            "Negative peak is maximum false local probability; mass is its area-weighted mean. "
            "Diagnostic pictures intentionally show extremes; use review-audit.json and "
            "model-review/*/proteins.csv for coverage, counts and all individual results.\n"
        )
        work.metrics.log("reviewed_models", evaluated)
        return {
            "source": source,
            "reviewed_models": evaluated,
            "missing": missing,
            "report_summary": summary,
            "retrained_models": 0,
        }
