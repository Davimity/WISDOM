"""Post-HPO sparse concept calibration for one frozen WISDOM v1 checkpoint."""

from __future__ import annotations

import csv
import json
import math
import torch
import hashlib
import inspect
import lambdaforge as lf
import matplotlib.pyplot as plt

from pathlib import Path
from torch import Tensor
from copy import deepcopy
from typing import Any, cast
from itertools import pairwise
from collections.abc import Mapping, Sequence

from scipy.optimize import linear_sum_assignment

from wisdom.models.WisdomV1 import WisdomV1
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.Training import _create_model, _device_batch, _model_inputs
from wisdom.interpretability.EmbeddingScaler import EmbeddingScaler
from wisdom.interpretability.SparseConceptModel import SparseConceptModel

plt.switch_backend("Agg")


class SparseConceptDiscovery(lf.Work):
    """Discover sparse concepts after, and independently from, predictor HPO."""

    def run(
        self,
        checkpoint                    : Path,
        dataset                       : Path,
        subset                        : str = "replicate-00/train-25",
        lambda_values                 : Sequence[float] = (),
        selected_lambda               : float | None = None,
        calibration_seeds             : Sequence[int] = (17, 29, 43),
        maximum_points_per_protein    : int = 0,
        epochs                        : int = 80,
        patience                      : int = 12,
        batch_size                    : int = 4096,
        learning_rate                 : float = 1.0e-3,
        near_dead_threshold           : float = 0.001,
        dominant_threshold            : float = 0.95,
        redundancy_threshold          : float = 0.95,
        stability_threshold           : float = 0.80,
        top_activations_per_concept   : int = 10,
        output_directory              : str | None = None,
        overwrite_output              : bool = False,
    ) -> dict[str, Any]:
        """Calibrate and fit a sparse bottleneck without labels or test data.

        Training statistics and optimization use frozen train embeddings. Validation measures
        reconstruction, prediction fidelity, sparsity, and seed stability. The held-out test split
        and every DNA surface target remain unopened. ``K_probe=H``; a deterministic Pareto knee
        selects lambda unless ``selected_lambda`` is supplied, after which a clean ``K_final``
        model is trained around that lambda.

        Args:
            checkpoint: Explicit ``best-model`` artifact selected after completed V1 HPO.
            dataset: Managed WISDOM dataset root resolved by LambdaForge.
            subset: Training dilution used to fit both predictor concepts and scaler.
            lambda_values: Manual non-negative calibration grid; empty uses seven log values.
            selected_lambda: Optional explicit calibration choice instead of automatic knee.
            calibration_seeds: Small independent seed set used to assess permutation-invariant
                decoder stability.
            maximum_points_per_protein: Uniform point cap per protein; zero keeps every point.
            epochs: Maximum optimization epochs per sparse candidate.
            patience: Validation epochs without improvement before stopping a candidate.
            batch_size: Surface embeddings per sparse optimizer step.
            learning_rate: Adam learning rate for the sparse model only.
            near_dead_threshold: Maximum activation rate classified as practically near-dead.
            dominant_threshold: Minimum activation rate classified as globally dominant.
            redundancy_threshold: Decoder cosine similarity marking potential redundancy.
            stability_threshold: Matched decoder cosine required for a stable concept.
            top_activations_per_concept: Highest train points retained for later visualization.
            output_directory: Optional durable external copy of the interpretability directory.
            overwrite_output: Permit replacement of that external copy after successful fitting.

        Returns:
            JSON-compatible selected lambda, suggested width, stability, and fidelity summary.

        Raises:
            ValueError: If the checkpoint is not final V1, a parameter is invalid, or data/model
                contracts disagree.
            OSError: If checkpoint, dataset, or output files cannot be read or written.
        """
        lambdas = tuple(lambda_values) or (0.0, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2, 1.0e-1, 1.0)
        seeds   = tuple(int(seed) for seed in calibration_seeds)
        if not seeds or not lambdas or any(value < 0.0 for value in lambdas):
            raise ValueError("calibration requires seeds and non-negative lambda values")
        if selected_lambda is not None and selected_lambda < 0.0:
            raise ValueError("selected_lambda cannot be negative")
        if maximum_points_per_protein < 0 or epochs < 1 or patience < 1 or batch_size < 1:
            raise ValueError("sampling and optimization counts are invalid")
        if learning_rate <= 0.0 or top_activations_per_concept < 1:
            raise ValueError("learning rate and top-activation count must be positive")
        if not 0.0 <= near_dead_threshold < dominant_threshold <= 1.0:
            raise ValueError("dead/dominant activation thresholds are inconsistent")
        if not 0.0 < redundancy_threshold <= 1.0 or not 0.0 < stability_threshold <= 1.0:
            raise ValueError("redundancy and stability thresholds must lie in (0,1]")

        output = Path(
            self.outputs.directory(
                "interpretability",
                role="interpretability",
                publish_to=output_directory,
                overwrite=overwrite_output,
            )
        )
        calibration_root = output / "calibration"
        final_root       = output / "final"
        calibration_root.mkdir()
        final_root.mkdir()

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        predictor, predictor_state, data_parameters = self._load_predictor(checkpoint, device)
        predictor_digest = self._parameter_digest(predictor)

        self.log("Extracting frozen pre-local-head embeddings from train and validation only")
        train = self._extract(
            predictor,
            dataset,
            "train",
            subset,
            data_parameters,
            maximum_points_per_protein,
            seeds[0],
            device,
        )
        validation = self._extract(
            predictor,
            dataset,
            "val",
            subset,
            data_parameters,
            maximum_points_per_protein,
            seeds[0] + 1,
            device,
        )
        scaler          = EmbeddingScaler.fit(train["embeddings"])
        train_standard  = scaler.transform(train["embeddings"])
        val_standard    = scaler.transform(validation["embeddings"])
        embedding_width = train_standard.shape[1]
        torch.save(scaler.state(), calibration_root / "embedding_scaler.pt")

        # Phase A deliberately starts with K=H. Lambda is the only important calibration axis.

        calibration_rows: list[dict[str, Any]] = []
        calibration_models: dict[tuple[float, int], SparseConceptModel] = {}
        candidate_total = len(lambdas) * len(seeds)
        completed       = 0
        for sparse_lambda in lambdas:
            for seed in seeds:
                model, metrics = self._fit_candidate(
                    train_standard,
                    val_standard,
                    train,
                    validation,
                    predictor.local_head,
                    scaler,
                    embedding_width,
                    float(sparse_lambda),
                    seed,
                    epochs,
                    patience,
                    batch_size,
                    learning_rate,
                    device,
                )
                calibration_models[(float(sparse_lambda), seed)] = model.cpu()
                calibration_rows.append(
                    {"phase": 1, "lambda": float(sparse_lambda), "seed": seed, **metrics}
                )
                completed += 1
                self.progress.update(
                    completed=completed,
                    total=candidate_total,
                    message=f"probe lambda={sparse_lambda:g}, seed={seed}",
                )
                self.log(
                    f"[Sparse calibration] lambda={sparse_lambda:g} seed={seed} "
                    f"reconstruction_nmse={metrics['reconstruction_nmse']:.4f} "
                    f"local_pearson={self._metric_text(metrics['local_logit_pearson'])} "
                    f"active={metrics['mean_active_features']:.2f}/{embedding_width}"
                )

        grouped = self._aggregate(calibration_rows)
        chosen_lambda = (
            float(selected_lambda)
            if selected_lambda is not None
            else self._pareto_knee(grouped)
        )
        if chosen_lambda not in {float(value) for value in lambdas}:
            raise ValueError("selected_lambda must be one of the calibrated lambda_values")

        chosen_models = [calibration_models[(chosen_lambda, seed)] for seed in seeds]
        stability     = self._stability(chosen_models, stability_threshold)
        stability_scores = stability["scores"]
        if not isinstance(stability_scores, Tensor):
            raise RuntimeError("concept stability did not return per-concept scores")
        representative = chosen_models[0]
        chosen_metrics = self._measure(
            representative.to(device),
            val_standard.to(device),
            validation,
            predictor.local_head,
            scaler,
        )
        activation_rates = chosen_metrics.pop("activation_rates")
        decoder_cosines  = chosen_metrics.pop("decoder_cosines")
        stable_mask      = stability_scores >= stability_threshold
        live_mask        = activation_rates > near_dead_threshold
        redundant_mask  = decoder_cosines >= redundancy_threshold
        useful_mask     = self._independent_concepts(
            representative.decoder.weight.detach().float().cpu(),
            stable_mask & live_mask,
            stability_scores,
            redundancy_threshold,
        )
        suggested_width  = max(1, int(useful_mask.sum()))

        # Phase B retrains a clean smaller model; the probe is never merely pruned and reused.

        local_lambdas = self._local_lambdas(chosen_lambda)
        final_rows: list[dict[str, Any]] = []
        final_models: dict[tuple[float, int], SparseConceptModel] = {}
        for sparse_lambda in local_lambdas:
            for seed in seeds:
                model, metrics = self._fit_candidate(
                    train_standard,
                    val_standard,
                    train,
                    validation,
                    predictor.local_head,
                    scaler,
                    suggested_width,
                    sparse_lambda,
                    seed,
                    epochs,
                    patience,
                    batch_size,
                    learning_rate,
                    device,
                )
                final_models[(sparse_lambda, seed)] = model.cpu()
                final_rows.append(
                    {"phase": 2, "lambda": sparse_lambda, "seed": seed, **metrics}
                )
                self.log(
                    f"[Sparse final] K={suggested_width} lambda={sparse_lambda:g} seed={seed} "
                    f"reconstruction_nmse={metrics['reconstruction_nmse']:.4f} "
                    f"local_pearson={self._metric_text(metrics['local_logit_pearson'])} "
                    f"active={metrics['mean_active_features']:.2f}/{suggested_width}"
                )

        final_grouped = self._aggregate(final_rows)
        final_lambda  = self._pareto_knee(final_grouped)
        eligible      = [row for row in final_rows if row["lambda"] == final_lambda]
        best_row      = min(eligible, key=lambda row: float(row["selection_error"]))
        final_seed    = int(best_row["seed"])
        final_model   = final_models[(final_lambda, final_seed)]
        comparison_models  = [final_model] + [
            final_models[(final_lambda, seed)]
            for seed in seeds
            if seed != final_seed
        ]
        final_stability = self._stability(comparison_models, stability_threshold)
        final_stability_scores = final_stability["scores"]
        if not isinstance(final_stability_scores, Tensor):
            raise RuntimeError("final concept stability did not return per-concept scores")
        final_measure = self._measure(
            final_model.to(device),
            val_standard.to(device),
            validation,
            predictor.local_head,
            scaler,
        )
        final_activation_rates = cast(Tensor, final_measure.pop("activation_rates"))
        final_decoder_cosines  = cast(Tensor, final_measure.pop("decoder_cosines"))
        final_dead             = final_activation_rates == 0.0
        final_near_dead        = (
            (final_activation_rates > 0.0)
            & (final_activation_rates <= near_dead_threshold)
        )
        final_redundant        = final_decoder_cosines >= redundancy_threshold
        final_stable           = final_stability_scores >= stability_threshold

        concept_rows, top_rows = self._concept_reports(
            final_model,
            train_standard,
            val_standard,
            train,
            validation,
            predictor.local_head,
            scaler,
            near_dead_threshold,
            dominant_threshold,
            redundancy_threshold,
            final_stability_scores,
            stability_threshold,
            top_activations_per_concept,
        )

        if self._parameter_digest(predictor) != predictor_digest:
            raise RuntimeError("frozen WISDOM parameters changed during concept discovery")

        torch.save(
            {
                "model_state_dict": final_model.cpu().state_dict(),
                "embedding_dim": embedding_width,
                "concept_count": suggested_width,
                "lambda": final_lambda,
                "seed": final_seed,
                "scaler": scaler.state(),
                "predictor_checkpoint_sha256": self._file_digest(checkpoint),
            },
            final_root / "concept_model.pt",
        )
        torch.save(scaler.state(), final_root / "embedding_scaler.pt")
        self._write_csv(
            calibration_root / "calibration_results.csv",
            calibration_rows + final_rows,
        )
        self._write_csv(calibration_root / "calibration_curve.csv", grouped)
        self._write_sampling_manifest(
            calibration_root / "sampling.jsonl",
            {"train": train, "validation": validation},
        )
        self._plot_calibration_curve(
            grouped,
            chosen_lambda,
            calibration_root / "sparsity-fidelity.png",
        )
        self._write_csv(final_root / "concept_report.csv", concept_rows)
        self._write_csv(final_root / "top_activations.csv", top_rows)

        summary = {
            "predictor_model_version": predictor_state["model_version"],
            "predictor_frozen": True,
            "labels_used": False,
            "surface_ground_truth_used": False,
            "test_used": False,
            "subset": subset,
            "sampled_train_points": len(train_standard),
            "sampled_validation_points": len(val_standard),
            "K_probe": embedding_width,
            "selected_probe_lambda": chosen_lambda,
            "dead_features": int((activation_rates == 0.0).sum()),
            "near_dead_features": int(
                ((activation_rates > 0.0) & (activation_rates <= near_dead_threshold)).sum()
            ),
            "stable_features": int(stable_mask.sum()),
            "redundant_features": int(redundant_mask.sum()),
            "suggested_K_final": suggested_width,
            "final_lambda": final_lambda,
            "final_seed": final_seed,
            "final_dead_features": int(final_dead.sum()),
            "final_near_dead_features": int(final_near_dead.sum()),
            "final_stable_features": int(final_stable.sum()),
            "final_unstable_features": int((~final_stable).sum()),
            "final_redundant_features": int(final_redundant.sum()),
            "final_validation": {
                key: float(value)
                for key, value in final_measure.items()
                if isinstance(value, (float, int))
            },
        }
        config = {
            "lambda_values": list(lambdas),
            "calibration_seeds": list(seeds),
            "maximum_points_per_protein": maximum_points_per_protein,
            "near_dead_threshold": near_dead_threshold,
            "dominant_threshold": dominant_threshold,
            "redundancy_threshold": redundancy_threshold,
            "stability_threshold": stability_threshold,
            "epochs": epochs,
            "patience": patience,
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "selected_probe_lambda": chosen_lambda,
            "suggested_K_final": suggested_width,
            "final_lambda": final_lambda,
            "final_seed": final_seed,
        }
        config_paths = (
            output / "config.yaml",
            calibration_root / "config.yaml",
            final_root / "config.yaml",
        )
        for config_path in config_paths:
            config_path.write_text(
                json.dumps(config, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        self.metrics.log("suggested_concept_count", float(suggested_width))
        self.metrics.log("final_dead_features", float(final_dead.sum()))
        self.metrics.log("final_stable_features", float(final_stable.sum()))
        self.metrics.log("final_redundant_features", float(final_redundant.sum()))
        self.metrics.log(
            "final_mean_active_concepts",
            float(final_measure["mean_active_features"]),
        )
        self.metrics.log("final_reconstruction_nmse", float(final_measure["reconstruction_nmse"]))
        final_local_pearson = final_measure["local_logit_pearson"]
        if final_local_pearson is not None:
            self.metrics.log("final_local_logit_pearson", float(final_local_pearson))
        self.log(
            f"Sparse interpretation complete: K_probe={embedding_width}, "
            f"K_final={suggested_width}, lambda={final_lambda:g}, "
            f"dead={int(final_dead.sum())}, stable={int(final_stable.sum())}, "
            f"redundant={int(final_redundant.sum())}, "
            f"mean_active={float(final_measure['mean_active_features']):.2f}"
        )
        return summary

    def _load_predictor(
        self,
        checkpoint: Path,
        device    : torch.device,
    ) -> tuple[WisdomV1, Mapping[str, Any], Mapping[str, Any]]:
        """Restore and freeze the explicit post-HPO V1 predictor.

        Args:
            checkpoint: File artifact or directory containing ``best-model.pt``.
            device: Inference device.

        Returns:
            Frozen predictor, checkpoint mapping, and runtime collator parameters.

        Raises:
            ValueError: If the artifact is not one compatible final V1 checkpoint.
        """
        path = checkpoint / "best-model.pt" if checkpoint.is_dir() else checkpoint
        state = torch.load(path, map_location="cpu", weights_only=True)
        if not isinstance(state, Mapping) or state.get("model_version") != 1:
            raise ValueError("interpretability requires an explicit WisdomV1 best-model checkpoint")
        parameters = state.get("model_parameters")
        if not isinstance(parameters, Mapping):
            raise ValueError("predictor checkpoint lacks model_parameters")
        predictor_module, _ = _create_model(1, parameters)
        if not isinstance(predictor_module, WisdomV1):
            raise ValueError("checkpoint did not reconstruct a WisdomV1 predictor")
        predictor = predictor_module
        predictor.load_state_dict(state["state_dict"])
        predictor.to(device).eval()
        predictor.requires_grad_(False)
        data_parameters = state.get("data_parameters", {})
        if not isinstance(data_parameters, Mapping):
            raise ValueError("predictor data_parameters must be a mapping")
        return (
            predictor,
            cast(Mapping[str, Any], state),
            cast(Mapping[str, Any], data_parameters),
        )

    def _extract(
        self,
        predictor                    : WisdomV1,
        dataset                      : Path,
        split                        : str,
        subset                       : str,
        data_parameters              : Mapping[str, Any],
        maximum_points_per_protein   : int,
        seed                         : int,
        device                       : torch.device,
    ) -> dict[str, Any]:
        """Extract uniformly sampled surface representations in protein order.

        Args:
            predictor: Frozen V1 model.
            dataset: Managed dataset root.
            split: ``train`` or ``val``; test is intentionally unsupported here.
            subset: Training-view name; validation remains complete by dataset contract.
            data_parameters: Collator budgets saved with the predictor checkpoint.
            maximum_points_per_protein: Uniform point cap, or zero for all points.
            seed: Sampling seed.
            device: Predictor inference device.

        Returns:
            CPU embeddings, local logits, coordinates, point indices, identifiers, and bag pointer.
        """
        if split not in {"train", "val"}:
            raise ValueError("concept extraction is restricted to train and validation")
        source   = WisdomDataset(dataset, split, subset=subset, include_surface_geometry=True)
        collator = WisdomCollator(
            atom_spatial_k=int(data_parameters.get("atom_spatial_k", 16)),
            surface_atom_k=int(data_parameters.get("surface_atom_k", 16)),
            diffusion_spectral_modes=int(data_parameters.get("diffusion_spectral_modes", 128)),
            relation_mode=str(data_parameters.get("relation_mode", "full_relational")),
            curvature_scale_count=int(data_parameters.get("curvature_scale_count", 0)),
        )
        generator = torch.Generator().manual_seed(seed)
        embeddings: list[Tensor] = []
        logits: list[Tensor]     = []
        positions: list[Tensor]  = []
        point_ids: list[Tensor]  = []
        proteins: list[str]      = []
        pointer                  = [0]
        signature = inspect.signature(predictor.encode_surface).parameters

        with torch.inference_mode():
            for sample_index in range(len(source)):
                sample = source[sample_index]
                host    = collator((sample,))
                tensors = _device_batch(host, device)
                arguments = {
                    name: value
                    for name, value in _model_inputs(tensors).items()
                    if name in signature
                }
                surface, local = predictor.encode_surface(**arguments)
                count = len(surface)
                if maximum_points_per_protein and count > maximum_points_per_protein:
                    selected = torch.randperm(count, generator=generator)[
                        :maximum_points_per_protein
                    ]
                    selected = selected.sort().values
                else:
                    selected = torch.arange(count)
                device_selected = selected.to(device)
                embeddings.append(surface[device_selected].float().cpu())
                logits.append(local[device_selected].float().cpu())
                positions.append(host["surface_positions"][selected].float())
                point_ids.append(selected)
                proteins.append(str(sample["identifier"]))
                pointer.append(pointer[-1] + len(selected))

        return {
            "embeddings": torch.cat(embeddings),
            "local_logits": torch.cat(logits),
            "positions": torch.cat(positions),
            "point_ids": torch.cat(point_ids),
            "proteins": proteins,
            "ptr": torch.tensor(pointer, dtype=torch.long),
        }

    def _fit_candidate(
        self,
        training               : Tensor,
        validation             : Tensor,
        training_metadata      : Mapping[str, Any],
        validation_metadata    : Mapping[str, Any],
        local_head             : torch.nn.Linear,
        scaler                 : EmbeddingScaler,
        concept_count          : int,
        sparse_lambda          : float,
        seed                   : int,
        epochs                 : int,
        patience               : int,
        batch_size             : int,
        learning_rate          : float,
        device                 : torch.device,
    ) -> tuple[SparseConceptModel, dict[str, float | int | None]]:
        """Train one sparse candidate and retain its best label-free validation state.

        Args:
            training: Standardized train embeddings ``[N,H]``.
            validation: Standardized validation embeddings ``[V,H]``.
            training_metadata: Frozen train logits used only to normalize the fidelity scale.
            validation_metadata: Original validation logits and protein pointer.
            local_head: Frozen predictor head mapping physical embeddings to local logits.
            scaler: Train-only transform used to recover physical embeddings.
            concept_count: Candidate width ``K``.
            sparse_lambda: Non-negative mean-ReLU penalty.
            seed: Reproducible initialization and minibatch seed.
            epochs: Maximum optimization epochs.
            patience: Validation plateaus before stopping.
            batch_size: Train points per optimizer step.
            learning_rate: Adam learning rate.
            device: Sparse optimization device.

        Returns:
            Best CPU model and its complete validation metrics.
        """
        torch.manual_seed(seed)
        model     = SparseConceptModel(training.shape[1], concept_count).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
        generator = torch.Generator().manual_seed(seed)
        local_head = local_head.to(device)
        local_variance = training_metadata["local_logits"].float().var(correction=0).clamp_min(1e-8)
        best_error = float("inf")
        best_state = deepcopy(model.state_dict())
        stale      = 0

        for _epoch in range(epochs):
            model.train()
            order = torch.randperm(len(training), generator=generator)
            for start in range(0, len(order), batch_size):
                batch = training[order[start : start + batch_size]].to(device)
                concepts, reconstructed = model(batch)
                original_logits = local_head(scaler.inverse(batch)).squeeze(-1)
                rebuilt_logits  = local_head(scaler.inverse(reconstructed)).squeeze(-1)
                reconstruction  = (reconstructed - batch).square().mean()
                fidelity = (
                    (rebuilt_logits - original_logits).square().mean()
                    / local_variance.to(device)
                )
                loss            = reconstruction + fidelity + sparse_lambda * concepts.mean()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                model.normalize_decoder()

            measured = self._measure(
                model,
                validation.to(device),
                validation_metadata,
                local_head,
                scaler,
            )
            error = float(measured["selection_error"])
            if error < best_error - 1.0e-5:
                best_error = error
                best_state = deepcopy(model.state_dict())
                stale      = 0
            else:
                stale += 1
            if stale >= patience:
                break

        model.load_state_dict(best_state)
        metrics = self._measure(
            model,
            validation.to(device),
            validation_metadata,
            local_head,
            scaler,
        )
        scalar_metrics: dict[str, float | int | None] = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (float, int)) or value is None
        }
        return model, scalar_metrics

    def _measure(
        self,
        model      : SparseConceptModel,
        standardized: Tensor,
        metadata   : Mapping[str, Any],
        local_head : torch.nn.Linear,
        scaler     : EmbeddingScaler,
    ) -> dict[str, Any]:
        """Measure reconstruction, prediction fidelity, sparsity, and redundancy.

        Args:
            model: Sparse model under evaluation.
            standardized: Standardized split embeddings ``[N,H]``.
            metadata: Original local logits and protein pointer for the same sampled points.
            local_head: Frozen Wisdom local head.
            scaler: Train-only standardization transform.

        Returns:
            Scalar metrics plus per-concept activation rates and maximum decoder similarities.
        """
        model.eval()
        device = next(model.parameters()).device
        values = standardized.to(device)
        with torch.inference_mode():
            concepts, reconstruction = model(values)
            rebuilt_logits = local_head(scaler.inverse(reconstruction)).squeeze(-1).float().cpu()
        original_logits = metadata["local_logits"].float().cpu()
        reconstruction_nmse = float((reconstruction.cpu() - standardized.cpu()).square().mean())
        local_mse = float((rebuilt_logits - original_logits).square().mean())
        local_variance = float(original_logits.var(correction=0))
        fidelity_scale = max(local_variance, 1.0e-8)
        local_pearson = self._correlation(original_logits, rebuilt_logits)
        local_r2      = 1.0 - local_mse / local_variance if local_variance > 1.0e-8 else None
        original_protein = self._bag_max(original_logits, metadata["ptr"])
        rebuilt_protein  = self._bag_max(rebuilt_logits, metadata["ptr"])
        protein_mae      = float((original_protein - rebuilt_protein).abs().mean())
        protein_corr     = self._correlation(original_protein, rebuilt_protein)
        probability_gap = float(
            (torch.sigmoid(original_protein) - torch.sigmoid(rebuilt_protein)).abs().mean()
        )
        active = concepts.cpu() > 0.0
        active_counts = active.sum(dim=1).float()
        activation_rates = active.float().mean(dim=0)
        weights = model.decoder.weight.detach().float().cpu()
        cosines = weights.T @ weights
        cosines.fill_diagonal_(0.0)
        decoder_cosines = cosines.max(dim=1).values
        selection_error = reconstruction_nmse + local_mse / fidelity_scale + probability_gap
        return {
            "reconstruction_nmse": reconstruction_nmse,
            "local_logit_mse": local_mse,
            "local_logit_pearson": local_pearson,
            "local_logit_r2": local_r2,
            "local_logit_count": len(original_logits),
            "protein_logit_mae": protein_mae,
            "protein_logit_pearson": protein_corr,
            "protein_probability_mae": probability_gap,
            "protein_logit_count": len(original_protein),
            "mean_active_features": float(active_counts.mean()),
            "median_active_features": float(active_counts.median()),
            "p90_active_features": float(torch.quantile(active_counts, 0.90)),
            "active_fraction": float(active.float().mean()),
            "dead_features": int((activation_rates == 0.0).sum()),
            "selection_error": selection_error,
            "activation_rates": activation_rates,
            "decoder_cosines": decoder_cosines,
        }

    @staticmethod
    def _aggregate(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, float]]:
        """Average calibration evidence across seeds for each lambda.

        Args:
            rows: Per-seed candidate metrics.

        Returns:
            Lambda-sorted means used by deterministic Pareto selection.
        """
        groups: dict[float, list[Mapping[str, Any]]] = {}
        for row in rows:
            groups.setdefault(float(row["lambda"]), []).append(row)
        return [
            {
                "lambda": value,
                "active_fraction": (
                    sum(float(row["active_fraction"]) for row in selected) / len(selected)
                ),
                "selection_error": (
                    sum(float(row["selection_error"]) for row in selected) / len(selected)
                ),
            }
            for value, selected in sorted(groups.items())
        ]

    @staticmethod
    def _pareto_knee(rows: Sequence[Mapping[str, float]]) -> float:
        """Choose the maximum-bend point on the non-dominated error/sparsity frontier.

        Both axes are minimized and normalized to ``[0,1]``. Dominated points are removed. The
        selected point has the greatest perpendicular distance below the line joining the two
        frontier extremes, making the result deterministic and scale-independent.

        Args:
            rows: Lambda, active-fraction, and selection-error means.

        Returns:
            Lambda at the Pareto knee, or nearest-to-utopia point for a short frontier.
        """
        ordered = sorted(rows, key=lambda row: (row["active_fraction"], row["selection_error"]))
        frontier: list[Mapping[str, float]] = []
        best_error = float("inf")
        for row in ordered:
            if row["selection_error"] < best_error:
                frontier.append(row)
                best_error = row["selection_error"]
        if len(frontier) < 3:
            selected = min(
                frontier or rows,
                key=lambda row: row["active_fraction"] + row["selection_error"],
            )
            return float(selected["lambda"])
        xs = torch.tensor([row["active_fraction"] for row in frontier])
        ys = torch.tensor([row["selection_error"] for row in frontier])
        x  = (xs - xs.min()) / (xs.max() - xs.min()).clamp_min(1.0e-12)
        y  = (ys - ys.min()) / (ys.max() - ys.min()).clamp_min(1.0e-12)
        start = torch.stack((x[0], y[0]))
        end   = torch.stack((x[-1], y[-1]))
        line  = end - start
        points = torch.stack((x, y), dim=1)
        distances = torch.abs(
            line[0] * (start[1] - points[:, 1]) - (start[0] - points[:, 0]) * line[1]
        ) / line.norm().clamp_min(1.0e-12)
        return float(frontier[int(distances.argmax())]["lambda"])

    @staticmethod
    def _stability(
        models   : Sequence[SparseConceptModel],
        threshold: float,
    ) -> dict[str, Tensor | int]:
        """Match decoder directions across seeds with Hungarian assignment.

        Args:
            models: Same-width models fitted at one lambda.
            threshold: Mean matched cosine required for stability.

        Returns:
            Reference-order cosine scores and stable/unstable counts.
        """
        reference = models[0].decoder.weight.detach().float().cpu().T
        score_rows = [torch.ones(len(reference))]
        for model in models[1:]:
            candidate = model.decoder.weight.detach().float().cpu().T
            cosine    = reference @ candidate.T
            left, right = linear_sum_assignment((-cosine).numpy())
            matched = torch.zeros(len(reference))
            matched[torch.from_numpy(left)] = cosine[left, right]
            score_rows.append(matched)
        scores = torch.stack(score_rows).mean(dim=0)
        stable = scores >= threshold
        return {
            "scores": scores,
            "stable": int(stable.sum()),
            "unstable": int((~stable).sum()),
        }

    @staticmethod
    def _independent_concepts(
        decoder          : Tensor,
        eligible         : Tensor,
        stability_scores : Tensor,
        threshold        : float,
    ) -> Tensor:
        """Keep one representative from each strongly similar eligible decoder direction.

        Concepts are considered in decreasing stability order. A candidate survives only when its
        positive cosine with every already retained direction is below ``threshold``. Codes are
        non-negative, so opposite decoder directions are not interchangeable duplicates. This
        greedy rule removes matching directions without deleting both members of a redundant pair.

        Args:
            decoder: Unit-normalized decoder columns with shape ``[H,K]``.
            eligible: Live-and-stable concept mask ``bool [K]``.
            stability_scores: Seed-matching score for each reference concept ``[K]``.
            threshold: Positive cosine at or above which a later concept is redundant.

        Returns:
            Boolean mask ``[K]`` containing the retained representatives.
        """
        retained = torch.zeros_like(eligible, dtype=torch.bool)
        ordered  = torch.argsort(stability_scores, descending=True)
        for candidate in ordered.tolist():
            if not bool(eligible[candidate]):
                continue
            previous = retained.nonzero(as_tuple=False).flatten()
            if len(previous):
                similarities = decoder[:, candidate] @ decoder[:, previous]
                if bool(torch.any(similarities >= threshold)):
                    continue
            retained[candidate] = True
        return retained

    def _concept_reports(
        self,
        model                         : SparseConceptModel,
        train_standard                : Tensor,
        val_standard                  : Tensor,
        train_metadata                : Mapping[str, Any],
        validation_metadata           : Mapping[str, Any],
        local_head                    : torch.nn.Linear,
        scaler                        : EmbeddingScaler,
        near_dead_threshold           : float,
        dominant_threshold            : float,
        redundancy_threshold          : float,
        stability_scores              : Tensor,
        stability_threshold           : float,
        top_activations_per_concept   : int,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Build per-concept diagnostics, knockouts, and visualization references.

        Args:
            model: Selected final concept model.
            train_standard: Standardized training embeddings.
            val_standard: Standardized validation embeddings.
            train_metadata: Training protein IDs, point IDs, coordinates, and pointer.
            validation_metadata: Validation metadata aligned to ``val_standard``.
            local_head: Frozen Wisdom head.
            scaler: Train-only embedding scaler.
            near_dead_threshold: Practical near-dead activation-rate boundary.
            dominant_threshold: Dominant activation-rate boundary.
            redundancy_threshold: Decoder similarity warning boundary.
            stability_scores: Reference-order matched decoder cosines across final seeds.
            stability_threshold: Minimum mean matched cosine classified as stable.
            top_activations_per_concept: References retained per concept.

        Returns:
            Concept-table rows and top-activation rows.
        """
        device = next(model.parameters()).device
        with torch.inference_mode():
            train_codes, _ = model(train_standard.to(device))
            val_codes, val_reconstruction = model(val_standard.to(device))
        train_codes = train_codes.cpu()
        val_codes   = val_codes.cpu()
        decoder     = model.decoder.weight.detach().float().cpu()
        similarities = decoder.T @ decoder
        similarities.fill_diagonal_(0.0)
        maximum_similarity = similarities.max(dim=1).values

        # Local knockout effects have a closed form because decoder and frozen local head are
        # linear. Protein effects still recompute MAX within each validation bag.

        head_weight = local_head.weight.detach().float().cpu().squeeze(0)
        physical_decoder = scaler.scale[:, None] * decoder
        local_slopes = head_weight @ physical_decoder
        baseline_local = (
            local_head(scaler.inverse(val_reconstruction).to(device))
            .squeeze(-1)
            .detach()
            .cpu()
        )
        baseline_bags  = self._bag_max(baseline_local, validation_metadata["ptr"])
        concept_rows: list[dict[str, Any]] = []
        top_rows: list[dict[str, Any]]     = []
        train_owner = torch.bucketize(
            torch.arange(len(train_codes)),
            train_metadata["ptr"][1:],
            right=True,
        )
        for concept in range(train_codes.shape[1]):
            train_values = train_codes[:, concept]
            val_values   = val_codes[:, concept]
            train_rate   = float((train_values > 0.0).float().mean())
            val_rate     = float((val_values > 0.0).float().mean())
            local_delta  = val_values * local_slopes[concept]
            knocked_bags = self._bag_max(
                baseline_local - local_delta,
                validation_metadata["ptr"],
            )
            nonzero = train_values[train_values > 0.0]
            concept_rows.append(
                {
                    "concept_id": concept,
                    "activation_rate_train": train_rate,
                    "activation_rate_val": val_rate,
                    "mean_nonzero_activation": float(nonzero.mean()) if len(nonzero) else 0.0,
                    "variance": float(train_values.var(correction=0)),
                    "dead": train_rate == 0.0,
                    "near_dead": 0.0 < train_rate <= near_dead_threshold,
                    "dominant": train_rate >= dominant_threshold,
                    "stable": float(stability_scores[concept]) >= stability_threshold,
                    "stability_score": float(stability_scores[concept]),
                    "max_decoder_similarity": float(maximum_similarity[concept]),
                    "potentially_redundant": (
                        float(maximum_similarity[concept]) >= redundancy_threshold
                    ),
                    "reconstruction_importance": float(
                        train_values.mean() * decoder[:, concept].norm()
                    ),
                    "local_logit_knockout_effect": float(local_delta.abs().mean()),
                    "protein_logit_knockout_effect": float(
                        (baseline_bags - knocked_bags).abs().mean()
                    ),
                }
            )
            top_count = min(top_activations_per_concept, len(train_values))
            for rank, point in enumerate(torch.topk(train_values, top_count).indices.tolist(), 1):
                owner = int(train_owner[point])
                coordinate = train_metadata["positions"][point]
                top_rows.append(
                    {
                        "concept_id": concept,
                        "rank": rank,
                        "activation": float(train_values[point]),
                        "protein": train_metadata["proteins"][owner],
                        "surface_point_index": int(train_metadata["point_ids"][point]),
                        "x": float(coordinate[0]),
                        "y": float(coordinate[1]),
                        "z": float(coordinate[2]),
                    }
                )
        return concept_rows, top_rows

    @staticmethod
    def _local_lambdas(selected: float) -> tuple[float, ...]:
        """Return a three-point logarithmic neighborhood around probe lambda.

        Args:
            selected: Probe lambda selected by knee or user override.

        Returns:
            Sorted unique local values used for clean final retraining.
        """
        if selected == 0.0:
            return (0.0, 1.0e-6, 1.0e-5)
        return (selected / math.sqrt(10.0), selected, selected * math.sqrt(10.0))

    @staticmethod
    def _bag_max(values: Tensor, pointer: Tensor) -> Tensor:
        """Apply the frozen V1 existential MAX rule to ordered protein bags.

        Args:
            values: Point logits ``[N]``.
            pointer: CPU prefix boundaries ``[B+1]``.

        Returns:
            Protein logits ``[B]``.
        """
        return torch.stack([
            values[int(start) : int(stop)].max()
            for start, stop in pairwise(pointer)
        ])

    @staticmethod
    def _correlation(first: Tensor, second: Tensor) -> float | None:
        """Compute Pearson correlation without inventing a value for constant vectors.

        Args:
            first: First aligned vector.
            second: Second aligned vector.

        Returns:
            Pearson correlation, or ``None`` when either variance is numerically zero.
        """
        first  = first.float() - first.float().mean()
        second = second.float() - second.float().mean()
        denominator = first.norm() * second.norm()
        if float(denominator) <= 1.0e-12:
            return None
        return float((first @ second) / denominator)

    @staticmethod
    def _metric_text(value: float | int | None) -> str:
        """Format one optional metric without converting undefined evidence to zero.

        Args:
            value: Numeric metric or ``None`` when its mathematical denominator vanished.

        Returns:
            Four-decimal value or the literal ``unavailable``.
        """
        return "unavailable" if value is None else f"{float(value):.4f}"

    @staticmethod
    def _parameter_digest(model: torch.nn.Module) -> str:
        """Fingerprint predictor parameters to prove they remain frozen.

        Args:
            model: Frozen predictor.

        Returns:
            SHA-256 over ordered CPU tensor bytes.
        """
        digest = hashlib.sha256()
        for name, value in model.state_dict().items():
            digest.update(name.encode("utf-8"))
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
        return digest.hexdigest()

    @staticmethod
    def _file_digest(path: Path) -> str:
        """Hash an explicit checkpoint file for interpretability provenance.

        Args:
            path: Checkpoint file or artifact directory.

        Returns:
            Lowercase SHA-256 digest.
        """
        selected = path / "best-model.pt" if path.is_dir() else path
        digest   = hashlib.sha256()
        with selected.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    @staticmethod
    def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
        """Write one stable union-schema diagnostic table.

        Args:
            path: Destination CSV path.
            rows: JSON-compatible row mappings.
        """
        fields = sorted({str(field) for row in rows for field in row})
        with path.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)

    @staticmethod
    def _write_sampling_manifest(
        path   : Path,
        splits : Mapping[str, Mapping[str, Any]],
    ) -> None:
        """Record every sampled surface-point index without duplicating its NPZ coordinates.

        Args:
            path: Destination JSONL path.
            splits: Train and validation extraction metadata with protein pointers and point IDs.

        Each line identifies one protein and the exact original surface-point indices used for
        concept fitting or validation. Coordinates remain in the immutable dataset and can be
        recovered from those indices; top activations additionally copy coordinates into their CSV.
        """
        with path.open("w", encoding="utf-8") as stream:
            for split, metadata in splits.items():
                pointer   = cast(Tensor, metadata["ptr"])
                point_ids = cast(Tensor, metadata["point_ids"])
                proteins  = cast(Sequence[str], metadata["proteins"])
                for protein, start, stop in zip(
                    proteins,
                    pointer[:-1].tolist(),
                    pointer[1:].tolist(),
                    strict=True,
                ):
                    record = {
                        "split": split,
                        "protein": protein,
                        "surface_point_indices": point_ids[start:stop].tolist(),
                    }
                    stream.write(json.dumps(record, separators=(",", ":")) + "\n")

    @staticmethod
    def _plot_calibration_curve(
        rows            : Sequence[Mapping[str, float]],
        selected_lambda : float,
        path            : Path,
    ) -> None:
        """Render the label-free sparsity/fidelity trade-off and selected knee.

        Args:
            rows: Seed-averaged lambda, active-fraction, and selection-error records.
            selected_lambda: Probe lambda selected automatically or supplied by the researcher.
            path: Destination PNG path.
        """
        ordered  = sorted(rows, key=lambda row: float(row["active_fraction"]), reverse=True)
        selected = next(row for row in ordered if float(row["lambda"]) == selected_lambda)

        figure, axis = plt.subplots(figsize=(7.0, 4.5), constrained_layout=True)
        axis.plot(
            [float(row["active_fraction"]) for row in ordered],
            [float(row["selection_error"]) for row in ordered],
            marker="o",
            color="#2563eb",
        )
        axis.scatter(
            [float(selected["active_fraction"])],
            [float(selected["selection_error"])],
            s=90,
            color="#dc2626",
            label=f"selected λ={selected_lambda:g}",
            zorder=3,
        )
        axis.set_xlabel("Active concept fraction (lower is sparser)")
        axis.set_ylabel("Reconstruction + fidelity error (lower is better)")
        axis.set_title("Sparse concept calibration")
        axis.grid(alpha=0.25)
        axis.legend()
        figure.savefig(path, dpi=180)
        plt.close(figure)
