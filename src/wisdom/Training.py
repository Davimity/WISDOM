"""Version-resolved trainable WISDOM Work for LambdaForge 0.13."""

from __future__ import annotations

import time
import json
import torch
import inspect
import importlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from torch import Tensor
from collections.abc import Mapping
from torch.nn import functional as F
from torch.utils.data import DataLoader

from wisdom.models.WisdomV1 import WisdomV1
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite
from wisdom.evaluation.SurfaceMetricSuite import SurfaceMetricSuite
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder


_MODEL_INPUT_NAMES = (
    "atomic_numbers",
    "residue_type_ids",
    "atom_edge_index",
    "atom_edge_types",
    "surface_curvatures",
    "surface_atom_neighbors",
    "surface_atom_distances",
    "surface_atom_normal_offsets",
    "surface_atom_tangential_distances",
    "surface_atom_mask",
    "surface_area_weights",
    "surface_batch",
    "surface_operators",
    "surface_ptr",
)

_V3_INPUT_NAMES = (
    "surface_positions",
    "surface_normals",
    "surface_neighbors",
    "surface_neighbor_mask",
)


class Training(lf.Work):
    """Train and evaluate one WISDOM model with framework-owned run services."""

    def run(
        self,
        dataset              : Path,
        model_version        : int = 1,
        subset               : str = "full",
        hidden_dim           : int = 128,
        embedding_dim        : int = 32,
        use_residue_type     : bool = True,
        atomic_layers        : int = 2,
        projection_depth     : int = 1,
        surface_layers       : int = 2,
        atom_spatial_k       : int = 16,
        surface_atom_k       : int = 16,
        diffusion_spectral_modes: int = 128,
        surface_atom_radius  : float = 6.0,
        surface_chunk_size   : int = 8192,
        atomic_message_chunk_size: int = 65536,
        dropout              : float = 0.2,
        surface_encoder_type : str = "diffusion",
        surface_patch_size   : int = 64,
        pooling_type         : str = "max",
        topk_fraction        : float = 0.05,
        attention_hidden_dim : int = 32,
        regional_diffusion_scale: float = 2.5,
        log_sum_exp_beta     : float = 5.0,
        learning_rate        : float = 3.0e-4,
        weight_decay         : float = 1.0e-4,
        batch_size           : int = 2,
        epochs               : int = 100,
        patience             : int = 30,
        minimum_delta         : float = 1.0e-3,
        precision            : str = "auto",
        surface_metrics      : bool = True,
        data_workers         : int = 0,
    ) -> dict[str, Any]:
        """Train and evaluate one compatible WISDOM generation from a managed DatasetVersion.

        LambdaForge resolves ``dataset``, expands seeds and search parameters, binds the current
        seed, records metrics, and ranks the YAML objective. WISDOM owns only the PyTorch
        scientific loop and its model/data contracts.

        Args:
            dataset: Resolved managed dataset root containing LambdaForge ``index.jsonl``.
            model_version: Architecture generation resolved from ``WisdomV{N}`` by convention.
            subset: Full data or a deterministic training-view name.
            hidden_dim: Shared atom/surface latent width.
            embedding_dim: Element and optional residue embedding width.
            use_residue_type: Include learned residue category features when true.
            atomic_layers: Relation-aware atomic graph layer count.
            projection_depth: Atom-context/curvature projection MLP depth.
            surface_layers: Surface-encoder block count.
            atom_spatial_k: Runtime atomic spatial-neighbor rank budget.
            surface_atom_k: Runtime nearest-atom budget per surface point.
            diffusion_spectral_modes: Runtime low-frequency mode budget per protein.
            surface_atom_radius: Physical transfer cutoff used for feature normalization.
            surface_chunk_size: Maximum surface points in one atom-transfer chunk.
            atomic_message_chunk_size: Maximum atomic messages materialized per RGCN chunk.
            dropout: Dropout probability in ``[0,1)``.
            surface_encoder_type: V3 surface-propagation hypothesis; v1 requires ``diffusion``.
            surface_patch_size: V3 serialized-attention patch bound.
            pooling_type: V2 pooling hypothesis; v1 requires ``max``.
            topk_fraction: V2 fraction retained by top-k mean pooling.
            attention_hidden_dim: V2 attention score-network hidden width.
            regional_diffusion_scale: V2 physical smoothing length before regional MAX, in Å.
            log_sum_exp_beta: V2 normalized log-sum-exp inverse temperature.
            learning_rate: Positive AdamW learning rate.
            weight_decay: Non-negative AdamW decoupled weight decay.
            batch_size: Positive number of disjoint protein graphs per optimizer step.
            epochs: Positive maximum training epoch count.
            patience: Validation epochs without a meaningful AUPRC improvement before stopping.
            minimum_delta: Minimum absolute validation AUPRC increase that resets ``patience``.
            precision: ``auto``, ``float32``, ``bfloat16``, or ``float16`` activation policy.
            surface_metrics: Report evaluation-only local metrics on validation and final test
                without exposing them to loss, checkpoint selection, pruning, or HPO ranking.
            data_workers: Non-negative DataLoader subprocess count per split.

        Returns:
            Best validation epoch/metrics, stopping reason, and held-out test metrics. Adaptively
            pruned Runs return ``test=None`` because they are excluded from HPO ranking.

        Raises:
            ValueError: If a model, data view, parameter, or metric contract is invalid.
            RuntimeError: If validation AUPRC is mathematically undefined for every epoch.
            OSError: If managed arrays or checkpoint/report artifacts cannot be accessed.
        """
        return _train_wisdom(
            self,
            dataset=dataset,
            model_version=model_version,
            subset=subset,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            use_residue_type=use_residue_type,
            atomic_layers=atomic_layers,
            projection_depth=projection_depth,
            surface_layers=surface_layers,
            atom_spatial_k=atom_spatial_k,
            surface_atom_k=surface_atom_k,
            diffusion_spectral_modes=diffusion_spectral_modes,
            surface_atom_radius=surface_atom_radius,
            surface_chunk_size=surface_chunk_size,
            atomic_message_chunk_size=atomic_message_chunk_size,
            dropout=dropout,
            surface_encoder_type=surface_encoder_type,
            surface_patch_size=surface_patch_size,
            pooling_type=pooling_type,
            topk_fraction=topk_fraction,
            attention_hidden_dim=attention_hidden_dim,
            regional_diffusion_scale=regional_diffusion_scale,
            log_sum_exp_beta=log_sum_exp_beta,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            minimum_delta=minimum_delta,
            precision=precision,
            surface_metrics=surface_metrics,
            data_workers=data_workers,
            seed=self.seed or 0,
        )


def _train_wisdom(
    work                 : Training,
    dataset              : Path,
    model_version        : int = 1,
    subset               : str = "full",
    hidden_dim           : int = 128,
    embedding_dim        : int = 32,
    use_residue_type     : bool = True,
    atomic_layers        : int = 2,
    projection_depth     : int = 1,
    surface_layers       : int = 2,
    atom_spatial_k       : int = 16,
    surface_atom_k       : int = 16,
    diffusion_spectral_modes: int = 128,
    surface_atom_radius  : float = 6.0,
    surface_chunk_size   : int = 8192,
    atomic_message_chunk_size: int = 65536,
    dropout              : float = 0.2,
    surface_encoder_type : str = "diffusion",
    surface_patch_size   : int = 64,
    pooling_type         : str = "max",
    topk_fraction        : float = 0.05,
    attention_hidden_dim : int = 32,
    regional_diffusion_scale: float = 2.5,
    log_sum_exp_beta     : float = 5.0,
    learning_rate        : float = 3.0e-4,
    weight_decay         : float = 1.0e-4,
    batch_size           : int = 2,
    epochs               : int = 100,
    patience             : int = 30,
    minimum_delta         : float = 1.0e-3,
    precision            : str = "auto",
    surface_metrics      : bool = True,
    data_workers         : int = 0,
    seed                 : int = 0,
) -> dict[str, Any]:
    """Train and evaluate one compatible WISDOM generation from a managed DatasetVersion.

    This private implementation keeps the public ``Training.run`` signature declarative while
    carrying out the ordinary PyTorch loop.

    Args:
        work: Active LambdaForge Work providing run-owned metrics and output services.
        dataset: Resolved managed dataset root containing LambdaForge ``index.jsonl``.
        model_version: Architecture generation resolved from ``WisdomV{N}`` by convention.
        subset: Full data or a deterministic view such as ``replicate-00/train-25``.
        hidden_dim: Shared atom/surface latent width.
        embedding_dim: Element and optional residue embedding width.
        use_residue_type: Include learned residue category features when true.
        atomic_layers: Relation-aware atomic graph layer count.
        projection_depth: Atom-context/curvature projection MLP depth.
        surface_layers: Surface-encoder block count.
        atom_spatial_k: Runtime atomic spatial-neighbor rank budget.
        surface_atom_k: Runtime nearest-atom budget per surface point.
        diffusion_spectral_modes: Runtime low-frequency mode budget per protein.
        surface_atom_radius: Transfer geometry normalization radius in Å.
        surface_chunk_size: Maximum transfer points per activation chunk.
        atomic_message_chunk_size: Maximum RGCN messages per chunk.
        dropout: Dropout probability in ``[0,1)``.
        surface_encoder_type: V3 controlled surface encoder name.
        surface_patch_size: V3 serialized-attention patch bound.
        pooling_type: V2 protein-level pooling hypothesis; v1 requires ``max``.
        topk_fraction: V2 fraction retained by top-k mean pooling.
        attention_hidden_dim: V2 attention score-network hidden width.
        regional_diffusion_scale: V2 regional diffusion length in Å.
        log_sum_exp_beta: V2 normalized log-sum-exp inverse temperature.
        learning_rate: Positive AdamW learning rate.
        weight_decay: Non-negative AdamW decoupled weight decay.
        batch_size: Positive number of disjoint protein graphs per optimizer step.
        epochs: Positive maximum training epoch count.
        patience: Validation epochs without a meaningful AUPRC improvement before stopping.
        minimum_delta: Minimum absolute validation AUPRC increase that resets ``patience``.
        precision: ``auto``, ``float32``, ``bfloat16``, or ``float16`` activation policy.
        surface_metrics: Report evaluation-only local metrics on validation and final test without
            using them for optimization or model selection.
        data_workers: Non-negative DataLoader subprocess count per split.
        seed: Reproducible seed injected by LambdaForge for each expanded run.

    Returns:
        Best validation epoch/metrics, stopping reason, and held-out test metrics. Adaptively
        pruned Runs return ``test=None`` because they are excluded from HPO ranking.

    Raises:
        ValueError: If a model, data view, parameter, or metric contract is invalid.
        RuntimeError: If validation AUPRC is mathematically undefined for every epoch.
        OSError: If managed arrays or checkpoint/report artifacts cannot be read or written.
    """
    if isinstance(model_version, bool) or model_version < 1:
        raise ValueError("model_version must be a positive integer")
    if model_version == 1 and pooling_type != "max":
        raise ValueError("WISDOM v1 fixes pooling_type='max'")
    if model_version == 1 and surface_encoder_type != "diffusion":
        raise ValueError("WISDOM v1 fixes surface_encoder_type='diffusion'")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("optimizer learning rate must be positive and weight decay non-negative")
    if batch_size < 1 or epochs < 1 or data_workers < 0:
        raise ValueError("batch size/epochs must be positive and data workers non-negative")
    if patience < 1 or minimum_delta < 0.0:
        raise ValueError("patience must be positive and minimum_delta must be non-negative")
    if precision not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError("precision must be auto, bfloat16, float16, or float32")
    if atom_spatial_k < 1 or surface_atom_k < 1 or diffusion_spectral_modes < 1:
        raise ValueError("atom K, surface-atom J, and spectral-mode budget must be positive")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    supports_bfloat16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    if precision == "auto":
        effective_precision = "bfloat16" if supports_bfloat16 else "float32"
    elif (device.type != "cuda" and precision != "float32") or (
        precision == "bfloat16" and not supports_bfloat16
    ):
        effective_precision = "float32"
    else:
        effective_precision = precision

    autocast_dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }.get(effective_precision)
    use_autocast = autocast_dtype is not None
    scaler = torch.amp.GradScaler("cuda", enabled=effective_precision == "float16")

    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")

    trial_index = work.trial.index if work.trial is not None else 0
    run_label   = f"trial={trial_index} seed={seed}"

    work.log(f"[Training {run_label}] loading managed train, validation, and test splits")

    datasets = {
        split: WisdomDataset(
            dataset,
            split,
            subset=subset,
            include_surface_targets=surface_metrics and split in {"val", "test"},
            include_surface_geometry=model_version == 3,
        )
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            value,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=data_workers,
            collate_fn=WisdomCollator(
                atom_spatial_k=atom_spatial_k,
                surface_atom_k=surface_atom_k,
                diffusion_spectral_modes=diffusion_spectral_modes,
            ),
            generator=generator if split == "train" else None,
            pin_memory=device.type == "cuda",
        )
        for split, value in datasets.items()
    }

    # Infer the model input width from the immutable data contract instead of duplicating the
    # preprocessing scale count in a training YAML. All splits must expose the same [M,S,3] shape.

    curvature_widths: dict[str, int] = {}

    for split, split_dataset in datasets.items():
        sample     = split_dataset[0]
        curvature = _tensor(sample, "surface_curvatures")

        if curvature.ndim != 3 or curvature.shape[2] != 3:
            raise ValueError(
                f"{split} surface_curvatures must have shape [M,S,3], got "
                f"{tuple(curvature.shape)}"
            )

        curvature_widths[split] = int(curvature.shape[1] * curvature.shape[2])

    if len(set(curvature_widths.values())) != 1:
        raise ValueError(f"dataset splits disagree on curvature feature width: {curvature_widths}")

    curvature_features = curvature_widths["train"]

    available_parameters = {
        "hidden_dim":           hidden_dim,
        "embedding_dim":        embedding_dim,
        "use_residue_type":     use_residue_type,
        "atomic_layers":        atomic_layers,
        "projection_depth":     projection_depth,
        "surface_layers":       surface_layers,
        "atom_spatial_k":       atom_spatial_k,
        "surface_atom_k":       surface_atom_k,
        "diffusion_spectral_modes": diffusion_spectral_modes,
        "surface_atom_radius":  surface_atom_radius,
        "surface_chunk_size":   surface_chunk_size,
        "atomic_message_chunk_size": atomic_message_chunk_size,
        "dropout":              dropout,
        "surface_encoder_type": surface_encoder_type,
        "surface_patch_size":   surface_patch_size,
        "pooling_type":         pooling_type,
        "topk_fraction":        topk_fraction,
        "attention_hidden_dim": attention_hidden_dim,
        "regional_diffusion_scale": regional_diffusion_scale,
        "log_sum_exp_beta":     log_sum_exp_beta,
        "curvature_features":   curvature_features,
    }
    model, model_parameters = _create_model(model_version, available_parameters)
    architecture_name       = str(getattr(model, "ARCHITECTURE_NAME", type(model).__name__))
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    split_sizes = {split: len(split_dataset) for split, split_dataset in datasets.items()}
    split_storage = {
        split: split_dataset.storage_bytes()
        for split, split_dataset in datasets.items()
    }
    preprocessing_bytes = sum(values["total"] for values in split_storage.values())

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    parameter_mib   = parameter_bytes / 2**20

    work.metrics.log("parameter_count", float(parameter_count))
    work.metrics.log("preprocessing_bytes", float(preprocessing_bytes))

    work.log(
        f"[Training {run_label}] starting on {device.type}; splits={split_sizes}; "
        f"architecture={architecture_name}; curvature_features={curvature_features}; "
        f"batch_size={batch_size}; "
        f"precision={effective_precision}; parameters={parameter_count:,} "
        f"({parameter_mib:.2f} MiB in FP32); preprocessing_bytes={preprocessing_bytes:,}; "
        f"maximum_epochs={epochs}; patience={patience}"
    )

    best_auprc                 = float("-inf")
    best_epoch                 = 0
    epochs_completed           = 0
    epochs_without_improvement = 0
    peak_allocated_gib         = 0.0
    peak_reserved_gib          = 0.0

    stop_reason: str | None = None

    checkpoint = work.run_dir / "best-model.pt"

    # Train up to the safety ceiling. Validation patience decides the useful duration of an
    # ordinary Run, while LambdaForge may cooperatively prune a weak HPO candidate.

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        model.train()
        loss_sum = 0.0
        examples = 0

        maximum_atoms          = 0
        maximum_surface_points = 0
        maximum_atomic_edges   = 0
        maximum_atomic_degree  = 0
        mean_atomic_degree_sum = 0.0
        atom_count_sum          = 0
        surface_point_sum       = 0
        training_batches       = 0
        spectral_modes_seen    : list[int] = []

        for batch in loaders["train"]:
            atomic_numbers = _tensor(batch, "atomic_numbers")
            maximum_surface_points = max(
                maximum_surface_points,
                int(_tensor(batch, "surface_area_weights").shape[0]),
            )
            maximum_atoms = max(maximum_atoms, len(atomic_numbers))
            atom_count_sum += len(atomic_numbers)
            surface_point_sum += int(_tensor(batch, "surface_area_weights").shape[0])
            active_edges  = _tensor(batch, "atom_edge_index")
            maximum_atomic_edges = max(maximum_atomic_edges, active_edges.shape[1])
            degree = torch.bincount(active_edges[1], minlength=len(atomic_numbers))
            maximum_atomic_degree = max(maximum_atomic_degree, int(degree.max()))
            mean_atomic_degree_sum += float(degree.float().mean())
            training_batches += 1
            spectral_modes_seen.extend(
                len(operator["eigenvalues"])
                for operator in _operators(batch)
            )

            tensors = _device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype or torch.bfloat16,
                enabled=use_autocast,
            ):
                output = model(**_model_inputs(tensors))
                target = _tensor(tensors, "target")
                loss   = F.binary_cross_entropy_with_logits(output["logits"], target)

            if scaler.is_enabled():
                scaler.scale(loss).backward()  # type: ignore[no-untyped-call]
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()

            count     = len(target)
            loss_sum += float(loss.detach()) * count
            examples += count

            # The model returns point-level diagnostics in addition to protein logits. Explicitly
            # release the completed batch so it cannot overlap the next batch or validation pass.

            del tensors, output, target, loss

        optimizer.zero_grad(set_to_none=True)

        training_seconds = time.perf_counter() - epoch_started
        train_throughput = examples / max(training_seconds, 1.0e-9)
        train_loss       = loss_sum / max(1, examples)
        validation, surface_validation = _evaluate(
            model,
            loaders["val"],
            device,
            autocast_dtype,
        )

        work.metrics.log("loss", train_loss, step=epoch, split="train")
        work.metrics.log(
            "maximum_surface_points",
            float(maximum_surface_points),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "maximum_atoms",
            float(maximum_atoms),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "maximum_active_atomic_edges",
            float(maximum_atomic_edges),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "maximum_atomic_degree",
            float(maximum_atomic_degree),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "mean_atomic_degree",
            mean_atomic_degree_sum / max(1, training_batches),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "mean_atoms_per_batch",
            atom_count_sum / max(1, training_batches),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "mean_surface_points_per_batch",
            surface_point_sum / max(1, training_batches),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "active_atom_spatial_k",
            float(atom_spatial_k),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "active_surface_atom_k",
            float(surface_atom_k),
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "train_seconds",
            training_seconds,
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "train_proteins_per_second",
            train_throughput,
            step=epoch,
            split="train",
        )
        if spectral_modes_seen:
            work.metrics.log(
                "mean_spectral_modes",
                sum(spectral_modes_seen) / len(spectral_modes_seen),
                step=epoch,
                split="train",
            )

        for name, value in validation.items():
            if value is not None:
                work.metrics.log(name, value, step=epoch, split="val")
        for name, value in surface_validation.items():
            if value is not None:
                work.metrics.log(name, value, step=epoch, split="val")

        objective        = validation["auprc"]
        epochs_completed = epoch

        if objective is not None and objective > best_auprc + minimum_delta:
            best_auprc                 = objective
            best_epoch                 = epoch
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_version":    model_version,
                    "model_parameters": model_parameters,
                    "pooling_type":     pooling_type,
                    "state_dict":       model.state_dict(),
                    "seed":             seed,
                    "epoch":            epoch,
                    "val_auprc":        objective,
                },
                checkpoint,
            )
        elif objective is not None:
            epochs_without_improvement += 1

        # One compact, labelled line per epoch remains readable with two interleaved GPU Runs.
        # The complete metric suite is still stored structurally by LambdaForge above.

        auroc    = validation["auroc"]
        balanced = validation["balanced_accuracy"]

        auprc_text    = "n/a" if objective is None else f"{objective:.4f}"
        auroc_text    = "n/a" if auroc is None else f"{auroc:.4f}"
        balanced_text = "n/a" if balanced is None else f"{balanced:.4f}"
        best_text     = "n/a" if best_epoch == 0 else f"{best_auprc:.4f}"

        surface_micro_text = _metric_text(surface_validation, "surface_micro_auprc")
        surface_macro_text = _metric_text(
            surface_validation,
            "surface_positive_macro_auprc",
        )
        surface_auroc_text = _metric_text(surface_validation, "surface_positive_macro_auroc")

        progress_message = (
            f"{run_label}; loss={train_loss:.5f}; val_auprc={auprc_text}; "
            f"surface_macro_auprc={surface_macro_text}; best={best_text}"
        )

        memory_text = ""
        if device.type == "cuda":
            allocated_gib = torch.cuda.memory_allocated(device) / 2**30
            reserved_gib  = torch.cuda.memory_reserved(device) / 2**30
            peak_gib      = torch.cuda.max_memory_allocated(device) / 2**30
            peak_reserved = torch.cuda.max_memory_reserved(device) / 2**30
            peak_allocated_gib = max(peak_allocated_gib, peak_gib)
            peak_reserved_gib  = max(peak_reserved_gib, peak_reserved)

            memory_text = (
                f" cuda_allocated={allocated_gib:.2f}GiB"
                f" cuda_reserved={reserved_gib:.2f}GiB cuda_peak={peak_gib:.2f}GiB"
            )
            work.metrics.log("cuda_allocated_gib", allocated_gib, step=epoch, split="train")
            work.metrics.log("cuda_reserved_gib", reserved_gib, step=epoch, split="train")
            work.metrics.log("cuda_peak_gib", peak_gib, step=epoch, split="train")

            progress_message += f"; cuda_peak={peak_gib:.2f}GiB"

        epoch_seconds = time.perf_counter() - epoch_started
        proteins_per_second = examples / max(epoch_seconds, 1.0e-9)
        work.metrics.log("epoch_seconds", epoch_seconds, step=epoch, split="train")
        work.metrics.log(
            "proteins_per_second",
            proteins_per_second,
            step=epoch,
            split="train",
        )

        work.progress.update(completed=epoch, total=epochs, message=progress_message)
        work.log(
            f"[Training {run_label}] epoch={epoch}/{epochs} loss={train_loss:.5f} "
            f"protein_val[auprc={auprc_text},auroc={auroc_text},"
            f"balanced_accuracy={balanced_text},best_auprc={best_text}] "
            f"surface_val[micro_auprc={surface_micro_text},"
            f"positive_macro_auprc={surface_macro_text},"
            f"positive_macro_auroc={surface_auroc_text}] "
            f"patience={epochs_without_improvement}/{patience} "
            f"batch_cost=atoms:{maximum_atoms:,},points:{maximum_surface_points:,},"
            f"atomic_edges:{maximum_atomic_edges:,},degree_max:{maximum_atomic_degree},"
            f"K:{atom_spatial_k},J:{surface_atom_k},modes:"
            f"{max(spectral_modes_seen, default=0)} train_throughput={train_throughput:.2f}/s "
            f"epoch_throughput={proteins_per_second:.2f}/s"
            f"{memory_text}"
        )

        # A framework pruning request is checked only after metrics and the best checkpoint have
        # been persisted. The Run then returns normally and LambdaForge records it as pruned.

        if work.stop_requested:
            stop_reason = "adaptive-hpo"
            work.log(f"Stopping candidate after epoch {epoch}: adaptive HPO pruning requested")
            break

        # Patience prevents an otherwise healthy Run from spending epochs on a validation plateau.
        # AUPRC must improve by minimum_delta, so negligible floating-point changes do not reset it.

        if epochs_without_improvement >= patience:
            stop_reason = "validation-patience"
            work.log(
                f"Stopping after epoch {epoch}: validation AUPRC did not improve by "
                f"{minimum_delta:g} for {patience} epochs"
            )
            break

    if best_epoch == 0:
        raise RuntimeError("validation AUPRC remained undefined; both classes are required")

    # Candidate ranking must use the best validation checkpoint rather than the final plateau
    # observation. This unstepped summary does not alter LambdaForge's per-epoch pruning history.

    work.metrics.log("auprc", best_auprc, split="val")

    test_metrics: dict[str, float | None] | None = None
    test_surface_metrics: dict[str, float | None] | None = None

    # Pruned HPO Runs are excluded from candidate scores and need no held-out test evaluation.
    # Completed Runs, including those stopped by patience, evaluate test exactly once at their
    # validation-selected checkpoint.

    if stop_reason != "adaptive-hpo":
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])
        test_metrics, test_surface_metrics = _evaluate(
            model,
            loaders["test"],
            device,
            autocast_dtype,
        )

        for name, value in test_metrics.items():
            if value is not None:
                work.metrics.log(name, value, split="test")
        for name, value in test_surface_metrics.items():
            if value is not None:
                work.metrics.log(name, value, split="test")

    report = {
        "model_version":               model_version,
        "architecture":                architecture_name,
        "pooling_type":                pooling_type,
        "subset":                      subset,
        "seed":                        seed,
        "epochs_completed":            epochs_completed,
        "best_epoch":                  best_epoch,
        "best_val_auprc":              best_auprc,
        "early_stopping_patience":     patience,
        "early_stopping_minimum_delta": minimum_delta,
        "precision":                   effective_precision,
        "surface_metrics_enabled":     surface_metrics,
        "stop_reason":                 stop_reason,
        "test":                        test_metrics,
        "test_surface":                test_surface_metrics,
        "curvature_features":          curvature_features,
        "parameter_count":             parameter_count,
        "preprocessing_bytes":         preprocessing_bytes,
        "atom_spatial_k":              atom_spatial_k,
        "surface_atom_k":              surface_atom_k,
        "diffusion_spectral_modes":    diffusion_spectral_modes,
        "peak_cuda_allocated_gib":     peak_allocated_gib if device.type == "cuda" else None,
        "peak_cuda_reserved_gib":      peak_reserved_gib if device.type == "cuda" else None,
        "split_sizes":                 split_sizes,
        "split_storage_bytes":         split_storage,
    }
    report_path = work.run_dir / "evaluation.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    work.outputs.artifact("best-model", checkpoint, role="checkpoint")
    work.outputs.artifact(
        "evaluation",
        report_path,
        role="report",
        media_type="application/json",
    )
    return report


def _evaluate(
    model         : torch.nn.Module,
    loader        : DataLoader[Any],
    device        : torch.device,
    autocast_dtype: torch.dtype | None,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Evaluate one explicit split with definition-aware LambdaForge metrics.

    Args:
        model: Trained WISDOM model placed on ``device``.
        loader: Non-empty explicit-split graph DataLoader.
        device: CPU or CUDA device receiving tensors.
        autocast_dtype: CUDA mixed-precision dtype, or ``None`` for full float32.

    Returns:
        Protein metric mapping and optional surface metric mapping. Mathematically undefined values
        remain ``None``.
    """
    probabilities     : list[Tensor] = []
    targets           : list[Tensor] = []
    surface_scores    : list[Tensor] = []
    surface_targets   : list[Tensor] = []
    surface_validity  : list[Tensor] = []
    surface_owners    : list[Tensor] = []
    surface_bag_labels: list[Tensor] = []

    protein_offset = 0
    mixed_dtype = (
        autocast_dtype if autocast_dtype in {torch.bfloat16, torch.float16} else None
    )

    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = _device_batch(batch, device)

            with torch.autocast(
                device_type=device.type,
                dtype=mixed_dtype or torch.bfloat16,
                enabled=mixed_dtype is not None,
            ):
                output = model(**_model_inputs(tensors))

            probabilities.append(torch.sigmoid(output["logits"]).cpu())
            protein_target = _tensor(tensors, "target").cpu()
            targets.append(protein_target)

            if "surface_target_hard" in batch and "surface_valid_mask" in batch:
                surface_scores.append(torch.sigmoid(output["surface_logits"]).float().cpu())
                surface_targets.append(_tensor(batch, "surface_target_hard").cpu())
                surface_validity.append(_tensor(batch, "surface_valid_mask").cpu())
                surface_owners.append(_tensor(batch, "surface_batch").cpu() + protein_offset)
                surface_bag_labels.append(protein_target)

            protein_offset += len(protein_target)

            del tensors, output

    protein_metrics = BinaryMetricSuite().compute(torch.cat(probabilities), torch.cat(targets))
    if not surface_scores:
        return protein_metrics, {}

    local_metrics = SurfaceMetricSuite().compute(
        torch.cat(surface_scores),
        torch.cat(surface_targets),
        torch.cat(surface_validity),
        torch.cat(surface_owners),
        torch.cat(surface_bag_labels),
    )
    return protein_metrics, local_metrics


def _metric_text(metrics: Mapping[str, float | None], name: str) -> str:
    """Format one optional metric for compact interleaved training logs.

    Args:
        metrics: Evaluation metric mapping.
        name: Metric key to render.

    Returns:
        Four-decimal value or ``n/a`` when the diagnostic is unavailable or undefined.
    """
    value = metrics.get(name)
    return "n/a" if value is None else f"{value:.4f}"


def _create_model(
    model_version       : int,
    available_parameters: Mapping[str, Any],
) -> tuple[torch.nn.Module, dict[str, Any]]:
    """Instantiate a WISDOM generation through a stable module/class naming convention.

    A trainable generation lives in ``wisdom.models.WisdomV{N}`` and exposes a class with the
    same name. Constructor arguments shared with ``Training.run`` are forwarded automatically;
    parameters irrelevant to that generation are omitted. Consequently a future generation that
    reuses the established training and input contracts needs only its model module and YAML.

    Args:
        model_version: Positive architecture generation number ``N``.
        available_parameters: Research parameters exposed by the common Training Work.

    Returns:
        Instantiated PyTorch model and the exact constructor arguments stored in its checkpoint.

    Raises:
        ValueError: If the convention does not resolve to a PyTorch module class.
        TypeError: If the resolved constructor rejects the compatible parameter mapping.
    """
    class_name = f"WisdomV{model_version}"
    module_name = f"wisdom.models.{class_name}"

    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        if error.name == module_name:
            raise ValueError(f"unsupported WISDOM model version: {model_version}") from error
        raise

    model_class = getattr(module, class_name, None)
    if not isinstance(model_class, type) or not issubclass(model_class, torch.nn.Module):
        raise ValueError(f"{module_name} must expose a torch module class named {class_name}")

    accepted = inspect.signature(model_class.__init__).parameters
    parameters = {
        name: value
        for name, value in available_parameters.items()
        if name in accepted
    }
    model = model_class(**parameters)

    # V1 is a fixed scientific hypothesis, not a name that may silently resolve to the retired
    # dense surface-graph implementation. Keep this check at construction time so an accidental
    # source regression fails before loading an epoch of data or allocating CUDA activations.

    if model_version == 1 and (
        type(model) is not WisdomV1
        or getattr(model, "ARCHITECTURE_NAME", None) != "bounded-atomic-diffusionnet"
        or getattr(model, "STRUCTURAL_SCHEMA_VERSION", None)
        != WisdomDataset.STRUCTURAL_SCHEMA_VERSION
        or not isinstance(getattr(model, "surface_encoder", None), DiffusionSurfaceEncoder)
    ):
        raise RuntimeError(
            "WISDOM v1 must use schema-3 bounded atomic topology and DiffusionSurfaceEncoder"
        )

    return model, parameters


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Recursively move only model inputs and the protein target to one device.

    Args:
        batch: Collated WISDOM graph mapping, which may also contain point-level diagnostics.
        device: Destination CPU or CUDA device.

    Returns:
        New mapping containing model tensors/operator packs and the global target on ``device``.
        DNA sidecars, identifiers, tiers, and unused diagnostics stay on the host.
    """
    selected_names = [*_MODEL_INPUT_NAMES, "target"]
    selected_names.extend(name for name in _V3_INPUT_NAMES if name in batch)
    return {
        name: _move_to_device(batch[name], device)
        for name in selected_names
    }


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Select the bounded common model inputs and optional WISDOM v3 geometry.

    Args:
        batch: Device-resident collated graph mapping.

    Returns:
        Keyword mapping accepted by the active trainable model generation.
    """
    names = list(_MODEL_INPUT_NAMES)
    names.extend(name for name in _V3_INPUT_NAMES if name in batch)
    return {name: batch[name] for name in names}


def _move_to_device(value: Any, device: torch.device) -> Any:
    """Move tensors nested in explicit model-input containers without touching other data.

    Args:
        value: Tensor, mapping, list, tuple, or scalar selected by ``_device_batch``.
        device: Destination CPU or CUDA device.

    Returns:
        Container of the same kind with every tensor moved non-blockingly.
    """
    if isinstance(value, Tensor):
        return value.to(device, non_blocking=True)
    if isinstance(value, Mapping):
        return {key: _move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(item, device) for item in value)
    return value


def _operators(batch: Mapping[str, Any]) -> list[Mapping[str, Tensor]]:
    """Return one validated list of per-protein intrinsic operator mappings.

    Args:
        batch: Collated host or device batch.

    Returns:
        Ordered operator packs aligned to ``surface_ptr``.

    Raises:
        ValueError: If the field is not a list of tensor mappings.
    """
    value = batch.get("surface_operators")
    if not isinstance(value, list) or not all(isinstance(item, Mapping) for item in value):
        raise ValueError("training batch field 'surface_operators' must be a list of mappings")
    return value


def _tensor(batch: Mapping[str, Any], name: str) -> Tensor:
    """Return one required tensor with a precise training-contract failure.

    Args:
        batch: Collated/device-resident mapping.
        name: Required tensor key.

    Returns:
        Tensor stored under ``name``.

    Raises:
        ValueError: If the requested field is absent or not tensor-valued.
    """
    value = batch.get(name)
    if not isinstance(value, Tensor):
        raise ValueError(f"training batch field {name!r} must be a tensor")
    return value
