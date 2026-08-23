"""Trainable WISDOM v1/v2 Work for LambdaForge 0.12."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import lambdaforge as lf
import torch
from torch import Tensor
from torch.nn import functional as F
from torch.utils.data import DataLoader

from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite
from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.WisdomV2 import WisdomV2


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
        dropout              : float = 0.2,
        pooling_type         : str = "max",
        topk_fraction        : float = 0.05,
        attention_hidden_dim : int = 32,
        regional_levels      : int = 1,
        log_sum_exp_beta     : float = 5.0,
        learning_rate        : float = 3.0e-4,
        weight_decay         : float = 1.0e-4,
        batch_size           : int = 2,
        epochs               : int = 100,
        data_workers         : int = 0,
    ) -> dict[str, Any]:
        """Train and evaluate WISDOM v1 or v2 from one managed DatasetVersion.

        LambdaForge resolves ``dataset``, expands seeds and search parameters, binds the current
        seed, records metrics, and ranks the YAML objective. WISDOM owns only the PyTorch
        scientific loop and its model/data contracts.

        Args:
            dataset: Resolved managed dataset root containing LambdaForge ``index.jsonl``.
            model_version: Supported architecture generation, either ``1`` or ``2``.
            subset: Full data or a deterministic training-view name.
            hidden_dim: Shared atom/surface latent width.
            embedding_dim: Element and optional residue embedding width.
            use_residue_type: Include learned residue category features when true.
            atomic_layers: Relation-aware atomic graph layer count.
            projection_depth: Atom-context/curvature projection MLP depth.
            surface_layers: Surface graph layer count.
            dropout: Dropout probability in ``[0,1)``.
            pooling_type: V2 pooling hypothesis; v1 requires ``max``.
            topk_fraction: V2 fraction retained by top-k mean pooling.
            attention_hidden_dim: V2 attention score-network hidden width.
            regional_levels: V2 smoothing rounds before regional MAX.
            log_sum_exp_beta: V2 normalized log-sum-exp inverse temperature.
            learning_rate: Positive AdamW learning rate.
            weight_decay: Non-negative AdamW decoupled weight decay.
            batch_size: Positive number of disjoint protein graphs per optimizer step.
            epochs: Positive maximum training epoch count.
            data_workers: Non-negative DataLoader subprocess count per split.

        Returns:
            Best validation epoch/metrics and final held-out test metrics.

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
            dropout=dropout,
            pooling_type=pooling_type,
            topk_fraction=topk_fraction,
            attention_hidden_dim=attention_hidden_dim,
            regional_levels=regional_levels,
            log_sum_exp_beta=log_sum_exp_beta,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            batch_size=batch_size,
            epochs=epochs,
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
    dropout              : float = 0.2,
    pooling_type         : str = "max",
    topk_fraction        : float = 0.05,
    attention_hidden_dim : int = 32,
    regional_levels      : int = 1,
    log_sum_exp_beta     : float = 5.0,
    learning_rate        : float = 3.0e-4,
    weight_decay         : float = 1.0e-4,
    batch_size           : int = 2,
    epochs               : int = 100,
    data_workers         : int = 0,
    seed                 : int = 0,
) -> dict[str, Any]:
    """Train and evaluate WISDOM v1 or v2 from one managed DatasetVersion.

    This private implementation keeps the public ``Training.run`` signature declarative while
    carrying out the ordinary PyTorch loop.

    Args:
        work: Active LambdaForge Work providing run-owned metrics and output services.
        dataset: Resolved managed dataset root containing LambdaForge ``index.jsonl``.
        model_version: Supported architecture generation, either ``1`` or ``2``.
        subset: Full data or a deterministic view name such as ``25pct``.
        hidden_dim: Shared atom/surface latent width.
        embedding_dim: Element and optional residue embedding width.
        use_residue_type: Include learned residue category features when true.
        atomic_layers: Relation-aware atomic graph layer count.
        projection_depth: Atom-context/curvature projection MLP depth.
        surface_layers: Surface graph layer count.
        dropout: Dropout probability in ``[0,1)``.
        pooling_type: V2 protein-level pooling hypothesis; v1 requires ``max``.
        topk_fraction: V2 fraction retained by top-k mean pooling.
        attention_hidden_dim: V2 attention score-network hidden width.
        regional_levels: V2 graph-neighbourhood smoothing rounds before regional MAX.
        log_sum_exp_beta: V2 normalized log-sum-exp inverse temperature.
        learning_rate: Positive AdamW learning rate.
        weight_decay: Non-negative AdamW decoupled weight decay.
        batch_size: Positive number of disjoint protein graphs per optimizer step.
        epochs: Positive maximum training epoch count.
        data_workers: Non-negative DataLoader subprocess count per split.
        seed: Reproducible seed injected by LambdaForge for each expanded run.

    Returns:
        Best validation epoch/metrics and final held-out test metrics.

    Raises:
        ValueError: If a model, data view, parameter, or metric contract is invalid.
        OSError: If managed arrays or checkpoint/report artifacts cannot be read or written.
    """
    if model_version not in {1, 2}:
        raise ValueError("model_version must be 1 or 2")
    if model_version == 1 and pooling_type != "max":
        raise ValueError("WISDOM v1 fixes pooling_type='max'")
    if learning_rate <= 0.0 or weight_decay < 0.0:
        raise ValueError("optimizer learning rate must be positive and weight decay non-negative")
    if batch_size < 1 or epochs < 1 or data_workers < 0:
        raise ValueError("batch size/epochs must be positive and data workers non-negative")

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    datasets = {
        split: WisdomDataset(dataset, split, subset=subset)
        for split in ("train", "val", "test")
    }
    generator = torch.Generator().manual_seed(seed)
    loaders = {
        split: DataLoader(
            value,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=data_workers,
            collate_fn=WisdomCollator(),
            generator=generator if split == "train" else None,
            pin_memory=device.type == "cuda",
        )
        for split, value in datasets.items()
    }

    model_parameters = {
        "hidden_dim": hidden_dim,
        "embedding_dim": embedding_dim,
        "use_residue_type": use_residue_type,
        "atomic_layers": atomic_layers,
        "projection_depth": projection_depth,
        "surface_layers": surface_layers,
        "dropout": dropout,
    }
    model: WisdomV1
    if model_version == 1:
        model = WisdomV1(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            use_residue_type=use_residue_type,
            atomic_layers=atomic_layers,
            projection_depth=projection_depth,
            surface_layers=surface_layers,
            dropout=dropout,
        )
    else:
        model = WisdomV2(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            use_residue_type=use_residue_type,
            atomic_layers=atomic_layers,
            projection_depth=projection_depth,
            surface_layers=surface_layers,
            dropout=dropout,
            pooling_type=pooling_type,
            topk_fraction=topk_fraction,
            attention_hidden_dim=attention_hidden_dim,
            regional_levels=regional_levels,
            log_sum_exp_beta=log_sum_exp_beta,
        )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    best_auprc = float("-inf")
    best_epoch = 0
    checkpoint = work.run_dir / "best-model.pt"
    for epoch in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        examples = 0
        for batch in loaders["train"]:
            tensors = _device_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            output = model(**_model_inputs(tensors))
            target = _tensor(tensors, "target")
            loss   = F.binary_cross_entropy_with_logits(output["logits"], target)
            loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
            count     = len(target)
            loss_sum += float(loss.detach()) * count
            examples += count

        train_loss = loss_sum / max(1, examples)
        validation = _evaluate(model, loaders["val"], device)
        work.metrics.log("loss", train_loss, step=epoch, split="train")
        for name, value in validation.items():
            if value is not None:
                work.metrics.log(name, value, step=epoch, split="val")

        objective = validation["auprc"]
        if objective is not None and objective > best_auprc:
            best_auprc = objective
            best_epoch = epoch
            torch.save(
                {
                    "model_version": model_version,
                    "model_parameters": model_parameters,
                    "pooling_type": pooling_type,
                    "state_dict": model.state_dict(),
                    "seed": seed,
                    "epoch": epoch,
                    "val_auprc": objective,
                },
                checkpoint,
            )
    if best_epoch == 0:
        raise RuntimeError("validation AUPRC remained undefined; both classes are required")

    saved = torch.load(checkpoint, map_location=device, weights_only=True)
    model.load_state_dict(saved["state_dict"])
    test_metrics = _evaluate(model, loaders["test"], device)
    for name, value in test_metrics.items():
        if value is not None:
            work.metrics.log(name, value, split="test")

    report = {
        "model_version": model_version,
        "pooling_type": pooling_type,
        "subset": subset,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_auprc": best_auprc,
        "test": test_metrics,
        "split_sizes": {split: len(value) for split, value in datasets.items()},
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
    model : WisdomV1,
    loader: DataLoader[Any],
    device: torch.device,
) -> dict[str, float | None]:
    """Evaluate one explicit split with definition-aware LambdaForge metrics.

    Args:
        model: Trained WISDOM model placed on ``device``.
        loader: Non-empty explicit-split graph DataLoader.
        device: CPU or CUDA device receiving tensors.

    Returns:
        Complete binary metric mapping; mathematically undefined values remain ``None``.
    """
    probabilities: list[Tensor] = []
    targets      : list[Tensor] = []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            tensors = _device_batch(batch, device)
            output  = model(**_model_inputs(tensors))
            probabilities.append(torch.sigmoid(output["logits"]).cpu())
            targets.append(_tensor(tensors, "target").cpu())
    return BinaryMetricSuite().compute(torch.cat(probabilities), torch.cat(targets))


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move tensor values to one training device while preserving identity strings.

    Args:
        batch: Collated WISDOM graph mapping.
        device: Destination CPU or CUDA device.

    Returns:
        New mapping whose tensor values reside on ``device``.
    """
    return {
        name: value.to(device, non_blocking=True) if isinstance(value, Tensor) else value
        for name, value in batch.items()
    }


def _model_inputs(batch: Mapping[str, Any]) -> dict[str, Tensor]:
    """Select the nine graph tensors shared by WISDOM v1 and v2 forward methods.

    Args:
        batch: Device-resident collated graph mapping.

    Returns:
        Keyword mapping accepted by either trainable model generation.
    """
    names = (
        "atomic_numbers",
        "residue_type_ids",
        "atom_edge_index",
        "atom_edge_types",
        "surface_curvatures",
        "surface_edge_index",
        "surface_atom_edge_index",
        "surface_area_weights",
        "surface_batch",
    )
    return {name: _tensor(batch, name) for name in names}


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
