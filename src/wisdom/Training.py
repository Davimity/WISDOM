"""Version-resolved trainable WISDOM Work for LambdaForge 0.14."""

from __future__ import annotations

import time
import json
import math
import torch
import inspect
import importlib
import lambdaforge as lf

from pathlib import Path
from torch import Tensor
from typing import Any, cast
from scipy.stats import spearmanr
from torch.nn import functional as F
from torch.utils.data import DataLoader
from collections.abc import Mapping, Sequence

from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.DiffusionBlock import DiffusionBlock
from wisdom.models.WeakSurfaceLoss import WeakSurfaceLoss
from wisdom.models.ModelWeightAverage import ModelWeightAverage
from wisdom.models.ArchitectureSpike import ArchitectureSpike
from wisdom.models.WeakLossProfile import WeakLossProfile
from wisdom.models.StabilityProfile import StabilityProfile
from wisdom.models.InitializationProfile import InitializationProfile
from wisdom.models.OptimizationProfile import OptimizationProfile
from wisdom.models.WeightAveragingMode import WeightAveragingMode
from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.evaluation.BinaryMetricSuite import BinaryMetricSuite
from wisdom.evaluation.SurfaceMetricSuite import SurfaceMetricSuite
from wisdom.evaluation.SubgroupMetricSuite import SubgroupMetricSuite
from wisdom.evaluation.OptimizationDiagnostics import OptimizationDiagnostics
from wisdom.evaluation.SurfaceFaithfulnessAudit import SurfaceFaithfulnessAudit
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.evaluation.SurfacePredictionReport import SurfacePredictionReport
from wisdom.evaluation.SurfaceVisualizationMode import SurfaceVisualizationMode


_MODEL_INPUT_NAMES = (
    "atomic_numbers",
    "residue_type_ids",
    "atom_role_ids",
    "atom_hybridization_ids",
    "formal_charges",
    "atom_aromaticity",
    "atom_hbond_donor",
    "atom_hbond_acceptor",
    "residue_hydropathy",
    "residue_polarity",
    "atom_edge_index",
    "atom_edge_is_spatial",
    "atom_edge_is_covalent",
    "atom_edge_distance",
    "atom_edge_bond_order",
    "atom_edge_same_residue",
    "atom_edge_same_chain",
    "atom_edge_residue_separation",
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

_VECTOR_INPUT_NAMES = ("atom_positions",)


class Training(lf.Work):
    """Train and evaluate one WISDOM model with framework-owned run services."""

    def run(
        self,
        dataset              : Path,
        model_version        : int = 1,
        subset               : str = "full",
        hidden_dim           : int = 128,
        embedding_dim        : int = 32,
        residue_embedding_dim: int | None = None,
        atomic_layers        : int = 2,
        projection_depth     : int = 1,
        surface_layers       : int = 2,
        atom_spatial_k       : int = 16,
        surface_atom_k       : int = 16,
        diffusion_spectral_modes: int = 128,
        surface_atom_radius  : float = 6.0,
        surface_geometry_transfer: bool = False,
        surface_chunk_size   : int = 8192,
        atomic_message_chunk_size: int = 65536,
        vector_atomic_channels: int = 0,
        interaction_round     : str = "single",
        atomic_edge_distance_scale: float = 6.0,
        dropout              : float = 0.2,
        surface_encoder_type : str = "diffusion",
        surface_patch_size   : int = 64,
        pooling_type         : str = "max",
        topk_fraction        : float = 0.05,
        attention_hidden_dim : int = 32,
        regional_diffusion_scale: float = 2.5,
        log_sum_exp_beta     : float = 5.0,
        head_type            : str = "single",
        global_context_dim   : int = 16,
        detach_global_context: bool = True,
        direct_head_weight   : float = 1.0,
        surface_head_weight  : float = 1.0,
        learning_rate        : float = 3.0e-4,
        weight_decay         : float = 1.0e-4,
        gate_lambda          : float = 1.0e-3,
        architecture_spike            : str   = "custom",
        weak_loss_profile              : str   = "custom",
        initialization_profile        : str   = "custom",
        optimization_profile          : str   = "custom",
        stability_profile             : str | None = None,
        embedding_initialization       : str   = "current",
        gate_initial_active            : float = 0.95,
        gate_warmup_fraction           : float = 0.0,
        gate_ramp_fraction             : float = 0.0,
        diffusion_time_initialization  : str   = "current",
        diffusion_residual_initialization: str = "current",
        atomic_residual_initialization : str   = "current",
        neutral_edge_initialization    : bool  = False,
        neutral_transfer_initialization: bool  = False,
        physics_distance_initialization: bool  = False,
        local_head_weight_std          : float | None = None,
        calibrate_local_head_bias      : bool  = False,
        calibration_batches           : int   = 4,
        linear_initialization          : str   = "current",
        learning_rate_warmup_fraction  : float = 0.0,
        gradient_clip_norm             : float | None = None,
        optimization_diagnostics       : bool  = False,
        weight_averaging               : str   = "none",
        ema_decay                      : float = 0.99,
        swa_start_fraction             : float = 0.80,
        pooling_curriculum_end_beta    : float | None = None,
        pooling_curriculum_hold_fraction: float = 0.30,
        negative_surface_lambda      : float = 0.0,
        positive_existence_lambda    : float = 0.0,
        regional_positive_lambda     : float = 0.0,
        regional_loss_length         : float = 2.5,
        regional_ranking_lambda      : float = 0.0,
        regional_ranking_margin      : float = 0.5,
        cardinality_lambda           : float = 0.0,
        minimum_positive_area        : float = 0.0,
        maximum_positive_area        : float | None = None,
        total_variation_lambda       : float = 0.0,
        dirichlet_lambda             : float = 0.0,
        batch_size                    : int   = 2,
        epochs                        : int   = 100,
        patience                      : int   = 30,
        minimum_delta                 : float = 1.0e-3,
        precision                     : str   = "auto",
        surface_metrics               : bool  = True,
        surface_metrics_interval      : int   = 0,
        surface_visualization         : str   = "viewer",
        surface_prediction_threshold  : float = 0.5,
        surface_visualization_maximum : int   = 12,
        faithfulness_audit            : bool  = False,
        evaluate_test                 : bool  = True,
        data_workers                  : int   = 4,
        initialization_seed           : int | None = None,
        training_seed                 : int | None = None,
        save_initial_checkpoint       : bool = False,
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
            residue_embedding_dim: Independent residue embedding width; ``None`` reuses
                ``embedding_dim`` for backward compatibility.
            atomic_layers: Relation-aware atomic graph layer count.
            projection_depth: Atom-context/curvature projection MLP depth.
            surface_layers: Surface-encoder block count.
            atom_spatial_k: Runtime atomic spatial-neighbor rank budget.
            surface_atom_k: Runtime nearest-atom budget per surface point.
            diffusion_spectral_modes: Runtime low-frequency mode budget per protein.
            surface_atom_radius: Physical transfer cutoff used for feature normalization.
            surface_geometry_transfer: Let the existing gated surface geometry interact with each
                candidate atom when computing atom-to-surface attention scores.
            surface_chunk_size: Maximum surface points in one atom-transfer chunk.
            atomic_message_chunk_size: Maximum atomic messages materialized per RGCN chunk.
            vector_atomic_channels: Equivariant vector-state channels; zero disables spike C.
            interaction_round: ``single``, added-depth control, or bidirectional fusion spike B.
            atomic_edge_distance_scale: Atomic distance normalization scale in ångströms.
            dropout: Dropout probability in ``[0,1)``.
            surface_encoder_type: V3 surface-propagation hypothesis; v1 requires ``diffusion``.
            surface_patch_size: V3 serialized-attention patch bound.
            pooling_type: V2 pooling hypothesis; v1 requires ``max``.
            topk_fraction: V2 fraction retained by top-k mean pooling.
            attention_hidden_dim: V2 attention score-network hidden width.
            regional_diffusion_scale: V2 physical smoothing length before regional MAX, in Å.
            log_sum_exp_beta: V2 normalized log-sum-exp inverse temperature.
            head_type: V2+ ``single``, ``dual``, ``global_context``, or ``film`` hypothesis.
            global_context_dim: D2/D2b global context bottleneck width.
            detach_global_context: Stop local gradients through the global context path.
            direct_head_weight: BCE weight for the direct protein classifier.
            surface_head_weight: BCE weight for the surface-derived protein classifier.
            learning_rate: Positive AdamW learning rate.
            weight_decay: Non-negative AdamW decoupled weight decay.
            gate_lambda: Non-negative multiplier for normalized expected L0 gate cost.
            architecture_spike: Named single-factor point-conditioning, feedback, depth-control,
                or vector-state hypothesis. ``custom`` preserves the detailed model arguments.
            weak_loss_profile: Named sequential weak-supervision candidate. ``custom`` preserves
                the detailed weak-loss coefficients.
            initialization_profile: Named model-initialization candidate. It changes no optimizer
                behavior, so its winner can remain active during the optimizer screen.
            optimization_profile: Named optimizer-stability candidate. It changes no model
                initialization, so it composes with ``initialization_profile``.
            stability_profile: Deprecated combined profile accepted for archived YAMLs. It cannot
                be combined with either new named profile.
            embedding_initialization: Categorical embedding scale policy.
            gate_initial_active: Initial Hard-Concrete activity probability.
            gate_warmup_fraction: Initial training fraction with all semantic gates forced on and
                no gate penalty.
            gate_ramp_fraction: Following fraction over which learned gates activate and their L0
                weight increases linearly to ``gate_lambda``.
            diffusion_time_initialization: Initial DiffusionNet physical time schedule.
            diffusion_residual_initialization: Residual scaling policy for DiffusionNet blocks.
            atomic_residual_initialization: Residual scaling policy for atomic graph blocks.
            neutral_edge_initialization: Start edge-attribute residuals at zero.
            neutral_transfer_initialization: Start orientation/content transfer residuals at zero.
            physics_distance_initialization: Start distance attention from trainable ``-d/R``.
            local_head_weight_std: Optional small-normal scale for local-head weights.
            calibrate_local_head_bias: Match initial predicted prevalence to the training labels.
            calibration_batches: Maximum representative train batches used by bias calibration.
            linear_initialization: Explicit dense-layer initialization policy.
            learning_rate_warmup_fraction: Fraction of epochs used for linear LR warm-up.
            gradient_clip_norm: Optional global gradient-norm ceiling; ``None`` disables clipping.
            optimization_diagnostics: Record component gradient norms, activation distributions,
                and sampled gate saturation once per optimization step and aggregate by epoch.
            weight_averaging: ``none``, ``ema``, or ``swa`` evaluation/checkpoint weights.
            ema_decay: EMA retention coefficient in ``[0,1)``.
            swa_start_fraction: Training fraction after which optimizer states enter SWA.
            pooling_curriculum_end_beta: Optional final LogSumExp beta. ``None`` keeps pooling
                fixed; a value activates the plan's smooth-to-sharp MIL curriculum from beta one.
            pooling_curriculum_hold_fraction: Initial training fraction held at beta one before a
                linear rise to the final value.
            negative_surface_lambda: Weight of the weak negative-bag surface loss. Zero preserves
                the global-BCE baseline. A positive value penalizes area-weighted local positive
                evidence only on globally negative training proteins and never reads surface GT.
            positive_existence_lambda: Weight requiring at least one positive local logit in each
                globally positive training protein.
            regional_positive_lambda: Weight requiring positive evidence after fixed intrinsic
                heat diffusion over the surface.
            regional_loss_length: Physical heat-diffusion length in angstroms for regional losses.
            regional_ranking_lambda: Weight ranking positive regional evidence above negative
                regional evidence within each mixed-class batch.
            regional_ranking_margin: Required regional positive-versus-negative logit margin.
            cardinality_lambda: Weight of broad positive surface-area mass constraints.
            minimum_positive_area: Lower positive probability-area fraction for positive bags.
            maximum_positive_area: Optional upper positive probability-area fraction. ``None``
                disables the upper constraint.
            total_variation_lambda: Weight of local probability variation over stored surface
                neighbours.
            dirichlet_lambda: Weight of normalized intrinsic spectral energy. It is an alternative
                to total variation and cannot be enabled simultaneously.
            batch_size: Positive number of disjoint protein graphs per optimizer step.
            epochs: Positive maximum training epoch count.
            patience: Validation epochs without a meaningful protein-score improvement before
                stopping.
            minimum_delta: Minimum absolute protein-score increase that resets ``patience``.
            precision: ``auto``, ``float32``, ``bfloat16``, or ``float16`` activation policy.
            surface_metrics: Report evaluation-only local metrics on validation and final test
                without exposing them to loss or checkpoint selection. Validation surface metrics
                contribute to architecture-level HPO only through ``wisdom_hpo_score``.
            surface_metrics_interval: Validation-epoch interval for local metrics. Zero evaluates
                them only once after restoring the best checkpoint; a positive value also evaluates
                them every N epochs. The final held-out test evaluation remains unchanged.
            surface_visualization: Final artifact level: ``none`` writes no local map, ``viewer``
                writes compact HTML/PLY reports, and ``full`` additionally writes point-aligned NPZ
                prediction data. This setting never changes metrics, loss, or checkpoint choice.
            surface_prediction_threshold: Initial cutoff in ``[0,1]`` for stored and interactive
                hard surface predictions. HTML viewers can change it without rerunning the model.
            surface_visualization_maximum: Maximum class-balanced HTML/PLY proteins per evaluated
                split. Zero renders all. Prediction NPZ files cover every protein only when
                ``surface_visualization`` is ``full``.
            faithfulness_audit: On the restored best checkpoint, measure prediction changes after
                equal-area top/random/bottom surface deletion and insertion without using local GT.
            evaluate_test: Evaluate the held-out test split once after selecting the validation
                checkpoint. Experimental screening may disable it to prevent repeated test access.
            data_workers: Persistent training-loader subprocesses used to overlap NPZ decoding
                with GPU work. Validation and test use half this count, rounded down to one.
            initialization_seed: Optional model-parameter seed. ``None`` reuses LambdaForge's run
                seed. Fix it while varying run seeds to isolate post-initialization stochasticity.
            training_seed: Optional shuffle/dropout/gate seed. ``None`` reuses LambdaForge's run
                seed. Fix it while varying run seeds to isolate initialization variability.
            save_initial_checkpoint: Publish the epoch-zero model state for seed-factorization
                audits. It is never used automatically as a trained checkpoint.

        Returns:
            Best validation epoch/metrics, stopping reason, optional surface-report summary, and
            held-out test metrics. Adaptively pruned Runs return ``test=None`` and no final
            surface report because they are excluded from HPO ranking.

        Raises:
            ValueError: If a model, data view, parameter, or metric contract is invalid.
            RuntimeError: If the protein-level validation score is undefined for every epoch.
            OSError: If managed arrays or checkpoint/report artifacts cannot be accessed.
        """
        return _train_wisdom(
            self,
            dataset=dataset,
            model_version=model_version,
            subset=subset,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            residue_embedding_dim=residue_embedding_dim,
            atomic_layers=atomic_layers,
            projection_depth=projection_depth,
            surface_layers=surface_layers,
            atom_spatial_k=atom_spatial_k,
            surface_atom_k=surface_atom_k,
            diffusion_spectral_modes=diffusion_spectral_modes,
            surface_atom_radius=surface_atom_radius,
            surface_geometry_transfer=surface_geometry_transfer,
            surface_chunk_size=surface_chunk_size,
            atomic_message_chunk_size=atomic_message_chunk_size,
            vector_atomic_channels=vector_atomic_channels,
            interaction_round=interaction_round,
            atomic_edge_distance_scale=atomic_edge_distance_scale,
            dropout=dropout,
            surface_encoder_type=surface_encoder_type,
            surface_patch_size=surface_patch_size,
            pooling_type=pooling_type,
            topk_fraction=topk_fraction,
            attention_hidden_dim=attention_hidden_dim,
            regional_diffusion_scale=regional_diffusion_scale,
            log_sum_exp_beta=log_sum_exp_beta,
            head_type=head_type,
            global_context_dim=global_context_dim,
            detach_global_context=detach_global_context,
            direct_head_weight=direct_head_weight,
            surface_head_weight=surface_head_weight,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            gate_lambda=gate_lambda,
            architecture_spike=architecture_spike,
            weak_loss_profile=weak_loss_profile,
            initialization_profile=initialization_profile,
            optimization_profile=optimization_profile,
            stability_profile=stability_profile,
            embedding_initialization=embedding_initialization,
            gate_initial_active=gate_initial_active,
            gate_warmup_fraction=gate_warmup_fraction,
            gate_ramp_fraction=gate_ramp_fraction,
            diffusion_time_initialization=diffusion_time_initialization,
            diffusion_residual_initialization=diffusion_residual_initialization,
            atomic_residual_initialization=atomic_residual_initialization,
            neutral_edge_initialization=neutral_edge_initialization,
            neutral_transfer_initialization=neutral_transfer_initialization,
            physics_distance_initialization=physics_distance_initialization,
            local_head_weight_std=local_head_weight_std,
            calibrate_local_head_bias=calibrate_local_head_bias,
            calibration_batches=calibration_batches,
            linear_initialization=linear_initialization,
            learning_rate_warmup_fraction=learning_rate_warmup_fraction,
            gradient_clip_norm=gradient_clip_norm,
            optimization_diagnostics=optimization_diagnostics,
            weight_averaging=weight_averaging,
            ema_decay=ema_decay,
            swa_start_fraction=swa_start_fraction,
            pooling_curriculum_end_beta=pooling_curriculum_end_beta,
            pooling_curriculum_hold_fraction=pooling_curriculum_hold_fraction,
            negative_surface_lambda=negative_surface_lambda,
            positive_existence_lambda=positive_existence_lambda,
            regional_positive_lambda=regional_positive_lambda,
            regional_loss_length=regional_loss_length,
            regional_ranking_lambda=regional_ranking_lambda,
            regional_ranking_margin=regional_ranking_margin,
            cardinality_lambda=cardinality_lambda,
            minimum_positive_area=minimum_positive_area,
            maximum_positive_area=maximum_positive_area,
            total_variation_lambda=total_variation_lambda,
            dirichlet_lambda=dirichlet_lambda,
            batch_size=batch_size,
            epochs=epochs,
            patience=patience,
            minimum_delta=minimum_delta,
            precision=precision,
            surface_metrics=surface_metrics,
            surface_metrics_interval=surface_metrics_interval,
            surface_visualization=surface_visualization,
            surface_prediction_threshold=surface_prediction_threshold,
            surface_visualization_maximum=surface_visualization_maximum,
            faithfulness_audit=faithfulness_audit,
            evaluate_test=evaluate_test,
            data_workers=data_workers,
            initialization_seed=initialization_seed,
            training_seed=training_seed,
            save_initial_checkpoint=save_initial_checkpoint,
            seed=self.seed or 0,
        )


def _train_wisdom(
    work                 : Training,
    dataset              : Path,
    model_version        : int = 1,
    subset               : str = "full",
    hidden_dim           : int = 128,
    embedding_dim        : int = 32,
    residue_embedding_dim: int | None = None,
    atomic_layers        : int = 2,
    projection_depth     : int = 1,
    surface_layers       : int = 2,
    atom_spatial_k       : int = 16,
    surface_atom_k       : int = 16,
    diffusion_spectral_modes: int = 128,
    surface_atom_radius  : float = 6.0,
    surface_geometry_transfer: bool = False,
    surface_chunk_size   : int = 8192,
    atomic_message_chunk_size: int = 65536,
    vector_atomic_channels: int = 0,
    interaction_round     : str = "single",
    atomic_edge_distance_scale: float = 6.0,
    dropout              : float = 0.2,
    surface_encoder_type : str = "diffusion",
    surface_patch_size   : int = 64,
    pooling_type         : str = "max",
    topk_fraction        : float = 0.05,
    attention_hidden_dim : int = 32,
    regional_diffusion_scale: float = 2.5,
    log_sum_exp_beta     : float = 5.0,
    head_type            : str = "single",
    global_context_dim   : int = 16,
    detach_global_context: bool = True,
    direct_head_weight   : float = 1.0,
    surface_head_weight  : float = 1.0,
    learning_rate        : float = 3.0e-4,
    weight_decay         : float = 1.0e-4,
    gate_lambda          : float = 1.0e-3,
    architecture_spike            : str   = "custom",
    weak_loss_profile              : str   = "custom",
    initialization_profile        : str   = "custom",
    optimization_profile          : str   = "custom",
    stability_profile             : str | None = None,
    embedding_initialization       : str   = "current",
    gate_initial_active            : float = 0.95,
    gate_warmup_fraction           : float = 0.0,
    gate_ramp_fraction             : float = 0.0,
    diffusion_time_initialization  : str   = "current",
    diffusion_residual_initialization: str = "current",
    atomic_residual_initialization : str   = "current",
    neutral_edge_initialization    : bool  = False,
    neutral_transfer_initialization: bool  = False,
    physics_distance_initialization: bool  = False,
    local_head_weight_std          : float | None = None,
    calibrate_local_head_bias      : bool  = False,
    calibration_batches           : int   = 4,
    linear_initialization          : str   = "current",
    learning_rate_warmup_fraction  : float = 0.0,
    gradient_clip_norm             : float | None = None,
    optimization_diagnostics       : bool  = False,
    weight_averaging               : str   = "none",
    ema_decay                      : float = 0.99,
    swa_start_fraction             : float = 0.80,
    pooling_curriculum_end_beta    : float | None = None,
    pooling_curriculum_hold_fraction: float = 0.30,
    negative_surface_lambda      : float = 0.0,
    positive_existence_lambda    : float = 0.0,
    regional_positive_lambda     : float = 0.0,
    regional_loss_length         : float = 2.5,
    regional_ranking_lambda      : float = 0.0,
    regional_ranking_margin      : float = 0.5,
    cardinality_lambda           : float = 0.0,
    minimum_positive_area        : float = 0.0,
    maximum_positive_area        : float | None = None,
    total_variation_lambda       : float = 0.0,
    dirichlet_lambda             : float = 0.0,
    batch_size                    : int   = 2,
    epochs                        : int   = 100,
    patience                      : int   = 30,
    minimum_delta                 : float = 1.0e-3,
    precision                     : str   = "auto",
    surface_metrics               : bool  = True,
    surface_metrics_interval      : int   = 0,
    surface_visualization         : str   = "viewer",
    surface_prediction_threshold  : float = 0.5,
    surface_visualization_maximum : int   = 12,
    faithfulness_audit            : bool  = False,
    evaluate_test                 : bool  = True,
    data_workers                  : int   = 4,
    initialization_seed           : int | None = None,
    training_seed                 : int | None = None,
    save_initial_checkpoint       : bool = False,
    seed                          : int   = 0,
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
        residue_embedding_dim: Independent residue embedding width or ``None`` for the legacy
            shared width.
        atomic_layers: Relation-aware atomic graph layer count.
        projection_depth: Atom-context/curvature projection MLP depth.
        surface_layers: Surface-encoder block count.
        atom_spatial_k: Runtime atomic spatial-neighbor rank budget.
        surface_atom_k: Runtime nearest-atom budget per surface point.
        diffusion_spectral_modes: Runtime low-frequency mode budget per protein.
        surface_atom_radius: Transfer geometry normalization radius in Å.
        surface_geometry_transfer: Whether existing gated surface geometry conditions relative
            atom-to-surface candidate scores.
        surface_chunk_size: Maximum transfer points per activation chunk.
        atomic_message_chunk_size: Maximum RGCN messages per chunk.
        vector_atomic_channels: Equivariant atom-vector channel count; zero keeps scalar atoms.
        interaction_round: Single fusion, equal-depth control, or bidirectional feedback round.
        atomic_edge_distance_scale: Atomic distance normalization scale in ångströms.
        dropout: Dropout probability in ``[0,1)``.
        surface_encoder_type: V3 controlled surface encoder name.
        surface_patch_size: V3 serialized-attention patch bound.
        pooling_type: V2 protein-level pooling hypothesis; v1 requires ``max``.
        topk_fraction: V2 fraction retained by top-k mean pooling.
        attention_hidden_dim: V2 attention score-network hidden width.
        regional_diffusion_scale: V2 regional diffusion length in Å.
        log_sum_exp_beta: V2 normalized log-sum-exp inverse temperature.
        head_type: V2+ global/local head relationship.
        global_context_dim: D2/D2b context bottleneck width.
        detach_global_context: Whether local gradients stop at global context.
        direct_head_weight: Direct-head protein BCE multiplier.
        surface_head_weight: Surface-derived protein BCE multiplier.
        learning_rate: Positive AdamW learning rate.
        weight_decay: Non-negative AdamW decoupled weight decay.
        gate_lambda: Non-negative multiplier for the normalized expected L0 gate cost.
        architecture_spike: Named one-factor architecture candidate or ``custom``.
        weak_loss_profile: Named sequential weak-loss candidate or ``custom``.
        initialization_profile: Named initialization candidate or ``custom`` for explicit values.
        optimization_profile: Named optimizer candidate or ``custom`` for explicit values.
        stability_profile: Deprecated combined profile. It cannot be mixed with the new profiles.
        embedding_initialization: Categorical embedding scale policy.
        gate_initial_active: Initial Hard-Concrete activity probability.
        gate_warmup_fraction: Initial all-on/no-L0 fraction of training.
        gate_ramp_fraction: Subsequent learned-gate linear L0 ramp fraction.
        diffusion_time_initialization: Initial DiffusionNet physical time schedule.
        diffusion_residual_initialization: Diffusion residual scaling policy.
        atomic_residual_initialization: Atomic graph residual scaling policy.
        neutral_edge_initialization: Start edge-conditioner corrections at zero.
        neutral_transfer_initialization: Start orientation/content transfer corrections at zero.
        physics_distance_initialization: Start transfer scores from a trainable ``-d/R`` prior.
        local_head_weight_std: Optional small-normal scale for local-head weights.
        calibrate_local_head_bias: Match initial protein prevalence on representative train batches.
        calibration_batches: Maximum train batches used for that one-time calibration.
        linear_initialization: Dense-layer initialization policy.
        learning_rate_warmup_fraction: Fraction of epochs used for linear LR warm-up.
        gradient_clip_norm: Optional global gradient-norm ceiling.
        optimization_diagnostics: Whether to aggregate component gradients, activations, and gate
            boundary fractions for stability diagnosis.
        weight_averaging: ``none``, ``ema``, or ``swa`` validation/checkpoint policy.
        ema_decay: EMA retention coefficient.
        swa_start_fraction: Fraction after which model states contribute to SWA.
        pooling_curriculum_end_beta: Optional final beta for smooth-to-sharp LogSumExp pooling.
        pooling_curriculum_hold_fraction: Initial fraction held at beta one.
        negative_surface_lambda: Non-negative multiplier for negative-bag surface supervision.
            The loss is the mean, over globally negative proteins, of area-weighted
            ``softplus(surface_logit)`` and uses no point annotation.
        positive_existence_lambda: Multiplier requiring a positive point in positive protein bags.
        regional_positive_lambda: Multiplier requiring a positive spectrally diffused region.
        regional_loss_length: Intrinsic heat-diffusion length in angstroms for regional losses.
        regional_ranking_lambda: Multiplier ranking positive regional maxima above negatives.
        regional_ranking_margin: Required positive-versus-negative regional logit margin.
        cardinality_lambda: Multiplier for broad positive probability-area constraints.
        minimum_positive_area: Lower positive probability-area fraction for positive proteins.
        maximum_positive_area: Optional upper positive probability-area fraction.
        total_variation_lambda: Multiplier for edgewise surface probability variation.
        dirichlet_lambda: Multiplier for normalized intrinsic spectral energy; mutually exclusive
            with total variation.
        batch_size: Positive number of disjoint protein graphs per optimizer step.
        epochs: Positive maximum training epoch count.
        patience: Validation epochs without a meaningful protein-score improvement before stopping.
        minimum_delta: Minimum absolute protein-score increase that resets ``patience``.
        precision: ``auto``, ``float32``, ``bfloat16``, or ``float16`` activation policy.
        surface_metrics: Report evaluation-only local metrics on validation and final test without
            using them for the loss or checkpoint selection. Validation surface metrics enter only
            the architecture-level HPO score.
        surface_metrics_interval: Validation-epoch interval for local metrics. Zero evaluates them
            only on the restored best checkpoint; a positive integer additionally evaluates every
            N epochs. Ignored when ``surface_metrics`` is false.
        surface_visualization: ``none``, ``viewer``, or ``full`` final surface artifact level.
            Viewer mode emits HTML/PLY only; full mode also emits point-aligned prediction NPZ.
        surface_prediction_threshold: Initial probability cutoff in ``[0,1]`` for hard maps.
        surface_visualization_maximum: Maximum balanced HTML/PLY sample per evaluated split. Zero
            renders all proteins. Numerical NPZ predictions retain full split coverage only in
            ``full`` visualization mode.
        faithfulness_audit: Whether final best-checkpoint evaluation performs V8 surface deletion
            and insertion controls without accessing local ground truth.
        evaluate_test: Evaluate held-out test exactly once after validation selection. False keeps
            HPO and screening completely isolated from test members.
        data_workers: Persistent training-loader subprocesses used to overlap NPZ decoding with
            GPU work. Validation and test use half this count, rounded down to one.
        initialization_seed: Optional parameter-initialization seed or ``None`` for ``seed``.
        training_seed: Optional data-order/dropout/gate seed or ``None`` for ``seed``.
        save_initial_checkpoint: Whether to publish an epoch-zero parameter snapshot.
        seed: Reproducible seed injected by LambdaForge for each expanded run.

    Returns:
        Best validation epoch/metrics, stopping reason, optional surface-report summary, and
        held-out test metrics. Adaptively pruned Runs return ``test=None`` and no final surface
        report because they are excluded from HPO ranking.

    Raises:
        ValueError: If a model, data view, parameter, or metric contract is invalid.
        RuntimeError: If the protein-level validation score is undefined for every epoch.
        OSError: If managed arrays or checkpoint/report artifacts cannot be read or written.
    """
    architecture = ArchitectureSpike(architecture_spike)
    architecture_overrides = architecture.overrides()
    surface_geometry_transfer = bool(
        architecture_overrides.get("surface_geometry_transfer", surface_geometry_transfer)
    )
    interaction_round = str(
        architecture_overrides.get("interaction_round", interaction_round)
    )
    vector_atomic_channels = cast(
        int,
        architecture_overrides.get("vector_atomic_channels", vector_atomic_channels),
    )

    loss_profile = WeakLossProfile(weak_loss_profile)
    loss_overrides = loss_profile.overrides()
    negative_surface_lambda = cast(
        float,
        loss_overrides.get("negative_surface_lambda", negative_surface_lambda),
    )
    positive_existence_lambda = cast(
        float,
        loss_overrides.get("positive_existence_lambda", positive_existence_lambda),
    )
    regional_positive_lambda = cast(
        float,
        loss_overrides.get("regional_positive_lambda", regional_positive_lambda),
    )
    regional_ranking_lambda = cast(
        float,
        loss_overrides.get("regional_ranking_lambda", regional_ranking_lambda),
    )
    cardinality_lambda = cast(
        float,
        loss_overrides.get("cardinality_lambda", cardinality_lambda),
    )
    minimum_positive_area = cast(
        float,
        loss_overrides.get("minimum_positive_area", minimum_positive_area),
    )
    maximum_positive_area = loss_overrides.get(
        "maximum_positive_area",
        maximum_positive_area,
    )
    total_variation_lambda = cast(
        float,
        loss_overrides.get("total_variation_lambda", total_variation_lambda),
    )
    dirichlet_lambda = cast(
        float,
        loss_overrides.get("dirichlet_lambda", dirichlet_lambda),
    )

    # Initialization and optimization are independent scientific decisions. The legacy selector
    # remains reproducible, but mixing it with either new selector would make precedence ambiguous.

    initialization = InitializationProfile(initialization_profile)
    optimization   = OptimizationProfile(optimization_profile)
    legacy_profile = StabilityProfile(stability_profile or "custom")

    if legacy_profile is not StabilityProfile.CUSTOM and (
        initialization is not InitializationProfile.CUSTOM
        or optimization is not OptimizationProfile.CUSTOM
    ):
        raise ValueError(
            "legacy stability_profile cannot be combined with initialization_profile or "
            "optimization_profile"
        )

    profile_overrides = {
        **legacy_profile.overrides(),
        **initialization.overrides(),
        **optimization.overrides(),
    }
    embedding_initialization = str(
        profile_overrides.get("embedding_initialization", embedding_initialization)
    )
    gate_initial_active = cast(
        float,
        profile_overrides.get("gate_initial_active", gate_initial_active),
    )
    gate_warmup_fraction = cast(
        float,
        profile_overrides.get("gate_warmup_fraction", gate_warmup_fraction),
    )
    gate_ramp_fraction = cast(
        float,
        profile_overrides.get("gate_ramp_fraction", gate_ramp_fraction),
    )
    diffusion_time_initialization = str(
        profile_overrides.get(
            "diffusion_time_initialization",
            diffusion_time_initialization,
        )
    )
    diffusion_residual_initialization = str(
        profile_overrides.get(
            "diffusion_residual_initialization",
            diffusion_residual_initialization,
        )
    )
    atomic_residual_initialization = str(
        profile_overrides.get(
            "atomic_residual_initialization",
            atomic_residual_initialization,
        )
    )
    neutral_edge_initialization = bool(
        profile_overrides.get("neutral_edge_initialization", neutral_edge_initialization)
    )
    neutral_transfer_initialization = bool(
        profile_overrides.get(
            "neutral_transfer_initialization",
            neutral_transfer_initialization,
        )
    )
    local_head_weight_std = cast(
        float | None,
        profile_overrides.get("local_head_weight_std", local_head_weight_std),
    )
    calibrate_local_head_bias = bool(
        profile_overrides.get("calibrate_local_head_bias", calibrate_local_head_bias)
    )
    linear_initialization = str(
        profile_overrides.get("linear_initialization", linear_initialization)
    )
    learning_rate_warmup_fraction = cast(
        float,
        profile_overrides.get(
            "learning_rate_warmup_fraction",
            learning_rate_warmup_fraction,
        ),
    )
    gradient_clip_norm = cast(
        float | None,
        profile_overrides.get("gradient_clip_norm", gradient_clip_norm),
    )
    weight_averaging = str(profile_overrides.get("weight_averaging", weight_averaging))
    ema_decay = cast(float, profile_overrides.get("ema_decay", ema_decay))
    swa_start_fraction = cast(
        float,
        profile_overrides.get("swa_start_fraction", swa_start_fraction),
    )

    if isinstance(model_version, bool) or model_version < 1:
        raise ValueError("model_version must be a positive integer")
    if model_version == 1 and pooling_type != "max":
        raise ValueError("WISDOM v1 fixes pooling_type='max'")
    if model_version == 1 and (interaction_round != "single" or vector_atomic_channels != 0):
        raise ValueError("WISDOM v1 fixes single interaction and scalar atomic states")
    if model_version == 1 and head_type != "single":
        raise ValueError("WISDOM v1 fixes head_type='single'")
    if model_version == 1 and surface_encoder_type != "diffusion":
        raise ValueError("WISDOM v1 fixes surface_encoder_type='diffusion'")
    if pooling_curriculum_end_beta is not None and (
        pooling_curriculum_end_beta <= 0.0
        or model_version < 2
        or pooling_type != "log_sum_exp"
    ):
        raise ValueError("pooling beta curriculum requires v2+ LogSumExp and a positive final beta")
    if (
        learning_rate <= 0.0
        or weight_decay < 0.0
        or gate_lambda < 0.0
        or negative_surface_lambda < 0.0
    ):
        raise ValueError("optimizer rate, weight decay, or loss regularization is invalid")
    if batch_size < 1 or epochs < 1 or data_workers < 0:
        raise ValueError("batch size/epochs must be positive and data workers non-negative")
    if (
        isinstance(surface_metrics_interval, bool)
        or not isinstance(surface_metrics_interval, int)
        or surface_metrics_interval < 0
    ):
        raise ValueError("surface_metrics_interval must be a non-negative integer")
    if not 0.0 <= surface_prediction_threshold <= 1.0:
        raise ValueError("surface_prediction_threshold must lie in [0,1]")
    if surface_visualization_maximum < 0:
        raise ValueError("surface_visualization_maximum cannot be negative")
    if patience < 1 or minimum_delta < 0.0:
        raise ValueError("patience must be positive and minimum_delta must be non-negative")
    if precision not in {"auto", "bfloat16", "float16", "float32"}:
        raise ValueError("precision must be auto, bfloat16, float16, or float32")
    if atom_spatial_k < 1 or surface_atom_k < 1 or diffusion_spectral_modes < 1:
        raise ValueError("atom K, surface-atom J, and spectral-mode budget must be positive")
    if vector_atomic_channels < 0:
        raise ValueError("vector_atomic_channels cannot be negative")
    if global_context_dim < 1 or direct_head_weight < 0.0 or surface_head_weight < 0.0:
        raise ValueError("global context width and head-loss weights are invalid")
    if head_type != "single" and direct_head_weight == 0.0 and surface_head_weight == 0.0:
        raise ValueError("a dual-head model requires at least one nonzero protein loss")
    if not 0.0 < gate_initial_active < 1.0:
        raise ValueError("gate_initial_active must lie strictly between zero and one")
    if (
        not 0.0 <= gate_warmup_fraction <= 1.0
        or not 0.0 <= gate_ramp_fraction <= 1.0
        or gate_warmup_fraction + gate_ramp_fraction > 1.0
    ):
        raise ValueError(
            "gate warm-up and ramp fractions must be non-negative and sum to at most one"
        )
    if not 0.0 <= learning_rate_warmup_fraction <= 1.0:
        raise ValueError("learning_rate_warmup_fraction must lie in [0,1]")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0.0:
        raise ValueError("gradient_clip_norm must be positive when supplied")
    if calibration_batches < 1:
        raise ValueError("calibration_batches must be positive")
    averaging_mode = WeightAveragingMode(weight_averaging)
    if not 0.0 <= ema_decay < 1.0 or not 0.0 <= swa_start_fraction < 1.0:
        raise ValueError("EMA decay and SWA start fraction are invalid")
    if not 0.0 <= pooling_curriculum_hold_fraction < 1.0:
        raise ValueError("pooling_curriculum_hold_fraction must lie in [0,1)")

    # Weak priors use only global labels and immutable geometry. Constructing the cohesive loss
    # object before data loading also determines whether the stored neighbour topology is needed.

    weak_surface_loss = WeakSurfaceLoss(
        negative_weight=negative_surface_lambda,
        positive_existence_weight=positive_existence_lambda,
        regional_positive_weight=regional_positive_lambda,
        regional_length=regional_loss_length,
        regional_ranking_weight=regional_ranking_lambda,
        ranking_margin=regional_ranking_margin,
        cardinality_weight=cardinality_lambda,
        minimum_positive_area=minimum_positive_area,
        maximum_positive_area=maximum_positive_area,
        total_variation_weight=total_variation_lambda,
        dirichlet_weight=dirichlet_lambda,
    )
    weak_surface_configuration = {
        "weak_loss_profile":           loss_profile.value,
        "negative_surface_lambda":     negative_surface_lambda,
        "positive_existence_lambda":   positive_existence_lambda,
        "regional_positive_lambda":    regional_positive_lambda,
        "regional_loss_length":        regional_loss_length,
        "regional_ranking_lambda":     regional_ranking_lambda,
        "regional_ranking_margin":     regional_ranking_margin,
        "cardinality_lambda":          cardinality_lambda,
        "minimum_positive_area":       minimum_positive_area,
        "maximum_positive_area":       maximum_positive_area,
        "total_variation_lambda":      total_variation_lambda,
        "dirichlet_lambda":            dirichlet_lambda,
    }
    initialization_configuration = {
        "architecture_spike":               architecture.value,
        "initialization_profile":           initialization.value,
        "optimization_profile":             optimization.value,
        "stability_profile":                legacy_profile.value,
        "embedding_initialization":         embedding_initialization,
        "gate_initial_active":              gate_initial_active,
        "gate_warmup_fraction":             gate_warmup_fraction,
        "gate_ramp_fraction":               gate_ramp_fraction,
        "diffusion_time_initialization":    diffusion_time_initialization,
        "diffusion_residual_initialization": diffusion_residual_initialization,
        "atomic_residual_initialization":   atomic_residual_initialization,
        "neutral_edge_initialization":      neutral_edge_initialization,
        "neutral_transfer_initialization":  neutral_transfer_initialization,
        "physics_distance_initialization":  physics_distance_initialization,
        "local_head_weight_std":            local_head_weight_std,
        "calibrate_local_head_bias":        calibrate_local_head_bias,
        "calibration_batches":              calibration_batches,
        "linear_initialization":            linear_initialization,
        "learning_rate_warmup_fraction":    learning_rate_warmup_fraction,
        "gradient_clip_norm":                gradient_clip_norm,
        "optimization_diagnostics":          optimization_diagnostics,
        "weight_averaging":                  averaging_mode.value,
        "ema_decay":                         ema_decay,
        "swa_start_fraction":                 swa_start_fraction,
        "pooling_curriculum_end_beta":        pooling_curriculum_end_beta,
        "pooling_curriculum_hold_fraction":   pooling_curriculum_hold_fraction,
    }

    visualization_mode = SurfaceVisualizationMode(surface_visualization)
    surface_predictions = visualization_mode is not SurfaceVisualizationMode.NONE
    surface_prediction_npz = visualization_mode is SurfaceVisualizationMode.FULL

    effective_initialization_seed = seed if initialization_seed is None else initialization_seed
    effective_training_seed       = seed if training_seed is None else training_seed
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
        torch.set_float32_matmul_precision("medium")

    trial_index = work.trial.index if work.trial is not None else 0
    run_label = (
        f"trial={trial_index} seed={seed} init_seed={effective_initialization_seed} "
        f"training_seed={effective_training_seed}"
    )

    requested_splits = ("train", "val", "test") if evaluate_test else ("train", "val")
    work.log(f"[Training {run_label}] loading managed splits: {requested_splits}")

    datasets = {
        split: WisdomDataset(
            dataset,
            split,
            subset=subset,
            include_surface_targets=False,
            include_surface_geometry=(
                model_version == 3 or weak_surface_loss.needs_surface_neighbors
            ),
            include_atom_geometry=vector_atomic_channels > 0,
        )
        for split in requested_splits
    }
    loader_datasets = dict(datasets)

    # Surface targets live in separate sidecars and are not needed for global validation. A second
    # validation view prevents ordinary epochs from decoding those arrays or retaining every point
    # score merely to compute diagnostics that cannot select the model. Prediction-only reports need
    # identifiers here but read ground/soft targets directly only for the selected HTML sample.

    if surface_metrics or surface_predictions or faithfulness_audit:
        loader_datasets["val_surface"] = WisdomDataset(
            dataset,
            "val",
            subset=subset,
            include_surface_targets=surface_metrics,
            include_surface_geometry=(
                model_version == 3 or weak_surface_loss.needs_surface_neighbors
            ),
            include_atom_geometry=vector_atomic_channels > 0,
        )
        if evaluate_test:
            loader_datasets["test_surface"] = WisdomDataset(
                dataset,
                "test",
                subset=subset,
                include_surface_targets=surface_metrics,
                include_surface_geometry=(
                    model_version == 3 or weak_surface_loss.needs_surface_neighbors
                ),
                include_atom_geometry=vector_atomic_channels > 0,
            )

    generator = torch.Generator().manual_seed(effective_training_seed)
    collator  = WisdomCollator(
        atom_spatial_k=atom_spatial_k,
        surface_atom_k=surface_atom_k,
        diffusion_spectral_modes=diffusion_spectral_modes,
    )
    loaders: dict[str, DataLoader[Any]] = {}
    evaluation_workers = max(1, data_workers // 2) if data_workers else 0

    # Persistent training workers overlap compressed-NPZ decoding with the previous GPU batch.
    # Evaluation uses half as many temporary workers so the idle training pool plus the active
    # validation pool and trainer remain inside the per-Run CPU allocation.

    for split, split_dataset in loader_datasets.items():
        split_workers = (
            data_workers
            if split == "train"
            else evaluation_workers
        )
        loader_options: dict[str, Any] = {}
        if split_workers:
            loader_options.update(
                persistent_workers=split == "train",
                prefetch_factor=1,
            )

        loaders[split] = DataLoader(
            split_dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=split_workers,
            collate_fn=collator,
            generator=generator if split == "train" else None,
            pin_memory=device.type == "cuda",
            **loader_options,
        )

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

        curvature_widths[split] = int(curvature.shape[1] * 3)

    if len(set(curvature_widths.values())) != 1:
        raise ValueError(f"dataset splits disagree on curvature feature width: {curvature_widths}")

    curvature_features = curvature_widths["train"]

    available_parameters = {
        "hidden_dim":           hidden_dim,
        "embedding_dim":        embedding_dim,
        "residue_embedding_dim": residue_embedding_dim,
        "atomic_layers":        atomic_layers,
        "projection_depth":     projection_depth,
        "surface_layers":       surface_layers,
        "atom_spatial_k":       atom_spatial_k,
        "surface_atom_k":       surface_atom_k,
        "diffusion_spectral_modes": diffusion_spectral_modes,
        "surface_atom_radius":  surface_atom_radius,
        "surface_geometry_transfer": surface_geometry_transfer,
        "surface_chunk_size":   surface_chunk_size,
        "atomic_message_chunk_size": atomic_message_chunk_size,
        "vector_atomic_channels": vector_atomic_channels,
        "interaction_round": interaction_round,
        "atomic_edge_distance_scale": atomic_edge_distance_scale,
        "dropout":              dropout,
        "surface_encoder_type": surface_encoder_type,
        "surface_patch_size":   surface_patch_size,
        "pooling_type":         pooling_type,
        "topk_fraction":        topk_fraction,
        "attention_hidden_dim": attention_hidden_dim,
        "regional_diffusion_scale": regional_diffusion_scale,
        "log_sum_exp_beta":     log_sum_exp_beta,
        "head_type":            head_type,
        "global_context_dim":   global_context_dim,
        "detach_global_context": detach_global_context,
        "curvature_features":   curvature_features,
        "embedding_initialization": embedding_initialization,
        "gate_initial_active":  gate_initial_active,
        "diffusion_time_initialization": diffusion_time_initialization,
        "diffusion_residual_initialization": diffusion_residual_initialization,
        "atomic_residual_initialization": atomic_residual_initialization,
        "neutral_edge_initialization": neutral_edge_initialization,
        "neutral_transfer_initialization": neutral_transfer_initialization,
        "physics_distance_initialization": physics_distance_initialization,
        "local_head_weight_std": local_head_weight_std,
        "linear_initialization": linear_initialization,
    }
    # Parameter RNG and training RNG are deliberately separated. This makes the plan's two seed
    # decomposition experiments possible without changing LambdaForge's run-level seed semantics.

    torch.manual_seed(effective_initialization_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_initialization_seed)

    model, model_parameters = _create_model(model_version, available_parameters)
    architecture_name       = str(getattr(model, "ARCHITECTURE_NAME", type(model).__name__))
    model.to(device)
    initial_diffusion_times = _diffusion_time_summary(model)
    for statistic in ("minimum", "mean", "median", "maximum"):
        value = initial_diffusion_times.get(statistic)
        if isinstance(value, float):
            work.metrics.log(f"initial_diffusion_time_{statistic}", value)

    calibrated_local_bias: float | None = None
    if calibrate_local_head_bias:
        calibration_loader = DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            collate_fn=collator,
        )
        calibrated_local_bias = _calibrate_local_head_bias(
            model,
            calibration_loader,
            device,
            autocast_dtype,
            calibration_batches,
        )
        work.metrics.log("initial_local_head_bias", calibrated_local_bias)

    initial_checkpoint = work.run_dir / "initial-model.pt"
    if save_initial_checkpoint:
        torch.save(
            {
                "model_state":         model.state_dict(),
                "model_parameters":    model_parameters,
                "initialization_seed": effective_initialization_seed,
            },
            initial_checkpoint,
        )

    # Reset every stochastic training source after model creation and optional deterministic bias
    # calibration. Dropout, Hard-Concrete samples, and any training-time torch randomness now vary
    # only with the explicitly recorded training seed.

    torch.manual_seed(effective_training_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(effective_training_seed)

    gate_parameters = tuple(model.semantic_gates.parameters())
    gate_ids        = {id(parameter) for parameter in gate_parameters}
    model_parameters_without_gates = tuple(
        parameter for parameter in model.parameters() if id(parameter) not in gate_ids
    )
    optimizer = torch.optim.AdamW(
        (
            {"params": model_parameters_without_gates, "weight_decay": weight_decay},
            {"params": gate_parameters, "weight_decay": 0.0},
        ),
        lr=learning_rate,
    )
    optimization_monitor = OptimizationDiagnostics(model) if optimization_diagnostics else None
    weight_average = (
        ModelWeightAverage(
            model,
            averaging_mode,
            ema_decay=ema_decay,
            swa_start_epoch=max(1, math.ceil(swa_start_fraction * epochs)),
        )
        if averaging_mode is not WeightAveragingMode.NONE
        else None
    )

    split_sizes = {split: len(split_dataset) for split, split_dataset in datasets.items()}
    split_storage = {
        split: split_dataset.storage_bytes()
        for split, split_dataset in datasets.items()
    }
    preprocessing_bytes = sum(values["total"] for values in split_storage.values())

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    gate_parameter_count = sum(parameter.numel() for parameter in gate_parameters)
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size()
        for parameter in model.parameters()
    )
    parameter_mib   = parameter_bytes / 2**20

    evidence_parameter_count = int(
        getattr(model, "evidence_head_parameter_count", 0)
    )
    evidence_added_parameters = int(
        getattr(model, "evidence_head_added_parameter_count", 0)
    )
    evidence_input_width = int(getattr(model, "evidence_input_width", hidden_dim))

    work.metrics.log("parameter_count", float(parameter_count))
    work.metrics.log("gate_parameter_count", float(gate_parameter_count))
    work.metrics.log("semantic_gate_count", float(len(model.semantic_gates.names)))
    work.metrics.log("evidence_head_parameter_count", float(evidence_parameter_count))
    work.metrics.log("evidence_head_added_parameter_count", float(evidence_added_parameters))
    work.metrics.log("evidence_input_width", float(evidence_input_width))
    work.metrics.log("preprocessing_bytes", float(preprocessing_bytes))

    if not surface_metrics:
        surface_schedule = "disabled"
    elif surface_metrics_interval == 0:
        surface_schedule = "best-checkpoint-only"
    else:
        surface_schedule = f"every-{surface_metrics_interval}-epochs-plus-best"

    work.log(
        f"[Training {run_label}] starting on {device.type}; splits={split_sizes}; "
        f"architecture={architecture_name}; curvature_features={curvature_features}; "
        f"atomic_edge_distance_scale={atomic_edge_distance_scale:g}A; "
        f"semantic_gates={len(model.semantic_gates.names)}; gate_lambda={gate_lambda:g}; "
        f"evidence_input_width={evidence_input_width}; "
        f"evidence_head_parameters={evidence_parameter_count:,} "
        f"(added={evidence_added_parameters:,}); "
        f"batch_size={batch_size}; "
        f"data_workers=train:{data_workers}/eval:{evaluation_workers}; "
        f"precision={effective_precision}; "
        f"matmul_precision={torch.get_float32_matmul_precision()}; parameters={parameter_count:,} "
        f"({parameter_mib:.2f} MiB in FP32); preprocessing_bytes={preprocessing_bytes:,}; "
        f"maximum_epochs={epochs}; patience={patience}; "
        f"weak_surface_losses={weak_surface_configuration}; "
        f"initialization={initialization_configuration}; "
        f"surface_metrics={surface_schedule}; surface_visualization={visualization_mode.value}"
    )

    best_auprc                 = float("-inf")
    best_selection_utility     = float("-inf")
    best_val_loss              = float("inf")
    best_validation_metrics    : dict[str, float | None] = {}
    best_epoch                 = 0
    epochs_completed           = 0
    epochs_without_improvement = 0
    patience_reference_utility = float("-inf")
    peak_allocated_gib         = 0.0
    peak_reserved_gib          = 0.0
    surface_curve              : list[dict[str, float]] = []
    subgroup_curves            : dict[str, list[dict[str, float]]] = {}

    stop_reason: str | None = None

    checkpoint = work.run_dir / "best-model.pt"

    maximum_atoms          = 0
    maximum_surface_points = 0
    maximum_atomic_edges   = 0
    maximum_atomic_degree  = 0
    mean_atomic_degree_sum = 0.0
    atom_count_sum          = 0
    surface_point_sum       = 0
    profiled_batches        = 0
    spectral_modes_seen     : list[int] = []

    # Train up to the safety ceiling. Validation patience decides the useful duration of an
    # ordinary Run, while LambdaForge may cooperatively prune a weak HPO candidate.

    for epoch in range(1, epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # Delay semantic pruning until the representation has learned a useful signal. During the
        # following ramp, gates are learned normally while their expected-L0 pressure rises from
        # zero to the requested value. LR warm-up is independent and affects every optimizer group.

        training_fraction = (epoch - 1) / max(1, epochs)
        if training_fraction < gate_warmup_fraction:
            model.semantic_gates.set_override("all_on")
            effective_gate_lambda = 0.0
            gate_phase = "all_on"
        else:
            model.semantic_gates.set_override("learned")
            ramp_progress = training_fraction - gate_warmup_fraction
            if gate_ramp_fraction > 0.0 and ramp_progress < gate_ramp_fraction:
                effective_gate_lambda = gate_lambda * ramp_progress / gate_ramp_fraction
                gate_phase = "ramp"
            else:
                effective_gate_lambda = gate_lambda
                gate_phase = "learned"

        warmup_epochs = math.ceil(learning_rate_warmup_fraction * epochs)
        learning_rate_scale = min(1.0, epoch / warmup_epochs) if warmup_epochs else 1.0
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = learning_rate * learning_rate_scale

        current_pooling_beta: float | None = None
        if pooling_curriculum_end_beta is not None:
            if training_fraction <= pooling_curriculum_hold_fraction:
                current_pooling_beta = 1.0
            else:
                sharp_progress = (
                    training_fraction - pooling_curriculum_hold_fraction
                ) / (1.0 - pooling_curriculum_hold_fraction)
                current_pooling_beta = 1.0 + sharp_progress * (
                    pooling_curriculum_end_beta - 1.0
                )
            pooling_head = cast(Any, model).pooling_head
            pooling_head.set_log_sum_exp_beta(current_pooling_beta)
        if optimization_monitor is not None:
            optimization_monitor.begin_epoch()

        model.train()
        task_loss_sum           = torch.zeros((), device=device)
        direct_task_loss_sum    = torch.zeros((), device=device)
        surface_task_loss_sum   = torch.zeros((), device=device)
        gate_regularization_sum = torch.zeros((), device=device)
        total_loss_sum          = torch.zeros((), device=device)
        weak_loss_sums = {
            name: torch.zeros((), device=device)
            for name in (
                "negative",
                "positive_existence",
                "regional_positive",
                "regional_ranking",
                "cardinality",
                "total_variation",
                "dirichlet",
                "total",
            )
        }
        examples                = 0
        data_wait_seconds       = 0.0
        previous_batch_finished = time.perf_counter()

        for batch in loaders["train"]:
            data_wait_seconds += time.perf_counter() - previous_batch_finished

            if epoch == 1:
                atomic_numbers = _tensor(batch, "atomic_numbers")
                surface_points = int(_tensor(batch, "surface_area_weights").shape[0])
                active_edges   = _tensor(batch, "atom_edge_index")
                degree         = torch.bincount(
                    active_edges.flatten(),
                    minlength=len(atomic_numbers),
                )

                maximum_atoms          = max(maximum_atoms, len(atomic_numbers))
                maximum_surface_points = max(maximum_surface_points, surface_points)
                directional_messages   = 2 * active_edges.shape[1]
                maximum_atomic_edges   = max(maximum_atomic_edges, directional_messages)
                maximum_atomic_degree  = max(maximum_atomic_degree, int(degree.max()))
                mean_atomic_degree_sum += float(degree.float().mean())
                atom_count_sum          += len(atomic_numbers)
                surface_point_sum       += surface_points
                profiled_batches        += 1
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
                if optimization_monitor is not None:
                    optimization_monitor.observe_forward(output)
                target = _tensor(tensors, "target")
                direct_task_loss = F.binary_cross_entropy_with_logits(output["logits"], target)
                if "surface_protein_logits" in output:
                    surface_task_loss = F.binary_cross_entropy_with_logits(
                        output["surface_protein_logits"],
                        target,
                    )
                    task_loss = (
                        direct_head_weight * direct_task_loss
                        + surface_head_weight * surface_task_loss
                    )
                else:
                    surface_task_loss = direct_task_loss.new_zeros(())
                    task_loss = direct_task_loss
                weak_losses = weak_surface_loss(
                    logits=output["surface_logits"],
                    area_weights=_tensor(tensors, "surface_area_weights"),
                    owners=_tensor(tensors, "surface_batch"),
                    protein_targets=target,
                    operators=_operators(tensors),
                    surface_ptr=_tensor(tensors, "surface_ptr"),
                    surface_neighbors=(
                        _tensor(tensors, "surface_neighbors")
                        if weak_surface_loss.needs_surface_neighbors
                        else None
                    ),
                    neighbor_mask=(
                        _tensor(tensors, "surface_neighbor_mask")
                        if weak_surface_loss.needs_surface_neighbors
                        else None
                    ),
                )
                gate_regularization = model.gate_regularization()
                total_loss = (
                    task_loss
                    + effective_gate_lambda * gate_regularization
                    + weak_losses["total"]
                )

            if scaler.is_enabled():
                scaler.scale(total_loss).backward()  # type: ignore[no-untyped-call]
                if gradient_clip_norm is not None or optimization_monitor is not None:
                    scaler.unscale_(optimizer)
                if optimization_monitor is not None:
                    optimization_monitor.observe_backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                total_loss.backward()  # type: ignore[no-untyped-call]
                if optimization_monitor is not None:
                    optimization_monitor.observe_backward()
                if gradient_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
                optimizer.step()
            if weight_average is not None:
                weight_average.update(model, epoch)

            count                   = len(target)
            task_loss_sum           += task_loss.detach() * count
            direct_task_loss_sum    += direct_task_loss.detach() * count
            surface_task_loss_sum   += surface_task_loss.detach() * count
            gate_regularization_sum += gate_regularization.detach() * count
            total_loss_sum          += total_loss.detach() * count
            for name, weak_value in weak_losses.items():
                weak_loss_sums[name] += weak_value.detach() * count
            examples += count

            # The model returns point-level diagnostics in addition to protein logits. Explicitly
            # release the completed batch so it cannot overlap the next batch or validation pass.

            del (
                tensors,
                output,
                target,
                task_loss,
                direct_task_loss,
                surface_task_loss,
                weak_losses,
                gate_regularization,
                total_loss,
            )
            previous_batch_finished = time.perf_counter()

        optimizer.zero_grad(set_to_none=True)

        if device.type == "cuda":
            torch.cuda.synchronize(device)

        training_seconds = time.perf_counter() - epoch_started
        train_throughput = examples / max(training_seconds, 1.0e-9)
        train_task_loss = float(task_loss_sum) / max(1, examples)
        train_direct_task_loss = float(direct_task_loss_sum) / max(1, examples)
        train_surface_task_loss = float(surface_task_loss_sum) / max(1, examples)
        train_weak_losses = {
            name: float(value) / max(1, examples)
            for name, value in weak_loss_sums.items()
        }
        train_negative_surface_loss = train_weak_losses["negative"]
        train_gate_regularization = float(gate_regularization_sum) / max(1, examples)
        train_total_loss = float(total_loss_sum) / max(1, examples)

        # Global validation is required every epoch for checkpoint selection, patience, and HPO.
        # Local diagnostics use the sidecar-enabled view only on explicitly scheduled epochs.

        surface_metrics_due = (
            surface_metrics
            and surface_metrics_interval > 0
            and epoch % surface_metrics_interval == 0
        )
        validation_loader = (
            loaders["val_surface"]
            if surface_metrics_due
            else loaders["val"]
        )
        averaged_validation = weight_average is not None and weight_average.ready
        if averaged_validation:
            assert weight_average is not None
            weight_average.apply(model)

        validation_started = time.perf_counter()
        validation, surface_validation = _evaluate(
            model,
            validation_loader,
            device,
            autocast_dtype,
            faithfulness_audit=False,
        )
        validation_seconds = time.perf_counter() - validation_started
        work.metrics.log(
            "averaged_validation_weights",
            float(averaged_validation),
            step=epoch,
            split="val",
        )

        work.metrics.log("loss", train_total_loss, step=epoch, split="train")
        work.metrics.log("task_loss", train_task_loss, step=epoch, split="train")
        work.metrics.log("direct_task_loss", train_direct_task_loss, step=epoch, split="train")
        if head_type != "single":
            work.metrics.log(
                "surface_task_loss",
                train_surface_task_loss,
                step=epoch,
                split="train",
            )
        for name, weak_value in train_weak_losses.items():
            work.metrics.log(
                f"weak_surface_{name}",
                weak_value,
                step=epoch,
                split="train",
            )
        work.metrics.log(
            "negative_surface_loss",
            train_negative_surface_loss,
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "gate_regularization", train_gate_regularization, step=epoch, split="train"
        )
        work.metrics.log("total_loss", train_total_loss, step=epoch, split="train")
        work.metrics.log("learning_rate", learning_rate * learning_rate_scale, step=epoch)
        work.metrics.log("effective_gate_lambda", effective_gate_lambda, step=epoch)
        if current_pooling_beta is not None:
            work.metrics.log("pooling_beta", current_pooling_beta, step=epoch)
        if optimization_monitor is not None:
            for name, diagnostic_value in optimization_monitor.metrics().items():
                work.metrics.log(name, diagnostic_value, step=epoch, split="train")
        gate_expected = float(model.gate_regularization().detach().cpu())
        gate_deterministic = sum(
            float(model.semantic_gates.deterministic_value(name).detach().cpu()) > 0.0
            for name in model.semantic_gates.names
        ) / len(model.semantic_gates.names)
        work.metrics.log(
            "gate_expected_active_fraction", gate_expected, step=epoch, split="train"
        )
        work.metrics.log(
            "gate_deterministic_active_fraction", gate_deterministic, step=epoch, split="train"
        )

        if epoch == 1:
            static_metrics = {
                "maximum_surface_points":       float(maximum_surface_points),
                "maximum_atoms":                float(maximum_atoms),
                "maximum_active_atomic_edges":  float(maximum_atomic_edges),
                "maximum_atomic_degree":        float(maximum_atomic_degree),
                "mean_atomic_degree":           mean_atomic_degree_sum / max(1, profiled_batches),
                "mean_atoms_per_batch":         atom_count_sum / max(1, profiled_batches),
                "mean_surface_points_per_batch": surface_point_sum / max(1, profiled_batches),
                "active_atom_spatial_k":         float(atom_spatial_k),
                "active_surface_atom_k":         float(surface_atom_k),
            }
            if spectral_modes_seen:
                static_metrics["mean_spectral_modes"] = (
                    sum(spectral_modes_seen) / len(spectral_modes_seen)
                )
            for name, metric_value in static_metrics.items():
                work.metrics.log(name, metric_value, step=epoch, split="train")

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
        work.metrics.log(
            "data_wait_seconds",
            data_wait_seconds,
            step=epoch,
            split="train",
        )
        work.metrics.log(
            "validation_seconds",
            validation_seconds,
            step=epoch,
            split="val",
        )
        work.metrics.log(
            "surface_metrics_computed",
            float(surface_metrics_due),
            step=epoch,
            split="val",
        )
        for name, optional_metric in validation.items():
            if optional_metric is not None:
                work.metrics.log(name, optional_metric, step=epoch, split="val")
        for name, optional_metric in surface_validation.items():
            if optional_metric is not None:
                work.metrics.log(name, optional_metric, step=epoch, split="val")

        # MCC itself remains unavailable when a thresholded predictor emits only one class. The
        # HPO component is deliberately distinct: it maps a defined MCC from [-1,1] to [0,1] and
        # assigns the worst utility, zero, to an undefined degenerate decision rule. This lets HPO
        # reject the candidate without publishing a fabricated MCC or failing the entire Run.

        mcc_defined   = validation["mcc"] is not None
        mcc_objective = _mcc_objective(validation["mcc"])
        work.metrics.log("mcc_defined", float(mcc_defined), step=epoch, split="val")
        work.metrics.log("mcc_objective", mcc_objective, step=epoch, split="val")

        objective         = validation["auprc"]
        selection_utility = _validation_utility(validation)
        validation_loss   = validation["loss"]
        if validation_loss is None:
            raise RuntimeError("validation loss is unavailable for a non-empty split")
        if selection_utility is not None:
            work.metrics.log("selection_utility", selection_utility, step=epoch, split="val")

        # Development-time surface evidence remains outside loss and checkpoint selection. When
        # available, it is paired with the global score from the same epoch so LambdaForge can
        # display localization, selection regret, rank coupling, and the frozen WISDOM utility.

        surface_score = surface_validation.get("surface_positive_macro_auprc")
        coupling_metrics: dict[str, float | None] = {}
        subgroup_coupling_metrics: dict[str, float | None] = {}
        if selection_utility is not None and surface_score is not None:
            surface_curve.append(
                {
                    "epoch":   float(epoch),
                    "global":  selection_utility,
                    "surface": float(surface_score),
                }
            )
            coupling_metrics = _surface_coupling(surface_curve)
            for name, optional_metric in coupling_metrics.items():
                if optional_metric is not None:
                    work.metrics.log(name, optional_metric, step=epoch, split="val")

        # Dataset-size, surface-size, phenotype, and prevalence strata remain diagnostic. Each
        # subgroup builds its own G/S trajectory and W value, but none enters checkpoint selection
        # or the top-level HPO objective.

        subgroup_coupling_metrics = _subgroup_coupling(
            surface_validation,
            subgroup_curves,
            epoch,
        )
        for name, optional_metric in subgroup_coupling_metrics.items():
            if optional_metric is not None:
                work.metrics.log(name, optional_metric, step=epoch, split="val")

        epochs_completed = epoch

        checkpoint_improved = (
            selection_utility is not None
            and selection_utility > best_selection_utility
        )
        patience_improved   = (
            selection_utility is not None
            and selection_utility > patience_reference_utility + minimum_delta
        )

        # The saved checkpoint is the exact argmax of G required by surface regret. minimum_delta
        # controls only patience, so a small real improvement cannot make the report describe a
        # different epoch from the weights that are eventually evaluated and published.

        if checkpoint_improved:
            if objective is None:
                raise RuntimeError("protein validation score G cannot exist without AUPRC")
            best_auprc              = objective
            best_selection_utility  = cast(float, selection_utility)
            best_val_loss           = validation_loss
            best_validation_metrics = dict(validation)
            best_epoch              = epoch

            torch.save(
                {
                    "model_version":    model_version,
                    "model_parameters": model_parameters,
                    "pooling_type":     pooling_type,
                    "head_type":        head_type,
                    "direct_head_weight": direct_head_weight,
                    "surface_head_weight": surface_head_weight,
                    "surface_encoder_type": surface_encoder_type,
                    "gate_lambda":      gate_lambda,
                    "weak_surface_loss": weak_surface_configuration,
                    "initialization": initialization_configuration,
                    "semantic_gate_definition": tuple(
                        (name, model.semantic_gates.parents[name])
                        for name in model.semantic_gates.names
                    ),
                    "hard_concrete": model.gate_summary()["hard_concrete"],
                    "data_parameters": {
                        "atom_spatial_k":           atom_spatial_k,
                        "surface_atom_k":           surface_atom_k,
                        "diffusion_spectral_modes": diffusion_spectral_modes,
                    },
                    "state_dict":       model.state_dict(),
                    "seed":             seed,
                    "initialization_seed": effective_initialization_seed,
                    "training_seed":       effective_training_seed,
                    "epoch":            epoch,
                    "val_auprc":        objective,
                    "val_loss":         best_val_loss,
                    "validation_metrics": best_validation_metrics,
                    "selection_utility": best_selection_utility,
                },
                checkpoint,
            )

        if averaged_validation:
            assert weight_average is not None
            weight_average.restore(model)

        if patience_improved:
            patience_reference_utility = cast(float, selection_utility)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        work.metrics.log(
            "patience_used",
            float(epochs_without_improvement),
            step=epoch,
            split="val",
        )
        work.metrics.log(
            "patience_remaining",
            float(max(0, patience - epochs_without_improvement)),
            step=epoch,
            split="val",
        )

        # One compact, labelled line per epoch remains readable with ten interleaved GPU Runs.
        # The complete metric suite is still stored structurally by LambdaForge above.

        auroc    = validation["auroc"]
        balanced = validation["balanced_accuracy"]
        mcc      = validation["mcc"]

        auprc_text    = "n/a" if objective is None else f"{objective:.4f}"
        auroc_text    = "n/a" if auroc is None else f"{auroc:.4f}"
        balanced_text = "n/a" if balanced is None else f"{balanced:.4f}"
        mcc_text      = "n/a" if mcc is None else f"{mcc:.4f}"
        best_text     = "n/a" if best_epoch == 0 else f"{best_auprc:.4f}"
        best_utility_text = (
            "n/a" if best_epoch == 0 else f"{best_selection_utility:.4f}"
        )
        utility_text  = (
            "n/a" if selection_utility is None else f"{selection_utility:.4f}"
        )
        wisdom_text   = _metric_text(coupling_metrics, "wisdom_hpo_score")
        regret_text   = _metric_text(coupling_metrics, "surface_selection_regret")
        coupling_text = _metric_text(coupling_metrics, "surface_coupling_score")
        val_loss_text = _metric_text(validation, "loss")

        surface_micro_text = _metric_text(surface_validation, "surface_micro_auprc")
        surface_macro_text = _metric_text(
            surface_validation,
            "surface_positive_macro_auprc",
        )
        surface_auroc_text = _metric_text(surface_validation, "surface_positive_macro_auroc")
        surface_status = (
            "computed"
            if surface_metrics_due
            else "deferred" if surface_metrics else "disabled"
        )

        progress_message = (
            f"{run_label}; train_loss={train_total_loss:.5f}; val_loss={val_loss_text}; "
            f"val_auprc={auprc_text}; utility={utility_text}; "
            f"surface_macro_auprc={surface_macro_text}; "
            f"wisdom_score={wisdom_text}; "
            f"patience={epochs_without_improvement}/{patience}; best={best_text}"
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
            f"[Training {run_label}] epoch={epoch}/{epochs} "
            f"train_task={train_task_loss:.5f} "
            f"negative_surface={train_negative_surface_loss:.5f} "
            f"gate_L0={train_gate_regularization:.5f} "
            f"train_total={train_total_loss:.5f} "
            f"protein_val[loss={val_loss_text},auprc={auprc_text},auroc={auroc_text},"
            f"balanced_accuracy={balanced_text},mcc={mcc_text},utility={utility_text},"
            f"best_utility={best_utility_text},best_auprc={best_text}] "
            f"surface_val[status={surface_status},micro_auprc={surface_micro_text},"
            f"positive_macro_auprc={surface_macro_text},"
            f"positive_macro_auroc={surface_auroc_text},regret={regret_text},"
            f"coupling={coupling_text},wisdom_score={wisdom_text}] "
            f"optimization[lr={learning_rate * learning_rate_scale:.3e},"
            f"gate_phase={gate_phase},gate_lambda={effective_gate_lambda:.3e}] "
            f"patience={epochs_without_improvement}/{patience} "
            f"batch_cost=atoms:{maximum_atoms:,},points:{maximum_surface_points:,},"
            f"atomic_edges:{maximum_atomic_edges:,},degree_max:{maximum_atomic_degree},"
            f"K:{atom_spatial_k},J:{surface_atom_k},modes:"
            f"{max(spectral_modes_seen, default=0)} train_throughput={train_throughput:.2f}/s "
            f"epoch_throughput={proteins_per_second:.2f}/s "
            f"data_wait={data_wait_seconds:.1f}s validation={validation_seconds:.1f}s"
            f"{memory_text}"
        )

        # A framework pruning request is checked only after metrics and the best checkpoint have
        # been persisted. The Run then returns normally and LambdaForge records it as pruned.

        if work.stop_requested:
            stop_reason = "adaptive-hpo"
            work.log(f"Stopping candidate after epoch {epoch}: adaptive HPO pruning requested")
            break

        # Patience prevents an otherwise healthy Run from spending epochs on a validation plateau.
        # Protein score G must improve by minimum_delta, so negligible changes do not reset it.

        if epochs_without_improvement >= patience:
            stop_reason = "validation-patience"
            work.log(
                f"Stopping after epoch {epoch}: protein validation score G did not improve by "
                f"{minimum_delta:g} for {patience} epochs"
            )
            break

    if best_epoch == 0:
        raise RuntimeError(
            "protein validation score G remained undefined; validation AUPRC and AUROC require "
            "both target classes"
        )

    # Candidate ranking must use the best validation checkpoint rather than the final plateau
    # observation. This unstepped summary does not alter LambdaForge's per-epoch pruning history.

    for name in ("auprc", "auroc", "balanced_accuracy", "mcc"):
        value = best_validation_metrics.get(name)
        if value is not None:
            work.metrics.log(name, value, split="val")
    work.metrics.log(
        "mcc_defined",
        float(best_validation_metrics.get("mcc") is not None),
        split="val",
    )
    work.metrics.log(
        "mcc_objective",
        _mcc_objective(best_validation_metrics.get("mcc")),
        split="val",
    )

    best_surface_metrics      : dict[str, float | None] | None = None
    test_metrics              : dict[str, float | None] | None = None
    test_surface_metrics      : dict[str, float | None] | None = None
    surface_prediction_reports: list[dict[str, Any]] = []
    prediction_root           : Path | None = None

    # Pruned HPO Runs are excluded from candidate scores and need no held-out test evaluation.
    # Completed Runs, including those stopped by patience, evaluate test exactly once at their
    # validation-selected checkpoint.

    if stop_reason != "adaptive-hpo":
        saved = torch.load(checkpoint, map_location=device, weights_only=True)
        model.load_state_dict(saved["state_dict"])

        if surface_predictions:
            prediction_root = Path(
                work.outputs.directory(
                    "surface-predictions",
                    role="visualization",
                )
            )

        # Local validation is always reported once for the exact checkpoint selected by the global
        # protein score G. This is the only local pass when the configured interval is zero.

        if surface_metrics or surface_predictions or faithfulness_audit:
            work.log(
                f"[Training {run_label}] evaluating best-checkpoint validation surface maps"
            )
            validation_predictions = (
                SurfacePredictionReport(
                    loader_datasets["val_surface"],
                    prediction_root,
                    "validation",
                    surface_prediction_threshold,
                    surface_visualization_maximum,
                    surface_prediction_npz,
                )
                if prediction_root is not None
                else None
            )
            _, best_surface_metrics = _evaluate(
                model,
                loaders["val_surface"],
                device,
                autocast_dtype,
                validation_predictions,
                faithfulness_audit=faithfulness_audit,
            )

            if surface_metrics or faithfulness_audit:
                for name, optional_metric in best_surface_metrics.items():
                    if optional_metric is not None:
                        work.metrics.log(name, optional_metric, split="val")

            if validation_predictions is not None:
                prediction_report = validation_predictions.publish(
                    best_surface_metrics,
                    best_epoch,
                )
                surface_prediction_reports.append(prediction_report)
                work.metrics.log(
                    "surface_prediction_proteins",
                    prediction_report["predicted_proteins"],
                    split="val",
                )
                work.metrics.log(
                    "surface_visualized_proteins",
                    prediction_report["visualized_proteins"],
                    split="val",
                )
                work.log(
                    f"[Training {run_label}] wrote validation surface maps for "
                    f"{prediction_report['predicted_proteins']} proteins and interactive reports "
                    f"for {prediction_report['visualized_proteins']}"
                )

        if evaluate_test:
            work.log(f"[Training {run_label}] evaluating held-out test once")
            test_predictions = (
                SurfacePredictionReport(
                    loader_datasets["test_surface"],
                    prediction_root,
                    "test",
                    surface_prediction_threshold,
                    surface_visualization_maximum,
                    surface_prediction_npz,
                )
                if prediction_root is not None
                else None
            )
            test_metrics, test_surface_metrics = _evaluate(
                model,
                loaders["test_surface"]
                if surface_metrics or surface_predictions or faithfulness_audit
                else loaders["test"],
                device,
                autocast_dtype,
                test_predictions,
                faithfulness_audit=faithfulness_audit,
            )

            for name, optional_metric in test_metrics.items():
                if optional_metric is not None:
                    work.metrics.log(name, optional_metric, split="test")
            if surface_metrics:
                for name, optional_metric in test_surface_metrics.items():
                    if optional_metric is not None:
                        work.metrics.log(name, optional_metric, split="test")

            if test_predictions is not None:
                prediction_report = test_predictions.publish(
                    test_surface_metrics,
                    best_epoch,
                )
                surface_prediction_reports.append(prediction_report)
                work.metrics.log(
                    "surface_prediction_proteins",
                    prediction_report["predicted_proteins"],
                    split="test",
                )
                work.metrics.log(
                    "surface_visualized_proteins",
                    prediction_report["visualized_proteins"],
                    split="test",
                )
                work.log(
                    f"[Training {run_label}] wrote test surface maps for "
                    f"{prediction_report['predicted_proteins']} proteins and interactive reports "
                    f"for {prediction_report['visualized_proteins']}"
                )

        if prediction_root is not None and surface_prediction_reports:
            SurfacePredictionReport.publish_index(
                prediction_root,
                surface_prediction_reports,
            )

    # Complete Runs expose one unstepped HPO result. If surface metrics were deferred during
    # training, the restored global checkpoint supplies its single surface observation; regret and
    # rank correlation then remain explicitly unavailable until a per-epoch schedule is enabled.

    final_coupling         : dict[str, float | None] = {}
    final_subgroup_coupling: dict[str, float | None] = {}
    if best_surface_metrics is not None:
        best_surface_score = best_surface_metrics.get("surface_positive_macro_auprc")
        if best_surface_score is not None:
            if not any(int(point["epoch"]) == best_epoch for point in surface_curve):
                surface_curve.append(
                    {
                        "epoch":   float(best_epoch),
                        "global":  best_selection_utility,
                        "surface": float(best_surface_score),
                    }
                )
                surface_curve.sort(key=lambda point: point["epoch"])
            final_coupling = _surface_coupling(surface_curve)
            for name, optional_metric in final_coupling.items():
                if optional_metric is not None:
                    work.metrics.log(name, optional_metric, split="val")

        final_subgroup_coupling = _subgroup_coupling(
            best_surface_metrics,
            subgroup_curves,
            best_epoch,
        )
        for name, optional_metric in final_subgroup_coupling.items():
            if optional_metric is not None:
                work.metrics.log(name, optional_metric, split="val")

    final_diffusion_times = _diffusion_time_summary(model)
    for statistic in ("minimum", "mean", "median", "maximum"):
        value = final_diffusion_times.get(statistic)
        if isinstance(value, float):
            work.metrics.log(f"final_diffusion_time_{statistic}", value)

    report = {
        "model_version":                  model_version,
        "architecture":                   architecture_name,
        "pooling_type":                   pooling_type,
        "head_type":                      head_type,
        "direct_head_weight":             direct_head_weight,
        "surface_head_weight":            surface_head_weight,
        "surface_encoder_type":           surface_encoder_type,
        "gate_lambda":                    gate_lambda,
        "weak_surface_loss":              weak_surface_configuration,
        "initialization":                 initialization_configuration,
        "initial_diffusion_times":         initial_diffusion_times,
        "final_diffusion_times":           final_diffusion_times,
        "calibrated_local_head_bias":     calibrated_local_bias,
        "model_parameters":               model_parameters,
        "subset":                         subset,
        "seed":                           seed,
        "initialization_seed":            effective_initialization_seed,
        "training_seed":                  effective_training_seed,
        "epochs_completed":               epochs_completed,
        "best_epoch":                     best_epoch,
        "best_val_auprc":                 best_auprc,
        "best_selection_utility":          best_selection_utility,
        "best_val_loss":                  best_val_loss,
        "early_stopping_patience":        patience,
        "early_stopping_minimum_delta":   minimum_delta,
        "precision":                      effective_precision,
        "surface_metrics_enabled":         surface_metrics,
        "surface_metrics_interval":        surface_metrics_interval,
        "surface_visualization":           visualization_mode.value,
        "surface_prediction_threshold":    surface_prediction_threshold,
        "surface_visualization_maximum":   surface_visualization_maximum,
        "faithfulness_audit":              faithfulness_audit,
        "surface_prediction_reports":      surface_prediction_reports,
        "surface_coupling":                final_coupling,
        "surface_metric_curve":            surface_curve,
        "subgroup_coupling":               final_subgroup_coupling,
        "subgroup_metric_curves":          subgroup_curves,
        "test_evaluated":                evaluate_test,
        "stop_reason":                   stop_reason,
        "best_validation_surface":       best_surface_metrics,
        "test":                          test_metrics,
        "test_surface":                  test_surface_metrics,
        "curvature_features":            curvature_features,
        "parameter_count":               parameter_count,
        "gate_parameter_count":          gate_parameter_count,
        "semantic_gate_count":           len(model.semantic_gates.names),
        "evidence_head_parameter_count": evidence_parameter_count,
        "evidence_head_added_parameter_count": evidence_added_parameters,
        "evidence_input_width":        evidence_input_width,
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
    gate_summary_path = work.run_dir / "gate_summary.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    gate_summary_path.write_text(
        json.dumps(model.gate_summary(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    work.outputs.artifact("best-model", checkpoint, role="checkpoint")
    if save_initial_checkpoint:
        work.outputs.artifact("initial-model", initial_checkpoint, role="checkpoint")
    work.outputs.artifact(
        "gate-summary",
        gate_summary_path,
        role="report",
        media_type="application/json",
    )
    work.outputs.artifact(
        "evaluation",
        report_path,
        role="report",
        media_type="application/json",
    )
    if optimization_monitor is not None:
        optimization_monitor.close()
    return report


def _diffusion_time_summary(model: torch.nn.Module) -> dict[str, Any]:
    """Summarize every learned DiffusionNet heat time without changing model state.

    A DiffusionNet block owns one positive heat time per hidden channel. If ``t`` is expressed in
    square ångströms, its characteristic spatial length is approximately ``sqrt(t)`` ångströms.
    WISDOM stores per-block extrema, mean, and median so initialization profiles can be compared
    with the distribution reached by the globally selected checkpoint. The same statistics over
    every discovered channel form the top-level LambdaForge metrics.

    Args:
        model: WISDOM module that may contain one or more ``DiffusionSurfaceEncoder`` instances.

    Returns:
        JSON-compatible mapping with a ``blocks`` list and aggregate ``minimum``, ``mean``,
        ``median``, and ``maximum`` values. A non-DiffusionNet encoder returns only an empty block
        list because no heat-time parameter exists.
    """
    blocks     : list[dict[str, float | int | str]] = []
    all_values : list[Tensor] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, DiffusionSurfaceEncoder):
            continue
        for block_index, block in enumerate(module.blocks):
            typed_block = cast(DiffusionBlock, block)
            values      = typed_block.diffusion_times.detach().float().cpu()
            all_values.append(values)
            blocks.append(
                {
                    "module":   module_name,
                    "block":    block_index,
                    "channels": len(values),
                    "minimum":  float(values.min()),
                    "mean":     float(values.mean()),
                    "median":   float(values.median()),
                    "maximum":  float(values.max()),
                }
            )

    summary: dict[str, Any] = {"blocks": blocks}
    if all_values:
        values = torch.cat(all_values)
        summary.update(
            {
                "minimum": float(values.min()),
                "mean":    float(values.mean()),
                "median":  float(values.median()),
                "maximum": float(values.max()),
            }
        )
    return summary


def _calibrate_local_head_bias(
    model              : torch.nn.Module,
    loader             : DataLoader[Any],
    device             : torch.device,
    autocast_dtype     : torch.dtype | None,
    calibration_batches: int,
) -> float:
    """Match initial pooled prevalence by shifting the shared local-head bias.

    Every current WISDOM pooling is equivariant to adding the same scalar to all local logits in
    one protein. The method temporarily sets the local-head bias to zero, observes representative
    training logits, and solves ``mean(sigmoid(logit+b)) = train_prevalence`` by bisection. Local
    weights remain small but nonzero, avoiding MAX ties while removing the random-extreme positive
    bias induced by thousands of surface points.

    Args:
        model: Newly initialized WISDOM model exposing a biased ``local_head`` linear layer.
        loader: Deterministic training loader used only for initial calibration.
        device: Device receiving each calibration batch.
        autocast_dtype: Optional CUDA mixed-precision dtype.
        calibration_batches: Positive maximum number of batches to observe.

    Returns:
        Calibrated scalar bias assigned to ``model.local_head.bias``.

    Raises:
        ValueError: If the model lacks a compatible head or the calibration subset has one class.
    """
    local_head = getattr(model, "local_head", None)
    if not isinstance(local_head, torch.nn.Linear) or local_head.bias is None:
        raise ValueError("local-head calibration requires one biased linear local_head")

    local_head.bias.data.zero_()
    logits : list[Tensor] = []
    targets: list[Tensor] = []
    model.eval()
    with torch.inference_mode():
        for batch_index, batch in enumerate(loader):
            if batch_index >= calibration_batches:
                break
            tensors = _device_batch(batch, device)
            with torch.autocast(
                device_type=device.type,
                dtype=autocast_dtype or torch.bfloat16,
                enabled=autocast_dtype is not None,
            ):
                output = model(**_model_inputs(tensors))
            logits.append(output["logits"].float())
            targets.append(_tensor(tensors, "target").float())

    observed_logits  = torch.cat(logits)
    observed_targets = torch.cat(targets)
    prevalence       = float(observed_targets.mean())
    if prevalence <= 0.0 or prevalence >= 1.0:
        raise ValueError("local-head calibration requires both training classes")

    lower = -30.0
    upper = 30.0
    for _ in range(64):
        midpoint = 0.5 * (lower + upper)
        predicted = float(torch.sigmoid(observed_logits + midpoint).mean())
        if predicted < prevalence:
            lower = midpoint
        else:
            upper = midpoint
    calibrated = 0.5 * (lower + upper)
    local_head.bias.data.fill_(calibrated)
    return calibrated


def _evaluate(
    model         : torch.nn.Module,
    loader        : DataLoader[Any],
    device        : torch.device,
    autocast_dtype: torch.dtype | None,
    prediction_report: SurfacePredictionReport | None = None,
    faithfulness_audit: bool = False,
) -> tuple[dict[str, float | None], dict[str, float | None]]:
    """Evaluate one explicit split with definition-aware LambdaForge metrics.

    Args:
        model: Trained WISDOM model placed on ``device``.
        loader: Non-empty explicit-split graph DataLoader.
        device: CPU or CUDA device receiving tensors.
        autocast_dtype: CUDA mixed-precision dtype, or ``None`` for full float32.
        prediction_report: Optional split collector that persists point probabilities after the
            complete evaluation succeeds.
        faithfulness_audit: Compute ground-truth-free deletion/insertion diagnostics. This is
            intended for final restored checkpoints, not every training epoch.

    Returns:
        Protein metric mapping including mean binary cross-entropy loss, model-only forward time,
        throughput, and latency, plus an optional surface metric mapping. Mathematically undefined
        values remain ``None``.
    """
    logits            : list[Tensor] = []
    surface_protein_logits: list[Tensor] = []
    head_disagreements   : list[Tensor] = []
    targets           : list[Tensor] = []
    surface_scores    : list[Tensor] = []
    surface_targets   : list[Tensor] = []
    surface_validity  : list[Tensor] = []
    surface_owners    : list[Tensor] = []
    surface_areas     : list[Tensor] = []
    surface_bag_labels: list[Tensor] = []
    surface_attention : list[Tensor] = []
    atom_counts       : list[Tensor] = []
    surface_counts    : list[Tensor] = []
    global_phenotypes : list[str] = []
    interface_phenotypes: list[str] = []
    tiers             : list[str] = []
    attention_available = True
    faithfulness_sums : dict[str, float] = {}
    faithfulness_count = 0
    cpu_forward_seconds = 0.0
    cuda_forward_events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []

    protein_offset = 0
    mixed_dtype = (
        autocast_dtype if autocast_dtype in {torch.bfloat16, torch.float16} else None
    )

    model.eval()
    with torch.inference_mode():
        for batch in loader:
            tensors = _device_batch(batch, device)

            # CUDA events measure only enqueued model work without synchronizing every batch.
            # CPU evaluation uses the monotonic wall clock around the same forward boundary.

            if device.type == "cuda":
                event_factory = cast(Any, torch.cuda.Event)
                forward_start = event_factory(enable_timing=True)
                forward_stop  = event_factory(enable_timing=True)
                forward_start.record()
            else:
                cpu_forward_started = time.perf_counter()

            with torch.autocast(
                device_type=device.type,
                dtype=mixed_dtype or torch.bfloat16,
                enabled=mixed_dtype is not None,
            ):
                output = model(**_model_inputs(tensors))

            if device.type == "cuda":
                forward_stop.record()
                cuda_forward_events.append((forward_start, forward_stop))
            else:
                cpu_forward_seconds += time.perf_counter() - cpu_forward_started

            if prediction_report is not None:
                prediction_report.collect(batch, output)
            if faithfulness_audit:
                batch_faithfulness, batch_positive_count = SurfaceFaithfulnessAudit().compute(
                    model,
                    output,
                    tensors,
                )
                faithfulness_count += batch_positive_count
                for name, value in batch_faithfulness.items():
                    faithfulness_sums[name] = faithfulness_sums.get(name, 0.0) + value

            logits.append(output["logits"].float())
            if "surface_protein_logits" in output:
                surface_protein_logits.append(output["surface_protein_logits"].float())
                head_disagreements.append(output["head_disagreement"].float())
            protein_target = _tensor(batch, "target")
            targets.append(protein_target)

            # Retain only compact per-protein audit metadata. Point and atom counts are derived
            # from the actual collated tensors, so subgroup boundaries describe the representation
            # consumed by this run rather than stale catalog estimates.

            batch_size = len(protein_target)
            atom_counts.append(
                torch.bincount(_tensor(batch, "atom_batch"), minlength=batch_size).cpu()
            )
            surface_counts.append(
                torch.diff(_tensor(batch, "surface_ptr")).cpu()
            )
            for name, destination in (
                ("global_phenotype", global_phenotypes),
                ("interface_phenotype", interface_phenotypes),
                ("tier", tiers),
            ):
                values = batch.get(name)
                if values is None:
                    destination.extend("unspecified" for _ in range(batch_size))
                    continue
                if not isinstance(values, list) or len(values) != batch_size:
                    raise ValueError(f"evaluation batch field {name!r} must align with proteins")
                destination.extend(str(value) for value in values)

            if "surface_target_hard" in batch and "surface_valid_mask" in batch:
                surface_scores.append(output["surface_logits"].float().cpu())
                surface_targets.append(_tensor(batch, "surface_target_hard").cpu())
                surface_validity.append(_tensor(batch, "surface_valid_mask").cpu())
                surface_owners.append(_tensor(batch, "surface_batch").cpu() + protein_offset)
                surface_areas.append(_tensor(batch, "surface_area_weights").cpu())
                surface_bag_labels.append(protein_target)
                if "attention_weights" in output:
                    surface_attention.append(output["attention_weights"].float().cpu())
                else:
                    attention_available = False

            protein_offset += len(protein_target)

            del tensors, output

    if cuda_forward_events:
        torch.cuda.synchronize(device)
        forward_seconds = math.fsum(
            start.elapsed_time(stop) / 1000.0
            for start, stop in cuda_forward_events
        )
    else:
        forward_seconds = cpu_forward_seconds

    protein_logits  = torch.cat(logits).cpu()
    protein_targets = torch.cat(targets)
    protein_metrics = BinaryMetricSuite().compute(torch.sigmoid(protein_logits), protein_targets)
    protein_metrics["loss"] = float(
        F.binary_cross_entropy_with_logits(protein_logits, protein_targets)
    )
    protein_count = len(protein_targets)
    protein_metrics["forward_seconds"] = forward_seconds
    protein_metrics["forward_proteins_per_second"] = (
        protein_count / max(forward_seconds, 1.0e-9)
    )
    protein_metrics["forward_milliseconds_per_protein"] = (
        1000.0 * forward_seconds / protein_count
    )
    if surface_protein_logits:
        surface_global_logits = torch.cat(surface_protein_logits).cpu()
        surface_global_metrics = BinaryMetricSuite().compute(
            torch.sigmoid(surface_global_logits),
            protein_targets,
        )
        protein_metrics.update(
            {
                f"surface_derived_{name}": value
                for name, value in surface_global_metrics.items()
            }
        )
        protein_metrics["surface_derived_loss"] = float(
            F.binary_cross_entropy_with_logits(surface_global_logits, protein_targets)
        )
        protein_metrics["head_disagreement"] = float(torch.cat(head_disagreements).mean())
    if not surface_scores:
        faithfulness_metrics: dict[str, float | None] = (
            {
                name: value / faithfulness_count
                for name, value in faithfulness_sums.items()
            }
            if faithfulness_count
            else {}
        )
        return protein_metrics, faithfulness_metrics

    local_metrics = SurfaceMetricSuite().compute(
        torch.sigmoid(torch.cat(surface_scores)).cpu(),
        torch.cat(surface_targets),
        torch.cat(surface_validity),
        torch.cat(surface_owners),
        torch.cat(surface_bag_labels),
        torch.cat(surface_areas),
        torch.sigmoid(protein_logits),
        torch.cat(surface_attention) if attention_available and surface_attention else None,
        torch.cat(head_disagreements).cpu() if head_disagreements else None,
    )
    local_metrics.update(
        SubgroupMetricSuite().compute(
            torch.sigmoid(protein_logits),
            protein_targets,
            torch.cat(atom_counts),
            torch.cat(surface_counts),
            torch.sigmoid(torch.cat(surface_scores)),
            torch.cat(surface_targets),
            torch.cat(surface_validity),
            torch.cat(surface_owners),
            global_phenotypes,
            interface_phenotypes,
            tiers,
        )
    )
    if faithfulness_count:
        local_metrics.update(
            {
                name: value / faithfulness_count
                for name, value in faithfulness_sums.items()
            }
        )
    return protein_metrics, local_metrics


def _validation_utility(metrics: Mapping[str, float | None]) -> float | None:
    """Compute the frozen global checkpoint-selection score.

    The checkpoint must be selectable without surface ground truth. Its score is therefore
    ``G = 0.70 * ProteinAUPRC + 0.30 * ProteinAUROC``. Both terms are threshold-free, so a
    candidate cannot change checkpoint selection merely because its probabilities cross 0.5 at a
    different calibration. Surface quality influences architecture-level HPO only after this
    global-only checkpoint choice has been reproduced.

    Args:
        metrics: Protein-level validation metrics from one epoch.

    Returns:
        Global utility in ``[0,1]``, or ``None`` when a required metric is undefined.
    """
    auprc = metrics.get("auprc")
    auroc = metrics.get("auroc")
    if auprc is None or auroc is None:
        return None
    utility = 0.70 * float(auprc) + 0.30 * float(auroc)
    return min(1.0, max(0.0, utility))


def _surface_coupling(
    curve: Sequence[Mapping[str, float]],
) -> dict[str, float | None]:
    """Measure whether global checkpoint selection preserves surface localization quality.

    For observations ``(G_t,S_t)``, the globally selected epoch is ``argmax G_t`` and surface
    regret is ``max S_t - S_argmaxG``. Spearman correlation measures whether the ordering induced
    by global and surface quality agrees. The primary correlation discards the first 30% of the
    observed curve; the complete-curve value remains diagnostic. Regret is normalized against the
    frozen unacceptable gap ``R_cap=0.20`` before coupling. The frozen scores are
    ``C_R = 1 - min(regret / 0.20, 1)``,
    ``C = 0.70 C_R + 0.30(rho+1)/2`` and
    ``W = 0.35G + 0.45S + 0.20C``. When rank correlation is undefined, ``rho=0`` contributes a
    neutral 0.5 to the total score while the scientific correlation remains ``None``.

    Args:
        curve: Ordered epoch records containing finite ``epoch``, ``global``, and ``surface``
            values. Surface values are positive-protein macro AUPRC.

    Returns:
        Global/surface scores at the globally selected epoch, the independently best surface epoch
        and score, regret, both Spearman diagnostics, coupling, and the scalar WISDOM HPO score.

    Raises:
        ValueError: If the curve is empty or contains a non-finite/out-of-range score.
    """
    if not curve:
        raise ValueError("surface coupling requires at least one paired epoch")

    global_values  = [float(point["global"]) for point in curve]
    surface_values = [float(point["surface"]) for point in curve]
    if any(
        not math.isfinite(value) or not 0.0 <= value <= 1.0
        for value in (*global_values, *surface_values)
    ):
        raise ValueError("surface coupling scores must be finite values in [0,1]")

    global_index       = max(range(len(curve)), key=global_values.__getitem__)
    surface_best_index = max(range(len(curve)), key=surface_values.__getitem__)
    global_score       = global_values[global_index]
    surface_score      = surface_values[global_index]
    regret             = max(0.0, surface_values[surface_best_index] - surface_score)

    correlation_all = _spearman(global_values, surface_values)
    late_start      = math.ceil(0.30 * len(curve))
    correlation     = _spearman(global_values[late_start:], surface_values[late_start:])
    correlation_for_score = 0.0 if correlation is None else correlation
    regret_component      = 1.0 - min(regret / 0.20, 1.0)
    coupling = 0.70 * regret_component + 0.30 * ((correlation_for_score + 1.0) / 2.0)
    wisdom_score          = 0.35 * global_score + 0.45 * surface_score + 0.20 * coupling

    return {
        "wisdom_hpo_global":           global_score,
        "wisdom_hpo_surface":          surface_score,
        "wisdom_hpo_coupling":         coupling,
        "wisdom_hpo_score":            wisdom_score,
        "protein_global_score":        global_score,
        "surface_selected_score":      surface_score,
        "global_selected_epoch":       float(curve[global_index]["epoch"]),
        "surface_best_epoch":          float(curve[surface_best_index]["epoch"]),
        "surface_best_score":          surface_values[surface_best_index],
        "surface_selection_regret":    regret,
        "surface_regret_component":    regret_component,
        "global_surface_spearman":    correlation,
        "global_surface_spearman_all": correlation_all,
        "surface_coupling_score":      coupling,
    }


def _subgroup_coupling(
    surface_metrics: Mapping[str, float | None],
    curves         : dict[str, list[dict[str, float]]],
    epoch          : int,
) -> dict[str, float | None]:
    """Update subgroup trajectories and derive their independent WISDOM utilities.

    Each subgroup is selected by its own threshold-free protein score ``G`` and evaluated with its
    aligned positive-protein surface AUPRC ``S``. The same regret, late-training Spearman coupling,
    and frozen ``W`` formula used by the complete validation population are then applied. Groups
    lacking both global classes or locally evaluable positives remain absent instead of receiving a
    fabricated value.

    Args:
        surface_metrics: One validation pass containing paired ``*_global_score`` and
            ``*_surface_auprc`` subgroup metrics.
        curves: Mutable trajectory mapping retained across validation epochs.
        epoch: Positive epoch represented by this validation pass.

    Returns:
        Flat subgroup metrics for plotting: global and surface score at the subgroup-selected
        checkpoint, regret, coupling, and WISDOM score.
    """
    output: dict[str, float | None] = {}
    suffix = "_global_score"
    prefixes = sorted(
        name[:-len(suffix)]
        for name in surface_metrics
        if name.startswith("subgroup_") and name.endswith(suffix)
    )
    for prefix in prefixes:
        global_score  = surface_metrics.get(f"{prefix}_global_score")
        surface_score = surface_metrics.get(f"{prefix}_surface_auprc")
        if global_score is None or surface_score is None:
            continue

        curve = curves.setdefault(prefix, [])
        observation = {
            "epoch":   float(epoch),
            "global":  float(global_score),
            "surface": float(surface_score),
        }
        existing = next(
            (index for index, point in enumerate(curve) if int(point["epoch"]) == epoch),
            None,
        )
        if existing is None:
            curve.append(observation)
            curve.sort(key=lambda point: point["epoch"])
        else:
            curve[existing] = observation

        coupling = _surface_coupling(curve)
        renamed = {
            "global_at_selection":         coupling["wisdom_hpo_global"],
            "surface_at_global_selection": coupling["wisdom_hpo_surface"],
            "surface_regret":              coupling["surface_selection_regret"],
            "coupling":                    coupling["wisdom_hpo_coupling"],
            "wisdom_score":                coupling["wisdom_hpo_score"],
        }
        output.update({f"{prefix}_{name}": value for name, value in renamed.items()})
    return output


def _spearman(first: Sequence[float], second: Sequence[float]) -> float | None:
    """Return a finite Spearman rank correlation while preserving undefined cases.

    Args:
        first: First aligned numeric sequence.
        second: Second aligned numeric sequence.

    Returns:
        Correlation in ``[-1,1]`` or ``None`` for fewer than two observations or a constant rank.

    Raises:
        ValueError: If sequence lengths differ.
    """
    if len(first) != len(second):
        raise ValueError("Spearman inputs must be aligned")
    if len(first) < 2 or len(set(first)) < 2 or len(set(second)) < 2:
        return None

    value = float(spearmanr(first, second).statistic)
    return value if math.isfinite(value) else None


def _mcc_objective(mcc: float | None) -> float:
    """Convert optional MCC evidence into a total HPO utility component.

    A constant thresholded prediction makes the conventional MCC denominator zero, so
    ``BinaryMetricSuite`` correctly reports MCC as unavailable. HPO nevertheless needs one finite
    value from every candidate. This separate component assigns zero—the worst normalized
    utility—to that degenerate rule. It does not claim that the undefined scientific MCC equals
    ``-1`` or ``0``.

    Args:
        mcc: Defined Matthews correlation coefficient in ``[-1,1]``, or ``None`` when its
            denominator vanishes.

    Returns:
        MCC mapped linearly to ``[0,1]``, or zero for an undefined one-class prediction.
    """
    if mcc is None:
        return 0.0
    return min(1.0, max(0.0, (float(mcc) + 1.0) / 2.0))


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
) -> tuple[WisdomV1, dict[str, Any]]:
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

    accepted       = inspect.signature(model_class.__init__).parameters
    accepts_kwargs = any(
        parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in accepted.values()
    )
    parameters = {
        name: value
        for name, value in available_parameters.items()
        if accepts_kwargs or name in accepted
    }
    if model_version == 2:
        parameters.pop("surface_encoder_type", None)
        parameters.pop("surface_patch_size", None)
    model = cast(WisdomV1, model_class(**parameters))

    # V1 is a fixed scientific hypothesis, not a name that may silently resolve to the retired
    # dense surface-graph implementation. Keep this check at construction time so an accidental
    # source regression fails before loading an epoch of data or allocating CUDA activations.

    expected_surface_encoder = (
        isinstance(getattr(model, "surface_encoder", None), DiffusionSurfaceEncoder)
        if int(available_parameters.get("surface_layers", 2)) > 0
        else isinstance(getattr(model, "surface_encoder", None), torch.nn.Identity)
    )
    if model_version == 1 and (
        type(model) is not WisdomV1
        or getattr(model, "ARCHITECTURE_NAME", None) != "semantic-gated-diffusionnet"
        or getattr(model, "STRUCTURAL_SCHEMA_VERSION", None)
        != WisdomDataset.STRUCTURAL_SCHEMA_VERSION
        or not expected_surface_encoder
    ):
        raise RuntimeError(
            "WISDOM v1 must use schema-3 bounded topology and the semantic-gated DiffusionNet "
            "backbone"
        )

    return model, parameters


def _device_batch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    """Move model tensors to one device while retaining prefix boundaries on the host.

    Args:
        batch: Collated WISDOM graph mapping, which may also contain point-level diagnostics.
        device: Destination CPU or CUDA device.

    Returns:
        New mapping containing model tensors/operator packs and the global target on ``device``.
        ``surface_ptr`` stays on CPU because it supplies Python slice boundaries; moving it to
        CUDA would force one device synchronization for every protein. DNA sidecars, identifiers,
        tiers, and unused diagnostics also stay on the host.
    """
    selected_names = [*_MODEL_INPUT_NAMES, "target"]
    selected_names.extend(name for name in _V3_INPUT_NAMES if name in batch)
    selected_names.extend(name for name in _VECTOR_INPUT_NAMES if name in batch)
    return {
        name: batch[name] if name == "surface_ptr" else _move_to_device(batch[name], device)
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
    names.extend(name for name in _VECTOR_INPUT_NAMES if name in batch)
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
