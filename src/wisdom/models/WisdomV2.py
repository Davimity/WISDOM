"""Independent pooling and optional evidence refinement on the semantic-gated backbone."""

import torch

from torch import Tensor
from typing import Any, ClassVar
from lambdaforge.nn import Scatter
from collections.abc import Sequence
from wisdom.models.HeadType import HeadType
from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.PoolingType import PoolingType
from wisdom.models.GlobalSurfaceHeads import GlobalSurfaceHeads
from wisdom.models.ProteinPoolingHead import ProteinPoolingHead
from wisdom.models.refinement import build_surface_evidence_refiner
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext
from wisdom.models.refinement.SurfaceEvidenceRefinerType import SurfaceEvidenceRefinerType


class WisdomV2(WisdomV1):
    """Expose orthogonal pooling/refinement choices while retaining the shared backbone."""

    output_schema: ClassVar[dict[str, Any]] = {
        "logits":                      "Tensor[B]",
        "surface_logits":              "Tensor[M]",
        "raw_surface_logits":          "Tensor[M] before evidence refinement",
        "refined_surface_logits":      "Tensor[M] entering pooling",
        "raw_surface_probabilities":   "Tensor[M]",
        "refined_surface_probabilities": "Tensor[M]",
        "surface_probabilities":       "Tensor[M]",
        "localization_scores":         "Tensor[M]",
        "positive_area_fraction":      "Tensor[B]",
        "localization_entropy":        "Tensor[B]",
        "maximum_surface_probability": "Tensor[B]",
        "surface_protein_logits":       "Tensor[B] when a direct global head is active",
        "head_disagreement":            "Tensor[B] when a direct global head is active",
    }

    def __init__(
        self,
        pooling_type             : PoolingType | str = PoolingType.MAX,
        topk_fraction            : float = 0.05,
        attention_hidden_dim     : int = 32,
        regional_diffusion_scale: float = 2.5,
        log_sum_exp_beta         : float = 5.0,
        pooling_area_mode        : str = "legacy",
        attention_variant        : str = "simple",
        regional_scale_mode      : str = "fixed",
        regional_diffusion_scale_init: float = 1.5,
        log_sum_exp_mode          : str = "fixed",
        log_sum_exp_beta_init     : float = 5.0,
        autopool_alpha            : float = 1.0,
        autopool_alpha_mode       : str = "fixed",
        autopool_alpha_init       : float = 1.0,
        gem_power                 : float = 1.0,
        gem_power_mode            : str = "fixed",
        gem_power_init            : float = 4.0,
        max_mean_lambda           : float = 0.5,
        max_mean_lambda_mode      : str = "fixed",
        max_mean_lambda_init      : float = 0.5,
        regional_diffusion_scale_bounds: Sequence[float] = (0.05, 12.0),
        log_sum_exp_beta_bounds        : Sequence[float] = (0.25, 200.0),
        autopool_alpha_bounds          : Sequence[float] = (0.0, 50.0),
        gem_power_bounds               : Sequence[float] = (1.0, 32.0),
        surface_refiner_type                   : str   = "none",
        surface_refiner_heat_length            : float = 3.0,
        surface_refiner_heat_length_init       : float = 3.0,
        surface_refiner_strength               : float = 0.5,
        surface_refiner_steps                  : int   = 2,
        surface_refiner_geometry_sigma         : float = 2.0,
        surface_refiner_normal_sigma           : float = 0.25,
        surface_refiner_curvature_sigma        : float = 1.0,
        surface_refiner_embedding_temperature  : float = 0.5,
        surface_refiner_embedding_detach       : bool  = True,
        surface_refiner_learned_hidden_dim     : int   = 32,
        surface_refiner_tv_lambda              : float = 0.05,
        surface_refiner_tv_steps               : int   = 5,
        surface_refiner_tv_step_size           : float = 0.5,
        surface_refiner_tv_epsilon             : float = 0.1,
        surface_refiner_crf_strength           : float = 0.5,
        surface_refiner_crf_steps              : int   = 5,
        surface_refiner_crf_damping            : float = 0.5,
        head_type                : HeadType | str = HeadType.SINGLE,
        global_context_dim       : int = 16,
        detach_global_context    : bool = True,
        **backbone: Any,
    ) -> None:
        """Build the v1 backbone and one reusable multiple-instance pooling head.

        Args:
            pooling_type: MAX, mean, attention, top-k, local-mean/MAX, or log-sum-exp.
            topk_fraction: Fraction retained by top-k mean.
            attention_hidden_dim: Hidden width of learned attention scores.
            regional_diffusion_scale: Heat length in ångströms before regional MAX.
            log_sum_exp_beta: Positive normalized log-sum-exp inverse temperature.
            pooling_area_mode: point, area, or legacy (area mean, point other families).
            attention_variant: simple tanh or gated tanh-times-sigmoid attention.
            regional_scale_mode: fixed or learned regional heat length.
            regional_diffusion_scale_init: Learned length initialization inside its bounds, in Å.
            log_sum_exp_mode: fixed, learned, or curriculum LSE temperature.
            log_sum_exp_beta_init: Learned beta initialization strictly inside its bounds.
            autopool_alpha: Fixed non-negative probability attention sharpness.
            autopool_alpha_mode: fixed or learned AutoPool alpha.
            autopool_alpha_init: Learned alpha initialization strictly inside its bounds.
            gem_power: Fixed probability-space power >= 1.
            gem_power_mode: fixed or learned GeM power.
            gem_power_init: Learned power initialization strictly inside its bounds.
            max_mean_lambda: Fixed convex MAX-MEAN coefficient in [0, 1].
            max_mean_lambda_mode: fixed or learned mixture.
            max_mean_lambda_init: Learned coefficient initialization in (0, 1).
            regional_diffusion_scale_bounds: Positive learned length limits in Å; default
                (0.05, 12) preserves old checkpoints. Saved constructor values restore new limits.
            log_sum_exp_beta_bounds: Positive learned beta limits; default (0.25, 200).
            autopool_alpha_bounds: Nonnegative learned alpha limits; default (0, 50).
            gem_power_bounds: Learned power limits with lower at least one; default (1, 32).
            surface_refiner_type: Evidence operator: none, heat, learned_heat,
                geometric_anisotropic, embedding_anisotropic, learned_anisotropic, graph_tv or
                crf.
            surface_refiner_heat_length: Fixed heat length in Å; zero is exact identity.
            surface_refiner_heat_length_init: Learned heat initial length strictly inside
                (0.05,12) Å.
            surface_refiner_strength: Anisotropic convex mixing fraction in [0,1].
            surface_refiner_steps: Nonnegative anisotropic update count.
            surface_refiner_geometry_sigma: Positive spatial conductance bandwidth in Å.
            surface_refiner_normal_sigma: Positive dimensionless normal-misalignment bandwidth.
            surface_refiner_curvature_sigma: Positive signed-log curvature conductance bandwidth.
            surface_refiner_embedding_temperature: Positive cosine-distance bandwidth.
            surface_refiner_embedding_detach: Detach only embedding guides, not the
                logit/backbone gradient path.
            surface_refiner_learned_hidden_dim: Positive hidden width of the small learned
                symmetric edge scorer.
            surface_refiner_tv_lambda: Nonnegative coefficient of smoothed graph TV.
            surface_refiner_tv_steps: Nonnegative unrolled TV descent count.
            surface_refiner_tv_step_size: Relaxation in (0,1] of the stabilized diagonal descent
                step.
            surface_refiner_tv_epsilon: Positive smoothed-absolute-value scale in logit units.
            surface_refiner_crf_strength: Nonnegative attractive Potts coefficient.
            surface_refiner_crf_steps: Nonnegative mean-field iteration count.
            surface_refiner_crf_damping: New-proposal fraction in (0,1].
            head_type: ``single`` control, D1 ``dual``, D2 ``global_context``, or D2b ``film``.
            global_context_dim: D2/D2b bottleneck width.
            detach_global_context: Prevent local gradients from modifying the direct context path.
            **backbone: Exact gated V1 constructor values fixed from the winning V1 study.
        """
        super().__init__(**backbone)

        self.pooling_type = PoolingType(pooling_type)
        self.pooling_head = ProteinPoolingHead(
            hidden_dim               = self.surface_evidence_dim,
            pooling_type             = self.pooling_type,
            dropout                  = self.dropout_probability,
            topk_fraction            = topk_fraction,
            attention_hidden_dim     = attention_hidden_dim,
            regional_diffusion_scale = regional_diffusion_scale,
            log_sum_exp_beta         = log_sum_exp_beta,
            pooling_area_mode        = pooling_area_mode,
            attention_variant        = attention_variant,
            regional_scale_mode      = regional_scale_mode,
            regional_diffusion_scale_init = regional_diffusion_scale_init,
            log_sum_exp_mode          = log_sum_exp_mode,
            log_sum_exp_beta_init     = log_sum_exp_beta_init,
            autopool_alpha            = autopool_alpha,
            autopool_alpha_mode       = autopool_alpha_mode,
            autopool_alpha_init       = autopool_alpha_init,
            gem_power                 = gem_power,
            gem_power_mode            = gem_power_mode,
            gem_power_init            = gem_power_init,
            max_mean_lambda           = max_mean_lambda,
            max_mean_lambda_mode      = max_mean_lambda_mode,
            max_mean_lambda_init      = max_mean_lambda_init,
            regional_diffusion_scale_bounds = regional_diffusion_scale_bounds,
            log_sum_exp_beta_bounds         = log_sum_exp_beta_bounds,
            autopool_alpha_bounds           = autopool_alpha_bounds,
            gem_power_bounds                = gem_power_bounds,
        )
        self.head_type = HeadType(head_type)
        self.global_surface_heads = (
            GlobalSurfaceHeads(
                hidden_dim=self.surface_evidence_dim,
                head_type=self.head_type,
                context_dim=global_context_dim,
                dropout=self.dropout_probability,
                detach_context=detach_global_context,
            )
            if self.head_type is not HeadType.SINGLE
            else None
        )

        self.surface_refiner_type = SurfaceEvidenceRefinerType(surface_refiner_type)
        self.surface_refiner = build_surface_evidence_refiner(
            self.surface_refiner_type,
            self.surface_evidence_dim,
            surface_refiner_heat_length=surface_refiner_heat_length,
            surface_refiner_heat_length_init=surface_refiner_heat_length_init,
            surface_refiner_strength=surface_refiner_strength,
            surface_refiner_steps=surface_refiner_steps,
            surface_refiner_geometry_sigma=surface_refiner_geometry_sigma,
            surface_refiner_normal_sigma=surface_refiner_normal_sigma,
            surface_refiner_curvature_sigma=surface_refiner_curvature_sigma,
            surface_refiner_embedding_temperature=surface_refiner_embedding_temperature,
            surface_refiner_embedding_detach=surface_refiner_embedding_detach,
            surface_refiner_learned_hidden_dim=surface_refiner_learned_hidden_dim,
            surface_refiner_tv_lambda=surface_refiner_tv_lambda,
            surface_refiner_tv_steps=surface_refiner_tv_steps,
            surface_refiner_tv_step_size=surface_refiner_tv_step_size,
            surface_refiner_tv_epsilon=surface_refiner_tv_epsilon,
            surface_refiner_crf_strength=surface_refiner_crf_strength,
            surface_refiner_crf_steps=surface_refiner_crf_steps,
            surface_refiner_crf_damping=surface_refiner_crf_damping,
        )

    def forward(self, **inputs: Any) -> dict[str, Tensor]:  # type: ignore[override]
        """Encode, refine local evidence differentiably, then pool and expose diagnostics.

        Args:
            **inputs: Named tensors from ``WisdomCollator`` accepted by ``WisdomV1.encode_surface``
                plus surface area weights and point-to-protein owners. Graph refiners also
                receive existing positions, normals and masked neighbor tables.

        Returns:
            Protein logits and raw/refined point evidence, normalized localization weights,
            and per-protein summaries. Attention pooling additionally returns its point weights;
            refinement changes its evidence values, not the embedding-based weight network.
        """
        area_weights = inputs.pop("surface_area_weights")
        owners       = inputs.pop("surface_batch")
        operators    = inputs["surface_operators"]
        surface_ptr  = inputs["surface_ptr"]

        embeddings, surface_logits = self.encode_surface(**inputs)
        protein_count = len(surface_ptr) - 1
        head_output: dict[str, Tensor] = {}
        if self.global_surface_heads is not None:
            head_output = self.global_surface_heads(
                embeddings,
                area_weights,
                owners,
                protein_count,
            )
            surface_logits = head_output.get("conditioned_surface_logits", surface_logits)
        # Evidence refinement is inside the training forward: protein BCE differentiates
        # through this operator to the local head. Attention still scores unchanged embeddings.

        raw_surface_logits = surface_logits
        context = SurfaceEvidenceContext(
            embeddings, area_weights, owners, surface_ptr, operators,
            positions=inputs.get("surface_positions"),
            normals=inputs.get("surface_normals"),
            curvatures=inputs.get("surface_curvatures"),
            neighbors=inputs.get("surface_neighbors"),
            neighbor_mask=inputs.get("surface_neighbor_mask"),
        )
        surface_logits = self.refine_surface_logits(raw_surface_logits, context)
        pooled = self.pool_refined_surface_logits(
            surface_logits, embeddings, area_weights, owners, operators, surface_ptr
        )

        epsilon       = torch.finfo(surface_logits.dtype).eps
        area_sum      = Scatter.sum(area_weights, owners, protein_count)
        area          = area_weights / area_sum[owners].clamp_min(epsilon)
        localization  = Scatter.segment_softmax(
            surface_logits + area.clamp_min(epsilon).log(), owners, protein_count
        )
        probabilities = torch.sigmoid(surface_logits)
        positive_area = Scatter.sum(
            area * (probabilities >= 0.5).to(surface_logits.dtype), owners, protein_count
        )
        entropy = -Scatter.sum(
            localization * localization.clamp_min(epsilon).log(), owners, protein_count
        )
        counts  = Scatter.sum(surface_logits.new_ones(len(surface_logits)), owners, protein_count)
        entropy = torch.where(
            counts > 1.0,
            entropy / counts.log().clamp_min(epsilon),
            torch.zeros_like(entropy),
        )

        output = {
            "logits": pooled["logits"],
            "surface_logits": surface_logits,
            "raw_surface_logits": raw_surface_logits,
            "refined_surface_logits": surface_logits,
            "raw_surface_probabilities": (
                probabilities if raw_surface_logits is surface_logits
                else torch.sigmoid(raw_surface_logits)
            ),
            "refined_surface_probabilities": probabilities,
            "surface_embeddings": embeddings,
            "surface_evidence_features": embeddings,
            "surface_probabilities": probabilities,
            "localization_scores": localization,
            "positive_area_fraction": positive_area,
            "localization_entropy": entropy,
            "maximum_surface_probability": Scatter.maximum(
                probabilities, owners, protein_count
            ),
        }
        if "attention_weights" in pooled:
            output["attention_weights"] = pooled["attention_weights"]
        if "direct_logits" in head_output:
            output["surface_protein_logits"] = pooled["logits"]
            output["logits"]                 = head_output["direct_logits"]
            output["head_disagreement"] = torch.abs(
                torch.sigmoid(head_output["direct_logits"])
                - torch.sigmoid(pooled["logits"])
            )
        return output

    def refine_surface_logits(
        self, raw_surface_logits: Tensor, context: SurfaceEvidenceContext,
    ) -> Tensor:
        """Expose the differentiable evidence boundary independently of pooling.

        Args:
            raw_surface_logits: Local head evidence [M] before refinement.
            context: Existing geometry/representations with no GT fields.

        Returns:
            The operative [M] field entering pooling, identity for refiner=none.
        """
        return self.surface_refiner(raw_surface_logits, context)

    def pool_refined_surface_logits(
        self,
        surface_logits      : Tensor,
        surface_embeddings  : Tensor,
        surface_area_weights: Tensor,
        surface_batch       : Tensor,
        surface_operators   : Any,
        surface_ptr         : Tensor,
    ) -> dict[str, Tensor]:
        """Pool already refined evidence without applying the refiner a second time.

        Args:
            surface_logits: Local evidence ``[M]``.
            surface_embeddings: Surface features ``[M,H]``.
            surface_area_weights: Represented-area weights ``[M]``.
            surface_batch: Point-to-protein owners ``[M]``.
            surface_operators: Ordered per-protein diffusion operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.

        Returns:
            Protein logits and optional attention weights from ``ProteinPoolingHead``.
        """
        # Mixed-precision model forward passes may expose BF16 surface embeddings after leaving
        # their autocast context. The diagnostic pooling boundary is called later by the
        # faithfulness audit, where the attention scorer still owns FP32 parameters. Recompute
        # attention scores in the scorer's parameter dtype so PyTorch receives compatible matrix
        # operands without changing the trained weights or the stored local logits.

        if self.pooling_type is PoolingType.ATTENTION:
            parameter = next(self.pooling_head.parameters())
            surface_embeddings = surface_embeddings.to(dtype=parameter.dtype)

        return self.pooling_head(
            surface_logits,
            surface_embeddings,
            surface_area_weights,
            surface_batch,
            surface_operators,
            surface_ptr,
        )

    def pool_surface_logits(
        self,
        surface_logits      : Tensor,
        surface_embeddings  : Tensor,
        surface_area_weights: Tensor,
        surface_batch       : Tensor,
        surface_operators   : Any,
        surface_ptr         : Tensor,
        context             : SurfaceEvidenceContext | None = None,
    ) -> dict[str, Tensor]:
        """Apply the complete raw -> refiner -> pooling boundary for interventions.

        Args:
            surface_logits: Raw local-head evidence [M], not an already refined field.
            surface_embeddings: Existing point embeddings [M,H].
            surface_area_weights: Point areas [M].
            surface_batch: Protein owners [M].
            surface_operators: Original per-protein spectral packs.
            surface_ptr: Point prefix boundaries [B+1].
            context: Required original full-surface context for a nonidentity refiner.

        Returns:
            Protein logits and optional unchanged embedding-derived attention weights.

        Raises:
            ValueError: If refinement is active but its full context is unavailable.
        """
        if context is None:
            if self.surface_refiner_type is not SurfaceEvidenceRefinerType.NONE:
                raise ValueError("active evidence refinement requires the original surface context")
            context = SurfaceEvidenceContext(
                surface_embeddings, surface_area_weights, surface_batch,
                surface_ptr, surface_operators,
            )
        refined = self.refine_surface_logits(surface_logits, context)
        return self.pool_refined_surface_logits(
            refined, surface_embeddings, surface_area_weights, surface_batch,
            surface_operators, surface_ptr,
        )
