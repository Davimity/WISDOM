"""WISDOM v2 pooling study on the semantic-gated v1 backbone."""

from typing import Any, ClassVar

import torch
from lambdaforge.nn import Scatter
from torch import Tensor

from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.PoolingType import PoolingType
from wisdom.models.ProteinPoolingHead import ProteinPoolingHead


class WisdomV2(WisdomV1):
    """Relearn semantic gates while changing only protein-level pooling."""

    output_schema: ClassVar[dict[str, Any]] = {
        "logits":                      "Tensor[B]",
        "surface_logits":              "Tensor[M]",
        "surface_probabilities":       "Tensor[M]",
        "localization_scores":         "Tensor[M]",
        "positive_area_fraction":      "Tensor[B]",
        "localization_entropy":        "Tensor[B]",
        "maximum_surface_probability": "Tensor[B]",
    }

    def __init__(
        self,
        pooling_type             : PoolingType | str = PoolingType.MAX,
        topk_fraction            : float = 0.05,
        attention_hidden_dim     : int = 32,
        regional_diffusion_scale: float = 2.5,
        log_sum_exp_beta         : float = 5.0,
        **backbone: Any,
    ) -> None:
        """Build the v1 backbone and one reusable multiple-instance pooling head.

        Args:
            pooling_type: MAX, mean, attention, top-k, local-mean/MAX, or log-sum-exp.
            topk_fraction: Fraction retained by top-k mean.
            attention_hidden_dim: Hidden width of learned attention scores.
            regional_diffusion_scale: Heat length in ångströms before regional MAX.
            log_sum_exp_beta: Positive normalized log-sum-exp inverse temperature.
            **backbone: Exact gated V1 constructor values fixed from the winning V1 study.
        """
        super().__init__(**backbone)

        self.pooling_type = PoolingType(pooling_type)
        self.pooling_head = ProteinPoolingHead(
            hidden_dim               = self.hidden_dim,
            pooling_type             = self.pooling_type,
            dropout                  = self.dropout_probability,
            topk_fraction            = topk_fraction,
            attention_hidden_dim     = attention_hidden_dim,
            regional_diffusion_scale = regional_diffusion_scale,
            log_sum_exp_beta         = log_sum_exp_beta,
        )

    def forward(self, **inputs: Any) -> dict[str, Tensor]:  # type: ignore[override]
        """Encode one batch, apply the chosen pooling, and expose map diagnostics.

        Args:
            **inputs: Named tensors from ``WisdomCollator`` accepted by ``WisdomV1.encode_surface``
                plus surface area weights and point-to-protein owners.

        Returns:
            Protein/local logits, probabilities, normalized localization weights, and compact
            per-protein map summaries. Attention pooling additionally returns its point weights.
        """
        area_weights = inputs.pop("surface_area_weights")
        owners       = inputs.pop("surface_batch")
        operators    = inputs["surface_operators"]
        surface_ptr  = inputs["surface_ptr"]

        embeddings, surface_logits = self.encode_surface(**inputs)
        pooled = self.pooling_head(
            surface_logits, embeddings, area_weights, owners, operators, surface_ptr
        )

        protein_count = len(surface_ptr) - 1
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
        return output

    def pool_surface_logits(
        self,
        surface_logits      : Tensor,
        surface_embeddings  : Tensor,
        surface_area_weights: Tensor,
        surface_batch       : Tensor,
        surface_operators   : Any,
        surface_ptr         : Tensor,
    ) -> dict[str, Tensor]:
        """Expose the reusable pooling boundary for diagnostics and focused tests.

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
        return self.pooling_head(
            surface_logits,
            surface_embeddings,
            surface_area_weights,
            surface_batch,
            surface_operators,
            surface_ptr,
        )
