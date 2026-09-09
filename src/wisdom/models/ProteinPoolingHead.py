"""Reusable protein-level multiple-instance pooling for WISDOM v2 and v3."""

from collections.abc import Mapping, Sequence

import torch
from lambdaforge.nn import Scatter
from lambdaforge.nn.pooling import (
    FractionalTopKMeanPooling,
    LogSumExpPooling,
    SparseAttentionPooling,
    SparseMaxPooling,
)
from torch import Tensor, nn

from wisdom.models.PoolingType import PoolingType
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder


class ProteinPoolingHead(nn.Module):
    """Map point evidence to one protein logit using one explicit MIL hypothesis."""

    def __init__(
        self,
        hidden_dim             : int,
        pooling_type           : PoolingType | str,
        dropout                : float,
        topk_fraction          : float = 0.05,
        attention_hidden_dim   : int = 32,
        regional_diffusion_scale: float = 2.5,
        log_sum_exp_beta       : float = 5.0,
    ) -> None:
        """Build pooling-specific lightweight modules.

        Args:
            hidden_dim: Surface embedding width.
            pooling_type: MAX, mean, attention, top-k, local-mean/MAX, or log-sum-exp.
            dropout: Dropout used only by learned attention.
            topk_fraction: Fraction retained by top-k mean pooling.
            attention_hidden_dim: Hidden width of the attention scorer.
            regional_diffusion_scale: Heat length in ångströms before regional MAX.
            log_sum_exp_beta: Positive inverse temperature for normalized log-sum-exp.
        """
        super().__init__()
        self.pooling_type             = PoolingType(pooling_type)
        self.topk_fraction            = float(topk_fraction)
        self.regional_diffusion_scale = float(regional_diffusion_scale)
        self.sparse_max               = SparseMaxPooling()
        self.attention = (
            SparseAttentionPooling(hidden_dim, attention_hidden_dim, dropout=dropout)
            if self.pooling_type is PoolingType.ATTENTION
            else None
        )
        self.topk = FractionalTopKMeanPooling(fraction=topk_fraction)
        self.log_sum_exp = LogSumExpPooling(beta=log_sum_exp_beta, normalize=True)

    def forward(
        self,
        logits        : Tensor,
        embeddings    : Tensor,
        area_weights  : Tensor,
        owners        : Tensor,
        operators     : Sequence[Mapping[str, Tensor]],
        surface_ptr   : Tensor,
    ) -> dict[str, Tensor]:
        """Pool one ordered disjoint surface batch.

        Args:
            logits: Local evidence ``[M]``.
            embeddings: Surface embeddings ``[M,H]``.
            area_weights: Represented-area weights ``[M]``.
            owners: Point-to-protein indices ``[M]``.
            operators: Per-protein diffusion operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.

        Returns:
            Protein ``logits[B]`` and optional point attention weights.
        """
        protein_count = len(surface_ptr) - 1
        if self.pooling_type is PoolingType.MAX:
            return {"logits": self.sparse_max(logits[:, None], owners, protein_count)[:, 0]}

        area_sum = Scatter.sum(area_weights, owners, protein_count)
        area     = area_weights / area_sum[owners].clamp_min(torch.finfo(logits.dtype).eps)
        if self.pooling_type is PoolingType.MEAN:
            return {"logits": Scatter.sum(logits * area, owners, protein_count)}
        if self.pooling_type is PoolingType.ATTENTION:
            if self.attention is None:
                raise RuntimeError("attention pooling was not constructed")
            weights, _ = self.attention.weights(embeddings, owners, protein_count)
            return {
                "logits": Scatter.sum(logits * weights, owners, protein_count),
                "attention_weights": weights,
            }
        if self.pooling_type is PoolingType.LOCAL_MEAN_MAX:
            smoothed = DiffusionSurfaceEncoder.diffuse(
                logits, operators, surface_ptr, length=self.regional_diffusion_scale
            )
            return {"logits": self.sparse_max(smoothed[:, None], owners, protein_count)[:, 0]}

        counts    = torch.bincount(owners, minlength=protein_count)
        maximum   = int(counts.max())
        dense     = logits.new_zeros((protein_count, maximum, 1))
        mask      = torch.zeros((protein_count, maximum), dtype=torch.bool, device=logits.device)
        positions = torch.arange(len(logits), device=logits.device)
        starts    = torch.cumsum(counts, dim=0) - counts
        local     = positions - starts[owners]
        dense[owners, local, 0] = logits
        mask[owners, local]     = True

        pooled = (
            self.topk(dense, mask)
            if self.pooling_type is PoolingType.TOPK
            else self.log_sum_exp(dense, mask)
        )
        return {"logits": pooled[:, 0]}
