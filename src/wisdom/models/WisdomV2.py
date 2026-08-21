"""WISDOM v2 with controlled protein-level pooling alternatives."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from lambdaforge.nn import Scatter
from lambdaforge.nn.pooling import (
    FractionalTopKMeanPooling,
    LogSumExpPooling,
    SparseAttentionPooling,
    SparseMaxPooling,
)
from torch import Tensor

from wisdom.models.PoolingType import PoolingType
from wisdom.models.WisdomV1 import WisdomV1


class WisdomV2(WisdomV1):
    """Keep the v1 backbone fixed while varying only the weak MIL pooling rule."""

    output_schema: ClassVar[dict[str, Any]] = {
        "logits": "Tensor[B]",
        "surface_logits": "Tensor[M]",
        "surface_probabilities": "Tensor[M]",
        "localization_scores": "Tensor[M]",
        "positive_area_fraction": "Tensor[B]",
        "localization_entropy": "Tensor[B]",
        "maximum_surface_probability": "Tensor[B]",
    }

    def __init__(
        self,
        hidden_dim          : int               = 128,
        embedding_dim       : int               = 32,
        use_residue_type    : bool              = True,
        atomic_layers       : int               = 2,
        projection_depth    : int               = 1,
        surface_layers      : int               = 2,
        dropout             : float             = 0.2,
        atomic_number_count : int               = 119,
        residue_type_count  : int               = 21,
        curvature_features  : int               = 6,
        pooling_type        : PoolingType | str = PoolingType.MAX,
        topk_fraction       : float             = 0.05,
        attention_hidden_dim: int               = 32,
        regional_levels     : int               = 1,
        log_sum_exp_beta    : float             = 5.0,
    ) -> None:
        """Construct the v1 encoder and exactly one configured pooling hypothesis.

        The graph encoders, feature families, local head, and optimization interface are inherited
        unchanged from v1. MAX is therefore an exact control. Mean uses represented surface area;
        attention learns normalized weights from surface embeddings; TOP-K averages a fixed
        fraction of local logits; local-mean/MAX smooths area-weighted graph neighborhoods before
        taking a maximum; and normalized log-sum-exp is a differentiable maximum approximation.

        Args:
            hidden_dim: Shared atomic and surface latent width.
            embedding_dim: Width of element and optional residue embeddings.
            use_residue_type: Include the residue-category atom embedding when true.
            atomic_layers: Relation-aware atomic graph-convolution count.
            projection_depth: Curvature/atom projection MLP depth.
            surface_layers: Surface graph-convolution count.
            dropout: Shared encoder/projection dropout probability.
            atomic_number_count: Element embedding table size.
            residue_type_count: Residue-category embedding table size.
            curvature_features: Flattened curvature input width.
            pooling_type: Closed v2 hypothesis from :class:`PoolingType`.
            topk_fraction: Fraction of points retained by TOP-K mean pooling.
            attention_hidden_dim: Hidden score width for sparse attention pooling.
            regional_levels: Number of graph-neighborhood smoothing rounds before local MAX.
            log_sum_exp_beta: Positive inverse temperature for normalized log-sum-exp.

        Raises:
            ValueError: If a pooling name or pooling-specific numeric bound is invalid.
            TypeError: Propagated from v1 for an invalid feature switch.
        """
        super().__init__(
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
            use_residue_type=use_residue_type,
            atomic_layers=atomic_layers,
            projection_depth=projection_depth,
            surface_layers=surface_layers,
            dropout=dropout,
            atomic_number_count=atomic_number_count,
            residue_type_count=residue_type_count,
            curvature_features=curvature_features,
        )
        try:
            self.pooling_type = PoolingType(pooling_type)
        except ValueError as error:
            raise ValueError(f"unsupported WISDOM v2 pooling type: {pooling_type!r}") from error
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must lie in (0,1]")
        if attention_hidden_dim < 1 or regional_levels < 1 or log_sum_exp_beta <= 0.0:
            raise ValueError(
                "attention width, regional levels, and log-sum-exp beta must be positive"
            )

        self.topk_fraction    = float(topk_fraction)
        self.regional_levels  = int(regional_levels)
        self.log_sum_exp_beta = float(log_sum_exp_beta)
        self.sparse_max       = SparseMaxPooling()
        self.attention_pooling = (
            SparseAttentionPooling(hidden_dim, attention_hidden_dim, dropout=dropout)
            if self.pooling_type is PoolingType.ATTENTION
            else None
        )
        self.topk_pooling = FractionalTopKMeanPooling(fraction=topk_fraction)
        self.log_sum_exp_pooling = LogSumExpPooling(beta=log_sum_exp_beta, normalize=True)

    def forward(
        self,
        atomic_numbers          : Tensor,
        residue_type_ids        : Tensor,
        atom_edge_index         : Tensor,
        atom_edge_types         : Tensor,
        surface_curvatures      : Tensor,
        surface_edge_index      : Tensor,
        surface_atom_edge_index : Tensor,
        surface_area_weights    : Tensor,
        surface_batch           : Tensor,
    ) -> dict[str, Tensor]:
        """Encode the fixed graphs, pool local logits, and expose map diagnostics.

        Args:
            atomic_numbers: Integer element IDs with shape ``[N]``.
            residue_type_ids: Integer residue categories with shape ``[N]``.
            atom_edge_index: Bidirectional atomic edges with shape ``[2,Ea]``.
            atom_edge_types: R-GCN relations with shape ``[Ea]``.
            surface_curvatures: Multiscale invariants with shape ``[M,S,3]``.
            surface_edge_index: Bidirectional surface edges with shape ``[2,Es]``.
            surface_atom_edge_index: Surface-to-atom incidence with shape ``[2,Esa]``.
            surface_area_weights: Positive represented-area weights with shape ``[M]``.
            surface_batch: Ordered point-to-protein owner IDs with shape ``[M]``.

        Returns:
            Protein/local logits, normalized localization scores, and area/entropy diagnostics.

        Raises:
            ValueError: If inherited graph validation or pooling output validation fails.
        """
        protein_count = self.validate_surface_bags(
            surface_area_weights,
            surface_batch,
            len(surface_curvatures),
        )
        embeddings, surface_logits = self.encode_surface(
            atomic_numbers=atomic_numbers,
            residue_type_ids=residue_type_ids,
            atom_edge_index=atom_edge_index,
            atom_edge_types=atom_edge_types,
            surface_curvatures=surface_curvatures,
            surface_edge_index=surface_edge_index,
            surface_atom_edge_index=surface_atom_edge_index,
        )
        pooled = self.pool_surface_logits(
            surface_logits,
            embeddings,
            surface_edge_index,
            surface_area_weights,
            surface_batch,
        )

        epsilon         = torch.finfo(surface_logits.dtype).eps
        area_sum        = Scatter.sum(surface_area_weights, surface_batch, protein_count)
        normalized_area = surface_area_weights / area_sum[surface_batch].clamp_min(epsilon)
        localization = Scatter.segment_softmax(
            surface_logits + normalized_area.clamp_min(epsilon).log(),
            surface_batch,
            protein_count,
        )
        probabilities = torch.sigmoid(surface_logits)
        positive_area = Scatter.sum(
            normalized_area * (probabilities >= 0.5).to(surface_logits.dtype),
            surface_batch,
            protein_count,
        )
        maximum_probability = Scatter.maximum(probabilities, surface_batch, protein_count)
        entropy = -Scatter.sum(
            localization * localization.clamp_min(epsilon).log(),
            surface_batch,
            protein_count,
        )
        counts = Scatter.sum(
            surface_logits.new_ones(len(surface_logits)),
            surface_batch,
            protein_count,
        )
        entropy_scale = counts.log().clamp_min(epsilon)
        entropy = torch.where(counts > 1.0, entropy / entropy_scale, torch.zeros_like(entropy))

        output = {
            "logits": pooled["logits"],
            "surface_logits": surface_logits,
            "surface_probabilities": probabilities,
            "localization_scores": localization,
            "positive_area_fraction": positive_area,
            "localization_entropy": entropy,
            "maximum_surface_probability": maximum_probability,
        }
        if "attention_weights" in pooled:
            output["attention_weights"] = pooled["attention_weights"]
        return output

    def pool_surface_logits(
        self,
        surface_logits       : Tensor,
        surface_embeddings   : Tensor,
        surface_edge_index   : Tensor,
        surface_area_weights : Tensor,
        surface_batch        : Tensor,
    ) -> dict[str, Tensor]:
        """Apply the configured v2 pooling to already encoded surface instances.

        Args:
            surface_logits: Local scalar logits with shape ``[M]``.
            surface_embeddings: Fixed-backbone embeddings with shape ``[M,H]``.
            surface_edge_index: Bidirectional surface graph with shape ``[2,E]``.
            surface_area_weights: Positive represented-area weights with shape ``[M]``.
            surface_batch: Ordered point-to-protein owners with shape ``[M]``.

        Returns:
            Protein logits ``[B]`` and attention weights ``[M]`` when attention is selected.

        Raises:
            ValueError: If the tensor shapes or selected attention module are inconsistent.
        """
        point_count = len(surface_logits)
        if surface_logits.ndim != 1 or surface_embeddings.shape != (point_count, self.hidden_dim):
            raise ValueError("surface logits/embeddings have inconsistent shapes")
        protein_count = self.validate_surface_bags(
            surface_area_weights,
            surface_batch,
            point_count,
        )
        if self.pooling_type is PoolingType.MAX:
            logits = self.sparse_max(surface_logits[:, None], surface_batch, protein_count)[:, 0]
            return {"logits": logits}

        area_sum        = Scatter.sum(surface_area_weights, surface_batch, protein_count)
        normalized_area = surface_area_weights / area_sum[surface_batch]
        if self.pooling_type is PoolingType.MEAN:
            logits = Scatter.sum(
                surface_logits * normalized_area,
                surface_batch,
                protein_count,
            )
            return {"logits": logits}
        if self.pooling_type is PoolingType.ATTENTION:
            if self.attention_pooling is None:
                raise ValueError("attention pooling module is unavailable")
            weights, _ = self.attention_pooling.weights(
                surface_embeddings,
                surface_batch,
                protein_count,
            )
            logits = Scatter.sum(surface_logits * weights, surface_batch, protein_count)
            return {"logits": logits, "attention_weights": weights}
        if self.pooling_type is PoolingType.LOCAL_MEAN_MAX:
            smoothed = surface_logits
            source   = surface_edge_index[0]
            target   = surface_edge_index[1]
            for _ in range(self.regional_levels):
                numerator = Scatter.sum(
                    smoothed[source] * surface_area_weights[source],
                    target,
                    point_count,
                ) + smoothed * surface_area_weights
                denominator = Scatter.sum(
                    surface_area_weights[source],
                    target,
                    point_count,
                ) + surface_area_weights
                smoothed = numerator / denominator
            logits = self.sparse_max(smoothed[:, None], surface_batch, protein_count)[:, 0]
            return {"logits": logits}

        dense, mask = self._dense_logits(surface_logits, surface_batch, protein_count)
        if self.pooling_type is PoolingType.TOPK:
            logits = self.topk_pooling(dense, mask)[:, 0]
        else:
            logits = self.log_sum_exp_pooling(dense, mask)[:, 0]
        return {"logits": logits}

    @staticmethod
    def _dense_logits(
        logits       : Tensor,
        surface_batch: Tensor,
        protein_count: int,
    ) -> tuple[Tensor, Tensor]:
        """Pad only scalar logits for LambdaForge dense set-pooling operators.

        Args:
            logits: Local values with shape ``[M]``.
            surface_batch: Ordered owner IDs with shape ``[M]``.
            protein_count: Number of non-empty protein bags.

        Returns:
            Dense logits ``[B,M_max,1]`` and Boolean validity mask ``[B,M_max]``.
        """
        counts    = torch.bincount(surface_batch, minlength=protein_count)
        starts    = torch.cumsum(counts, dim=0) - counts
        local_ids = torch.arange(len(logits), device=logits.device) - starts[surface_batch]
        maximum   = int(counts.max().item())
        dense     = logits.new_zeros((protein_count, maximum, 1))
        mask      = torch.zeros(
            (protein_count, maximum),
            dtype=torch.bool,
            device=logits.device,
        )
        dense[surface_batch, local_ids, 0] = logits
        mask[surface_batch, local_ids]     = True
        return dense, mask
