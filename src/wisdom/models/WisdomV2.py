"""WISDOM v2 controlled pooling hypotheses on the fixed DiffusionNet backbone."""

from __future__ import annotations

from typing import Any, ClassVar
from collections.abc import Mapping, Sequence

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
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder


class WisdomV2(WisdomV1):
    """Keep the complete v1 representation fixed while varying only protein pooling."""

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
        hidden_dim               : int = 128,
        embedding_dim            : int = 32,
        residue_embedding_dim    : int | None = None,
        use_residue_type         : bool = True,
        atom_feature_preset      : str = "legacy",
        use_element              : bool = True,
        use_formal_charge        : bool = False,
        use_aromaticity          : bool = False,
        use_hbond_donor          : bool = False,
        use_hbond_acceptor       : bool = False,
        use_hybridization        : bool = False,
        use_atom_role            : bool = False,
        use_residue_hydropathy   : bool = False,
        use_residue_polarity     : bool = False,
        atomic_layers            : int = 2,
        projection_depth         : int = 1,
        surface_layers           : int = 2,
        dropout                  : float = 0.2,
        atomic_number_count      : int = 119,
        residue_type_count       : int = 21,
        curvature_features       : int = 6,
        atom_spatial_k           : int = 16,
        surface_atom_k           : int = 16,
        diffusion_spectral_modes : int = 128,
        surface_atom_radius      : float = 6.0,
        surface_chunk_size       : int = 8192,
        atomic_message_chunk_size: int = 65536,
        transfer_geometry        : str = "full",
        surface_feature_mode     : str = "chemistry_geometry",
        use_mean_curvature       : bool = True,
        use_gaussian_curvature   : bool = True,
        use_curvedness           : bool = True,
        use_shape_index          : bool = False,
        pooling_type             : PoolingType | str = PoolingType.MAX,
        topk_fraction            : float = 0.05,
        attention_hidden_dim     : int = 32,
        regional_diffusion_scale : float = 2.5,
        log_sum_exp_beta         : float = 5.0,
    ) -> None:
        """Construct v1 exactly and select one multiple-instance pooling rule.

        Args:
            hidden_dim: Fixed reviewed v1 latent width.
            embedding_dim: Fixed reviewed v1 embedding width.
            residue_embedding_dim: Independent residue embedding width or the element width.
            use_residue_type: Fixed reviewed v1 residue-feature switch.
            atom_feature_preset: Coherent generic atom-feature family inherited from v1.
            use_element: Include element identity in a custom feature set.
            use_formal_charge: Include scaled formal charge.
            use_aromaticity: Include conservative aromatic identity.
            use_hbond_donor: Include conservative donor identity.
            use_hbond_acceptor: Include conservative acceptor identity.
            use_hybridization: Include graph-derived hybridization identity.
            use_atom_role: Include structural atom role.
            use_residue_hydropathy: Include normalized residue hydropathy.
            use_residue_polarity: Include coarse residue polarity.
            atomic_layers: Fixed reviewed v1 atomic depth.
            projection_depth: Fixed reviewed v1 pointwise projection depth.
            surface_layers: Fixed reviewed v1 DiffusionNet depth.
            dropout: Fixed reviewed v1 dropout.
            atomic_number_count: Element embedding table size.
            residue_type_count: Residue embedding table size.
            curvature_features: Dataset-derived flattened curvature width.
            atom_spatial_k: Fixed reviewed runtime K.
            surface_atom_k: Fixed reviewed runtime J.
            diffusion_spectral_modes: Fixed reviewed runtime spectral budget.
            surface_atom_radius: Transfer geometry normalization radius in Å.
            surface_chunk_size: Atom-transfer point chunk size.
            atomic_message_chunk_size: RGCN edge-message chunk size.
            transfer_geometry: Distance-only or full invariant transfer geometry.
            surface_feature_mode: Chemistry-only, geometry-only, or combined input.
            use_mean_curvature: Include mean curvature.
            use_gaussian_curvature: Include Gaussian curvature.
            use_curvedness: Include curvedness.
            use_shape_index: Include derived bounded shape index.
            pooling_type: Closed v2 pooling hypothesis.
            topk_fraction: Point fraction retained by top-k mean.
            attention_hidden_dim: Hidden width of learned attention scoring.
            regional_diffusion_scale: Physical diffusion length in Å before regional MAX.
            log_sum_exp_beta: Positive normalized log-sum-exp inverse temperature.

        Raises:
            ValueError: If a pooling name or pooling-specific value is invalid.
        """
        super().__init__(
            hidden_dim                = hidden_dim,
            embedding_dim             = embedding_dim,
            residue_embedding_dim     = residue_embedding_dim,
            use_residue_type          = use_residue_type,
            atom_feature_preset       = atom_feature_preset,
            use_element               = use_element,
            use_formal_charge         = use_formal_charge,
            use_aromaticity           = use_aromaticity,
            use_hbond_donor           = use_hbond_donor,
            use_hbond_acceptor        = use_hbond_acceptor,
            use_hybridization         = use_hybridization,
            use_atom_role             = use_atom_role,
            use_residue_hydropathy    = use_residue_hydropathy,
            use_residue_polarity      = use_residue_polarity,
            atomic_layers             = atomic_layers,
            projection_depth          = projection_depth,
            surface_layers            = surface_layers,
            dropout                   = dropout,
            atomic_number_count       = atomic_number_count,
            residue_type_count        = residue_type_count,
            curvature_features        = curvature_features,
            atom_spatial_k            = atom_spatial_k,
            surface_atom_k            = surface_atom_k,
            diffusion_spectral_modes  = diffusion_spectral_modes,
            surface_atom_radius       = surface_atom_radius,
            surface_chunk_size        = surface_chunk_size,
            atomic_message_chunk_size = atomic_message_chunk_size,
            transfer_geometry         = transfer_geometry,
            surface_feature_mode      = surface_feature_mode,
            use_mean_curvature        = use_mean_curvature,
            use_gaussian_curvature    = use_gaussian_curvature,
            use_curvedness            = use_curvedness,
            use_shape_index           = use_shape_index,
        )
        try:
            self.pooling_type = PoolingType(pooling_type)
        except ValueError as error:
            raise ValueError(f"unsupported WISDOM v2 pooling type: {pooling_type!r}") from error
        if not 0.0 < topk_fraction <= 1.0:
            raise ValueError("topk_fraction must lie in (0,1]")
        if (
            attention_hidden_dim < 1
            or regional_diffusion_scale <= 0.0
            or log_sum_exp_beta <= 0.0
        ):
            raise ValueError(
                "attention width, regional scale, and log-sum-exp beta must be positive"
            )

        self.topk_fraction            = float(topk_fraction)
        self.regional_diffusion_scale = float(regional_diffusion_scale)
        self.log_sum_exp_beta         = float(log_sum_exp_beta)
        self.sparse_max               = SparseMaxPooling()
        self.attention_pooling = (
            SparseAttentionPooling(hidden_dim, attention_hidden_dim, dropout=dropout)
            if self.pooling_type is PoolingType.ATTENTION
            else None
        )
        self.topk_pooling       = FractionalTopKMeanPooling(fraction=topk_fraction)
        self.log_sum_exp_pooling = LogSumExpPooling(beta=log_sum_exp_beta, normalize=True)

    def forward(
        self,
        atomic_numbers                    : Tensor,
        residue_type_ids                  : Tensor,
        atom_edge_index                   : Tensor,
        atom_edge_types                   : Tensor,
        surface_curvatures                : Tensor,
        surface_atom_neighbors            : Tensor,
        surface_atom_distances            : Tensor,
        surface_atom_normal_offsets       : Tensor,
        surface_atom_tangential_distances : Tensor,
        surface_atom_mask                 : Tensor,
        surface_area_weights              : Tensor,
        surface_batch                     : Tensor,
        surface_operators                 : Sequence[Mapping[str, Tensor]],
        surface_ptr                       : Tensor,
        surface_positions                 : Tensor | None = None,
        surface_normals                   : Tensor | None = None,
        surface_neighbors                 : Tensor | None = None,
        surface_neighbor_mask             : Tensor | None = None,
        atom_role_ids                     : Tensor | None = None,
        atom_hybridization_ids            : Tensor | None = None,
        formal_charges                    : Tensor | None = None,
        atom_aromaticity                  : Tensor | None = None,
        atom_hbond_donor                  : Tensor | None = None,
        atom_hbond_acceptor               : Tensor | None = None,
        residue_hydropathy                : Tensor | None = None,
        residue_polarity                  : Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Encode the fixed v1 backbone, pool local logits, and expose map diagnostics.

        Args:
            atomic_numbers: Element IDs ``[N]``.
            residue_type_ids: Residue IDs ``[N]``.
            atom_role_ids: Structural atom-role IDs ``[N]``.
            atom_hybridization_ids: Hybridization IDs ``[N]``.
            formal_charges: Scaled formal charges ``[N]``.
            atom_aromaticity: Aromatic flags ``[N]``.
            atom_hbond_donor: Donor flags ``[N]``.
            atom_hbond_acceptor: Acceptor flags ``[N]``.
            residue_hydropathy: Normalized residue hydropathy ``[N]``.
            residue_polarity: Polar-residue flags ``[N]``.
            atom_edge_index: Active bidirectional atomic topology ``[2,Ea]``.
            atom_edge_types: Atomic relations ``[Ea]``.
            surface_curvatures: Curvature invariants ``[M,S,3]``.
            surface_atom_neighbors: Atom IDs ``[M,J]``.
            surface_atom_distances: Atom distances ``[M,J]``.
            surface_atom_normal_offsets: Signed normal offsets ``[M,J]``.
            surface_atom_tangential_distances: Tangential magnitudes ``[M,J]``.
            surface_atom_mask: Valid transfer entries ``[M,J]``.
            surface_area_weights: Positive represented-area weights ``[M]``.
            surface_batch: Point-to-protein ownership ``[M]``.
            surface_operators: Per-protein intrinsic operator packs.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            surface_positions: Optional V3 positions; unused by v2.
            surface_normals: Optional V3 normals; unused by v2.
            surface_neighbors: Optional V3 neighbors; unused by v2.
            surface_neighbor_mask: Optional V3 neighbor mask; unused by v2.

        Returns:
            Protein/local logits and localization diagnostics.
        """
        protein_count = self.validate_surface_bags(
            surface_area_weights,
            surface_batch,
            len(surface_curvatures),
            len(surface_ptr) - 1,
        )
        embeddings, surface_logits = self.encode_surface(
            atomic_numbers                    = atomic_numbers,
            residue_type_ids                  = residue_type_ids,
            atom_edge_index                   = atom_edge_index,
            atom_edge_types                   = atom_edge_types,
            surface_curvatures                = surface_curvatures,
            surface_atom_neighbors            = surface_atom_neighbors,
            surface_atom_distances            = surface_atom_distances,
            surface_atom_normal_offsets       = surface_atom_normal_offsets,
            surface_atom_tangential_distances = surface_atom_tangential_distances,
            surface_atom_mask                 = surface_atom_mask,
            surface_operators                 = surface_operators,
            surface_ptr                       = surface_ptr,
            atom_role_ids                     = atom_role_ids,
            atom_hybridization_ids             = atom_hybridization_ids,
            formal_charges                    = formal_charges,
            atom_aromaticity                  = atom_aromaticity,
            atom_hbond_donor                  = atom_hbond_donor,
            atom_hbond_acceptor               = atom_hbond_acceptor,
            residue_hydropathy                = residue_hydropathy,
            residue_polarity                  = residue_polarity,
        )
        pooled = self.pool_surface_logits(
            surface_logits,
            embeddings,
            surface_area_weights,
            surface_batch,
            surface_operators,
            surface_ptr,
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
            "maximum_surface_probability": maximum_probability,
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
        surface_operators   : Sequence[Mapping[str, Tensor]],
        surface_ptr         : Tensor,
    ) -> dict[str, Tensor]:
        """Apply exactly one configured v2 pooling hypothesis.

        Args:
            surface_logits: Local logits ``[M]``.
            surface_embeddings: Fixed-backbone embeddings ``[M,H]``.
            surface_area_weights: Positive represented-area weights ``[M]``.
            surface_batch: Point-to-protein owners ``[M]``.
            surface_operators: Per-protein diffusion operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.

        Returns:
            Protein logits ``[B]`` plus optional learned attention weights ``[M]``.
        """
        point_count = len(surface_logits)
        if surface_logits.ndim != 1 or surface_embeddings.shape != (point_count, self.hidden_dim):
            raise ValueError("surface logits/embeddings have inconsistent shapes")
        protein_count = self.validate_surface_bags(
            surface_area_weights,
            surface_batch,
            point_count,
            len(surface_ptr) - 1,
        )
        if self.pooling_type is PoolingType.MAX:
            logits = self.sparse_max(surface_logits[:, None], surface_batch, protein_count)[:, 0]
            return {"logits": logits}

        area_sum        = Scatter.sum(surface_area_weights, surface_batch, protein_count)
        normalized_area = surface_area_weights / area_sum[surface_batch]
        if self.pooling_type is PoolingType.MEAN:
            return {
                "logits": Scatter.sum(
                    surface_logits * normalized_area,
                    surface_batch,
                    protein_count,
                )
            }
        if self.pooling_type is PoolingType.ATTENTION:
            if self.attention_pooling is None:
                raise ValueError("attention pooling module is unavailable")
            weights, _ = self.attention_pooling.weights(
                surface_embeddings,
                surface_batch,
                protein_count,
            )
            return {
                "logits": Scatter.sum(surface_logits * weights, surface_batch, protein_count),
                "attention_weights": weights,
            }
        if self.pooling_type is PoolingType.LOCAL_MEAN_MAX:
            smoothed = DiffusionSurfaceEncoder.diffuse_scalar(
                surface_logits,
                surface_operators,
                surface_ptr,
                time=self.regional_diffusion_scale**2,
            )
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
        """Pad scalar logits only for LambdaForge dense set-pooling primitives.

        Args:
            logits: Local values ``[M]``.
            surface_batch: Ordered owner IDs ``[M]``.
            protein_count: Non-empty protein bag count ``B``.

        Returns:
            Dense values ``[B,Mmax,1]`` and validity mask ``[B,Mmax]``.
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
