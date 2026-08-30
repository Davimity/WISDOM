"""WISDOM v3 controlled molecular-surface encoder comparison."""

from __future__ import annotations

from typing import Any, ClassVar
from collections.abc import Mapping, Sequence

from torch import Tensor

from wisdom.models.WisdomV1 import WisdomV1
from wisdom.models.SurfaceEncoderType import SurfaceEncoderType
from wisdom.models.PTV3SurfaceEncoder import PTV3SurfaceEncoder
from wisdom.models.DMASIFSurfaceEncoder import DMASIFSurfaceEncoder
from wisdom.models.DeltaConvSurfaceEncoder import DeltaConvSurfaceEncoder
from wisdom.models.PointMambaSurfaceEncoder import PointMambaSurfaceEncoder


class WisdomV3(WisdomV1):
    """Hold atomic encoding, transfer, local head, and MAX pooling fixed across surface encoders."""

    output_schema: ClassVar[dict[str, Any]] = {
        "logits": "Tensor[B]",
        "surface_logits": "Tensor[M]",
    }

    def __init__(
        self,
        hidden_dim               : int = 128,
        embedding_dim            : int = 32,
        use_residue_type         : bool = True,
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
        surface_encoder_type     : SurfaceEncoderType | str = SurfaceEncoderType.DIFFUSION,
        surface_patch_size       : int = 64,
    ) -> None:
        """Construct one controlled v3 surface-propagation hypothesis.

        Args:
            hidden_dim: Fixed shared latent width.
            embedding_dim: Fixed atom-category embedding width.
            use_residue_type: Fixed residue-category feature switch.
            atomic_layers: Fixed atomic RGCN depth.
            projection_depth: Fixed atom/curvature projection depth.
            surface_layers: Depth shared by every compared surface encoder.
            dropout: Shared dropout probability.
            atomic_number_count: Element embedding table size.
            residue_type_count: Residue embedding table size.
            curvature_features: Dataset-derived flattened curvature width.
            atom_spatial_k: Fixed runtime atomic K.
            surface_atom_k: Fixed runtime transfer J.
            diffusion_spectral_modes: Fixed runtime spectral budget.
            surface_atom_radius: Transfer normalization radius in Å.
            surface_chunk_size: Transfer point chunk size.
            atomic_message_chunk_size: RGCN edge-message chunk size.
            surface_encoder_type: Diffusion, dMaSIF-like, DeltaConv, compact PTv3, or PointMamba.
            surface_patch_size: Maximum serialized PTv3 attention window size.

        Raises:
            ValueError: If the encoder type is unsupported.
        """
        super().__init__(
            hidden_dim                = hidden_dim,
            embedding_dim             = embedding_dim,
            use_residue_type          = use_residue_type,
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
        )
        try:
            self.surface_encoder_type = SurfaceEncoderType(surface_encoder_type)
        except ValueError as error:
            raise ValueError(
                f"unsupported WISDOM v3 surface encoder: {surface_encoder_type!r}"
            ) from error

        if self.surface_encoder_type is SurfaceEncoderType.DMASIF:
            self.surface_encoder = DMASIFSurfaceEncoder(hidden_dim, surface_layers, dropout)
        elif self.surface_encoder_type is SurfaceEncoderType.DELTACONV:
            self.surface_encoder = DeltaConvSurfaceEncoder(hidden_dim, surface_layers, dropout)
        elif self.surface_encoder_type is SurfaceEncoderType.PTV3:
            self.surface_encoder = PTV3SurfaceEncoder(
                hidden_dim,
                surface_layers,
                dropout,
                surface_patch_size,
            )
        elif self.surface_encoder_type is SurfaceEncoderType.POINT_MAMBA:
            self.surface_encoder = PointMambaSurfaceEncoder(hidden_dim, surface_layers, dropout)

    def encode_surface_features(
        self,
        features             : Tensor,
        surface_operators    : Sequence[Mapping[str, Tensor]],
        surface_ptr          : Tensor,
        surface_positions    : Tensor | None,
        surface_normals      : Tensor | None,
        surface_neighbors    : Tensor | None,
        surface_neighbor_mask: Tensor | None,
    ) -> Tensor:
        """Route identical initial surface features through one selected v3 encoder.

        Args:
            features: Initial surface features ``[M,H]``.
            surface_operators: Per-protein intrinsic operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            surface_positions: Required surface coordinates ``[M,3]``.
            surface_normals: Required outward normals ``[M,3]``.
            surface_neighbors: Required bounded global neighbor IDs ``[M,Ks]``.
            surface_neighbor_mask: Required neighbor validity mask ``[M,Ks]``.

        Returns:
            Surface embeddings ``[M,H]`` in unchanged point order.

        Raises:
            ValueError: If v3 geometry is absent.
        """
        if (
            surface_positions is None
            or surface_normals is None
            or surface_neighbors is None
            or surface_neighbor_mask is None
        ):
            raise ValueError("WISDOM v3 requires positions, normals, and bounded surface neighbors")

        if self.surface_encoder_type is SurfaceEncoderType.DIFFUSION:
            return self.surface_encoder(features, surface_operators, surface_ptr)
        if self.surface_encoder_type is SurfaceEncoderType.DMASIF:
            return self.surface_encoder(
                features,
                surface_positions,
                surface_normals,
                surface_neighbors,
                surface_neighbor_mask,
            )
        if self.surface_encoder_type is SurfaceEncoderType.DELTACONV:
            return self.surface_encoder(features, surface_operators, surface_ptr)
        return self.surface_encoder(features, surface_positions, surface_ptr)

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
    ) -> dict[str, Tensor]:
        """Predict with fixed MAX pooling after the selected v3 surface mechanism.

        Args:
            atomic_numbers: Element IDs ``[N]``.
            residue_type_ids: Residue IDs ``[N]``.
            atom_edge_index: Active atomic topology ``[2,Ea]``.
            atom_edge_types: Atomic relation IDs ``[Ea]``.
            surface_curvatures: Curvature invariants ``[M,S,3]``.
            surface_atom_neighbors: Atom IDs ``[M,J]``.
            surface_atom_distances: Atom distances ``[M,J]``.
            surface_atom_normal_offsets: Normal offsets ``[M,J]``.
            surface_atom_tangential_distances: Tangential magnitudes ``[M,J]``.
            surface_atom_mask: Valid atom-neighbor entries ``[M,J]``.
            surface_area_weights: Positive surface weights ``[M]``.
            surface_batch: Point-to-protein owners ``[M]``.
            surface_operators: Per-protein intrinsic operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            surface_positions: Surface coordinates ``[M,3]``.
            surface_normals: Outward normals ``[M,3]``.
            surface_neighbors: Bounded surface IDs ``[M,Ks]``.
            surface_neighbor_mask: Valid surface-neighbor mask ``[M,Ks]``.

        Returns:
            Protein logits ``[B]`` and surface logits ``[M]``.
        """
        protein_count = self.validate_surface_bags(
            surface_area_weights,
            surface_batch,
            len(surface_curvatures),
        )
        if (
            surface_positions is None
            or surface_normals is None
            or surface_neighbors is None
            or surface_neighbor_mask is None
        ):
            raise ValueError("WISDOM v3 requires all surface geometry tensors")
        _, surface_logits = self.encode_surface(
            atomic_numbers,
            residue_type_ids,
            atom_edge_index,
            atom_edge_types,
            surface_curvatures,
            surface_atom_neighbors,
            surface_atom_distances,
            surface_atom_normal_offsets,
            surface_atom_tangential_distances,
            surface_atom_mask,
            surface_operators,
            surface_ptr,
            surface_positions,
            surface_normals,
            surface_neighbors,
            surface_neighbor_mask,
        )
        protein_logits = self.global_max_pooling(
            surface_logits.unsqueeze(-1),
            surface_batch,
            protein_count,
        ).squeeze(-1)
        return {"logits": protein_logits, "surface_logits": surface_logits}
