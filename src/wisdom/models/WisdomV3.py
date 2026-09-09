"""WISDOM v3 surface-encoder study with gated inputs and a fixed v2 pooling."""

from typing import Any
from collections.abc import Mapping, Sequence

from torch import Tensor

from wisdom.models.WisdomV2 import WisdomV2
from wisdom.models.SurfaceEncoderType import SurfaceEncoderType
from wisdom.models.PTV3SurfaceEncoder import PTV3SurfaceEncoder
from wisdom.models.DMASIFSurfaceEncoder import DMASIFSurfaceEncoder
from wisdom.models.DeltaConvSurfaceEncoder import DeltaConvSurfaceEncoder
from wisdom.models.PointMambaSurfaceEncoder import PointMambaSurfaceEncoder


class WisdomV3(WisdomV2):
    """Compare surface propagation while retaining the selected v2 pooling explicitly."""

    def __init__(
        self,
        surface_encoder_type: SurfaceEncoderType | str = SurfaceEncoderType.DIFFUSION,
        surface_patch_size  : int = 64,
        **v2_parameters     : Any,
    ) -> None:
        """Build the gated V2 model and replace only its surface encoder.

        Args:
            surface_encoder_type: DiffusionNet, dMaSIF-like, DeltaConv, PTv3, or PointMamba.
            surface_patch_size: Maximum compact PTv3 attention window.
            **v2_parameters: Fixed winning gated-backbone and pooling parameters from V1/V2.

        Raises:
            ValueError: If the encoder name is unsupported.
        """
        super().__init__(**v2_parameters)
        self.surface_encoder_type = SurfaceEncoderType(surface_encoder_type)

        if self.surface_encoder_type is SurfaceEncoderType.DMASIF:
            self.surface_encoder = DMASIFSurfaceEncoder(
                self.hidden_dim, self.surface_layers, self.dropout_probability
            )
        elif self.surface_encoder_type is SurfaceEncoderType.DELTACONV:
            self.surface_encoder = DeltaConvSurfaceEncoder(
                self.hidden_dim, self.surface_layers, self.dropout_probability
            )
        elif self.surface_encoder_type is SurfaceEncoderType.PTV3:
            self.surface_encoder = PTV3SurfaceEncoder(
                self.hidden_dim,
                self.surface_layers,
                self.dropout_probability,
                surface_patch_size,
            )
        elif self.surface_encoder_type is SurfaceEncoderType.POINT_MAMBA:
            self.surface_encoder = PointMambaSurfaceEncoder(
                self.hidden_dim, self.surface_layers, self.dropout_probability
            )

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
        """Route identical gated initial features through one selected surface encoder.

        Args:
            features: Initial point features ``[M,H]``.
            surface_operators: Per-protein intrinsic operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            surface_positions: Surface coordinates ``[M,3]``.
            surface_normals: Unit surface normals ``[M,3]``.
            surface_neighbors: Bounded point neighbours ``[M,K]``.
            surface_neighbor_mask: Valid-neighbour mask ``[M,K]``.

        Returns:
            Surface embeddings ``[M,H]`` in unchanged point order.

        Raises:
            ValueError: If a coordinate-based encoder lacks required geometry.
        """
        if self.surface_encoder_type is SurfaceEncoderType.DIFFUSION:
            return self.surface_encoder(features, surface_operators, surface_ptr)
        if self.surface_encoder_type is SurfaceEncoderType.DELTACONV:
            return self.surface_encoder(features, surface_operators, surface_ptr)
        if surface_positions is None or surface_normals is None:
            raise ValueError("the selected WISDOM v3 encoder requires positions and normals")
        if self.surface_encoder_type is SurfaceEncoderType.DMASIF:
            if surface_neighbors is None or surface_neighbor_mask is None:
                raise ValueError("dMaSIF requires bounded surface neighbours")
            return self.surface_encoder(
                features,
                surface_positions,
                surface_normals,
                surface_neighbors,
                surface_neighbor_mask,
            )
        return self.surface_encoder(features, surface_positions, surface_ptr)
