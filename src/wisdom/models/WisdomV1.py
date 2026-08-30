"""WISDOM v1 bounded atomic, geometric-transfer, and DiffusionNet backbone."""

from __future__ import annotations

from typing import Any, ClassVar
from collections.abc import Mapping, Sequence

import torch
from lambdaforge.nn.models import MLP, Model, RelationalGCN
from lambdaforge.nn.pooling import SparseMaxPooling
from torch import Tensor, nn

from wisdom.models.SurfaceAtomTransfer import SurfaceAtomTransfer
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder


class WisdomV1(Model):
    """Model discrete atomic chemistry, local transfer, and an intrinsic molecular surface."""

    # These identifiers make the fixed v1 scientific contract inspectable by Training and tests.

    ARCHITECTURE_NAME: ClassVar[str] = "bounded-atomic-diffusionnet"
    STRUCTURAL_SCHEMA_VERSION: ClassVar[str] = "3.0"

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
    ) -> None:
        """Build the v1 backbone while fixing protein aggregation to existential MAX.

        Atomic categories pass through LambdaForge's chunked relational GCN. Each surface point
        then gathers at most ``J`` atom embeddings through learned invariant geometric weights.
        Curvature and chemical context form scalar surface features for DiffusionNet; no persisted
        surface-radius graph and no complete ``[E_surface,H]`` message tensor exists.

        Args:
            hidden_dim: Shared atom/surface latent width ``H``.
            embedding_dim: Width of element and optional residue embeddings.
            use_residue_type: Include the learned residue-category embedding when true.
            atomic_layers: Relation-aware atomic message-passing depth.
            projection_depth: Pointwise chemistry/curvature projection depth.
            surface_layers: Number of DiffusionNet blocks.
            dropout: Shared dropout probability in ``[0,1)``.
            atomic_number_count: Element embedding table size.
            residue_type_count: Residue embedding table size including unknown ID zero.
            curvature_features: Flattened ``3*S`` curvature width derived from the dataset.
            atom_spatial_k: Runtime spatial rank budget recorded with this model.
            surface_atom_k: Runtime nearest-atom budget recorded with this model.
            diffusion_spectral_modes: Runtime low-frequency mode budget recorded with this model.
            surface_atom_radius: Physical transfer cutoff used to normalize geometric features.
            surface_chunk_size: Maximum points per atom-transfer activation chunk.
            atomic_message_chunk_size: Maximum atomic edges processed per RGCN message chunk.

        Raises:
            TypeError: If ``use_residue_type`` is not Boolean.
            ValueError: If a dimension/depth/budget/radius is non-positive or dropout is invalid.
        """
        super().__init__()

        dimensions = (
            hidden_dim,
            embedding_dim,
            atomic_layers,
            projection_depth,
            surface_layers,
            atomic_number_count,
            residue_type_count,
            curvature_features,
            atom_spatial_k,
            surface_atom_k,
            diffusion_spectral_modes,
            surface_chunk_size,
            atomic_message_chunk_size,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions
        ):
            raise ValueError("WISDOM dimensions, depths, and budgets must be positive integers")
        if not isinstance(use_residue_type, bool):
            raise TypeError("use_residue_type must be Boolean")
        if not 0.0 <= dropout < 1.0 or surface_atom_radius <= 0.0:
            raise ValueError("dropout or surface atom radius is invalid")

        self.atomic_number_embedding = nn.Embedding(atomic_number_count, embedding_dim)
        self.residue_type_embedding  = (
            nn.Embedding(residue_type_count, embedding_dim) if use_residue_type else None
        )

        atomic_input_width = embedding_dim * (2 if use_residue_type else 1)
        projection_width   = curvature_features + hidden_dim

        self.atomic_encoder = RelationalGCN(
            in_channels       = atomic_input_width,
            out_channels      = hidden_dim,
            num_relations     = 3,
            hidden_channels   = [hidden_dim] * (atomic_layers - 1),
            aggregation       = "mean",
            dropout           = dropout,
            residual          = True,
            message_chunk_size = atomic_message_chunk_size,
        )
        self.surface_atom_transfer = SurfaceAtomTransfer(
            hidden_dim = hidden_dim,
            radius     = surface_atom_radius,
            chunk_size = surface_chunk_size,
        )
        self.surface_projection = MLP(
            in_features = projection_width,
            out_features = hidden_dim,
            hidden      = [hidden_dim] * (projection_depth - 1),
            dropout     = dropout,
            residual    = True,
        )
        self.surface_encoder: nn.Module = DiffusionSurfaceEncoder(
            hidden_dim,
            layers  = surface_layers,
            dropout = dropout,
        )
        self.local_head         = nn.Linear(hidden_dim, 1)
        self.global_max_pooling = SparseMaxPooling()

        self.hidden_dim               = hidden_dim
        self.curvature_features       = curvature_features
        self.residue_type_count       = residue_type_count
        self.use_residue_type         = use_residue_type
        self.atomic_layers            = atomic_layers
        self.projection_depth         = projection_depth
        self.surface_layers           = surface_layers
        self.atom_spatial_k           = atom_spatial_k
        self.surface_atom_k           = surface_atom_k
        self.diffusion_spectral_modes = diffusion_spectral_modes
        self.dropout_probability      = float(dropout)

    def encode_surface(
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
        surface_operators                 : Sequence[Mapping[str, Tensor]],
        surface_ptr                       : Tensor,
        surface_positions                 : Tensor | None = None,
        surface_normals                   : Tensor | None = None,
        surface_neighbors                 : Tensor | None = None,
        surface_neighbor_mask             : Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode atomic chemistry and the intrinsic surface in exact point order.

        Args:
            atomic_numbers: Element IDs ``long [N]``.
            residue_type_ids: Residue IDs ``long [N]``.
            atom_edge_index: Active bidirectional atomic pairs ``long [2,Ea]``.
            atom_edge_types: Relation IDs ``0=spatial,1=covalent,2=both`` with shape ``[Ea]``.
            surface_curvatures: Invariant curvature values ``float [M,S,3]``.
            surface_atom_neighbors: Offset atom IDs ``long [M,J]`` with ``-1`` sentinels.
            surface_atom_distances: Atom distances ``float [M,J]`` in ångströms.
            surface_atom_normal_offsets: Signed normal offsets ``float [M,J]``.
            surface_atom_tangential_distances: Tangential magnitudes ``float [M,J]``.
            surface_atom_mask: Valid atom-neighbor mask ``bool [M,J]``.
            surface_operators: Ordered per-protein DiffusionNet operators.
            surface_ptr: Point prefix boundaries ``long [B+1]``.
            surface_positions: Optional V3 coordinates ``[M,3]``; unused by v1.
            surface_normals: Optional V3 normals ``[M,3]``; unused by v1.
            surface_neighbors: Optional V3 bounded surface IDs ``[M,Ks]``; unused by v1.
            surface_neighbor_mask: Optional V3 neighbor mask ``[M,Ks]``; unused by v1.

        Returns:
            Surface embeddings ``[M,H]`` and weakly supervised local logits ``[M]``.

        Raises:
            ValueError: If categories, relations, curvature width, endpoints, or encoder outputs
                disagree with the model contract.
        """
        if atomic_numbers.ndim != 1 or not len(atomic_numbers):
            raise ValueError("atomic_numbers must contain at least one value with shape [N]")
        if residue_type_ids.shape != atomic_numbers.shape:
            raise ValueError("residue_type_ids must have the same shape [N]")
        if atom_edge_index.ndim != 2 or atom_edge_index.shape[0] != 2:
            raise ValueError("atom_edge_index must have shape [2,Ea]")
        if atom_edge_types.shape != (atom_edge_index.shape[1],):
            raise ValueError("atom_edge_types must have shape [Ea]")
        if surface_curvatures.ndim != 3 or surface_curvatures.shape[2] != 3:
            raise ValueError("surface_curvatures must have shape [M,S,3]")

        atom_count    = len(atomic_numbers)
        surface_count = len(surface_curvatures)
        observed_width = int(surface_curvatures.shape[1] * 3)
        if surface_count == 0 or observed_width != self.curvature_features:
            raise ValueError(
                "flattened surface curvature width disagrees with model configuration: "
                f"observed={observed_width}, expected={self.curvature_features}"
            )
        if atomic_numbers.min() < 0 or (
            atomic_numbers.max() >= self.atomic_number_embedding.num_embeddings
        ):
            raise ValueError("atomic number is outside the configured embedding table")
        if residue_type_ids.min() < 0 or residue_type_ids.max() >= self.residue_type_count:
            raise ValueError("residue type is outside the configured embedding table")
        if atom_edge_types.numel() and (atom_edge_types.min() < 0 or atom_edge_types.max() > 2):
            raise ValueError("atom relation IDs must be 0=spatial, 1=covalent, or 2=both")
        if atom_edge_index.numel() and (
            atom_edge_index.min() < 0 or atom_edge_index.max() >= atom_count
        ):
            raise ValueError("atomic graph endpoint is out of range")

        atom_features = self.atomic_number_embedding(atomic_numbers)
        if self.residue_type_embedding is not None:
            atom_features = torch.cat(
                (atom_features, self.residue_type_embedding(residue_type_ids)),
                dim=1,
            )
        atom_embeddings = self.atomic_encoder(atom_features, atom_edge_index, atom_edge_types)

        atomic_context = self.surface_atom_transfer(
            atom_embeddings,
            surface_atom_neighbors,
            surface_atom_distances,
            surface_atom_normal_offsets,
            surface_atom_tangential_distances,
            surface_atom_mask,
        )
        curvature_features = surface_curvatures.flatten(start_dim=1)
        initial_surface = self.surface_projection(
            torch.cat((curvature_features, atomic_context), dim=1)
        )
        surface_embeddings = self.encode_surface_features(
            initial_surface,
            surface_operators,
            surface_ptr,
            surface_positions,
            surface_normals,
            surface_neighbors,
            surface_neighbor_mask,
        )
        if surface_embeddings.shape != (surface_count, self.hidden_dim):
            raise ValueError("surface encoder output must have shape [M,hidden_dim]")
        return surface_embeddings, self.local_head(surface_embeddings).squeeze(-1)

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
        """Apply the fixed v1 DiffusionNet surface mechanism.

        Args:
            features: Initial scalar surface features ``[M,H]``.
            surface_operators: Per-protein intrinsic operator packs.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            surface_positions: Optional coordinates reserved for v3; ignored by v1.
            surface_normals: Optional normals reserved for v3; ignored by v1.
            surface_neighbors: Optional bounded neighbors reserved for v3; ignored by v1.
            surface_neighbor_mask: Optional neighbor mask reserved for v3; ignored by v1.

        Returns:
            DiffusionNet embeddings ``[M,H]``.
        """
        del surface_positions, surface_normals, surface_neighbors, surface_neighbor_mask
        return self.surface_encoder(features, surface_operators, surface_ptr)

    def validate_surface_bags(
        self,
        surface_area_weights: Tensor,
        surface_batch       : Tensor,
        surface_count       : int,
    ) -> int:
        """Validate ordered non-empty protein bags and return their count.

        Args:
            surface_area_weights: Positive finite weights ``float [M]``.
            surface_batch: Ordered consecutive protein IDs ``long [M]``.
            surface_count: Expected point count ``M``.

        Returns:
            Number of protein bags ``B``.

        Raises:
            ValueError: If shapes, weights, or ordered ownership IDs are invalid.
        """
        if surface_count < 1 or surface_area_weights.shape != (surface_count,):
            raise ValueError("surface area weights must have non-empty shape [M]")
        if surface_batch.shape != (surface_count,):
            raise ValueError("surface_batch must have shape [M]")
        if not torch.isfinite(surface_area_weights).all() or torch.any(surface_area_weights <= 0):
            raise ValueError("surface area weights must be finite and positive")
        if surface_batch.min() != 0 or not torch.all(surface_batch[1:] >= surface_batch[:-1]):
            raise ValueError("surface_batch must contain ordered consecutive protein IDs")

        protein_count = int(surface_batch.max().item()) + 1
        if torch.any(torch.bincount(surface_batch, minlength=protein_count) == 0):
            raise ValueError("surface_batch must not skip protein IDs")
        return protein_count

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
    ) -> dict[str, Tensor]:
        """Predict one protein logit through MAX multiple-instance pooling.

        Args:
            atomic_numbers: Element IDs ``[N]``.
            residue_type_ids: Residue IDs ``[N]``.
            atom_edge_index: Active bidirectional atomic topology ``[2,Ea]``.
            atom_edge_types: Atomic relation IDs ``[Ea]``.
            surface_curvatures: Curvature invariants ``[M,S,3]``.
            surface_atom_neighbors: Atom IDs ``[M,J]``.
            surface_atom_distances: Atom distances ``[M,J]``.
            surface_atom_normal_offsets: Normal offsets ``[M,J]``.
            surface_atom_tangential_distances: Tangential magnitudes ``[M,J]``.
            surface_atom_mask: Valid transfer entries ``[M,J]``.
            surface_area_weights: Positive represented-area weights ``[M]``.
            surface_batch: Point-to-protein owners ``[M]``.
            surface_operators: Per-protein intrinsic operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.

        Returns:
            Protein logits ``[B]`` and surface logits ``[M]``.
        """
        protein_count = self.validate_surface_bags(
            surface_area_weights,
            surface_batch,
            len(surface_curvatures),
        )
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
        )
        protein_logits = self.global_max_pooling(
            surface_logits.unsqueeze(-1),
            surface_batch,
            protein_count,
        ).squeeze(-1)
        return {"logits": protein_logits, "surface_logits": surface_logits}
