"""WISDOM v1 semantic-gated atomic and DiffusionNet surface backbone."""

from __future__ import annotations

from typing import Any, ClassVar
from collections.abc import Mapping, Sequence

import torch
from lambdaforge.nn.models import MLP, Model
from lambdaforge.nn.pooling import SparseMaxPooling
from torch import Tensor, nn

from wisdom.models.GatedAtomicEncoder import GatedAtomicEncoder
from wisdom.models.SurfaceAtomTransfer import SurfaceAtomTransfer
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.gating.SemanticGateRegistry import SemanticGateRegistry


class WisdomV1(Model):
    """Learn which generic physical inputs a fixed DiffusionNet/MAX backbone needs."""

    ARCHITECTURE_NAME: ClassVar[str] = "semantic-gated-diffusionnet"
    STRUCTURAL_SCHEMA_VERSION: ClassVar[str] = "3.0"

    output_schema: ClassVar[dict[str, Any]] = {
        "logits":         "Tensor[B]",
        "surface_logits": "Tensor[M]",
    }

    def __init__(
        self,
        hidden_dim                 : int = 128,
        embedding_dim              : int = 32,
        residue_embedding_dim      : int | None = None,
        atomic_layers              : int = 2,
        projection_depth           : int = 1,
        surface_layers             : int = 2,
        dropout                    : float = 0.2,
        atomic_number_count        : int = 119,
        residue_type_count         : int = 21,
        curvature_features         : int = 6,
        atom_spatial_k             : int = 16,
        surface_atom_k             : int = 16,
        diffusion_spectral_modes   : int = 128,
        surface_atom_radius        : float = 6.0,
        surface_chunk_size         : int = 8192,
        atomic_message_chunk_size  : int = 65536,
        atomic_edge_distance_scale : float = 6.0,
    ) -> None:
        """Build the complete semantic input superset and fixed v1 hypothesis.

        Element identity is always active. Every other generic atom descriptor, each physical
        graph branch, its predictive edge enrichments, transfer orientation/content, the complete
        atom context, and every curvature descriptor at every stored scale receive a global
        Hard-Concrete gate. Tensor widths remain fixed; a closed gate writes zeros rather than
        rebuilding the network.

        Args:
            hidden_dim: Shared atom/surface latent width ``H``.
            embedding_dim: Element embedding width.
            residue_embedding_dim: Residue embedding width; ``None`` uses ``embedding_dim``.
            atomic_layers: Positive two-branch atomic message-passing depth.
            projection_depth: Positive surface projection MLP depth.
            surface_layers: Positive DiffusionNet block count.
            dropout: Shared dropout probability in ``[0,1)``.
            atomic_number_count: Element vocabulary size.
            residue_type_count: Residue vocabulary size including unknown zero.
            curvature_features: Persisted flattened ``3*S`` curvature width; it determines ``S``.
            atom_spatial_k: Runtime spatial-neighbour budget recorded with the model.
            surface_atom_k: Runtime atom-to-surface neighbour budget.
            diffusion_spectral_modes: Runtime spectral mode budget.
            surface_atom_radius: Transfer cutoff and distance normalization radius in ångströms.
            surface_chunk_size: Maximum points per transfer activation chunk.
            atomic_message_chunk_size: Recorded edge-work budget for reproducibility.
            atomic_edge_distance_scale: Atomic edge-distance normalization scale in ångströms.

        Raises:
            ValueError: If a dimension is invalid or curvature width is not ``3*S``.
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
            atom_spatial_k,
            surface_atom_k,
            diffusion_spectral_modes,
            surface_chunk_size,
            atomic_message_chunk_size,
        )
        if any(isinstance(value, bool) or value < 1 for value in dimensions):
            raise ValueError("WISDOM dimensions, depths, and budgets must be positive integers")
        if curvature_features < 3 or curvature_features % 3:
            raise ValueError("curvature_features must equal three times the stored scale count")
        if not 0.0 <= dropout < 1.0 or min(surface_atom_radius, atomic_edge_distance_scale) <= 0.0:
            raise ValueError("dropout or a physical distance scale is invalid")

        residue_width = embedding_dim if residue_embedding_dim is None else residue_embedding_dim
        if residue_width < 1:
            raise ValueError("residue_embedding_dim must be positive")

        self.curvature_scale_count = curvature_features // 3
        gate_definitions = self._gate_definitions(self.curvature_scale_count)
        self.semantic_gates = SemanticGateRegistry(gate_definitions)

        self.atomic_number_embedding = nn.Embedding(atomic_number_count, embedding_dim)
        self.residue_type_embedding   = nn.Embedding(residue_type_count, residue_width)
        self.atom_role_embedding      = nn.Embedding(7, 8)
        self.hybridization_embedding  = nn.Embedding(4, 4)

        atomic_input_width = embedding_dim + residue_width + 8 + 4 + 6
        self.atomic_encoder = GatedAtomicEncoder(
            input_dim      = atomic_input_width,
            hidden_dim     = hidden_dim,
            layers         = atomic_layers,
            dropout        = dropout,
            distance_scale = atomic_edge_distance_scale,
            message_chunk_size = atomic_message_chunk_size,
        )
        self.surface_atom_transfer = SurfaceAtomTransfer(
            hidden_dim = hidden_dim,
            radius     = surface_atom_radius,
            chunk_size = surface_chunk_size,
        )
        surface_input_width = hidden_dim + 4 * self.curvature_scale_count
        self.surface_projection = MLP(
            in_features  = surface_input_width,
            out_features = hidden_dim,
            hidden       = [hidden_dim] * (projection_depth - 1),
            dropout      = dropout,
            residual     = True,
        )
        self.surface_encoder: nn.Module = DiffusionSurfaceEncoder(
            hidden_dim,
            layers  = surface_layers,
            dropout = dropout,
        )
        self.local_head         = nn.Linear(hidden_dim, 1)
        self.global_max_pooling = SparseMaxPooling()

        self.hidden_dim                  = hidden_dim
        self.curvature_features          = curvature_features
        self.atomic_layers               = atomic_layers
        self.projection_depth            = projection_depth
        self.surface_layers              = surface_layers
        self.atom_spatial_k              = atom_spatial_k
        self.surface_atom_k              = surface_atom_k
        self.diffusion_spectral_modes    = diffusion_spectral_modes
        self.atomic_message_chunk_size   = atomic_message_chunk_size
        self.dropout_probability         = float(dropout)
        self.atomic_input_width          = atomic_input_width
        self.evidence_input_width        = hidden_dim
        self.evidence_head_parameter_count = hidden_dim + 1
        self.evidence_head_added_parameter_count = 0

    @staticmethod
    def _gate_definitions(scale_count: int) -> tuple[tuple[str, str | None], ...]:
        """Define stable gate names and the hierarchy used only by the L0 cost.

        Args:
            scale_count: Number of persisted curvature scales.

        Returns:
            Ordered ``(name,parent)`` pairs; every parent precedes its children.
        """
        atom_parent = "surface.atom_context"
        definitions: list[tuple[str, str | None]] = [(atom_parent, None)]
        definitions.extend(
            (f"atom.{name}", atom_parent)
            for name in (
                "residue_type",
                "formal_charge",
                "aromaticity",
                "hbond_donor",
                "hbond_acceptor",
                "hybridization",
                "atom_role",
                "residue_hydropathy",
                "residue_polarity",
            )
        )
        for branch in ("spatial", "covalent"):
            definitions.append((f"atomic_graph.{branch}", atom_parent))

        definitions.extend(
            (
                ("edge.spatial.distance", "atomic_graph.spatial"),
                ("edge.spatial.same_residue", "atomic_graph.spatial"),
                ("edge.spatial.same_chain", "atomic_graph.spatial"),
                ("edge.spatial.residue_separation", "atomic_graph.spatial"),
                ("edge.covalent.distance", "atomic_graph.covalent"),
                ("edge.covalent.bond_order", "atomic_graph.covalent"),
                ("edge.covalent.same_residue", "atomic_graph.covalent"),
                ("transfer.orientation", atom_parent),
                ("transfer.atom_content", atom_parent),
            )
        )
        for descriptor in ("mean", "gaussian", "curvedness", "shape_index"):
            definitions.extend(
                (f"surface.curvature.{descriptor}.scale_{index}", None)
                for index in range(scale_count)
            )
        return tuple(definitions)

    def gate_regularization(self) -> Tensor:
        """Return the normalized expected L0 semantic-path cost in ``[0,1]``."""
        return self.semantic_gates.regularization()

    def gate_summary(self) -> dict[str, object]:
        """Return JSON-compatible learned gate probabilities and deterministic values."""
        return self.semantic_gates.summary()

    def encode_surface(
        self,
        atomic_numbers                    : Tensor,
        residue_type_ids                  : Tensor,
        atom_edge_index                   : Tensor,
        atom_edge_is_spatial              : Tensor,
        atom_edge_is_covalent             : Tensor,
        atom_edge_distance                : Tensor,
        atom_edge_bond_order              : Tensor,
        atom_edge_same_residue            : Tensor,
        atom_edge_same_chain              : Tensor,
        atom_edge_residue_separation      : Tensor,
        surface_curvatures                : Tensor,
        surface_atom_neighbors            : Tensor,
        surface_atom_distances            : Tensor,
        surface_atom_normal_offsets       : Tensor,
        surface_atom_tangential_distances : Tensor,
        surface_atom_mask                 : Tensor,
        surface_operators                 : Sequence[Mapping[str, Tensor]],
        surface_ptr                       : Tensor,
        atom_role_ids                     : Tensor,
        atom_hybridization_ids            : Tensor,
        formal_charges                    : Tensor,
        atom_aromaticity                  : Tensor,
        atom_hbond_donor                  : Tensor,
        atom_hbond_acceptor               : Tensor,
        residue_hydropathy                : Tensor,
        residue_polarity                  : Tensor,
        surface_positions                 : Tensor | None = None,
        surface_normals                   : Tensor | None = None,
        surface_neighbors                 : Tensor | None = None,
        surface_neighbor_mask             : Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode atoms, transfer context, gate curvature channels, and propagate on the surface.

        Args:
            atomic_numbers: Element IDs ``[N]``; this information is always active.
            residue_type_ids: Residue IDs ``[N]``.
            atom_edge_index: Directed union topology ``[2,E]``.
            atom_edge_is_spatial: Spatial branch membership ``bool [E]``.
            atom_edge_is_covalent: Covalent branch membership ``bool [E]``.
            atom_edge_distance: Edge distance in ångströms ``[E]``.
            atom_edge_bond_order: Numeric bond order ``[E]``.
            atom_edge_same_residue: Same-residue flags ``[E]``.
            atom_edge_same_chain: Same-chain flags ``[E]``.
            atom_edge_residue_separation: Sequence-index separation ``[E]``.
            surface_curvatures: Mean, Gaussian, and curvedness values ``[M,S,3]``.
            surface_atom_neighbors: Bounded atom IDs ``[M,J]``.
            surface_atom_distances: Atom distances in ångströms ``[M,J]``.
            surface_atom_normal_offsets: Signed normal offsets ``[M,J]``.
            surface_atom_tangential_distances: Tangential distances ``[M,J]``.
            surface_atom_mask: Valid transfer mask ``[M,J]``.
            surface_operators: Per-protein intrinsic DiffusionNet operators.
            surface_ptr: Surface prefix boundaries ``[B+1]``.
            atom_role_ids: Atom role IDs ``[N]``.
            atom_hybridization_ids: Hybridization IDs ``[N]``.
            formal_charges: Formal charges ``[N]``.
            atom_aromaticity: Aromatic flags ``[N]``.
            atom_hbond_donor: Donor flags ``[N]``.
            atom_hbond_acceptor: Acceptor flags ``[N]``.
            residue_hydropathy: Normalized hydropathy ``[N]``.
            residue_polarity: Polarity flags ``[N]``.
            surface_positions: Optional V3 point positions ``[M,3]``.
            surface_normals: Optional V3 normals ``[M,3]``.
            surface_neighbors: Optional V3 neighbours ``[M,K]``.
            surface_neighbor_mask: Optional V3 validity ``[M,K]``.

        Returns:
            Surface embeddings ``[M,H]`` and local logits ``[M]``.
        """
        gates = self.semantic_gates.sample(training=self.training)

        atom_features = torch.cat(
            (
                self.atomic_number_embedding(atomic_numbers),
                self.residue_type_embedding(residue_type_ids) * gates["atom.residue_type"],
                self.atom_role_embedding(atom_role_ids) * gates["atom.atom_role"],
                self.hybridization_embedding(atom_hybridization_ids)
                * gates["atom.hybridization"],
                formal_charges.float().unsqueeze(1) * gates["atom.formal_charge"],
                atom_aromaticity.float().unsqueeze(1) * gates["atom.aromaticity"],
                atom_hbond_donor.float().unsqueeze(1) * gates["atom.hbond_donor"],
                atom_hbond_acceptor.float().unsqueeze(1) * gates["atom.hbond_acceptor"],
                residue_hydropathy.float().unsqueeze(1) * gates["atom.residue_hydropathy"],
                residue_polarity.float().unsqueeze(1) * gates["atom.residue_polarity"],
            ),
            dim=1,
        )
        atom_embeddings = self.atomic_encoder(
            atom_features,
            atom_edge_index,
            atom_edge_is_spatial,
            atom_edge_is_covalent,
            atom_edge_distance,
            atom_edge_bond_order,
            atom_edge_same_residue,
            atom_edge_same_chain,
            atom_edge_residue_separation,
            gates,
        )
        atom_context = self.surface_atom_transfer(
            atom_embeddings,
            surface_atom_neighbors,
            surface_atom_distances,
            surface_atom_normal_offsets,
            surface_atom_tangential_distances,
            surface_atom_mask,
            gates,
        ) * gates["surface.atom_context"]

        curvature = self.curvature_inputs(surface_curvatures, gates)
        initial   = self.surface_projection(torch.cat((atom_context, curvature), dim=1))
        encoded   = self.encode_surface_features(
            initial,
            surface_operators,
            surface_ptr,
            surface_positions,
            surface_normals,
            surface_neighbors,
            surface_neighbor_mask,
        )
        logits = self.local_head(encoded).squeeze(-1)
        return encoded, logits

    def curvature_inputs(
        self,
        curvatures: Tensor,
        gates     : Mapping[str, Tensor],
    ) -> Tensor:
        """Derive shape index and gate every normalized descriptor/scale channel.

        Mean curvature ``H`` and curvedness have inverse-length units; Gaussian curvature ``K``
        has inverse-length-squared units. The persisted scale index fixes their physical context,
        while ``tanh`` provides a deterministic bounded canonical transform. Shape index is
        ``(2/pi) atan2(2H, 2 sqrt(max(H²-K,0)))`` and already lies in ``[-1,1]``.

        Args:
            curvatures: Persisted ``(H,K,C)`` values ``[M,S,3]``.
            gates: One-forward semantic gate mapping.

        Returns:
            Fixed-width gated descriptors ``[M,4*S]``.

        Raises:
            ValueError: If the persisted scale count differs from model construction.
        """
        if curvatures.ndim != 3 or curvatures.shape[1:] != (self.curvature_scale_count, 3):
            raise ValueError("surface_curvatures must have the model's [M,S,3] shape")

        mean       = curvatures[:, :, 0]
        gaussian   = curvatures[:, :, 1]
        curvedness = curvatures[:, :, 2]
        difference = 2.0 * torch.sqrt((mean.square() - gaussian).clamp_min(0.0))
        shape      = (2.0 / torch.pi) * torch.atan2(2.0 * mean, difference)
        shape      = torch.where(
            (mean == 0.0) & (difference == 0.0), torch.zeros_like(shape), shape
        )

        descriptors = {
            "mean":        torch.tanh(mean),
            "gaussian":    torch.tanh(gaussian),
            "curvedness":  torch.tanh(curvedness),
            "shape_index": shape,
        }
        channels = [
            values[:, index] * gates[f"surface.curvature.{name}.scale_{index}"]
            for name, values in descriptors.items()
            for index in range(self.curvature_scale_count)
        ]
        return torch.stack(channels, dim=1)

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
        """Apply the fixed v1 DiffusionNet surface encoder.

        Args:
            features: Initial surface features ``[M,H]``.
            surface_operators: Per-protein intrinsic operator packs.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            surface_positions: Unused V3 positions.
            surface_normals: Unused V3 normals.
            surface_neighbors: Unused V3 neighbours.
            surface_neighbor_mask: Unused V3 mask.

        Returns:
            DiffusionNet embeddings ``[M,H]``.
        """
        del surface_positions, surface_normals, surface_neighbors, surface_neighbor_mask
        return self.surface_encoder(features, surface_operators, surface_ptr)

    def forward(
        self,
        atomic_numbers                    : Tensor,
        residue_type_ids                  : Tensor,
        atom_edge_index                   : Tensor,
        atom_edge_is_spatial              : Tensor,
        atom_edge_is_covalent             : Tensor,
        atom_edge_distance                : Tensor,
        atom_edge_bond_order              : Tensor,
        atom_edge_same_residue            : Tensor,
        atom_edge_same_chain              : Tensor,
        atom_edge_residue_separation      : Tensor,
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
        atom_role_ids                     : Tensor,
        atom_hybridization_ids            : Tensor,
        formal_charges                    : Tensor,
        atom_aromaticity                  : Tensor,
        atom_hbond_donor                  : Tensor,
        atom_hbond_acceptor               : Tensor,
        residue_hydropathy                : Tensor,
        residue_polarity                  : Tensor,
        surface_positions                 : Tensor | None = None,
        surface_normals                   : Tensor | None = None,
        surface_neighbors                 : Tensor | None = None,
        surface_neighbor_mask             : Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict local surface evidence and one MAX-pooled protein logit.

        Args:
            atomic_numbers: Element IDs ``[N]``.
            residue_type_ids: Residue IDs ``[N]``.
            atom_edge_index: Directed atomic topology ``[2,E]``.
            atom_edge_is_spatial: Spatial membership ``[E]``.
            atom_edge_is_covalent: Covalent membership ``[E]``.
            atom_edge_distance: Distances in ångströms ``[E]``.
            atom_edge_bond_order: Numeric bond orders ``[E]``.
            atom_edge_same_residue: Same-residue flags ``[E]``.
            atom_edge_same_chain: Same-chain flags ``[E]``.
            atom_edge_residue_separation: Residue-index separation ``[E]``.
            surface_curvatures: Surface descriptors ``[M,S,3]``.
            surface_atom_neighbors: Nearby atom IDs ``[M,J]``.
            surface_atom_distances: Nearby atom distances ``[M,J]``.
            surface_atom_normal_offsets: Signed normal offsets ``[M,J]``.
            surface_atom_tangential_distances: Tangential distances ``[M,J]``.
            surface_atom_mask: Valid transfer entries ``[M,J]``.
            surface_area_weights: Represented-area weights ``[M]``.
            surface_batch: Point-to-protein owners ``[M]``.
            surface_operators: Per-protein DiffusionNet operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            atom_role_ids: Atom-role IDs ``[N]``.
            atom_hybridization_ids: Hybridization IDs ``[N]``.
            formal_charges: Formal charges ``[N]``.
            atom_aromaticity: Aromatic flags ``[N]``.
            atom_hbond_donor: Donor flags ``[N]``.
            atom_hbond_acceptor: Acceptor flags ``[N]``.
            residue_hydropathy: Normalized hydropathy ``[N]``.
            residue_polarity: Polarity flags ``[N]``.
            surface_positions: Optional V3 positions ``[M,3]``.
            surface_normals: Optional V3 normals ``[M,3]``.
            surface_neighbors: Optional V3 neighbours ``[M,K]``.
            surface_neighbor_mask: Optional V3 validity mask ``[M,K]``.

        Returns:
            Essential ``surface_logits[M]`` and ``logits[B]`` outputs.
        """
        del surface_area_weights
        _, surface_logits = self.encode_surface(
            atomic_numbers,
            residue_type_ids,
            atom_edge_index,
            atom_edge_is_spatial,
            atom_edge_is_covalent,
            atom_edge_distance,
            atom_edge_bond_order,
            atom_edge_same_residue,
            atom_edge_same_chain,
            atom_edge_residue_separation,
            surface_curvatures,
            surface_atom_neighbors,
            surface_atom_distances,
            surface_atom_normal_offsets,
            surface_atom_tangential_distances,
            surface_atom_mask,
            surface_operators,
            surface_ptr,
            atom_role_ids,
            atom_hybridization_ids,
            formal_charges,
            atom_aromaticity,
            atom_hbond_donor,
            atom_hbond_acceptor,
            residue_hydropathy,
            residue_polarity,
            surface_positions,
            surface_normals,
            surface_neighbors,
            surface_neighbor_mask,
        )
        protein_count = len(surface_ptr) - 1
        protein_logits = self.global_max_pooling(
            surface_logits.unsqueeze(-1), surface_batch, protein_count
        ).squeeze(-1)
        return {"logits": protein_logits, "surface_logits": surface_logits}
