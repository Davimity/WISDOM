"""Two-branch sparse atomic encoder with globally shared semantic gates."""

import torch

from collections.abc import Mapping
from torch import Tensor, nn


class GatedAtomicEncoder(nn.Module):
    """Propagate spatial and covalent messages without conflating physical meanings."""

    def __init__(
        self,
        input_dim         : int,
        hidden_dim        : int,
        layers            : int,
        dropout           : float,
        distance_scale    : float,
        message_chunk_size: int,
        residue_scale     : float = 32.0,
    ) -> None:
        """Build residual sparse message layers and small edge conditioners.

        Args:
            input_dim: Concatenated atom-feature width.
            hidden_dim: Atom latent width.
            layers: Positive number of spatial/covalent message layers.
            dropout: Residual dropout probability.
            distance_scale: Ångström scale used to normalize edge distances.
            message_chunk_size: Maximum branch messages materialized together.
            residue_scale: Sequence-distance scale used to normalize residue separation.

        Raises:
            ValueError: If a dimension, layer count, or physical scale is invalid.
        """
        super().__init__()
        if (
            min(input_dim, hidden_dim, layers, message_chunk_size) < 1
            or distance_scale <= 0.0
            or residue_scale <= 0.0
        ):
            raise ValueError("atomic encoder dimensions, layers, and scales must be positive")

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.spatial_messages = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)
        )
        self.covalent_messages = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)
        )
        self.spatial_conditioners = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1, bias=False))
            for _ in range(layers)
        )
        self.covalent_conditioners = nn.ModuleList(
            nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 1, bias=False))
            for _ in range(layers)
        )
        self.normalizations = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.dropout        = nn.Dropout(dropout)
        self.distance_scale = float(distance_scale)
        self.message_chunk_size = message_chunk_size
        self.residue_scale  = float(residue_scale)

    def forward(
        self,
        atom_features          : Tensor,
        edge_index             : Tensor,
        edge_is_spatial        : Tensor,
        edge_is_covalent       : Tensor,
        edge_distance          : Tensor,
        edge_bond_order        : Tensor,
        edge_same_residue      : Tensor,
        edge_same_chain        : Tensor,
        edge_residue_separation: Tensor,
        gates                  : Mapping[str, Tensor],
    ) -> Tensor:
        """Apply both physical branches with one shared gate sample per semantic source.

        Edge attributes modulate a valid attribute-free base message through a centered residual
        scorer. Thus disabling all edge-attribute gates preserves message passing, while disabling
        a branch gate removes only that physical branch. A pair that is both spatial and covalent
        contributes once to each branch.

        Args:
            atom_features: Concatenated element/chemistry features ``[N,F]``.
            edge_index: Directed atom pairs ``[2,E]``.
            edge_is_spatial: Spatial-neighbour membership ``bool [E]``.
            edge_is_covalent: Covalent-bond membership ``bool [E]``.
            edge_distance: Center distance in ångströms ``[E]``.
            edge_bond_order: Numeric bond order ``[E]``.
            edge_same_residue: Same-residue indicator ``[E]``.
            edge_same_chain: Same-chain indicator ``[E]``.
            edge_residue_separation: Absolute sequence-index separation ``[E]``.
            gates: One-forward semantic gate mapping from ``SemanticGateRegistry``.

        Returns:
            Encoded atom features ``[N,H]``.
        """
        hidden = self.input_projection(atom_features)
        source, target = edge_index

        spatial_features = torch.stack(
            (
                (edge_distance / self.distance_scale).clamp(max=2.0)
                * gates["edge.spatial.distance"],
                edge_same_residue.float() * gates["edge.spatial.same_residue"],
                edge_same_chain.float() * gates["edge.spatial.same_chain"],
                (edge_residue_separation.float() / self.residue_scale).clamp(max=2.0)
                * gates["edge.spatial.residue_separation"],
            ),
            dim=1,
        )
        covalent_features = torch.stack(
            (
                (edge_distance / self.distance_scale).clamp(max=2.0)
                * gates["edge.covalent.distance"],
                edge_bond_order.float().clamp(min=0.0, max=3.0) / 3.0
                * gates["edge.covalent.bond_order"],
                edge_same_residue.float() * gates["edge.covalent.same_residue"],
            ),
            dim=1,
        )

        layers = zip(
                self.spatial_messages,
                self.covalent_messages,
                self.spatial_conditioners,
                self.covalent_conditioners,
                self.normalizations,
                strict=True,
        )
        for spatial, covalent, spatial_conditioner, covalent_conditioner, normalization in layers:
            spatial_delta = self._branch(
                hidden,
                source,
                target,
                edge_is_spatial,
                spatial,
                spatial_conditioner,
                spatial_features,
            )
            covalent_delta = self._branch(
                hidden,
                source,
                target,
                edge_is_covalent,
                covalent,
                covalent_conditioner,
                covalent_features,
            )
            hidden = normalization(
                hidden
                + self.dropout(gates["atomic_graph.spatial"] * spatial_delta)
                + self.dropout(gates["atomic_graph.covalent"] * covalent_delta)
            )
            hidden = torch.nn.functional.silu(hidden)

        return hidden

    def _branch(
        self,
        hidden    : Tensor,
        source    : Tensor,
        target    : Tensor,
        mask      : Tensor,
        message   : nn.Module,
        conditioner: nn.Module,
        attributes: Tensor,
    ) -> Tensor:
        """Aggregate one sparse mean branch without constructing a dense adjacency matrix.

        Args:
            hidden: Current atom states ``[N,H]``.
            source: Directed source IDs ``[E]``.
            target: Directed destination IDs ``[E]``.
            mask: Membership of this physical branch ``bool [E]``.
            message: Attribute-free message projection.
            conditioner: Scalar residual scorer for gated edge attributes.
            attributes: Gated and normalized branch attributes ``[E,A]``.

        Returns:
            Mean aggregated branch update ``[N,H]``; zero for atoms without branch neighbours.
        """
        selected_source = source[mask]
        selected_target = target[mask]
        if not selected_source.numel():
            return torch.zeros_like(hidden)

        selected_attributes = attributes[mask]
        aggregated = torch.zeros_like(hidden)
        degree = torch.zeros(len(hidden), device=hidden.device, dtype=hidden.dtype)

        # Bound temporary [edge,H] activations while preserving the exact sparse sum and degree.

        for start in range(0, len(selected_source), self.message_chunk_size):
            stop                 = start + self.message_chunk_size
            chunk_source         = selected_source[start:stop]
            chunk_target         = selected_target[start:stop]
            chunk_attributes     = selected_attributes[start:stop]
            zero_attributes      = torch.zeros_like(chunk_attributes)
            residual_scale       = conditioner(chunk_attributes) - conditioner(zero_attributes)
            messages             = message(hidden[chunk_source]) * (1.0 + residual_scale)
            messages             = messages.to(hidden.dtype)

            aggregated.index_add_(0, chunk_target, messages)
            degree.index_add_(
                0,
                chunk_target,
                torch.ones_like(chunk_target, dtype=hidden.dtype),
            )

        return aggregated / degree.clamp_min(1.0).unsqueeze(1)
