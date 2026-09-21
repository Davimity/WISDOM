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
        residual_initial_scale: float | None = None,
        neutral_edge_initialization: bool = False,
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
            residual_initial_scale: Optional learnable per-channel scale for the combined spatial
                and covalent residual update. ``None`` preserves the historical path.
            neutral_edge_initialization: Zero the final edge-conditioner layers so epoch zero uses
                attribute-free message passing before learning physical corrections.

        Raises:
            ValueError: If a dimension, layer count, or physical scale is invalid.
        """
        super().__init__()
        if (
            min(input_dim, hidden_dim, layers, message_chunk_size) < 1
            or distance_scale <= 0.0
            or residue_scale <= 0.0
            or (residual_initial_scale is not None and residual_initial_scale < 0.0)
        ):
            raise ValueError("atomic encoder dimensions, layers, and scales must be positive")

        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.spatial_messages = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)
        )
        self.covalent_messages = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)
        )

        # Moving W after aggregation is exact only for a linear map without an additive bias.
        # Keep this contract adjacent to construction so a future architectural edit fails loudly.

        if any(
            message.bias is not None
            for message in (*self.spatial_messages, *self.covalent_messages)
        ):
            raise RuntimeError("factorized atomic messages require bias-free linear maps")
        self.spatial_conditioners = nn.ModuleList(
            nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1, bias=False))
            for _ in range(layers)
        )
        self.covalent_conditioners = nn.ModuleList(
            nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 1, bias=False))
            for _ in range(layers)
        )
        if neutral_edge_initialization:
            self.initialize_neutral_edge_residuals()

        self.normalizations = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.residual_scales = nn.ParameterList(
            nn.Parameter(torch.full((hidden_dim,), residual_initial_scale))
            for _ in range(layers)
        ) if residual_initial_scale is not None else nn.ParameterList()
        self.dropout        = nn.Dropout(dropout)
        self.distance_scale = float(distance_scale)
        self.message_chunk_size = message_chunk_size
        self.residue_scale  = float(residue_scale)

    def initialize_neutral_edge_residuals(self) -> None:
        """Zero edge-conditioner outputs while preserving the attribute-free message path.

        The conditioner is mathematically a correction ``c(e)-c(0)``. Zeroing only its final
        layer makes that correction exactly zero at initialization without disabling message
        passing or preventing later learning.
        """
        for conditioner in (*self.spatial_conditioners, *self.covalent_conditioners):
            nn.init.zeros_(conditioner[-1].weight)

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
            edge_index: Unique undirected atom pairs ``[2,E]`` satisfying ``src < dst``.
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

        # Each persisted pair is unique and undirected. Branch topology, gated attributes, and
        # degree therefore depend on this forward's gate sample but not on the layer index. Prepare
        # them once and let each layer emit the two directed messages without duplicating tensors.

        spatial_branch = self._prepare_branch(
            source,
            target,
            edge_is_spatial,
            spatial_features,
            len(hidden),
            hidden.dtype,
        )
        covalent_branch = self._prepare_branch(
            source,
            target,
            edge_is_covalent,
            covalent_features,
            len(hidden),
            hidden.dtype,
        )

        layers = zip(
                range(len(self.normalizations)),
                self.spatial_messages,
                self.covalent_messages,
                self.spatial_conditioners,
                self.covalent_conditioners,
                self.normalizations,
                strict=True,
        )
        for (
            layer_index,
            spatial,
            covalent,
            spatial_conditioner,
            covalent_conditioner,
            normalization,
        ) in layers:
            spatial_delta = self._branch(
                hidden,
                spatial_branch,
                spatial,
                spatial_conditioner,
            )
            covalent_delta = self._branch(
                hidden,
                covalent_branch,
                covalent,
                covalent_conditioner,
            )
            residual = (
                self.dropout(gates["atomic_graph.spatial"] * spatial_delta)
                + self.dropout(gates["atomic_graph.covalent"] * covalent_delta)
            )
            if len(self.residual_scales):
                residual = residual * self.residual_scales[layer_index]
            hidden = normalization(hidden + residual)
            hidden = torch.nn.functional.silu(hidden)

        return hidden

    def _branch(
        self,
        hidden     : Tensor,
        branch     : tuple[Tensor, Tensor, Tensor, Tensor],
        message    : nn.Module,
        conditioner: nn.Module,
    ) -> Tensor:
        """Aggregate one exact factorized undirected mean-message branch.

        For an edge ``j -> i``, the former implementation accumulated
        ``a_ji * W h_j``. Because ``W`` is a shared linear map with no bias,
        ``mean_j(a_ji * W h_j) = W(mean_j(a_ji * h_j))``. Applying ``W`` after the sparse mean
        changes work from ``O(E*H²)`` to ``O(E*H + N*H²)`` without changing the represented
        function. Each compact pair contributes once in each direction, exactly as the previous
        physically duplicated edge list did.

        Args:
            hidden: Current atom states ``[N,H]``.
            branch: Selected compact sources, targets, attributes, and undirected degree.
            message: Shared bias-free attribute-free message projection.
            conditioner: Scalar residual scorer for gated edge attributes.

        Returns:
            Mean aggregated branch update ``[N,H]``; zero for atoms without branch neighbours.
        """
        source, target, attributes, degree = branch
        if not source.numel():
            return torch.zeros_like(hidden)

        aggregated = torch.zeros_like(hidden)

        # Centering makes a fully gated-off attribute vector recover scale one. The zero response
        # is identical for every edge, so evaluating it once avoids one MLP pass per edge/layer.

        zero_attributes = torch.zeros(
            (1, attributes.shape[1]),
            device=attributes.device,
            dtype=attributes.dtype,
        )
        zero_response = conditioner(zero_attributes)

        # Bound temporary [edge,H] activations. One compact pair creates two directional hidden
        # vectors, so half the configured message budget is consumed per endpoint direction. Both
        # index additions preserve the former contributions without duplicating edge attributes.

        pair_chunk_size = max(1, self.message_chunk_size // 2)
        for start in range(0, len(source), pair_chunk_size):
            stop                 = start + pair_chunk_size
            chunk_source         = source[start:stop]
            chunk_target         = target[start:stop]
            chunk_attributes     = attributes[start:stop]
            residual_scale       = conditioner(chunk_attributes) - zero_response
            scaled_source        = hidden[chunk_source] * (1.0 + residual_scale)
            scaled_target        = hidden[chunk_target] * (1.0 + residual_scale)

            aggregated.index_add_(0, chunk_target, scaled_source.to(hidden.dtype))
            aggregated.index_add_(0, chunk_source, scaled_target.to(hidden.dtype))

        mean = aggregated / degree.clamp_min(1.0).unsqueeze(1)
        return message(mean)

    @staticmethod
    def _prepare_branch(
        source    : Tensor,
        target    : Tensor,
        mask      : Tensor,
        attributes: Tensor,
        atom_count: int,
        dtype     : torch.dtype,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """Select one branch once and count its bidirectional compact-edge degree.

        Args:
            source: First endpoint of every compact pair ``long [E]``.
            target: Second endpoint of every compact pair ``long [E]``.
            mask: Membership of each pair in this physical branch ``bool [E]``.
            attributes: Normalized gated attributes ``float [E,A]``.
            atom_count: Number of atoms defining the output degree vector.
            dtype: Current hidden-state dtype used by mean normalization.

        Returns:
            Selected sources, targets, attributes, and integer degree ``[N]``. Each selected pair
            increments both endpoint degrees because both message directions are evaluated.
        """
        selected_source     = source[mask]
        selected_target     = target[mask]
        selected_attributes = attributes[mask]

        endpoints = torch.cat((selected_source, selected_target))
        degree    = torch.bincount(endpoints, minlength=atom_count).to(dtype=dtype)
        return selected_source, selected_target, selected_attributes, degree
