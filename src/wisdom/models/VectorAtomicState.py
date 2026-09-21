"""Small equivariant vector-state augmentation for the pre-freeze atomic spike."""

import torch
from torch import Tensor, nn


class VectorAtomicState(nn.Module):
    """Build directional atom channels and feed only rotational invariants back to scalars."""

    def __init__(self, hidden_dim: int, vector_channels: int) -> None:
        """Construct one scalar-to-vector-to-scalar message round.

        Args:
            hidden_dim: Scalar atom-state width ``H``.
            vector_channels: Number ``C_v`` of three-dimensional equivariant channels.

        Raises:
            ValueError: If either width is non-positive.
        """
        super().__init__()
        if hidden_dim < 1 or vector_channels < 1:
            raise ValueError("scalar and vector atomic widths must be positive")

        self.edge_coefficients = nn.Linear(2 * hidden_dim + 1, vector_channels)
        self.scalar_update = nn.Sequential(
            nn.Linear(hidden_dim + vector_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.normalization   = nn.LayerNorm(hidden_dim)
        self.hidden_dim      = hidden_dim
        self.vector_channels = vector_channels

    def forward(
        self,
        scalar_states : Tensor,
        positions     : Tensor,
        edge_index    : Tensor,
        edge_distances: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Produce equivariant vector channels and invariantly enriched scalar atom states.

        For each stored undirected pair ``(i,j)``, a learned scalar coefficient per vector channel
        multiplies the unit direction ``r_ij/||r_ij||``. Opposite signs are accumulated at the two
        endpoints, so vectors rotate with the structure. Their channel norms are invariant and can
        safely update scalar states without choosing a coordinate frame.

        Args:
            scalar_states: Atom features ``[N,H]``.
            positions: Centered atom coordinates ``[N,3]`` in ångströms.
            edge_index: Unique undirected pairs ``[2,E]``.
            edge_distances: Non-negative pair distances ``[E]`` in ångströms.

        Returns:
            Updated scalar states ``[N,H]`` and equivariant vectors ``[N,C_v,3]``.

        Raises:
            ValueError: If array shapes disagree.
        """
        if scalar_states.ndim != 2 or scalar_states.shape[1] != self.hidden_dim:
            raise ValueError("scalar atom states must have shape [N,H]")
        if positions.shape != (len(scalar_states), 3):
            raise ValueError("atom positions must have shape [N,3]")
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("atomic edge index must have shape [2,E]")
        if edge_distances.shape != (edge_index.shape[1],):
            raise ValueError("atomic edge distances must have shape [E]")

        source, target = edge_index
        offsets = positions[target] - positions[source]
        directions = offsets / edge_distances.clamp_min(1.0e-6)[:, None]
        coefficients = self.edge_coefficients(
            torch.cat(
                (
                    scalar_states[source],
                    scalar_states[target],
                    edge_distances[:, None],
                ),
                dim=1,
            )
        )
        messages = coefficients[:, :, None] * directions[:, None, :]
        vectors  = scalar_states.new_zeros(
            (len(scalar_states), self.vector_channels, 3)
        )
        vectors.index_add_(0, target, messages)
        vectors.index_add_(0, source, -messages)

        degree = torch.bincount(edge_index.flatten(), minlength=len(scalar_states)).clamp_min(1)
        vectors = vectors / degree.to(vectors.dtype)[:, None, None]
        invariants = torch.linalg.vector_norm(vectors, dim=2)
        update = self.scalar_update(torch.cat((scalar_states, invariants), dim=1))
        return self.normalization(scalar_states + update), vectors
