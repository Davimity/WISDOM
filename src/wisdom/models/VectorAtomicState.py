"""Small equivariant vector-state augmentation for the pre-freeze atomic spike."""

import torch

from torch import Tensor, nn
from wisdom.models.ArchitectureImplementation import ArchitectureImplementation


class VectorAtomicState(nn.Module):
    """Build directional atom channels and feed only rotational invariants back to scalars."""

    def __init__(
        self, hidden_dim: int, vector_channels: int, implementation: str = "controlled",
    ) -> None:
        """Construct one scalar-to-vector-to-scalar message round.

        Args:
            hidden_dim: Scalar atom-state width ``H``.
            vector_channels: Number ``C_v`` of three-dimensional equivariant channels.
            implementation: controlled symmetrizes edge coefficients, accumulates vectors in
                FP32 and preserves isolated atoms. legacy restores archived ordered scoring.

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
        self.implementation  = ArchitectureImplementation(implementation)

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
        safely update scalar states without choosing a coordinate frame. Controlled coefficients
        average the original linear scorer over both endpoint orders; no vector state is carried
        into another vector message-passing layer. Isolated atoms keep their original scalars.

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
        controlled = self.implementation is ArchitectureImplementation.CONTROLLED
        if controlled:
            # An undirected pair has no chemically meaningful source ordering. Averaging
            # the same scorer in both orders preserves its state_dict and rotation scalarity.

            reverse = self.edge_coefficients(torch.cat(
                (scalar_states[target], scalar_states[source], edge_distances[:, None]), dim=1,
            ))
            coefficients = 0.5 * (coefficients + reverse)
        messages = coefficients[:, :, None] * directions[:, None, :]
        vectors  = (scalar_states.float() if controlled else scalar_states).new_zeros(
            (len(scalar_states), self.vector_channels, 3)
        )
        messages = messages.to(vectors.dtype)
        vectors.index_add_(0, target, messages)
        vectors.index_add_(0, source, -messages)

        degree = torch.bincount(edge_index.flatten(), minlength=len(scalar_states))
        vectors = vectors / degree.clamp_min(1).to(vectors.dtype)[:, None, None]
        invariants = torch.linalg.vector_norm(vectors, dim=2)
        update = self.scalar_update(torch.cat((scalar_states, invariants), dim=1))
        updated = self.normalization(scalar_states + update)
        if controlled:
            updated = torch.where((degree > 0)[:, None], updated, scalar_states)
        return updated, vectors
