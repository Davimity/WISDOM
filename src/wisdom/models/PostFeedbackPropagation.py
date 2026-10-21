"""Independent post-feedback propagation, with an exact identity-start option."""

import torch

from torch import Tensor, nn
from collections.abc import Mapping
from torch.utils.checkpoint import checkpoint
from wisdom.models.GatedAtomicEncoder import GatedAtomicEncoder


class PostFeedbackPropagation(nn.Module):
    """Reuse hidden-state graph blocks, or a separately labelled local-capacity control."""

    def __init__(
        self,
        hidden_dim        : int,
        layers            : int = 2,
        dropout           : float = 0.2,
        distance_scale    : float = 6.0,
        message_chunk_size: int = 65536,
        initial_scale     : float = 0.1,
        nodewise          : bool = False,
        activation_checkpointing: bool = False,
    ) -> None:
        """Allocate independent GNN2 weights without an input-feature projection.

        Args:
            hidden_dim: Already encoded scalar width H; no categorical features are rebuilt.
            layers: One or two graph message layers; the D8 local control requires two.
            dropout: Existing per-relation dropout probability.
            distance_scale: Existing edge-distance normalization in Å.
            message_chunk_size: Existing bounded mean-message work budget.
            initial_scale: Initial learned scalar gamma; zero gives exact identity, not LayerNorm.
            nodewise: Use H→2H→H SiLU capacity instead of communicating along edges.
            activation_checkpointing: Recompute differentiable blocks during backward,
                preserving RNG.

        Raises:
            ValueError: A depth outside the bounded study or an incompatible control is requested.
        """
        super().__init__()
        if layers not in {1, 2} or (nodewise and layers != 2):
            raise ValueError("post-feedback propagation supports one/two layers; nodewise uses two")
        self.nodewise = nodewise
        self.activation_checkpointing = activation_checkpointing
        self.scale = nn.Parameter(torch.tensor(float(initial_scale)))
        self.transform = (
            nn.Sequential(nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(),
                          nn.Linear(2 * hidden_dim, hidden_dim))
            if nodewise else GatedAtomicEncoder(
                hidden_dim, hidden_dim, layers, dropout, distance_scale, message_chunk_size,
                project_inputs=False,
            )
        )

    def forward(
        self,
        states                 : Tensor,
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
        """Return A + gamma*(F(A,E)-A), or A + gamma*MLP(A) for D8.

        The original encoder includes SiLU and LayerNorm: it is not A+Delta. Subtract its
        complete input explicitly before residual mixing. No normalization follows this mix,
        so gamma=0 is genuinely identity. Closing a graph relation reuses the existing gates.

        Args:
            states: Already constructed atomic states A [N,H], with a live autograd path.
            edge_index: Original unique undirected pairs [2,E]; never copied or rebuilt.
            edge_is_spatial: Original spatial membership bool [E].
            edge_is_covalent: Original covalent membership bool [E], including overlaps.
            edge_distance: Original center distances [E], Å.
            edge_bond_order: Numeric bond orders [E].
            edge_same_residue: Same-residue flags [E].
            edge_same_chain: Same-chain flags [E].
            edge_residue_separation: Absolute sequence separations [E].
            gates: The same one-forward semantic sample used by GNN1 and both transfers.

        Returns:
            Updated scalar states [N,H]. Coordinates, topology, operators and labels are untouched.
        """
        arguments = (states,) if self.nodewise else (
            states, edge_index, edge_is_spatial, edge_is_covalent, edge_distance,
            edge_bond_order, edge_same_residue, edge_same_chain, edge_residue_separation, gates,
        )
        if self.activation_checkpointing and self.training and torch.is_grad_enabled():
            transformed = checkpoint(self.transform, *arguments, use_reentrant=False,
                                     preserve_rng_state=True)
        else:
            transformed = self.transform(*arguments)
        delta = transformed if self.nodewise else transformed - states
        return states + self.scale * delta
