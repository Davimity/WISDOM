"""Rigid-invariant sparse atomic-state query heads for electrostatic and two LJ targets."""

import torch

from torch import Tensor, nn
from lambdaforge.nn import Scatter
from collections.abc import Mapping


class FieldDecoder(nn.Module):
    """Sum local atom/radial messages and retain a global scalar summary for distant sources."""

    def __init__(self, atom_dim: int, hidden_dim: int = 64, radial_count: int = 16,
                 radius: float = 12.0, chunk_size: int = 65536,
                 output_mode: str = "three", state_control: str = "normal") -> None:
        """Declare the decoder, keeping force-field parameters outside its input contract.

        Args:
            atom_dim: Existing encoder output width H, shared with ordinary WISDOM.
            hidden_dim: Message/decoder width, default 64.
            radial_count: Fixed Gaussian distance basis count, default 16.
            radius: Stored local candidate radius in Å, default 12; not an electrostatic cutoff.
            chunk_size: Maximum gathered pair messages, default 65536, bounding activation memory.
            output_mode: three distinct final outputs or one probe-conditioned shared output.
            state_control: normal, zero, permuted or detached; controls are explicit ablations,
                not silent remedies for a weak encoder. Permutation never crosses proteins.
        """
        super().__init__()
        if output_mode not in {"three", "conditioned"} or state_control not in {
            "normal", "zero", "permuted", "detached"
        } or min(atom_dim, hidden_dim, radial_count, chunk_size) < 1 or radius <= 0:
            raise ValueError("invalid VF decoder dimensions, support or control")
        self.radius = radius
        self.chunk_size = chunk_size
        self.state_control = state_control
        self.output_mode = output_mode
        self.register_buffer("centers", torch.linspace(0, radius, radial_count))
        self.radial_width = radius / max(1, radial_count - 1)
        self.message = nn.Sequential(nn.Linear(atom_dim + radial_count, hidden_dim), nn.SiLU(),
                                     nn.Linear(hidden_dim, hidden_dim))
        self.global_projection = nn.Linear(atom_dim, hidden_dim)
        self.global_distance = nn.Linear(1, hidden_dim, bias=False)
        self.element_head = nn.Linear(atom_dim, 119)
        if output_mode == "three":
            self.heads = nn.ModuleList(nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(), nn.Linear(hidden_dim, 1)) for _ in range(3))
        else:
            self.probe_embedding = nn.Embedding(3, 8)
            self.head = nn.Sequential(nn.Linear(2 * hidden_dim + 8, hidden_dim), nn.SiLU(),
                                      nn.Linear(hidden_dim, 1))

    def forward(self, states: Tensor, atoms: Tensor, atom_batch: Tensor,
                queries: Mapping[str, Tensor]) -> Tensor:
        """Predict three normalized fields without receiving target values or field parameters.

        Args:
            states: Encoder embeddings [N,H], retaining autograd connectivity to SchNet.
            atoms: Centered retained atom coordinates [N,3], Å.
            atom_batch: Explicit protein ownership [N].
            queries: positions [Q,3] Å, owners [Q], pair_query [E] and pair_atom [E]; atom IDs
                are disjoint-global and query IDs refer only to this sampled query batch.

        Returns:
            Predicted transformed/normalized potential, LJ1 and LJ2 [Q,3]. Local cost is O(E),
            global summary O(N+Q); no Q-by-N attention/distance tensor is constructed. The global
            summary is a learned mean, not an exact APBS or far-field physical solution.

        Raises:
            ValueError: Sparse pair ownership mixes distinct proteins.
        """
        if self.state_control == "zero":
            states = states * 0
        elif self.state_control == "detached":
            states = states.detach()
        elif self.state_control == "permuted":
            permutation = torch.arange(len(states), device=states.device)
            for owner in range(int(queries["protein_count"].item())):
                ids = torch.nonzero(atom_batch == owner).flatten()
                permutation[ids] = ids.flip(0)
            states = states[permutation]
        rows, indices = queries["pair_query"], queries["pair_atom"]
        count = len(queries["positions"])
        context = states.new_zeros((count, self.global_projection.out_features))

        # Only integer ownership is checked here; physical target validation occurs once at the
        # immutable dataset boundary, not by recomputing APBS or geometry in an optimizer step.

        if not torch.equal(atom_batch[indices], queries["owners"][rows]):
            raise ValueError("VF query pairs cross protein ownership boundaries")
        for start in range(0, len(rows), self.chunk_size):
            query_ids, atom_ids = rows[start:start + self.chunk_size], indices[
                start:start + self.chunk_size]
            distance = torch.linalg.vector_norm(
                queries["positions"][query_ids].float() - atoms[atom_ids].float(), dim=1)
            radial = torch.exp(-0.5 * ((distance[:, None] - self.centers) / self.radial_width)**2)
            messages = self.message(torch.cat((states[atom_ids], radial.to(states.dtype)), dim=1))
            context.index_add_(0, query_ids, messages.to(context.dtype))
        protein_count = int(queries["protein_count"].item())
        global_state = self.global_projection(Scatter.mean(states, atom_batch, protein_count))
        centroid = Scatter.mean(atoms.float(), atom_batch, protein_count)
        global_radius = torch.linalg.vector_norm(
            queries["positions"].float() - centroid[queries["owners"]], dim=1, keepdim=True)
        radial_position = global_radius / (self.radius + global_radius)
        global_context = (global_state[queries["owners"]]
                          + self.global_distance(radial_position.to(states.dtype)))
        outputs = []
        for channel in range(3):
            far = global_context if channel == 0 else torch.zeros_like(global_context)
            joined = torch.cat((context, far), dim=1)
            if self.output_mode == "three":
                outputs.append(self.heads[channel](joined))
            else:
                probe = self.probe_embedding.weight[channel].expand(count, -1)
                outputs.append(self.head(torch.cat((joined, probe), dim=1)))
        return torch.cat(outputs, dim=1)
