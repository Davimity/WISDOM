"""Physical CA graph communication using LambdaForge's public SchNet interactions."""

import json
import torch

from typing import Any, cast
from torch import Tensor, nn
from collections.abc import Mapping
from lambdaforge.nn import SchNetLayer
from wisdom.models.residue.ResidueTopology import ResidueTopology
from wisdom.models.residue.ResidueGraphMode import ResidueGraphMode


class ResidueGraphEncoder(nn.Module):
    """Refine context without replacing its original path or crossing protein namespaces."""

    def __init__(self, context_dim: int, mode: str = "hybrid", hidden_dim: int = 64,
                 layers: int = 2, radius: float = 10.0, k: int = 16,
                 radial_count: int = 16, initial_scale: float = 1.0,
                 zero_initialize: bool = True, topology: str = "descending",
                 atomic_dim: int = 128, null_atomic_message: bool = False,
                 surface_radius: float = 12.0, surface_k: int = 32) -> None:
        """Declare one independent residue network, initially preserving the common predictor.

        Args:
            context_dim: Projected residue context/output width, default inherited from V10.
            mode: pointwise, null, sequence, contact or hybrid; see ResidueGraphMode.
            hidden_dim: Residue latent width, default 64; no atomic/surface width is changed.
            layers: Number of residue interactions, default two.
            radius: CA spatial radius in Å, default 10; verified peptide edges survive it.
            k: Active directed nearest-CA prefix before union symmetrization, default 16.
            radial_count: LF Gaussian basis count, default 16.
            initial_scale: Multiplicative alpha for the whole new branch, default one.
            zero_initialize: Zero the last projection, not both it and alpha; default true.
            topology: descending, roundtrip or parallel; each executes this network once.
            atomic_dim: Atomic output width for bounded roundtrip/parallel adapters, default 128.
            null_atomic_message: Supply zeros instead of pooled atoms in the roundtrip control.
            surface_radius: Minimum-heavy-atom support radius in Å for the parallel branch.
            surface_k: Maximum distinct residues per surface point, default 32.

        Raises:
            ValueError: Invalid dimensions, graph choices, or a doubly zeroed learning path.
        """
        super().__init__()
        self.mode = ResidueGraphMode(mode)
        self.topology = ResidueTopology(topology)
        if min(context_dim, hidden_dim, layers, k, radial_count) < 1 or radius <= 0:
            raise ValueError("residue dimensions, radius and budgets must be positive")
        if zero_initialize and initial_scale == 0:
            raise ValueError("zero residue output and zero alpha would block all branch gradients")
        self.radius, self.k = radius, k
        self.scale = float(initial_scale)
        self.null_atomic_message = null_atomic_message
        self.surface_radius, self.surface_k = surface_radius, surface_k
        self.input = nn.Linear(context_dim, hidden_dim)
        self.layers = nn.ModuleList(
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
                          nn.Linear(hidden_dim, hidden_dim))
            if self.mode is ResidueGraphMode.POINTWISE else
            SchNetLayer(hidden_dim, radial_count, radius, edge_channels=3)
            for _ in range(layers))
        self.output = nn.Linear(hidden_dim, context_dim)
        self.atomic_input = (nn.Linear(atomic_dim, context_dim, bias=False)
                             if self.topology is ResidueTopology.ROUNDTRIP else None)
        self.atomic_output = (nn.Linear(context_dim, atomic_dim, bias=False)
                              if self.topology is not ResidueTopology.DESCENDING else None)
        if zero_initialize:
            nn.init.zeros_(self.output.weight)
            nn.init.zeros_(self.output.bias)

    def forward(self, context: Tensor, valid_context: Tensor, member: Mapping[str, Any],
                atoms: Tensor | None = None, parent: Tensor | None = None) -> Tensor:
        """Return the masked correction [R,context_dim] in one physical protein namespace.

        Args:
            context: Projected original residue features [R,d], not global XYZ or labels.
            valid_context: External coverage [R]; missing CA/context keeps the old atom route.
            member: Validated CA graph sidecar, including local indices and stored support.
            atoms: Optional [N,atomic_dim] states for the single roundtrip summary.
            parent: Local atom-to-residue addresses [N], mandatory with atoms.

        Returns:
            Alpha times a learned correction. Initially zero when requested; after one backward
            the output projection can learn, and subsequent steps propagate gradients upstream.

        Raises:
            ValueError: Missing CA graph, insufficient cached radius/K or ownership mismatch.
        """
        if "ca_edge_index" not in member:
            raise ValueError("V14 requires the independently generated CA residue graph; enrich @9")
        metadata = json.loads(member["metadata_json"])["parameters"]
        if self.radius > metadata["ca_radius"] or self.k > metadata["ca_k_max"]:
            raise ValueError("requested residue radius/K exceeds stored CA graph support")
        if len(context) != len(member["backbone_valid"]):
            raise ValueError("external residue context and physical CA nodes are not aligned")
        valid = member["backbone_valid"][:, 1].bool() & valid_context.bool()
        if atoms is not None:
            if parent is None or len(parent) != len(atoms) or self.atomic_input is None:
                raise ValueError("roundtrip requires exact atom ownership and its declared adapter")
            pooled = atoms.new_zeros((len(context), atoms.shape[1]))
            if not self.null_atomic_message:
                pooled.index_add_(0, parent, atoms)
                pooled /= torch.bincount(parent, minlength=len(context)).clamp_min(1)[:, None]
            context = context + self.atomic_input(pooled)
        hidden = torch.nn.functional.silu(self.input(context)) * valid[:, None]
        edges = member["ca_edge_index"].long()
        ranks = member["ca_edge_rank"].long()
        relation = member["ca_edge_relation"].long()
        distance = member["ca_edge_distance"].float()
        spatial = (((ranks > 0) & (ranks <= self.k)).any(0)) & (distance <= self.radius)
        sequence = (relation & 1) != 0
        chemical = (relation & 12) != 0
        active = (sequence | chemical if self.mode is ResidueGraphMode.SEQUENCE else spatial
                  if self.mode is ResidueGraphMode.CONTACT else spatial | sequence | chemical)
        active &= valid[edges[0]] & valid[edges[1]]
        selected = edges[:, active]
        directed = torch.cat((selected, selected.flip(0)), dim=1)
        lengths = distance[active].repeat(2)
        features = torch.stack((sequence, spatial, chemical), dim=1)[active].float().repeat(2, 1)
        for layer in self.layers:
            if self.mode is ResidueGraphMode.POINTWISE:
                hidden = hidden + layer(hidden)
            elif self.mode is ResidueGraphMode.NULL:
                interaction = cast(SchNetLayer, layer)
                radial, _ = interaction.radial_inputs(lengths)
                hidden = interaction.forward_from_radial(hidden, directed, radial,
                    torch.zeros_like(lengths), features)
            else:
                hidden = layer(hidden, directed, lengths, features)
            hidden = hidden * valid[:, None]
        return self.scale * self.output(hidden) * valid[:, None]
