"""Relation-aware WISDOM composition of public LambdaForge graph operators."""

import torch

from typing import Any
from torch import Tensor, nn
from inspect import signature
from lambdaforge.nn.models.graph import edge_geometry
from collections.abc import Mapping, Sequence, Iterable
from wisdom.models.AtomicEncoderType import AtomicEncoderType
from wisdom.models.GatedAtomicEncoder import GatedAtomicEncoder
from lambdaforge.nn import ECMP, GVP, GCNLayer, GINLayer, GATv2Layer
from lambdaforge.nn import StructuralGraphTransformerLayer, GraphStructuralEncoding
from lambdaforge.nn import GINELayer, NNConvLayer, PNALayer, SchNetLayer, PaiNNLayer
from lambdaforge.nn import GraphSAGELayer, RelationalGCNLayer, GraphTransformerLayer
from lambdaforge.nn import DimeNet, DimeNetBasis, GraphTriplets, GVPGNNLayer, EGNNLayer


class AtomicGraphEncoder(nn.Module):
    """Keep scientific relations/gates outside LF while delegating all GNN equations to LF."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        layers: int,
        dropout: float,
        distance_scale: float,
        message_chunk_size: int,
        encoder_type: str,
        radial_count: int = 16,
        vector_channels: int = 16,
        basis_count: int = 4,
        spatial_cutoff: float = 6.0,
        covalent_cutoff: float = 64.0,
        degree_log_means: Sequence[float] | None = None,
        maximum_triplets: int = 1_000_000,
        maximum_pairwise_bytes: int = 134_217_728,
        allow_dense_global: bool = False,
    ) -> None:
        """Build a fixed atomic candidate without constructing topology or reading datasets.

        Args:
            input_dim: Generic scalar atom-feature width F.
            hidden_dim: Shared scalar output width D; also the internal scalar width.
            layers: Number of atomic interaction blocks, not equivalent physical reach by family.
            dropout: Scalar/residual dropout probability in [0,1).
            distance_scale: Baseline scalar distance normalization in Å.
            message_chunk_size: LF matrix-filter message chunk budget.
            encoder_type: Exact AtomicEncoderType candidate, never a combined architecture.
            radial_count: Radial basis count; DimeNet uses three spherical orders.
            vector_channels: Persistent GVP polar channels; PaiNN ties its channels to D.
            basis_count: NNConv basis count, inactive for the full matrix variant.
            spatial_cutoff: Fixed spatial basis support in Å, without adding/removing edges.
            covalent_cutoff: Explicit finite DimeNet covalent support in Å. Supplied covalent
                distances beyond this bound fail, rather than silently deleting a relation.
            degree_log_means: Training-only mean log(1+degree), covalent then spatial, for PNA.
            maximum_triplets: Hard typed-triplet construction bound, not a total VRAM estimate.
            maximum_pairwise_bytes: Bound for principal triplet/path/attention tensors in bytes.
            allow_dense_global: Explicit consumer permission for the global-only dense exception.

        Raises:
            ValueError: If a required training statistic or explicit global permission is absent.
        """
        super().__init__()
        self.kind = AtomicEncoderType(encoder_type)
        self.require_public_api(encoder_type)
        self.hidden_dim, self.layers = hidden_dim, layers
        self.distance_scale = distance_scale
        self.maximum_triplets = maximum_triplets
        self.maximum_pairwise_bytes = maximum_pairwise_bytes
        self.spatial_cutoff, self.covalent_cutoff = spatial_cutoff, covalent_cutoff
        self.diagnostics: dict[str, Tensor] = {}
        self.reference: GatedAtomicEncoder | None = None
        self.directional: Any = None
        self.radial_count = radial_count
        self.readout: nn.Module = nn.Identity()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.normalizations = nn.ModuleList(nn.LayerNorm(hidden_dim) for _ in range(layers))
        self.dropout = nn.Dropout(dropout)
        self.covalent = nn.ModuleList()
        self.spatial = nn.ModuleList()
        self.roots = nn.ModuleList()
        self.covalent_conditioners = nn.ModuleList()
        self.spatial_conditioners = nn.ModuleList()
        self.radial_bases = nn.ModuleList()
        self.covalent_message_sources = nn.ModuleList()
        self.spatial_message_sources = nn.ModuleList()

        # Controls use the historical encoder verbatim under a declared source mask.
        # Its weights are copied by WisdomV1 after the shared initialization finishes.

        if self.kind in {
            AtomicEncoderType.BASELINE_NO_ATTRIBUTES,
            AtomicEncoderType.BASELINE_RADIAL,
        }:
            self.reference = GatedAtomicEncoder(
                input_dim,
                hidden_dim,
                layers,
                dropout,
                distance_scale,
                message_chunk_size,
            )
            self.input_projection.requires_grad_(False)
            self.normalizations.requires_grad_(False)
            if self.kind is AtomicEncoderType.BASELINE_NO_ATTRIBUTES:
                self.reference.spatial_conditioners.requires_grad_(False)
                self.reference.covalent_conditioners.requires_grad_(False)
            return

        if (
            self.kind
            in {
                AtomicEncoderType.STRUCTURAL_GLOBAL,
                AtomicEncoderType.GLOBAL_NO_STRUCTURE,
            }
            and not allow_dense_global
        ):
            raise ValueError("global atomic attention requires explicit allow_dense_global=True")

        if self.kind in {AtomicEncoderType.DIMENET, AtomicEncoderType.DIMENET_PLUS_PLUS}:
            self.directional = DimeNet(
                hidden_dim,
                hidden_dim,
                hidden_channels=hidden_dim,
                num_layers=layers,
                num_radial=radial_count,
                num_spherical=3,
                cutoff=spatial_cutoff,
                edge_channels=6,
                interaction_type=(
                    "bilinear" if self.kind is AtomicEncoderType.DIMENET else "factorized"
                ),
                max_triplet_bytes=maximum_pairwise_bytes,
            )
            # External LF bases own both declared relation supports. The unused internal basis
            # remains serializable but is frozen and excluded from trainable capacity.

            self.directional.basis.requires_grad_(False)
            self.radial_bases.extend(
                [
                    DimeNetBasis(radial_count, 3, covalent_cutoff),
                    DimeNetBasis(radial_count, 3, spatial_cutoff),
                ]
            )
            self.normalizations.requires_grad_(False)
            return

        if self.kind is AtomicEncoderType.PNA_FULL and degree_log_means is None:
            raise ValueError("PNA degree scalers require frozen training-only degree statistics")
        degree_means = tuple(degree_log_means or (1.0, 1.0))
        for index in range(layers):
            for relation, edge_width, target, messages in (
                (0, 3, self.covalent, self.covalent_message_sources),
                (1, 4, self.spatial, self.spatial_message_sources),
            ):
                # The message stream is identical in MPNN and PNA. Only LF aggregation and
                # node-update operators change; sender/receiver order is not silently swapped.

                seed = torch.initial_seed() ^ ((index + 1) * 104729 + relation)
                with torch.random.fork_rng(devices=[]):
                    torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
                    target.append(
                        self._operator(
                            edge_width,
                            hidden_dim,
                            radial_count,
                            vector_channels,
                            basis_count,
                            message_chunk_size,
                            dropout,
                            degree_means[relation],
                        )
                    )
                if self.kind.value.startswith("pna"):
                    with torch.random.fork_rng(devices=[]):
                        torch.set_rng_state(torch.Generator().manual_seed(seed).get_state())
                        source = ECMP(
                            hidden_dim,
                            hidden_dim,
                            num_layers=1,
                            edge_dim=edge_width,
                            dropout=dropout,
                            aggregation="mean",
                        )
                    source.input_proj.requires_grad_(False)
                    source.update_mlps.requires_grad_(False)
                    source.norms.requires_grad_(False)
                    messages.append(source)
                    target[index].pre_mlp.requires_grad_(False)
            self.roots.append(nn.Linear(hidden_dim, hidden_dim, bias=False))
            conditioned = self.kind in {
                AtomicEncoderType.GCN_CONDITIONED,
                AtomicEncoderType.SAGE_CONDITIONED,
            }
            if conditioned:
                self.covalent_conditioners.append(
                    nn.Sequential(nn.Linear(3, 8), nn.SiLU(), nn.Linear(8, 1, bias=False))
                )
                self.spatial_conditioners.append(
                    nn.Sequential(nn.Linear(4, 8), nn.SiLU(), nn.Linear(8, 1, bias=False))
                )
            if self.kind not in {
                AtomicEncoderType.SAGE_STRUCTURAL,
                AtomicEncoderType.SAGE_CONDITIONED,
                AtomicEncoderType.RGCN_CANONICAL,
            }:
                self.roots[index].requires_grad_(False)
            if self.kind in {
                AtomicEncoderType.GIN,
                AtomicEncoderType.GINE,
                AtomicEncoderType.GINE_NO_ATTRIBUTES,
            }:
                self.spatial[index].mlp.requires_grad_(False)
            if self.kind in {
                AtomicEncoderType.PNA_LIGHT,
                AtomicEncoderType.PNA_IDENTITY,
                AtomicEncoderType.PNA_FULL,
            }:
                self.spatial[index].post_mlp.requires_grad_(False)
            if self.kind in {AtomicEncoderType.MPNN_MEAN, AtomicEncoderType.MPNN_SUM}:
                for operator in (self.covalent[index], self.spatial[index]):
                    operator.input_proj.requires_grad_(False)
                self.spatial[index].update_mlps.requires_grad_(False)
                self.spatial[index].norms.requires_grad_(False)
        if self.kind in {
            AtomicEncoderType.GIN,
            AtomicEncoderType.GINE,
            AtomicEncoderType.GINE_NO_ATTRIBUTES,
            AtomicEncoderType.MPNN_MEAN,
            AtomicEncoderType.MPNN_SUM,
            AtomicEncoderType.PNA_LIGHT,
            AtomicEncoderType.PNA_IDENTITY,
            AtomicEncoderType.PNA_FULL,
            AtomicEncoderType.GCN_CANONICAL,
        }:
            self.normalizations.requires_grad_(False)
        if self.kind is AtomicEncoderType.GCN_CANONICAL:
            # The canonical control shares W across typed incidences and adds one self-loop,
            # unlike the relation-specific residual GCN candidates. Do not label these identical.

            for index in range(layers):
                self.covalent[index] = GCNLayer(
                    hidden_dim, hidden_dim, bias=False, add_self_loops=True
                )
                self.spatial[index].requires_grad_(False)
        if self.kind is AtomicEncoderType.GVP:
            self.readout = GVP(hidden_dim, vector_channels, hidden_dim, 0)

    def _operator(
        self,
        edge_width: int,
        width: int,
        radial: int,
        vectors: int,
        bases: int,
        chunk: int,
        dropout: float,
        degree_mean: float,
    ) -> Any:
        """Instantiate one concrete LF relation operator; never reproduce its message equation.

        Args:
            edge_width: Relation's gated scalar attribute width, three or four.
            width: Scalar node width D.
            radial: Number of radial basis functions.
            vectors: Persistent GVP vector-channel width.
            bases: NNConv matrix-basis count.
            chunk: Full NNConv temporary message chunk bound.
            dropout: Common scalar/dropout probability.
            degree_mean: Frozen training mean log(1+degree) for this relation.

        Returns:
            Eager public LF operator with fully allocated trainable parameters.

        Raises:
            ValueError: If no public composition is declared for the selected candidate.
        """
        kind = self.kind.value
        if kind.startswith("gcn"):
            return GCNLayer(width, width, bias=False, add_self_loops=False)
        if kind.startswith("sage"):
            return GraphSAGELayer(width, width, root_weight=False, bias=False)
        if kind.startswith("rgcn"):
            return RelationalGCNLayer(width, width, 1, root_weight=False, bias=False)
        if self.kind is AtomicEncoderType.GIN:
            return GINLayer(width, width, epsilon=0, trainable_epsilon=False)
        if kind.startswith("gine"):
            return GINELayer(width, width, edge_width, epsilon=0, trainable_epsilon=False)
        if kind.startswith("mpnn"):
            return ECMP(
                width,
                width,
                num_layers=1,
                edge_dim=edge_width,
                aggregation="sum" if self.kind is AtomicEncoderType.MPNN_SUM else "mean",
                dropout=dropout,
            )
        if kind.startswith("nnconv"):
            return NNConvLayer(
                width,
                width,
                edge_width,
                hidden_channels=width,
                num_bases=bases if self.kind is AtomicEncoderType.NNCONV_BASES else None,
                root_weight=False,
                center_conditioner=(self.kind is AtomicEncoderType.NNCONV_BASES),
                message_chunk_size=chunk,
            )
        if kind.startswith("pna"):
            return PNALayer(
                width,
                width,
                edge_channels=edge_width,
                activation="relu",
                aggregators=("mean", "max", "std")
                if self.kind is AtomicEncoderType.PNA_LIGHT
                else ("mean", "max", "min", "std"),
                scalers=("identity", "amplification", "attenuation")
                if self.kind is AtomicEncoderType.PNA_FULL
                else ("identity",),
                average_log_degree=max(degree_mean, 1.0e-12),
                dropout=dropout,
            )
        if kind.startswith("gatv2"):
            heads = 4 if self.kind is AtomicEncoderType.GATV2_4 else 2
            return GATv2Layer(
                width,
                width // heads,
                num_heads=heads,
                edge_channels=edge_width,
                add_self_loops=False,
                bias=False,
            )
        if kind.startswith("schnet"):
            return SchNetLayer(
                width,
                radial,
                self.spatial_cutoff,
                edge_channels=edge_width,
                cutoff=self.spatial_cutoff if edge_width == 4 else None,
            )
        if self.kind is AtomicEncoderType.PAINN:
            return PaiNNLayer(
                width,
                radial,
                self.spatial_cutoff,
                edge_channels=edge_width,
                cutoff=self.spatial_cutoff if edge_width == 4 else None,
            )
        if self.kind is AtomicEncoderType.GVP:
            return GVPGNNLayer(width, vectors, edge_channels=edge_width, dropout=dropout)
        if kind.startswith("egnn"):
            return EGNNLayer(
                width,
                width,
                edge_channels=edge_width,
                message_channels=width,
                update_coordinates=self.kind is AtomicEncoderType.EGNN_LATENT,
                coordinate_aggregation="sum",
                message_aggregation="sum",
            )
        if kind.startswith("transformer"):
            return GraphTransformerLayer(
                width,
                width,
                num_heads=2,
                edge_channels=edge_width,
                add_self_loops=False,
                feature_dropout=dropout,
            )
        if self.kind in {
            AtomicEncoderType.STRUCTURAL_GLOBAL,
            AtomicEncoderType.GLOBAL_NO_STRUCTURE,
        }:
            return StructuralGraphTransformerLayer(
                width, 2, 128, 32, dropout, self.maximum_pairwise_bytes
            )
        raise ValueError(f"no public LF composition for atomic encoder {self.kind.value}")

    def forward(
        self,
        atom_features: Tensor,
        edge_index: Tensor,
        edge_is_spatial: Tensor,
        edge_is_covalent: Tensor,
        edge_distance: Tensor,
        edge_bond_order: Tensor,
        edge_same_residue: Tensor,
        edge_same_chain: Tensor,
        edge_residue_separation: Tensor,
        gates: Mapping[str, Tensor],
        positions: Tensor | None = None,
        atom_batch: Tensor | None = None,
    ) -> Tensor:
        """Apply fixed relation topology and return scalar atom embeddings in unchanged order.

        Args:
            atom_features: Gated invariant scalar descriptors [N,F].
            edge_index: Unique undirected original pairs [2,E].
            edge_is_spatial: Spatial membership bool [E].
            edge_is_covalent: Covalent membership bool [E]; overlap is retained.
            edge_distance: Original finite center distances [E], Å.
            edge_bond_order: Original numeric covalent bond orders [E].
            edge_same_residue: Same-residue flags [E].
            edge_same_chain: Same-chain flags [E].
            edge_residue_separation: Absolute sequence separations [E].
            gates: One-forward shared semantic scalar gate values.
            positions: Physical atom positions [N,3], Å, required by geometric candidates.
            atom_batch: Explicit protein ownership [N], required by global attention.

        Returns:
            Invariant scalar states [N,D]. Physical positions, topology and targets are untouched.

        Raises:
            ValueError: Required geometry/ownership is missing or outside the declared contract.
            MemoryError: A bounded LF directional/global representation exceeds its byte guard.
        """
        arguments = (
            atom_features,
            edge_index,
            edge_is_spatial,
            edge_is_covalent,
            edge_distance,
            edge_bond_order,
            edge_same_residue,
            edge_same_chain,
            edge_residue_separation,
            gates,
        )
        if self.reference is not None:
            return self.reference(*arguments)
        hidden = self.input_projection(atom_features)
        covalent_attributes = torch.stack(
            (
                (edge_distance / self.distance_scale).clamp(max=2)
                * gates["edge.covalent.distance"],
                edge_bond_order.float().clamp(0, 3) / 3 * gates["edge.covalent.bond_order"],
                edge_same_residue.float() * gates["edge.covalent.same_residue"],
            ),
            dim=1,
        )
        spatial_attributes = torch.stack(
            (
                (edge_distance / self.distance_scale).clamp(max=2) * gates["edge.spatial.distance"],
                edge_same_residue.float() * gates["edge.spatial.same_residue"],
                edge_same_chain.float() * gates["edge.spatial.same_chain"],
                (edge_residue_separation.float() / 32).clamp(max=2)
                * gates["edge.spatial.residue_separation"],
            ),
            dim=1,
        )
        branches = []
        for relation, mask, attributes in (
            ("covalent", edge_is_covalent, covalent_attributes),
            ("spatial", edge_is_spatial, spatial_attributes),
        ):
            pairs = edge_index[:, mask.bool()]
            directed = torch.cat((pairs, pairs.flip(0)), dim=1)
            branches.append(
                (
                    directed,
                    attributes[mask.bool()].repeat(2, 1),
                    edge_distance[mask.bool()].repeat(2),
                    gates[f"atomic_graph.{relation}"],
                    gates[f"edge.{relation}.distance"],
                )
            )
        if self.directional is not None:
            if positions is None:
                raise ValueError("directional atom encoding requires physical atom_positions")
            return self._directional(hidden, branches, positions)

        vector_width = (
            self.hidden_dim
            if self.kind is AtomicEncoderType.PAINN
            else (self.covalent[0].vector_channels if self.kind is AtomicEncoderType.GVP else 0)
        )
        vectors = hidden.new_zeros((len(hidden), vector_width, 3))
        latent_positions = positions
        for index, (raw_covalent, raw_spatial, norm) in enumerate(
            zip(
                self.covalent,
                self.spatial,
                self.normalizations,
                strict=True,
            )
        ):
            covalent: Any = raw_covalent
            spatial: Any = raw_spatial
            if self.kind is AtomicEncoderType.GCN_CANONICAL:
                edges = torch.cat([branch[0] for branch in branches], dim=1)
                modulation = torch.cat([branch[3].expand(len(branch[2])) for branch in branches])
                hidden = torch.nn.functional.silu(covalent(hidden, edges, modulation))
                continue
            operators = (covalent, spatial)
            aggregates, vector_deltas, position_deltas = [], [], []
            for branch_id, (raw_operator, branch) in enumerate(
                zip(operators, branches, strict=True)
            ):
                operator: Any = raw_operator
                edges, attributes, distances, relation_gate, distance_gate = branch
                if self.kind in {
                    AtomicEncoderType.GIN,
                    AtomicEncoderType.GINE,
                    AtomicEncoderType.GINE_NO_ATTRIBUTES,
                }:
                    messages = (
                        operator.messages(hidden, edges)
                        if self.kind is AtomicEncoderType.GIN
                        else operator.messages(hidden, edges, attributes)
                    )
                    delta = operator.aggregate(messages, edges, len(hidden))
                elif self.kind in {AtomicEncoderType.MPNN_MEAN, AtomicEncoderType.MPNN_SUM}:
                    messages = operator.messages(hidden, edges, attributes)
                    delta = operator.aggregate(messages, edges, len(hidden))
                elif self.kind in {
                    AtomicEncoderType.PNA_LIGHT,
                    AtomicEncoderType.PNA_IDENTITY,
                    AtomicEncoderType.PNA_FULL,
                }:
                    message_sources = (
                        self.covalent_message_sources
                        if branch_id == 0
                        else self.spatial_message_sources
                    )
                    message_source: Any = message_sources[index]
                    delta = operator.aggregate(
                        message_source.messages(hidden, edges, attributes),
                        edges,
                        len(hidden),
                        degree_dtype=hidden.dtype,
                    )
                elif self.kind.value.startswith(("gcn", "sage")):
                    weights = None
                    if len(self.covalent_conditioners):
                        conditioner = (
                            self.covalent_conditioners
                            if branch_id == 0
                            else self.spatial_conditioners
                        )[index]
                        weights = (
                            1 + conditioner(attributes) - conditioner(torch.zeros_like(attributes))
                        ).squeeze(-1)
                    delta = operator(hidden, edges, message_weights=weights)
                elif self.kind.value.startswith("rgcn"):
                    delta = operator(
                        hidden,
                        edges,
                        torch.zeros(edges.shape[1], device=edges.device, dtype=torch.long),
                    )
                elif self.kind.value.startswith("schnet"):
                    delta = (
                        operator(
                            hidden, edges, distances, attributes, distance_weights=distance_gate
                        )
                        - hidden
                    )
                elif self.kind in {AtomicEncoderType.GVP, AtomicEncoderType.PAINN}:
                    if positions is None:
                        raise ValueError(
                            "persistent vector encoders require physical atom_positions"
                        )
                    _, directions = edge_geometry(positions, edges)
                    if self.kind is AtomicEncoderType.GVP:
                        scalar_out, vector_out = operator(
                            hidden, vectors, edges, directions[:, None, :], attributes
                        )
                    else:
                        scalar_out, vector_out = operator(
                            hidden,
                            vectors,
                            edges,
                            distances,
                            directions,
                            attributes,
                            distance_weights=distance_gate,
                        )
                    delta = scalar_out - hidden
                    vector_deltas.append(relation_gate * (vector_out - vectors))
                elif self.kind in {AtomicEncoderType.EGNN_FIXED, AtomicEncoderType.EGNN_LATENT}:
                    if latent_positions is None:
                        raise ValueError(
                            "EGNN requires physical atom_positions as its initial state"
                        )
                    # LF requires features and coordinates to share dtype. Keep coordinate
                    # arithmetic in FP32 rather than rounding physical positions to BF16.

                    controlled_positions = distance_gate * latent_positions.float()
                    with torch.autocast(device_type=hidden.device.type, enabled=False):
                        scalar_out, position_out = operator(
                            hidden.float(), edges, controlled_positions, attributes.float()
                        )
                    delta = scalar_out.to(hidden.dtype) - hidden
                    position_deltas.append(relation_gate * (position_out - controlled_positions))
                elif self.kind in {
                    AtomicEncoderType.STRUCTURAL_GLOBAL,
                    AtomicEncoderType.GLOBAL_NO_STRUCTURE,
                }:
                    delta = self._global(operator, hidden, edges, atom_batch) - hidden
                else:
                    delta = operator(hidden, edges, attributes)
                    if self.kind.value.startswith("transformer"):
                        delta = delta - hidden
                # Gates act after min/max/std and after complete residual block differences.
                # A closed relation cannot continue sending messages through biases or roots.

                aggregates.append(relation_gate * delta)
            combined = aggregates[0] + aggregates[1]
            if self.kind in {
                AtomicEncoderType.GIN,
                AtomicEncoderType.GINE,
                AtomicEncoderType.GINE_NO_ATTRIBUTES,
                AtomicEncoderType.MPNN_MEAN,
                AtomicEncoderType.MPNN_SUM,
                AtomicEncoderType.PNA_LIGHT,
                AtomicEncoderType.PNA_IDENTITY,
                AtomicEncoderType.PNA_FULL,
            }:
                hidden = covalent.update(hidden, combined)
            elif self.kind in {
                AtomicEncoderType.SAGE_STRUCTURAL,
                AtomicEncoderType.SAGE_CONDITIONED,
                AtomicEncoderType.RGCN_CANONICAL,
            }:
                hidden = torch.nn.functional.silu(norm(self.roots[index](hidden) + combined))
            else:
                hidden = torch.nn.functional.silu(norm(hidden + self.dropout(combined)))
            if vector_deltas:
                vectors = vectors + vector_deltas[0] + vector_deltas[1]
            if position_deltas:
                assert latent_positions is not None
                # Coordinate displacement uses total fixed incidence degree, not one mean
                # per relation; latent geometry never changes the supplied graph or surface.

                degree = (
                    torch.stack(
                        [torch.bincount(branch[0][1], minlength=len(hidden)) for branch in branches]
                    )
                    .sum(dim=0)
                    .clamp_min(1)
                    .to(hidden.dtype)
                )
                latent_positions = (
                    latent_positions + (position_deltas[0] + position_deltas[1]) / degree[:, None]
                )
        if self.kind is AtomicEncoderType.GVP:
            hidden, _ = self.readout(hidden, vectors)
        if vector_width:
            self.diagnostics["vector_rms"] = vectors.detach().float().square().mean().sqrt()
        if self.kind is AtomicEncoderType.EGNN_LATENT:
            assert positions is not None and latent_positions is not None
            self.diagnostics["latent_displacement_rms"] = (
                (latent_positions - positions).detach().float().square().mean().sqrt()
            )
        return hidden

    def _directional(self, hidden: Tensor, branches: list[Any], positions: Tensor) -> Tensor:
        """Compose public typed-triplet bases, preserving all mixed relation paths.

        Args:
            hidden: Projected scalar atom states [N,D].
            branches: Covalent/spatial directed edges, attributes, lengths and source gates.
            positions: Unchanged physical coordinates [N,3], Å.

        Returns:
            LF DimeNet node readout [N,D]; memory/support violations raise explicitly.
        """
        assert self.directional is not None
        edges = torch.cat([branch[0] for branch in branches], dim=1)
        distances = torch.cat([branch[2] for branch in branches])
        edge_weights = torch.cat([branch[3].expand(len(branch[2])) for branch in branches])
        attributes = []
        for index, branch in enumerate(branches):
            scalar = torch.nn.functional.pad(branch[1], (0, 4 - branch[1].shape[1]))
            relation = scalar.new_zeros((len(scalar), 2))
            relation[:, index] = 1
            attributes.append(torch.cat((scalar, relation), dim=1))
            if torch.any(
                branch[2] >= (self.covalent_cutoff if index == 0 else self.spatial_cutoff)
            ):
                raise ValueError("directional relation support would remove a supplied atomic edge")
        triplets = GraphTriplets.from_edges(edges, len(hidden), max_triplets=self.maximum_triplets)
        incoming, _ = triplets.validate(edges.shape[1], hidden.device)
        if incoming.numel() * max(self.hidden_dim, self.radial_count * 3) * 4 > (
            self.maximum_pairwise_bytes
        ):
            raise MemoryError(
                "typed directional bases exceed the configured principal tensor budget"
            )
        angles = triplets.angles(positions, edges)
        relation_is_covalent = (
            torch.arange(edges.shape[1], device=edges.device) < branches[0][0].shape[1]
        )
        radial, spherical = [], []
        for basis, branch in zip(self.radial_bases, branches, strict=True):
            r, s = basis(distances, angles, triplets, distance_weights=branch[4])
            radial.append(r)
            spherical.append(s)
        r = torch.where(relation_is_covalent[:, None], radial[0], radial[1])
        s = torch.where(relation_is_covalent[incoming, None], spherical[0], spherical[1])
        self.diagnostics["typed_triplets"] = hidden.new_tensor(float(len(incoming)))
        return self.directional.forward_from_basis(
            hidden, edges, r, s, triplets, torch.cat(attributes), message_weights=edge_weights
        )

    def _global(
        self, operator: Any, hidden: Tensor, edges: Tensor, atom_batch: Tensor | None
    ) -> Tensor:
        """Apply bounded public structural attention separately within each declared protein.

        Args:
            operator: Concrete LF global structural attention block.
            hidden: Scalar atom states [N,D].
            edges: Directed edges for exactly one relation, [2,E].
            atom_batch: Explicit protein ownership [N]; disconnected components are NOT proteins.

        Returns:
            Concatenated [N,D] states with original ordering and no inter-protein attention.

        Raises:
            ValueError: Protein ownership is missing or an edge crosses protein boundaries.
            MemoryError: LF's path/score guard rejects the proposed protein size.
        """
        if atom_batch is None:
            raise ValueError("global atomic attention requires explicit atom_batch ownership")
        if torch.any(atom_batch[edges[0]] != atom_batch[edges[1]]):
            raise ValueError("atomic edge crosses declared protein boundaries")
        output = torch.zeros_like(hidden)
        for protein in torch.unique(atom_batch, sorted=True):
            ids = torch.where(atom_batch == protein)[0]
            inverse = torch.full((len(hidden),), -1, dtype=torch.long, device=hidden.device)
            inverse[ids] = torch.arange(len(ids), device=hidden.device)
            selected = edges[:, atom_batch[edges[0]] == protein]
            encoding = GraphStructuralEncoding.from_edges(
                inverse[selected].cpu(),
                len(ids),
                max_pairwise_bytes=self.maximum_pairwise_bytes,
            )
            codes = encoding.spatial_indices.to(hidden.device)
            indegree, outdegree = (
                encoding.in_degree.to(hidden.device),
                encoding.out_degree.to(hidden.device),
            )
            if self.kind is AtomicEncoderType.GLOBAL_NO_STRUCTURE:
                codes, indegree, outdegree = (
                    torch.zeros_like(codes),
                    torch.zeros_like(indegree),
                    torch.zeros_like(outdegree),
                )
            output[ids] = operator(hidden[ids][None], codes[None], indegree[None], outdegree[None])[
                0
            ]
        return output

    @staticmethod
    def require_public_api(encoder_type: str) -> None:
        """Reject older editable LF APIs before expensive data ingestion or graph construction.

        Args:
            encoder_type: Exact candidate whose public message/radial contract is required.

        Raises:
            RuntimeError: The installed LF checkout predates the necessary public API. There is
                deliberately no compatibility shim, equation reimplementation or fallback layer.
        """
        kind = AtomicEncoderType(encoder_type)
        required: list[tuple[Any, str]] = []
        if kind is AtomicEncoderType.GCN_CONDITIONED:
            required.append((GCNLayer.forward, "message_weights"))
        if kind is AtomicEncoderType.SAGE_CONDITIONED:
            required.append((GraphSAGELayer.forward, "message_weights"))
        if kind in {AtomicEncoderType.SCHNET, AtomicEncoderType.SCHNET_RADIAL}:
            required.append((SchNetLayer.forward, "distance_weights"))
        if kind is AtomicEncoderType.PAINN:
            required.append((PaiNNLayer.forward, "distance_weights"))
        if kind in {AtomicEncoderType.DIMENET, AtomicEncoderType.DIMENET_PLUS_PLUS}:
            required.extend(
                (
                    (DimeNetBasis.forward, "distance_weights"),
                    (DimeNet.forward_from_basis, "message_weights"),
                )
            )
        for method, parameter in required:
            if parameter not in signature(method).parameters:
                raise RuntimeError(
                    f"atomic encoder {kind.value} requires the current public LambdaForge API: "
                    f"{method.__qualname__}(..., {parameter}=...). Update LambdaForge, then run "
                    "./install.sh locally or lf clusters bootstrap <cluster> --project . remotely."
                )

    @staticmethod
    def training_degree_statistics(
        samples: Iterable[Mapping[str, Any]], spatial_k: int
    ) -> tuple[float, float]:
        """Fit PNA log-degree means on the exact training view, never validation or local GT.

        Args:
            samples: Validated training samples in the dataset's deterministic order.
            spatial_k: The same nested topology prefix used by WisdomCollator.

        Returns:
            Atom-weighted mean log(1+degree), covalent then spatial. Empty relations return zero.
        """
        totals = torch.zeros(2, dtype=torch.float64)
        count = 0
        for sample in samples:
            nodes = len(sample["atomic_numbers"])
            edges = sample["atom_edge_index"]
            masks = (sample["atom_edge_is_covalent"], sample["atom_edge_is_spatial"])
            if "atom_edge_spatial_rank" in sample:
                masks = (masks[0], masks[1] & (sample["atom_edge_spatial_rank"] <= spatial_k))
            for relation, mask in enumerate(masks):
                degree = torch.bincount(edges[:, mask.bool()].flatten(), minlength=nodes)
                totals[relation] += degree.double().log1p().sum()
            count += nodes
        return float(totals[0] / max(count, 1)), float(totals[1] / max(count, 1))
