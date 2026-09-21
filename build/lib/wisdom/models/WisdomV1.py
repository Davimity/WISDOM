"""WISDOM v1 semantic-gated atomic and DiffusionNet surface backbone."""

from __future__ import annotations

from typing import Any, ClassVar
from collections.abc import Mapping, Sequence

import torch
from lambdaforge.nn.models import MLP, Model
from lambdaforge.nn.pooling import SparseMaxPooling
from torch import Tensor, nn

from wisdom.models.GatedAtomicEncoder import GatedAtomicEncoder
from wisdom.models.LinearInitialization import LinearInitialization
from wisdom.models.ResidualInitialization import ResidualInitialization
from wisdom.models.EmbeddingInitialization import EmbeddingInitialization
from wisdom.models.DiffusionTimeInitialization import DiffusionTimeInitialization
from wisdom.models.InteractionRound import InteractionRound
from wisdom.models.PhysicalEmbeddingInitializer import PhysicalEmbeddingInitializer
from wisdom.models.SurfaceAtomTransfer import SurfaceAtomTransfer
from wisdom.models.SurfaceAtomFeedback import SurfaceAtomFeedback
from wisdom.models.VectorAtomicState import VectorAtomicState
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.gating.SemanticGateRegistry import SemanticGateRegistry


class WisdomV1(Model):
    """Learn which generic physical inputs a fixed DiffusionNet/MAX backbone needs."""

    ARCHITECTURE_NAME: ClassVar[str] = "semantic-gated-diffusionnet"
    STRUCTURAL_SCHEMA_VERSION: ClassVar[str] = "3.0"

    output_schema: ClassVar[dict[str, Any]] = {
        "logits":             "Tensor[B]",
        "surface_logits":     "Tensor[M]",
        "surface_embeddings": "Tensor[M,H]",
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
        surface_geometry_transfer  : bool = False,
        surface_chunk_size         : int = 8192,
        atomic_message_chunk_size  : int = 65536,
        vector_atomic_channels     : int = 0,
        interaction_round          : str = "single",
        atomic_edge_distance_scale : float = 6.0,
        embedding_initialization    : str = "current",
        gate_initial_active         : float = 0.95,
        diffusion_time_initialization: str = "current",
        diffusion_residual_initialization: str = "current",
        atomic_residual_initialization: str = "current",
        neutral_edge_initialization : bool = False,
        neutral_transfer_initialization: bool = False,
        physics_distance_initialization: bool = False,
        local_head_weight_std       : float | None = None,
        linear_initialization       : str = "current",
    ) -> None:
        """Build the complete semantic input superset and fixed v1 hypothesis.

        Element identity is always active. Every other generic atom descriptor, each physical
        graph branch, its predictive edge enrichments, transfer orientation/content, the optional
        point-geometry transfer interaction, the complete atom context, and every curvature
        descriptor at every stored scale receive a global Hard-Concrete gate. Tensor widths remain
        fixed; a closed gate writes zeros rather than rebuilding the network.

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
            surface_geometry_transfer: Whether existing gated surface curvature descriptors
                condition the relative atom-to-surface attention scores.
            surface_chunk_size: Maximum points per transfer activation chunk.
            atomic_message_chunk_size: Recorded edge-work budget for reproducibility.
            vector_atomic_channels: Equivariant atomic vector channels; zero preserves the scalar
                baseline and positive values activate isolated spike C.
            interaction_round: ``single`` baseline, equally deep ``depth_control``, or one
                ``bidirectional`` surface-to-atom-to-surface feedback round.
            atomic_edge_distance_scale: Atomic edge-distance normalization scale in ångströms.
            embedding_initialization: ``current``, ``fan_scaled``, ``small_normal``, or
                task-independent ``physical`` categorical table initialization.
            gate_initial_active: Initial analytic probability ``P(z>0)`` for every semantic gate.
            diffusion_time_initialization: ``current``, ``broader_lengths``, or ``same_scale``
                initial DiffusionNet time schedule.
            diffusion_residual_initialization: ``current``, ``rezero``, or ``layerscale`` for
                DiffusionNet residual branches.
            atomic_residual_initialization: Equivalent residual policy for atomic graph updates.
            neutral_edge_initialization: Start edge-attribute conditioners as zero corrections.
            neutral_transfer_initialization: Start orientation/content transfer as zero corrections.
            physics_distance_initialization: Start atom transfer from a trainable ``-d/R`` prior.
            local_head_weight_std: Optional small-normal local-head weight scale.
            linear_initialization: ``current``, ``activation_aware``, or ``orthogonal`` policy.

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
        if vector_atomic_channels < 0:
            raise ValueError("vector_atomic_channels cannot be negative")
        if local_head_weight_std is not None and local_head_weight_std <= 0.0:
            raise ValueError("local_head_weight_std must be positive when supplied")

        embedding_mode = EmbeddingInitialization(embedding_initialization)
        diffusion_time_mode = DiffusionTimeInitialization(diffusion_time_initialization)
        diffusion_residual_mode = ResidualInitialization(diffusion_residual_initialization)
        atomic_residual_mode = ResidualInitialization(atomic_residual_initialization)
        linear_mode = LinearInitialization(linear_initialization)
        interaction_mode = InteractionRound(interaction_round)

        residue_width = embedding_dim if residue_embedding_dim is None else residue_embedding_dim
        if residue_width < 1:
            raise ValueError("residue_embedding_dim must be positive")

        self.curvature_scale_count = curvature_features // 3
        gate_definitions = self._gate_definitions(
            self.curvature_scale_count,
            surface_geometry_transfer,
        )
        self.semantic_gates = SemanticGateRegistry(
            gate_definitions,
            initial_active=gate_initial_active,
        )

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
            residual_initial_scale=self._residual_scale(atomic_residual_mode),
        )
        self.surface_atom_transfer = SurfaceAtomTransfer(
            hidden_dim        = hidden_dim,
            radius            = surface_atom_radius,
            chunk_size        = surface_chunk_size,
            point_feature_dim = (
                4 * self.curvature_scale_count if surface_geometry_transfer else 0
            ),
            physics_distance_initialization=physics_distance_initialization,
        )
        self.vector_atomic_state = (
            VectorAtomicState(hidden_dim, vector_atomic_channels)
            if vector_atomic_channels > 0
            else None
        )
        self.surface_atom_feedback = (
            SurfaceAtomFeedback(hidden_dim)
            if interaction_mode is InteractionRound.BIDIRECTIONAL
            else None
        )
        self.second_surface_transfer = (
            SurfaceAtomTransfer(
                hidden_dim=hidden_dim,
                radius=surface_atom_radius,
                chunk_size=surface_chunk_size,
                point_feature_dim=(
                    4 * self.curvature_scale_count if surface_geometry_transfer else 0
                ),
            )
            if interaction_mode is not InteractionRound.SINGLE
            else None
        )
        self.second_surface_projection = (
            MLP(
                in_features=2 * hidden_dim,
                out_features=hidden_dim,
                hidden=[hidden_dim],
                dropout=dropout,
                residual=True,
            )
            if interaction_mode is not InteractionRound.SINGLE
            else None
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
            initial_times=self._diffusion_times(surface_layers, diffusion_time_mode),
            residual_initial_scale=self._residual_scale(diffusion_residual_mode),
        )
        self.second_surface_encoder: nn.Module | None = (
            DiffusionSurfaceEncoder(
                hidden_dim,
                layers=surface_layers,
                dropout=dropout,
                initial_times=self._diffusion_times(surface_layers, diffusion_time_mode),
                residual_initial_scale=self._residual_scale(diffusion_residual_mode),
            )
            if interaction_mode is not InteractionRound.SINGLE
            else None
        )
        self.local_head         = nn.Linear(hidden_dim, 1)
        self.global_max_pooling = SparseMaxPooling()

        self._initialize_linears(linear_mode)
        self._initialize_embeddings(embedding_mode)
        if neutral_edge_initialization:
            self.atomic_encoder.initialize_neutral_edge_residuals()
        if neutral_transfer_initialization:
            self.surface_atom_transfer.initialize_neutral_residuals()
            if self.second_surface_transfer is not None:
                self.second_surface_transfer.initialize_neutral_residuals()
        if physics_distance_initialization:
            self.surface_atom_transfer.initialize_physics_distance()
        if local_head_weight_std is not None:
            nn.init.normal_(self.local_head.weight, mean=0.0, std=local_head_weight_std)

        self.hidden_dim                  = hidden_dim
        self.curvature_features          = curvature_features
        self.atomic_layers               = atomic_layers
        self.projection_depth            = projection_depth
        self.surface_layers              = surface_layers
        self.atom_spatial_k              = atom_spatial_k
        self.surface_atom_k              = surface_atom_k
        self.diffusion_spectral_modes    = diffusion_spectral_modes
        self.surface_geometry_transfer   = surface_geometry_transfer
        self.atomic_message_chunk_size   = atomic_message_chunk_size
        self.vector_atomic_channels      = vector_atomic_channels
        self.interaction_round           = interaction_mode
        self.surface_atom_radius         = float(surface_atom_radius)
        self.dropout_probability         = float(dropout)
        self.atomic_input_width          = atomic_input_width
        self.evidence_input_width        = hidden_dim
        self.evidence_head_parameter_count = hidden_dim + 1
        self.evidence_head_added_parameter_count = 0

    @staticmethod
    def _residual_scale(mode: ResidualInitialization) -> float | None:
        """Translate a named residual policy into its initial learnable branch scale.

        Args:
            mode: Historical, ReZero, or LayerScale-like initialization.

        Returns:
            ``None`` for the historical unscaled path, zero for ReZero, or ``0.01`` for
            LayerScale-like initialization.
        """
        return {
            ResidualInitialization.CURRENT:    None,
            ResidualInitialization.REZERO:     0.0,
            ResidualInitialization.LAYERSCALE: 1.0e-2,
        }[mode]

    @staticmethod
    def _diffusion_times(
        layers: int,
        mode  : DiffusionTimeInitialization,
    ) -> tuple[float, ...]:
        """Build one physical initial time per DiffusionNet block.

        Args:
            layers: Positive number of surface blocks.
            mode: Historical powers of two, broader powers-of-two lengths, or equal times.

        Returns:
            Positive diffusion times in square ångströms. For four broader blocks this is exactly
            ``(0.25, 1, 4, 16)`` as prescribed by the experimental plan.
        """
        if mode is DiffusionTimeInitialization.CURRENT:
            return tuple(float(2**index) for index in range(layers))
        if mode is DiffusionTimeInitialization.SAME_SCALE:
            return (1.0,) * layers
        return tuple(float((0.5 * 2**index) ** 2) for index in range(layers))

    def _initialize_embeddings(self, mode: EmbeddingInitialization) -> None:
        """Apply one shared scale policy to every categorical lookup table.

        Args:
            mode: PyTorch defaults, width-scaled normal, or the small-normal control.
        """
        if mode is EmbeddingInitialization.CURRENT:
            return
        if mode is EmbeddingInitialization.PHYSICAL:
            PhysicalEmbeddingInitializer().initialize(
                self.atomic_number_embedding,
                self.residue_type_embedding,
            )
            for embedding in (self.atom_role_embedding, self.hybridization_embedding):
                nn.init.normal_(
                    embedding.weight,
                    mean=0.0,
                    std=embedding.embedding_dim ** -0.5,
                )
            return
        for embedding in (
            self.atomic_number_embedding,
            self.residue_type_embedding,
            self.atom_role_embedding,
            self.hybridization_embedding,
        ):
            standard_deviation = (
                embedding.embedding_dim ** -0.5
                if mode is EmbeddingInitialization.FAN_SCALED
                else 0.02
            )
            nn.init.normal_(embedding.weight, mean=0.0, std=standard_deviation)

    def _initialize_linears(self, mode: LinearInitialization) -> None:
        """Apply an explicit dense-layer policy without touching embeddings or gates.

        Args:
            mode: Current module defaults, activation-aware fan scaling, or orthogonal hidden
                transforms with Xavier fallbacks for non-square projections.
        """
        if mode is LinearInitialization.CURRENT:
            return
        for module in self.modules():
            if not isinstance(module, nn.Linear):
                continue
            if (
                mode is LinearInitialization.ORTHOGONAL
                and module.in_features == module.out_features
            ):
                nn.init.orthogonal_(module.weight)
            elif mode is LinearInitialization.ACTIVATION_AWARE and module.out_features > 1:
                nn.init.kaiming_uniform_(module.weight, nonlinearity="relu")
            else:
                nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    @staticmethod
    def _gate_definitions(
        scale_count              : int,
        surface_geometry_transfer: bool,
    ) -> tuple[tuple[str, str | None], ...]:
        """Define stable gate names and the hierarchy used only by the L0 cost.

        Args:
            scale_count: Number of persisted curvature scales.
            surface_geometry_transfer: Whether to include the point-geometry interaction gate.

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
        if surface_geometry_transfer:
            definitions.append(("transfer.point_geometry", atom_parent))

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
        atom_positions                    : Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode atoms, transfer context, gate curvature channels, and propagate on the surface.

        Args:
            atomic_numbers: Element IDs ``[N]``; this information is always active.
            residue_type_ids: Residue IDs ``[N]``.
            atom_edge_index: Compact undirected union topology ``[2,E]`` with ``src < dst``.
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
            atom_positions: Optional centered atom coordinates ``[N,3]`` required by spike C.

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
        if self.vector_atomic_state is not None:
            if atom_positions is None:
                raise ValueError("vector atomic state requires atom_positions")
            atom_embeddings, _ = self.vector_atomic_state(
                atom_embeddings,
                atom_positions,
                atom_edge_index,
                atom_edge_distance,
            )

        # The same transformed, per-scale gated geometry feeds both the established surface
        # projection and the optional transfer interaction. No second representation or
        # preprocessing feature is introduced by the spike.

        curvature = self.curvature_inputs(surface_curvatures, gates)
        atom_context = self.surface_atom_transfer(
            atom_embeddings,
            surface_atom_neighbors,
            surface_atom_distances,
            surface_atom_normal_offsets,
            surface_atom_tangential_distances,
            surface_atom_mask,
            gates,
            point_features=(curvature if self.surface_geometry_transfer else None),
        ) * gates["surface.atom_context"]

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

        # Spike B adds exactly one distinct second round. The depth control repeats transfer and
        # surface propagation without feedback; the bidirectional candidate first lets S1 update
        # atoms, isolating whether the cross-domain return path matters beyond added capacity.

        if self.interaction_round is not InteractionRound.SINGLE:
            assert self.second_surface_transfer is not None
            assert self.second_surface_projection is not None
            assert self.second_surface_encoder is not None
            second_atoms = atom_embeddings
            if self.surface_atom_feedback is not None:
                second_atoms = self.surface_atom_feedback(
                    atom_embeddings,
                    encoded,
                    surface_atom_neighbors,
                    surface_atom_distances,
                    surface_atom_mask,
                    self.surface_atom_radius,
                )
            second_context = self.second_surface_transfer(
                second_atoms,
                surface_atom_neighbors,
                surface_atom_distances,
                surface_atom_normal_offsets,
                surface_atom_tangential_distances,
                surface_atom_mask,
                gates,
                point_features=(curvature if self.surface_geometry_transfer else None),
            ) * gates["surface.atom_context"]
            second_initial = self.second_surface_projection(
                torch.cat((encoded, second_context), dim=1)
            )
            encoded = self.second_surface_encoder(
                second_initial,
                surface_operators,
                surface_ptr,
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
        atom_positions                    : Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict local surface evidence and one MAX-pooled protein logit.

        Args:
            atomic_numbers: Element IDs ``[N]``.
            residue_type_ids: Residue IDs ``[N]``.
            atom_edge_index: Compact undirected atomic topology ``[2,E]`` with ``src < dst``.
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
            atom_positions: Optional atom coordinates ``[N,3]`` for vector-state spike C.

        Returns:
            Essential ``surface_logits[M]`` and ``logits[B]`` outputs.
        """
        del surface_area_weights
        surface_embeddings, surface_logits = self.encode_surface(
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
            atom_positions,
        )
        protein_count = len(surface_ptr) - 1
        protein_logits = self.global_max_pooling(
            surface_logits.unsqueeze(-1), surface_batch, protein_count
        ).squeeze(-1)
        return {
            "logits":             protein_logits,
            "surface_logits":     surface_logits,
            "surface_embeddings": surface_embeddings,
        }
