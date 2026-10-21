"""WISDOM v1 semantic-gated atomic and DiffusionNet surface backbone."""

from __future__ import annotations

import torch

from torch import Tensor, nn
from typing import Any, ClassVar
from lambdaforge.nn.models import MLP, Model
from collections.abc import Mapping, Sequence
from lambdaforge.nn.pooling import SparseMaxPooling
from wisdom.models.InteractionRound import InteractionRound
from wisdom.models.AtomicEncoderType import AtomicEncoderType
from wisdom.models.VectorAtomicState import VectorAtomicState
from wisdom.models.AtomicGraphEncoder import AtomicGraphEncoder
from wisdom.models.GatedAtomicEncoder import GatedAtomicEncoder
from wisdom.models.SurfaceAtomFeedback import SurfaceAtomFeedback
from wisdom.models.SurfaceAtomTransfer import SurfaceAtomTransfer
from wisdom.models.LinearInitialization import LinearInitialization
from wisdom.models.ResidualInitialization import ResidualInitialization
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.EmbeddingInitialization import EmbeddingInitialization
from wisdom.models.FeedbackPropagationMode import FeedbackPropagationMode
from wisdom.models.PostFeedbackPropagation import PostFeedbackPropagation
from wisdom.models.ArchitectureImplementation import GateCostNormalization
from wisdom.models.gating.SemanticGateRegistry import SemanticGateRegistry
from wisdom.models.SurfaceRepresentationMode import SurfaceRepresentationMode
from wisdom.models.DiffusionTimeInitialization import DiffusionTimeInitialization
from wisdom.models.PhysicalEmbeddingInitializer import PhysicalEmbeddingInitializer
from wisdom.models.ArchitectureImplementation import ArchitectureImplementation, FeedbackWeighting


class WisdomV1(Model):
    """Learn which generic physical inputs a fixed DiffusionNet/MAX backbone needs."""

    ARCHITECTURE_NAME: ClassVar[str] = "semantic-gated-diffusionnet"
    STRUCTURAL_SCHEMA_VERSION: ClassVar[str] = "3.0"

    output_schema: ClassVar[dict[str, Any]] = {
        "logits":             "Tensor[B]",
        "surface_logits":     "Tensor[M]",
        "surface_evidence_features": "Tensor[M,D]",
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
        atomic_encoder_type: str = "baseline",
        atomic_radial_count: int = 16,
        atomic_vector_channels: int = 16,
        atomic_basis_count: int = 4,
        atomic_spatial_cutoff: float = 6.0,
        atomic_covalent_cutoff: float = 64.0,
        atomic_maximum_triplets: int = 1_000_000,
        atomic_maximum_pairwise_bytes: int = 134_217_728,
        atomic_allow_dense_global: bool = False,
        atomic_degree_log_means: Sequence[float] | None = None,
        feedback_propagation       : str = "none",
        post_feedback_initial_scale: float = 0.1,
        surface_stage_layers       : Sequence[int] | None = None,
        checkpoint_surface         : bool = False,
        checkpoint_post_feedback   : bool = False,
        vector_atomic_channels     : int = 0,
        interaction_round          : str = "single",
        architecture_implementation : str = "legacy",
        feedback_weighting          : str = "point",
        gate_cost_normalization     : str = "historical_mean",
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
        surface_representation_mode : str = "learned",
        explicit_feature_dim        : int = 0,
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
            atomic_encoder_type: Exact V9 atomic operator/control; baseline keeps historical keys.
            atomic_radial_count: Gaussian/sinusoidal/Bessel radial basis count, positive.
            atomic_vector_channels: Positive GVP polar-channel count; PaiNN uses hidden_dim.
            atomic_basis_count: Positive NNConv basis count, inactive for full filters.
            atomic_spatial_cutoff: Spatial radial support in Å, without generating new neighbors.
            atomic_covalent_cutoff: Finite DimeNet covalent support in Å; out-of-support edges fail.
            atomic_maximum_triplets: Hard bound on explicit typed directional triplet count.
            atomic_maximum_pairwise_bytes: Principal directional/path/attention tensor byte guard,
                not a forecast of total backward or optimizer VRAM.
            atomic_allow_dense_global: Explicit permission for the bounded global-only experiment.
            atomic_degree_log_means: Derived frozen training log-degree means for full PNA;
                persisted in checkpoints, not an independently tuned input.
            feedback_propagation: none preserves old routes; d0-d8 selects a bounded V8d control.
            post_feedback_initial_scale: Initial learned gamma of the separate GNN/local update;
                zero is exact identity and initially blocks its internal weight gradients.
            surface_stage_layers: Optional positive [first,second] DiffusionNet block counts,
                only for two-stage V8d; None keeps surface_layers blocks in each stage.
            checkpoint_surface: Recompute surface blocks during backward without detaching.
            checkpoint_post_feedback: Recompute the added atomic transform during backward.
            vector_atomic_channels: Equivariant atomic vector channels; zero preserves the scalar
                baseline and positive values activate isolated spike C.
            interaction_round: ``single`` baseline, equally deep ``depth_control``, or one
                ``bidirectional`` round. depth_control_parameter_matched adds the identical
                atomic update but supplies a zero message, never S1 or detached S1.
            architecture_implementation: legacy preserves archived module layout, RNG and
                numerical behavior; controlled isolates new-module initialization and enables
                audited feedback/vector fixes. Saved constructor parameters restore the policy.
            feedback_weighting: point (historical distance mean) or area (distance times
                represented area), used only in feedback/null-control rounds.
            gate_cost_normalization: historical_mean divides by all available gates;
                baseline_count divides by the scalar baseline count, preserving the per-gate
                coefficient when point_geometry adds a gate. The latter cost may exceed one.
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
            surface_representation_mode: ``learned`` (default), ``explicit`` or late ``hybrid``.
            explicit_feature_dim: Number K of selected fixed fields; positive outside learned mode.

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
        self.architecture_implementation = ArchitectureImplementation(architecture_implementation)
        self.feedback_weighting          = FeedbackWeighting(feedback_weighting)
        self.gate_cost_normalization     = GateCostNormalization(gate_cost_normalization)
        controlled = self.architecture_implementation is ArchitectureImplementation.CONTROLLED
        self.feedback_propagation = FeedbackPropagationMode(feedback_propagation)
        if self.feedback_propagation is not FeedbackPropagationMode.NONE:
            if (not controlled or surface_geometry_transfer or vector_atomic_channels
                    or atomic_encoder_type != "baseline"
                    or surface_representation_mode != "learned"):
                raise ValueError(
                    "V8d requires controlled scalar baseline atoms and learned surfaces"
                )
            interaction_mode = self.feedback_propagation.interaction()

        second_surface_layers = surface_layers
        if surface_stage_layers is not None:
            if (self.feedback_propagation is FeedbackPropagationMode.NONE
                    or interaction_mode is InteractionRound.SINGLE
                    or len(surface_stage_layers) != 2
                    or any(value < 1 for value in surface_stage_layers)):
                raise ValueError(
                    "surface_stage_layers requires two positive counts for two-stage V8d"
                )
            surface_layers, second_surface_layers = surface_stage_layers
        if not controlled and (
            interaction_mode is InteractionRound.DEPTH_CONTROL_PARAMETER_MATCHED
            or self.feedback_weighting is not FeedbackWeighting.POINT
        ):
            raise ValueError("matched control and area feedback require controlled architecture")

        residue_width = embedding_dim if residue_embedding_dim is None else residue_embedding_dim
        if residue_width < 1:
            raise ValueError("residue_embedding_dim must be positive")

        self.atomic_encoder_type = AtomicEncoderType(atomic_encoder_type)
        if self.atomic_encoder_type is not AtomicEncoderType.BASELINE and (
            not controlled or surface_geometry_transfer or vector_atomic_channels
            or interaction_mode is not InteractionRound.SINGLE
            or SurfaceRepresentationMode(surface_representation_mode)
            is not SurfaceRepresentationMode.LEARNED
        ):
            raise ValueError(
                "V9 changes only the controlled learned atomic encoder, not V8/surface hypotheses"
            )

        self.curvature_scale_count = curvature_features // 3
        gate_definitions = self._gate_definitions(
            self.curvature_scale_count,
            surface_geometry_transfer and not controlled,
        )
        self.baseline_gate_count = len(self._gate_definitions(self.curvature_scale_count, False))
        if controlled and surface_geometry_transfer:
            # Append the extra gate so shared gate indices/initial states remain identical.

            gate_definitions += (("transfer.point_geometry", "surface.atom_context"),)
        self.semantic_gates = SemanticGateRegistry(
            gate_definitions,
            initial_active=gate_initial_active,
            inactive_names=self.atomic_encoder_type.inactive_sources(),
        )

        self.atomic_number_embedding = nn.Embedding(atomic_number_count, embedding_dim)
        self.residue_type_embedding   = nn.Embedding(residue_type_count, residue_width)
        self.atom_role_embedding      = nn.Embedding(7, 8)
        self.hybridization_embedding  = nn.Embedding(4, 4)

        atomic_input_width = embedding_dim + residue_width + 8 + 4 + 6
        self.atomic_encoder: Any = GatedAtomicEncoder(
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
                4 * self.curvature_scale_count
                if surface_geometry_transfer and not controlled else 0
            ),
            physics_distance_initialization=physics_distance_initialization,
        )
        self.vector_atomic_state = (
            VectorAtomicState(hidden_dim, vector_atomic_channels, implementation="legacy")
            if vector_atomic_channels > 0 and not controlled
            else None
        )
        self.surface_atom_feedback = (
            SurfaceAtomFeedback(hidden_dim, implementation="legacy")
            if interaction_mode is InteractionRound.BIDIRECTIONAL and not controlled
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
            if interaction_mode is not InteractionRound.SINGLE and not controlled
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
            if interaction_mode is not InteractionRound.SINGLE and not controlled
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
            if interaction_mode is not InteractionRound.SINGLE and not controlled
            else None
        )
        self.surface_representation_mode = SurfaceRepresentationMode(surface_representation_mode)
        if self.surface_representation_mode is not SurfaceRepresentationMode.LEARNED:
            if explicit_feature_dim < 1:
                raise ValueError("explicit/hybrid representations require explicit_feature_dim > 0")
        elif explicit_feature_dim:
            raise ValueError("learned representation must not request explicit feature channels")
        self.explicit_feature_dim = explicit_feature_dim
        self.surface_evidence_dim = (
            hidden_dim if self.surface_representation_mode is SurfaceRepresentationMode.LEARNED
            else explicit_feature_dim + (
                hidden_dim if self.surface_representation_mode is SurfaceRepresentationMode.HYBRID
                else 0
            )
        )
        self.local_head         = nn.Linear(self.surface_evidence_dim, 1)
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
        self.second_surface_layers       = second_surface_layers
        self.atom_spatial_k              = atom_spatial_k
        self.surface_atom_k              = surface_atom_k
        self.diffusion_spectral_modes    = diffusion_spectral_modes
        self.surface_geometry_transfer   = surface_geometry_transfer
        self.atomic_message_chunk_size   = atomic_message_chunk_size
        self.vector_atomic_channels      = vector_atomic_channels
        self.interaction_round           = interaction_mode
        self.surface_atom_radius         = float(surface_atom_radius)
        self.dropout_probability         = float(dropout)
        self.linear_initialization       = linear_mode
        self.atomic_input_width          = atomic_input_width
        self.evidence_input_width        = self.surface_evidence_dim
        self.evidence_head_parameter_count = self.surface_evidence_dim + 1
        self.evidence_head_added_parameter_count = 0
        if controlled:
            self._initialize_controlled_architecture(
                diffusion_time_mode, diffusion_residual_mode,
                neutral_transfer_initialization, physics_distance_initialization,
            )
        if self.atomic_encoder_type is not AtomicEncoderType.BASELINE:
            # Allocate alternatives after the entire common backbone. A private stream cannot
            # move V2 head/pooling initialization; copy the shared input transform explicitly.

            reference_encoder = self.atomic_encoder
            with torch.random.fork_rng(devices=[]):
                stream = torch.Generator().manual_seed(torch.initial_seed() ^ 0x563941)
                torch.set_rng_state(stream.get_state())
                alternative = AtomicGraphEncoder(
                    atomic_input_width, hidden_dim, atomic_layers, dropout,
                    atomic_edge_distance_scale, atomic_message_chunk_size, atomic_encoder_type,
                    radial_count=atomic_radial_count, vector_channels=atomic_vector_channels,
                    basis_count=atomic_basis_count, spatial_cutoff=atomic_spatial_cutoff,
                    covalent_cutoff=atomic_covalent_cutoff,
                    degree_log_means=atomic_degree_log_means,
                    maximum_triplets=atomic_maximum_triplets,
                    maximum_pairwise_bytes=atomic_maximum_pairwise_bytes,
                    allow_dense_global=atomic_allow_dense_global,
                )
                self._initialize_linears(linear_mode, root=alternative)
            if alternative.reference is not None:
                alternative.reference.load_state_dict(reference_encoder.state_dict(), strict=True)
            else:
                alternative.input_projection.load_state_dict(reference_encoder.input_projection.state_dict())
            self.atomic_encoder = alternative

        # GNN2 has no categorical-input projection and owns independent weights. Its private
        # stream leaves GNN1, both surface stages and all V2 heads exactly seed-paired.

        self.post_feedback_propagation: PostFeedbackPropagation | None = None
        mode = self.feedback_propagation
        if mode.layers() or mode is FeedbackPropagationMode.D8:
            with torch.random.fork_rng(devices=[]):
                stream = torch.Generator().manual_seed(torch.initial_seed() ^ 0x503844)
                torch.set_rng_state(stream.get_state())
                self.post_feedback_propagation = PostFeedbackPropagation(
                    hidden_dim, mode.layers() or 2, dropout,
                    atomic_edge_distance_scale, atomic_message_chunk_size,
                    initial_scale=post_feedback_initial_scale,
                    nodewise=mode is FeedbackPropagationMode.D8,
                    activation_checkpointing=checkpoint_post_feedback,
                )
                self._initialize_linears(linear_mode, root=self.post_feedback_propagation)
        if checkpoint_surface:
            if not isinstance(self.surface_encoder, DiffusionSurfaceEncoder):
                raise ValueError("checkpoint_surface requires the DiffusionNet surface encoder")
            self.surface_encoder.activation_checkpointing = True
            if isinstance(self.second_surface_encoder, DiffusionSurfaceEncoder):
                self.second_surface_encoder.activation_checkpointing = True

        # Explicit mode skips the learned branch, not just its output. Freeze unused tensors
        # so optimizers and trainable-parameter accounting reflect that experimental control.

        if self.surface_representation_mode is SurfaceRepresentationMode.EXPLICIT:
            for name, parameter in self.named_parameters():
                if not name.startswith("local_head."):
                    parameter.requires_grad_(False)

    def _initialize_controlled_architecture(
        self,
        diffusion_time_mode: DiffusionTimeInitialization,
        residual_mode: ResidualInitialization,
        neutral_transfer: bool,
        physics_distance: bool,
    ) -> None:
        """Construct only added V8 modules after the unchanged shared backbone is initialized.

        Each module family uses a private CPU RNG stream derived from the initialization seed.
        Exiting the fork restores the state used later by V2 pooling/heads/refiners. Depth,
        matched depth and bidirectional therefore share identical second-round parameters,
        regardless of feedback capacity, under every supported linear initialization policy.

        Args:
            diffusion_time_mode: Frozen initial surface time schedule for the second encoder.
            residual_mode: Frozen initial surface residual scaling for the second encoder.
            neutral_transfer: Whether orientation/content corrections start at zero.
            physics_distance: Whether the second transfer starts from its trainable -d/R prior.
        """
        seed = torch.initial_seed()
        with torch.random.fork_rng(devices=[]):
            if self.surface_geometry_transfer:
                torch.set_rng_state(torch.Generator().manual_seed(seed ^ 0x503847).get_state())
                self.surface_atom_transfer.enable_point_geometry(4 * self.curvature_scale_count)
                for name in ("point_geometry_projection", "point_atom_projection",
                             "point_pair_projection", "point_geometry_scorer"):
                    self._initialize_linears(self.linear_initialization,
                                             root=getattr(self.surface_atom_transfer, name))

            if self.vector_atomic_channels:
                torch.set_rng_state(torch.Generator().manual_seed(seed ^ 0x563843).get_state())
                self.vector_atomic_state = VectorAtomicState(
                    self.hidden_dim, self.vector_atomic_channels, implementation="controlled",
                )
                self._initialize_linears(self.linear_initialization, root=self.vector_atomic_state)

            if self.interaction_round in {
                InteractionRound.BIDIRECTIONAL, InteractionRound.DEPTH_CONTROL_PARAMETER_MATCHED,
            }:
                torch.set_rng_state(torch.Generator().manual_seed(seed ^ 0x463842).get_state())
                self.surface_atom_feedback = SurfaceAtomFeedback(
                    self.hidden_dim, weighting=self.feedback_weighting.value,
                    implementation="controlled",
                )
                self._initialize_linears(
                    self.linear_initialization, root=self.surface_atom_feedback,
                )

            if self.interaction_round is not InteractionRound.SINGLE:
                torch.set_rng_state(torch.Generator().manual_seed(seed ^ 0x533842).get_state())
                self.second_surface_transfer = SurfaceAtomTransfer(
                    self.hidden_dim, radius=self.surface_atom_radius,
                    chunk_size=self.surface_atom_transfer.chunk_size,
                    point_feature_dim=(4 * self.curvature_scale_count
                                       if self.surface_geometry_transfer else 0),
                    physics_distance_initialization=physics_distance,
                )
                self.second_surface_projection = MLP(
                    in_features=2 * self.hidden_dim, out_features=self.hidden_dim,
                    hidden=[self.hidden_dim], dropout=self.dropout_probability, residual=True,
                )
                self.second_surface_encoder = DiffusionSurfaceEncoder(
                    self.hidden_dim, layers=self.second_surface_layers,
                    dropout=self.dropout_probability,
                    initial_times=self._diffusion_times(
                        self.second_surface_layers, diffusion_time_mode,
                    ),
                    residual_initial_scale=self._residual_scale(residual_mode),
                )
                for module in (self.second_surface_transfer, self.second_surface_projection,
                               self.second_surface_encoder):
                    self._initialize_linears(self.linear_initialization, root=module)
                if neutral_transfer:
                    self.second_surface_transfer.initialize_neutral_residuals()
                if physics_distance:
                    self.second_surface_transfer.initialize_physics_distance()

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

    def _initialize_linears(
        self, mode: LinearInitialization, root: nn.Module | None = None,
    ) -> None:
        """Apply an explicit dense-layer policy without touching embeddings or gates.

        Args:
            mode: Current module defaults, activation-aware fan scaling, or orthogonal hidden
                transforms with Xavier fallbacks for non-square projections.
            root: Optional isolated subtree; None initializes this model exactly as historically.
                Supplying new heads avoids reinitializing shared backbone/pooling parameters.
        """
        if mode is LinearInitialization.CURRENT:
            return
        for module in (self if root is None else root).modules():
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
        """Return expected path L0 cost under the saved normalization policy.

        historical_mean remains the original mean in [0,1]. baseline_count divides the
        path-probability sum by the unchanged scalar gate count, retaining each shared gate's
        loss coefficient as optional gates are added; its upper bound is N/N_baseline.
        """
        if self.surface_representation_mode is SurfaceRepresentationMode.EXPLICIT:
            return self.local_head.weight.sum() * 0.0
        cost = self.semantic_gates.regularization()
        if self.gate_cost_normalization is GateCostNormalization.BASELINE_COUNT:
            cost = cost * (len(self.semantic_gates.names) / self.baseline_gate_count)
        return cost

    def gate_summary(self) -> dict[str, object]:
        """Return JSON-compatible learned gate probabilities and deterministic values."""
        return {
            **self.semantic_gates.summary(),
            "cost_normalization": self.gate_cost_normalization.value,
            "baseline_gate_count": self.baseline_gate_count,
        }

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
        surface_explicit_features         : Tensor | None = None,
        compute_logits                    : bool = True,
        surface_area_weights              : Tensor | None = None,
        atom_batch                        : Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
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
            atom_batch: Explicit protein ownership [N], required for bounded global attention.
            atom_positions: Optional centered coordinates ``[N,3]``, Å; required by spike C
                and V9 directional, persistent-vector and EGNN encoders.
            surface_explicit_features: Selected train-normalized fixed fields ``[M,K]``;
                never an annotation or a ligand-dependent input.
            compute_logits: False skips the unconditioned local head when V7 replaces it;
                the returned logit is None. True preserves all historical V1/V2 calls.
            surface_area_weights: Positive represented areas [M] in Å²; needed only when
                feedback_weighting=area and a feedback or null-control round is active.

        Returns:
            Surface evidence Z ``[M,D]`` and optional local logits ``[M]``. D is H, K or H+K
            in learned, explicit or hybrid mode respectively.
        """
        if self.surface_representation_mode is not SurfaceRepresentationMode.LEARNED:
            if surface_explicit_features is None or surface_explicit_features.shape != (
                len(surface_curvatures), self.explicit_feature_dim
            ):
                raise ValueError("selected surface fields must have shape [M,K]")
            # Explicit-only returns before sampling gates, atomic encoding, transfer or
            # DiffusionNet. Its prediction cannot indirectly depend on learned embeddings.

            if self.surface_representation_mode is SurfaceRepresentationMode.EXPLICIT:
                logits = (self.local_head(surface_explicit_features).squeeze(-1)
                          if compute_logits else None)
                return surface_explicit_features, logits

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
        encoder_arguments = (
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
        atom_embeddings = (
            self.atomic_encoder(*encoder_arguments, positions=atom_positions, atom_batch=atom_batch)
            if isinstance(self.atomic_encoder, AtomicGraphEncoder)
            else self.atomic_encoder(*encoder_arguments)
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

        # All V8d stages reference the same physical data and one derived sparse gradient tensor
        # per protein. Only this forward owns the derived container; no learned state is cached.

        if (self.feedback_propagation is not FeedbackPropagationMode.NONE
                and self.interaction_round is not InteractionRound.SINGLE):
            surface_operators = tuple(
                dict(operator, stacked_gradient=DiffusionSurfaceEncoder.stacked_gradient(
                    operator, int(surface_ptr[index + 1] - surface_ptr[index]),
                ))
                for index, operator in enumerate(surface_operators)
            )

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
        # atoms. The null-message control has the same update capacity/coverage but never reads
        # S1 through that path; all second rounds still receive S1 through the same projection.

        if self.interaction_round is not InteractionRound.SINGLE:
            assert self.second_surface_transfer is not None
            assert self.second_surface_projection is not None
            assert self.second_surface_encoder is not None
            second_atoms = atom_embeddings
            if self.surface_atom_feedback is not None:
                null_message = (
                    self.interaction_round is InteractionRound.DEPTH_CONTROL_PARAMETER_MATCHED
                )
                second_atoms = self.surface_atom_feedback(
                    atom_embeddings,
                    None if null_message else encoded,
                    surface_atom_neighbors,
                    surface_atom_distances,
                    surface_atom_mask,
                    self.surface_atom_radius,
                    area_weights=surface_area_weights,
                )
            if self.post_feedback_propagation is not None:
                second_atoms = self.post_feedback_propagation(
                    second_atoms, *encoder_arguments[1:],
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
        if self.surface_representation_mode is SurfaceRepresentationMode.HYBRID:
            assert surface_explicit_features is not None
            encoded = torch.cat((encoded, surface_explicit_features.to(encoded.dtype)), dim=1)
        logits = self.local_head(encoded).squeeze(-1) if compute_logits else None
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
        surface_explicit_features         : Tensor | None = None,
        atom_batch                        : Tensor | None = None,
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
            atom_positions: Optional centered coordinates ``[N,3]``, Å, for spike C and
                directional, persistent-vector or EGNN atomic encoders.
            surface_explicit_features: Optional frozen train-normalized fields ``[M,K]``;
                mandatory in explicit/hybrid mode, never a surface reference target.
            atom_batch: Explicit protein ownership [N], required for bounded global attention.

        Returns:
            Essential ``surface_logits[M]`` and ``logits[B]`` outputs.
        """
        surface_evidence_features, surface_logits = self.encode_surface(
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
            surface_explicit_features,
            surface_area_weights=surface_area_weights,
            atom_batch=atom_batch,
        )
        protein_count = len(surface_ptr) - 1
        assert surface_logits is not None
        protein_logits = self.global_max_pooling(
            surface_logits.unsqueeze(-1), surface_batch, protein_count
        ).squeeze(-1)
        return {
            "logits":             protein_logits,
            "surface_logits":     surface_logits,
            **self.surface_representation_outputs(
                surface_evidence_features, surface_explicit_features),
        }

    def surface_representation_outputs(
        self, evidence: Tensor, explicit: Tensor | None,
    ) -> dict[str, Tensor]:
        """Expose the actual H/X/Z representations without creating placeholder features.

        Args:
            evidence: Point evidence Z [M,D]: H, X, or concatenated [H,X].
            explicit: Frozen normalized X [M,K] supplied by the batch, or None.

        Returns:
            Always Z as surface_evidence_features; H as surface_learned_embeddings only when
            a learned encoder exists; X as surface_explicit_features only when consumed.
            The deprecated surface_embeddings alias denotes H, never X or concatenated Z.

        Raises:
            ValueError: Explicit/hybrid mode lacks its real frozen feature matrix.
        """
        output = {"surface_evidence_features": evidence}
        mode   = self.surface_representation_mode
        if mode is not SurfaceRepresentationMode.EXPLICIT:
            width = (evidence.shape[1] - self.explicit_feature_dim
                     if mode is SurfaceRepresentationMode.HYBRID else evidence.shape[1])
            learned = evidence[:, :width]
            output.update(surface_learned_embeddings=learned, surface_embeddings=learned)
        if mode is not SurfaceRepresentationMode.LEARNED:
            if explicit is None:
                raise ValueError("explicit/hybrid evidence requires real explicit surface features")
            output["surface_explicit_features"] = explicit
        return output
