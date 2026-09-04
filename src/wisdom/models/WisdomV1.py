"""WISDOM v1 bounded atomic, geometric-transfer, and DiffusionNet backbone."""

from __future__ import annotations

from typing import Any, ClassVar
from collections.abc import Mapping, Sequence

import torch
from lambdaforge.nn.models import MLP, Model, RelationalGCN
from lambdaforge.nn.pooling import SparseMaxPooling
from torch import Tensor, nn

from wisdom.models.SurfaceAtomTransfer import SurfaceAtomTransfer
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder


class WisdomV1(Model):
    """Model discrete atomic chemistry, local transfer, and an intrinsic molecular surface."""

    # These identifiers make the fixed v1 scientific contract inspectable by Training and tests.

    ARCHITECTURE_NAME: ClassVar[str] = "bounded-atomic-diffusionnet"
    STRUCTURAL_SCHEMA_VERSION: ClassVar[str] = "3.0"

    output_schema: ClassVar[dict[str, Any]] = {
        "logits": "Tensor[B]",
        "surface_logits": "Tensor[M]",
    }

    def __init__(
        self,
        hidden_dim               : int = 128,
        embedding_dim            : int = 32,
        residue_embedding_dim    : int | None = None,
        use_residue_type         : bool = True,
        atom_feature_preset      : str = "legacy",
        use_element              : bool = True,
        use_formal_charge        : bool = False,
        use_aromaticity          : bool = False,
        use_hbond_donor          : bool = False,
        use_hbond_acceptor       : bool = False,
        use_hybridization        : bool = False,
        use_atom_role            : bool = False,
        use_residue_hydropathy   : bool = False,
        use_residue_polarity     : bool = False,
        atomic_layers            : int = 2,
        projection_depth         : int = 1,
        surface_layers           : int = 2,
        dropout                  : float = 0.2,
        atomic_number_count      : int = 119,
        residue_type_count       : int = 21,
        curvature_features       : int = 6,
        atom_spatial_k           : int = 16,
        surface_atom_k           : int = 16,
        diffusion_spectral_modes : int = 128,
        surface_atom_radius      : float = 6.0,
        surface_chunk_size       : int = 8192,
        atomic_message_chunk_size: int = 65536,
        transfer_geometry        : str = "full",
        surface_feature_mode     : str = "chemistry_geometry",
        use_mean_curvature       : bool = True,
        use_gaussian_curvature   : bool = True,
        use_curvedness           : bool = True,
        use_shape_index          : bool = False,
    ) -> None:
        """Build the v1 backbone while fixing protein aggregation to existential MAX.

        Atomic categories pass through LambdaForge's chunked relational GCN. Each surface point
        then gathers at most ``J`` atom embeddings through learned invariant geometric weights.
        Curvature and chemical context form scalar surface features for DiffusionNet; no persisted
        surface-radius graph and no complete ``[E_surface,H]`` message tensor exists.

        Args:
            hidden_dim: Shared atom/surface latent width ``H``.
            embedding_dim: Width of element and optional residue embeddings.
            residue_embedding_dim: Independent residue embedding width; ``None`` preserves the
                legacy shared ``embedding_dim`` value.
            use_residue_type: Include the learned residue-category embedding when true.
            atom_feature_preset: Convenient feature bundle: ``legacy``, ``identity``,
                ``identity_residue``, ``identity_chemistry``, ``identity_structural``,
                ``full_generic``, ``constant``, or ``custom``.
            use_element: Include learned element identity in the custom feature set.
            use_formal_charge: Include the source structure's formal charge.
            use_aromaticity: Include a conservative aromatic-atom flag.
            use_hbond_donor: Include a conservative hydrogen-bond donor flag.
            use_hbond_acceptor: Include a conservative hydrogen-bond acceptor flag.
            use_hybridization: Include graph-derived sp/sp2/sp3 category identity.
            use_atom_role: Include backbone/side-chain/metal/other role identity.
            use_residue_hydropathy: Include normalized Kyte--Doolittle hydropathy.
            use_residue_polarity: Include a coarse polar-residue flag.
            atomic_layers: Relation-aware atomic message-passing depth.
            projection_depth: Pointwise chemistry/curvature projection depth.
            surface_layers: Number of DiffusionNet blocks.
            dropout: Shared dropout probability in ``[0,1)``.
            atomic_number_count: Element embedding table size.
            residue_type_count: Residue embedding table size including unknown ID zero.
            curvature_features: Flattened ``3*S`` curvature width derived from the dataset.
            atom_spatial_k: Runtime spatial rank budget recorded with this model.
            surface_atom_k: Runtime nearest-atom budget recorded with this model.
            diffusion_spectral_modes: Runtime low-frequency mode budget recorded with this model.
            surface_atom_radius: Physical transfer cutoff used to normalize geometric features.
            surface_chunk_size: Maximum points per atom-transfer activation chunk.
            atomic_message_chunk_size: Maximum atomic edges processed per RGCN message chunk.
            transfer_geometry: ``distance_only`` or invariant ``full`` ``(d,z,rho)`` transfer.
            surface_feature_mode: ``chemistry_only``, ``geometry_only``, or their concatenation.
            use_mean_curvature: Include mean curvature at every retained scale.
            use_gaussian_curvature: Include Gaussian curvature at every retained scale.
            use_curvedness: Include curvedness at every retained scale.
            use_shape_index: Derive and include bounded shape index at every retained scale.

        Raises:
            TypeError: If a feature switch is not Boolean.
            ValueError: If a dimension/depth/budget/radius is non-positive or dropout is invalid.
        """
        super().__init__()

        dimensions = (
            hidden_dim,
            embedding_dim,
            projection_depth,
            atomic_number_count,
            residue_type_count,
            atom_spatial_k,
            surface_atom_k,
            diffusion_spectral_modes,
            surface_chunk_size,
            atomic_message_chunk_size,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions
        ):
            raise ValueError("WISDOM dimensions, depths, and budgets must be positive integers")
        if atomic_layers < 0 or surface_layers < 0:
            raise ValueError("atomic and surface layer counts cannot be negative")
        if isinstance(curvature_features, bool) or curvature_features < 0:
            raise ValueError("curvature_features cannot be negative")
        if not 0.0 <= dropout < 1.0 or surface_atom_radius <= 0.0:
            raise ValueError("dropout or surface atom radius is invalid")

        feature_switches = self.resolve_atom_features(
            atom_feature_preset,
            {
                "element":              use_element,
                "residue_type":         use_residue_type,
                "formal_charge":        use_formal_charge,
                "aromaticity":          use_aromaticity,
                "hbond_donor":          use_hbond_donor,
                "hbond_acceptor":       use_hbond_acceptor,
                "hybridization":        use_hybridization,
                "atom_role":            use_atom_role,
                "residue_hydropathy":   use_residue_hydropathy,
                "residue_polarity":     use_residue_polarity,
            },
        )
        if transfer_geometry not in {"distance_only", "full"}:
            raise ValueError("transfer_geometry must be distance_only or full")
        if surface_feature_mode not in {
            "chemistry_only",
            "geometry_only",
            "chemistry_geometry",
        }:
            raise ValueError("unsupported surface feature mode")

        curvature_switches = (
            use_mean_curvature,
            use_gaussian_curvature,
            use_curvedness,
            use_shape_index,
        )
        if not all(isinstance(value, bool) for value in curvature_switches):
            raise TypeError("curvature feature switches must be Boolean")
        if surface_feature_mode != "chemistry_only" and not any(curvature_switches):
            raise ValueError("a geometry surface mode requires at least one curvature descriptor")

        residue_width = embedding_dim if residue_embedding_dim is None else residue_embedding_dim
        if residue_width < 1:
            raise ValueError("residue_embedding_dim must be positive")

        self.atomic_number_embedding = (
            nn.Embedding(atomic_number_count, embedding_dim)
            if feature_switches["element"]
            else None
        )
        self.residue_type_embedding  = (
            nn.Embedding(residue_type_count, residue_width)
            if feature_switches["residue_type"]
            else None
        )
        self.atom_role_embedding = (
            nn.Embedding(7, 8) if feature_switches["atom_role"] else None
        )
        self.hybridization_embedding = (
            nn.Embedding(4, 4) if feature_switches["hybridization"] else None
        )
        self.constant_atom_embedding = (
            nn.Parameter(torch.empty(embedding_dim).normal_(std=0.02))
            if not any(feature_switches.values())
            else None
        )

        atomic_input_width = (
            (embedding_dim if feature_switches["element"] else 0)
            + (residue_width if feature_switches["residue_type"] else 0)
            + (8 if feature_switches["atom_role"] else 0)
            + (4 if feature_switches["hybridization"] else 0)
            + sum(
                int(feature_switches[name])
                for name in (
                    "formal_charge",
                    "aromaticity",
                    "hbond_donor",
                    "hbond_acceptor",
                    "residue_hydropathy",
                    "residue_polarity",
                )
            )
        )
        if self.constant_atom_embedding is not None:
            atomic_input_width = embedding_dim

        projection_width = (
            (hidden_dim if surface_feature_mode != "geometry_only" else 0)
            + (curvature_features if surface_feature_mode != "chemistry_only" else 0)
        )

        self.atomic_encoder: nn.Module = (
            RelationalGCN(
                in_channels        = atomic_input_width,
                out_channels       = hidden_dim,
                num_relations      = 3,
                hidden_channels    = [hidden_dim] * (atomic_layers - 1),
                aggregation        = "mean",
                dropout            = dropout,
                residual           = True,
                message_chunk_size = atomic_message_chunk_size,
            )
            if atomic_layers
            else nn.Linear(atomic_input_width, hidden_dim)
        )
        self.surface_atom_transfer = SurfaceAtomTransfer(
            hidden_dim   = hidden_dim,
            radius       = surface_atom_radius,
            chunk_size   = surface_chunk_size,
            geometry_mode = transfer_geometry,
        )
        self.surface_projection = MLP(
            in_features = projection_width,
            out_features = hidden_dim,
            hidden      = [hidden_dim] * (projection_depth - 1),
            dropout     = dropout,
            residual    = True,
        )
        self.surface_encoder: nn.Module = (
            DiffusionSurfaceEncoder(hidden_dim, layers=surface_layers, dropout=dropout)
            if surface_layers
            else nn.Identity()
        )
        self.local_head         = nn.Linear(hidden_dim, 1)
        self.global_max_pooling = SparseMaxPooling()

        self.hidden_dim               = hidden_dim
        self.curvature_features       = curvature_features
        self.residue_type_count       = residue_type_count
        self.use_residue_type         = feature_switches["residue_type"]
        self.atom_features            = feature_switches
        self.atomic_input_width       = atomic_input_width
        self.surface_feature_mode     = surface_feature_mode
        self.curvature_switches       = curvature_switches
        self.atomic_layers            = atomic_layers
        self.projection_depth         = projection_depth
        self.surface_layers           = surface_layers
        self.atom_spatial_k           = atom_spatial_k
        self.surface_atom_k           = surface_atom_k
        self.diffusion_spectral_modes = diffusion_spectral_modes
        self.dropout_probability      = float(dropout)

    @staticmethod
    def resolve_atom_features(
        preset  : str,
        switches: Mapping[str, bool],
    ) -> dict[str, bool]:
        """Resolve one convenient atom-feature family or explicit custom switches.

        Args:
            preset: Named feature family or ``custom``.
            switches: Individual Boolean choices used only by ``custom``; ``legacy`` preserves
                the historical element plus optional-residue behavior.

        Returns:
            Complete feature-name mapping. ``constant`` deliberately returns every feature false
            so the model uses one shared learned atom vector.

        Raises:
            TypeError: If a custom switch is not Boolean.
            ValueError: If ``preset`` is unknown.
        """
        if not all(isinstance(value, bool) for value in switches.values()):
            raise TypeError("atom feature switches must be Boolean")

        presets = {
            "identity":            {"element"},
            "identity_residue":    {"element", "residue_type"},
            "identity_chemistry":  {
                "element", "formal_charge", "aromaticity", "hbond_donor",
                "hbond_acceptor", "hybridization",
            },
            "identity_structural": {"element", "atom_role"},
            "full_generic":        set(switches),
            "constant":            set(),
        }
        if preset == "custom":
            return dict(switches)
        if preset == "legacy":
            return {
                name: value if name in {"element", "residue_type"} else False
                for name, value in switches.items()
            }
        if preset not in presets:
            raise ValueError(f"unsupported atom feature preset: {preset!r}")
        enabled = presets[preset]
        return {name: name in enabled for name in switches}

    def encode_surface(
        self,
        atomic_numbers                    : Tensor,
        residue_type_ids                  : Tensor,
        atom_edge_index                   : Tensor,
        atom_edge_types                   : Tensor,
        surface_curvatures                : Tensor,
        surface_atom_neighbors            : Tensor,
        surface_atom_distances            : Tensor,
        surface_atom_normal_offsets       : Tensor,
        surface_atom_tangential_distances : Tensor,
        surface_atom_mask                 : Tensor,
        surface_operators                 : Sequence[Mapping[str, Tensor]],
        surface_ptr                       : Tensor,
        surface_positions                 : Tensor | None = None,
        surface_normals                   : Tensor | None = None,
        surface_neighbors                 : Tensor | None = None,
        surface_neighbor_mask             : Tensor | None = None,
        atom_role_ids                     : Tensor | None = None,
        atom_hybridization_ids            : Tensor | None = None,
        formal_charges                    : Tensor | None = None,
        atom_aromaticity                  : Tensor | None = None,
        atom_hbond_donor                  : Tensor | None = None,
        atom_hbond_acceptor               : Tensor | None = None,
        residue_hydropathy                : Tensor | None = None,
        residue_polarity                  : Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Encode atomic chemistry and the intrinsic surface in exact point order.

        Args:
            atomic_numbers: Element IDs ``long [N]``.
            residue_type_ids: Residue IDs ``long [N]``.
            atom_role_ids: Coarse structural-role IDs ``long [N]``.
            atom_hybridization_ids: Unknown/sp/sp2/sp3 IDs ``long [N]``.
            formal_charges: Formal charges ``float [N]`` in elementary-charge units.
            atom_aromaticity: Aromatic-atom flags ``bool [N]``.
            atom_hbond_donor: Conservative donor flags ``bool [N]``.
            atom_hbond_acceptor: Conservative acceptor flags ``bool [N]``.
            residue_hydropathy: Normalized residue hydropathy ``float [N]``.
            residue_polarity: Coarse polar-residue flags ``bool [N]``.
            atom_edge_index: Active bidirectional atomic pairs ``long [2,Ea]``.
            atom_edge_types: Relation IDs ``0=spatial,1=covalent,2=both`` with shape ``[Ea]``.
            surface_curvatures: Invariant curvature values ``float [M,S,3]``.
            surface_atom_neighbors: Offset atom IDs ``long [M,J]`` with ``-1`` sentinels.
            surface_atom_distances: Atom distances ``float [M,J]`` in ångströms.
            surface_atom_normal_offsets: Signed normal offsets ``float [M,J]``.
            surface_atom_tangential_distances: Tangential magnitudes ``float [M,J]``.
            surface_atom_mask: Valid atom-neighbor mask ``bool [M,J]``.
            surface_operators: Ordered per-protein DiffusionNet operators.
            surface_ptr: Point prefix boundaries ``long [B+1]``.
            surface_positions: Optional V3 coordinates ``[M,3]``; unused by v1.
            surface_normals: Optional V3 normals ``[M,3]``; unused by v1.
            surface_neighbors: Optional V3 bounded surface IDs ``[M,Ks]``; unused by v1.
            surface_neighbor_mask: Optional V3 neighbor mask ``[M,Ks]``; unused by v1.

        Returns:
            Surface embeddings ``[M,H]`` and weakly supervised local logits ``[M]``.

        Raises:
            ValueError: If tensor shapes, curvature width, or encoder outputs disagree with the
                model contract. Dataset publication owns exhaustive value/range validation.
        """
        if atomic_numbers.ndim != 1 or not len(atomic_numbers):
            raise ValueError("atomic_numbers must contain at least one value with shape [N]")
        if residue_type_ids.shape != atomic_numbers.shape:
            raise ValueError("residue_type_ids must have the same shape [N]")
        if atom_edge_index.ndim != 2 or atom_edge_index.shape[0] != 2:
            raise ValueError("atom_edge_index must have shape [2,Ea]")
        if atom_edge_types.shape != (atom_edge_index.shape[1],):
            raise ValueError("atom_edge_types must have shape [Ea]")
        if surface_curvatures.ndim != 3 or surface_curvatures.shape[2] != 3:
            raise ValueError("surface_curvatures must have shape [M,S,3]")

        integer_default = torch.zeros_like(atomic_numbers)
        float_default   = torch.zeros_like(atomic_numbers, dtype=torch.float32)
        atom_role_ids          = integer_default if atom_role_ids is None else atom_role_ids
        atom_hybridization_ids = (
            integer_default if atom_hybridization_ids is None else atom_hybridization_ids
        )
        formal_charges         = float_default if formal_charges is None else formal_charges
        atom_aromaticity       = float_default if atom_aromaticity is None else atom_aromaticity
        atom_hbond_donor       = float_default if atom_hbond_donor is None else atom_hbond_donor
        atom_hbond_acceptor    = (
            float_default if atom_hbond_acceptor is None else atom_hbond_acceptor
        )
        residue_hydropathy     = (
            float_default if residue_hydropathy is None else residue_hydropathy
        )
        residue_polarity       = float_default if residue_polarity is None else residue_polarity

        surface_count = len(surface_curvatures)
        curvature_features = self.curvature_inputs(surface_curvatures)
        observed_width     = int(curvature_features.shape[1])
        if (
            surface_count == 0
            or (
                self.surface_feature_mode != "chemistry_only"
                and observed_width != self.curvature_features
            )
        ):
            raise ValueError(
                "flattened surface curvature width disagrees with model configuration: "
                f"observed={observed_width}, expected={self.curvature_features}"
            )
        atom_feature_parts: list[Tensor] = []
        if self.atomic_number_embedding is not None:
            atom_feature_parts.append(self.atomic_number_embedding(atomic_numbers))
        if self.residue_type_embedding is not None:
            atom_feature_parts.append(self.residue_type_embedding(residue_type_ids))
        if self.atom_role_embedding is not None:
            atom_feature_parts.append(self.atom_role_embedding(atom_role_ids))
        if self.hybridization_embedding is not None:
            atom_feature_parts.append(self.hybridization_embedding(atom_hybridization_ids))

        scalar_features = {
            "formal_charge":      formal_charges,
            "aromaticity":        atom_aromaticity,
            "hbond_donor":        atom_hbond_donor,
            "hbond_acceptor":     atom_hbond_acceptor,
            "residue_hydropathy": residue_hydropathy,
            "residue_polarity":   residue_polarity,
        }
        for name, values in scalar_features.items():
            if self.atom_features[name]:
                atom_feature_parts.append(values.to(torch.float32).unsqueeze(1))

        if self.constant_atom_embedding is not None:
            atom_features = self.constant_atom_embedding.unsqueeze(0).expand(
                len(atomic_numbers), -1
            )
        else:
            atom_features = torch.cat(atom_feature_parts, dim=1)

        if self.atomic_layers:
            atom_embeddings = self.atomic_encoder(atom_features, atom_edge_index, atom_edge_types)
        else:
            atom_embeddings = self.atomic_encoder(atom_features)

        surface_parts: list[Tensor] = []
        if self.surface_feature_mode != "geometry_only":
            surface_parts.append(
                self.surface_atom_transfer(
                    atom_embeddings,
                    surface_atom_neighbors,
                    surface_atom_distances,
                    surface_atom_normal_offsets,
                    surface_atom_tangential_distances,
                    surface_atom_mask,
                )
            )
        if self.surface_feature_mode != "chemistry_only":
            surface_parts.append(curvature_features)

        initial_surface = self.surface_projection(torch.cat(surface_parts, dim=1))
        surface_embeddings = self.encode_surface_features(
            initial_surface,
            surface_operators,
            surface_ptr,
            surface_positions,
            surface_normals,
            surface_neighbors,
            surface_neighbor_mask,
        )
        if surface_embeddings.shape != (surface_count, self.hidden_dim):
            raise ValueError("surface encoder output must have shape [M,hidden_dim]")
        return surface_embeddings, self.local_head(surface_embeddings).squeeze(-1)

    def curvature_inputs(self, curvatures: Tensor) -> Tensor:
        """Select invariant curvature descriptors and optionally derive shape index.

        For principal curvatures ``k1 >= k2``, mean curvature is ``H=(k1+k2)/2`` and Gaussian
        curvature is ``K=k1*k2``. WISDOM reconstructs ``k1-k2=2*sqrt(max(H²-K,0))`` and uses the
        bounded Koenderink shape index ``(2/pi)*atan2(2H,k1-k2)``. It is zero at numerically flat
        points where both numerator and denominator vanish.

        Args:
            curvatures: Mean, Gaussian, and curvedness values ``[M,S,3]``.

        Returns:
            Flattened selected descriptors ``[M,curvature_features]``.
        """
        selected: list[Tensor] = []
        for enabled, index in zip(self.curvature_switches[:3], range(3), strict=True):
            if enabled:
                selected.append(curvatures[:, :, index])
        if self.curvature_switches[3]:
            mean      = curvatures[:, :, 0]
            gaussian  = curvatures[:, :, 1]
            difference = 2.0 * torch.sqrt((mean.square() - gaussian).clamp_min(0.0))
            shape_index = (2.0 / torch.pi) * torch.atan2(2.0 * mean, difference)
            shape_index = torch.where(
                (mean == 0.0) & (difference == 0.0),
                torch.zeros_like(shape_index),
                shape_index,
            )
            selected.append(shape_index)
        if not selected:
            return curvatures.new_empty((len(curvatures), 0))
        return torch.stack(selected, dim=2).flatten(start_dim=1)

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
        """Apply the fixed v1 DiffusionNet surface mechanism.

        Args:
            features: Initial scalar surface features ``[M,H]``.
            surface_operators: Per-protein intrinsic operator packs.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            surface_positions: Optional V3 positions; unused by v1.
            surface_normals: Optional V3 normals; unused by v1.
            surface_neighbors: Optional V3 bounded neighbors; unused by v1.
            surface_neighbor_mask: Optional V3 neighbor mask; unused by v1.

        Returns:
            DiffusionNet embeddings ``[M,H]``.
        """
        del surface_positions, surface_normals, surface_neighbors, surface_neighbor_mask
        if self.surface_layers == 0:
            return self.surface_encoder(features)
        return self.surface_encoder(features, surface_operators, surface_ptr)

    def validate_surface_bags(
        self,
        surface_area_weights: Tensor,
        surface_batch       : Tensor,
        surface_count       : int,
        protein_count       : int,
    ) -> int:
        """Validate surface ownership shapes without synchronizing CUDA values to the host.

        Args:
            surface_area_weights: Positive finite weights ``float [M]``.
            surface_batch: Ordered consecutive protein IDs ``long [M]``.
            surface_count: Expected point count ``M``.
            protein_count: Number of proteins derived from the CPU-resident prefix array.

        Returns:
            Supplied number of protein bags ``B``.

        Raises:
            ValueError: If tensor shapes or the protein count are invalid.
        """
        if surface_count < 1 or surface_area_weights.shape != (surface_count,):
            raise ValueError("surface area weights must have non-empty shape [M]")
        if surface_batch.shape != (surface_count,):
            raise ValueError("surface_batch must have shape [M]")
        if protein_count < 1:
            raise ValueError("surface_ptr must describe at least one protein")
        return protein_count

    def forward(
        self,
        atomic_numbers                    : Tensor,
        residue_type_ids                  : Tensor,
        atom_edge_index                   : Tensor,
        atom_edge_types                   : Tensor,
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
        surface_positions                 : Tensor | None = None,
        surface_normals                   : Tensor | None = None,
        surface_neighbors                 : Tensor | None = None,
        surface_neighbor_mask             : Tensor | None = None,
        atom_role_ids                     : Tensor | None = None,
        atom_hybridization_ids            : Tensor | None = None,
        formal_charges                    : Tensor | None = None,
        atom_aromaticity                  : Tensor | None = None,
        atom_hbond_donor                  : Tensor | None = None,
        atom_hbond_acceptor               : Tensor | None = None,
        residue_hydropathy                : Tensor | None = None,
        residue_polarity                  : Tensor | None = None,
    ) -> dict[str, Tensor]:
        """Predict one protein logit through MAX multiple-instance pooling.

        Args:
            atomic_numbers: Element IDs ``[N]``.
            residue_type_ids: Residue IDs ``[N]``.
            atom_role_ids: Structural-role IDs ``[N]``.
            atom_hybridization_ids: Hybridization category IDs ``[N]``.
            formal_charges: Formal charges ``[N]`` in elementary-charge units.
            atom_aromaticity: Aromatic flags ``[N]``.
            atom_hbond_donor: Donor flags ``[N]``.
            atom_hbond_acceptor: Acceptor flags ``[N]``.
            residue_hydropathy: Normalized hydropathy ``[N]``.
            residue_polarity: Polar-residue flags ``[N]``.
            atom_edge_index: Active bidirectional atomic topology ``[2,Ea]``.
            atom_edge_types: Atomic relation IDs ``[Ea]``.
            surface_curvatures: Curvature invariants ``[M,S,3]``.
            surface_atom_neighbors: Atom IDs ``[M,J]``.
            surface_atom_distances: Atom distances ``[M,J]``.
            surface_atom_normal_offsets: Normal offsets ``[M,J]``.
            surface_atom_tangential_distances: Tangential magnitudes ``[M,J]``.
            surface_atom_mask: Valid transfer entries ``[M,J]``.
            surface_area_weights: Positive represented-area weights ``[M]``.
            surface_batch: Point-to-protein owners ``[M]``.
            surface_operators: Per-protein intrinsic operators.
            surface_ptr: Point prefix boundaries ``[B+1]``.
            surface_positions: Optional V3 coordinates; unused by v1.
            surface_normals: Optional V3 normals; unused by v1.
            surface_neighbors: Optional V3 bounded neighbours; unused by v1.
            surface_neighbor_mask: Optional V3 neighbour validity; unused by v1.

        Returns:
            Protein logits ``[B]`` and surface logits ``[M]``.
        """
        protein_count = self.validate_surface_bags(
            surface_area_weights,
            surface_batch,
            len(surface_curvatures),
            len(surface_ptr) - 1,
        )
        _, surface_logits = self.encode_surface(
            atomic_numbers                     = atomic_numbers,
            residue_type_ids                   = residue_type_ids,
            atom_edge_index                    = atom_edge_index,
            atom_edge_types                    = atom_edge_types,
            surface_curvatures                 = surface_curvatures,
            surface_atom_neighbors             = surface_atom_neighbors,
            surface_atom_distances             = surface_atom_distances,
            surface_atom_normal_offsets        = surface_atom_normal_offsets,
            surface_atom_tangential_distances  = surface_atom_tangential_distances,
            surface_atom_mask                  = surface_atom_mask,
            surface_operators                  = surface_operators,
            surface_ptr                        = surface_ptr,
            atom_role_ids                      = atom_role_ids,
            atom_hybridization_ids             = atom_hybridization_ids,
            formal_charges                     = formal_charges,
            atom_aromaticity                   = atom_aromaticity,
            atom_hbond_donor                   = atom_hbond_donor,
            atom_hbond_acceptor                = atom_hbond_acceptor,
            residue_hydropathy                 = residue_hydropathy,
            residue_polarity                   = residue_polarity,
            surface_positions                 = surface_positions,
            surface_normals                   = surface_normals,
            surface_neighbors                 = surface_neighbors,
            surface_neighbor_mask             = surface_neighbor_mask,
        )
        protein_logits = self.global_max_pooling(
            surface_logits.unsqueeze(-1),
            surface_batch,
            protein_count,
        ).squeeze(-1)
        return {"logits": protein_logits, "surface_logits": surface_logits}
