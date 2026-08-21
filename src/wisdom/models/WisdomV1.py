"""WISDOM v1 atom-to-surface multiple-instance baseline."""

from __future__ import annotations

from typing import Any, ClassVar

import torch
from lambdaforge.nn import Scatter
from lambdaforge.nn.models import GCN, MLP, Model, RelationalGCN
from lambdaforge.nn.pooling import SparseMaxPooling
from torch import Tensor, nn


class WisdomV1(Model):
    """Encode atomic and surface graphs before existential MAX MIL pooling."""

    output_schema: ClassVar[dict[str, Any]] = {
        "logits": "Tensor[B]",
        "surface_logits": "Tensor[M]",
        "surface_embeddings": "Tensor[M,H]",
    }

    def __init__(
        self,
        hidden_dim          : int = 128,
        embedding_dim       : int = 32,
        use_residue_type    : bool = True,
        atomic_layers       : int = 2,
        projection_depth    : int = 1,
        surface_layers      : int = 2,
        dropout             : float = 0.2,
        atomic_number_count : int = 119,
        residue_type_count  : int = 21,
        curvature_features  : int = 6,
    ) -> None:
        """Build the complete v1 backbone from independent conceptual parameters.

        LambdaForge supplies the relational graph network, multilayer perceptron, surface graph
        network, sparse scatter operation, and sparse MAX pooling. WISDOM only composes them into
        its domain-specific atom-to-surface flow. ``atomic_layers`` and ``surface_layers`` count
        graph convolutions, while ``projection_depth`` counts linear projection layers. All hidden
        widths are ``hidden_dim`` so an HPO candidate cannot create inconsistent nested widths.

        Args:
            hidden_dim: Shared output width of the atomic encoder, surface projection, and surface
                encoder.
            embedding_dim: Width of each learned categorical embedding. When residue identity is
                disabled, this is the complete atomic input width; otherwise two embeddings are
                concatenated and the input width is ``2 * embedding_dim``.
            use_residue_type: Whether atomic features include a learned residue-category embedding
                in addition to the required atomic-number embedding.
            atomic_layers: Number of relation-aware graph convolutions, from one to four in the v1
                search protocol.
            projection_depth: Number of linear layers that fuse flattened curvature with aggregated
                atomic context. A value of one is a single linear projection.
            surface_layers: Number of graph convolutions on the fixed surface graph.
            dropout: Shared dropout probability used by the projection and both graph encoders.
            atomic_number_count: Number of element embedding rows; every atomic number must be
                smaller than this value.
            residue_type_count: Number of residue embedding rows, including ID zero for unknown
                residues.
            curvature_features: Flattened curvature width, equal to three invariants times the
                number of preprocessing scales.

        Raises:
            TypeError: If the residue-feature switch is not Boolean.
            ValueError: If a dimension, depth, table size, or dropout probability cannot define the
                documented tensor contract.
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
            curvature_features,
        )
        if any(
            isinstance(value, bool) or not isinstance(value, int) or value < 1
            for value in dimensions
        ):
            raise ValueError(
                "all WISDOM v1 dimensions, depths, and category counts must be positive integers"
            )
        if not isinstance(use_residue_type, bool):
            raise TypeError("use_residue_type must be Boolean")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0,1)")

        self.atomic_number_embedding = nn.Embedding(atomic_number_count, embedding_dim)
        self.residue_type_embedding  = (
            nn.Embedding(residue_type_count, embedding_dim) if use_residue_type else None
        )

        atomic_input_width = embedding_dim * (2 if use_residue_type else 1)
        projection_width   = curvature_features + hidden_dim

        self.atomic_encoder = RelationalGCN(
            in_channels=atomic_input_width,
            out_channels=hidden_dim,
            num_relations=3,
            hidden_channels=[hidden_dim] * (atomic_layers - 1),
            aggregation="mean",
            dropout=dropout,
            residual=True,
        )
        self.surface_projection = MLP(
            in_features=projection_width,
            out_features=hidden_dim,
            hidden=[hidden_dim] * (projection_depth - 1),
            dropout=dropout,
            residual=True,
        )
        self.surface_encoder = GCN(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            hidden_channels=[hidden_dim] * (surface_layers - 1),
            dropout=dropout,
            residual=True,
        )
        self.local_head         = nn.Linear(hidden_dim, 1)
        self.global_max_pooling = SparseMaxPooling()

        self.hidden_dim          = hidden_dim
        self.curvature_features  = curvature_features
        self.residue_type_count  = residue_type_count
        self.use_residue_type    = use_residue_type
        self.atomic_layers       = atomic_layers
        self.projection_depth    = projection_depth
        self.surface_layers      = surface_layers
        self.dropout_probability = float(dropout)

    def encode_surface(
        self,
        atomic_numbers          : Tensor,
        residue_type_ids        : Tensor,
        atom_edge_index         : Tensor,
        atom_edge_types         : Tensor,
        surface_curvatures      : Tensor,
        surface_edge_index      : Tensor,
        surface_atom_edge_index : Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Encode both sparse graphs and produce one unsupervised logit per surface point.

        Atomic number, and optionally residue identity, are embedded and propagated through an
        R-GCN whose relation IDs distinguish spatial-only, covalent-only, and combined edges. For
        surface point ``p`` with incident atom set ``A(p)``, the fixed cross-domain transfer is
        ``a_p = |A(p)|^-1 * sum_{i in A(p)} h_i``. LambdaForge ``Scatter.mean`` evaluates this
        expression without constructing a dense point-by-atom matrix. Flattened multiscale
        curvature is concatenated with ``a_p`` and propagated on the sparse surface graph.

        Args:
            atomic_numbers: Integer element IDs with shape ``[N]``.
            residue_type_ids: Integer residue IDs with shape ``[N]``. Values are validated even
                when ``use_residue_type`` is false so malformed batches do not pass silently.
            atom_edge_index: Bidirectional directed atomic graph with shape ``[2,Ea]``.
            atom_edge_types: R-GCN relation IDs in ``{0,1,2}`` with shape ``[Ea]``.
            surface_curvatures: Invariant ``(H,K,C)`` values with shape ``[M,S,3]``.
            surface_edge_index: Bidirectional directed surface graph with shape ``[2,Es]``.
            surface_atom_edge_index: Incidence ``[surface,atom]`` with shape ``[2,Esa]``.

        Returns:
            A pair containing surface embeddings ``float [M,hidden_dim]`` and local logits
            ``float [M]`` in the exact input point order.

        Raises:
            ValueError: If ranks, feature widths, category values, relation values, graph endpoints,
                or produced encoder shapes violate the model contract.
        """
        if atomic_numbers.ndim != 1 or not len(atomic_numbers):
            raise ValueError("atomic_numbers must contain at least one value with shape [N]")
        if residue_type_ids.shape != atomic_numbers.shape:
            raise ValueError("residue_type_ids must have the same shape [N] as atomic_numbers")
        if atom_edge_index.ndim != 2 or atom_edge_index.shape[0] != 2:
            raise ValueError("atom_edge_index must have shape [2,Ea]")
        if atom_edge_types.shape != (atom_edge_index.shape[1],):
            raise ValueError("atom_edge_types must have shape [Ea]")
        if surface_curvatures.ndim != 3 or surface_curvatures.shape[2] != 3:
            raise ValueError("surface_curvatures must have shape [M,S,3]")

        atom_count    = len(atomic_numbers)
        surface_count = len(surface_curvatures)
        if surface_count == 0:
            raise ValueError("WISDOM requires at least one surface point")
        if surface_curvatures.shape[1] * 3 != self.curvature_features:
            raise ValueError("flattened surface curvature width disagrees with model configuration")
        if surface_edge_index.ndim != 2 or surface_edge_index.shape[0] != 2:
            raise ValueError("surface_edge_index must have shape [2,Es]")
        if surface_atom_edge_index.ndim != 2 or surface_atom_edge_index.shape[0] != 2:
            raise ValueError("surface_atom_edge_index must have shape [2,Esa]")

        if (
            atomic_numbers.min() < 0
            or atomic_numbers.max() >= self.atomic_number_embedding.num_embeddings
        ):
            raise ValueError("atomic number is outside the configured embedding table")
        if residue_type_ids.min() < 0 or residue_type_ids.max() >= self.residue_type_count:
            raise ValueError("residue type is outside the configured embedding table")
        if atom_edge_types.numel() and (atom_edge_types.min() < 0 or atom_edge_types.max() > 2):
            raise ValueError("atom relation IDs must be 0=spatial, 1=covalent, or 2=both")
        if atom_edge_index.numel() and (
            atom_edge_index.min() < 0 or atom_edge_index.max() >= atom_count
        ):
            raise ValueError("atomic graph endpoints are out of range")
        if surface_edge_index.numel() and (
            surface_edge_index.min() < 0 or surface_edge_index.max() >= surface_count
        ):
            raise ValueError("surface graph endpoints are out of range")

        # Atomic features contain exactly the feature families selected by the v1 HPO candidate.
        atom_features = self.atomic_number_embedding(atomic_numbers)
        if self.residue_type_embedding is not None:
            residue_features = self.residue_type_embedding(residue_type_ids)
            atom_features    = torch.cat((atom_features, residue_features), dim=1)

        atom_embeddings = self.atomic_encoder(atom_features, atom_edge_index, atom_edge_types)
        if atom_embeddings.shape != (atom_count, self.hidden_dim):
            raise ValueError("atomic encoder output must have shape [N,hidden_dim]")

        # Every surface point receives the arithmetic mean of only its associated atom embeddings.
        surface_ids = surface_atom_edge_index[0]
        atom_ids    = surface_atom_edge_index[1]
        if surface_ids.numel() and (
            surface_ids.min() < 0
            or surface_ids.max() >= surface_count
            or atom_ids.min() < 0
            or atom_ids.max() >= atom_count
        ):
            raise ValueError("surface-to-atom endpoints are out of range")
        aggregated_atoms = Scatter.mean(atom_embeddings[atom_ids], surface_ids, surface_count)

        curvature_features = surface_curvatures.flatten(start_dim=1)
        surface_features   = self.surface_projection(
            torch.cat((curvature_features, aggregated_atoms), dim=1)
        )
        surface_embeddings = self.surface_encoder(surface_features, surface_edge_index)
        if surface_embeddings.shape != (surface_count, self.hidden_dim):
            raise ValueError("surface encoder output must have shape [M,hidden_dim]")

        surface_logits = self.local_head(surface_embeddings).squeeze(-1)
        return surface_embeddings, surface_logits

    def validate_surface_bags(
        self,
        surface_area_weights : Tensor,
        surface_batch        : Tensor,
        surface_count        : int,
    ) -> int:
        """Validate disjoint protein ownership and return the number of non-empty bags.

        Args:
            surface_area_weights: Positive finite represented-area values with shape ``[M]``.
            surface_batch: Ordered consecutive integer protein IDs with shape ``[M]``.
            surface_count: Expected number ``M`` of surface points.

        Returns:
            Number of proteins ``B = max(surface_batch) + 1``.

        Raises:
            ValueError: If shapes, area values, or protein IDs violate the disjoint batching
                contract.
        """
        if surface_count < 1:
            raise ValueError("surface_count must be positive")
        if (
            surface_area_weights.shape != (surface_count,)
            or surface_batch.shape != (surface_count,)
        ):
            raise ValueError("surface weights and batch IDs must have shape [M]")
        if not torch.isfinite(surface_area_weights).all() or torch.any(surface_area_weights <= 0):
            raise ValueError("surface area weights must be finite and positive")
        if surface_batch.min() != 0 or not torch.all(surface_batch[1:] >= surface_batch[:-1]):
            raise ValueError("surface_batch must contain ordered consecutive protein IDs")

        protein_count = int(surface_batch.max().item()) + 1
        if torch.any(torch.bincount(surface_batch, minlength=protein_count) == 0):
            raise ValueError("surface_batch must not skip protein IDs")
        return protein_count

    def forward(
        self,
        atomic_numbers          : Tensor,
        residue_type_ids        : Tensor,
        atom_edge_index         : Tensor,
        atom_edge_types         : Tensor,
        surface_curvatures      : Tensor,
        surface_edge_index      : Tensor,
        surface_atom_edge_index : Tensor,
        surface_area_weights    : Tensor,
        surface_batch           : Tensor,
    ) -> dict[str, Tensor]:
        """Predict one protein logit with MAX multiple-instance pooling.

        After :meth:`encode_surface` produces local logits ``l_p``, protein ``b`` receives
        ``L_b = max_{p: batch(p)=b} l_p``. This is the existential MIL assumption: one sufficiently
        positive surface point can make the complete protein positive. Surface area does not alter
        v1 pooling; its tensor is validated and retained in the common interface because v2 uses it
        for physically weighted regional consensus.

        Args:
            atomic_numbers: Integer element IDs with shape ``[N]``.
            residue_type_ids: Integer residue IDs with shape ``[N]``.
            atom_edge_index: Bidirectional atomic edges with shape ``[2,Ea]``.
            atom_edge_types: Atomic relation IDs with shape ``[Ea]``.
            surface_curvatures: Multiscale curvature with shape ``[M,S,3]``.
            surface_edge_index: Bidirectional surface edges with shape ``[2,Es]``.
            surface_atom_edge_index: Surface-to-atom incidence with shape ``[2,Esa]``.
            surface_area_weights: Positive represented-area weights with shape ``[M]``.
            surface_batch: Ordered, consecutive protein IDs with shape ``[M]``.

        Returns:
            Mapping with protein logits ``[B]`` and local surface logits ``[M]``.

        Raises:
            ValueError: If the encoding contract fails, weights are not finite and positive, or
                batch IDs do not form non-empty consecutive groups beginning at zero.
        """
        protein_count = self.validate_surface_bags(
            surface_area_weights,
            surface_batch,
            len(surface_curvatures),
        )

        surface_embeddings, surface_logits = self.encode_surface(
            atomic_numbers=atomic_numbers,
            residue_type_ids=residue_type_ids,
            atom_edge_index=atom_edge_index,
            atom_edge_types=atom_edge_types,
            surface_curvatures=surface_curvatures,
            surface_edge_index=surface_edge_index,
            surface_atom_edge_index=surface_atom_edge_index,
        )

        # LambdaForge's sparse pooling keeps proteins disjoint without padding their point clouds.
        protein_logits = self.global_max_pooling(
            surface_logits.unsqueeze(-1),
            surface_batch,
            protein_count,
        ).squeeze(-1)
        return {
            "logits": protein_logits,
            "surface_logits": surface_logits,
            "surface_embeddings": surface_embeddings,
        }
