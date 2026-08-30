"""Compact scalar/vector DeltaConv surface encoder."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from wisdom.models.DiffusionBlock import DiffusionBlock
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder


class DeltaConvSurfaceEncoder(nn.Module):
    """Alternate scalar and tangent-vector features through gradient/divergence operators."""

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0) -> None:
        """Build a compact coordinate-independent DeltaConv hypothesis.

        DeltaConv (Wiersma et al., ACM TOG 2022) learns compositions of vector-calculus operators.
        WISDOM reuses the same precomputed tangent gradients as DiffusionNet, maintains a two-axis
        vector feature, and returns scalar features so atom transfer, local head, and pooling remain
        fixed across the v3 comparison.

        Args:
            hidden_dim: Scalar/vector channel width ``H``.
            layers: Number of differential update blocks.
            dropout: Scalar update dropout probability in ``[0,1)``.

        Raises:
            ValueError: If dimensions or dropout are invalid.
        """
        super().__init__()
        if hidden_dim < 1 or layers < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("DeltaConv encoder dimensions or dropout are invalid")

        self.vector_mixing = nn.ModuleList(
            nn.Linear(hidden_dim, hidden_dim, bias=False) for _ in range(layers)
        )
        self.scalar_updates = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3 * hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )

    def forward(
        self,
        features : Tensor,
        operators: Sequence[Mapping[str, Tensor]],
        surface_ptr: Tensor,
    ) -> Tensor:
        """Apply differential blocks independently to every protein surface.

        Args:
            features: Concatenated scalar surface features ``[M_total,H]``.
            operators: Per-protein mass and sparse tangent gradients.
            surface_ptr: Prefix point boundaries ``[B+1]``.

        Returns:
            Concatenated scalar embeddings ``[M_total,H]`` in unchanged point order.
        """
        outputs: list[Tensor] = []
        for protein_index, operator in enumerate(operators):
            start  = int(surface_ptr[protein_index])
            stop   = int(surface_ptr[protein_index + 1])
            values = features[start:stop].float()
            mass   = operator["mass"].float()
            gradient_x, gradient_y = DiffusionSurfaceEncoder.sparse_gradients(
                operator,
                stop - start,
            )
            gradient_x = gradient_x.float()
            gradient_y = gradient_y.float()

            vector_x = torch.zeros_like(values)
            vector_y = torch.zeros_like(values)
            for mixing, update in zip(
                self.vector_mixing,
                self.scalar_updates,
                strict=True,
            ):
                vector_x = vector_x + mixing(DiffusionBlock.sparse_multiply(gradient_x, values))
                vector_y = vector_y + mixing(DiffusionBlock.sparse_multiply(gradient_y, values))
                divergence = (
                    DiffusionBlock.sparse_multiply(
                        gradient_x.transpose(0, 1),
                        mass[:, None] * vector_x,
                    )
                    + DiffusionBlock.sparse_multiply(
                        gradient_y.transpose(0, 1),
                        mass[:, None] * vector_y,
                    )
                ) / mass[:, None]
                vector_norm = torch.sqrt(vector_x.square() + vector_y.square() + 1.0e-8)
                values = values + update(torch.cat((values, divergence, vector_norm), dim=1))
            outputs.append(values.to(features.dtype))
        return torch.cat(outputs, dim=0)
