"""Compact serialized Point Transformer V3 surface encoder."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class PTV3SurfaceEncoder(nn.Module):
    """Apply bounded window attention along deterministic Morton surface serializations."""

    def __init__(
        self,
        hidden_dim: int,
        layers    : int = 2,
        dropout   : float = 0.0,
        patch_size: int = 64,
    ) -> None:
        """Build a compact PTv3-inspired serialized-attention comparison.

        Point Transformer V3 obtains efficiency from point serialization and patch attention. This
        implementation keeps that mechanism while omitting hierarchical voxel pooling, because
        WISDOM must preserve one output per immutable surface point for localization.

        Args:
            hidden_dim: Surface feature width ``H``.
            layers: Number of alternating serialized-attention blocks.
            dropout: Attention/output dropout in ``[0,1)``.
            patch_size: Maximum points in one attention window, bounding quadratic attention.

        Raises:
            ValueError: If dimensions/patch size are non-positive or dropout is invalid.
        """
        super().__init__()
        if hidden_dim < 1 or layers < 1 or patch_size < 2 or not 0.0 <= dropout < 1.0:
            raise ValueError("PTv3 encoder dimensions, patch size, or dropout are invalid")

        heads = 4 if hidden_dim % 4 == 0 else 1
        self.radial_projection = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.attention = nn.ModuleList(
            nn.MultiheadAttention(
                hidden_dim,
                heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(layers)
        )
        self.updates = nn.ModuleList(
            nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, 2 * hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(2 * hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )
        self.patch_size = patch_size

    def forward(
        self,
        features : Tensor,
        positions: Tensor,
        surface_ptr: Tensor,
    ) -> Tensor:
        """Encode each protein by bounded Morton-ordered local attention.

        Args:
            features: Concatenated surface features ``[M_total,H]``.
            positions: Surface coordinates ``[M_total,3]`` in ångströms.
            surface_ptr: Prefix protein boundaries ``[B+1]``.

        Returns:
            Surface embeddings ``[M_total,H]`` restored to original point order.
        """
        outputs: list[Tensor] = []
        for protein_index in range(len(surface_ptr) - 1):
            start = int(surface_ptr[protein_index])
            stop  = int(surface_ptr[protein_index + 1])
            local_positions = positions[start:stop]
            centered = local_positions - local_positions.mean(dim=0, keepdim=True)
            extent   = centered.abs().amax().clamp_min(1.0e-6)
            normalized_for_order = centered / extent
            order = torch.argsort(
                self._morton_codes(normalized_for_order),
                stable=True,
            )
            inverse    = torch.empty_like(order)
            inverse[order] = torch.arange(len(order), device=order.device)

            # The normalized coordinates above choose only a serialization. The learned feature
            # receives physical centroid radius in ångströms, not a unit-box XYZ vector, so global
            # translation/rotation cannot enter the MLP and molecular scale remains observable.

            radial_distance = torch.linalg.vector_norm(centered, dim=1, keepdim=True)
            values = features[start:stop] + self.radial_projection(radial_distance)
            values = values[order]
            for layer_index, (attention, update) in enumerate(
                zip(self.attention, self.updates, strict=True)
            ):
                shift = self.patch_size // 2 if layer_index % 2 else 0
                shifted = torch.roll(values, shifts=-shift, dims=0) if shift else values
                chunks: list[Tensor] = []
                for chunk_start in range(0, len(shifted), self.patch_size):
                    chunk = shifted[chunk_start : chunk_start + self.patch_size].unsqueeze(0)
                    attended, _ = attention(chunk, chunk, chunk, need_weights=False)
                    chunks.append((chunk + attended + update(chunk + attended)).squeeze(0))
                values = torch.cat(chunks)
                if shift:
                    values = torch.roll(values, shifts=shift, dims=0)
            outputs.append(values[inverse])
        return torch.cat(outputs)

    @staticmethod
    def _morton_codes(normalized_positions: Tensor) -> Tensor:
        """Interleave ten quantized coordinate bits into deterministic 3D Morton codes.

        Args:
            normalized_positions: Centered coordinates within approximately ``[-1,1]``.

        Returns:
            Integer Morton code ``long [M]``.
        """
        quantized = ((normalized_positions + 1.0) * 511.5).clamp(0, 1023).long()
        codes = torch.zeros(len(quantized), dtype=torch.long, device=quantized.device)
        for bit in range(10):
            codes |= ((quantized[:, 0] >> bit) & 1) << (3 * bit)
            codes |= ((quantized[:, 1] >> bit) & 1) << (3 * bit + 1)
            codes |= ((quantized[:, 2] >> bit) & 1) << (3 * bit + 2)
        return codes
