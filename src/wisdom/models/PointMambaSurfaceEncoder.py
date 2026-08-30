"""Compact serialized state-space surface encoder inspired by PointMamba."""

from __future__ import annotations

import torch
from torch import Tensor, nn

from wisdom.models.PTV3SurfaceEncoder import PTV3SurfaceEncoder


class PointMambaSurfaceEncoder(nn.Module):
    """Run bidirectional gated diagonal state scans over Morton-serialized surface points."""

    def __init__(self, hidden_dim: int, layers: int = 2, dropout: float = 0.0) -> None:
        """Build a dependency-free compact state-space comparison.

        PointMamba (Liang et al., NeurIPS 2024) combines point serialization with Mamba state-space
        blocks. Its maintained reference requires a specialized Mamba/CUDA stack not otherwise
        needed by WISDOM. This controlled implementation tests the same linear-memory global-state
        hypothesis using bidirectional learned diagonal scans, without claiming checkpoint or
        layer-level compatibility with the reference repository.

        Args:
            hidden_dim: Surface feature and recurrent-state width ``H``.
            layers: Number of residual state-space scans.
            dropout: Residual output dropout in ``[0,1)``.

        Raises:
            ValueError: If dimensions or dropout are invalid.
        """
        super().__init__()
        if hidden_dim < 1 or layers < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("PointMamba encoder dimensions or dropout are invalid")

        self.gates = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.inputs = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(layers))
        self.outputs = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
            )
            for _ in range(layers)
        )

    def forward(
        self,
        features : Tensor,
        positions: Tensor,
        surface_ptr: Tensor,
    ) -> Tensor:
        """Encode each protein independently and restore its immutable surface order.

        Args:
            features: Concatenated scalar surface features ``[M_total,H]``.
            positions: Surface coordinates ``[M_total,3]`` used only for deterministic ordering.
            surface_ptr: Prefix protein boundaries ``[B+1]``.

        Returns:
            Concatenated surface embeddings ``[M_total,H]``.
        """
        outputs: list[Tensor] = []
        for protein_index in range(len(surface_ptr) - 1):
            start = int(surface_ptr[protein_index])
            stop  = int(surface_ptr[protein_index + 1])
            local_positions = positions[start:stop]
            centered = local_positions - local_positions.mean(dim=0, keepdim=True)
            extent   = centered.abs().amax().clamp_min(1.0e-6)
            order    = torch.argsort(
                PTV3SurfaceEncoder._morton_codes(centered / extent),
                stable=True,
            )
            inverse = torch.empty_like(order)
            inverse[order] = torch.arange(len(order), device=order.device)

            values = features[start:stop][order]
            for gate_layer, input_layer, output_layer in zip(
                self.gates,
                self.inputs,
                self.outputs,
                strict=True,
            ):
                gates      = torch.sigmoid(gate_layer(values))
                candidates = torch.tanh(input_layer(values))
                forward    = self._scan(gates, candidates)
                backward   = self._scan(gates.flip(0), candidates.flip(0)).flip(0)
                values     = values + output_layer(torch.cat((forward, backward), dim=1))
            outputs.append(values[inverse])
        return torch.cat(outputs)

    @staticmethod
    def _scan(gates: Tensor, candidates: Tensor) -> Tensor:
        """Evaluate one differentiable diagonal state recurrence.

        Args:
            gates: Retention gates ``[M,H]`` in ``(0,1)``.
            candidates: Proposed states ``[M,H]``.

        Returns:
            States satisfying ``h_i=g_i*h_(i-1)+(1-g_i)*u_i`` with shape ``[M,H]``.
        """
        state = torch.zeros_like(candidates[0])
        states: list[Tensor] = []
        for index in range(len(candidates)):
            state = gates[index] * state + (1.0 - gates[index]) * candidates[index]
            states.append(state)
        return torch.stack(states)
