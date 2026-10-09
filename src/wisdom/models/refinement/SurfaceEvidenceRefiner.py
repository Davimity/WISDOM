"""Orthogonal scalar-evidence refinement boundary and shared sparse message primitive."""

import torch

from torch import Tensor, nn
from abc import ABC, abstractmethod
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext


class SurfaceEvidenceRefiner(nn.Module, ABC):
    """Transform local logits before MIL pooling, preserving point order and autograd."""

    @abstractmethod
    def forward(self, logits: Tensor, context: SurfaceEvidenceContext) -> Tensor:
        """Refine logits using geometry or representations, never surface targets.

        Args:
            logits: Unnormalized evidence [M], differentiable with respect to the local head.
            context: Borrowed geometry and embeddings for disjoint proteins.

        Returns:
            Differentiable refined evidence [M] in the same order.
        """
        raise NotImplementedError

    def parameter_values(self) -> dict[str, float]:
        """Return small interpretable diagnostics for logging outside the forward graph.

        Returns:
            Scalar values only; fixed operators without diagnostics return an empty mapping.
        """
        return {}

    @staticmethod
    def messages(
        values : Tensor,
        left   : Tensor,
        right  : Tensor,
        weights: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Accumulate both directions of each unique undirected edge.

        Args:
            values: Scalar point field [M].
            left: First endpoint IDs [E].
            right: Second endpoint IDs [E].
            weights: Nonnegative symmetric edge weights [E].

        Returns:
            Unnormalized neighbor sum and weighted degree, each [M]. This vectorized
            scatter keeps the gradient to values/weights and uses O(M+E) storage.
        """
        sums   = torch.zeros_like(values)
        degree = torch.zeros_like(values)
        sums   = sums.index_add(0, left, weights * values[right])
        sums   = sums.index_add(0, right, weights * values[left])
        degree = degree.index_add(0, left, weights).index_add(0, right, weights)
        return sums, degree

    @classmethod
    def residual(
        cls,
        logits  : Tensor,
        left    : Tensor,
        right   : Tensor,
        weights : Tensor,
        strength: float,
        steps   : int,
    ) -> Tensor:
        """Apply stable convex, row-normalized neighbor updates a fixed number of times.

        Args:
            logits: Input evidence [M].
            left: First edge endpoints [E].
            right: Second edge endpoints [E].
            weights: Symmetric conductances [E] in [0,1].
            strength: Convex mixing coefficient in [0,1].
            steps: Nonnegative number of unrolled updates.

        Returns:
            Repeated (1-strength)*z + strength*neighbor_mean, same dtype as input.
            Zero-degree points keep their input exactly; computation uses FP32.
        """
        values = logits.float()
        for _ in range(steps):
            sums, degree = cls.messages(values, left, right, weights.float())
            mean = sums / degree.clamp_min(torch.finfo(values.dtype).tiny)
            values = torch.where(
                degree > 0.0, (1.0 - strength) * values + strength * mean, values,
            )
        return values.to(logits.dtype)
