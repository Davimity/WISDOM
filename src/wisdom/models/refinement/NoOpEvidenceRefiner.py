"""Exact legacy evidence control, without parameters or state-dict entries."""

from torch import Tensor
from wisdom.models.refinement.SurfaceEvidenceRefiner import SurfaceEvidenceRefiner
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext


class NoOpEvidenceRefiner(SurfaceEvidenceRefiner):
    """Return the same tensor object; no projection or numerical approximation occurs."""

    def forward(self, logits: Tensor, context: SurfaceEvidenceContext) -> Tensor:
        """Preserve the historical field exactly.

        Args:
            logits: Original evidence [M].
            context: Unused borrowed batch references.

        Returns:
            logits itself, preserving values, storage and autograd.
        """
        return logits
