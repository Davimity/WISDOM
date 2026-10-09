"""Fixed physical-length heat refinement through the existing spectral primitive."""

from torch import Tensor
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.refinement.SurfaceEvidenceRefiner import SurfaceEvidenceRefiner
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext


class HeatEvidenceRefiner(SurfaceEvidenceRefiner):
    """Diffuse logits, rather than changing DiffusionNet's embedding propagation."""

    def __init__(self, length: float = 3.0) -> None:
        """Set a fixed diffusion length.

        Args:
            length: Nonnegative physical length in Å; zero is exact identity.

        Raises:
            ValueError: If length is negative.
        """
        super().__init__()
        if length < 0.0:
            raise ValueError("surface refiner heat length must be nonnegative")
        self.length = length

    def forward(self, logits: Tensor, context: SurfaceEvidenceContext) -> Tensor:
        """Use Phi exp(-length² Lambda) Phi^T A independently per protein.

        Args:
            logits: Raw evidence [M].
            context: Existing spectral packs and point boundaries.

        Returns:
            Same-shaped heat-refined field with gradient to logits. Positive lengths
            use truncated modes; length=0 bypasses the projection entirely.
        """
        return DiffusionSurfaceEncoder.diffuse(
            logits, context.operators, context.surface_ptr, self.length,
        )
