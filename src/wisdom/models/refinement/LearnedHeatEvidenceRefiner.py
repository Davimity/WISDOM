"""Heat evidence refinement with exactly one shared bounded physical-length parameter."""

from torch import Tensor
from wisdom.models.BoundedScalar import BoundedScalar
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from wisdom.models.refinement.SurfaceEvidenceRefiner import SurfaceEvidenceRefiner
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext


class LearnedHeatEvidenceRefiner(SurfaceEvidenceRefiner):
    """Learn one global heat length, not a length per point or protein."""

    def __init__(self, initial_length: float = 3.0) -> None:
        """Initialize the log-coordinate bounded length.

        Args:
            initial_length: Initial Å length strictly inside (0.05,12).

        Raises:
            ValueError: If the initialization is outside the strict domain.
        """
        super().__init__()
        self.length = BoundedScalar(initial_length, 0.05, 12.0, logarithmic=True)

    def forward(self, logits: Tensor, context: SurfaceEvidenceContext) -> Tensor:
        """Diffuse evidence with gradient to both logits and the learned length.

        Args:
            logits: Raw evidence [M].
            context: Existing spectral packs and boundaries.

        Returns:
            Phi exp(-length² Lambda) Phi^T A logits, retaining the tensor length in autograd.
        """
        return DiffusionSurfaceEncoder.diffuse(
            logits, context.operators, context.surface_ptr, self.length(),
        )

    def parameter_values(self) -> dict[str, float]:
        """Expose the physical length restored/trained, not its unconstrained coordinate.

        Returns:
            surface_refiner_heat_length in Å, detached only for scalar logging.
        """
        return {"surface_refiner_heat_length": float(self.length().detach().cpu())}
