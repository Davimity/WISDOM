"""Local geometric conductance and stable residual smoothing of scalar evidence."""

from torch import Tensor
from wisdom.models.refinement.SurfaceEvidenceRefiner import SurfaceEvidenceRefiner
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext


class GeometricAnisotropicEvidenceRefiner(SurfaceEvidenceRefiner):
    """Reduce evidence exchange across sharp spatial, normal or curvature differences."""

    def __init__(
        self,
        strength       : float = 0.5,
        steps          : int = 2,
        geometry_sigma : float = 2.0,
        normal_sigma   : float = 0.25,
        curvature_sigma: float = 1.0,
    ) -> None:
        """Configure a fixed local geometry operator without trainable parameters.

        Args:
            strength: Convex evidence-mixing fraction in [0,1].
            steps: Nonnegative update count.
            geometry_sigma: Positive spatial bandwidth in Å.
            normal_sigma: Positive dimensionless normal-misalignment bandwidth.
            curvature_sigma: Positive signed-log curvature bandwidth.

        Raises:
            ValueError: If mixing cannot remain convex or bandwidths are nonpositive.
        """
        super().__init__()
        if not 0.0 <= strength <= 1.0 or steps < 0:
            raise ValueError("surface refiner requires strength in [0,1] and nonnegative steps")
        if min(geometry_sigma, normal_sigma, curvature_sigma) <= 0.0:
            raise ValueError("surface refiner geometry bandwidths must be positive")
        self.strength        = strength
        self.steps           = steps
        self.geometry_sigma  = geometry_sigma
        self.normal_sigma    = normal_sigma
        self.curvature_sigma = curvature_sigma

    def conductance(
        self, context: SurfaceEvidenceContext, left: Tensor, right: Tensor,
    ) -> Tensor:
        """Obtain generic symmetric geometric edge weights, independent of labels.

        Args:
            context: Stored surface geometry.
            left: First edge endpoints [E].
            right: Second edge endpoints [E].

        Returns:
            Gaussian spatial/curvature times exponential normal compatibility [E].
        """
        return context.weights(
            left, right, self.geometry_sigma, self.normal_sigma, self.curvature_sigma,
        )

    def forward(self, logits: Tensor, context: SurfaceEvidenceContext) -> Tensor:
        """Perform row-normalized convex updates on the stored neighborhood union.

        Args:
            logits: Raw local evidence [M].
            context: Existing positions, normals, curvatures and neighbor IDs.

        Returns:
            Refined [M] evidence with intact autograd to logits and no protein mixing.
        """
        left, right = context.edges()
        weights = self.conductance(context, left, right)
        return self.residual(logits, left, right, weights, self.strength, self.steps)
