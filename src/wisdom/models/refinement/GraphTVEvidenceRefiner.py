"""Truncated differentiable descent for area-weighted, smoothed graph total variation."""

import torch

from torch import Tensor
from wisdom.models.refinement.SurfaceEvidenceRefiner import SurfaceEvidenceRefiner
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext


class GraphTVEvidenceRefiner(SurfaceEvidenceRefiner):
    """Unroll a fixed number of diagonally stabilized steps; do not claim an exact solve."""

    def __init__(
        self,
        weight   : float = 0.05,
        steps    : int = 5,
        step_size: float = 0.5,
        epsilon  : float = 0.1,
    ) -> None:
        """Set conservative explicit descent parameters.

        Args:
            weight: Nonnegative TV coefficient lambda.
            steps: Nonnegative solver iteration count.
            step_size: Relaxation fraction in (0,1] of a conservative diagonal step.
            epsilon: Positive smooth-absolute-value scale in logit units.

        Raises:
            ValueError: If the descent cannot use positive, stable settings.
        """
        super().__init__()
        if weight < 0.0 or steps < 0 or not 0.0 < step_size <= 1.0 or epsilon <= 0.0:
            raise ValueError("graph TV requires lambda>=0, steps>=0, step_size in (0,1], epsilon>0")
        self.weight    = weight
        self.steps     = steps
        self.step_size = step_size
        self.epsilon   = epsilon

    def forward(self, logits: Tensor, context: SurfaceEvidenceContext) -> Tensor:
        """Minimize 0.5*sum a_i(z_i-l_i)² + lambda*sum w_ij*sqrt((z_i-z_j)²+eps²).

        Args:
            logits: Raw [M] unary evidence l, also the initial iterate.
            context: Areas and sparse stored geometry. Areas are normalized to mean one
                separately per protein so lambda has comparable scale across proteins.

        Returns:
            Five (by default) differentiable FP32 descent steps, returned in input dtype.
            The diagonal a_i+2*lambda*degree_i/eps bounds smoothed-TV curvature;
            step_size<=1 keeps a conservative stable step. Zero lambda is exact identity.
        """
        if self.weight == 0.0 or self.steps == 0:
            return logits
        left, right = context.edges()
        weights = context.weights(left, right, 2.0, 0.25, 1.0)
        raw     = logits.float()
        values  = raw
        areas   = context.area_weights.float()
        totals  = areas.new_zeros(len(context.surface_ptr) - 1)
        totals  = totals.index_add(0, context.owners, areas)
        counts  = torch.diff(context.surface_ptr).to(areas.device)
        mass    = areas / (totals / counts)[context.owners].clamp_min(1.0e-12)
        mass    = mass.clamp_min(1.0e-6)
        _, degree = self.messages(raw, left, right, weights)
        diagonal = mass + 2.0 * self.weight * degree / self.epsilon

        # Unrolled optimization retains every iterate in autograd; no numerical solver detaches l.

        for _ in range(self.steps):
            difference = values[left] - values[right]
            flux = weights * difference / torch.sqrt(difference.square() + self.epsilon ** 2)
            gradient = mass * (values - raw)
            gradient = gradient.index_add(0, left, self.weight * flux)
            gradient = gradient.index_add(0, right, -self.weight * flux)
            values = values - self.step_size * gradient / diagonal
        return values.to(logits.dtype)
