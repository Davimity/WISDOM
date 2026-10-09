"""Sparse binary Potts CRF with differentiable damped mean-field updates."""

import torch

from torch import Tensor
from wisdom.models.refinement.SurfaceEvidenceRefiner import SurfaceEvidenceRefiner
from wisdom.models.refinement.SurfaceEvidenceContext import SurfaceEvidenceContext


class CRFEvidenceRefiner(SurfaceEvidenceRefiner):
    """Refine logits with fixed attractive pairwise compatibility, not DenseCRF."""

    def __init__(self, strength: float = 0.5, steps: int = 5, damping: float = 0.5) -> None:
        """Set the nonnegative Potts coefficient and fixed mean-field schedule.

        Args:
            strength: Nonnegative pairwise coefficient lambda.
            steps: Nonnegative number of mean-field iterations.
            damping: Fraction of each new mean-field probability in (0,1].

        Raises:
            ValueError: If smoothing is repulsive or damping/iteration settings are invalid.
        """
        super().__init__()
        if strength < 0.0 or steps < 0 or not 0.0 < damping <= 1.0:
            raise ValueError("CRF requires strength>=0, steps>=0 and damping in (0,1]")
        self.strength = strength
        self.steps    = steps
        self.damping  = damping

    def forward(self, logits: Tensor, context: SurfaceEvidenceContext) -> Tensor:
        """Unroll mean-field for E(y)=-sum l_i*y_i+lambda*sum c_ij*[y_i!=y_j].

        Args:
            logits: Binary unary log-odds [M].
            context: Sparse generic geometry, without targets.

        Returns:
            logit(q) [M] after damped updates towards sigmoid(l+lambda*C*(2q-1)).
            c_ij=w_ij/max(degree_i,degree_j) is symmetric and has row sums <=1,
            preserving an undirected Potts energy and bounded messages. q is clamped
            to FP32 epsilon before logit; isolated points and zero strength are exact identity.
        """
        if self.strength == 0.0 or self.steps == 0:
            return logits
        left, right = context.edges()
        weights = context.weights(left, right, 2.0, 0.25, 1.0)
        raw = logits.float()
        _, degree = self.messages(raw, left, right, weights)
        weights = weights / torch.maximum(degree[left], degree[right]).clamp_min(1.0e-30)
        probability = torch.sigmoid(raw)
        for _ in range(self.steps):
            message, _ = self.messages(2.0 * probability - 1.0, left, right, weights)
            proposal = torch.sigmoid(raw + self.strength * message)
            probability = (1.0 - self.damping) * probability + self.damping * proposal
        epsilon = torch.finfo(raw.dtype).eps
        refined = torch.logit(probability.clamp(epsilon, 1.0 - epsilon))
        return torch.where(degree > 0.0, refined, raw).to(logits.dtype)
