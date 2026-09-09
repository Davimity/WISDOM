"""One globally shared Hard-Concrete gate with an analytic L0 probability."""

import math
import torch

from torch import Tensor, nn


class HardConcreteGate(nn.Module):
    """Learn whether one named semantic source remains active.

    The implementation follows the stretched binary-concrete construction from Louizos et al.
    A logistic sample is relaxed with temperature ``beta``, stretched from ``(0, 1)`` to
    ``(gamma, zeta)``, and clamped back to ``[0, 1]``. Clamping creates exact zero and one values
    while preserving a reparameterized gradient in the interior.
    """

    BETA             = 2.0 / 3.0
    GAMMA            = -0.1
    ZETA             = 1.1
    INITIAL_ACTIVE   = 0.95
    SAMPLING_EPSILON = 1.0e-6

    def __init__(self) -> None:
        """Initialize ``log_alpha`` so the analytic probability of being nonzero is 0.95."""
        super().__init__()

        logit_probability = math.log(self.INITIAL_ACTIVE / (1.0 - self.INITIAL_ACTIVE))
        stretch_offset    = self.BETA * math.log(-self.GAMMA / self.ZETA)

        self.log_alpha = nn.Parameter(torch.tensor(logit_probability + stretch_offset))

    def probability_active(self) -> Tensor:
        """Return the analytic probability ``P(z > 0)`` as a scalar float32 tensor."""
        log_alpha = self.log_alpha.float()
        threshold = self.BETA * math.log(-self.GAMMA / self.ZETA)
        return torch.sigmoid(log_alpha - threshold)

    def deterministic_value(self) -> Tensor:
        """Return the reproducible stretched-sigmoid gate used outside training."""
        probability = torch.sigmoid(self.log_alpha.float())
        stretched   = probability * (self.ZETA - self.GAMMA) + self.GAMMA
        return stretched.clamp(0.0, 1.0)

    def forward(self, training: bool = True) -> Tensor:
        """Sample one gate during training or return its deterministic evaluation value.

        Args:
            training: Whether to draw a reparameterized stochastic sample.

        Returns:
            Scalar float32 tensor in ``[0,1]``. Training samples may be exactly zero or one.
        """
        if not training:
            return self.deterministic_value()

        uniform = torch.rand((), device=self.log_alpha.device, dtype=torch.float32)
        uniform = uniform.clamp(self.SAMPLING_EPSILON, 1.0 - self.SAMPLING_EPSILON)
        logistic = torch.log(uniform) - torch.log1p(-uniform)
        relaxed  = torch.sigmoid((logistic + self.log_alpha.float()) / self.BETA)
        stretched = relaxed * (self.ZETA - self.GAMMA) + self.GAMMA
        return stretched.clamp(0.0, 1.0)

    def extra_repr(self) -> str:
        """Expose fixed Hard-Concrete constants in module diagnostics."""
        return f"beta={self.BETA:g}, gamma={self.GAMMA:g}, zeta={self.ZETA:g}"
