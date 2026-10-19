"""Explicit protein-only loss balance for the four global/surface head hypotheses."""

import math

from torch import Tensor
from collections.abc import Mapping
from torch.nn import functional as F
from wisdom.models.HeadType import HeadType, HeadLossBalance


class ProteinTaskLoss:
    """Balance two protein BCEs without changing the scale of weak priors or reading local GT."""

    def __init__(
        self,
        head_type       : str = "single",
        balance         : str = "legacy_sum",
        surface_fraction: float = 0.5,
        direct_weight   : float = 1.0,
        surface_weight  : float = 1.0,
    ) -> None:
        """Resolve one explicit weighting policy before data loading or model allocation.

        Args:
            head_type: single, dual, global_context or film.
            balance: legacy_sum retains historical w_d*BCE_d+w_s*BCE_s; normalized uses
                (1-rho)*BCE_d+rho*BCE_s. Single always uses its original BCE unchanged.
            surface_fraction: rho in [0,1], used only by normalized multi-head models.
                rho=1 is rejected: an unsupervised direct head cannot select checkpoints.
            direct_weight: Nonnegative historical direct coefficient; positive for multi-head.
            surface_weight: Nonnegative historical surface coefficient.

        Raises:
            ValueError: Nonfinite/negative weights, an invalid fraction or policy, ambiguous
                normalized-plus-legacy weights, or a zero supervised direct-head coefficient.
        """
        self.head_type = HeadType(head_type)
        self.balance   = HeadLossBalance(balance)

        # Reject ambiguous policies rather than silently reinterpreting archived coefficients.

        if not all(math.isfinite(v) for v in (surface_fraction, direct_weight, surface_weight)):
            raise ValueError("protein head-loss coefficients must be finite")
        if not 0.0 <= surface_fraction <= 1.0 or min(direct_weight, surface_weight) < 0.0:
            raise ValueError("surface_loss_fraction must be in [0,1]; legacy weights must be >=0")
        if self.balance is HeadLossBalance.NORMALIZED:
            if (direct_weight, surface_weight) != (1.0, 1.0):
                raise ValueError("normalized head loss cannot also override legacy head weights")
            self.direct_weight, self.surface_weight = 1.0 - surface_fraction, surface_fraction
        else:
            if surface_fraction != 0.5:
                raise ValueError("surface_loss_fraction is inactive with legacy_sum")
            self.direct_weight, self.surface_weight = direct_weight, surface_weight
        if self.head_type is not HeadType.SINGLE and self.direct_weight <= 0.0:
            raise ValueError("primary direct head requires a positive supervised BCE coefficient")
        if self.head_type is HeadType.SINGLE:
            self.direct_weight, self.surface_weight = 1.0, 0.0

    def __call__(self, output: Mapping[str, Tensor], target: Tensor) -> dict[str, Tensor]:
        """Compute exact BCE components using global binary labels only.

        Args:
            output: logits[B] is the selected primary branch; multi-head models additionally
                supply surface_protein_logits[B]. No surface target is accepted or inspected.
            target: Protein labels [B], floating 0/1 on the logit device.

        Returns:
            total plus direct and surface unweighted BCE scalars. Single's surface component
            is differentiable zero; its total is exactly the original BCE, not rescaled.

        Raises:
            ValueError: The output branch contract disagrees with the configured head type.
        """
        multi = "surface_protein_logits" in output
        if multi != (self.head_type is not HeadType.SINGLE):
            raise ValueError("protein output branches disagree with head_type")
        direct  = F.binary_cross_entropy_with_logits(output["logits"], target)
        surface = (F.binary_cross_entropy_with_logits(output["surface_protein_logits"], target)
                   if multi else direct * 0.0)
        total   = (self.direct_weight * direct + self.surface_weight * surface) if multi else direct
        return {"total": total, "direct": direct, "surface": surface}
