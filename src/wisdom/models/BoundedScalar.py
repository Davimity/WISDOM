"""Trainable scalar whose physical value remains inside an interpretable interval."""

import math

import torch

from torch import Tensor, nn


class BoundedScalar(nn.Module):
    """Represent one bounded scalar through a stable unconstrained parameter."""

    def __init__(
        self,
        initial_value: float,
        lower_bound : float,
        upper_bound : float,
        logarithmic : bool,
    ) -> None:
        """Initialize a scalar through a sigmoid in linear or logarithmic coordinates.

        Let ``theta`` be the stored unconstrained parameter and ``q=sigmoid(theta)``. In linear
        mode the returned value is ``lower + q(upper-lower)``. In logarithmic mode it is
        ``exp(log(lower) + q(log(upper)-log(lower)))``. The latter gives comparable resolution
        across orders of magnitude while both mappings stay within the declared bounds.
        At machine precision, saturation can reach an endpoint and produce a zero gradient.

        Args:
            initial_value: Interpretable initial scalar strictly inside the bounds.
            lower_bound: Smallest permitted value. It must be positive in logarithmic mode.
            upper_bound: Largest permitted value, strictly greater than ``lower_bound``.
            logarithmic: Interpolate in log-space when true and ordinary value-space otherwise.

        Raises:
            ValueError: If bounds or the initial value cannot define the requested transform.
        """
        super().__init__()

        if not lower_bound < initial_value < upper_bound:
            raise ValueError("bounded scalar initialization must lie strictly inside its bounds")
        if logarithmic and lower_bound <= 0.0:
            raise ValueError("logarithmic bounded scalars require a positive lower bound")

        self.lower_bound = float(lower_bound)
        self.upper_bound = float(upper_bound)
        self.logarithmic = bool(logarithmic)

        if self.logarithmic:
            lower       = math.log(self.lower_bound)
            upper       = math.log(self.upper_bound)
            initial     = math.log(initial_value)
            normalized  = (initial - lower) / (upper - lower)
        else:
            normalized = (initial_value - self.lower_bound) / (
                self.upper_bound - self.lower_bound
            )

        raw = math.log(normalized / (1.0 - normalized))
        self.raw = nn.Parameter(torch.tensor(raw, dtype=torch.float32))

    def forward(self) -> Tensor:
        """Return the differentiable scalar in its physical domain.

        Returns:
            Scalar float tensor on the same device as ``raw``. Its value stays within the
            configured bounds and retains autograd to ``raw`` until numerical saturation.
        """
        fraction = torch.sigmoid(self.raw)
        if self.logarithmic:
            lower = self.raw.new_tensor(math.log(self.lower_bound))
            width = self.raw.new_tensor(
                math.log(self.upper_bound) - math.log(self.lower_bound)
            )
            # The exponential may round just outside an endpoint (e.g. exp(log(200))). This
            # guards arithmetic roundoff inside the transform, not optimizer state after a step.

            return torch.exp(lower + fraction * width).clamp(self.lower_bound, self.upper_bound)

        lower = self.raw.new_tensor(self.lower_bound)
        width = self.raw.new_tensor(self.upper_bound - self.lower_bound)
        return lower + fraction * width
