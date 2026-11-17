"""Scalar distance representation composed with the public LF SchNet interaction."""

import torch

from torch import nn


class DirectRadialBasis(nn.Module):
    """Expose d/r as one radial channel; LF still owns cutoff, gating, messages and updates."""

    def __init__(self, radius: float) -> None:
        """Bind a fixed physical scale without learned or population-dependent statistics.

        Args:
            radius: Positive reference radius in angstroms, identical for both relations.

        Raises:
            ValueError: The physical normalization scale is not positive.
        """
        super().__init__()
        if radius <= 0:
            raise ValueError("direct radial radius must be positive")
        self.radius = radius
        self.num_radial = 1

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        """Return d/r with shape [E,1], without clipping or changing supplied edges.

        Args:
            distances: Finite nonnegative physical edge lengths [E], in angstroms.

        Returns:
            Floating [E,1] normalized lengths. LF applies the existing distance gate and cutoff
            after this representation, including support=1+g*(cutoff(d)-1).
        """
        return distances[:, None] / self.radius
