"""One intrinsic DiffusionNet feature block."""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class DiffusionBlock(nn.Module):
    """Combine learned spectral diffusion and frame-invariant gradient products."""

    def __init__(
        self,
        hidden_dim  : int,
        dropout     : float = 0.0,
        initial_time: float = 1.0,
        residual_initial_scale: float | None = None,
    ) -> None:
        """Create channelwise positive diffusion times and the residual pointwise MLP.

        This follows the core DiffusionNet block of Sharp et al. (2022): diffuse scalar channels,
        compute tangent gradients, convert them to frame-invariant learned inner products, and
        update features pointwise through a residual MLP.

        Args:
            hidden_dim: Scalar feature width ``H``.
            dropout: Pointwise MLP dropout probability in ``[0,1)``.
            initial_time: Initial physical diffusion time in Å². A diffusion length is roughly
                ``sqrt(t)`` ångströms.
            residual_initial_scale: Optional learnable per-channel residual scale. ``None`` keeps
                the historical unscaled update; zero implements ReZero and ``0.01`` implements
                the planned LayerScale-like start.

        Raises:
            ValueError: If width/time is non-positive or dropout leaves ``[0,1)``.
        """
        super().__init__()
        if (
            hidden_dim < 1
            or initial_time <= 0.0
            or not 0.0 <= dropout < 1.0
            or (residual_initial_scale is not None and residual_initial_scale < 0.0)
        ):
            raise ValueError("DiffusionBlock width/time/dropout is invalid")

        initial_raw = torch.log(torch.expm1(torch.tensor(initial_time)))
        self.raw_times = nn.Parameter(initial_raw.repeat(hidden_dim))
        self.gradient_mixing = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(3 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_scale = (
            nn.Parameter(torch.full((hidden_dim,), residual_initial_scale))
            if residual_initial_scale is not None
            else None
        )

    @property
    def diffusion_times(self) -> Tensor:
        """Return strictly positive learned channel times in physical Å².

        Returns:
            ``float [H]`` values ``softplus(raw_time)+1e-8``.
        """
        return F.softplus(self.raw_times) + 1.0e-8

    def forward(
        self,
        features    : Tensor,
        mass        : Tensor,
        eigenvalues : Tensor,
        eigenvectors: Tensor,
        gradient    : Tensor,
    ) -> Tensor:
        """Apply one residual intrinsic feature update.

        Spectral heat diffusion is
        ``Phi exp(-Lambda t) Phi^T M X``. Tangent gradients are ``gx=Gx X`` and ``gy=Gy X``.
        The learned scalar feature ``gx*A(gx) + gy*A(gy)`` is invariant to any orthonormal change
        of the two tangent-frame axes, unlike directly concatenating ``gx`` and ``gy``.

        Args:
            features: Scalar surface features ``float [M,H]``.
            mass: Positive lumped mass ``float [M]``.
            eigenvalues: Low Laplacian spectrum ``float [Q]`` in Å⁻².
            eigenvectors: Mass-orthonormal modes ``float [M,Q]``.
            gradient: Vertically stacked sparse tangent derivatives ``[2M,M]``. Its first ``M``
                rows contain ``Gx`` and its last ``M`` rows contain ``Gy``.

        Returns:
            Updated scalar features ``float [M,H]``.
        """
        input_dtype    = features.dtype
        values         = features.float()
        mass32         = mass.float()
        eigenvalues32  = eigenvalues.float()
        eigenvectors32 = eigenvectors.float()

        coefficients = eigenvectors32.T @ (mass32[:, None] * values)
        attenuation  = torch.exp(-eigenvalues32[:, None] * self.diffusion_times.float()[None, :])
        diffused     = eigenvectors32 @ (attenuation * coefficients)

        # Stacking gives [Gx; Gy] X = [Gx X; Gy X] in one sparse launch. The same bias-free
        # gradient_mixing map acts independently on each row, so it is also exactly equivalent to
        # applying the map separately to gx and gy before the invariant elementwise product.

        point_count     = len(values)
        gradients       = self.sparse_multiply(gradient, values)
        grad_x           = gradients[:point_count]
        grad_y           = gradients[point_count:]
        mixed_gradients  = self.gradient_mixing(gradients)
        mixed_x          = mixed_gradients[:point_count]
        mixed_y          = mixed_gradients[point_count:]
        invariant_gradient = torch.tanh(grad_x * mixed_x + grad_y * mixed_y)

        update = self.mlp(torch.cat((values, diffused, invariant_gradient), dim=1))
        if self.residual_scale is not None:
            update = update * self.residual_scale
        return (values + update).to(input_dtype)

    @staticmethod
    def sparse_multiply(operator: Tensor, values: Tensor) -> Tensor:
        """Apply one sparse surface operator through a float32 autocast island.

        CUDA implements sparse COO multiplication for float32 but not for BF16 through
        ``addmm_sparse_cuda``. Merely converting the operands with ``.float()`` is insufficient:
        an enclosing autocast context casts the operation back to BF16. This method temporarily
        disables autocast only for the sparse multiplication, preserving mixed precision in the
        dense learned layers around it.

        Args:
            operator: Sparse derivative matrix ``float [M,M]``.
            values: Dense point features ``float [M,H]``.

        Returns:
            Float32 derivative features ``[M,H]`` with autograd connected to ``values``.
        """
        with torch.autocast(device_type=values.device.type, enabled=False):
            return torch.sparse.mm(operator.float(), values.float())
