"""One intrinsic DiffusionNet feature block."""

from __future__ import annotations

import torch

from torch import Tensor, nn
from torch.nn import functional as F
from wisdom.models.surface.DiffusionReadout import DiffusionReadout


class DiffusionBlock(nn.Module):
    """Combine learned spectral diffusion and frame-invariant gradient products."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.0,
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
        self.readout_mode = DiffusionReadout.SINGLE
        self.use_gradient = True
        self.extra_raw_times: nn.Parameter | None = None
        self.mixture_logits: nn.Parameter | None = None

    def configure(
        self,
        readout_mode: str = "single",
        readouts: int = 1,
        scale_multipliers: tuple[float, ...] = (0.5, 2.0),
        use_gradient: bool = True,
    ) -> None:
        """Configure V13 reads after all shared parameters have received paired initialization.

        Args:
            readout_mode: single, concat, mixture or capacity-control duplicate.
            readouts: One, two or three simultaneous readings; no automatic escalation.
            scale_multipliers: Positive initial time multipliers, one per reading when >1.
            use_gradient: False supplies zero Z to the unchanged nominal gradient MLP columns.

        Raises:
            ValueError: Incompatible read count/mode or missing nonpositive time multipliers.
        """
        mode = DiffusionReadout(readout_mode)
        if readouts not in {1, 2, 3} or (mode is DiffusionReadout.SINGLE) != (readouts == 1):
            raise ValueError("single requires one readout; other modes require two or three")
        if readouts > 1 and (len(scale_multipliers) != readouts or min(scale_multipliers) <= 0):
            raise ValueError("provide one positive time multiplier per simultaneous reading")
        self.readout_mode, self.use_gradient = mode, use_gradient
        if readouts == 1:
            return
        # Existing raw_times remain the first reading; extra readings use independent parameters.
        # Scale controls are declared in TIME units (Å²), not interpreted as lengths.

        times = (
            self.diffusion_times.detach()[None]
            * self.raw_times.new_tensor(scale_multipliers)[:, None]
        )
        with torch.no_grad():
            self.raw_times.copy_(times[0] + torch.log(-torch.expm1(-times[0])))
        self.extra_raw_times = nn.Parameter(times[1:] + torch.log(-torch.expm1(-times[1:])))
        if mode is DiffusionReadout.MIXTURE:
            self.mixture_logits = nn.Parameter(torch.zeros_like(times))
        else:
            old = self.mlp[0]
            assert isinstance(old, nn.Linear)
            width = old.out_features
            replacement = nn.Linear((readouts + 2) * width, width)
            with torch.no_grad():
                replacement.weight.zero_()
                replacement.weight[:, : 2 * width].copy_(old.weight[:, : 2 * width])
                replacement.weight[:, -width:].copy_(old.weight[:, 2 * width :])
                replacement.bias.copy_(old.bias)
            self.mlp[0] = replacement

    @property
    def diffusion_times(self) -> Tensor:
        """Return strictly positive learned channel times in physical Å².

        Returns:
            ``float [H]`` values ``softplus(raw_time)+1e-8``.
        """
        return F.softplus(self.raw_times) + 1.0e-8

    def forward(
        self,
        features: Tensor,
        mass: Tensor,
        eigenvalues: Tensor,
        eigenvectors: Tensor,
        gradient: Tensor,
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
        input_dtype = features.dtype
        values = features.float()
        mass32 = mass.float()
        eigenvalues32 = eigenvalues.float()
        eigenvectors32 = eigenvectors.float()

        coefficients = eigenvectors32.T @ (mass32[:, None] * values)
        times = self.diffusion_times.float()[None]
        if self.extra_raw_times is not None:
            times = torch.cat((times, F.softplus(self.extra_raw_times.float()) + 1e-8))
        if self.readout_mode is DiffusionReadout.DUPLICATE:
            times = times[:1].expand_as(times)
        readings = [
            eigenvectors32 @ (torch.exp(-eigenvalues32[:, None] * time[None]) * coefficients)
            for time in times
        ]
        if self.mixture_logits is not None:
            diffused = [(torch.stack(readings) * self.mixture_logits.softmax(0)[:, None]).sum(0)]
        else:
            diffused = readings

        # Stacking gives [Gx; Gy] X = [Gx X; Gy X] in one sparse launch. The same bias-free
        # gradient_mixing map acts independently on each row, so it is also exactly equivalent to
        # applying the map separately to gx and gy before the invariant elementwise product.

        point_count = len(values)
        invariant_gradient = torch.zeros_like(values)
        if self.use_gradient:
            gradients = self.sparse_multiply(gradient, values)
            grad_x = gradients[:point_count]
            grad_y = gradients[point_count:]
            mixed_gradients = self.gradient_mixing(gradients)
            mixed_x = mixed_gradients[:point_count]
            mixed_y = mixed_gradients[point_count:]
            invariant_gradient = torch.tanh(grad_x * mixed_x + grad_y * mixed_y)

        update = self.mlp(torch.cat((values, *diffused, invariant_gradient), dim=1))
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
