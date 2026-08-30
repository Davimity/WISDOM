"""Shared DiffusionNet encoder for independently packed protein surfaces."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import Tensor, nn

from wisdom.models.DiffusionBlock import DiffusionBlock


class DiffusionSurfaceEncoder(nn.Module):
    """Apply shared intrinsic blocks separately to every protein operator pack."""

    def __init__(
        self,
        hidden_dim: int,
        layers    : int = 2,
        dropout   : float = 0.0,
    ) -> None:
        """Build a compact DiffusionNet with physical multiscale time initialization.

        The implementation follows Nicholas Sharp et al., *DiffusionNet: Discretization Agnostic
        Learning on Surfaces* (ACM TOG 2022), and the authors' MIT-licensed reference repository.
        WISDOM keeps only the scalar spectral diffusion and invariant gradient-product mechanisms
        needed by its precomputed molecular point clouds.

        Args:
            hidden_dim: Surface feature width ``H``.
            layers: Number of residual DiffusionNet blocks.
            dropout: Pointwise block-MLP dropout in ``[0,1)``.

        Raises:
            ValueError: If width/layer count is non-positive or dropout is invalid.
        """
        super().__init__()
        if hidden_dim < 1 or layers < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("DiffusionSurfaceEncoder dimensions or dropout are invalid")

        self.blocks = nn.ModuleList(
            DiffusionBlock(
                hidden_dim,
                dropout=dropout,
                initial_time=float(2**index),
            )
            for index in range(layers)
        )
        self.hidden_dim = hidden_dim

    def forward(
        self,
        features : Tensor,
        operators: Sequence[Mapping[str, Tensor]],
        surface_ptr: Tensor,
    ) -> Tensor:
        """Encode disjoint surfaces without constructing a giant block-diagonal operator.

        Args:
            features: Concatenated surface features ``float [M_total,H]``.
            operators: Ordered per-protein mass, eigenpair, and sparse-gradient mappings.
            surface_ptr: Prefix boundaries ``long [B+1]`` into ``features``.

        Returns:
            Concatenated embeddings ``float [M_total,H]`` in unchanged point order.

        Raises:
            ValueError: If feature width, pointer boundaries, or operator dimensions disagree.
        """
        if features.ndim != 2 or features.shape[1] != self.hidden_dim:
            raise ValueError("diffusion input must have shape [M,H]")
        if surface_ptr.ndim != 1 or len(surface_ptr) != len(operators) + 1:
            raise ValueError("surface_ptr must have shape [B+1] aligned with operator packs")
        if int(surface_ptr[0]) != 0 or int(surface_ptr[-1]) != len(features):
            raise ValueError("surface_ptr boundaries disagree with concatenated features")

        outputs: list[Tensor] = []
        for protein_index, operator in enumerate(operators):
            start = int(surface_ptr[protein_index])
            stop  = int(surface_ptr[protein_index + 1])
            local = features[start:stop]
            gradient_x, gradient_y = self.sparse_gradients(operator, stop - start)

            for block in self.blocks:
                local = block(
                    local,
                    operator["mass"],
                    operator["eigenvalues"],
                    operator["eigenvectors"],
                    gradient_x,
                    gradient_y,
                )
            outputs.append(local)
        return torch.cat(outputs, dim=0)

    @staticmethod
    def diffuse_scalar(
        values    : Tensor,
        operators : Sequence[Mapping[str, Tensor]],
        surface_ptr: Tensor,
        time      : float,
    ) -> Tensor:
        """Diffuse one scalar field at a fixed physical scale for regional pooling.

        Args:
            values: Concatenated scalar values ``float [M_total]``.
            operators: Ordered per-protein mass and low eigenpairs.
            surface_ptr: Prefix point boundaries ``long [B+1]``.
            time: Positive heat time in Å², usually ``regional_scale**2``.

        Returns:
            Spectrally smoothed values ``float [M_total]`` in unchanged point order.

        Raises:
            ValueError: If ``time`` is not positive.
        """
        if time <= 0.0:
            raise ValueError("regional diffusion time must be positive")

        outputs: list[Tensor] = []
        for protein_index, operator in enumerate(operators):
            start = int(surface_ptr[protein_index])
            stop  = int(surface_ptr[protein_index + 1])
            local = values[start:stop].float()
            mass  = operator["mass"].float()
            phi   = operator["eigenvectors"].float()
            lambdas = operator["eigenvalues"].float()
            coefficients = phi.T @ (mass * local)
            outputs.append(phi @ (torch.exp(-lambdas * time) * coefficients))
        return torch.cat(outputs).to(values.dtype)

    @staticmethod
    def sparse_gradients(
        operator  : Mapping[str, Tensor],
        point_count: int,
    ) -> tuple[Tensor, Tensor]:
        """Materialize two coalesced sparse matrices from one compact COO operator pack.

        Args:
            operator: Mapping containing shared ``gradient_index`` and x/y values.
            point_count: Local surface size defining both sparse dimensions.

        Returns:
            Coalesced tangent derivative matrices ``(Gx,Gy)`` with shape ``[M,M]``.
        """
        index = operator["gradient_index"]
        size  = (point_count, point_count)
        gradient_x = torch.sparse_coo_tensor(
            index,
            operator["gradient_x"],
            size,
            check_invariants=False,
        ).coalesce()
        gradient_y = torch.sparse_coo_tensor(
            index,
            operator["gradient_y"],
            size,
            check_invariants=False,
        ).coalesce()
        return gradient_x, gradient_y
