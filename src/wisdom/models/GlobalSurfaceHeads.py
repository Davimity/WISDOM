"""Shared-backbone global classifier and optional context-conditioned localizer."""

import torch

from torch import Tensor, nn
from lambdaforge.nn import Scatter
from wisdom.models.HeadType import HeadType
from wisdom.models.HeadType import GlobalReduction, HeadImplementation


class GlobalSurfaceHeads(nn.Module):
    """Implement D1, D2, and D2b without changing the WISDOM backbone or pooling."""

    def __init__(
        self,
        hidden_dim        : int,
        head_type         : HeadType | str = HeadType.DUAL,
        context_dim       : int = 16,
        dropout           : float = 0.0,
        detach_context    : bool = True,
        implementation    : str = "controlled",
        global_reduction  : str = "area_mean",
        film_modulation_scale: float = 1.0,
    ) -> None:
        """Construct the direct global head and one controlled local conditioning mechanism.

        Args:
            hidden_dim: Actual surface evidence width D (H, K or H+K); parameter name retained.
            head_type: ``dual`` (D1), ``global_context`` (D2), or ``film`` (D2b).
            context_dim: Small global context bottleneck width.
            dropout: Direct-head dropout probability.
            detach_context: Stop local-loss gradients through the global context path.
                Context projection and pointwise backbone paths still learn.
            implementation: controlled constructs only participating modules; legacy recreates
                the historical complete state_dict and constructor RNG consumption exactly.
            global_reduction: area_mean, unweighted mean, or maximum of Z within each protein.
            film_modulation_scale: Positive multiplier on FiLM gamma/beta corrections.

        Raises:
            ValueError: If dimensions/dropout are invalid or ``single`` is requested.
        """
        super().__init__()
        self.head_type = HeadType(head_type)
        self.implementation = HeadImplementation(implementation)
        self.global_reduction = GlobalReduction(global_reduction)
        self.film_modulation_scale = float(film_modulation_scale)
        if self.head_type is HeadType.SINGLE:
            raise ValueError("GlobalSurfaceHeads is not needed for the single-head control")
        if hidden_dim < 1 or context_dim < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("global/surface head dimensions or dropout are invalid")
        if not 0.0 < film_modulation_scale < float("inf"):
            raise ValueError("film_modulation_scale must be finite and positive")
        if self.head_type is not HeadType.FILM and film_modulation_scale != 1.0:
            raise ValueError("film_modulation_scale is active only for film")
        if self.implementation is HeadImplementation.LEGACY and (
            self.global_reduction is not GlobalReduction.AREA_MEAN or film_modulation_scale != 1.0
        ):
            raise ValueError("legacy heads require historical area_mean and unit FiLM scale")

        self.global_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        # Historical construction order is part of old checkpoints and seeded experiments.
        # Controlled heads allocate only their own branch; legacy is an explicit reader path.

        legacy = self.implementation is HeadImplementation.LEGACY
        if legacy or self.head_type is not HeadType.DUAL:
            self.context_projection = nn.Sequential(nn.Linear(hidden_dim, context_dim), nn.SiLU())
        if legacy or self.head_type is HeadType.GLOBAL_CONTEXT:
            self.context_local_head = nn.Linear(hidden_dim + context_dim, 1)
        if legacy or self.head_type is HeadType.FILM:
            self.film_head = nn.Linear(context_dim, 2 * hidden_dim)
            self.film_local_head = nn.Linear(hidden_dim, 1)
            nn.init.zeros_(self.film_head.weight)
            nn.init.zeros_(self.film_head.bias)

        self.hidden_dim     = hidden_dim
        self.context_dim    = context_dim
        self.detach_context = detach_context

    def forward(
        self,
        surface_evidence_features: Tensor,
        area_weights      : Tensor,
        owners           : Tensor,
        protein_count    : int,
    ) -> dict[str, Tensor]:
        """Predict direct protein logits and, for D2/D2b, conditioned local logits.

        The direct classifier consumes an area-weighted mean, point mean or featurewise max
        within each protein. Context heads concatenate its projected embedding to each point;
        FiLM applies ``(1+scale*gamma_i)*S_ip+scale*beta_i`` with zero-start gamma and beta.
        detach_context cuts only the reduction-to-projection gradient: the projection and
        pointwise backbone route remain differentiable. Dual retains the original local head.

        Args:
            surface_evidence_features: Actual Z point states ``[M,D]``: H, X or [H,X].
            area_weights: Positive represented-area weights ``[M]``.
            owners: Point-to-protein owner IDs ``[M]``.
            protein_count: Number of proteins ``B``.

        Returns:
            ``direct_logits[B]``, ``global_context[B,D]``, and optional
            ``conditioned_surface_logits[M]``.
        """
        if self.implementation is HeadImplementation.LEGACY:
            area_sum = Scatter.sum(area_weights, owners, protein_count)
            area = area_weights / area_sum[owners].clamp_min(
                torch.finfo(surface_evidence_features.dtype).eps
            )
            global_context = Scatter.sum(surface_evidence_features * area[:, None],
                                         owners, protein_count)
        else:
            # Accumulate reductions in FP32 under AMP; never mix areas from different proteins.

            states = surface_evidence_features.float()
            if self.global_reduction is GlobalReduction.MAX:
                global_context = Scatter.maximum(states, owners, protein_count)
            else:
                weights = (
                    area_weights.float() if self.global_reduction is GlobalReduction.AREA_MEAN
                    else states.new_ones(len(states))
                )
                totals = Scatter.sum(weights, owners, protein_count)
                weights = weights / totals[owners].clamp_min(torch.finfo(torch.float32).eps)
                global_context = Scatter.sum(states * weights[:, None], owners, protein_count)
        direct_logits = self.global_head(global_context).squeeze(-1)
        output = {
            "direct_logits": direct_logits,
            "global_context": global_context,
        }
        if self.head_type is HeadType.DUAL:
            return output

        context_source = global_context.detach() if self.detach_context else global_context
        context = self.context_projection(context_source)
        point_context = context[owners]
        if self.head_type is HeadType.GLOBAL_CONTEXT:
            conditioned = self.context_local_head(
                torch.cat((surface_evidence_features, point_context), dim=1)
            ).squeeze(-1)
        else:
            gamma, beta = self.film_head(point_context).chunk(2, dim=1)
            scale = self.film_modulation_scale
            conditioned_state = (1.0 + scale * gamma) * surface_evidence_features + scale * beta
            conditioned = self.film_local_head(conditioned_state).squeeze(-1)
        output["conditioned_surface_logits"] = conditioned
        return output
