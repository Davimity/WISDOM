"""Shared-backbone global classifier and optional context-conditioned localizer."""

import torch
from lambdaforge.nn import Scatter
from torch import Tensor, nn

from wisdom.models.HeadType import HeadType


class GlobalSurfaceHeads(nn.Module):
    """Implement D1, D2, and D2b without changing the WISDOM backbone or pooling."""

    def __init__(
        self,
        hidden_dim        : int,
        head_type         : HeadType | str = HeadType.DUAL,
        context_dim       : int = 16,
        dropout           : float = 0.0,
        detach_context    : bool = True,
    ) -> None:
        """Construct the direct global head and one controlled local conditioning mechanism.

        Args:
            hidden_dim: Shared surface embedding width ``H``.
            head_type: ``dual`` (D1), ``global_context`` (D2), or ``film`` (D2b).
            context_dim: Small global context bottleneck width.
            dropout: Direct-head dropout probability.
            detach_context: Stop local-loss gradients through the global context path.

        Raises:
            ValueError: If dimensions/dropout are invalid or ``single`` is requested.
        """
        super().__init__()
        self.head_type = HeadType(head_type)
        if self.head_type is HeadType.SINGLE:
            raise ValueError("GlobalSurfaceHeads is not needed for the single-head control")
        if hidden_dim < 1 or context_dim < 1 or not 0.0 <= dropout < 1.0:
            raise ValueError("global/surface head dimensions or dropout are invalid")

        self.global_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.context_projection = nn.Sequential(
            nn.Linear(hidden_dim, context_dim),
            nn.SiLU(),
        )
        self.context_local_head = nn.Linear(hidden_dim + context_dim, 1)
        self.film_head = nn.Linear(context_dim, 2 * hidden_dim)
        self.film_local_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.film_head.weight)
        nn.init.zeros_(self.film_head.bias)

        self.hidden_dim     = hidden_dim
        self.context_dim    = context_dim
        self.detach_context = detach_context

    def forward(
        self,
        surface_embeddings: Tensor,
        area_weights      : Tensor,
        owners           : Tensor,
        protein_count    : int,
    ) -> dict[str, Tensor]:
        """Predict direct protein logits and, for D2/D2b, conditioned local logits.

        Surface context is the represented-area-weighted mean
        ``g_i=sum_p a_ip*S_ip``. The direct head classifies ``g_i``. D2 concatenates a detached
        bottleneck to each point; D2b applies zero-initialized FiLM
        ``(1+gamma_i)*S_ip+beta_i``. D1 returns only the direct logit so the caller retains its
        established local head unchanged.

        Args:
            surface_embeddings: Shared surface states ``[M,H]``.
            area_weights: Positive represented-area weights ``[M]``.
            owners: Point-to-protein owner IDs ``[M]``.
            protein_count: Number of proteins ``B``.

        Returns:
            ``direct_logits[B]``, ``global_context[B,H]``, and optional
            ``conditioned_surface_logits[M]``.
        """
        area_sum = Scatter.sum(area_weights, owners, protein_count)
        area     = area_weights / area_sum[owners].clamp_min(
            torch.finfo(surface_embeddings.dtype).eps
        )
        global_context = Scatter.sum(
            surface_embeddings * area[:, None],
            owners,
            protein_count,
        )
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
                torch.cat((surface_embeddings, point_context), dim=1)
            ).squeeze(-1)
        else:
            gamma, beta = self.film_head(point_context).chunk(2, dim=1)
            conditioned_state = (1.0 + gamma) * surface_embeddings + beta
            conditioned = self.film_local_head(conditioned_state).squeeze(-1)
        output["conditioned_surface_logits"] = conditioned
        return output
