"""Independent V11 summary, value, head and support interventions on historical scores."""

import torch

from torch import Tensor, nn
from lambdaforge.nn import Entmax15
from wisdom.models.pooling.BoundedScalar import BoundedScalar
from wisdom.models.communication.TransferMessage import TransferMessage
from wisdom.models.communication.TransferSummary import TransferSummary


class TransferExperiment(nn.Module):
    """Keep changes explicit and chunk-bounded; empty rows always carry no atom information."""

    def __init__(
        self,
        hidden_dim: int,
        point_dim: int = 0,
        summary: str = "mean",
        message: str = "atom",
        rank: int = 16,
        heads: int = 1,
        shared_head_weights: bool = False,
        channel_attention: bool = False,
        normalization: str = "softmax",
        normalization_radius: float | None = None,
        learned_radius: bool = False,
        radius_bounds: tuple[float, float] = (1.0, 12.0),
        radius_initial: float = 6.0,
        radius_temperature: float = 0.5,
        epsilon: float = 1e-8,
        summary_repeats: int = 2,
        support: str = "effective",
        active_radius: float | None = None,
        active_k: int | None = None,
    ) -> None:
        """Bind one-factor interventions; combining them is an explicit later integration.

        Args:
            hidden_dim: Atomic/output width D; heads must divide it.
            point_dim: Existing gated curvature width, required only by point_pair values.
            summary: mean/mean_std/mean_max/mean_std_max/mean_support/duplicate_mean.
            message: atom baseline, pair MLP, point_pair MLP or factored low_rank correction.
            rank: Positive bottleneck width of the low-rank value correction.
            heads: Scalar attention heads with D/heads channels each.
            shared_head_weights: Capacity control: all heads use the original scalar weights.
            channel_attention: One separately normalized weight per output channel; exploratory.
            normalization: softmax or native LF entmax15; no learned exponent.
            normalization_radius: Fixed Å scorer scale independent of the active support radius.
            learned_radius: Add stable log-sigmoid radial support inside the fixed cached mask.
            radius_bounds: Global effective-radius bounds in Å, upper <= consumer cache radius.
            radius_initial: Strictly interior initial effective radius, in Å.
            radius_temperature: Positive soft-transition width in Å, not a hard cutoff.
            epsilon: Positive FP32 root/denominator floor.
            summary_repeats: Mean copies in the matched projection-capacity control, default two.
            support: effective uses 1/sum(alpha²); count uses log1p(valid count).
            active_radius: Optional hard Å sub-support within the declared cached candidate set.
            active_k: Optional nearest-candidate prefix within that same stored support.

        Raises:
            ValueError: Incompatible heads, message geometry or unsupported normalization/support.
        """
        super().__init__()
        self.summary = TransferSummary(summary)
        self.message = TransferMessage(message)
        self.heads = hidden_dim if channel_attention else heads
        if hidden_dim % self.heads or normalization not in {"softmax", "entmax15"}:
            raise ValueError("transfer heads must divide D; normalization is softmax/entmax15")
        if support not in {"effective", "count"} or epsilon <= 0 or radius_temperature <= 0:
            raise ValueError("transfer support/floors/temperature are invalid")
        if self.message is TransferMessage.POINT_PAIR and point_dim < 1:
            raise ValueError("point_pair requires existing gated point geometry")
        if self.heads > 1 and self.summary is not TransferSummary.MEAN:
            raise ValueError("isolate multi-head and summary experiments before combining them")
        if self.message is TransferMessage.LOW_RANK and self.summary is not TransferSummary.MEAN:
            raise ValueError("low_rank values require mean summary; isolate moment experiments")
        self.normalization = normalization
        self.active_radius, self.active_k = active_radius, active_k
        self.normalization_radius = normalization_radius
        self.shared_head_weights = shared_head_weights
        self.epsilon, self.temperature, self.support = epsilon, radius_temperature, support
        self.radius = (
            BoundedScalar(radius_initial, *radius_bounds, logarithmic=False)
            if learned_radius
            else None
        )
        self.radius_bounds = radius_bounds
        self.summary_repeats = summary_repeats
        self.entmax = Entmax15(dim=1)
        self.value_correction: nn.Module | None = None
        self.atom_low_rank: nn.Module | None = None
        self.pair_low_rank: nn.Module | None = None
        if self.message in {TransferMessage.PAIR, TransferMessage.POINT_PAIR}:
            self.value_correction = nn.Sequential(
                nn.Linear(
                    hidden_dim
                    + 3
                    + (point_dim if self.message is TransferMessage.POINT_PAIR else 0),
                    hidden_dim,
                ),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            final = self.value_correction[-1]
            assert isinstance(final, nn.Linear)
            nn.init.zeros_(final.weight)
            nn.init.zeros_(final.bias)
        elif self.message is TransferMessage.LOW_RANK:
            self.atom_low_rank = nn.Linear(hidden_dim, rank, bias=False)
            self.pair_low_rank = nn.Sequential(nn.Linear(3, rank), nn.SiLU())
            self.value_correction = nn.Linear(rank, hidden_dim, bias=False)
            nn.init.zeros_(self.value_correction.weight)
        self.head_score = (
            nn.Linear(hidden_dim + 3, self.heads, bias=False) if self.heads > 1 else None
        )
        self.value_projection = (
            nn.Linear(hidden_dim, hidden_dim, bias=False) if self.heads > 1 else None
        )
        self.head_projection = (
            nn.Linear(hidden_dim, hidden_dim, bias=False) if self.heads > 1 else None
        )
        if self.head_score is not None:
            assert self.value_projection is not None and self.head_projection is not None
            nn.init.zeros_(self.head_score.weight)
            nn.init.eye_(self.value_projection.weight)
            nn.init.eye_(self.head_projection.weight)
        sizes = {
            TransferSummary.MEAN: hidden_dim,
            TransferSummary.MEAN_STD: 2 * hidden_dim,
            TransferSummary.MEAN_MAX: 2 * hidden_dim,
            TransferSummary.MEAN_STD_MAX: 3 * hidden_dim,
            TransferSummary.MEAN_SUPPORT: hidden_dim + 1,
            TransferSummary.DUPLICATE_MEAN: summary_repeats * hidden_dim,
        }
        self.summary_projection = (
            nn.Linear(sizes[self.summary], hidden_dim)
            if (self.summary is not TransferSummary.MEAN)
            else None
        )
        if self.summary_projection is not None:
            nn.init.zeros_(self.summary_projection.weight)
            nn.init.zeros_(self.summary_projection.bias)
            with torch.no_grad():
                self.summary_projection.weight[:, :hidden_dim].copy_(torch.eye(hidden_dim))
        self.diagnostics: dict[str, Tensor] = {}

    def forward(
        self,
        scores: Tensor,
        gathered: Tensor,
        geometry: Tensor,
        distances: Tensor,
        valid: Tensor,
        point_features: Tensor | None = None,
        low_rank_atoms: Tensor | None = None,
        content_gate: Tensor | None = None,
    ) -> Tensor:
        """Change one chunk's values/weights/summary with stable FP32 moments and empty masking.

        Args:
            scores: Historical scalar candidate scores [C,J], before masked normalization.
            gathered: Atom embeddings [C,J,D] in the exact stored candidate order.
            geometry: Invariant normalized [distance,normal_offset,tangential] per pair [C,J,3].
            distances: Unnormalized physical center distances [C,J], in Å.
            valid: Cached candidate validity after the explicit hard consumer radius [C,J].
            point_features: Existing gated surface geometry [C,G], only for point_pair messages.
            low_rank_atoms: Optional preprojected gathered [C,J,rank] values for B3.
            content_gate: Existing scalar semantic content gate for additional head scores.

        Returns:
            Context [C,D]; no invented neighbor rescue or nonzero bias on unsupported rows.
        """
        dtype = gathered.dtype
        if self.active_radius is not None:
            valid = valid & (distances <= self.active_radius)
        if self.active_k is not None:
            prefix = torch.arange(valid.shape[1], device=valid.device) < self.active_k
            valid = valid & prefix[None]
        values = gathered.float()
        logits = scores.float().unsqueeze(-1)
        if self.head_score is not None and not self.shared_head_weights:
            content = gathered if content_gate is None else gathered * content_gate
            logits = logits + self.head_score(torch.cat((content, geometry), dim=-1)).float()
        elif self.heads > 1:
            logits = logits.expand(-1, -1, self.heads)
        if self.radius is not None:
            radial_gate = torch.nn.functional.logsigmoid(
                (self.radius() - distances.float()) / self.temperature
            )
            logits = logits + radial_gate.unsqueeze(-1)
        logits = logits.masked_fill(~valid.unsqueeze(-1), -1e4)
        weights = self.entmax(logits) if self.normalization == "entmax15" else logits.softmax(1)
        weights = weights.masked_fill(~valid.unsqueeze(-1), 0.0)
        weights = weights / weights.sum(1, keepdim=True).clamp_min(self.epsilon)
        if self.message in {TransferMessage.PAIR, TransferMessage.POINT_PAIR}:
            assert self.value_correction is not None
            parts = [gathered, geometry]
            if self.message is TransferMessage.POINT_PAIR:
                assert point_features is not None
                parts.append(point_features[:, None].expand(-1, gathered.shape[1], -1))
            values = values + self.value_correction(torch.cat(parts, dim=-1)).float()
        if self.heads > 1:
            assert self.value_projection is not None and self.head_projection is not None
            values = (
                self.value_projection(values.to(dtype))
                .float()
                .reshape(len(values), values.shape[1], self.heads, -1)
            )
            output = (weights.unsqueeze(-1) * values).sum(1).flatten(1)
            output = self.head_projection(output.to(dtype)).float()
        else:
            alpha = weights[..., 0]
            mean = torch.bmm(alpha.unsqueeze(1), values).squeeze(1)
            if self.message is TransferMessage.LOW_RANK:
                assert self.atom_low_rank is not None and self.pair_low_rank is not None
                assert self.value_correction is not None
                low = low_rank_atoms if low_rank_atoms is not None else self.atom_low_rank(gathered)
                modulated = low * self.pair_low_rank(geometry)
                mean = (
                    mean
                    + self.value_correction(
                        torch.bmm(alpha.to(modulated.dtype).unsqueeze(1), modulated).squeeze(1)
                    ).float()
                )
            parts = [mean]
            if self.summary in {TransferSummary.MEAN_STD, TransferSummary.MEAN_STD_MAX}:
                # Center before squaring, rather than E[h²]-E[h]² cancellation near equal values.

                variance = (alpha[..., None] * (values - mean[:, None]).square()).sum(1)
                deviation = torch.where(variance > 0, variance.clamp_min(self.epsilon).sqrt(),
                                        torch.zeros_like(variance))
                parts.append(deviation)
            if self.summary in {TransferSummary.MEAN_MAX, TransferSummary.MEAN_STD_MAX}:
                parts.append(
                    values.masked_fill(~valid[..., None], -torch.inf).amax(1).nan_to_num(neginf=0.0)
                )
            if self.summary is TransferSummary.MEAN_SUPPORT:
                support = (
                    valid.sum(1).float().log1p()
                    if self.support == "count"
                    else alpha.square().sum(1).clamp_min(self.epsilon).reciprocal()
                )
                parts.append(support[:, None])
            if self.summary is TransferSummary.DUPLICATE_MEAN:
                parts = [mean] * self.summary_repeats
            output = (
                self.summary_projection(torch.cat(parts, dim=1))
                if self.summary_projection is not None
                else mean
            )
        # The context above retains its graph; telemetry must not retain another autograd branch.

        weights = weights.detach()
        supported = valid.any(1)
        entropy = -(weights * weights.clamp_min(self.epsilon).log()).sum(1).mean(1)
        mean_weights = weights.mean(2, keepdim=True)
        divergence = (weights * (weights.clamp_min(self.epsilon).log()
            - mean_weights.clamp_min(self.epsilon).log())).sum(1).mean(1)
        self.diagnostics = {
            "rows": scores.new_tensor(len(scores), dtype=torch.float32).detach(),
            "supported_rows": supported.sum().detach(),
            "valid_pairs": valid.sum().detach(),
            "empty_rows": (~valid.any(1)).sum().detach(),
            "entropy_sum": (entropy * supported).sum().detach(),
            "head_divergence_sum": (divergence * supported).sum().detach(),
            "attended_distance_sum": (weights * distances[..., None]).sum(1).mean(1).sum().detach(),
            "effective_support_sum": (
                weights.square().sum(1).clamp_min(self.epsilon).reciprocal()
                * valid.any(1, keepdim=True)
            )
            .mean(1).sum()
            .detach(),
        }
        return (output * valid.any(1, keepdim=True)).to(dtype)
