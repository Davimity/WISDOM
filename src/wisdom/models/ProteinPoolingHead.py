"""Measure-aware multiple-instance pooling with interpretable bounded parameters."""

import math
import torch

from torch import Tensor, nn
from itertools import pairwise
from lambdaforge.nn import Scatter
from collections.abc import Mapping, Sequence
from wisdom.models.PoolingType import PoolingType
from wisdom.models.BoundedScalar import BoundedScalar
from wisdom.models.PoolingAreaMode import PoolingAreaMode
from wisdom.models.AttentionVariant import AttentionVariant
from wisdom.models.PoolingParameterMode import PoolingParameterMode
from wisdom.models.DiffusionSurfaceEncoder import DiffusionSurfaceEncoder
from lambdaforge.nn.pooling import SparseAttentionPooling, SparseMaxPooling


class ProteinPoolingHead(nn.Module):
    """Aggregate local evidence without changing the surface predictor or its supervision."""

    runtime_beta: Tensor
    scale_logits: nn.Parameter

    def __init__(
        self,
        hidden_dim                   : int,
        pooling_type                 : PoolingType | str,
        dropout                      : float,
        topk_fraction                : float = 0.05,
        attention_hidden_dim         : int = 32,
        regional_diffusion_scale     : float = 2.5,
        log_sum_exp_beta             : float = 5.0,
        pooling_area_mode            : str = "legacy",
        attention_variant            : str = "simple",
        regional_scale_mode          : str = "fixed",
        regional_diffusion_scale_init: float = 1.5,
        log_sum_exp_mode             : str = "fixed",
        log_sum_exp_beta_init        : float = 5.0,
        autopool_alpha               : float = 1.0,
        autopool_alpha_mode          : str = "fixed",
        autopool_alpha_init          : float = 1.0,
        gem_power                    : float = 1.0,
        gem_power_mode               : str = "fixed",
        gem_power_init               : float = 4.0,
        max_mean_lambda              : float = 0.5,
        max_mean_lambda_mode         : str = "fixed",
        max_mean_lambda_init         : float = 0.5,
        regional_diffusion_scale_bounds: Sequence[float] = (0.05, 12.0),
        log_sum_exp_beta_bounds        : Sequence[float] = (0.25, 200.0),
        autopool_alpha_bounds          : Sequence[float] = (0.0, 50.0),
        gem_power_bounds               : Sequence[float] = (1.0, 32.0),
    ) -> None:
        """Construct only the parameters used by the selected family.

        Args:
            hidden_dim: Point embedding width H.
            pooling_type: Family in PoolingType; no new encoder is introduced.
            dropout: Attention scorer dropout probability, shared with the frozen backbone.
            topk_fraction: Point or represented-area fraction in (0, 1].
            attention_hidden_dim: Width of the simple or gated attention scorer.
            regional_diffusion_scale: Fixed heat length in angstroms, including exact identity 0.
            log_sum_exp_beta: Fixed positive inverse temperature on local logits.
            pooling_area_mode: point or area; legacy retains area-mean and point other families.
            attention_variant: simple tanh scorer or gated tanh-times-sigmoid scorer.
            regional_scale_mode: fixed or learned heat length.
            regional_diffusion_scale_init: Learned length initialization inside its bounds, in Å.
            log_sum_exp_mode: fixed, learned, or externally scheduled curriculum.
            log_sum_exp_beta_init: Learned beta initialization strictly inside its bounds.
            autopool_alpha: Fixed non-negative sharpness on probabilities.
            autopool_alpha_mode: fixed or learned AutoPool sharpness.
            autopool_alpha_init: Learned initialization strictly inside its bounds.
            gem_power: Fixed probability power >= 1.
            gem_power_mode: fixed or learned probability power.
            gem_power_init: Learned initialization strictly inside its bounds.
            max_mean_lambda: Fixed MAX mixing coefficient in [0, 1].
            max_mean_lambda_mode: fixed or learned convex mixing coefficient.
            max_mean_lambda_init: Learned initialization inside (0, 1).
            regional_diffusion_scale_bounds: Positive lower/upper learned heat lengths in Å;
                default (0.05, 12) preserves historical checkpoint interpretation.
            log_sum_exp_beta_bounds: Positive lower/upper learned beta; default (0.25, 200).
            autopool_alpha_bounds: Nonnegative lower/upper learned alpha; default (0, 50).
            gem_power_bounds: Lower/upper learned power, lower at least one; default (1, 32).
                Bounds affect learned modes only; fixed and curriculum values are unchanged.

        Raises:
            ValueError: If a family/mode is unknown or a selected scalar violates its domain.
        """
        super().__init__()
        self.pooling_type = PoolingType(pooling_type)
        self.area_mode    = PoolingAreaMode(pooling_area_mode)
        self.variant      = AttentionVariant(attention_variant)
        self.beta_mode    = PoolingParameterMode(log_sum_exp_mode)
        self.topk_fraction = float(topk_fraction)
        self.regional_diffusion_scale = float(regional_diffusion_scale)
        self.sparse_max = SparseMaxPooling()

        # Keep the original simple-attention module and its state_dict names intact.

        self.attention = (
            SparseAttentionPooling(hidden_dim, attention_hidden_dim, dropout=dropout)
            if (
                self.pooling_type is PoolingType.ATTENTION
                and self.variant is AttentionVariant.SIMPLE
            )
            else None
        )
        if self.pooling_type is PoolingType.ATTENTION and self.variant is AttentionVariant.GATED:
            self.attention_value = nn.Linear(hidden_dim, attention_hidden_dim)
            self.attention_gate  = nn.Linear(hidden_dim, attention_hidden_dim)
            self.attention_drop  = nn.Dropout(dropout)
            self.attention_score = nn.Linear(attention_hidden_dim, 1)

        # One bounded scalar per applicable learned family; all join the ordinary optimizer.

        self.learned_scalar: BoundedScalar | None = None
        specifications = {
            PoolingType.LOG_SUM_EXP: (
                log_sum_exp_mode, log_sum_exp_beta_init, log_sum_exp_beta_bounds, True,
            ),
            PoolingType.LOCAL_MEAN_MAX: (
                regional_scale_mode, regional_diffusion_scale_init,
                regional_diffusion_scale_bounds, True,
            ),
            PoolingType.AUTOPOOL: (
                autopool_alpha_mode, autopool_alpha_init, autopool_alpha_bounds, False,
            ),
            PoolingType.GEM: (gem_power_mode, gem_power_init, gem_power_bounds, True),
            PoolingType.MAX_MEAN: (
                max_mean_lambda_mode, max_mean_lambda_init, (0.0, 1.0), False,
            ),
        }
        if self.pooling_type in specifications:
            mode, initial, bounds, logarithmic = specifications[self.pooling_type]
            selected_mode = PoolingParameterMode(mode)
            if (
                selected_mode is PoolingParameterMode.CURRICULUM
                and self.pooling_type is not PoolingType.LOG_SUM_EXP
            ):
                raise ValueError("curriculum is available only for log_sum_exp")
            if selected_mode is PoolingParameterMode.LEARNED:
                lower, upper = bounds
                if self.pooling_type is PoolingType.AUTOPOOL and lower < 0:
                    raise ValueError("learned AutoPool bounds must remain non-negative")
                if self.pooling_type is PoolingType.GEM and lower < 1:
                    raise ValueError("learned GeM bounds must remain at least one")
                self.learned_scalar = BoundedScalar(initial, lower, upper, logarithmic)

        self.fixed_alpha  = float(autopool_alpha)
        self.fixed_power  = float(gem_power)
        self.fixed_lambda = float(max_mean_lambda)
        # Only curricula add checkpoint state: old fixed heads had no temperature buffer.

        self.register_buffer(
            "runtime_beta", torch.tensor(log_sum_exp_beta, dtype=torch.float32),
            persistent=self.beta_mode is PoolingParameterMode.CURRICULUM,
        )
        if self.pooling_type is PoolingType.MULTISCALE_REGIONAL_MAX:
            self.scale_logits = nn.Parameter(torch.zeros(5))
        self.scales = (0.0, 1.5, 3.0, 6.0, 12.0)

        if self.pooling_type is PoolingType.TOPK and not 0 < topk_fraction <= 1:
            raise ValueError("topk_fraction must be in (0, 1]")
        if self.pooling_type is PoolingType.LOG_SUM_EXP and log_sum_exp_beta <= 0:
            raise ValueError("log_sum_exp_beta must be positive")
        if self.pooling_type is PoolingType.LOCAL_MEAN_MAX and regional_diffusion_scale < 0:
            raise ValueError("regional_diffusion_scale must be non-negative")
        if self.pooling_type is PoolingType.AUTOPOOL and autopool_alpha < 0:
            raise ValueError("autopool_alpha must be non-negative")
        if self.pooling_type is PoolingType.GEM and gem_power < 1:
            raise ValueError("gem_power must be at least one")
        if self.pooling_type is PoolingType.MAX_MEAN and not 0 <= max_mean_lambda <= 1:
            raise ValueError("max_mean_lambda must lie in [0, 1]")

    def forward(
        self,
        logits       : Tensor,
        embeddings   : Tensor,
        area_weights : Tensor,
        owners       : Tensor,
        operators    : Sequence[Mapping[str, Tensor]],
        surface_ptr  : Tensor,
    ) -> dict[str, Tensor]:
        """Pool an ordered disjoint batch without mixing proteins.

        For a protein's points p, let l_p be a local logit, h_p its embedding, and m_p a
        normalized measure (1/N for point mode, A_p/sum(A) for area mode). MAX returns
        max(l); mean returns sum(m*l); attention returns sum(softmax(s+log(m))*l), with
        s=w^T tanh(Vh) or w^T[tanh(Vh)*sigmoid(Uh)]. Point Top-K averages ceil(f*N) logits;
        area Top-K integrates the sorted logits over exactly f of normalized area, dividing
        by f. Regional variants use heat time length^2 before MAX, optionally mixed across
        five fixed lengths with learned convex weights. LSE returns log(sum(m*exp(beta*l)))/beta.

        Probability variants use p=sigmoid(l): linear softmax returns sum(m*p^2)/sum(m*p),
        AutoPool returns sum(softmax(alpha*p+log(m))*p), and GeM returns (sum(m*p^r))^(1/r).
        Their global result is converted to a clipped logit. MAX-MEAN instead mixes logits
        as lambda*max(l)+(1-lambda)*sum(m*l). All sums and maxima stay within one protein.

        Args:
            logits: Local real-valued evidence [M], with differentiable FP32/BF16 values.
            embeddings: Local features [M,H], used only by attention.
            area_weights: Positive represented areas [M]; normalized within each protein.
            owners: Integer point-to-protein indices [M].
            operators: Per-protein mass-orthonormal diffusion eigenpairs.
            surface_ptr: Prefix boundaries [B+1], allowing O(M log M) segmented top-k sorting.

        Returns:
            Protein logits [B], plus attention_weights [M] only for attention. Reduction
            arithmetic uses FP32 (FP64 inputs retain FP64), preventing BF16 area truncation.
            Probability families convert their result to a finite logit for unchanged BCE.
        """
        count = len(surface_ptr) - 1
        if self.pooling_type is PoolingType.MAX:
            return {"logits": self.sparse_max(logits[:, None], owners, count)[:, 0]}

        # A measure sums to one per protein. Area is an explicit experiment, not a default prior.

        values = logits if logits.dtype == torch.float64 else logits.float()
        use_area = self.area_mode is PoolingAreaMode.AREA or (
            self.area_mode is PoolingAreaMode.LEGACY and self.pooling_type is PoolingType.MEAN
        )
        mass    = area_weights.to(values.dtype) if use_area else torch.ones_like(values)
        measure = mass / Scatter.sum(mass, owners, count)[owners]
        mean    = Scatter.sum(measure * values, owners, count)

        if self.pooling_type is PoolingType.MEAN:
            return {"logits": mean}
        if self.pooling_type is PoolingType.ATTENTION:
            if self.attention is not None:
                dtype  = next(self.attention.parameters()).dtype
                scores = self.attention.scorer(embeddings.to(dtype)).squeeze(-1)
            else:
                local  = embeddings.to(self.attention_value.weight.dtype)
                gated  = (
                    torch.tanh(self.attention_value(local))
                    * torch.sigmoid(self.attention_gate(local))
                )
                scores = self.attention_score(self.attention_drop(gated)).squeeze(-1)
            scores = scores.to(values.dtype)
            if use_area:
                scores = scores + measure.log()
            weights = Scatter.segment_softmax(scores, owners, count)
            return {
                "logits":            Scatter.sum(values * weights, owners, count),
                "attention_weights": weights,
            }
        if self.pooling_type is PoolingType.TOPK:
            pooled = []
            boundaries = surface_ptr.tolist()
            for start, stop in pairwise(boundaries):
                local = values[start:stop]
                if use_area:
                    order    = torch.argsort(local, descending=True, stable=True)
                    selected = measure[start:stop][order]
                    previous = torch.cumsum(selected, dim=0) - selected
                    # Split the boundary point's represented area, never over-cover the region.

                    retained = torch.minimum(selected, (self.topk_fraction - previous).clamp_min(0))
                    pooled.append(torch.sum(retained * local[order]) / self.topk_fraction)
                else:
                    k = max(1, math.ceil(self.topk_fraction * len(local)))
                    pooled.append(torch.topk(local, k).values.mean())
            return {"logits": torch.stack(pooled)}
        if self.pooling_type is PoolingType.LOCAL_MEAN_MAX:
            length = (
                self.learned_scalar() if self.learned_scalar is not None
                else self.regional_diffusion_scale
            )
            smoothed = DiffusionSurfaceEncoder.diffuse(values, operators, surface_ptr, length)
            return {"logits": Scatter.maximum(smoothed, owners, count)}
        if self.pooling_type is PoolingType.MULTISCALE_REGIONAL_MAX:
            bank = torch.stack([
                Scatter.maximum(
                    DiffusionSurfaceEncoder.diffuse(values, operators, surface_ptr, scale),
                    owners, count,
                )
                for scale in self.scales
            ], dim=1)
            return {"logits": bank @ torch.softmax(self.scale_logits, dim=0).to(bank.dtype)}
        if self.pooling_type is PoolingType.LOG_SUM_EXP:
            beta = self.learned_scalar() if self.learned_scalar is not None else self.runtime_beta
            return {"logits": self._normalized_lse(values, measure, owners, count, beta)}
        if self.pooling_type is PoolingType.MAX_MEAN:
            mixture = (
                self.learned_scalar() if self.learned_scalar is not None else self.fixed_lambda
            )
            maximum = Scatter.maximum(values, owners, count)
            return {"logits": mixture * maximum + (1 - mixture) * mean}

        # Probability-space hypotheses retain the same final BCEWithLogits contract. GeM uses
        # log probabilities rather than direct p**r, which underflows for negative logits.

        probabilities = torch.sigmoid(values)
        epsilon       = torch.finfo(values.dtype).eps
        if self.pooling_type is PoolingType.LINEAR_SOFTMAX:
            numerator   = Scatter.sum(measure * probabilities.square(), owners, count)
            denominator = Scatter.sum(measure * probabilities, owners, count)
            probability = numerator / denominator.clamp_min(epsilon)
        elif self.pooling_type is PoolingType.AUTOPOOL:
            alpha   = self.learned_scalar() if self.learned_scalar is not None else self.fixed_alpha
            weights = Scatter.segment_softmax(alpha * probabilities + measure.log(), owners, count)
            probability = Scatter.sum(weights * probabilities, owners, count)
        elif self.pooling_type is PoolingType.GEM:
            power = (
                self.learned_scalar() if self.learned_scalar is not None
                else values.new_tensor(self.fixed_power)
            )
            log_probabilities = nn.functional.logsigmoid(values)
            log_mean = self._normalized_lse(log_probabilities, measure, owners, count, power)
            probability = torch.exp(log_mean)
        else:
            raise ValueError(f"unsupported pooling family: {self.pooling_type}")
        probability = probability.clamp(epsilon, 1 - epsilon)
        return {"logits": torch.logit(probability)}

    @staticmethod
    def _normalized_lse(
        values : Tensor,
        measure: Tensor,
        owners : Tensor,
        count  : int,
        beta   : Tensor,
    ) -> Tensor:
        """Compute log(sum(measure*exp(beta*x)))/beta with segmented max subtraction.

        Args:
            values: Ordered local real values [M].
            measure: Positive weights [M], summing to one per protein.
            owners: Protein indices [M].
            count: Number of proteins B.
            beta: Positive scalar tensor retaining autograd to its bounded parameter.

        Returns:
            Stable normalized log-sum-exp [B]. This also computes log GeM when values are log p.
        """
        maximum = Scatter.maximum(values, owners, count)
        shifted = beta * (values - maximum[owners])
        total   = Scatter.sum(measure * shifted.exp(), owners, count)
        return maximum + total.log() / beta

    def set_log_sum_exp_beta(self, beta: float) -> None:
        """Update a fixed or scheduled beta without detaching a learned parameter.

        Args:
            beta: Positive curriculum inverse temperature.

        Raises:
            ValueError: If beta is invalid, the family is not LSE, or beta is learned.
        """
        if (
            beta <= 0 or self.pooling_type is not PoolingType.LOG_SUM_EXP
            or self.learned_scalar is not None
        ):
            raise ValueError("a beta schedule requires a positive fixed/curriculum LSE temperature")
        self.runtime_beta.fill_(beta)

    def parameter_values(self) -> dict[str, float]:
        """Read interpretable values for epoch logging and selected-checkpoint reporting.

        Returns:
            Detached scalars with stable metric names. Detachment happens only for reporting,
            never inside pooling forward; scale-bank weights sum to one.
        """
        names: dict[PoolingType, tuple[str, float | Tensor]] = {
            PoolingType.LOG_SUM_EXP:   ("pooling_beta", self.runtime_beta),
            PoolingType.LOCAL_MEAN_MAX: ("pooling_diffusion_scale", self.regional_diffusion_scale),
            PoolingType.AUTOPOOL:      ("pooling_autopool_alpha", self.fixed_alpha),
            PoolingType.GEM:           ("pooling_gem_power", self.fixed_power),
            PoolingType.MAX_MEAN:      ("pooling_max_mean_lambda", self.fixed_lambda),
        }
        if self.pooling_type is PoolingType.MULTISCALE_REGIONAL_MAX:
            weights = torch.softmax(self.scale_logits.detach(), dim=0).cpu().tolist()
            return {
                f"pooling_scale_weight_{scale:g}": weight
                for scale, weight in zip(self.scales, weights, strict=True)
            }
        if self.pooling_type not in names:
            return {}
        name, fixed = names[self.pooling_type]
        value = self.learned_scalar().detach() if self.learned_scalar is not None else fixed
        return {name: float(value)}
