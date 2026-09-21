"""Closed optimizer-stability profiles used by the V1 construction campaign."""

from enum import Enum


class OptimizationProfile(str, Enum):
    """Name one optimization policy without replacing model initialization."""

    CUSTOM       = "custom"
    BASELINE     = "baseline"
    LR_WARMUP_05 = "lr_warmup_05"
    LR_WARMUP_10 = "lr_warmup_10"
    CLIP_1       = "clip_1"
    CLIP_5       = "clip_5"
    EMA_099      = "ema_099"
    EMA_0999     = "ema_0999"
    SWA          = "swa"

    def overrides(self) -> dict[str, str | float | bool | None]:
        """Return only the optimizer values changed by this profile.

        Returns:
            Training parameter overrides for one optimizer-stability hypothesis. ``custom`` and
            ``baseline`` return empty mappings, leaving explicit parameters authoritative.
        """
        profiles: dict[OptimizationProfile, dict[str, str | float | bool | None]] = {
            self.CUSTOM:       {},
            self.BASELINE:     {},
            self.LR_WARMUP_05: {"learning_rate_warmup_fraction": 0.05},
            self.LR_WARMUP_10: {"learning_rate_warmup_fraction": 0.10},
            self.CLIP_1:       {"gradient_clip_norm": 1.0},
            self.CLIP_5:       {"gradient_clip_norm": 5.0},
            self.EMA_099:      {"weight_averaging": "ema", "ema_decay": 0.99},
            self.EMA_0999:     {"weight_averaging": "ema", "ema_decay": 0.999},
            self.SWA:          {"weight_averaging": "swa", "swa_start_fraction": 0.80},
        }
        return profiles[self]
