"""Closed one-factor profiles for initialization and optimizer stability screens."""

from enum import Enum


class StabilityProfile(str, Enum):
    """Name candidates that alter one stability subsystem without hidden combinations."""

    CUSTOM               = "custom"
    BASELINE             = "baseline"
    EMBEDDING_FAN        = "embedding_fan"
    EMBEDDING_SMALL      = "embedding_small"
    EMBEDDING_PHYSICAL   = "embedding_physical"
    GATE_050             = "gate_050"
    GATE_099             = "gate_099"
    GATE_WARMUP          = "gate_warmup"
    DIFFUSION_REZERO     = "diffusion_rezero"
    DIFFUSION_LAYERSCALE = "diffusion_layerscale"
    ATOMIC_REZERO        = "atomic_rezero"
    ATOMIC_LAYERSCALE    = "atomic_layerscale"
    NEUTRAL_EDGES        = "neutral_edges"
    NEUTRAL_TRANSFER     = "neutral_transfer"
    LOCAL_SMALL          = "local_small"
    LOCAL_CALIBRATED     = "local_calibrated"
    ACTIVATION_AWARE     = "activation_aware"
    ORTHOGONAL           = "orthogonal"
    BROADER_DIFFUSION    = "broader_diffusion"
    SAME_DIFFUSION       = "same_diffusion"
    LR_WARMUP_05         = "lr_warmup_05"
    LR_WARMUP_10         = "lr_warmup_10"
    CLIP_1               = "clip_1"
    CLIP_5               = "clip_5"
    EMA_099              = "ema_099"
    EMA_0999             = "ema_0999"
    SWA                  = "swa"

    def overrides(self) -> dict[str, str | float | bool | None]:
        """Return the values intentionally changed from the historical baseline.

        Returns:
            Training parameter overrides for this one-factor candidate. ``custom`` returns an
            empty mapping so every explicit YAML value remains authoritative.
        """
        profiles: dict[StabilityProfile, dict[str, str | float | bool | None]] = {
            self.CUSTOM:               {},
            self.BASELINE:             {},
            self.EMBEDDING_FAN:        {"embedding_initialization": "fan_scaled"},
            self.EMBEDDING_SMALL:      {"embedding_initialization": "small_normal"},
            self.EMBEDDING_PHYSICAL:   {"embedding_initialization": "physical"},
            self.GATE_050:             {"gate_initial_active": 0.50},
            self.GATE_099:             {"gate_initial_active": 0.99},
            self.GATE_WARMUP:          {
                "gate_warmup_fraction": 0.10,
                "gate_ramp_fraction":   0.20,
            },
            self.DIFFUSION_REZERO:     {"diffusion_residual_initialization": "rezero"},
            self.DIFFUSION_LAYERSCALE: {"diffusion_residual_initialization": "layerscale"},
            self.ATOMIC_REZERO:        {"atomic_residual_initialization": "rezero"},
            self.ATOMIC_LAYERSCALE:    {"atomic_residual_initialization": "layerscale"},
            self.NEUTRAL_EDGES:        {"neutral_edge_initialization": True},
            self.NEUTRAL_TRANSFER:     {"neutral_transfer_initialization": True},
            self.LOCAL_SMALL:          {"local_head_weight_std": 0.01},
            self.LOCAL_CALIBRATED:     {
                "local_head_weight_std":     0.01,
                "calibrate_local_head_bias": True,
            },
            self.ACTIVATION_AWARE:     {"linear_initialization": "activation_aware"},
            self.ORTHOGONAL:           {"linear_initialization": "orthogonal"},
            self.BROADER_DIFFUSION:    {"diffusion_time_initialization": "broader_lengths"},
            self.SAME_DIFFUSION:       {"diffusion_time_initialization": "same_scale"},
            self.LR_WARMUP_05:         {"learning_rate_warmup_fraction": 0.05},
            self.LR_WARMUP_10:         {"learning_rate_warmup_fraction": 0.10},
            self.CLIP_1:               {"gradient_clip_norm": 1.0},
            self.CLIP_5:               {"gradient_clip_norm": 5.0},
            self.EMA_099:              {"weight_averaging": "ema", "ema_decay": 0.99},
            self.EMA_0999:             {"weight_averaging": "ema", "ema_decay": 0.999},
            self.SWA:                  {"weight_averaging": "swa", "swa_start_fraction": 0.80},
        }
        return profiles[self]
