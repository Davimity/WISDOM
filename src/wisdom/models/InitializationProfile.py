"""Closed initialization profiles used by the V1 construction campaign."""

from enum import Enum


class InitializationProfile(str, Enum):
    """Name one initialization policy without changing optimizer behavior."""

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

    def overrides(self) -> dict[str, str | float | bool | None]:
        """Return only the model-initialization values changed by this profile.

        Returns:
            Training parameter overrides for one initialization hypothesis. ``custom`` and
            ``baseline`` return empty mappings, leaving explicit parameters authoritative.
        """
        profiles: dict[InitializationProfile, dict[str, str | float | bool | None]] = {
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
        }
        return profiles[self]
