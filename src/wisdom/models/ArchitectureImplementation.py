"""Versioned architecture behavior and closed scientific execution policies."""

from enum import Enum


class ArchitectureImplementation(str, Enum):
    """Preserve archived scalar/vector behavior or use the paired, audited V8 contract."""

    LEGACY     = "legacy"
    CONTROLLED = "controlled"


class FeedbackWeighting(str, Enum):
    """Select the integration measure of surface-to-atom aggregation."""

    POINT = "point"
    AREA  = "area"


class GateCostNormalization(str, Enum):
    """Keep historical mean cost or preserve each baseline gate's individual coefficient."""

    HISTORICAL_MEAN = "historical_mean"
    BASELINE_COUNT  = "baseline_count"
