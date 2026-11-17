"""GateCostNormalization vocabulary; scientific values are unchanged."""

from enum import Enum


class GateCostNormalization(str, Enum):
    """Keep historical mean cost or preserve each baseline gate's individual coefficient."""

    HISTORICAL_MEAN = "historical_mean"
    BASELINE_COUNT  = "baseline_count"
