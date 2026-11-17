"""Optimization-time weight-averaging vocabulary."""

from enum import Enum


class WeightAveragingMode(str, Enum):
    """Name evaluation-weight policies used by optimization stability experiments."""

    NONE = "none"
    EMA  = "ema"
    SWA  = "swa"
