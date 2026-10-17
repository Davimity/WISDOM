"""Closed presentation profiles; neither profile changes inference or numerical metrics."""

from enum import Enum


class VisualizationContent(str, Enum):
    """Select lightweight predictions, sampled learned features, or full structural layers."""

    PREDICTIONS = "predictions"
    FEATURES    = "features"
    FULL        = "full"
