"""Closed presentation profiles; neither profile changes inference or numerical metrics."""

from enum import Enum


class VisualizationContent(str, Enum):
    """Select lightweight prediction maps or the complete structural inspector."""

    PREDICTIONS = "predictions"
    FULL        = "full"
