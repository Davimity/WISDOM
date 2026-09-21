"""Closed output levels for post-training surface inspection."""

from enum import Enum


class SurfaceVisualizationMode(str, Enum):
    """Control the cost and detail of final surface-prediction artifacts.

    ``NONE`` omits visualization entirely. ``VIEWER`` writes the compact HTML/PLY representation
    intended for routine experimental review. ``FULL`` additionally writes point-aligned NPZ data
    so downstream quantitative inspection can reproduce every displayed channel exactly.
    """

    NONE   = "none"
    VIEWER = "viewer"
    FULL   = "full"
