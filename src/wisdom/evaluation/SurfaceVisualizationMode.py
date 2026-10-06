"""Closed output levels for post-training surface inspection."""

from enum import Enum


class SurfaceVisualizationMode(str, Enum):
    """Control the cost and detail of final surface-prediction artifacts.

    ``NONE`` omits visualization entirely. ``REPORT`` embeds inspectors in the LF report only,
    without standalone pages or PLY. ``VIEWER`` writes the compact HTML/PLY representation
    intended for routine experimental review. ``FULL`` additionally writes point-aligned NPZ data
    so downstream quantitative inspection can reproduce every displayed channel exactly.
    """

    NONE   = "none"
    REPORT = "report"
    VIEWER = "viewer"
    FULL   = "full"
