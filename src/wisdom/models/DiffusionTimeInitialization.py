"""Diffusion-time initialization vocabulary."""

from enum import Enum


class DiffusionTimeInitialization(str, Enum):
    """Name physical time schedules used to initialize DiffusionNet blocks."""

    CURRENT         = "current"
    BROADER_LENGTHS = "broader_lengths"
    SAME_SCALE      = "same_scale"
