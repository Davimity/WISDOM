"""Closed distance representations for the SchNet filter-input ablation."""

from enum import Enum


class RadialEncoding(str, Enum):
    """Choose a fixed Gaussian expansion or one normalized scalar, never a constant filter."""

    GAUSSIAN = "gaussian"
    DIRECT = "direct"
