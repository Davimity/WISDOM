"""Closed scientific alternatives for VN01, not interchangeable loss implementations."""

from enum import Enum


class NegativeMiningMode(str, Enum):
    """Name the no-intervention, regional, area-matched, distributed and audit controls."""

    NONE        = "none"
    REGIONAL    = "regional"
    POINT       = "point"
    DISTRIBUTED = "distributed"
    AUDIT       = "audit"
