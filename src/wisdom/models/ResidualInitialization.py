"""Residual-branch initialization vocabulary."""

from enum import Enum


class ResidualInitialization(str, Enum):
    """Name the initial scale applied to learned residual branches."""

    CURRENT    = "current"
    REZERO     = "rezero"
    LAYERSCALE = "layerscale"
