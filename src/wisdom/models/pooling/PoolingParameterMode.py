"""Optimization policies for scalar pooling parameters."""

from enum import Enum


class PoolingParameterMode(str, Enum):
    """Distinguish fixed, learned, and externally scheduled pooling parameters."""

    FIXED      = "fixed"
    LEARNED    = "learned"
    CURRICULUM = "curriculum"
