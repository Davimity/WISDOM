"""Categorical embedding initialization vocabulary."""

from enum import Enum


class EmbeddingInitialization(str, Enum):
    """Name the supported categorical embedding initialization policies."""

    CURRENT      = "current"
    FAN_SCALED   = "fan_scaled"
    SMALL_NORMAL = "small_normal"
    PHYSICAL     = "physical"
