"""Dense-layer initialization vocabulary."""

from enum import Enum


class LinearInitialization(str, Enum):
    """Name the supported learned linear-layer initialization policies."""

    CURRENT          = "current"
    ACTIVATION_AWARE = "activation_aware"
    ORTHOGONAL       = "orthogonal"
