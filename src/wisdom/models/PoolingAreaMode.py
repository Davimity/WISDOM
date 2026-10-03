"""Surface-measure choices for protein-level pooling."""

from enum import Enum


class PoolingAreaMode(str, Enum):
    """Choose whether a pooling rule integrates points or represented surface area."""

    LEGACY = "legacy"
    POINT  = "point"
    AREA   = "area"
