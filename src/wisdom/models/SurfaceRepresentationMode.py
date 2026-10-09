"""Closed vocabulary of late-fusion surface representations."""

from enum import Enum


class SurfaceRepresentationMode(str, Enum):
    """Choose learned features, fixed projected chemistry, or their concatenation."""

    LEARNED = "learned"
    EXPLICIT = "explicit"
    HYBRID = "hybrid"
