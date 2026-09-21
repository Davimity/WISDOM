"""Closed atom/surface interaction-round hypotheses."""

from enum import Enum


class InteractionRound(str, Enum):
    """Pre-freeze fusion depth and feedback controls."""

    SINGLE        = "single"
    DEPTH_CONTROL = "depth_control"
    BIDIRECTIONAL = "bidirectional"
