"""Closed atom/surface interaction-round hypotheses."""

from enum import Enum


class InteractionRound(str, Enum):
    """Pre-freeze fusion depth and feedback controls."""

    SINGLE        = "single"
    DEPTH_CONTROL = "depth_control"
    DEPTH_CONTROL_PARAMETER_MATCHED = "depth_control_parameter_matched"
    BIDIRECTIONAL = "bidirectional"
