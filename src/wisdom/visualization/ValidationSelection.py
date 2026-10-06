"""Closed presentation policies for inference-only protein validation."""

from enum import Enum


class ValidationSelection(str, Enum):
    """Choose contrasting diagnostic cases, complete coverage, or exact requested IDs."""

    DIAGNOSTIC = "diagnostic"
    ALL        = "all"
    EXPLICIT   = "explicit"
