"""Difficulty tiers used by the curated DNA-binding benchmark."""

from enum import Enum


class DatasetTier(str, Enum):
    """Separate the distribution-matched core from deliberate challenges."""

    CORE      = "core"
    CHALLENGE = "challenge"
