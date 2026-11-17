"""HeadImplementation vocabulary; scientific values are unchanged."""

from enum import Enum


class HeadImplementation(str, Enum):
    """Legacy checkpoint layout or selected-module, paired-initialization implementation."""

    LEGACY = "legacy"
    CONTROLLED = "controlled"
