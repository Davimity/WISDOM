"""GlobalReduction vocabulary; scientific values are unchanged."""

from enum import Enum


class GlobalReduction(str, Enum):
    """Per-protein reduction of surface evidence, never a reduction across proteins."""

    AREA_MEAN = "area_mean"
    MEAN = "mean"
    MAX = "max"
