"""Closed pre-freeze global/local head hypotheses."""

from enum import Enum


class HeadType(str, Enum):
    """Relationship between protein classification and surface localization heads."""

    SINGLE         = "single"
    DUAL           = "dual"
    GLOBAL_CONTEXT = "global_context"
    FILM           = "film"


class HeadImplementation(str, Enum):
    """Legacy checkpoint layout or selected-module, paired-initialization implementation."""

    LEGACY = "legacy"
    CONTROLLED = "controlled"


class GlobalReduction(str, Enum):
    """Per-protein reduction of surface evidence, never a reduction across proteins."""

    AREA_MEAN = "area_mean"
    MEAN = "mean"
    MAX = "max"


class LocalHeadInitialization(str, Enum):
    """Independent conditioned classifier or explicit copy of the shared local classifier."""

    INDEPENDENT = "independent"
    COPY = "copy"


class HeadLossBalance(str, Enum):
    """Historical weighted BCE sum or a unit-sum convex combination."""

    LEGACY_SUM = "legacy_sum"
    NORMALIZED = "normalized"
