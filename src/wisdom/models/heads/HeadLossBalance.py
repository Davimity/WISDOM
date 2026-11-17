"""HeadLossBalance vocabulary; scientific values are unchanged."""

from enum import Enum


class HeadLossBalance(str, Enum):
    """Historical weighted BCE sum or a unit-sum convex combination."""

    LEGACY_SUM = "legacy_sum"
    NORMALIZED = "normalized"
