"""TransferSummary vocabulary; scientific values are unchanged."""

from enum import Enum


class TransferSummary(str, Enum):
    """Preserve different statistics without changing the output width."""

    MEAN = "mean"
    MEAN_STD = "mean_std"
    MEAN_MAX = "mean_max"
    MEAN_STD_MAX = "mean_std_max"
    MEAN_SUPPORT = "mean_support"
    DUPLICATE_MEAN = "duplicate_mean"
