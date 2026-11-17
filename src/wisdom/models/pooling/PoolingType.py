"""Closed set of protein-level pooling hypotheses studied by WISDOM v2."""

from enum import Enum


class PoolingType(str, Enum):
    """Name each controlled v2 multiple-instance pooling strategy."""

    MAX                     = "max"
    MEAN                    = "mean"
    ATTENTION               = "attention"
    TOPK                    = "topk"
    LOCAL_MEAN_MAX          = "local_mean_max"
    LOG_SUM_EXP             = "log_sum_exp"
    LINEAR_SOFTMAX          = "linear_softmax"
    AUTOPOOL                = "autopool"
    GEM                     = "gem"
    MAX_MEAN                = "max_mean"
    MULTISCALE_REGIONAL_MAX = "multiscale_regional_max"
