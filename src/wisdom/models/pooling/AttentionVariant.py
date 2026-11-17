"""Closed attention-score hypotheses used by the V5 pooling study."""

from enum import Enum


class AttentionVariant(str, Enum):
    """Select simple additive scoring or gated multiplicative scoring."""

    SIMPLE = "simple"
    GATED  = "gated"
