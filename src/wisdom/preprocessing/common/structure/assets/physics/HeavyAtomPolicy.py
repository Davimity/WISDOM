"""Versioned chemical admission choices, separate from numerical field convergence."""

from enum import Enum


class HeavyAtomPolicy(str, Enum):
    """Keep historical observed-only references distinct from bounded modeled completion."""

    STRICT = "strict"
    BOUNDED = "bounded"
