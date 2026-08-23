"""Atomic graph relation flags."""

from enum import IntFlag


class Relation(IntFlag):
    """Represent independently testable spatial and covalent edge semantics as bit flags."""

    SPATIAL = 1
    COVALENT = 2
