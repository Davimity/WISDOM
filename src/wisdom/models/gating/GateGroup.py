"""Disjoint scientific roles, independent of module-name prefixes."""

from enum import Enum


class GateGroup(str, Enum):
    """Assign direct L0 costs once; hierarchy still determines effective path activity."""

    SURFACE  = "surface"
    CHEMICAL = "chemical"
    OTHER    = "other"
