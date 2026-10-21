"""Parser-independent atom domain value."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Atom:
    """Represent one atom in a normalized Cartesian coordinate system.

    Attributes:
        index: Consecutive zero-based index in hierarchy traversal order.
        name: Structure atom name after surrounding whitespace removal.
        atomic_number: Chemical element atomic number supplied by Gemmi.
        position: Cartesian coordinates ``(x, y, z)`` in ångströms.
        formal_charge: Integral formal charge read from the source structure.
        occupancy: Observed mmCIF selected occupancy, or None when not explicitly available.
        b_iso: Explicit isotropic displacement in Å² with an eligible experimental mmCIF method;
            None when unavailable, including predicted-model confidence stored in B's column.
        altloc: Selected alternate-conformation code, empty for a blank code.
        alternate_count: Number of retained source alternatives sharing this atom name.
    """

    index          : int
    name           : str
    atomic_number  : int
    position       : tuple[float, float, float]
    formal_charge  : int = 0

    occupancy      : float | None = None
    b_iso          : float | None = None
    altloc         : str          = ""
    alternate_count: int          = 1
