"""Parser-independent residue domain value."""

from dataclasses import dataclass

from wisdom.utils.structure.models.Atom import Atom


@dataclass(frozen=True, slots=True)
class Residue:
    """Represent one residue and exclusively own its retained atoms.

    Attributes:
        name: Uppercase residue/component name.
        number: Author residue sequence number.
        insertion_code: Normalized insertion code; the empty string means absent.
        is_polymer: Whether Gemmi assigns the residue to a polymer entity.
        atoms: Deterministically selected atoms sorted by atom name.
    """

    name           : str
    number         : int
    insertion_code : str
    is_polymer     : bool
    atoms          : tuple[Atom, ...]
