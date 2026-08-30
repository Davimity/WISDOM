"""Parser-independent protein domain value."""

from dataclasses import dataclass

from wisdom.utils.structure.models.Atom import Atom
from wisdom.utils.structure.models.Chain import Chain
from wisdom.utils.structure.enums.BondType import BondType
from wisdom.utils.structure.enums.ConnectionType import ConnectionType


@dataclass(frozen=True, slots=True)
class Protein:
    """Represent a protein through one non-duplicated ownership hierarchy.

    Attributes:
        id: Stable source-derived protein identifier.
        chains: Retained chains; residues and atoms are reachable only through ownership.
        explicit_connections: Source-declared atom pairs with typed connection and bond semantics.
            These references point to the same immutable atoms owned by the hierarchy.
    """

    id                   : str
    chains               : tuple[Chain, ...]
    explicit_connections : tuple[tuple[Atom, Atom, ConnectionType, BondType], ...] = ()
