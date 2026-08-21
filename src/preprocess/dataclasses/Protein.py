"""Normalized protein domain value."""

from dataclasses import dataclass

from preprocess.dataclasses.Atom import Atom
from preprocess.dataclasses.Chain import Chain
from preprocess.enums.BondType import BondType
from preprocess.enums.ConnectionType import ConnectionType


@dataclass(frozen=True, slots=True)
class Protein:
    """Represent a parser-independent protein with a single ownership hierarchy.

    Attributes:
        id: Stable source-derived protein identifier used as the output stem.
        chains: Retained chains; residues and atoms are reachable only through ownership.
        explicit_connections: Source-declared atom pairs with typed connection and bond semantics.
            These references point to the same immutable atoms owned by the hierarchy.
    """

    id                   : str
    chains               : tuple[Chain, ...]
    explicit_connections : tuple[tuple[Atom, Atom, ConnectionType, BondType], ...] = ()
