"""AtomicInputSource vocabulary; scientific values are unchanged."""

from enum import Enum


class AtomicInputSource(str, Enum):
    """Name blocks before input projection, never individual proteins or atom instances."""

    ATOM_NAME = "atom_name"
    RESIDUE_ATOM = "residue_atom"
    BACKBONE = "backbone_torsions"
    SIDECHAIN = "sidechain_torsions"
    COVALENT = "covalent_statistics"
    GROUPS = "functional_groups"
    SASA = "atomic_sasa"
