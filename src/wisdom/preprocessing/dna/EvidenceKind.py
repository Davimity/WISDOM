"""Closed evidence vocabulary for DNA-binding candidate labels."""

from enum import Enum


class EvidenceKind(str, Enum):
    """Name evidence whose interpretation is auditable and machine-checkable."""

    BIOLOGICAL_ASSEMBLY_CONTACT = "biological_assembly_contact"
    CURATED_NOT_DNA_BINDING     = "curated_not_dna_binding"
    CURATED_DNA_BINDING         = "curated_dna_binding"
    ABSENCE_OF_DNA              = "absence_of_dna"
