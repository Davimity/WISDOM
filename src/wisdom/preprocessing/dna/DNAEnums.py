"""Small closed vocabularies shared by WISDOM-DNA curation components."""

from enum import Enum


class DNALabel(str, Enum):
    """Represent the four evidence states before a benchmark row is accepted."""

    POSITIVE = "positive"
    NEGATIVE = "negative"
    UNKNOWN  = "unknown"
    CONFLICT = "conflict"


class DatasetTier(str, Enum):
    """Separate the distribution-matched core from deliberate morphology challenges."""

    CORE      = "core"
    CHALLENGE = "challenge"


class DiscoveryMode(str, Enum):
    """Select bounded fixtures or the official live public-data services."""

    FIXTURE = "fixture"
    LIVE    = "live"


class EvidenceKind(str, Enum):
    """Name label evidence whose interpretation is auditable and machine-checkable."""

    BIOLOGICAL_ASSEMBLY_CONTACT = "biological_assembly_contact"
    CURATED_NOT_DNA_BINDING     = "curated_not_dna_binding"
    CURATED_DNA_BINDING         = "curated_dna_binding"
    ABSENCE_OF_DNA              = "absence_of_dna"
