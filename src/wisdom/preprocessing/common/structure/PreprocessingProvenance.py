"""Source and coordinate provenance for one normalized protein."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class PreprocessingProvenance:
    """Keep processing provenance outside the scientific protein hierarchy.

    Attributes:
        source_identifier: Exact non-comment TXT record that selected the structure.
        source_path: Absolute local path read by Gemmi.
        source_hash: SHA-256 digest of the exact source bytes.
        source_format: Normalized ``pdb`` or ``mmcif`` format label.
        selected_chains: Chain identifiers actually retained by the reader.
        coordinate_origin: Centroid subtracted from source coordinates, in ångströms.
    """

    source_identifier : str
    source_path       : str
    source_hash       : str
    source_format     : str
    selected_chains   : tuple[str, ...]
    coordinate_origin : tuple[float, float, float]
