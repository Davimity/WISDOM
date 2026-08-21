"""Resolved input structure."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class StructureSource:
    """Describe one validated and locally available structure ready for Gemmi.

    Attributes:
        identifier: Exact dataset TXT record used for reporting.
        protein_id: Source-derived output identifier without coordinate suffix.
        chains: Per-record remote chain selection, or an empty tuple.
        path: Absolute path to a local or cached coordinate file.
        sha256: Digest of the exact file bytes used for safe resume.
        format: Normalized ``pdb`` or ``mmcif`` label inferred from the suffix.
        is_local: Whether the dataset record was a path rather than a remote PDB ID.
    """

    identifier : str
    protein_id : str
    chains     : tuple[str, ...]
    path       : Path
    sha256     : str
    format     : str
    is_local   : bool

    @property
    def name(self) -> str:
        """Compose the deterministic output stem.

        Returns:
            ``protein_id`` followed by ``_`` and concatenated remote chain IDs when a per-record
            chain selector exists. Local filenames never acquire an inferred chain selector.
        """
        return self.protein_id + ("_" + "".join(self.chains) if self.chains else "")
