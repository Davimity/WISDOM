"""Resolve manifest records against local or LambdaForge-managed structure files."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

from wisdom.preprocessing.structure.dataclasses.StructureSource import StructureSource


class StructureCache:
    """Resolve supported dataset records into hashed local structure sources.

    The class owns input grammar, managed-cache lookup, format labels, and source hashing.
    LambdaForge downloads and validates remote files before scientific workers are launched.
    """

    _LOCAL_SUFFIXES = (".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz")
    _REMOTE_ID      = re.compile(r"^[A-Za-z0-9]{3,}$")
    _CHAIN          = re.compile(r"^[A-Za-z0-9]+$")

    def __init__(self, structure_dir: str | Path) -> None:
        """Bind one record resolver to its LambdaForge-managed structure directory.

        Args:
            structure_dir: Directory containing prefetched ``<pdb_id>.cif.gz`` files.
        """
        self.structure_dir = Path(structure_dir)

    def resolve(
        self,
        identifier  : str,
        relative_to : Path,
    ) -> StructureSource:
        """Resolve one local path or remote ``XYZ_CHAIN[_CHAIN...]`` record.

        Local records must end in PDB/PDBx/mmCIF, optionally gzip-compressed, and relative paths are
        based at ``relative_to``. In remote records the first underscore separates the PDB ID from
        the first complete mmCIF chain name; later underscores separate additional complete chain
        names. Therefore ``1ABC_AQ`` selects the single chain ``AQ``, while ``1ABC_A_Q`` selects
        chains ``A`` and ``Q``. A remote file must already have been downloaded through
        ``Work.cache.fetch`` before SHA-256 hashing. LambdaForge catches record-level failures
        around this method.

        Args:
            identifier: Nonempty, comment-free dataset record.
            relative_to: Directory against which relative local structure paths are resolved.

        Returns:
            One immutable, locally available and content-hashed structure source.

        Raises:
            OSError: If a local file or prefetched remote structure is absent.
            ValueError: If the record grammar or chain selection is invalid.
        """
        lower = identifier.lower()
        if lower.endswith(self._LOCAL_SUFFIXES):
            path = Path(identifier).expanduser()
            if not path.is_absolute():
                path = (relative_to / path).resolve()
            if not path.is_file():
                raise FileNotFoundError(f"structure file not found: {path}")
            protein_id = self.output_stem(identifier)
            chains     : tuple[str, ...] = ()
            is_local   = True
        else:
            fields     = identifier.split("_")
            protein_id = fields[0]
            chains     = tuple(fields[1:])
            if not self._REMOTE_ID.fullmatch(protein_id):
                raise ValueError(f"invalid protein identifier: {identifier!r}")
            if any(not chain or not self._CHAIN.fullmatch(chain) for chain in chains):
                raise ValueError(f"invalid chain selection: {identifier!r}")

            protein_id = protein_id.lower()
            path       = self.structure_dir / f"{protein_id}.cif.gz"
            is_local   = False
            if not path.is_file():
                raise FileNotFoundError(
                    f"{protein_id} is absent from the LambdaForge-managed structure cache"
                )

        # Exact source bytes accompany every record even though task identity is framework-owned.
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)

        suffixes = {suffix.lower() for suffix in path.suffixes}
        return StructureSource(
            identifier=identifier,
            protein_id=protein_id,
            chains=chains,
            path=path.resolve(),
            sha256=digest.hexdigest(),
            format="mmcif" if suffixes & {".cif", ".mmcif"} else "pdb",
            is_local=is_local,
        )

    @staticmethod
    def output_stem(identifier: str) -> str:
        """Derive the human-readable output stem without touching source bytes.

        Args:
            identifier: Raw supported manifest record, possibly pointing to a nonexistent file.

        Returns:
            Local coordinate filename without its recognized suffix, or normalized remote PDB ID
            plus its underscore-separated chain selector. Invalid remote grammar returns a
            harmless stem; full validation remains the responsibility of :meth:`resolve` inside
            the per-record LambdaForge failure boundary.

        Raises:
            ValueError: If a local coordinate filename has no stem before its suffix.
        """
        lower = identifier.lower()
        if lower.endswith(StructureCache._LOCAL_SUFFIXES):
            name = Path(identifier).name
            for suffix in StructureCache._LOCAL_SUFFIXES:
                if lower.endswith(suffix):
                    stem = name[: -len(suffix)]
                    if not stem:
                        raise ValueError(f"structure filename has no stem: {identifier}")
                    return stem

        fields     = identifier.split("_")
        protein_id = fields[0]
        chains     = fields[1:]
        return protein_id.lower() + (f"_{'_'.join(chains)}" if chains else "")
