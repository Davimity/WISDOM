"""Input parsing and race-safe local structure cache."""

from __future__ import annotations

import gzip
import hashlib
import os
import re
import time
import urllib.request
from pathlib import Path
from uuid import uuid4

from preprocess.dataclasses.StructureSource import StructureSource


class StructureCache:
    """Resolve supported dataset records into hashed local structure sources.

    The class owns input grammar, remote-cache paths, race-safe HTTP publication, format labels,
    and source hashing. Dataset iteration and concurrency belong to LambdaForge.
    """

    _LOCAL_SUFFIXES = (".pdb", ".cif", ".mmcif", ".pdb.gz", ".cif.gz", ".mmcif.gz")
    _REMOTE_ID      = re.compile(r"^[A-Za-z0-9]{3,}$")
    _CHAINS         = re.compile(r"^[A-Za-z0-9]+$")
    _REMOTE_URL     = "https://files.rcsb.org/download/{protein_id}.cif.gz"

    def __init__(
        self,
        raw_dir : str | Path,
        download: bool = True,
    ) -> None:
        """Bind one record resolver to its named download cache.

        Args:
            raw_dir: Run-owned directory used for downloaded PDBx/mmCIF files.
            download: Whether an absent remote entry may be downloaded from RCSB PDB.
        """
        self.raw_dir = Path(raw_dir)
        self.download = download

    def resolve(
        self,
        identifier  : str,
        relative_to : Path,
    ) -> StructureSource:
        """Resolve one local path or remote ``XYZ_ABC`` record.

        Local records must end in PDB/PDBx/mmCIF, optionally gzip-compressed, and relative paths are
        based at ``relative_to``. Remote records use the suffix characters after one underscore as
        individual chain IDs. An absent remote entry is downloaded atomically before SHA-256
        hashing. LambdaForge catches record-level failures around this method.

        Args:
            identifier: Nonempty, comment-free dataset record.
            relative_to: Directory against which relative local structure paths are resolved.

        Returns:
            One immutable, locally available and content-hashed structure source.

        Raises:
            OSError: If a local file is absent or a remote download cannot be completed.
            ValueError: If the record grammar, chain selection, or downloaded content is invalid.
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
            protein_id, separator, chain_text = identifier.rpartition("_")
            if not separator:
                protein_id, chain_text = identifier, ""
            if not self._REMOTE_ID.fullmatch(protein_id):
                raise ValueError(f"invalid protein identifier: {identifier!r}")
            if chain_text and not self._CHAINS.fullmatch(chain_text):
                raise ValueError(f"invalid chain selection: {identifier!r}")

            protein_id = protein_id.lower()
            chains     = tuple(chain_text)
            path       = self.raw_dir / f"{protein_id}.cif.gz"
            is_local   = False
            if not path.is_file():
                if not self.download:
                    raise FileNotFoundError(
                        f"{protein_id} is absent from cache and download=false"
                    )
                self.raw_dir.mkdir(parents=True, exist_ok=True)
                self._download(protein_id, path)

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

    @classmethod
    def output_stem(cls, identifier: str) -> str:
        """Derive the human-readable output stem without touching source bytes.

        Args:
            identifier: Raw supported manifest record, possibly pointing to a nonexistent file.

        Returns:
            Local coordinate filename without its recognized suffix, or normalized remote PDB ID
            plus its concatenated chain selector. Invalid remote grammar returns a harmless stem;
            full validation remains the responsibility of :meth:`resolve` inside the per-record
            LambdaForge failure boundary.

        Raises:
            ValueError: If a local coordinate filename has no stem before its suffix.
        """
        lower = identifier.lower()
        if lower.endswith(cls._LOCAL_SUFFIXES):
            name = Path(identifier).name
            for suffix in cls._LOCAL_SUFFIXES:
                if lower.endswith(suffix):
                    stem = name[: -len(suffix)]
                    if not stem:
                        raise ValueError(f"structure filename has no stem: {identifier}")
                    return stem

        protein_id, separator, chain_text = identifier.rpartition("_")
        if not separator:
            protein_id, chain_text = identifier, ""
        return protein_id.lower() + (f"_{chain_text}" if chain_text else "")

    def _download(
        self,
        protein_id : str,
        target     : Path,
    ) -> None:
        """Download and atomically publish one compressed RCSB PDBx/mmCIF entry.

        An exclusive ``.lock`` serializes competing writers. The owner streams one-mebibyte chunks
        to a PID/UUID temporary file, flushes and synchronizes it, checks that gzip yields payload,
        and publishes with ``os.replace``. Waiters poll every 0.1 seconds and give up after 180
        seconds. Cleanup runs for success and failure.

        Args:
            protein_id: Normalized remote PDB identifier used in the RCSB download URL.
            target: Final run-cache path, conventionally ``<id>.cif.gz``.

        Raises:
            TimeoutError: If another writer leaves the target unavailable for 180 seconds.
            OSError: If locking, HTTP streaming, synchronization, or publication fails.
            ValueError: If the downloaded gzip stream has no decompressed payload.
        """
        lock     = target.with_suffix(target.suffix + ".lock")
        deadline = time.monotonic() + 180.0

        # Become the sole cache writer, or wait until the current writer publishes its target.
        while True:
            try:
                descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
                os.close(descriptor)
                break
            except FileExistsError:
                if target.is_file():
                    return
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for cache lock: {lock}") from None
                time.sleep(0.1)

        temporary = target.with_name(f".{target.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            if target.is_file():
                return
            # Stream and synchronize bytes before validating gzip and atomically renaming the file.
            url = self._REMOTE_URL.format(protein_id=protein_id.upper())
            with (
                urllib.request.urlopen(url, timeout=60.0) as response,
                temporary.open("wb") as output,
            ):
                while chunk := response.read(1024 * 1024):
                    output.write(chunk)
                output.flush()
                os.fsync(output.fileno())
            with gzip.open(temporary, "rb") as compressed:
                if not compressed.read(16):
                    raise ValueError(f"empty structure downloaded from {url}")
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)
            lock.unlink(missing_ok=True)
