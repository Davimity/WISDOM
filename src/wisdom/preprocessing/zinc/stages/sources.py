"""Freeze mutable public observations as durable scientific inputs, not disposable cache."""

import json
import shutil
import lambdaforge as lf

from pathlib import Path
from urllib.parse import urlsplit
from collections.abc import Callable
from lambdaforge.work import ManagedFile
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive


def freeze_source(
    work                : lf.Work,
    url                 : str,
    key                 : str,
    requests_per_second : float,
    decompress          : str | None = None,
    validate            : Callable[[ManagedFile], bool] | None = None,
) -> ManagedFile:
    """Retain exact acquired bytes across cache collection and compatible code recovery.

    Args:
        work: Owning Work; LF supplies download retries, locks, integrity and atomic publication.
        url: Exact public HTTP(S) request, bound permanently to this observation key.
        key: Session-scoped scientific source identity, never a machine-specific cache path.
        requests_per_second: Native per-service request ceiling; not scientific identity.
        decompress: Native fetch decoding, None or gzip. Coordinate hashes cover decoded bytes.
        validate: Optional source-format guard applied by native cache and checkpoint services.

    Returns:
        Native checkpoint file, usable even after LF removes the download cache. The separate
        observation receipt pins the first byte digest and URL, including empty JSON results.

    Raises:
        RuntimeError: A request changes meaning or reacquisition cannot reproduce observed bytes.
        OSError: Native download, checkpoint or source reading fails. No partial source is exposed.
    """
    name        = f"acquisition-sources/{key}"
    observation = f"acquisition-observations/{key}.json"
    expected    = (json.loads(Path(work.checkpoints.file(observation)).read_text())
                   if work.checkpoints.exists(observation) else None)
    if expected is not None and (expected["url"] != url or
                                 expected["decompress"] != decompress):
        raise RuntimeError(f"Frozen Zn source {key} changed request; start a new Discovery session")

    # Only the first acquisition needs the disposable HTTP cache. LF owns the unpublished
    # checkpoint target; the scientific receipt prevents silently adopting a later revision.

    def build(path: Path) -> None:
        """Copy validated native download bytes into a durable LF build target.

        Args:
            path: Temporary checkpoint destination owned and atomically committed by LF.

        Raises:
            RuntimeError: A missing/corrupt source is reacquired with a different byte digest.
            OSError: Download or copy fails; LF retains already completed sources.
        """
        rate   = work.cache.rate_limit(urlsplit(url).netloc,
                                       requests_per_second=requests_per_second)
        source = work.cache.fetch(url, key=key, decompress=decompress, retries=5,
            retry_backoff=2.0, timeout=180, rate_limit=rate, validate=validate)
        if expected is not None:
            digest = ProteinArchive.sha256(Path(source))
            if digest != expected["sha256"]:
                raise RuntimeError(f"Frozen Zn source {key} cannot be restored byte-for-byte; "
                    "restore the original snapshot or start a new Discovery, not a mixed retry")
        shutil.copyfile(source, path)

    source = work.checkpoints.file(name, build=build, validate=validate)

    def record(path: Path) -> None:
        """Pin the first complete observation independently of the reconstructible cache.

        Args:
            path: Native receipt build target. The source has already passed byte validation.
        """
        digest = ProteinArchive.sha256(Path(source))
        path.write_text(json.dumps({"url": url, "decompress": decompress, "sha256": digest}))

    work.checkpoints.file(observation, build=record)
    return source
