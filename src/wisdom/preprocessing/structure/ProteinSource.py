"""Turn a TXT manifest into stable JSON-compatible protein records."""

from __future__ import annotations

import hashlib

from pathlib import Path
from collections import Counter
from collections.abc import Iterable

from wisdom.preprocessing.structure.StructureResolver import StructureResolver


class ProteinSource:
    """Expose unique manifest lines with deterministic output filenames."""

    def records(self, manifest: Path) -> Iterable[dict[str, object]]:
        """Yield deduplicated records in authored order.

        Empty lines and comments are ignored. The returned dictionaries are directly accepted by
        LambdaForge ``resume_map``; no project-specific record wrapper is necessary.

        Args:
            manifest: UTF-8 TXT file containing one structure identifier or local path per line.

        Yields:
            Dictionaries with stable ``key``, source ``identifier``, line number, and NPZ name.

        Raises:
            OSError: If the manifest cannot be read.
            ValueError: If a supported local filename has no usable stem.
        """
        entries: list[tuple[str, int]] = []
        seen   : set[str]              = set()

        # Preserve the first occurrence so final reports follow the authored manifest order.

        for line_number, raw_line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            identifier = raw_line.strip()
            if not identifier or identifier.startswith("#") or identifier in seen:
                continue

            seen.add(identifier)
            entries.append((identifier, line_number))

        stems  = [StructureResolver.output_stem(identifier) for identifier, _ in entries]
        counts = Counter(stems)

        # A short digest is needed only when different manifest strings share one readable stem.

        for index, ((identifier, line_number), stem) in enumerate(
            zip(entries, stems, strict=True)
        ):
            output_name = f"{stem}.npz"
            if counts[stem] > 1:
                digest      = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8]
                output_name = f"{stem}_{digest}_{index}.npz"

            yield {
                "key":         identifier,
                "identifier":  identifier,
                "source_line": line_number,
                "output_name": output_name,
            }
