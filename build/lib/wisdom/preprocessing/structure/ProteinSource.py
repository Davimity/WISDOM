"""Logical TXT input source for LambdaForge preprocessing."""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Iterable

from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace
from wisdom.preprocessing.structure.StructureCache import StructureCache


class ProteinSource:
    """Expose unique protein manifest lines as deterministic LambdaForge records."""

    def __init__(self, input_name: str = "protein_identifiers") -> None:
        """Select the logical task input containing protein identifiers.

        Args:
            input_name: Name declared in task ``inputs`` and resolved by ``context.input``.

        Raises:
            ValueError: If ``input_name`` is empty.
        """
        if not input_name.strip():
            raise ValueError("input_name cannot be empty")
        self.input_name = input_name

    def records(self, context: ProcessingWorkspace) -> Iterable[ProcessingRecord]:
        """Yield deduplicated, ordered manifest records with collision-safe output names.

        Empty and comment lines are ignored. Repeated records retain only their first occurrence.
        Output stems normally remain human-readable. Only records whose local/remote names collide
        receive a deterministic identifier digest, preventing parallel completion order from
        deciding filenames.

        Args:
            context: LambdaForge context resolving the named manifest input.

        Yields:
            One ``PreprocessingRecord`` per unique manifest line. Its key and value both preserve
            the exact line, while JSON metadata carries source line and prospective NPZ filename.

        Raises:
            OSError: If the named manifest cannot be read.
            ValueError: If a supported local filename has no usable stem.
        """
        manifest = context.input(self.input_name)
        entries  : list[tuple[str, int]] = []
        seen     : set[str]              = set()

        # Preserve the first physical occurrence so reports remain aligned with the authored TXT.
        for line_number, raw_line in enumerate(
            manifest.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            identifier = raw_line.strip()
            if not identifier or identifier.startswith("#") or identifier in seen:
                continue
            seen.add(identifier)
            entries.append((identifier, line_number))

        stems  = [StructureCache.output_stem(identifier) for identifier, _ in entries]
        counts = Counter(stems)
        for index, ((identifier, line_number), stem) in enumerate(
            zip(entries, stems, strict=True)
        ):
            if counts[stem] > 1:
                digest      = hashlib.sha256(identifier.encode("utf-8")).hexdigest()[:8]
                output_name = f"{stem}_{digest}_{index}.npz"
            else:
                output_name = f"{stem}.npz"
            yield ProcessingRecord(
                key=identifier,
                value=identifier,
                metadata={"source_line": line_number, "output_name": output_name},
            )
