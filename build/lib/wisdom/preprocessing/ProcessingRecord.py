"""Small keyed value passed between WISDOM preprocessing components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ProcessingRecord(dict[str, Any]):
    """Keep one stable map key beside its scientific value and routing metadata."""

    def __init__(
        self,
        key     : str | Mapping[str, Any],
        value   : Any = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create the JSON-compatible value exchanged by LambdaForge map workers.

        Args:
            key: Stable record identity, or a complete mapping returned by a map worker.
            value: Scientific candidate or result associated with a text key.
            metadata: Optional routing information such as an output filename.
        """
        if isinstance(key, Mapping):
            record   = key
            key      = str(record["key"])
            value    = record.get("value")
            metadata = record.get("metadata")
        super().__init__(key=key, value=value, metadata=dict(metadata or {}))

    @property
    def key(self) -> str:
        """Return the stable identity used by LambdaForge map.

        Returns:
            Record identity as text.
        """
        return str(self["key"])

    @property
    def value(self) -> Any:
        """Return the scientific payload without copying it.

        Returns:
            Candidate or transformed value stored by this record.
        """
        return self["value"]

    @property
    def metadata(self) -> dict[str, Any]:
        """Return an independent routing-metadata mapping.

        Returns:
            Copy of the record metadata.
        """
        return dict(self["metadata"])

    def with_value(self, value: Any) -> ProcessingRecord:
        """Return the same keyed record with a replacement scientific payload.

        Args:
            value: Scientific payload for the returned record.

        Returns:
            New record retaining this key and metadata.
        """
        return ProcessingRecord(self.key, value, self.metadata)
