"""Stable JSON-compatible record passed between WISDOM preprocessing components."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


class ProcessingRecord(dict[str, Any]):
    """Represent one keyed scientific item without owning execution or persistence.

    LambdaForge 0.12 deliberately removed its former preprocessing record abstraction. WISDOM
    still needs a small domain value that keeps a stable key beside a candidate value while its
    :class:`lambdaforge.Work` calls ``self.map``. Subclassing ``dict`` keeps the value directly
    JSON-compatible, which is required by LambdaForge's safe map checkpoints.
    """

    def __init__(
        self,
        key     : str,
        value   : Any,
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Create one uniquely keyed, JSON-compatible preprocessing value.

        Args:
            key: Non-empty stable identity used for parallel-map resume and duplicate detection.
            value: Scientific candidate or result payload associated with ``key``.
            metadata: Optional JSON-compatible routing facts such as the intended filename.

        Raises:
            ValueError: If ``key`` is empty.
        """
        selected = str(key).strip()
        if not selected:
            raise ValueError("processing record key cannot be empty")
        super().__init__(key=selected, value=value, metadata=dict(metadata or {}))

    @property
    def key(self) -> str:
        """Return the non-empty stable identity used by LambdaForge ``Work.map``.

        Returns:
            Stable record identity.
        """
        return str(self["key"])

    @property
    def value(self) -> Any:
        """Return the scientific payload without copying it.

        Returns:
            Candidate or transformed scientific value stored by this record.
        """
        return self["value"]

    @property
    def metadata(self) -> dict[str, Any]:
        """Return a mutable copy of routing metadata.

        Returns:
            JSON-compatible metadata mapping.
        """
        value = self.get("metadata", {})
        return dict(value) if isinstance(value, Mapping) else {}

    def with_value(self, value: Any) -> ProcessingRecord:
        """Create a record with the same identity and metadata but a new payload.

        Args:
            value: Replacement scientific payload.

        Returns:
            Independent record retaining this record's stable key and metadata.
        """
        return ProcessingRecord(self.key, value, self.metadata)

    @classmethod
    def restore(cls, value: Mapping[str, Any]) -> ProcessingRecord:
        """Restore a record returned from a LambdaForge JSON checkpoint.

        Args:
            value: Mapping containing ``key``, ``value``, and optional ``metadata`` fields.

        Returns:
            Validated :class:`ProcessingRecord` instance.

        Raises:
            ValueError: If the checkpoint lacks a non-empty key.
        """
        return cls(str(value.get("key", "")), value.get("value"), value.get("metadata", {}))
