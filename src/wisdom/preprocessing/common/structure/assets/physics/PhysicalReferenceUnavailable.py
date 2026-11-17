"""Distinguish a scientifically unsupported reference from an execution or storage failure."""

from typing import Any
from collections.abc import Mapping


class PhysicalReferenceUnavailable(ValueError):
    """Mark unsupported chemistry without treating missing physical supervision as zero energy.

    Only known preparation limitations may raise this exception. Infrastructure errors, unknown
    tool failures, invalid arrays and failed independent LJ audits must still stop publication.
    """

    def __init__(self, message: str, audit: Mapping[str, Any] | None = None) -> None:
        """Retain structured preparation limitations without treating them as physical evidence.

        Args:
            message: Human-readable reason why the unchanged observed structure is unsupported.
            audit: JSON-compatible atom-identity differences or known upstream limitations;
                absent coordinates are not synthesized by this diagnostic.
        """
        super().__init__(message)
        self.audit = dict(audit or {})
