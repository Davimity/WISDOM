"""Atomic persistence and scientific resume for universal protein NPZ files."""

from __future__ import annotations

import hashlib

from typing import Any
from pathlib import Path
from collections.abc import Mapping

from wisdom.preprocessing.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig


class ProteinSink:
    """Write validated geometry and summarize JSON-compatible map results."""

    def __init__(self) -> None:
        """Create an empty collection of per-protein reports."""
        self.records: dict[str, dict[str, Any]] = {}

    def write(self, record: Mapping[str, Any], output_root: Path) -> None:
        """Validate and atomically persist one transformed protein.

        Args:
            record: Mapping carrying a stable key plus arrays, metadata, output name, and report.
            output_root: Checkpoint-owned directory receiving universal NPZ files.

        Raises:
            TypeError: If arrays, metadata, or report do not follow the transform contract.
            ValueError: If the output filename or scientific arrays are invalid.
            OSError: If the archive cannot be published and reopened.
        """
        key         = str(record["key"])
        arrays      = record.get("arrays")
        metadata    = record.get("metadata")
        output_name = record.get("output_name")
        report      = record.get("report")

        if not isinstance(arrays, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("processed protein requires array and metadata mappings")
        if not isinstance(output_name, str) or Path(output_name).name != output_name:
            raise ValueError("processed protein output_name must be a safe filename")
        if not isinstance(report, Mapping):
            raise TypeError("processed protein requires report fields")

        scientific_config = metadata.get("config")
        if not isinstance(scientific_config, Mapping):
            raise TypeError("processed protein metadata requires scientific config")

        output_path = output_root / output_name
        archive     = ProteinArchive(PreprocessConfig(**dict(scientific_config)))
        archive.write(output_path, arrays, metadata)

        report_value               = dict(report)
        report_value["file_bytes"] = output_path.stat().st_size
        self.records[key]           = report_value

    def resume(
        self,
        record         : Mapping[str, Any],
        output_root    : Path,
        expected_config: PreprocessConfig,
    ) -> dict[str, Any] | None:
        """Return a fresh report when one existing NPZ is exactly reusable.

        Args:
            record: Current source record with ``key`` and deterministic ``output_name``.
            output_root: Directory containing prior universal NPZ files.
            expected_config: Current scientific configuration.

        Returns:
            JSON-compatible skipped report, or ``None`` when recomputation is required.
        """
        key         = str(record["key"])
        output_name = record.get("output_name")
        if not isinstance(output_name, str) or Path(output_name).name != output_name:
            return None

        path     = output_root / output_name
        archive  = ProteinArchive(expected_config)
        metadata = archive.read_metadata(path)
        if metadata is None or metadata.get("source_identifier") != key:
            return None

        # The expected filename is opened directly; no worker scans the complete NPZ directory.

        source_path = Path(str(metadata.get("source_path", "")))
        source_hash = str(metadata.get("source_hash", ""))
        if not source_path.is_file() or not source_hash:
            return None

        digest = hashlib.sha256()
        try:
            with source_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return None

        if digest.hexdigest() != source_hash or not archive.can_resume(path, source_hash):
            return None

        return {
            "key": key,
            "value": {
                "identifier":             key,
                "status":                 "skipped",
                "output":                 path.name,
                "atom_count":             metadata.get("atom_count"),
                "residue_count":          metadata.get("residue_count"),
                "atom_edge_count":        metadata.get("atom_edge_count"),
                "surface_point_count":    metadata.get("surface_point_count"),
                "atom_spatial_candidate_count": metadata.get(
                    "atom_spatial_candidate_count"
                ),
                "surface_atom_neighbor_count": metadata.get(
                    "surface_atom_neighbor_count"
                ),
                "diffusion_spectral_modes": metadata.get("diffusion_spectral_modes"),
                "diffusion_gradient_entries": metadata.get("diffusion_gradient_entries"),
                "file_bytes":             path.stat().st_size,
                "seconds":                0.0,
                "warnings":               metadata.get("warnings", []),
            },
        }

    def finalize(
        self,
        manifest   : Path,
        output_root: Path,
        report_path: Path,
    ) -> tuple[Path, Path]:
        """Write the ordered preprocessing report.

        Args:
            manifest: Input TXT whose order should be preserved in the report.
            output_root: Directory containing validated NPZ files.
            report_path: Destination JSON report.

        Returns:
            The geometry directory and report path.

        Raises:
            RuntimeError: If no processed or reusable protein exists.
            OSError: If the manifest or report cannot be read or written.
        """
        output_root.mkdir(parents=True, exist_ok=True)

        ordered = list(
            dict.fromkeys(
                line
                for raw_line in manifest.read_text(encoding="utf-8").splitlines()
                if (line := raw_line.strip()) and not line.startswith("#")
            )
        )
        records = [self.records[key] for key in ordered if key in self.records]
        records.extend(self.records[key] for key in sorted(set(self.records) - set(ordered)))

        statuses = ("processed", "skipped", "failed")
        counts   = {
            status: sum(record["status"] == status for record in records)
            for status in statuses
        }
        report = {"total": len(records), **counts, "records": records}
        ProteinArchive.write_report(report_path, report)

        if counts["processed"] + counts["skipped"] == 0:
            raise RuntimeError(
                "preprocessing produced no usable proteins; inspect preprocessing-report.json "
                f"for the {counts['failed']} recorded failure(s)"
            )

        return output_root, report_path
