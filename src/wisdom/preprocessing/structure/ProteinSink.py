"""Atomic WISDOM NPZ sink for LambdaForge preprocessing."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.StorageManager import StorageManager


class ProteinSink:
    """Publish validated NPZ records while LambdaForge owns progress and dataset identity."""

    def __init__(
        self,
        identifier_input: str = "protein_identifiers",
        dataset_output  : str = "processed",
        report_output   : str = "report",
    ) -> None:
        """Bind the logical task input/output names used by the sink.

        Args:
            identifier_input: Named TXT input used to restore authored record order in the report.
            dataset_output: Named output directory receiving per-protein NPZ files.
            report_output: Named output file receiving the compatibility human-readable report.

        Raises:
            ValueError: If any logical name is empty.
        """
        names = (identifier_input, dataset_output, report_output)
        if any(not name.strip() for name in names):
            raise ValueError("logical input and output names cannot be empty")

        self.identifier_input = identifier_input
        self.dataset_output   = dataset_output
        self.report_output    = report_output

        self.records  : dict[str, dict[str, Any]]                      = {}
        self._existing: dict[str, tuple[Path, dict[str, Any]]] | None = None

    def write(self, record: ProcessingRecord, context: ProcessingWorkspace) -> None:
        """Validate and atomically persist one transformed protein record.

        Args:
            record: Output from ``PreprocessPipeline`` containing arrays, metadata, filename and
                report fields while preserving the source's stable key.
            context: LambdaForge context resolving the named dataset directory.

        Raises:
            TypeError: If the transformed value does not follow the WISDOM sink contract.
            ValueError: If the filename is unsafe or scientific arrays fail validation.
            OSError: If atomic NPZ publication or verification fails.
        """
        if not isinstance(record.value, Mapping):
            raise TypeError("processed protein value must be a mapping")
        arrays      = record.value.get("arrays")
        metadata    = record.value.get("metadata")
        output_name = record.value.get("output_name")
        report      = record.value.get("report")
        if not isinstance(arrays, Mapping) or not isinstance(metadata, Mapping):
            raise TypeError("processed protein requires array and metadata mappings")
        if not isinstance(output_name, str) or Path(output_name).name != output_name:
            raise ValueError("processed protein output_name must be a safe filename")
        if not isinstance(report, Mapping):
            raise TypeError("processed protein requires report fields")

        scientific_config = metadata.get("config")
        if not isinstance(scientific_config, Mapping):
            raise TypeError("processed protein metadata requires scientific config")

        output_root = context.output(self.dataset_output)
        output_path = output_root / output_name
        storage     = StorageManager(PreprocessConfig(**dict(scientific_config)))
        storage.write(output_path, arrays, metadata)

        report_value               = dict(report)
        report_value["file_bytes"] = output_path.stat().st_size
        self.records[record.key]    = report_value
        if self._existing is not None:
            self._existing[record.key] = (output_path, dict(metadata))

    def is_complete(self, key: str, context: ProcessingWorkspace) -> bool:
        """Revalidate an existing NPZ and its exact source bytes before record resume.

        Args:
            key: Stable manifest identifier whose previous framework status is successful.
            context: LambdaForge context locating the named dataset output.

        Returns:
            ``True`` only when an NPZ for ``key`` exists, its current source file still hashes to
            the stored source digest, and the complete schema/numerical validation succeeds.
        """
        existing = self._existing_records(context).get(key)
        if existing is None:
            return False
        path, metadata = existing
        source_path    = Path(str(metadata.get("source_path", "")))
        source_hash    = str(metadata.get("source_hash", ""))
        if not source_path.is_file() or not source_hash:
            return False

        digest = hashlib.sha256()
        try:
            with source_path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        except OSError:
            return False
        scientific_config = metadata.get("config")
        if not isinstance(scientific_config, Mapping):
            return False
        storage = StorageManager(PreprocessConfig(**dict(scientific_config)))
        return digest.hexdigest() == source_hash and storage.can_resume(path, source_hash)

    def resume(
        self,
        record         : ProcessingRecord,
        context        : ProcessingWorkspace,
        expected_config: PreprocessConfig,
    ) -> ProcessingRecord | None:
        """Return a report for an exactly reusable archive, or request recomputation.

        Args:
            record: Current manifest record whose key and output name identify the expected NPZ.
            context: Workspace locating the checkpoint-owned processed directory.
            expected_config: Current scientific settings that must match stored metadata exactly.

        Returns:
            A compact ``skipped`` report when source hash, scientific configuration, schema, and
            numerical validation still match; otherwise ``None``.

        Raises:
            TypeError: If the record lacks a safe output filename.
        """
        if not self.is_complete(record.key, context):
            return None
        existing = self._existing_records(context).get(record.key)
        if existing is None:
            return None
        path, metadata = existing
        output_name    = record.metadata.get("output_name")
        if not isinstance(output_name, str) or path.name != output_name:
            return None
        if metadata.get("config_hash") != StorageManager(expected_config).config_hash:
            return None
        return record.with_value(
            {
                "identifier": record.key,
                "status": "skipped",
                "output": path.name,
                "atom_count": metadata.get("atom_count"),
                "residue_count": metadata.get("residue_count"),
                "atom_edge_count": metadata.get("atom_edge_count"),
                "surface_point_count": metadata.get("surface_point_count"),
                "surface_edge_count": metadata.get("surface_edge_count"),
                "surface_atom_edge_count": metadata.get("surface_atom_edge_count"),
                "file_bytes": path.stat().st_size,
                "seconds": 0.0,
                "warnings": metadata.get("warnings", []),
            }
        )

    def finalize(self, context: ProcessingWorkspace) -> tuple[Path, ...]:
        """Write the ordered compatibility report and declare scientific dataset artifacts.

        LambdaForge remains authoritative for Work-map progress, failures, Attempts, outputs, and
        immutable dataset identity. This domain report retains the exact identifier-to-NPZ mapping
        and scientific scale diagnostics that a generic map checkpoint cannot infer.

        Args:
            context: LambdaForge context resolving manifest input and named outputs.

        Returns:
            Artifact declarations for the NPZ dataset directory and compatibility JSON report.

        Raises:
            RuntimeError: If no processed or scientifically reusable NPZ exists.
            OSError: If manifest/report data cannot be read or atomically published.
        """
        output_root = context.output(self.dataset_output)
        report_path = context.output(self.report_output, create=True)
        output_root.mkdir(parents=True, exist_ok=True)

        # Every JSON map result is present here, including reports for scientifically resumed NPZs.
        merged: dict[str, dict[str, Any]] = dict(self.records)

        # Restore manifest order after asynchronous CPU transforms complete in arbitrary order.
        ordered_identifiers = list(
            dict.fromkeys(
                line
                for raw_line in context.input(self.identifier_input)
                .read_text(encoding="utf-8")
                .splitlines()
                if (line := raw_line.strip()) and not line.startswith("#")
            )
        )
        records = [merged[key] for key in ordered_identifiers if key in merged]
        records.extend(merged[key] for key in sorted(set(merged) - set(ordered_identifiers)))

        counts = {
            status: sum(record["status"] == status for record in records)
            for status in ("processed", "skipped", "failed")
        }
        report = {"total": len(records), **counts, "records": records}
        StorageManager.write_report(report_path, report)

        if counts["processed"] + counts["skipped"] == 0:
            raise RuntimeError(
                "preprocessing produced no usable proteins; inspect preprocessing-report.json "
                f"for the {counts['failed']} recorded failure(s)"
            )

        return output_root, report_path

    def _existing_records(
        self,
        context: ProcessingWorkspace,
    ) -> dict[str, tuple[Path, dict[str, Any]]]:
        """Index readable NPZ metadata once for resume and report reconstruction.

        Args:
            context: LambdaForge context locating the named dataset output.

        Returns:
            Mutable cache mapping exact source identifiers to NPZ paths and metadata.
        """
        if self._existing is None:
            self._existing = {}
            output_root    = context.output(self.dataset_output)
            storage        = StorageManager(PreprocessConfig())
            if output_root.is_dir():
                for path in sorted(output_root.glob("*.npz")):
                    metadata = storage.read_metadata(path)
                    if metadata is None:
                        continue
                    identifier = metadata.get("source_identifier")
                    if isinstance(identifier, str):
                        self._existing[identifier] = (path, metadata)
        return self._existing
