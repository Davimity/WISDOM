"""Independent validation of complete WISDOM preprocessing datasets."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np

from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.StorageManager import StorageManager


class DatasetValidator:
    """Audit dataset coverage, every NPZ archive, metadata, and report consistency."""

    def validate(
        self,
        processed_dir       : Path,
        preprocessing_report: Path,
        id_file             : Path,
    ) -> dict[str, Any]:
        """Validate one complete preprocessed dataset against its manifest and source report.

        Validation has three layers. First, the normalized manifest must correspond one-to-one and
        in order with successful source-report records. Second, the report must reference exactly
        the ``.npz`` files present in ``processed_dir``. Third, each archive is opened with
        ``allow_pickle=False`` and checked for the exact WISDOM schema, finite arrays, valid sparse
        indices, recomputed Euclidean distances, unit normals, normalized area weights, consistent
        metadata counts, provenance hashes, and agreement with its report record.

        Args:
            processed_dir: Directory containing one pickle-free WISDOM NPZ per expected protein.
            preprocessing_report: JSON report emitted by ``PreprocessPipeline`` for the dataset.
            id_file: Master TXT manifest whose non-empty, non-comment records define coverage.

        Returns:
            A JSON-compatible report with a global status, concise aggregate counts, global errors,
            and one ordered validation record per expected protein. Invalid data is reported rather
            than raised so callers can always publish actionable diagnostics.

        Raises:
            OSError: If the manifest, preprocessing report, or processed directory cannot be read.
            ValueError: If the preprocessing report root is not a JSON object.
        """
        # Match preprocessing normalization exactly: trim, ignore comments, and keep first use.
        identifiers = list(
            dict.fromkeys(
                line
                for raw_line in id_file.read_text(encoding="utf-8").splitlines()
                if (line := raw_line.strip()) and not line.startswith("#")
            )
        )

        report_value = json.loads(preprocessing_report.read_text(encoding="utf-8"))
        if not isinstance(report_value, dict):
            raise ValueError("preprocessing report root must be a JSON object")

        # Reject malformed, duplicate, failed, missing, or reordered preprocessing records.
        global_errors: list[str]           = []
        source_records_value               = report_value.get("records")
        source_records: list[dict[str, Any]] = []
        if not isinstance(source_records_value, list):
            global_errors.append("preprocessing report field 'records' must be a list")
        else:
            source_records = [
                cast(dict[str, Any], value)
                for value in source_records_value
                if isinstance(value, dict)
            ]
            if len(source_records) != len(source_records_value):
                global_errors.append("preprocessing report contains a non-object record")

        reported_identifiers = [str(record.get("identifier", "")) for record in source_records]
        if reported_identifiers != identifiers:
            global_errors.append(
                "preprocessing report identifiers do not exactly match manifest order and coverage"
            )
        if report_value.get("total") != len(source_records):
            global_errors.append("preprocessing report total does not equal its record count")
        if len(reported_identifiers) != len(set(reported_identifiers)):
            global_errors.append("preprocessing report contains duplicate identifiers")

        record_by_identifier = {
            str(record.get("identifier", "")): record for record in source_records
        }
        referenced_outputs: list[str] = []
        pending: list[tuple[str, Path, dict[str, Any]]] = []
        validation_records: list[dict[str, Any]]       = []

        # Resolve every expected protein to one safe basename and preserve manifest order.
        for identifier in identifiers:
            source_record = record_by_identifier.get(identifier)
            if source_record is None:
                validation_records.append(
                    {
                        "identifier": identifier,
                        "output": None,
                        "status": "invalid",
                        "errors": ["no preprocessing report record exists for this identifier"],
                    }
                )
                continue

            output_value = source_record.get("output")
            errors: list[str] = []
            if source_record.get("status") not in {"processed", "skipped"}:
                errors.append(
                    f"preprocessing status is {source_record.get('status')!r}, not successful"
                )
            if not isinstance(output_value, str) or not output_value.endswith(".npz"):
                errors.append("output must be an NPZ filename")
            elif Path(output_value).name != output_value:
                errors.append("output must be a basename without directory traversal")

            if errors:
                validation_records.append(
                    {
                        "identifier": identifier,
                        "output": output_value,
                        "status": "invalid",
                        "errors": errors,
                    }
                )
                continue

            output = cast(str, output_value)
            referenced_outputs.append(output)
            archive_path = processed_dir / output
            if not archive_path.is_file():
                validation_records.append(
                    {
                        "identifier": identifier,
                        "output": output,
                        "status": "invalid",
                        "errors": ["referenced NPZ file is missing"],
                    }
                )
                continue
            pending.append((identifier, archive_path, source_record))

        if len(referenced_outputs) != len(set(referenced_outputs)):
            global_errors.append("multiple preprocessing records reference the same NPZ file")

        existing_outputs = {path.name for path in processed_dir.glob("*.npz") if path.is_file()}
        unexpected_outputs = sorted(existing_outputs - set(referenced_outputs))
        if unexpected_outputs:
            global_errors.append(
                "processed directory contains unexpected NPZ files: "
                + ", ".join(unexpected_outputs)
            )

        # Keep scientific aggregation deterministic without owning another execution scheduler.
        inspected = [self._validate_archive(*values) for values in pending]
        validation_records.extend(inspected)

        order = {identifier: index for index, identifier in enumerate(identifiers)}
        validation_records.sort(
            key=lambda record: order.get(str(record["identifier"]), len(order))
        )

        # A dataset must use one schema and one scientific configuration across all proteins.
        config_hashes = {
            str(record["config_hash"])
            for record in validation_records
            if record.get("status") == "valid" and record.get("config_hash")
        }
        schema_versions = {
            str(record["schema_version"])
            for record in validation_records
            if record.get("status") == "valid" and record.get("schema_version")
        }
        if len(config_hashes) > 1:
            global_errors.append("valid NPZ files contain multiple scientific configuration hashes")
        if len(schema_versions) > 1:
            global_errors.append("valid NPZ files contain multiple schema versions")

        valid_count   = sum(record.get("status") == "valid" for record in validation_records)
        invalid_count = len(validation_records) - valid_count
        warning_count = sum(
            len(cast(list[Any], record.get("warnings", []))) for record in validation_records
        )
        file_bytes = sum(int(record.get("file_bytes", 0)) for record in validation_records)
        status     = "valid" if not global_errors and invalid_count == 0 else "invalid"

        surface_diagnostics = [
            cast(Mapping[str, Any], record["surface_diagnostics"])
            for record in validation_records
            if isinstance(record.get("surface_diagnostics"), Mapping)
        ]
        maximum_gap = max(
            (float(value["maximum_absolute_gap"]) for value in surface_diagnostics),
            default=float("nan"),
        )
        minimum_normal_cosine = min(
            (float(value["minimum_normal_cosine"]) for value in surface_diagnostics),
            default=float("nan"),
        )
        maximum_curvature = max(
            (
                float(value["maximum_dimensionless_curvature"])
                for value in surface_diagnostics
            ),
            default=float("nan"),
        )
        isolated_points = sum(
            int(value["isolated_surface_points"]) for value in surface_diagnostics
        )

        return {
            "status": status,
            "summary": {
                "expected_proteins": len(identifiers),
                "preprocessing_records": len(source_records),
                "archives_found": len(existing_outputs),
                "valid_proteins": valid_count,
                "invalid_proteins": invalid_count,
                "unexpected_archives": len(unexpected_outputs),
                "scientific_warnings": warning_count,
                "file_bytes": file_bytes,
                "config_hash": next(iter(config_hashes), None),
                "schema_version": next(iter(schema_versions), None),
                "maximum_surface_gap": maximum_gap,
                "minimum_normal_cosine": minimum_normal_cosine,
                "maximum_dimensionless_curvature": maximum_curvature,
                "isolated_surface_points": isolated_points,
            },
            "global_errors": global_errors,
            "unexpected_outputs": unexpected_outputs,
            "records": validation_records,
        }

    def format_summary(self, report: Mapping[str, Any]) -> str:
        """Render the detailed validation mapping as a concise human-readable report.

        Args:
            report: Complete mapping returned by ``validate``.

        Returns:
            Plain UTF-8 text containing the overall verdict, aggregate coverage, scientific warning
            count, global consistency errors, and every invalid protein with its precise reasons.
        """
        summary = cast(Mapping[str, Any], report["summary"])
        lines   = [
            "WISDOM preprocessing validation",
            "================================",
            f"Status: {str(report['status']).upper()}",
            "",
            f"Expected proteins:      {summary['expected_proteins']}",
            f"Preprocessing records:  {summary['preprocessing_records']}",
            f"NPZ archives found:     {summary['archives_found']}",
            f"Valid proteins:         {summary['valid_proteins']}",
            f"Invalid proteins:       {summary['invalid_proteins']}",
            f"Unexpected NPZ files:   {summary['unexpected_archives']}",
            f"Scientific warnings:    {summary['scientific_warnings']}",
            f"Validated NPZ bytes:    {summary['file_bytes']}",
            f"Schema version:         {summary['schema_version']}",
            f"Scientific config hash: {summary['config_hash']}",
            f"Maximum envelope gap:   {summary['maximum_surface_gap']} Å",
            f"Minimum normal cosine:  {summary['minimum_normal_cosine']}",
            f"Maximum C·radius:       {summary['maximum_dimensionless_curvature']}",
            f"Isolated surface points:{summary['isolated_surface_points']:>12}",
            "",
        ]

        global_errors = cast(list[str], report.get("global_errors", []))
        invalid_records = [
            cast(Mapping[str, Any], record)
            for record in cast(list[Any], report.get("records", []))
            if isinstance(record, Mapping) and record.get("status") != "valid"
        ]
        if global_errors:
            lines.append("Dataset errors:")
            lines.extend(f"- {error}" for error in global_errors)
            lines.append("")
        if invalid_records:
            lines.append("Invalid proteins:")
            for record in invalid_records:
                errors = "; ".join(str(value) for value in record.get("errors", []))
                lines.append(f"- {record.get('identifier')} ({record.get('output')}): {errors}")
            lines.append("")

        if report["status"] == "valid":
            lines.append(
                "All expected proteins passed archive, schema, numerical, provenance, and "
                "cross-report checks. Scientific warnings are retained diagnostics, not errors."
            )
        else:
            lines.append(
                "The dataset is not valid. Correct every error above before using these arrays."
            )
        lines.append("See validation-report.json for per-protein hashes, counts, and warnings.")
        return "\n".join(lines)

    @staticmethod
    def _validate_archive(
        identifier   : str,
        archive_path : Path,
        source_record: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Inspect one archive independently and compare it with its preprocessing record.

        Args:
            identifier: Exact normalized master-manifest record expected in provenance metadata.
            archive_path: Existing NPZ file referenced by the preprocessing report.
            source_record: Per-protein preprocessing record supplying expected counts and size.

        Returns:
            JSON-compatible valid/invalid record with SHA-256, byte/count diagnostics, schema and
            configuration identities, retained surface warnings, and actionable validation errors.
        """
        errors: list[str] = []
        result: dict[str, Any] = {
            "identifier": identifier,
            "output": archive_path.name,
            "status": "invalid",
            "errors": errors,
        }

        try:
            # Per-file SHA-256 makes a later archive comparison possible without trusting its name.
            digest = hashlib.sha256()
            with archive_path.open("rb") as handle:
                while chunk := handle.read(1024 * 1024):
                    digest.update(chunk)
            result["sha256"] = digest.hexdigest()
            result["file_bytes"] = archive_path.stat().st_size

            # Exact member names reject missing arrays, hidden extras, and pickle-bearing objects.
            with np.load(archive_path, allow_pickle=False) as archive:
                expected_names = {*StorageManager.ARRAY_NAMES, StorageManager.METADATA_NAME}
                actual_names   = set(archive.files)
                if actual_names != expected_names:
                    missing = sorted(expected_names - actual_names)
                    extra   = sorted(actual_names - expected_names)
                    errors.append(f"archive schema mismatch; missing={missing}, extra={extra}")

                arrays = {
                    name: archive[name]
                    for name in StorageManager.ARRAY_NAMES
                    if name in actual_names
                }
                metadata_array = archive[StorageManager.METADATA_NAME]
                if metadata_array.shape != () or metadata_array.dtype.kind != "U":
                    errors.append("metadata_json must be one scalar fixed-width Unicode value")
                metadata_value = json.loads(str(metadata_array.item()))

            if not isinstance(metadata_value, dict):
                errors.append("metadata_json root must be a JSON object")
                raise ValueError("metadata_json root must be a JSON object")
            metadata = cast(dict[str, Any], metadata_value)

            # Rebuild the producer's scientific configuration and reuse all numerical invariants.
            config_value = metadata.get("config")
            if not isinstance(config_value, dict):
                errors.append("metadata config must be a JSON object")
                raise ValueError("metadata config must be a JSON object")
            storage             = StorageManager(PreprocessConfig(**config_value))
            surface_diagnostics = storage.validate(arrays)

            # Provenance must identify this manifest record, schema, and exact scientific settings.
            if metadata.get("source_identifier") != identifier:
                errors.append("metadata source_identifier does not match the master manifest")
            if metadata.get("preprocessing_schema_version") != StorageManager.SCHEMA_VERSION:
                errors.append("metadata preprocessing schema version is unsupported")
            if metadata.get("config_hash") != storage.config_hash:
                errors.append("metadata config_hash does not match its canonical config")
            if not re.fullmatch(r"[0-9a-f]{64}", str(metadata.get("source_hash", ""))):
                errors.append("metadata source_hash is not a lowercase SHA-256 digest")

            coordinate_origin = np.asarray(metadata.get("coordinate_origin"), dtype=np.float64)
            if coordinate_origin.shape != (3,) or not np.isfinite(coordinate_origin).all():
                errors.append("metadata coordinate_origin must contain three finite coordinates")

            # Counts are recomputed from arrays, then compared with both metadata and source report.
            counts = {
                "atom_count": len(arrays["atom_positions"]),
                "residue_count": len(np.unique(arrays["residue_indices"])),
                "atom_edge_count": arrays["atom_edge_index"].shape[1],
                "surface_point_count": len(arrays["surface_positions"]),
                "surface_edge_count": arrays["surface_edge_index"].shape[1],
                "surface_atom_edge_count": arrays["surface_atom_edge_index"].shape[1],
            }
            for name, value in counts.items():
                if metadata.get(name) != value:
                    errors.append(f"metadata {name} does not equal recomputed value {value}")
                if name in source_record and source_record.get(name) != value:
                    errors.append(f"preprocessing report {name} does not equal {value}")
            if (
                "file_bytes" in source_record
                and source_record.get("file_bytes") != result["file_bytes"]
            ):
                errors.append("preprocessing report file_bytes does not equal current file size")

            warnings = metadata.get("warnings")
            if not isinstance(warnings, list) or not all(
                isinstance(value, str) for value in warnings
            ):
                errors.append("metadata warnings must be a list of strings")
                warnings = []
            if "warnings" in source_record and source_record.get("warnings") != warnings:
                errors.append("preprocessing report warnings differ from metadata warnings")

            result.update(
                {
                    **counts,
                    "array_bytes": sum(array.nbytes for array in arrays.values()),
                    "schema_version": metadata.get("preprocessing_schema_version"),
                    "config_hash": metadata.get("config_hash"),
                    "source_hash": metadata.get("source_hash"),
                    "warnings": warnings,
                    "surface_diagnostics": surface_diagnostics,
                }
            )
            if (
                "array_bytes" in source_record
                and source_record.get("array_bytes") != result["array_bytes"]
            ):
                errors.append("preprocessing report array_bytes does not equal loaded array bytes")
        except Exception as error:
            message = f"{type(error).__name__}: {error}"
            if message not in errors:
                errors.append(message)

        if not errors:
            result["status"] = "valid"
        return result
