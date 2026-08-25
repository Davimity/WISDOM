"""Join fixed DatasetDesign labels/assemblies to universal geometry by exact identity."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from typing import Any

from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace


class DNAAnnotationSource:
    """Expose designed catalog rows whose universal NPZ exists in the bound report."""

    def __init__(
        self,
        catalog_input        : str = "dataset_catalog",
        base_report_input    : str = "base_preprocessing_report",
        structure_cache_input: str = "structure_cache",
    ) -> None:
        """Bind the curated catalog and exact upstream preprocessing report.

        Args:
            catalog_input: Named input containing canonical ``catalog.csv`` from DatasetDesign.
            base_report_input: Named workflow-bound ``preprocessing-report.json`` input.
            structure_cache_input: Named directory containing geometry's downloaded RCSB
                ``*.cif.gz`` files.

        Raises:
            ValueError: If either logical name is empty.
        """
        names = (catalog_input, base_report_input, structure_cache_input)
        if any(not name.strip() for name in names):
            raise ValueError("annotation input names cannot be empty")
        self.catalog_input         = catalog_input
        self.base_report_input     = base_report_input
        self.structure_cache_input = structure_cache_input

    def records(self, context: ProcessingWorkspace) -> Iterable[ProcessingRecord]:
        """Yield one joined row per designed protein without guessing filenames.

        Args:
            context: LambdaForge task context resolving the two declared inputs.

        Yields:
            Records containing catalog provenance and the exact base NPZ path.

        Raises:
            ValueError: If headers, labels, identifiers, or upstream coverage disagree.
            OSError: If an input or cached coordinate file cannot be read or materialized.
        """
        report_path = context.input(self.base_report_input)
        report      = json.loads(report_path.read_text(encoding="utf-8"))
        records     = report.get("records") if isinstance(report, dict) else None
        if not isinstance(records, list):
            raise ValueError("base preprocessing report must contain a records list")
        outputs = {
            str(value["identifier"]): report_path.parent / "processed" / str(value["output"])
            for value in records
            if isinstance(value, dict)
            and value.get("status") in {"processed", "skipped"}
            and value.get("identifier")
            and value.get("output")
        }

        catalog_path = context.input(self.catalog_input)
        structure_cache = context.input(self.structure_cache_input)
        with catalog_path.open("r", encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream)
            required = {
                "base_identifier",
                "binding_residue_indices",
                "label",
                "local_gt_expected",
                "local_gt_method",
                "structure_path",
                "protein_chain",
                "dna_chains",
                "structure_sha256",
                "assembly_id",
                "protein_copy",
                "assembly_rotation",
                "assembly_translation",
            }
            if not required.issubset(reader.fieldnames or ()):
                missing = sorted(required - set(reader.fieldnames or ()))
                raise ValueError(f"DNA catalog is missing columns: {missing}")
            for line_number, row in enumerate(reader, start=2):
                identifier = str(row["base_identifier"]).strip()
                if identifier not in outputs:
                    raise ValueError(
                        f"DNA catalog line {line_number} has no successful base NPZ: {identifier}"
                    )
                value: dict[str, Any] = dict(row)
                value["label"]                   = int(row["label"])
                value["dna_chains"]              = json.loads(row["dna_chains"] or "[]")
                value["binding_residue_indices"] = json.loads(
                    row["binding_residue_indices"] or "[]"
                )
                value["assembly_rotation"] = json.loads(row["assembly_rotation"])
                value["assembly_translation"] = json.loads(row["assembly_translation"])
                value["protein_copy"] = int(row["protein_copy"])
                value["local_gt_expected"] = row["local_gt_expected"].lower() == "true"
                value["base_npz"]           = str(outputs[identifier].resolve())

                # Geometry already downloaded every selected PDB entry in parallel. The source
                # only joins its archive path. The transform content-addresses materialized bytes,
                # so chains sharing one PDB reuse the same verified uncompressed structure.
                pdb_id          = identifier.split("_", maxsplit=1)[0].lower()
                compressed_path = structure_cache / f"{pdb_id}.cif.gz"
                if not compressed_path.is_file():
                    raise FileNotFoundError(
                        f"geometry structure cache lacks {compressed_path.name}"
                    )
                value["structure_archive_path"] = str(compressed_path.resolve())
                yield ProcessingRecord(key=identifier, value=value)
