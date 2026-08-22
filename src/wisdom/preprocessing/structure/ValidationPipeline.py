"""LambdaForge orchestration for independent preprocessing validation."""

from __future__ import annotations

from lambdaforge.tasks import (
    ArtifactDeclaration,
    ArtifactType,
    Task,
    TaskContext,
    TaskOutput,
)

from wisdom.preprocessing.structure.DatasetValidator import DatasetValidator
from wisdom.preprocessing.structure.StorageManager import StorageManager


class ValidationPipeline(Task):
    """Resolve immutable inputs, audit every protein, and publish clear validation artifacts."""

    REPORT_NAME  = "validation-report.json"
    SUMMARY_NAME = "validation-summary.txt"

    def __init__(
        self,
        processed_input  : str = "processed_dataset",
        report_input     : str = "preprocessing_report",
        identifier_input : str = "protein_identifiers",
        report_output    : str = "validation_report",
        summary_output   : str = "validation_summary",
    ) -> None:
        """Store logical path names for one deterministic scientific-validation task.

        Args:
            processed_input: Named input for the fingerprinted preprocessing dataset directory.
            report_input: Named input for its matching compatibility report.
            identifier_input: Named input for the master protein TXT used by preprocessing.
            report_output: Named output for the detailed machine-readable validation report.
            summary_output: Named output for the concise human-readable validation verdict.

        Raises:
            ValueError: If any logical input/output name is empty.
        """
        names = (
            processed_input,
            report_input,
            identifier_input,
            report_output,
            summary_output,
        )
        if any(not name.strip() for name in names):
            raise ValueError("logical input and output names cannot be empty")

        self.processed_input  = processed_input
        self.report_input     = report_input
        self.identifier_input = identifier_input
        self.report_output    = report_output
        self.summary_output   = summary_output

    def run(self, context: TaskContext) -> TaskOutput:
        """Validate a complete preprocessing artifact and publish JSON plus plain-text results.

        LambdaForge first resolves and fingerprints the processed directory, preprocessing report,
        and master manifest. ``DatasetValidator`` then checks coverage and each archive. Both report
        forms are written atomically. An invalid dataset raises only after those files are written,
        so the failed run retains readable diagnostics and cannot be mistaken for valid output.

        Args:
            context: LambdaForge attempt context used to resolve declared inputs and safe outputs.

        Returns:
            Successful task output declaring the detailed JSON and concise text report artifacts,
            with validation counts exposed as scalar metrics.

        Raises:
            ValueError: If any global or per-protein validation check fails.
            OSError: If declared inputs cannot be read or reports cannot be published atomically.
        """
        # Named paths keep physical locations in YAML and out of domain implementation.
        processed_dir        = context.input(self.processed_input)
        preprocessing_report = context.input(self.report_input)
        id_file              = context.input(self.identifier_input)

        validator = DatasetValidator()
        report    = validator.validate(processed_dir, preprocessing_report, id_file)
        summary   = validator.format_summary(report)

        # Publish both machine-readable detail and a terminal-friendly verdict before failing.
        report_path  = context.output(self.report_output, create=True)
        summary_path = context.output(self.summary_output, create=True)
        StorageManager.write_report(report_path, report)
        StorageManager.write_text(summary_path, summary)

        counts = report["summary"]
        if report["status"] != "valid":
            raise ValueError(
                "preprocessed dataset is invalid; inspect validation-summary.txt and "
                "validation-report.json in this run directory"
            )

        outputs = {
            "status": report["status"],
            "report": report_path.relative_to(context.run_dir).as_posix(),
            "summary": summary_path.relative_to(context.run_dir).as_posix(),
            "valid_proteins": counts["valid_proteins"],
            "invalid_proteins": counts["invalid_proteins"],
            "scientific_warnings": counts["scientific_warnings"],
        }
        metrics = {
            "valid_proteins": counts["valid_proteins"],
            "invalid_proteins": counts["invalid_proteins"],
            "scientific_warnings": counts["scientific_warnings"],
        }
        return TaskOutput(
            outputs=outputs,
            metrics=metrics,
            artifacts=(
                ArtifactDeclaration(
                    path=report_path.relative_to(context.run_dir),
                    kind=ArtifactType.REPORT,
                    media_type="application/json",
                ),
                ArtifactDeclaration(
                    path=summary_path.relative_to(context.run_dir),
                    kind=ArtifactType.REPORT,
                    media_type="text/plain",
                ),
            ),
        )
