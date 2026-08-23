"""LambdaForge 0.12 Work for split-free WISDOM-DNA curation."""

from __future__ import annotations

import json
from functools import partial
from pathlib import Path
from typing import Any

import lambdaforge as lf

from wisdom.preprocessing.dna.DNACandidateCurator import DNACandidateCurator
from wisdom.preprocessing.dna.DNACurationSink import DNACurationSink
from wisdom.preprocessing.dna.DNADatasetSource import DNADatasetSource
from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace


class Selection(lf.Work):
    """Discover and verify a split-free population of DNA benchmark candidates."""

    def run(
        self,
        public_sources             : Path,
        workers                    : int   = 72,
        requests_per_second        : float = 4.0,
        retries                    : int   = 5,
        max_positive_candidates    : int   = 0,
        max_negative_candidates    : int   = 0,
        contact_distance           : float = 4.5,
        minimum_residues           : int   = 30,
        minimum_interface_residues : int   = 2,
        minimum_sequence_coverage  : float = 0.80,
        maximum_resolution         : float = 4.0,
        challenge_aspect_ratio     : float = 4.0,
    ) -> dict[str, Any]:
        """Discover, verify, deduplicate, and report candidate DNA benchmark proteins.

        The source reads checksum-pinned public evidence, ``self.map`` applies bounded concurrent
        network and structure checks with stable JSON checkpoints, and the sink resolves duplicate
        biological proteins only after every candidate has an auditable verdict. LambdaForge owns
        execution identity, retries, progress, resume, and artifact registration; this Work creates
        no train/validation/test split and publishes no DatasetVersion.

        Args:
            public_sources: Checksum-pinned DyProL, BTD-Combo, and BioLiP release manifest.
            workers: Positive number of concurrent I/O threads used for source candidates.
            requests_per_second: Shared mean public-service request-start ceiling.
            retries: Bounded attempts for transient HTTP failures.
            max_positive_candidates: Positive source-record cap; zero reads the full release.
            max_negative_candidates: Negative source-record cap; zero reads the full release.
            contact_distance: Protein-DNA heavy-atom contact cutoff in ångströms.
            minimum_residues: Minimum observed protein-chain residue count.
            minimum_interface_residues: Minimum distinct contacting residues for a positive.
            minimum_sequence_coverage: Minimum observed/source sequence fraction in ``(0,1]``.
            maximum_resolution: Worst accepted experimental resolution in ångströms.
            challenge_aspect_ratio: Elongation threshold recorded as a morphology tier.

        Returns:
            Curation verdict, logical-protein count, and registered output names.

        Raises:
            ValueError: If a scientific or operational argument violates its documented bound.
            RuntimeError: If no defensible positive or negative logical population survives.
            OSError: If public acquisition or atomic artifact construction fails.
        """
        if workers < 1:
            raise ValueError("workers must be a positive integer")

        curation_root = self.run_dir / "curation"
        report_root   = self.run_dir / "curation-report"
        cache_root    = self.cache.path("public-discovery")
        workspace     = ProcessingWorkspace(
            self.run_dir,
            inputs={"public_sources": public_sources},
            outputs={
                "curation": curation_root,
                "curation-report": report_root,
                "discovery-cache": cache_root,
            },
        )

        source = DNADatasetSource(
            mode="live",
            source_manifest_input="public_sources",
            cache_output="discovery-cache",
            max_positive_candidates=max_positive_candidates,
            max_negative_candidates=max_negative_candidates,
            requests_per_second=requests_per_second,
            retries=retries,
        )
        curator = DNACandidateCurator(
            contact_distance=contact_distance,
            minimum_residues=minimum_residues,
            minimum_interface_residues=minimum_interface_residues,
            challenge_aspect_ratio=challenge_aspect_ratio,
            minimum_sequence_coverage=minimum_sequence_coverage,
            maximum_resolution=maximum_resolution,
            cache_output="discovery-cache",
            requests_per_second=requests_per_second,
            retries=retries,
        )

        # Candidate verification is dominated by public-service latency, so bounded threads share
        # one process and rate limiter while LambdaForge checkpoints each stable candidate key.
        candidates = tuple(source.records(workspace))
        mapped     = self.map(
            candidates,
            partial(curator.transform, context=workspace),
            key=lambda record: record.key,
            workers=workers,
            executor="thread",
            resume=True,
            name="curate-candidates",
        )

        sink = DNACurationSink(
            dataset_output="curation",
            report_output="curation-report",
        )
        sink.records = {
            record.key: dict(record.value)
            for value in mapped
            for record in (ProcessingRecord.restore(value),)
        }
        sink.finalize(workspace)

        summary = json.loads((report_root / "summary.json").read_text(encoding="utf-8"))
        self.outputs.artifact("curation", curation_root, role="dataset")
        self.outputs.artifact("curation-report", report_root, role="report")
        self.metrics.log("curated_logical_proteins", int(summary["logical_protein_count"]))
        return {
            "verdict": summary["verdict"],
            "curated_logical_proteins": summary["logical_protein_count"],
            "curation_output": "curation",
            "report_output": "curation-report",
        }
