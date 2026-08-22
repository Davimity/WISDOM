"""One function-first LambdaForge Work for WISDOM-DNA protein selection."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import lambdaforge as lf
from lambdaforge.preprocessing import PreprocessingTask
from lambdaforge.tasks import TaskContext

from wisdom.preprocessing.dna.DNACandidateCurator import DNACandidateCurator
from wisdom.preprocessing.dna.DNADatasetSource import DNADatasetSource
from wisdom.preprocessing.dna.DNASelectionSink import DNASelectionSink


def select_dna(
    public_sources            : Path,
    dilutions                 : Sequence[float] = (0.10, 0.25, 0.50, 0.75),
    workers                   : int = 72,
    requests_per_second       : float = 4.0,
    retries                   : int = 5,
    max_positive_candidates   : int = 0,
    max_negative_candidates   : int = 0,
    contact_distance          : float = 4.5,
    minimum_residues          : int = 30,
    minimum_interface_residues: int = 2,
    minimum_sequence_coverage : float = 0.80,
    maximum_resolution        : float = 4.0,
    challenge_aspect_ratio    : float = 4.0,
    validation_fraction       : float = 0.20,
    reserve_fraction          : float = 0.10,
    selection_seed            : int = 2026,
) -> dict[str, Any]:
    """Discover, verify, cluster, balance, dilute, and report DNA benchmark members.

    This is the only public entry point for the lightweight selection action. LambdaForge 0.11
    resolves and fingerprints ``public_sources``, reserves the requested resources from YAML, and
    invokes this ordinary function. Internally, its public ``PreprocessingTask`` still owns stable
    record keys, the bounded thread pool, checkpoints, resume, aggregate progress, and failure
    policy. WISDOM owns only the scientific source, curator, and final selection rules.

    No universal surface or model input is generated here. The portable ``selection`` artifact
    contains exact balanced train/validation/test membership, positive reserve pools, evidence,
    external MMseqs2 cluster identities, and nested dilution views. The large discovery cache and
    per-candidate checkpoints remain run evidence and are not part of that portable artifact.

    Args:
        public_sources: Checksum-pinned JSON definition of DyProL, BTD-Combo, and BioLiP releases.
        dilutions: Strictly interior unit fractions used for nested balanced training views; the
            complete validation and test membership remains fixed in every view.
        workers: Positive number of concurrent I/O threads. Network request starts remain governed
            by one shared rate limiter, so workers overlap latency rather than increase API rate.
        requests_per_second: Positive mean start-rate ceiling shared by public-service clients.
        retries: Positive bounded HTTP attempt count with exponential backoff.
        max_positive_candidates: Non-negative source limit; zero selects the complete release.
        max_negative_candidates: Non-negative source limit; zero selects the complete release.
        contact_distance: Maximum protein-to-DNA heavy-atom centre distance in ångströms.
        minimum_residues: Minimum accepted observed protein-chain residue count.
        minimum_interface_residues: Minimum distinct contacting residues for a positive structure.
        minimum_sequence_coverage: Minimum observed/source sequence ratio in ``(0,1]``.
        maximum_resolution: Largest accepted experimental resolution in ångströms.
        challenge_aspect_ratio: Principal-axis elongation threshold for the challenge tier.
        validation_fraction: Fraction of development sequence clusters assigned to validation.
        reserve_fraction: Fraction of positive clusters reserved for local-GT replacement.
        selection_seed: Seed used for deterministic cluster, balance, and dilution ordering.

    Returns:
        JSON-compatible summary containing the verdict, selected record count, dilution statistics,
        and run-relative artifact locations.

    Raises:
        ValueError: If a scientific, operational, or path parameter violates its documented bound.
        RuntimeError: If evidence cannot yield non-empty exactly balanced leakage-free splits.
        OSError: If public acquisition, checkpoints, or atomic selection publication fails.
    """
    runtime    = lf.current()
    stage_root = runtime.run_dir / "selection-work"
    context    = TaskContext(
        name               = runtime.name,
        run_dir            = stage_root,
        source_dir         = runtime.source_dir,
        attempt_id         = runtime.attempt_id,
        config_fingerprint = f"{runtime.config_fingerprint}:selection-v1",
        resume             = True,
        inputs = (
            {
                "name":          "public_sources",
                "path":          str(public_sources),
                "resolved_path": str(public_sources.resolve()),
            },
        ),
        outputs = {
            "selection":             "selection",
            "selection-report":      "selection-report",
            "selection-checkpoints": "selection-checkpoints",
            "discovery-cache":       "discovery-cache",
        },
    )

    # LambdaForge retains record-level concurrency and resume, while these three cohesive classes
    # retain the scientific meanings of candidate discovery, verification, and final membership.
    task = PreprocessingTask(
        source = DNADatasetSource(
            mode = "live",
            source_manifest_input="public_sources",
            cache_output="discovery-cache",
            max_positive_candidates=max_positive_candidates,
            max_negative_candidates=max_negative_candidates,
            requests_per_second=requests_per_second,
            retries=retries,
        ),
        transforms=(
            DNACandidateCurator(
                contact_distance=contact_distance,
                minimum_residues=minimum_residues,
                minimum_interface_residues=minimum_interface_residues,
                challenge_aspect_ratio=challenge_aspect_ratio,
                minimum_sequence_coverage=minimum_sequence_coverage,
                maximum_resolution=maximum_resolution,
                cache_output="discovery-cache",
                requests_per_second=requests_per_second,
                retries=retries,
            ),
        ),
        sink=DNASelectionSink(
            dataset_output="selection",
            report_output="selection-report",
            checkpoint_output="selection-checkpoints",
            seed=selection_seed,
            validation_fraction=validation_fraction,
            reserve_fraction=reserve_fraction,
            dilutions=dilutions,
        ),
        workers=workers,
        workload="io",
        on_error="fail",
        checkpoint_interval=100,
        progress_interval_seconds=10.0,
    )
    task.run(context)

    selection = stage_root / "selection"
    report    = stage_root / "selection-report"
    summary   = json.loads((report / "summary.json").read_text(encoding="utf-8"))

    lf.artifact("selection", selection, role="dataset")
    lf.artifact("selection-report", report, role="report")
    lf.metric("selected_members", int(summary["logical_member_count"]))
    for split, counts in sorted(summary["split_class_balance"].items()):
        lf.metric(
            "selected_members",
            int(counts["positive"]) + int(counts["negative"]),
            split=str(split),
        )

    return {
        "verdict": summary["verdict"],
        "selected_members": summary["logical_member_count"],
        "dilutions": summary["dilutions"],
        "selection_artifact": "selection",
        "report_artifact": "selection-report",
    }
