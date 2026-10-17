"""Acquire Zn-positive candidates without manufacturing a negative protein population."""

import json
import uuid
import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections.abc import Sequence
from datetime import datetime, timezone
from wisdom.preprocessing.zinc.discovery import discover_candidates


class ZincDiscovery(lf.Work):
    """Freeze public Zn structure hits and acquired or researcher-supplied explicit negatives."""

    def run(
        self, skip: bool = False, existing_raw: Path | None = None,
        negative_evidence: Path | None = None, snapshot_label: str | None = None,
        workers: int = 8, maximum_entries: int = 0, requests_per_second: float = 2.0,
        coordination_cutoff: float = 3.0, minimum_occupancy: float = 0.5,
        minimum_donors: int = 2, minimum_residues: int = 2,
        output_directory: str | None = "../data/zinc/raw",
        negative_evidence_go_terms: Sequence[str] = ("GO:0008270", "GO:0046872"),
        acquire_functional_metadata: bool = True,
    ) -> dict[str, Any]:
        """Query RCSB Zn depositions and freeze sequence/assembly-aware candidate JSONL.

        Args:
            skip: Forward an exact existing_raw snapshot, performing no acquisition.
            existing_raw: Complete Discovery directory with raw.jsonl, provenance and structures;
                required when skipped, never a standalone unpinned JSONL.
            negative_evidence: Optional explicit sequence-bound negative JSONL; null acquires
                experimental GO NOT annotations and exact UniProt-to-PDB sequence mappings.
            snapshot_label: Optional descriptive researcher label; never the cache identity.
            workers: Concurrent native cache downloads, default 8.
            maximum_entries: Zero keeps the complete query; positive value selects a sorted
                discovery pilot, which must not be represented as the complete RCSB population.
            requests_per_second: Native shared service rate ceiling, default 2.
            coordination_cutoff: Donor-Zn center distance in ångströms, default 3.
            minimum_occupancy: Deposited Zn/donor occupancy cutoff, default 0.5.
            minimum_donors: Selected-copy N/O/S atom minimum, default 2.
            minimum_residues: Selected-copy coordinating residue minimum, default 2.
            output_directory: Native publication destination, default ../data/zinc/raw.
            negative_evidence_go_terms: Terms whose experimental NOT annotations are sought,
                default Zn/metal terms GO:0008270/GO:0046872; reviewed broader
                ion/cation ancestors GO:0043167/GO:0043169 are also accepted.
            acquire_functional_metadata: Freeze optional RCSB/UniProt descriptive metadata,
                default true; missing metadata never invents a family or a label.

        Returns:
            Candidate count and frozen raw output. Chosen assembly/copy maximizes accepted-site
            donor evidence among deposited alternatives; all alternatives remain in the audit.
            This is an explicit structural selection rule, not proof of physiological specificity.

        Raises:
            ValueError: Invalid explicit negative evidence or unrelated GO term.
            RuntimeError: Automatic evidence cannot map any eligible negative structure.
            OSError: Public query/structure download or publication failure.
        """
        if skip:
            if existing_raw is None:
                raise ValueError("existing_raw is required when Zn Discovery is skipped")
            self.outputs.artifact("zinc-raw", existing_raw, role="dataset-evidence")
            self.outputs.artifact("raw", existing_raw / "raw.jsonl", role="dataset-evidence")
            self.log("Zn discovery skipped; forwarding the exact observable-evidence snapshot")
            return {"skipped": True}

        # Native checkpoints retain the acquisition namespace across Attempts. A fresh Run
        # starts a fresh observation; a human label cannot accidentally pin old public data.

        session = self.checkpoints.file("discovery-session.json", build=self._start_session)
        identity = json.loads(Path(session).read_text())
        result = discover_candidates(
            self, negative_evidence, identity["session_id"], workers,
            maximum_entries, requests_per_second,
            coordination_cutoff, minimum_occupancy, minimum_donors, minimum_residues,
            output_directory, negative_evidence_go_terms, acquire_functional_metadata,
            snapshot_label, identity["started_at"],
        )
        return {key: value for key, value in result.items() if key != "raw_path"}

    @staticmethod
    def _start_session(path: Path) -> None:
        """Write a native resumable acquisition identity, not a scientific release identifier.

        Args:
            path: LF-owned checkpoint build target.

        Returns:
            None; UUID prevents cross-Run stale reuse and UTC time records acquisition start.

        Raises:
            OSError: Checkpoint cannot be written; native LF owns atomic publication.
        """
        path.write_text(json.dumps({"session_id": uuid.uuid4().hex,
                                    "started_at": datetime.now(timezone.utc).isoformat()}))
