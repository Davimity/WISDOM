"""Acquire Zn-positive candidates without manufacturing a negative protein population."""

import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections.abc import Sequence
from wisdom.preprocessing.zinc.discovery import discover_candidates


class ZincDiscovery(lf.Work):
    """Freeze public Zn structure hits and acquired or researcher-supplied explicit negatives."""

    def run(
        self, negative_evidence: Path | None = None, release_id: str = "zinc-pilot-1",
        workers: int = 8, maximum_entries: int = 0, requests_per_second: float = 2.0,
        coordination_cutoff: float = 3.0, minimum_occupancy: float = 0.5,
        minimum_donors: int = 2, minimum_residues: int = 2,
        output_directory: str | None = "../data/zinc/raw",
        negative_go_terms: Sequence[str] = ("GO:0008270", "GO:0046872"),
    ) -> dict[str, Any]:
        """Query RCSB Zn depositions and freeze sequence/assembly-aware candidate JSONL.

        Args:
            negative_evidence: Optional explicit sequence-bound negative JSONL; null acquires
                experimental GO NOT annotations and exact UniProt-to-PDB sequence mappings.
            release_id: Immutable discovery snapshot label, default zinc-pilot-1.
            workers: Concurrent native cache downloads, default 8.
            maximum_entries: Zero keeps the complete query; positive value selects a sorted
                discovery pilot, which must not be represented as the complete RCSB population.
            requests_per_second: Native shared service rate ceiling, default 2.
            coordination_cutoff: Donor-Zn center distance in ångströms, default 3.
            minimum_occupancy: Deposited Zn/donor occupancy cutoff, default 0.5.
            minimum_donors: Selected-copy N/O/S atom minimum, default 2.
            minimum_residues: Selected-copy coordinating residue minimum, default 2.
            output_directory: Native publication destination, default ../data/zinc/raw.
            negative_go_terms: Default Zn/metal terms GO:0008270/GO:0046872; reviewed broader
                ion/cation ancestors GO:0043167/GO:0043169 are also accepted.

        Returns:
            Candidate count and frozen raw output. Chosen assembly/copy maximizes accepted-site
            donor evidence among deposited alternatives; all alternatives remain in the audit.
            This is an explicit structural selection rule, not proof of physiological specificity.

        Raises:
            ValueError: Invalid explicit negative evidence or unrelated GO term.
            RuntimeError: Automatic evidence cannot map any eligible negative structure.
            OSError: Public query/structure download or publication failure.
        """
        result = discover_candidates(
            self, negative_evidence, release_id, workers, maximum_entries, requests_per_second,
            coordination_cutoff, minimum_occupancy, minimum_donors, minimum_residues,
            output_directory, negative_go_terms,
        )
        return {key: value for key, value in result.items() if key != "raw_path"}
