"""Acquire Zn-positive candidates without manufacturing a negative protein population."""

import json
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections import Counter
from urllib.parse import urlencode
from wisdom.preprocessing.zinc.evidence import load_evidence
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination


class ZincDiscovery(lf.Work):
    """Freeze public Zn structure hits together with researcher-supplied explicit negatives."""

    def run(
        self, negative_evidence: Path, release_id: str, workers: int = 8,
        maximum_entries: int = 0, requests_per_second: float = 2.0,
        coordination_cutoff: float = 3.0, minimum_occupancy: float = 0.5,
        minimum_donors: int = 2, minimum_residues: int = 2,
        output_directory: str | None = "../data/zinc/raw",
    ) -> dict[str, Any]:
        """Query RCSB Zn depositions and freeze sequence/assembly-aware candidate JSONL.

        Args:
            negative_evidence: Explicit curated/experimental sequence-bound negative JSONL.
                No assumed non-binding proteins are generated when this source is absent.
            release_id: Researcher-assigned immutable discovery snapshot label, no default.
            workers: Concurrent native cache downloads, default 8.
            maximum_entries: Zero keeps the complete query; positive value selects a sorted
                discovery pilot, which must not be represented as the complete RCSB population.
            requests_per_second: Native shared service rate ceiling, default 2.
            coordination_cutoff: Donor-Zn center distance in ångströms, default 3.
            minimum_occupancy: Deposited Zn/donor occupancy cutoff, default 0.5.
            minimum_donors: Selected-copy N/O/S atom minimum, default 2.
            minimum_residues: Selected-copy coordinating residue minimum, default 2.
            output_directory: Native publication destination, default ../data/zinc/raw.

        Returns:
            Candidate count and frozen raw output. Chosen assembly/copy maximizes accepted-site
            donor evidence among deposited alternatives; all alternatives remain in the audit.
            This is an explicit structural selection rule, not proof of physiological specificity.

        Raises:
            ValueError: Invalid explicit negative evidence or absent negative population.
            OSError: Public query/structure download or publication failure.
        """
        negatives = load_evidence(negative_evidence)
        if not negatives or any(int(row["label"]) != 0 for row in negatives):
            raise ValueError("Zn discovery needs a nonempty explicit negative-only evidence source")

        # 1. Freeze the official query response. An entry with ZN is not a positive protein:
        # assembly/copy-specific coordination is checked before constructing each candidate.

        query = {"query": {"type": "terminal", "service": "text", "parameters": {
            "attribute": ("rcsb_nonpolymer_entity."
                          "rcsb_nonpolymer_entity_container_identifiers.chem_comp_id"),
            "operator": "exact_match", "value": "ZN"}},
            "return_type": "entry", "request_options": {"return_all_hits": True}}
        url = "https://search.rcsb.org/rcsbsearch/v2/query?" + urlencode(
            {"json": json.dumps(query)})
        response = self.cache.fetch(url, key=f"zinc-discovery/{release_id}/rcsb-query.json",
                                     rate_limit=requests_per_second)
        identifiers = sorted(row["identifier"].lower()
                             for row in json.loads(Path(response).read_text())["result_set"])
        if maximum_entries:
            identifiers = identifiers[:maximum_entries]
        self.log(f"Inspecting {len(identifiers)} Zn entries; absence never creates negatives")

        # 2. Parallel managed downloads and deterministic best-copy evidence selection. Keep
        # alternative assembly-site summaries so the chosen copy can be reviewed independently.

        rows = self.resume_map(identifiers, partial(
            self._candidates, requests_per_second=requests_per_second,
            coordination_cutoff=coordination_cutoff, minimum_occupancy=minimum_occupancy,
            minimum_donors=minimum_donors, minimum_residues=minimum_residues,
        ), key=lambda value: value, workers=workers, executor="thread", name="zinc-discovery")
        candidates = [row for entry in rows for row in entry]
        root = Path(self.outputs.directory("zinc-raw", publish_to=output_directory,
                                            overwrite=True, role="dataset-evidence"))
        (root / "raw.jsonl").write_text("".join(
            json.dumps(row, sort_keys=True) + "\n" for row in [*candidates, *negatives]))
        (root / "rcsb-query.json").write_bytes(Path(response).read_bytes())
        (root / "provenance.json").write_text(json.dumps({
            "release_id": release_id, "query": query, "query_url": url,
            "query_sha256": hashlib.sha256(Path(response).read_bytes()).hexdigest(),
            "negative_source_sha256": hashlib.sha256(negative_evidence.read_bytes()).hexdigest(),
            "inspected_entries": len(identifiers), "pilot": bool(maximum_entries),
            "positive_candidates": len(candidates), "explicit_negatives": len(negatives),
            "negative_policy": "explicit experimental/curated NOT; sequence-bound only",
        }, indent=2))
        self.outputs.artifact("raw", root / "raw.jsonl", role="dataset-evidence")
        return {"positive_candidates": len(candidates), "explicit_negatives": len(negatives)}

    def _candidates(
        self, pdb: str, requests_per_second: float, coordination_cutoff: float,
        minimum_occupancy: float, minimum_donors: int, minimum_residues: int,
    ) -> list[dict[str, Any]]:
        """Inspect every declared assembly/copy in one deposition, preserving alternatives.

        Args:
            pdb: Public deposition ID.
            requests_per_second: Native shared download ceiling.
            coordination_cutoff: Maximum Zn-donor distance in ångströms.
            minimum_occupancy: Metal/donor occupancy minimum.
            minimum_donors: Selected-copy donor minimum.
            minimum_residues: Selected-copy residue minimum.

        Returns:
            One coordinate-pinned candidate per deposited chain, including chains without
            verified coordination so full-RAW leakage grouping retains potential bridges.
            Candidate labels are hypotheses; Selection must revalidate before acceptance.
        """
        source = self.cache.fetch(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
                                   key=f"structures/{pdb}.cif", decompress="gzip",
                                   timeout=180, retries=5, rate_limit=requests_per_second)
        structure = ProteinStructure(Path(source))
        verifier = ZincCoordination(coordination_cutoff, minimum_occupancy,
                                     minimum_donors, minimum_residues)
        alternatives: dict[str, list[dict[str, Any]]] = {}
        for descriptor in structure.structure.assemblies:
            assembly = structure.assembly(str(descriptor.name))
            copies: Counter[str] = Counter()
            for chain in assembly.protein_chains():
                copies[chain.name] += 1
                sites = verifier.analyse(assembly, chain.name, copies[chain.name])
                accepted = [site for site in sites if site["accepted"]]
                alternatives.setdefault(chain.name, []).append({
                    "assembly_id": str(descriptor.name), "protein_copy": copies[chain.name],
                    "accepted_sites": len(accepted),
                    "donor_count": sum(site["selected_donor_count"] for site in accepted),
                })
        rows = []
        for chain_name, choices in sorted(alternatives.items()):
            chosen = sorted(choices, key=lambda choice: (-choice["donor_count"],
                            choice["assembly_id"], choice["protein_copy"]))[0]
            sequence = structure.sequence(structure.structure[0].find_chain(chain_name))
            rows.append({"identifier": f"{pdb.upper()}_{chain_name}",
                         "sequence": sequence, "label": 1,
                         "assembly_id": chosen["assembly_id"],
                         "protein_copy": chosen["protein_copy"],
                         "origin": "rcsb_zinc_coordination", "assembly_alternatives": choices,
                         "structure_sha256": structure.sha256(),
                         "label_evidence": {"kind": "structural_coordination_candidate",
                                            "reference": f"https://www.rcsb.org/structure/{pdb}",
                                            "confidence": "structural_coordination_only"}})
        self.log(f"Zn discovery {pdb}: {len(rows)} chain candidates retained for full-RAW leakage",
                  level="debug")
        return rows
