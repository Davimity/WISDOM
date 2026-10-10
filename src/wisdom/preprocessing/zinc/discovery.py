"""Acquire Zn-positive candidates without manufacturing a negative protein population."""

import json
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections import Counter
from urllib.parse import urlencode
from collections.abc import Sequence
from wisdom.preprocessing.zinc.evidence import load_evidence
from wisdom.preprocessing.zinc.negatives import discover_negatives
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination


def discover_candidates(
    work: lf.Work, negative_evidence: Path | None, release_id: str, workers: int = 8,
    maximum_entries: int = 0, requests_per_second: float = 2.0,
    coordination_cutoff: float = 3.0, minimum_occupancy: float = 0.5,
    minimum_donors: int = 2, minimum_residues: int = 2,
    output_directory: str | None = None,
    negative_go_terms: Sequence[str] = ("GO:0008270", "GO:0046872"),
) -> dict[str, Any]:
    """Freeze candidates through the owning Work's cache, checkpoints and output services.

    Args:
        work: Executing Discovery or Selection Work; no nested runner is constructed.
        negative_evidence: Reviewed negative-only JSONL; null acquires experimental GO denials.
        release_id: Immutable query snapshot namespace; change it for a new public release.
        workers: Bounded concurrent entry inspections, default 8.
        maximum_entries: Sorted pilot entry cap; zero retains every query hit.
        requests_per_second: Shared native download rate ceiling, default 2.
        coordination_cutoff: Maximum donor-Zn distance in angstroms, default 3.
        minimum_occupancy: Deposited metal/donor occupancy minimum, default 0.5.
        minimum_donors: Minimum selected-copy donors, default 2.
        minimum_residues: Minimum distinct selected-copy donor residues, default 2.
        output_directory: Optional atomic external copy; null retains only managed outputs.
        negative_go_terms: Exact Zn/metal-binding denials searched when no file is supplied;
            default GO:0008270/GO:0046872. Only reviewed Zn-binding ancestors are accepted.

    Returns:
        Managed raw_path and candidate/negative counts. Persisted query and provenance accompany
        the JSONL; native resume_map retains completed per-entry inspections.

    Raises:
        ValueError: Invalid explicit negative evidence or unsupported GO term.
        RuntimeError: No experimental negative can be mapped to an exact structural sequence.
        OSError: Acquisition or output publication fails.
    """
    # Establish negative support first. Empty acquisition must stop before a costly positive scan.

    root = Path(work.outputs.directory("zinc-raw", publish_to=output_directory,
                                        overwrite=True, role="dataset-evidence"))
    automatic = negative_evidence is None
    if negative_evidence is None:
        negative_evidence = discover_negatives(work, root, release_id, workers,
                                               requests_per_second, negative_go_terms)
    negatives = load_evidence(negative_evidence)
    if not negatives:
        detail = ("See zinc-raw/negative-discovery.json." if automatic
                  else "The supplied negative_evidence file has no records.")
        raise RuntimeError(f"No exact-sequence experimental Zn negatives were found. {detail} "
                           "Supply reviewed negative_evidence or improve explicit evidence; "
                           "absent Zn never creates negatives. "
                           "The positive structure scan has not started.")
    if any(int(row["label"]) != 0 for row in negatives):
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
    rate = work.cache.rate_limit("search.rcsb.org", requests_per_second=requests_per_second)
    response = work.cache.fetch(url, key=f"zinc-discovery/{release_id}/rcsb-query.json",
                                 rate_limit=rate)
    identifiers = sorted(row["identifier"].lower()
                         for row in json.loads(Path(response).read_text())["result_set"])
    if maximum_entries:
        identifiers = identifiers[:maximum_entries]
    work.log(f"Inspecting {len(identifiers)} Zn entries; absence never creates negatives")

    # 2. Parallel managed downloads and deterministic best-copy evidence selection. Keep
    # alternative assembly-site summaries so the chosen copy can be reviewed independently.

    rows = work.resume_map(identifiers, partial(
        candidates_for_entry, work, requests_per_second=requests_per_second,
        coordination_cutoff=coordination_cutoff, minimum_occupancy=minimum_occupancy,
        minimum_donors=minimum_donors, minimum_residues=minimum_residues,
    ), key=lambda value: value, workers=workers, executor="thread", name="zinc-discovery")
    candidates = [row for entry in rows for row in entry]
    (root / "raw.jsonl").write_text("".join(
        json.dumps(row, sort_keys=True) + "\n" for row in [*candidates, *negatives]))
    (root / "rcsb-query.json").write_bytes(Path(response).read_bytes())
    (root / "provenance.json").write_text(json.dumps({
        "release_id": release_id, "query": query, "query_url": url,
        "query_sha256": hashlib.sha256(Path(response).read_bytes()).hexdigest(),
        "negative_source_sha256": hashlib.sha256(negative_evidence.read_bytes()).hexdigest(),
        "negative_source": "QuickGO experimental NOT / UniProt exact sequence" if automatic
                           else "researcher-supplied evidence",
        "inspected_entries": len(identifiers), "pilot": bool(maximum_entries),
        "positive_candidates": len(candidates), "explicit_negatives": len(negatives),
        "negative_policy": "explicit experimental/curated NOT; sequence-bound only",
    }, indent=2))
    work.outputs.artifact("raw", root / "raw.jsonl", role="dataset-evidence")
    return {"raw_path": root / "raw.jsonl", "positive_candidates": len(candidates),
            "explicit_negatives": len(negatives)}


def candidates_for_entry(
    work: lf.Work, pdb: str, requests_per_second: float, coordination_cutoff: float,
    minimum_occupancy: float, minimum_donors: int, minimum_residues: int,
) -> list[dict[str, Any]]:
    """Inspect every declared assembly/copy in one deposition, preserving alternatives.

    Args:
        work: Owning Work supplying native cache and logging services.
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

    Raises:
        OSError: The managed coordinate download or parser cannot read the deposition.
        ValueError: A deposited assembly or coordination record is inconsistent.
    """
    rate = work.cache.rate_limit("files.rcsb.org", requests_per_second=requests_per_second)
    source = work.cache.fetch(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
                               key=f"structures/{pdb}.cif", decompress="gzip",
                               timeout=180, retries=5, rate_limit=rate)
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
    work.log(f"Zn discovery {pdb}: {len(rows)} chain candidates retained for full-RAW leakage",
              level="debug")
    return rows
