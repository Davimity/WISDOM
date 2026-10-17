"""Acquire Zn-positive candidates without manufacturing a negative protein population."""

import json
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from urllib.parse import urlencode
from collections.abc import Sequence
from datetime import datetime, timezone
from collections import Counter, defaultdict
from wisdom.preprocessing.zinc.negatives import fetch_json
from wisdom.preprocessing.zinc.evidence import load_evidence
from wisdom.preprocessing.zinc.metadata import enrich_metadata
from wisdom.preprocessing.zinc.negatives import discover_negatives
from wisdom.preprocessing.zinc.negatives import write_source_receipt
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.common.snapshots import snapshot_structures
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination


def discover_candidates(
    work: lf.Work, negative_evidence: Path | None, release_id: str, workers: int = 8,
    maximum_entries: int = 0, requests_per_second: float = 2.0,
    coordination_cutoff: float = 3.0, minimum_occupancy: float = 0.5,
    minimum_donors: int = 2, minimum_residues: int = 2,
    output_directory: str | None = None,
    negative_evidence_go_terms: Sequence[str] = ("GO:0008270", "GO:0046872"),
    acquire_functional_metadata: bool = True, snapshot_label: str | None = None,
    started_at: str | None = None,
) -> dict[str, Any]:
    """Freeze candidates through the owning Work's cache, checkpoints and output services.

    Args:
        work: Executing Discovery Work; no nested runner is constructed.
        negative_evidence: Reviewed negative-only JSONL; null acquires experimental GO denials.
        release_id: Native resumable acquisition namespace; not the scientific snapshot digest.
        workers: Bounded concurrent entry inspections, default 8.
        maximum_entries: Sorted pilot entry cap; zero retains every query hit.
        requests_per_second: Shared native download rate ceiling, default 2.
        coordination_cutoff: Maximum donor-Zn distance in angstroms, default 3.
        minimum_occupancy: Deposited metal/donor occupancy minimum, default 0.5.
        minimum_donors: Minimum selected-copy donors, default 2.
        minimum_residues: Minimum distinct selected-copy donor residues, default 2.
        output_directory: Optional atomic external copy; null retains only managed outputs.
        negative_evidence_go_terms: Zn/metal-binding denials searched when no file is supplied;
            default GO:0008270/GO:0046872. Only reviewed Zn-binding ancestors are accepted.
        acquire_functional_metadata: Freeze descriptive RCSB/UniProt annotations, default true.
        snapshot_label: Optional human label; omitted from scientific snapshot identity.
        started_at: UTC acquisition-start timestamp; omitted from scientific identity.

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
        overwrite=output_directory is not None, role="dataset-evidence"))
    automatic = negative_evidence is None
    if negative_evidence is None:
        negative_evidence = discover_negatives(work, root, release_id, workers,
                                               requests_per_second, negative_evidence_go_terms)
    negatives = load_evidence(negative_evidence)
    (root / "negative-evidence.jsonl").write_bytes(negative_evidence.read_bytes())
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
        "attribute": "rcsb_nonpolymer_entity_container_identifiers.nonpolymer_comp_id",
        "operator": "exact_match", "value": "ZN"}},
        "return_type": "entry", "request_options": {"return_all_hits": True}}
    url = "https://search.rcsb.org/rcsbsearch/v2/query?" + urlencode(
        {"json": json.dumps(query, separators=(",", ":"))})
    query_key = f"zinc-discovery/{release_id}/rcsb-query.json"
    work.log("Fetching or restoring the complete RCSB ZN inventory; "
             "presence is candidate evidence, not a binding label")
    payload = fetch_json(work, url, query_key, root, requests_per_second)
    if payload.get("total_count", len(payload["result_set"])) != len(payload["result_set"]):
        raise RuntimeError("RCSB returned incomplete Zn query results; Discovery stopped")
    identifiers = sorted(row["identifier"].lower()
                         for row in payload["result_set"])
    total_entries = len(identifiers)
    if maximum_entries:
        identifiers = identifiers[:maximum_entries]
    work.log(f"Inspecting {len(identifiers)} Zn entries; absence never creates negatives")

    # 2. Parallel managed downloads and deterministic best-copy evidence selection. Keep
    # alternative assembly-site summaries so the chosen copy can be reviewed independently.

    rows = work.resume_map(identifiers, partial(
        candidates_for_entry, work, requests_per_second=requests_per_second,
        coordination_cutoff=coordination_cutoff, minimum_occupancy=minimum_occupancy,
        minimum_donors=minimum_donors, minimum_residues=minimum_residues,
        namespace=release_id,
    ), key=lambda value: value, workers=workers, executor="thread", name="zinc-discovery")
    candidates = [row for entry in rows for row in entry]

    # Freeze every RAW deposition, including later scientific rejects. A selected member can
    # remain connected to another through an excluded RAW bridge; no live revision is consulted.

    all_rows = [*candidates, *negatives]
    if acquire_functional_metadata:
        by_pdb: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in all_rows:
            by_pdb[row["identifier"].split("_", 1)[0].lower()].append(row)
        jobs = [{"pdb_id": pdb, "rows": values} for pdb, values in sorted(by_pdb.items())]
        work.log(f"Freezing descriptive metadata for {len(jobs)} depositions; "
                 "annotations do not define labels")
        metadata = work.resume_map(jobs, partial(enrich_metadata, work=work, root=root,
            namespace=release_id, requests_per_second=requests_per_second), key="pdb_id",
            workers=workers, executor="thread", name="zinc-functional-metadata")
        all_rows = [row for result in metadata for row in result["rows"]]
        for result in metadata:
            for source in result["sources"]:
                fetch_json(work, source["url"], source["key"], root, requests_per_second)
    paths = {}
    for pdb in sorted({row["identifier"].split("_", 1)[0].lower() for row in all_rows}):
        paths[pdb] = work.cache.fetch(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
            key=f"zinc-coordinates/{release_id}/{pdb}.cif", decompress="gzip", retries=5,
            timeout=180, rate_limit=work.cache.rate_limit("files.rcsb.org",
                                                        requests_per_second=requests_per_second))
    pinned = [{**row, "pdb_id": row["identifier"].split("_", 1)[0].lower(),
               "source_structure": str(paths[row["identifier"].split("_", 1)[0].lower()])}
              for row in all_rows]
    # Reviewed manual negatives need not predict a coordinate digest before acquisition.
    # Discovery binds missing digests to observed bytes; any supplied digest is still verified.

    digests = {pdb: hashlib.sha256(Path(path).read_bytes()).hexdigest()
               for pdb, path in paths.items()}
    for row, portable in zip(pinned, all_rows, strict=True):
        row.setdefault("structure_sha256", digests[row["pdb_id"]])
        portable.setdefault("structure_sha256", row["structure_sha256"])
    snapshot_structures(root / "structures", pinned)
    for pdb, path in paths.items():
        key = f"coordinates/{pdb}"
        digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
        receipt = work.checkpoints.file(f"coordinate-receipts/{release_id}/{pdb}.json",
            build=partial(write_source_receipt,
                url=f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
                key=key, digest=digest))
        record = json.loads(Path(receipt).read_text())
        record.update(file=f"structures/{pdb}.cif.gz", hash_scope="uncompressed_deposited_mmcif")
        target = root / "sources" / f"{key}.source.json"
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(record, indent=2))
    raw = root / "raw.jsonl"
    raw.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in all_rows))
    response = root / "sources" / query_key
    (root / "rcsb-query.json").write_bytes(response.read_bytes())
    source_manifest = [json.loads(path.read_text())
                       for path in sorted((root / "sources").rglob("*.source.json"))]
    policy = {"coordination_cutoff": coordination_cutoff, "minimum_occupancy": minimum_occupancy,
              "minimum_donors": minimum_donors, "minimum_residues": minimum_residues,
              "negative_evidence_go_terms": list(negative_evidence_go_terms),
              "maximum_entries": maximum_entries, "acquire_functional_metadata":
              acquire_functional_metadata}
    provenance = {
        "snapshot_label": snapshot_label, "query": query, "query_url": url,
        "acquisition_started_at": started_at,
        "acquisition_completed_at": datetime.now(timezone.utc).isoformat(),
        "sources": source_manifest, "scientific_parameters": policy,
        "raw_sha256": hashlib.sha256(raw.read_bytes()).hexdigest(),
        "query_sha256": hashlib.sha256(response.read_bytes()).hexdigest(),
        "negative_source_sha256": hashlib.sha256(negative_evidence.read_bytes()).hexdigest(),
        "negative_source": "QuickGO experimental NOT / UniProt exact sequence" if automatic
                           else "researcher-supplied evidence",
        "inspected_entries": len(identifiers), "pilot": bool(maximum_entries),
        "positive_candidates": len(candidates), "explicit_negatives": len(negatives),
        "negative_policy": "explicit experimental/curated NOT; sequence-bound only",
        "physiological_specificity_claimed": False,
    }
    scientific = {"parameters": policy, "raw_sha256": provenance["raw_sha256"],
                  "sources": sorted((row["url"], row["sha256"]) for row in source_manifest),
                  "coordinates": sorted((row["pdb_id"], row["structure_sha256"]) for row in pinned)}
    provenance["snapshot_id"] = "sha256:" + hashlib.sha256(
        json.dumps(scientific, sort_keys=True).encode()).hexdigest()
    (root / "provenance.json").write_text(json.dumps(provenance, indent=2))
    work.log(f"Discovery frozen: {len(candidates)} positive hypotheses and {len(negatives)} "
             f"explicit negative chains; snapshot {provenance['snapshot_id']}")
    acquisition = {"rcsb_entries_containing_Zn": total_entries, "inspected_entries": len(rows),
                   "candidate_chains": len(candidates),
                   "assembly_copies": sum(sum(len(row["assembly_alternatives"])
                                              for row in entry) for entry in rows),
                   "verified_candidate_chains": sum(any(choice["accepted_sites"]
                       for choice in row["assembly_alternatives"]) for row in candidates),
                   "explicit_negative_chains": len(negatives), "pilot": bool(maximum_entries)}
    (root / "acquisition.json").write_text(json.dumps(acquisition, indent=2))
    work.outputs.artifact("raw", root / "raw.jsonl", role="dataset-evidence")
    return {"raw_path": root / "raw.jsonl", "positive_candidates": len(candidates),
            "explicit_negatives": len(negatives)}


def candidates_for_entry(
    work: lf.Work, pdb: str, requests_per_second: float, coordination_cutoff: float,
    minimum_occupancy: float, minimum_donors: int, minimum_residues: int,
    namespace: str = "legacy",
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
        namespace: Native acquisition session separating distinct public revisions.

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
                               key=f"zinc-coordinates/{namespace}/{pdb}.cif", decompress="gzip",
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
                     "candidate_verified": bool(chosen["accepted_sites"]),
                     "structure_sha256": structure.sha256(),
                     "label_evidence": {"kind": "structural_coordination_candidate",
                                        "reference": f"https://www.rcsb.org/structure/{pdb}",
                                        "confidence": "structural_coordination_only"}})
    work.log(f"Zn discovery {pdb}: {len(rows)} chain candidates retained for full-RAW leakage",
              level="debug")
    return rows
