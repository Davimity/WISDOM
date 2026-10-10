"""Acquire experimental GO denials and map them to exact deposited protein sequences."""

import json
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections import Counter
from urllib.parse import urlencode, urlsplit
from collections.abc import Mapping, Sequence
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.preprocessing.zinc.evidence import EXPERIMENTAL_CODES, ZINC_NEGATIVE_TERMS


def fetch_json(work: lf.Work, url: str, key: str, root: Path,
               requests_per_second: float,
               sources: list[dict[str, str]] | None = None) -> dict[str, Any]:
    """Cache a public response and retain its exact bytes beside the acquisition audit.

    Args:
        work: Executing Work providing native cache and shared rate limits.
        url: Public JSON endpoint; no authentication or project-owned downloader is used.
        key: Release-specific relative source identity, also used beneath root/sources.
        root: Managed evidence output directory.
        requests_per_second: Per-service request ceiling, in requests per second.
        sources: Optional logical URL/key ledger for restoring outputs after map reuse.

    Returns:
        Parsed JSON object; the original response remains available for independent review.

    Raises:
        OSError: Native retries cannot acquire or store the response.
        ValueError: The service returns malformed JSON.
    """
    rate   = work.cache.rate_limit(urlsplit(url).netloc,
                                   requests_per_second=requests_per_second)
    source = work.cache.fetch(url, key=key, retries=5, timeout=90, rate_limit=rate)
    target = root / "sources" / key
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(Path(source).read_bytes())
    if sources is not None:
        sources.append({"url": url, "key": key})
    return json.loads(target.read_text())


def annotations(work: lf.Work, parameters: Mapping[str, str], namespace: str,
                root: Path, requests_per_second: float,
                sources: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """Freeze all QuickGO pages without allowing a short response to hide evidence.

    Args:
        work: Native download/logging owner.
        parameters: QuickGO search selectors, including explicit ontology traversal semantics.
        namespace: Release-specific key prefix separating each term or subject query.
        root: Managed evidence directory containing unmodified source JSON.
        requests_per_second: Shared QuickGO request ceiling.
        sources: Optional logical response ledger retained in the reconstructible map result.

    Returns:
        All annotation records in the frozen response, including subsequently excluded evidence.

    Raises:
        RuntimeError: Hit counts change during pagination or the response is incomplete.
        OSError: Native response acquisition fails.
    """
    rows: list[dict[str, Any]] = []
    page, total, expected      = 1, 1, None
    while page <= total:
        query   = dict(parameters, limit="200", page=str(page))
        url     = "https://www.ebi.ac.uk/QuickGO/services/annotation/search?" + urlencode(query)
        payload = fetch_json(work, url, f"{namespace}/page-{page}.json", root,
                             requests_per_second, sources)
        hits    = int(payload["numberOfHits"])
        if expected is not None and hits != expected:
            raise RuntimeError("QuickGO changed during pagination; use a new release_id")
        expected = hits
        total    = int(payload["pageInfo"]["total"])
        rows.extend(payload["results"])
        work.log(f"QuickGO {namespace}: page {page}/{max(total, 1)}, "
                 f"{len(rows)}/{hits} annotations")
        page += 1
    if len(rows) != expected:
        raise RuntimeError("Incomplete QuickGO snapshot; refusing a partial evidence inventory")
    return rows


def discover_negatives(
    work: lf.Work, root: Path, release_id: str, workers: int,
    requests_per_second: float, go_terms: Sequence[str],
) -> Path:
    """Build a conservative sequence-bound negative inventory before scanning Zn structures.

    Args:
        work: Work owning native downloads, reconstructible maps and outputs.
        root: Managed raw output directory; receives evidence JSONL and exclusion audit.
        release_id: Frozen source-response namespace, changed for a new acquisition.
        workers: Maximum concurrent UniProt/PDB mapping operations.
        requests_per_second: Per-service shared download ceiling.
        go_terms: Zn-binding term or reviewed is-a ancestors in ZINC_NEGATIVE_TERMS.

    Returns:
        Generated negative-evidence.jsonl, possibly empty. The caller must refuse an empty
        inventory before acquiring a massive positive population. Multiple structures of one
        sequence are candidates, not independent negative experiments or leakage groups.

    Raises:
        ValueError: An unrelated term is requested or a reviewed ancestor no longer entails Zn.
        RuntimeError: A public annotation snapshot is incomplete.
        OSError: Public acquisition or output writing fails.
    """
    if not go_terms or not set(go_terms) <= set(ZINC_NEGATIVE_TERMS):
        raise ValueError("negative_go_terms must contain only reviewed Zn-binding ancestors")

    # Exact term queries are essential: denying a more specific child activity does not
    # deny all Zn binding. Verify the broader terms against the frozen is-a ontology graph.

    namespace = f"zinc-negatives/{release_id}"
    ontology  = fetch_json(work, "https://www.ebi.ac.uk/QuickGO/services/ontology/go/terms/"
                           "GO:0008270/ancestors?relations=is_a",
                           f"{namespace}/ontology.json", root, requests_per_second)
    ancestors = set(ontology["results"][0]["ancestors"]) | {"GO:0008270"}
    if not set(go_terms) <= ancestors:
        raise ValueError("A negative GO term no longer entails Zn binding in this ontology")
    subjects: dict[str, list[dict[str, Any]]] = {}
    exclusions: list[dict[str, Any]] = []
    queried = 0
    for term in sorted(set(go_terms)):
        rows = annotations(work, {"goId": term, "goUsage": "exact",
                                  "qualifier": "NOT|enables"},
                           f"{namespace}/{term}", root, requests_per_second)
        queried += len(rows)
        for row in rows:
            subject = str(row["geneProductId"])
            reason  = None
            if row["goId"] != term or row["qualifier"] != "NOT|enables":
                reason = "not_an_exact_whole_product_denial"
            elif row["goEvidence"] not in EXPERIMENTAL_CODES or not row.get("reference"):
                reason = "not_referenced_experimental_evidence"
            elif row.get("extensions"):
                reason = "context_restricted_annotation"
            elif not subject.startswith("UniProtKB:") or "-" in subject:
                reason = "unsupported_subject_or_isoform"
            if reason:
                exclusions.append({"subject": subject, "annotation": row["id"], "reason": reason})
            else:
                subjects.setdefault(subject.split(":", 1)[1], []).append(row)
    work.log(f"Zn negatives: {queried} annotations, {len(subjects)} experimental subjects; "
             "mapping exact full sequences, never homologues")

    # Check contradicting experimental assertions before mapping. Native resume_map owns
    # restart reuse; its dependencies include the frozen GO/UniProt/coordinate responses.

    mapped = work.resume_map(
        [{"accession": accession, "annotations": rows}
         for accession, rows in sorted(subjects.items())],
        partial(map_subject, work=work, root=root, namespace=namespace,
                requests_per_second=requests_per_second),
        key=lambda item: item["accession"], workers=workers, executor="thread",
        name="zinc-negative-mapping",
    )
    candidates: dict[str, dict[str, Any]] = {}
    for result in mapped:
        # Reused map results must reconstruct the current attempt's portable source audit.
        # Fetching these release keys restores cached bytes, not a fresh public revision.

        for source in result["sources"]:
            fetch_json(work, source["url"], source["key"], root, requests_per_second)
        exclusions.extend(result["exclusions"])
        for row in result["rows"]:
            previous = candidates.get(row["identifier"])
            if previous is None:
                candidates[row["identifier"]] = row
            elif previous["sequence"] != row["sequence"]:
                raise ValueError("Conflicting exact-sequence mappings for one negative identity")
            else:
                previous["label_evidence"]["support"].extend(row["label_evidence"]["support"])
    output = root / "negative-evidence.jsonl"
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n"
                              for _, row in sorted(candidates.items())))
    audit = {
        "policy": "experimental NOT; exact canonical full-sequence mapping; no absence labels",
        "release_id": release_id, "go_terms": list(go_terms),
        "queried_annotations": queried, "experimental_subjects": len(subjects),
        "mapped_candidates": len(candidates),
        "unique_negative_sequences": len({row["sequence"] for row in candidates.values()}),
        "exclusion_counts": dict(Counter(row["reason"] for row in exclusions)),
        "exclusions": exclusions,
        "source_sha256": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sorted((root / "sources").rglob("*.json"))},
        "limitations": ["A GO denial remains conditional on experimental sensitivity and curation.",
                        "Multiple PDB chains do not multiply independent biological evidence.",
                        "Mapping and downstream quality/leakage checks may leave too few groups."],
    }
    (root / "negative-discovery.json").write_text(json.dumps(audit, indent=2))
    work.log(f"Zn negative mapping: {len(candidates)} candidates, "
             f"{audit['unique_negative_sequences']} exact sequences; see negative-discovery.json")
    return output


def map_subject(subject: Mapping[str, Any], work: lf.Work, root: Path, namespace: str,
                requests_per_second: float) -> dict[str, Any]:
    """Resolve one experimental subject to full-sequence-identical biological chain copies.

    Args:
        subject: Canonical UniProt accession and accepted experimental NOT annotations.
        work: Native cache and log owner.
        root: Managed directory retaining exact public source responses.
        namespace: Immutable acquisition release prefix.
        requests_per_second: Shared per-service download ceiling.

    Returns:
        Negative rows and explicit exclusion reasons. Tags, mutations, truncations and isoforms
        are not treated as the tested subject. No matching by sequence similarity is allowed.

    Raises:
        OSError: Native retries cannot acquire UniProt or deposited coordinates.
        ValueError: Downloaded data cannot be parsed; corrupt sources are not silently skipped.
    """
    accession = subject["accession"]
    exclusions: list[dict[str, Any]] = []
    sources: list[dict[str, str]] = []
    positive = annotations(work, {"geneProductId": f"UniProtKB:{accession}",
                                  "goId": "GO:0008270", "goUsage": "descendants"},
                           f"{namespace}/{accession}/assertions", root,
                           requests_per_second, sources)
    if any("NOT" not in row["qualifier"].split("|")
           and row["goEvidence"] in EXPERIMENTAL_CODES for row in positive):
        return {"rows": [], "sources": sources, "exclusions": [{"subject": accession,
                 "reason": "contradictory_experimental_Zn_annotation"}]}
    entry = fetch_json(work, f"https://rest.uniprot.org/uniprotkb/{accession}.json",
                       f"{namespace}/{accession}/uniprot.json", root, requests_per_second, sources)
    sequence = entry["sequence"]["value"].upper()
    pdbs     = sorted({row["id"].lower() for row in entry.get("uniProtKBCrossReferences", [])
                       if row["database"] == "PDB"})
    rows = []
    if not pdbs or "X" in sequence:
        return {"rows": [], "sources": sources, "exclusions": [{"subject": accession,
                 "reason": "no_PDB_cross_reference" if not pdbs else "ambiguous_subject_sequence"}]}
    for pdb in pdbs:
        rate   = work.cache.rate_limit("files.rcsb.org",
                                       requests_per_second=requests_per_second)
        source = work.cache.fetch(f"https://files.rcsb.org/download/{pdb.upper()}.cif.gz",
                                  key=f"structures/{pdb}.cif", decompress="gzip",
                                  retries=5, timeout=180, rate_limit=rate)
        structure = ProteinStructure(Path(source))
        matching  = []
        for chain in structure.structure[0]:
            try:
                if structure.sequence(chain) == sequence:
                    matching.append(chain.name)
            except ValueError:
                # Non-polymers and incomplete entity sequences cannot establish subject identity.

                continue
        copies: dict[str, dict[str, Any]] = {}
        for descriptor in sorted(structure.structure.assemblies, key=lambda item: item.name):
            assembly = structure.assembly(str(descriptor.name))
            for chain in assembly.protein_chains():
                if chain.name in matching:
                    copies.setdefault(chain.name, {"assembly_id": str(descriptor.name),
                                                   "protein_copy": 1})
        if not copies:
            exclusions.append({"subject": accession, "pdb": pdb,
                               "reason": "no_exact_full_sequence_in_declared_assembly"})
        for chain_name, copy in sorted(copies.items()):
            annotation = subject["annotations"][0]
            rows.append({
                "identifier": f"{pdb.upper()}_{chain_name}", "sequence": sequence, "label": 0,
                **copy, "origin": "quickgo_experimental_NOT",
                "structure_sha256": structure.sha256(),
                "label_evidence": {
                    "kind": "curated_not_annotation", "term": annotation["goId"],
                    "qualifier": "NOT", "evidence_code": annotation["goEvidence"],
                    "reference": annotation["reference"], "scope": "zinc_binding",
                    "sequence_sha256": hashlib.sha256(sequence.encode()).hexdigest(),
                    "uniprot": accession, "mapping": "exact_full_entity_sequence",
                    "support": subject["annotations"],
                },
            })
        work.log(f"Zn negative {accession}/{pdb}: {len(copies)} exact chain mappings",
                 level="debug")
    return {"rows": rows, "sources": sources, "exclusions": exclusions}
