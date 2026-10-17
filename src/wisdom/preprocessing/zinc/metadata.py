"""Freeze descriptive chain annotations without letting them define benchmark labels."""

from typing import Any
from pathlib import Path
from urllib.parse import urlencode
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.zinc.negatives import fetch_json


def enrich_metadata(
    job: Mapping[str, Any], work: Any, root: Path, namespace: str,
    requests_per_second: float,
) -> dict[str, Any]:
    """Map RCSB entity annotations to author chain names and exact UniProt sequences.

    Args:
        job: Lowercase pdb_id and RAW chain rows; labels and site evidence remain unchanged.
        work: Native cache/checkpoint owner.
        root: Portable output retaining original JSON responses and observation receipts.
        namespace: Native acquisition session separating mutable public revisions.
        requests_per_second: Shared request ceiling for each public service.

    Returns:
        Enriched rows and a logical source ledger for native map-output reconstruction.
        Pfam/InterPro/CATH describe the deposited entity. Extra UniProt annotations are used
        only for identical full sequences; partial/chimeric references remain descriptive links.
        Missing annotations are empty lists, not invented functional families.

    Raises:
        OSError: Native retries cannot freeze requested public responses.
        ValueError: A response is malformed; acquisition must not silently invent metadata.
        RuntimeError: RCSB returns a partial GraphQL response with explicit errors.
    """
    pdb = str(job["pdb_id"])
    query = ('{entry(entry_id:"' + pdb.upper() + '"){polymer_entities{'
        'rcsb_polymer_entity_container_identifiers{auth_asym_ids '
        'reference_sequence_identifiers{database_name database_accession}} '
        'rcsb_entity_source_organism{ncbi_taxonomy_id} '
        'rcsb_polymer_entity_annotation{annotation_id type} '
        'polymer_entity_instances{rcsb_polymer_instance_annotation{annotation_id type}}}}}')
    url = "https://data.rcsb.org/graphql?" + urlencode({"query": query})
    sources: list[dict[str, str]] = []
    payload = fetch_json(work, url, f"zinc-metadata/{namespace}/{pdb}/rcsb.json",
                         root, requests_per_second, sources)
    if payload.get("errors"):
        raise RuntimeError(f"RCSB metadata returned incomplete GraphQL data for {pdb}: "
                           f"{payload['errors']}")
    entry = (payload.get("data") or {}).get("entry")
    entities = entry.get("polymer_entities", []) if entry else []
    by_chain = {chain: entity for entity in entities for chain in
                entity["rcsb_polymer_entity_container_identifiers"]["auth_asym_ids"]}
    output = []
    uniprot_entries: dict[str, dict[str, Any]] = {}
    for original in job["rows"]:
        row = dict(original)
        entity = by_chain.get(row["identifier"].split("_", 1)[1], {})
        identifiers = entity.get("rcsb_polymer_entity_container_identifiers", {})
        references: Sequence[Mapping[str, Any]] = identifiers.get(
            "reference_sequence_identifiers") or []
        accessions = sorted({ref["database_accession"] for ref in references
                             if ref["database_name"] == "UniProt"})
        annotations = list(entity.get("rcsb_polymer_entity_annotation") or [])
        for instance in entity.get("polymer_entity_instances") or []:
            annotations.extend(instance.get("rcsb_polymer_instance_annotation") or [])
        fields: dict[str, set[str]] = {name: set() for name in ("pfam", "interpro", "cath", "ec")}
        for annotation in annotations:
            name = annotation["type"].lower()
            if name in fields:
                fields[name].add(str(annotation["annotation_id"]))
        taxa = {str(item["ncbi_taxonomy_id"]) for item in
                entity.get("rcsb_entity_source_organism") or [] if item.get("ncbi_taxonomy_id")}

        # Canonical UniProt functions cannot be transferred across a tag, mutation, truncation
        # or chimera. RCSB entity-domain annotations retain their own, weaker descriptive scope.

        exact = []
        missing = []
        for accession in accessions:
            if accession not in uniprot_entries:
                # A retired reference may legitimately return no entry. The official search
                # response freezes that absence as JSON, without hiding transport errors.

                response = fetch_json(work,
                    "https://rest.uniprot.org/uniprotkb/search?" + urlencode({
                        "query": f"accession:{accession}", "format": "json", "size": 2}),
                    f"zinc-metadata/{namespace}/uniprot/{accession}.json", root,
                    requests_per_second, sources)
                uniprot_entries[accession] = next((item for item in response["results"]
                    if item["primaryAccession"] == accession), {})
            canonical = uniprot_entries[accession]
            if not canonical:
                missing.append(accession)
                continue
            if canonical["sequence"]["value"].upper() != row["sequence"].upper():
                continue
            exact.append(accession)
            for reference in canonical.get("uniProtKBCrossReferences", []):
                name = reference["database"].lower()
                if name in fields:
                    fields[name].add(str(reference["id"]))
            description = canonical.get("proteinDescription", {})
            names = [description.get("recommendedName", {}),
                     *description.get("alternativeNames", [])]
            for name in names:
                fields["ec"].update(str(ec["value"]) for ec in name.get("ecNumbers", []))
            if canonical.get("organism", {}).get("taxonId"):
                taxa.add(str(canonical["organism"]["taxonId"]))
        row.update({name: sorted(values) for name, values in fields.items()})
        row.update(uniprot=accessions, exact_uniprot=exact, uniprot_missing=missing,
                   taxonomy=sorted(taxa),
                   functional_metadata_scope="deposited_entity; UniProt extras exact_sequence_only",
                   functional_metadata_status="available" if entity else "unavailable",
                   functional_metadata_errors=payload.get("errors", []))
        output.append(row)
    return {"rows": output, "sources": sources}
