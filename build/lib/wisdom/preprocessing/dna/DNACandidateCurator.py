"""Scientific verification and inexpensive profiling of DNA benchmark candidates."""

from __future__ import annotations

import gzip
import hashlib
import urllib.parse
from collections.abc import Mapping
from itertools import pairwise
from pathlib import Path
from typing import Any

import gemmi
import numpy as np
from scipy.spatial import cKDTree

from wisdom.preprocessing.dna.DNAEnums import DatasetTier, DNALabel, EvidenceKind
from wisdom.preprocessing.dna.PublicDataClient import PublicDataClient
from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace


class DNACandidateCurator:
    """Verify labels from evidence and biological-assembly atom contacts."""

    SEARCH_URL  = "https://search.rcsb.org/rcsbsearch/v2/query"
    DATA_URL    = "https://data.rcsb.org/graphql"
    QUICKGO_URL = "https://www.ebi.ac.uk/QuickGO/services/annotation/search"
    FILE_URL    = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

    AMINO_KINDS = frozenset(
        {
            gemmi.ResidueKind.AA,
            gemmi.ResidueKind.AAD,
            gemmi.ResidueKind.PAA,
            gemmi.ResidueKind.MAA,
        }
    )

    def __init__(
        self,
        contact_distance          : float = 4.5,
        minimum_residues          : int = 30,
        minimum_interface_residues: int = 2,
        challenge_aspect_ratio    : float = 4.0,
        minimum_sequence_coverage : float = 0.80,
        maximum_resolution        : float = 4.0,
        cache_output              : str = "discovery-cache",
        requests_per_second       : float = 2.0,
        retries                   : int = 4,
    ) -> None:
        """Configure physically interpretable acceptance and profiling thresholds.

        A positive protein chain requires at least one pair of non-hydrogen atoms, one from the
        selected protein and one from DNA in the same biological assembly, separated by at most
        ``contact_distance`` ångströms. The default 4.5 Å captures direct and close interfacial
        contacts while excluding DNA merely present elsewhere in the assembly. This geometric test
        verifies positive evidence; it never manufactures a negative label.

        Args:
            contact_distance: Maximum protein-heavy-atom to DNA-heavy-atom centre distance in Å.
            minimum_residues: Minimum number of amino-acid residues in an accepted protein chain.
            minimum_interface_residues: Minimum distinct contacting residues for a positive row.
            challenge_aspect_ratio: Principal-axis extent ratio above which a row enters the
                challenge tier rather than being silently filtered.
            minimum_sequence_coverage: Minimum observed/source residue ratio for a mapped chain.
            maximum_resolution: Largest accepted crystallographic/cryo-EM resolution in Å.
            cache_output: Named LambdaForge output used for mapped experimental structures.
            requests_per_second: Positive mean request-rate limit for RCSB and QuickGO.
            retries: Positive bounded public-service attempt count.

        Raises:
            ValueError: If a distance or count cannot define the documented rules.
        """
        if contact_distance <= 0.0 or minimum_residues < 1 or minimum_interface_residues < 1:
            raise ValueError("contact and minimum-size thresholds must be positive")
        if challenge_aspect_ratio <= 1.0:
            raise ValueError("challenge_aspect_ratio must exceed one")
        if not 0.0 < minimum_sequence_coverage <= 1.0 or maximum_resolution <= 0.0:
            raise ValueError("structure coverage and resolution bounds are invalid")
        if not cache_output.strip():
            raise ValueError("cache_output cannot be empty")

        self.contact_distance           = float(contact_distance)
        self.minimum_residues           = minimum_residues
        self.minimum_interface_residues = minimum_interface_residues
        self.challenge_aspect_ratio     = float(challenge_aspect_ratio)
        self.minimum_sequence_coverage  = float(minimum_sequence_coverage)
        self.maximum_resolution         = float(maximum_resolution)
        self.cache_output               = cache_output
        self.client                     = PublicDataClient(requests_per_second, retries)
        self.biolip_identifiers         : frozenset[str] | None = None

    def transform(
        self,
        record : ProcessingRecord,
        context: ProcessingWorkspace,
    ) -> ProcessingRecord:
        """Assign an evidence state, verify contacts, and calculate cheap descriptors.

        Gemmi reads the biological assembly. A sparse KD-tree radius query finds all DNA atoms
        within ``contact_distance`` of each protein atom without allocating a dense ``N x D``
        distance matrix. Principal-axis extents come from the eigenvalues of the coordinate
        covariance matrix and radius of gyration is ``sqrt(mean_i ||x_i-centroid||²)``.

        Args:
            record: Candidate mapping containing structure, chain, evidence, and cluster provenance.
            context: LambdaForge task context; unused because all paths are in the candidate record.

        Returns:
            Record containing a complete catalog row plus explicit inclusion/exclusion reasons.

        Raises:
            TypeError: If the candidate value is not a mapping.
            OSError: If the assembly cannot be read.
            ValueError: If coordinates or declared chains are malformed.
        """
        if not isinstance(record.value, Mapping):
            raise TypeError("DNA candidate value must be a mapping")
        candidate = dict(record.value)
        if not candidate.get("structure_path"):
            try:
                candidate = self._map_public_candidate(candidate, context)
            except (OSError, RuntimeError, ValueError, KeyError, TypeError) as error:
                return record.with_value(self._excluded(record, candidate, str(error)))

        structure_path = Path(str(candidate["structure_path"])).resolve()
        structure      = gemmi.read_structure(str(structure_path))
        if not structure:
            raise ValueError(f"{structure_path.name} contains no coordinate model")

        model         = structure[0]
        protein_chain = str(candidate["protein_chain"])
        chain         = model.find_chain(protein_chain)
        if chain is None:
            raise ValueError(f"protein chain {protein_chain!r} is absent from the assembly")

        protein_atoms, residue_ids, sequence = self._protein_atoms(chain)

        # Contact validation compares every author-declared biological assembly and retains the
        # strongest reproducible interface. Geometry still stores the selected deposited chain, so
        # assembly copies never become independent examples. Copies with the same observed amino-
        # acid sequence are pooled solely for this physical positive-evidence test.
        contact_models = [
            (
                str(assembly.name or index + 1),
                gemmi.make_assembly(
                    assembly,
                    model,
                    gemmi.HowToNameCopiedChain.AddNumber,
                ),
            )
            for index, assembly in enumerate(structure.assemblies)
        ] or [("asymmetric_unit", model)]
        assembly_results: list[
            tuple[int, int, str, list[tuple[float, float, float]], list[str], set[int]]
        ] = []
        for candidate_assembly_id, contact_model in contact_models:
            contact_protein_atoms: list[tuple[float, float, float]] = []
            contact_residue_ids  : list[int]                       = []
            residue_offset = 0
            for assembly_chain in contact_model:
                assembly_atoms, assembly_residues, assembly_sequence = self._protein_atoms(
                    assembly_chain
                )
                if assembly_sequence != sequence:
                    continue
                contact_protein_atoms.extend(assembly_atoms)
                contact_residue_ids.extend(value + residue_offset for value in assembly_residues)
                residue_offset += len(assembly_sequence)
            assembly_dna, assembly_dna_chains = self._dna_atoms(
                contact_model,
                None if structure.assemblies else candidate.get("dna_chains"),
            )
            pairs, residues_in_contact = self._contacts(
                contact_protein_atoms or protein_atoms,
                contact_residue_ids or residue_ids,
                assembly_dna,
            )
            assembly_results.append(
                (
                    len(residues_in_contact),
                    pairs,
                    candidate_assembly_id,
                    assembly_dna,
                    assembly_dna_chains,
                    residues_in_contact,
                )
            )
        (
            _,
            contact_pairs,
            assembly_id,
            dna_atoms,
            observed_dna_chains,
            interface_residues,
        ) = max(assembly_results, key=lambda value: (value[0], value[1], value[2]))
        asymmetric_dna, _ = self._dna_atoms(model, candidate.get("dna_chains"))
        _, deposited_interface_residues = self._contacts(
            protein_atoms,
            residue_ids,
            asymmetric_dna,
        )
        region_sizes, interface_spread = self._interface_profile(
            protein_atoms,
            residue_ids,
            deposited_interface_residues,
        )

        evidence = {
            value.value if isinstance(value, EvidenceKind) else str(value)
            for value in candidate.get("evidence", ())
        }
        positive_assertion = bool(
            evidence
            & {
                EvidenceKind.BIOLOGICAL_ASSEMBLY_CONTACT.value,
                EvidenceKind.CURATED_DNA_BINDING.value,
            }
        )
        negative_assertion = EvidenceKind.CURATED_NOT_DNA_BINDING.value in evidence
        source_binding_mask = str(candidate.get("binding_site_mask", ""))
        verified_positive  = (
            positive_assertion
            and (
                (bool(dna_atoms) and len(interface_residues) >= self.minimum_interface_residues)
                or (len(source_binding_mask) == len(sequence) and "1" in source_binding_mask)
            )
        )

        # Contradictory assertions are quarantined before quality filtering or split assignment.
        if negative_assertion and (positive_assertion or contact_pairs > 0):
            label  = DNALabel.CONFLICT
            reason = "explicit negative evidence conflicts with positive evidence or DNA contact"
        elif verified_positive:
            label  = DNALabel.POSITIVE
            reason = (
                "positive evidence verified by biological-assembly protein-DNA contact"
                if contact_pairs
                else "positive evidence and residue labels supplied by pinned DyProL release"
            )
        elif negative_assertion:
            label  = DNALabel.NEGATIVE
            reason = "explicit curated non-DNA-binding evidence"
        else:
            label  = DNALabel.UNKNOWN
            reason = (
                "positive assertion lacked a verified interface"
                if positive_assertion
                else "absence of DNA/contact is not evidence of non-DNA-binding"
            )

        residue_count = len(sequence)
        exclusion      = None
        if len(protein_chain) != 1 or not protein_chain.isalnum():
            exclusion = (
                "protein chain cannot be represented by the universal PDB_CHAIN selector syntax"
            )
        if residue_count < self.minimum_residues:
            exclusion = f"protein has {residue_count} residues; minimum is {self.minimum_residues}"
        if not protein_atoms:
            exclusion = "protein chain contains no finite heavy-atom coordinates"
        source_sequence = str(candidate.get("sequence", sequence)).upper()
        sequence_coverage = len(sequence) / max(1, len(source_sequence))
        if sequence_coverage < self.minimum_sequence_coverage:
            exclusion = (
                f"observed sequence coverage {sequence_coverage:.3f} is below "
                f"{self.minimum_sequence_coverage:.3f}"
            )
        if source_binding_mask and len(source_binding_mask) != len(sequence):
            exclusion = "DyProL binding-site mask does not align with observed structure residues"
        if source_binding_mask and source_sequence != sequence:
            exclusion = "DyProL source sequence does not exactly match observed structure residues"
        cluster_id = str(candidate.get("sequence_cluster_id", "")).strip()

        profile = self._profile(np.asarray(protein_atoms, dtype=np.float64))
        tier    = (
            DatasetTier.CHALLENGE
            if profile["aspect_ratio"] >= self.challenge_aspect_ratio
            else DatasetTier.CORE
        )
        source_hash = candidate.get("structure_sha256")
        if not source_hash:
            source_hash = hashlib.sha256(structure_path.read_bytes()).hexdigest()
        coordinate_bytes = np.asarray(protein_atoms, dtype="<f4").tobytes()
        protein_structure_hash = hashlib.sha256(
            sequence.encode("ascii") + coordinate_bytes
        ).hexdigest()
        reported_length = int(candidate.get("reported_sequence_length") or residue_count)
        missing_fraction = max(0.0, (reported_length - residue_count) / max(1, reported_length))
        protein_chain_count = sum(
            any(
                gemmi.find_tabulated_residue(residue.name).kind in self.AMINO_KINDS
                for residue in item
            )
            for item in model
        )

        value: dict[str, Any] = {
            "candidate_key": record.key,
            "base_identifier": f"{str(candidate['pdb_id']).upper()}_{protein_chain}",
            "pdb_id": str(candidate["pdb_id"]).upper(),
            "assembly_id": assembly_id,
            "protein_chain": protein_chain,
            "dna_chains": observed_dna_chains,
            "structure_path": str(structure_path),
            "structure_url": self.FILE_URL.format(pdb_id=str(candidate["pdb_id"]).upper()),
            "structure_sha256": str(source_hash),
            "sequence": sequence,
            "canonical_sequence": sequence,
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "sequence_cluster_id": cluster_id,
            "label_conflict_cluster_id": str(
                candidate.get("label_conflict_cluster_id", cluster_id)
            ),
            "protein_structure_sha256": protein_structure_hash,
            "label_state": label.value,
            "label": 1 if label is DNALabel.POSITIVE else 0 if label is DNALabel.NEGATIVE else None,
            "label_reason": reason,
            "positive_assertion": positive_assertion,
            "evidence": sorted(evidence),
            "evidence_sources": list(candidate.get("evidence_sources", ())),
            "included": exclusion is None and label in {DNALabel.POSITIVE, DNALabel.NEGATIVE},
            "exclusion_reason": exclusion,
            "tier": tier.value,
            "source_database": str(candidate.get("source_database", "")),
            "source_record": str(candidate.get("source_record", "")),
            "source_version": str(candidate.get("source_version", "")),
            "source_url": str(candidate.get("source_url", "")),
            "source_checksum": str(candidate.get("source_checksum", "")),
            "published_partition": str(candidate.get("published_partition", "development")),
            "query_version": str(candidate.get("query_version", "RCSB Search/Data API v2")),
            "query_date_utc": str(candidate.get("query_date_utc", "")),
            "uniprot_id": str(candidate.get("uniprot_id", "")),
            "uniprot_ids": list(candidate.get("uniprot_ids", ())),
            "negative_source": candidate.get("negative_source"),
            "negative_source_label": candidate.get("negative_source_label"),
            "negative_confidence": candidate.get("negative_confidence"),
            "no_positive_uniprot_annotation": candidate.get("no_positive_uniprot_annotation"),
            "explicit_negative_annotation": candidate.get("explicit_negative_annotation"),
            "no_known_pdb_dna_complex": candidate.get("no_known_pdb_dna_complex"),
            "no_biolip_dna_binding": candidate.get("no_biolip_dna_binding"),
            "homology_filter_threshold": candidate.get("homology_filter_threshold", 0.30),
            "local_gt_expected": bool(
                candidate.get("local_gt_expected", label is DNALabel.NEGATIVE)
            ),
            "local_gt_method": (
                "binding_residue_mask"
                if source_binding_mask
                else "dna_distance"
                if verified_positive and dna_atoms
                else "global_negative"
                if label is DNALabel.NEGATIVE
                else "none"
            ),
            "binding_residue_indices": [
                index for index, flag in enumerate(source_binding_mask) if flag == "1"
            ],
            "reference_complex_pdb_id": (
                str(candidate["pdb_id"]).upper() if verified_positive else None
            ),
            "experimental_method": candidate.get("experimental_method"),
            "structure_method": candidate.get("experimental_method"),
            "resolution_angstrom": candidate.get("resolution_angstrom"),
            "initial_release_date": candidate.get("initial_release_date"),
            "taxonomy_name": candidate.get("taxonomy_name"),
            "taxonomy": candidate.get("taxonomy_name"),
            "taxonomy_id": candidate.get("taxonomy_id"),
            "functional_description": candidate.get("functional_description"),
            "total_chain_count": len(model),
            "protein_chain_count": protein_chain_count,
            "residue_count": residue_count,
            "sequence_length": residue_count,
            "reported_sequence_length": reported_length,
            "missing_residue_fraction": missing_fraction,
            "sequence_coverage": sequence_coverage,
            "atom_count": len(protein_atoms),
            "dna_atom_count": len(dna_atoms),
            "contact_pair_count": contact_pairs,
            "interface_residue_count": len(interface_residues),
            "interface_residue_fraction": len(interface_residues) / max(1, residue_count),
            "interface_region_count": len(region_sizes),
            "interface_region_sizes": region_sizes,
            "largest_interface_region": max(region_sizes, default=0),
            "smallest_interface_region": min(region_sizes, default=0),
            "largest_interface_region_fraction": (
                max(region_sizes, default=0) / max(1, len(interface_residues))
            ),
            "interface_spread_angstrom": interface_spread,
            **profile,
        }
        return record.with_value(value)

    def _map_public_candidate(
        self,
        candidate: dict[str, Any],
        context  : ProcessingWorkspace,
    ) -> dict[str, Any]:
        """Map a public sequence record to one deterministic experimental PDB chain.

        DyProL already supplies a PDB-chain identity. BTD-Combo supplies an anonymized sequence, so
        the RCSB sequence service finds exact polymer-entity matches. Negative candidates are
        rejected if any exact match belongs to a DNA-containing PDB entry, if BioLiP records a DNA
        interaction for the mapped UniProt accession, or if QuickGO returns a non-negated DNA-
        binding annotation. The absence checks strengthen an existing published negative label;
        they never manufacture one.

        Args:
            candidate: Source record with class, sequence, partition, and provenance.
            context: LambdaForge context locating the shared public-data cache.

        Returns:
            Candidate extended with selected structure, chain, UniProt, and RCSB cluster metadata.

        Raises:
            RuntimeError: If no defensible experimental mapping survives or a conflict is found.
            ValueError: If the source class or supplied PDB-chain identity is invalid.
        """
        source_class = str(candidate.get("source_class", ""))
        if source_class == "positive":
            pdb_id = str(candidate.get("pdb_id", "")).upper()
            chain  = str(candidate.get("protein_chain", ""))
            if len(pdb_id) != 4 or len(chain) != 1:
                raise ValueError("DyProL candidate requires one PDB ID and one protein chain")
            structure_path = self._structure(pdb_id, context)
            structure      = gemmi.read_structure(str(structure_path))
            positive_chain = structure[0].find_chain(chain)
            if positive_chain is None:
                raise RuntimeError(f"DyProL chain {pdb_id}_{chain} is absent from RCSB coordinates")
            entity = structure.get_entity_of(positive_chain.get_polymer())
            if entity is None:
                raise RuntimeError(f"DyProL chain {pdb_id}_{chain} has no RCSB polymer entity")
            metadata = self._metadata(pdb_id, entity.name)
            resolution = metadata.get("resolution_angstrom")
            if resolution is not None and float(resolution) > self.maximum_resolution:
                raise RuntimeError(
                    f"structure resolution {float(resolution):.2f} Å exceeds "
                    f"{self.maximum_resolution:.2f} Å"
                )
            candidate.update(metadata)
            candidate["assembly_id"]    = "asymmetric_unit"
            candidate["structure_path"] = str(structure_path)
            candidate["structure_sha256"] = hashlib.sha256(
                structure_path.read_bytes()
            ).hexdigest()
            candidate["dna_chains"] = self._dna_chain_names(structure[0])
            return candidate

        if source_class != "negative":
            raise ValueError(f"unsupported public source class {source_class!r}")
        entities = self._sequence_entities(str(candidate["sequence"]))
        if not entities:
            raise RuntimeError("no exact experimental RCSB polymer-entity mapping")

        mapped: list[dict[str, Any]] = []
        known_dna_complex = False
        for identifier in entities:
            pdb_id, separator, entity_id = identifier.partition("_")
            if not separator:
                continue
            metadata = self._metadata(pdb_id, entity_id)
            if metadata.get("canonical_sequence") != str(candidate["sequence"]).upper():
                continue
            if int(metadata.get("entry_dna_polymer_count") or 0) > 0:
                known_dna_complex = True
            mapped.append(metadata)
        if known_dna_complex:
            raise RuntimeError(
                "negative rejected because an exact sequence has a known PDB DNA complex"
            )
        if not mapped:
            raise RuntimeError("RCSB sequence hits did not provide an exact full-sequence mapping")

        # An auditable negative needs UniProt identity before functional conflict checks can pass.
        mapped = [value for value in mapped if value.get("uniprot_ids")]
        if not mapped:
            raise RuntimeError("negative has no UniProt mapping for annotation conflict checks")
        biolip_path = Path(str(candidate["biolip_dna_uniprot_path"]))
        if self.biolip_identifiers is None:
            self.biolip_identifiers = frozenset(
                value.strip() for value in biolip_path.read_text().splitlines() if value.strip()
            )
        for metadata in mapped:
            accessions = {str(value) for value in metadata["uniprot_ids"]}
            if accessions & self.biolip_identifiers:
                raise RuntimeError("negative rejected because BioLiP records a DNA interaction")
            annotations = [self._quickgo(value) for value in sorted(accessions)]
            if any(result[0] for result in annotations):
                raise RuntimeError("negative rejected because QuickGO records DNA-binding function")
            metadata["explicit_negative_annotation"] = any(
                result[1] for result in annotations
            )

        # Prefer complete, high-resolution structures; stable identifiers break every tie.
        ranked = sorted(
            mapped,
            key=lambda value: (
                self._method_rank(str(value.get("experimental_method") or "")),
                float(value.get("resolution_angstrom") or float("inf")),
                str(value["pdb_id"]),
                str(value["entity_id"]),
            ),
        )
        for metadata in ranked:
            resolution = metadata.get("resolution_angstrom")
            if resolution is not None and float(resolution) > self.maximum_resolution:
                continue
            pdb_id = str(metadata["pdb_id"])
            structure_path = self._structure(pdb_id, context)
            structure      = gemmi.read_structure(str(structure_path))
            for chain_name in metadata["protein_chains"]:
                if len(chain_name) != 1 or not str(chain_name).isalnum():
                    continue
                negative_chain = structure[0].find_chain(str(chain_name))
                if negative_chain is None:
                    continue
                _, _, observed = self._protein_atoms(negative_chain)
                coverage = len(observed) / max(1, len(str(candidate["sequence"])))
                if coverage < self.minimum_sequence_coverage:
                    continue
                selected_candidate = dict(candidate)
                selected_candidate.update(metadata)
                selected_candidate.update(
                    {
                        "pdb_id": pdb_id,
                        "assembly_id": "asymmetric_unit",
                        "protein_chain": str(chain_name),
                        "dna_chains": [],
                        "structure_path": str(structure_path),
                        "structure_sha256": hashlib.sha256(
                            structure_path.read_bytes()
                        ).hexdigest(),
                        "uniprot_id": sorted(str(value) for value in metadata["uniprot_ids"])[0],
                        "negative_confidence": "high",
                        "no_positive_uniprot_annotation": True,
                        "no_known_pdb_dna_complex": True,
                        "no_biolip_dna_binding": True,
                    }
                )
                return selected_candidate
        raise RuntimeError("no mapped experimental structure passed quality and coverage filters")

    def _sequence_entities(self, sequence: str) -> tuple[str, ...]:
        """Return exact experimental RCSB polymer entities for one benchmark sequence.

        Args:
            sequence: Canonical amino-acid sequence.

        Returns:
            Sorted identifiers in ``PDB_ENTITY`` form.

        Raises:
            RuntimeError: If RCSB returns an invalid result set.
        """
        query: dict[str, Any] = {
            "query": {
                "type": "terminal",
                "service": "sequence",
                "parameters": {
                    "evalue_cutoff": 1.0e-10,
                    "identity_cutoff": 1.0,
                    "target": "pdb_protein_sequence",
                    "value": sequence,
                },
            },
            "return_type": "polymer_entity",
            "request_options": {
                "paginate": {"start": 0, "rows": 100},
                "results_content_type": ["experimental"],
            },
        }
        identifiers: set[str] = set()
        start                   = 0
        while True:
            query["request_options"]["paginate"]["start"] = start
            payload = self.client.json(self.SEARCH_URL, query)
            results = payload.get("result_set")
            if results is None:
                break
            if not isinstance(results, list):
                raise RuntimeError("RCSB sequence response has an invalid result_set")
            identifiers.update(
                str(value["identifier"]).upper()
                for value in results
                if isinstance(value, Mapping) and value.get("identifier")
            )
            start += len(results)
            total = int(payload.get("total_count", start))
            if not results or start >= total:
                break
        return tuple(sorted(identifiers))

    def _metadata(self, pdb_id: str, entity_id: str) -> dict[str, Any]:
        """Read structure quality, identity, chain, UniProt, and MMseqs2 metadata.

        Args:
            pdb_id: Four-character RCSB entry ID.
            entity_id: Polymer entity ID within the entry.

        Returns:
            Flat mapping used for deterministic structure choice and leakage control.

        Raises:
            RuntimeError: If required entity, chain, or 30%/90% clusters are absent.
        """
        query = {
            "query": (
                "query($entry:String!,$entity:String!){"
                "entry(entry_id:$entry){exptl{method}rcsb_entry_info{resolution_combined "
                "polymer_entity_count_DNA}rcsb_accession_info{initial_release_date}}"
                "polymer_entity(entry_id:$entry,entity_id:$entity){"
                "entity_poly{pdbx_seq_one_letter_code_can rcsb_sample_sequence_length}"
                "rcsb_polymer_entity_container_identifiers{auth_asym_ids "
                "reference_sequence_identifiers{database_name database_accession}}"
                "rcsb_polymer_entity_group_membership{aggregation_method group_id "
                "similarity_cutoff}"
                "rcsb_polymer_entity{pdbx_description}"
                "rcsb_entity_source_organism{ncbi_scientific_name ncbi_taxonomy_id}}}"
            ),
            "variables": {"entry": pdb_id.upper(), "entity": str(entity_id)},
        }
        payload = self.client.json(self.DATA_URL, query)
        try:
            data        = payload["data"]
            entity      = data["polymer_entity"]
            entry       = data["entry"]
            identifiers = entity["rcsb_polymer_entity_container_identifiers"]
            memberships = entity["rcsb_polymer_entity_group_membership"] or []
            cluster_30  = next(
                value
                for value in memberships
                if value.get("aggregation_method") == "sequence_identity"
                and float(value["similarity_cutoff"]) == 30.0
            )
            cluster_90 = next(
                value
                for value in memberships
                if value.get("aggregation_method") == "sequence_identity"
                and float(value["similarity_cutoff"]) == 90.0
            )
        except (KeyError, TypeError, StopIteration, ValueError) as error:
            raise RuntimeError(f"RCSB metadata is incomplete for {pdb_id}_{entity_id}") from error

        references = identifiers.get("reference_sequence_identifiers") or []
        uniprot_ids = sorted(
            {
                str(value["database_accession"])
                for value in references
                if value.get("database_name") == "UniProt" and value.get("database_accession")
            }
        )
        organisms   = entity.get("rcsb_entity_source_organism") or []
        organism    = organisms[0] if organisms and isinstance(organisms[0], Mapping) else {}
        methods     = entry.get("exptl") or []
        method      = methods[0].get("method") if methods else None
        resolutions = entry.get("rcsb_entry_info", {}).get("resolution_combined") or []
        sequence    = "".join(
            str(entity.get("entity_poly", {}).get("pdbx_seq_one_letter_code_can") or "").split()
        ).upper()
        return {
            "pdb_id": pdb_id.upper(),
            "entity_id": str(entity_id),
            "protein_chains": sorted(
                str(value) for value in identifiers.get("auth_asym_ids") or []
            ),
            "uniprot_ids": uniprot_ids,
            "uniprot_id": uniprot_ids[0] if uniprot_ids else "",
            "canonical_sequence": sequence,
            "sequence_cluster_id": f"rcsb-mmseqs2-30:{cluster_30['group_id']}",
            "label_conflict_cluster_id": f"rcsb-mmseqs2-90:{cluster_90['group_id']}",
            "experimental_method": method,
            "resolution_angstrom": float(resolutions[0]) if resolutions else None,
            "entry_dna_polymer_count": int(
                entry.get("rcsb_entry_info", {}).get("polymer_entity_count_DNA") or 0
            ),
            "reported_sequence_length": entity.get("entity_poly", {}).get(
                "rcsb_sample_sequence_length"
            ),
            "functional_description": entity.get("rcsb_polymer_entity", {}).get(
                "pdbx_description"
            ),
            "taxonomy_name": organism.get("ncbi_scientific_name"),
            "taxonomy_id": organism.get("ncbi_taxonomy_id"),
            "initial_release_date": entry.get("rcsb_accession_info", {}).get(
                "initial_release_date"
            ),
        }

    def _quickgo(self, uniprot_id: str) -> tuple[bool, bool]:
        """Check positive and explicit-negative DNA-binding GO annotations.

        Args:
            uniprot_id: Canonical UniProt accession mapped by RCSB.

        Returns:
            ``(positive_annotation, explicit_NOT_annotation)`` for GO:0003677 descendants.

        Raises:
            RuntimeError: If QuickGO returns a malformed annotation collection.
        """
        parameters = urllib.parse.urlencode(
            {
                "geneProductId": f"UniProtKB:{uniprot_id}",
                "goId": "GO:0003677",
                "goUsage": "descendants",
                "limit": 100,
            }
        )
        payload = self.client.json(f"{self.QUICKGO_URL}?{parameters}")
        results = payload.get("results", [])
        if not isinstance(results, list):
            raise RuntimeError("QuickGO returned an invalid annotation list")
        qualifiers = {
            str(value.get("qualifier", ""))
            for value in results
            if isinstance(value, Mapping)
        }
        positive = any(value and "NOT" not in value.upper() for value in qualifiers)
        explicit_negative = any("NOT" in value.upper() for value in qualifiers)
        return positive, explicit_negative

    def _structure(self, pdb_id: str, context: ProcessingWorkspace) -> Path:
        """Download and validate one experimental RCSB mmCIF entry atomically.

        Args:
            pdb_id: Four-character PDB entry identifier.
            context: LambdaForge context resolving the cache output.

        Returns:
            Absolute path to an uncompressed Gemmi-readable mmCIF file.

        Raises:
            RuntimeError: If compressed bytes or parsed coordinates are invalid.
        """
        root     = context.output(self.cache_output) / "structures"
        gz_path  = self.client.download(
            self.FILE_URL.format(pdb_id=pdb_id.upper()),
            root / f"{pdb_id.lower()}.cif.gz",
        )
        cif_path = root / f"{pdb_id.lower()}.cif"
        if not cif_path.is_file():
            self.client.atomic_write(cif_path, gzip.decompress(gz_path.read_bytes()))
        structure = gemmi.read_structure(str(cif_path))
        if not structure:
            raise RuntimeError(f"RCSB structure {pdb_id} contains no coordinate model")
        return cif_path.resolve()

    @classmethod
    def _dna_chain_names(cls, model: gemmi.Model) -> list[str]:
        """List exact model chains that contain at least one DNA residue.

        Args:
            model: First Gemmi coordinate model.

        Returns:
            Sorted DNA chain identifiers.
        """
        return sorted(
            chain.name
            for chain in model
            if any(
                gemmi.find_tabulated_residue(residue.name).kind is gemmi.ResidueKind.DNA
                for residue in chain
            )
        )

    @staticmethod
    def _method_rank(method: str) -> int:
        """Rank experimental methods for deterministic negative structure selection.

        Args:
            method: RCSB experimental-method description.

        Returns:
            Lower-is-better rank: X-ray, electron microscopy, NMR, then other experiments.
        """
        normalized = method.upper()
        if "X-RAY" in normalized:
            return 0
        if "ELECTRON" in normalized:
            return 1
        if "NMR" in normalized:
            return 2
        return 3

    @staticmethod
    def _excluded(
        record   : ProcessingRecord,
        candidate: Mapping[str, Any],
        reason   : str,
    ) -> dict[str, Any]:
        """Represent one failed mapping as auditable exclusion instead of a task error.

        Args:
            record: Stable LambdaForge source record.
            candidate: Original public-source candidate fields.
            reason: Exact mapping, annotation, conflict, or structure rejection reason.

        Returns:
            JSON-compatible row retained by the sink's exclusion report.
        """
        sequence = str(candidate.get("sequence", ""))
        return {
            "candidate_key": record.key,
            "base_identifier": record.key,
            "label": None,
            "label_state": DNALabel.UNKNOWN.value,
            "label_reason": reason,
            "included": False,
            "exclusion_reason": reason,
            "evidence": list(candidate.get("evidence", ())),
            "evidence_sources": list(candidate.get("evidence_sources", ())),
            "source_database": str(candidate.get("source_database", "")),
            "source_record": str(candidate.get("source_record", "")),
            "source_version": str(candidate.get("source_version", "")),
            "source_url": str(candidate.get("source_url", "")),
            "source_checksum": str(candidate.get("source_checksum", "")),
            "published_partition": str(candidate.get("published_partition", "")),
            "query_version": "RCSB Search/Data API v2 and QuickGO REST",
            "query_date_utc": str(candidate.get("query_date_utc", "")),
            "sequence_sha256": hashlib.sha256(sequence.encode("ascii")).hexdigest(),
            "sequence_length": len(sequence),
            "local_gt_expected": bool(candidate.get("local_gt_expected", False)),
            "negative_source": candidate.get("negative_source"),
            "negative_source_label": candidate.get("negative_source_label"),
        }

    def _protein_atoms(
        self,
        chain: gemmi.Chain,
    ) -> tuple[list[tuple[float, float, float]], list[int], str]:
        """Extract finite heavy atoms and one-letter sequence from an amino-acid chain.

        Args:
            chain: Gemmi chain selected as the protein candidate.

        Returns:
            Heavy-atom coordinates, matching zero-based residue IDs, and one-letter sequence.
        """
        coordinates: list[tuple[float, float, float]] = []
        residue_ids : list[int]                      = []
        sequence    : list[str]                      = []
        for residue in chain:
            info = gemmi.find_tabulated_residue(residue.name)
            if info.kind not in self.AMINO_KINDS:
                continue
            residue_index = len(sequence)
            sequence.append(info.one_letter_code or "X")
            for atom in residue:
                position = (atom.pos.x, atom.pos.y, atom.pos.z)
                if atom.element.atomic_number > 1 and np.isfinite(position).all():
                    coordinates.append(position)
                    residue_ids.append(residue_index)
        return coordinates, residue_ids, "".join(sequence)

    @staticmethod
    def _dna_atoms(
        model          : gemmi.Model,
        declared_chains: Any,
    ) -> tuple[list[tuple[float, float, float]], list[str]]:
        """Extract finite DNA heavy atoms while excluding RNA and unrelated ligands.

        Args:
            model: Gemmi biological assembly model.
            declared_chains: Optional discovery-time DNA chain identifiers to verify.

        Returns:
            DNA heavy-atom coordinates and sorted observed DNA chain identifiers.
        """
        declared   = {str(value) for value in declared_chains or ()}
        coordinates: list[tuple[float, float, float]] = []
        observed   : list[str]                        = []
        for chain in model:
            if declared and chain.name not in declared:
                continue
            chain_atoms: list[tuple[float, float, float]] = []
            for residue in chain:
                if gemmi.find_tabulated_residue(residue.name).kind is not gemmi.ResidueKind.DNA:
                    continue
                for atom in residue:
                    position = (atom.pos.x, atom.pos.y, atom.pos.z)
                    if atom.element.atomic_number > 1 and np.isfinite(position).all():
                        chain_atoms.append(position)
            if chain_atoms:
                observed.append(chain.name)
                coordinates.extend(chain_atoms)
        return coordinates, sorted(observed)

    def _contacts(
        self,
        protein_atoms : list[tuple[float, float, float]],
        residue_ids   : list[int],
        dna_atoms     : list[tuple[float, float, float]],
    ) -> tuple[int, set[int]]:
        """Count sparse protein-DNA atom contacts and the contacting protein residues.

        Args:
            protein_atoms: Protein heavy-atom coordinates in Å.
            residue_ids: Protein residue owner for every protein atom.
            dna_atoms: DNA heavy-atom coordinates in Å.

        Returns:
            Number of atom pairs within the cutoff and the set of contacting residue indices.
        """
        if not protein_atoms or not dna_atoms:
            return 0, set()
        neighbors = cKDTree(np.asarray(dna_atoms)).query_ball_point(
            np.asarray(protein_atoms),
            self.contact_distance,
        )
        pair_count = sum(len(values) for values in neighbors)
        residues   = {residue_ids[index] for index, values in enumerate(neighbors) if values}
        return pair_count, residues

    @staticmethod
    def _interface_profile(
        protein_atoms     : list[tuple[float, float, float]],
        residue_ids       : list[int],
        interface_residues: set[int],
    ) -> tuple[list[int], float]:
        """Estimate connected interface regions and their spatial spread cheaply.

        Contacting residues become graph nodes. Two nodes are joined when they are consecutive in
        sequence or any pair of their heavy atoms lies within 8 Å. The radius query remains sparse
        and captures spatially contiguous patches without constructing the WISDOM surface.

        Args:
            protein_atoms: Protein heavy-atom coordinates in Å.
            residue_ids: Residue owner of every coordinate.
            interface_residues: Residue indices with at least one DNA contact.

        Returns:
            Descending component sizes in residues and the radius of gyration of all interface
            heavy atoms in Å. Empty interfaces return ``([], 0.0)``.
        """
        if not interface_residues:
            return [], 0.0
        coordinates = np.asarray(protein_atoms, dtype=np.float64)
        owners      = np.asarray(residue_ids, dtype=np.int64)
        mask        = np.isin(owners, list(interface_residues))
        points      = coordinates[mask]
        point_owner = owners[mask]
        adjacency: dict[int, set[int]] = {residue: set() for residue in interface_residues}
        for first, second in cKDTree(points).query_pairs(8.0):
            left  = int(point_owner[first])
            right = int(point_owner[second])
            if left != right:
                adjacency[left].add(right)
                adjacency[right].add(left)
        ordered = sorted(interface_residues)
        for left, right in pairwise(ordered):
            if right == left + 1:
                adjacency[left].add(right)
                adjacency[right].add(left)

        remaining = set(interface_residues)
        sizes: list[int] = []
        while remaining:
            seed      = remaining.pop()
            component = {seed}
            frontier  = [seed]
            while frontier:
                neighbors = adjacency[frontier.pop()] & remaining
                remaining.difference_update(neighbors)
                component.update(neighbors)
                frontier.extend(neighbors)
            sizes.append(len(component))
        centered = points - points.mean(axis=0)
        spread   = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        return sorted(sizes, reverse=True), spread

    @staticmethod
    def _profile(coordinates: np.ndarray) -> dict[str, float]:
        """Compute translation-invariant global shape descriptors from protein atoms.

        Args:
            coordinates: Finite heavy-atom coordinates with shape ``[N,3]`` in Å.

        Returns:
            Radius of gyration, three principal extents, aspect ratio, and coordinate span.
        """
        if coordinates.size == 0:
            return {
                "radius_of_gyration": 0.0,
                "principal_extent_1": 0.0,
                "principal_extent_2": 0.0,
                "principal_extent_3": 0.0,
                "aspect_ratio": 0.0,
                "coordinate_span": 0.0,
            }
        centered     = coordinates - coordinates.mean(axis=0)
        gyration     = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
        covariance   = centered.T @ centered / max(1, len(centered))
        eigenvalues  = np.maximum(np.linalg.eigvalsh(covariance), 0.0)
        extents      = 2.0 * np.sqrt(eigenvalues[::-1])
        aspect_ratio = float(extents[0] / max(extents[-1], 1e-8))
        span         = float(np.linalg.norm(coordinates.max(axis=0) - coordinates.min(axis=0)))
        return {
            "radius_of_gyration": gyration,
            "principal_extent_1": float(extents[0]),
            "principal_extent_2": float(extents[1]),
            "principal_extent_3": float(extents[2]),
            "aspect_ratio": aspect_ratio,
            "coordinate_span": span,
        }
