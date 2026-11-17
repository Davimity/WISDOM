"""Verified physical residue identities and observed/deposited sequence correspondences."""

import gzip
import json
import gemmi
import hashlib
import numpy as np

from pathlib import Path
from typing import ClassVar
from collections.abc import Mapping
from wisdom.utils.structure.models.Protein import Protein
from wisdom.utils.structure.enums.AtomRole import AtomRole
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.io.SourceEvidenceLevel import SourceEvidenceLevel


class SequenceContext:
    """Separate physical residue ordinals, author addresses and sequence token positions."""

    FAMILY = "sequence_context"
    parameters: ClassVar[dict[str, str]] = {}

    def build(
        self, arrays: Mapping[str, np.ndarray], protein: Protein | None = None,
        source: Path | None = None, context: Mapping[str, np.ndarray] | None = None,
        model_index: int = 0,
    ) -> dict[str, np.ndarray]:
        """Build compact residue/chain tables without guessing missing sequence positions.

        Args:
            arrays: Validated schema-3 atom identities, ordinal parents, roles and covalent edges.
            protein: Optional source hierarchy already matched to every base atom.
            source: Its verified coordinate source. Only deposited label_seq/entity tables are
                used for complete sequence mapping; repeated amino acids are never aligned by type.
            context: Optional verified AtomicContext arrays retaining historical author evidence.
            model_index: Source coordinate model retained by the immutable base, default zero.

        Returns:
            Residue arrays [R], chain strings/offsets [C]/[C+1], flattened observed and complete
            token strings. Missing indices are -1 with explicit masks. All physical residues,
            including retained ligands/metals, keep a row, but only protein polymers enter tokens.

        Raises:
            ValueError: Nonconsecutive ordinals, conflicting source/sequence or author evidence.
        """
        parents = arrays["residue_indices"]
        count = int(parents.max()) + 1
        unique, first = np.unique(parents, return_index=True)
        if not np.array_equal(unique, np.arange(count)):
            raise ValueError("sequence context requires consecutive physical residue indices")
        chain = arrays["chain_indices"][first].astype(np.int32)
        chains = int(chain.max()) + 1
        names = arrays["residue_names"][first].astype(str)
        polymer = np.zeros(count, np.bool_)
        polymer_known = np.zeros(count, np.bool_)
        for i in range(count):
            roles = arrays["atom_role_ids"][parents == i]
            polymer[i] = bool(np.any(np.isin(roles, [AtomRole.BACKBONE, AtomRole.SIDECHAIN])))
            polymer_known[i] = bool(np.any(roles != AtomRole.UNKNOWN))
        author_number = np.zeros(count, np.int32)
        insertion = np.full(count, "", dtype="U4")
        author_valid = np.zeros(count, np.bool_)
        chain_names = np.full(chains, "", dtype="U64")
        if context is not None:
            author_number[:] = context["author_number"]
            insertion[:] = context["insertion_code"]
            author_valid[:] = context["author_valid"]
            polymer_known[author_valid] = True
            polymer[author_valid] = context["is_polymer"][author_valid]
            chain_names[:] = context["chain_author_id"]
        if protein is not None:
            residues = [r for c in protein.chains for r in c.residues]
            if len(residues) != count or [r.name for r in residues] != list(names):
                raise ValueError("sequence source residue correspondence mismatch")
            if context is not None and np.any(author_valid & (
                (author_number != [r.number for r in residues]) |
                (insertion != [r.insertion_code for r in residues])
            )):
                raise ValueError("sequence source conflicts with verified historical author IDs")
            polymer[:] = [r.is_polymer for r in residues]
            polymer_known[:] = True
            author_number[:] = [r.number for r in residues]
            insertion[:] = [r.insertion_code for r in residues]
            author_valid[:] = True
            chain_names[:] = [c.id for c in protein.chains]
        if protein is not None and any(len(chain.id) > 64 for chain in protein.chains):
            raise ValueError("sequence chain identifier exceeds the compact string contract")

        structure = None
        if source is not None:
            payload = source.read_bytes()
            if payload.startswith(b"\x1f\x8b"):
                payload = gzip.decompress(payload)
            structure = gemmi.read_structure_string(payload)
            structure.setup_entities()

        # Gemmi's component vocabulary distinguishes amino-acid polymers from water, nucleic
        # acids and ligands. Nonstandard mapped letters remain explicitly identified in base names.

        info = [gemmi.find_tabulated_residue(name) for name in names]
        protein_polymer = polymer & np.asarray([entry.is_amino_acid() for entry in info])
        if structure is not None:
            for c in range(chains):
                source_chain = structure[model_index].find_chain(str(chain_names[c]))
                entity = (structure.get_entity_of(source_chain.get_polymer())
                          if source_chain else None)
                if entity is not None and entity.polymer_type in (
                    gemmi.PolymerType.PeptideL, gemmi.PolymerType.PeptideD
                ):
                    protein_polymer[chain == c] = polymer[chain == c]
        letters = np.asarray([entry.one_letter_code.upper() if entry.is_amino_acid()
                              else "X" for entry in info], dtype="U1")
        letters[~np.isin(letters, list("ACDEFGHIKLMNPQRSTVWYOU"))] = "X"
        observed_index = np.full(count, -1, np.int32)
        local = np.zeros(count, np.int32)
        observed: list[str] = []
        observed_ptr = [0]
        for c in range(chains):
            rows = np.flatnonzero(chain == c)
            local[rows] = np.arange(len(rows))
            for row in rows[protein_polymer[rows]]:
                observed_index[row] = len(observed)
                observed.append(str(letters[row]))
            observed_ptr.append(len(observed))

        # Full sequences require deposited label_seq_id, not a guessed alignment from author
        # numbers or repeated letters. Unavailable complete chains stay empty, not observed-as-full.

        complete: list[str] = []
        complete_ptr = [0]
        complete_index = np.full(count, -1, np.int32)
        for c in range(chains):
            rows = np.flatnonzero((chain == c) & protein_polymer)
            raw_chain = next((candidate for candidate in structure[model_index]
                if candidate.name == chain_names[c]), None) if structure else None
            sequence = []
            if raw_chain is not None:
                assert structure is not None
                entity = structure.get_entity_of(raw_chain.get_polymer())
                if entity is not None and entity.full_sequence:
                    # Gemmi's sequence convenience conversion returns X for several modified
                    # components (MSE/PTR/PCA/LLP), while its residue dictionary knows their
                    # parent amino acid. Use that same dictionary on BOTH sides of the join;
                    # never repair a true mismatch by aligning a similar sequence instead.

                    components = [gemmi.find_tabulated_residue(name)
                                  for name in entity.full_sequence]
                    sequence = [component.one_letter_code.upper() if component.is_amino_acid()
                                else "X" for component in components]
                    sequence = [letter if letter in "ACDEFGHIKLMNPQRSTVWYOU" else "X"
                                for letter in sequence]
                addresses = {(r.seqid.num, r.seqid.icode.strip(), r.name): r.label_seq
                             for r in raw_chain if r.entity_type == gemmi.EntityType.Polymer}
                for row in rows:
                    address = (int(author_number[row]), str(insertion[row]), names[row])
                    label = addresses.get(address)
                    if label is None or not sequence or not author_valid[row]:
                        continue
                    position = int(label) - 1
                    if not 0 <= position < len(sequence) or sequence[position] != letters[row]:
                        token = (sequence[position] if 0 <= position < len(sequence)
                                 else "out-of-range")
                        raise ValueError("deposited sequence disagrees with a physical residue: "
                            f"chain={chain_names[c]} author={author_number[row]}"
                            f"{insertion[row]} component={names[row]} observed={letters[row]} "
                            f"label_seq_id={label} deposited={token}")
                    complete_index[row] = len(complete) + position
            complete.extend(sequence)
            complete_ptr.append(len(complete))

        # Fragments follow verified C--N covalent edges, not adjacent residue ordinals. An
        # unresolved break therefore cannot silently become a continuous Transformer sequence.

        segment = np.full(count, -1, np.int32)
        links: dict[int, set[int]] = {int(i): set() for i in np.flatnonzero(protein_polymer)}
        edges = arrays["atom_edge_index"][:, arrays["atom_edge_is_covalent"].astype(bool)].T
        atom_names = arrays["atom_names"]
        for a, b in edges:
            left, right = int(parents[a]), int(parents[b])
            if left != right and left in links and right in links and chain[left] == chain[right] \
                    and {str(atom_names[a]), str(atom_names[b])} == {"C", "N"}:
                links[left].add(right)
                links[right].add(left)
        next_segment = 0
        for row in links:
            if segment[row] >= 0:
                continue
            pending = [row]
            segment[row] = next_segment
            while pending:
                for neighbor in links[pending.pop()]:
                    if segment[neighbor] < 0:
                        segment[neighbor] = next_segment
                        pending.append(neighbor)
            next_segment += 1
        return {
            "residue_chain": chain, "residue_local_index": local,
            "canonical_letter": letters, "canonical_valid": protein_polymer & (letters != "X"),
            "is_protein_polymer": protein_polymer, "polymer_valid": polymer_known,
            "author_number": author_number, "insertion_code": insertion,
            "author_valid": author_valid, "chain_author_id": chain_names,
            "observed_index": observed_index, "observed_ptr": np.asarray(observed_ptr, np.int64),
            "observed_tokens": np.asarray(observed, dtype="U1"),
            "complete_index": complete_index, "complete_valid": complete_index >= 0,
            "complete_ptr": np.asarray(complete_ptr, np.int64),
            "complete_tokens": np.asarray(complete, dtype="U1"), "segment_index": segment,
        }

    def write(
        self, base: Path, output: Path, protein: Protein | None = None,
        source: Path | None = None, context: Mapping[str, np.ndarray] | None = None,
        quality: str = "unavailable",
        source_selection: Mapping[str, object] | None = None,
    ) -> None:
        """Persist independently verified identity, without copying atom arrays.

        Args:
            base: Exact universal archive.
            output: LF-managed temporary output.
            protein: Verified source hierarchy or None.
            source: Verified portable source file or None.
            context: Previously verified author evidence or None.
            quality: exact/equivalent/unavailable source-recovery status.
            source_selection: Optional declared assembly/copy provenance; direct frame matching
                is tried first, and only declared operations may recover a different frame.
        """
        quality = SourceEvidenceLevel(quality).value
        with np.load(base, allow_pickle=False) as stored:
            arrays = {key: stored[key] for key in stored.files if key != "metadata_json"}
            metadata = json.loads(str(stored["metadata_json"].item()))
        values = self.build(arrays, protein, source, context,
                            int(metadata.get("model_index", 0)))
        StructuralAsset.write(output, base, values, self.FAMILY, {}, {
            "algorithm": "gemmi-deposited-label-seq-and-observed-fragments-v1",
            "source_quality": quality, "missing_index": -1,
            "declared_source_selection": dict(source_selection or {}),
            "verified_source_sha256": hashlib.sha256(source.read_bytes()).hexdigest()
                if source is not None else None,
            "author_evidence_sha256": self.author_evidence_hash(context),
            "sequence_policy": "observed != complete; no ordinal/letter alignment",
            "monomer_normalization": "Gemmi residue dictionary parent letters on both sides",
            "coverage": {"residues": len(values["residue_chain"]),
                         "full_mapped": int(values["complete_valid"].sum())},
        })

    def validate(self, path: Path, base: Path) -> bool:
        """Audit physical parent tables, token ranges, masks and deposited mapping identity.

        Args:
            path: Persisted sequence context.
            base: Exact immutable archive defining residue names/ordinal parents.

        Returns:
            True after all correspondence and unavailable-value checks pass.

        Raises:
            ValueError: A residue crosses chains or is associated with an inconsistent token.
        """
        values, header = StructuralAsset.read(path, base, self.FAMILY)
        if header["parameters"] or header.get("algorithm") != (
            "gemmi-deposited-label-seq-and-observed-fragments-v1"
        ):
            raise ValueError("sequence context algorithm/configuration mismatch")
        integer32 = ("residue_chain", "residue_local_index", "author_number", "observed_index",
                     "complete_index", "segment_index")
        booleans = ("canonical_valid", "is_protein_polymer", "polymer_valid", "author_valid",
                    "complete_valid")
        if any(values[key].dtype != np.int32 for key in integer32) or any(
            values[key].dtype != np.bool_ for key in booleans
        ) or any(values[key].dtype != np.int64 for key in ("observed_ptr", "complete_ptr")):
            raise ValueError("sequence context index/mask dtype mismatch")
        with np.load(base, allow_pickle=False) as stored:
            _, first = np.unique(stored["residue_indices"], return_index=True)
            chain = stored["chain_indices"][first]
        if not np.array_equal(values["residue_chain"], chain):
            raise ValueError("sequence context residue/chain correspondence mismatch")
        count = len(chain)
        for key in ("canonical_letter", "author_number", "author_valid", "is_protein_polymer",
                    "observed_index", "complete_index", "complete_valid", "segment_index"):
            if values[key].shape != (count,):
                raise ValueError(f"sequence context invalid residue shape: {key}")
        for kind in ("observed", "complete"):
            pointer, tokens, indices = (values[f"{kind}_ptr"], values[f"{kind}_tokens"],
                                        values[f"{kind}_index"])
            if len(pointer) != int(chain.max()) + 2 or pointer[0] != 0 \
                    or pointer[-1] != len(tokens) or np.any(np.diff(pointer) < 0):
                raise ValueError("sequence context chain pointers are inconsistent")
            valid = indices >= 0
            if np.any(indices < -1) or np.any(indices[valid] >= len(tokens)):
                raise ValueError("sequence context token index out of range")
            if np.any(valid & ~values["is_protein_polymer"]):
                raise ValueError("non-protein residue cannot have a sequence token")
            if np.any(indices[valid] < pointer[chain[valid]]) or np.any(
                indices[valid] >= pointer[chain[valid] + 1]
            ) or not np.array_equal(tokens[indices[valid]], values["canonical_letter"][valid]):
                raise ValueError("sequence context residue-token identity/chain mismatch")
        if not np.array_equal(values["complete_valid"], values["complete_index"] >= 0):
            raise ValueError("sequence context full-mapping mask mismatch")
        unknown_author = ~values["author_valid"]
        if np.any(values["author_number"][unknown_author] != 0) or np.any(
            values["insertion_code"][unknown_author] != ""
        ):
            raise ValueError("unavailable author identities require masked empty placeholders")
        polymer = values["is_protein_polymer"]
        if not np.array_equal(values["observed_index"][polymer], np.arange(polymer.sum())):
            raise ValueError("observed sequence must preserve every physical polymer residue")
        for c in np.unique(chain):
            if not np.array_equal(values["residue_local_index"][chain == c],
                                  np.arange(np.count_nonzero(chain == c))):
                raise ValueError("sequence context local residue order mismatch")
            indices = values["complete_index"][(chain == c) & values["complete_valid"]]
            if np.any(np.diff(indices) <= 0):
                raise ValueError("complete sequence mapping is ambiguous or reorders residues")
        return True

    @staticmethod
    def author_evidence_hash(context: Mapping[str, np.ndarray] | None) -> str | None:
        """Fingerprint only historical identity fields actually consumed by sequence mapping.

        Args:
            context: Validated AtomicContext arrays or None; occupancy/torsions are irrelevant.

        Returns:
            Stable identity digest or None. Changing optional experimental quality cannot
            invalidate a sequence mapping when author addresses/polymer evidence are unchanged.
        """
        if context is None:
            return None
        fields = {name: context[name].tolist() for name in (
            "author_number", "insertion_code", "author_valid", "is_polymer", "chain_author_id")}
        return hashlib.sha256(json.dumps(fields, sort_keys=True).encode()).hexdigest()
