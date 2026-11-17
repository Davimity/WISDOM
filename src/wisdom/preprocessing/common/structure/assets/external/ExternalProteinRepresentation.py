"""Versioned per-residue frozen representations and independently audited static input tables."""

import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.context.SequenceContext import SequenceContext
from wisdom.preprocessing.common.structure.assets.external.ProteinRepresentationExtractor import (
    ProteinRepresentationExtractor,
)


class ExternalProteinRepresentation:
    """Join token outputs to physical residues; never expand frozen embeddings per atom."""

    def __init__(
        self, extractor: ProteinRepresentationExtractor, sequence_policy: str = "fragments",
    ) -> None:
        """Bind a frozen extractor independently of every WISDOM architecture/hyperparameter.

        Args:
            extractor: Initialized official local model or contract-compatible test double.
            sequence_policy: fragments uses observed peptide-connected runs; complete uses verified
                deposited full chains and masks structural residues without exact label_seq mapping.
        """
        if sequence_policy not in {"fragments", "complete"}:
            raise ValueError("external sequence policy must be fragments or complete")
        if extractor.kind == "saprot" and sequence_policy != "fragments":
            raise ValueError("SaProt requires observed structural tokens, not invented full 3Di")
        self.extractor = extractor
        self.parameters = {**extractor.metadata, "sequence_policy": sequence_policy}

    def write(
        self, base: Path, output: Path, sequence: Path,
        structure_tokens: Path | None = None, static: bool = False, family: str | None = None,
    ) -> None:
        """Persist contextual residue vectors or a separate static amino-acid input table.

        Args:
            base: Immutable structural input.
            output: LF-managed temporary output.
            sequence: Validated residue-to-token correspondence asset.
            structure_tokens: Required official Foldseek token asset for SaProt.
            static: Export actual input-weight prototypes instead of contextual outputs.
            family: Optional model-qualified asset name; None preserves the legacy family name.
        """
        SequenceContext().validate(sequence, base)
        context, _ = StructuralAsset.read(sequence, base, "sequence_context")
        if static:
            weights, alphabet = self.extractor.static_table()
            values = {"weights": weights, "alphabet": np.asarray(alphabet)}
            default_family = "static_embedding_table"
        else:
            count = len(context["residue_chain"])
            embeddings = np.zeros((count, self.extractor.dimension), np.float32)
            valid = np.zeros(count, np.bool_)
            token_position = np.full(count, -1, np.int32)
            tokens = context["canonical_letter"].astype(str)
            eligible = context["is_protein_polymer"]
            if self.extractor.kind == "saprot":
                if structure_tokens is None:
                    raise ValueError("SaProt requires an official Foldseek structure_tokens asset")
                structural, _ = StructuralAsset.read(structure_tokens, base, "structure_tokens")
                tokens = np.asarray([a + b for a, b in
                                     zip(tokens, structural["token"], strict=True)])
                eligible = eligible & structural["valid"]
            if self.parameters["sequence_policy"] == "complete":
                for chain in range(len(context["complete_ptr"]) - 1):
                    start, stop = context["complete_ptr"][chain:chain + 2]
                    rows = np.flatnonzero((context["residue_chain"] == chain)
                                          & context["complete_valid"] & eligible)
                    full = list(map(str, context["complete_tokens"][start:stop]))
                    if full:
                        encoded = self.extractor.encode(full)
                        positions = context["complete_index"][rows] - start
                        embeddings[rows] = encoded[positions]
                        token_position[rows] = positions
                        valid[rows] = True
            else:
                for segment in np.unique(context["segment_index"][eligible]):
                    if segment < 0:
                        continue
                    rows = np.flatnonzero((context["segment_index"] == segment) & eligible)
                    embeddings[rows] = self.extractor.encode(list(map(str, tokens[rows])))
                    token_position[rows] = np.arange(len(rows))
                    valid[rows] = True
            # Official 3Di can be unavailable for a short/incomplete chain. Retain a fully masked
            # SaProt asset rather than invent structural tokens or silently drop that protein.

            if not valid.any() and self.extractor.kind != "saprot":
                raise ValueError("external extraction has no verified residue/token correspondence")
            values = {"embedding": embeddings, "valid": valid, "token_position": token_position,
                      "residue_chain": context["residue_chain"]}
            default_family = "external_embeddings"
        if family not in {None, default_family, f"{default_family}_{self.extractor.kind}"}:
            raise ValueError("external representation family disagrees with the selected model")
        dependencies: dict[str, Any] = {"sequence_sha256": hashlib.sha256(
            sequence.read_bytes()).hexdigest()}
        if structure_tokens:
            dependencies["structure_tokens_sha256"] = hashlib.sha256(
                structure_tokens.read_bytes()).hexdigest()
        StructuralAsset.write(output, base, values, family or default_family, self.parameters,
                              {"algorithm": "frozen-residue-token-join-v1", **dependencies})

    @staticmethod
    def validate(path: Path, base: Path, static: bool = False, family: str | None = None) -> bool:
        """Validate frozen-representation shape, masks and checkpoint identity without loading it.

        Args:
            path: Persisted contextual or static table.
            base: Exact universal NPZ.
            static: Validate a static table rather than physical residue embeddings.
            family: Model-qualified family or None for historical single-model assets.

        Returns:
            True after numerical, fingerprint and residue support checks.

        Raises:
            ValueError: Bad checkpoint fingerprint, mask, dimensions or residue correspondence.
        """
        default_family = "static_embedding_table" if static else "external_embeddings"
        family = family or default_family
        values, header = StructuralAsset.read(path, base, family)
        params = header["parameters"]
        if family not in {default_family, f"{default_family}_{params.get('extractor')}"}:
            raise ValueError("external representation family/model identity mismatch")
        if header.get("algorithm") != "frozen-residue-token-join-v1" or len(header.get(
            "sequence_sha256", "")) != 64:
            raise ValueError("external representation lacks its exact sequence dependency")
        if len(params.get("checkpoint_sha256", "")) != 64 or not params.get("frozen"):
            raise ValueError("external representation lacks a frozen checkpoint identity")
        if static:
            if values["weights"].shape != (20, params["dimension"]) or not np.array_equal(
                values["alphabet"], list("ACDEFGHIKLMNPQRSTVWY")
            ) or values["weights"].dtype != np.float32:
                raise ValueError("static input table shape mismatch")
        else:
            with np.load(base, allow_pickle=False) as stored:
                count = int(stored["residue_indices"].max()) + 1
                _, first = np.unique(stored["residue_indices"], return_index=True)
                chain = stored["chain_indices"][first]
            if values["embedding"].shape != (count, params["dimension"]) or not np.array_equal(
                chain, values["residue_chain"]
            ) or values["valid"].shape != (count,) or values["valid"].dtype != np.bool_:
                raise ValueError("external embedding physical residue correspondence mismatch")
            if (values["embedding"].dtype != np.float32 or
                    values["token_position"].dtype != np.int32):
                raise ValueError("external embedding numeric/index dtype mismatch")
            if np.any(values["embedding"][~values["valid"]] != 0) or np.any(
                (values["token_position"] < 0) != ~values["valid"]
            ):
                raise ValueError("external unavailable representation mask mismatch")
        return True
