"""Explicit local-only frozen protein language models behind one residue-token interface."""

import json
import torch
import hashlib
import importlib
import numpy as np

from typing import Any
from pathlib import Path
from wisdom.preprocessing.common.structure.assets.external.ExternalModelKind import (
    ExternalModelKind,
)
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive


class ProteinRepresentationExtractor:
    """Load optional official model implementations only when explicitly requested."""

    def __init__(
        self, kind: str, checkpoint: Path, identifier: str, revision: str,
        layer: int = -1, maximum_length: int = 1022, device: str = "cpu",
    ) -> None:
        """Load a researcher-staged checkpoint; never download weights or substitute models.

        Args:
            kind: esm2 (35M), esmc (300M), or saprot (35M).
            checkpoint: Local complete pretrained directory including tokenizer/config/weights.
            identifier: Model repository identifier retained as evidence, not a network selector.
            revision: Researcher-staged immutable source revision.
            layer: Hidden-state index; -1 selects the final representation.
            maximum_length: Positive residue chunk limit excluding BOS/EOS; chunks are disjoint
                and processed independently. This loses cross-chunk context and is recorded.
            device: Explicit PyTorch device for this optional extraction, default CPU.

        Raises:
            ValueError: Missing identity, unknown family, incompatible checkpoint or tokenization.
            ImportError: Native Transformers dependencies are absent; no downloads are attempted.
        """
        kind = ExternalModelKind(kind).value
        if not checkpoint.is_dir() \
                or not identifier or not revision or maximum_length < 1:
            raise ValueError("external extraction needs a supported model and local checkpoint")
        entries = [(path.relative_to(checkpoint).as_posix(), ProteinArchive.sha256(path))
                   for path in sorted(checkpoint.rglob("*")) if path.is_file()]
        fingerprint = hashlib.sha256(json.dumps(entries).encode()).hexdigest()
        try:
            transformers = importlib.import_module("transformers")
        except ImportError as error:
            raise ImportError("Frozen extraction requires the current Transformers dependency; "
                              "run ./install.sh locally or lf clusters bootstrap <cluster> "
                              "--project . remotely. No checkpoint will be downloaded") from error
        self.model: Any
        if kind == "esmc":
            if not hasattr(transformers, "EsmcForMaskedLM"):
                raise ImportError("ESM-C requires Transformers>=5.20.2 with native Esmc support")
            loaded = transformers.EsmcForMaskedLM.from_pretrained(
                str(checkpoint), local_files_only=True, dtype=torch.float32,
                attn_implementation="sdpa", trust_remote_code=False)
            self.model = loaded.esmc
            self.tokenizer = transformers.EsmcTokenizer.from_pretrained(
                str(checkpoint), local_files_only=True, trust_remote_code=False)
        else:
            self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(checkpoint), local_files_only=True, trust_remote_code=False)
            loaded = transformers.AutoModelForMaskedLM.from_pretrained(
                str(checkpoint), local_files_only=True, trust_remote_code=False,
                dtype=torch.float32)
            self.model = loaded.esm
        self.model.eval().to(device)
        self.model.requires_grad_(False)
        self.kind = kind
        self.layer = layer
        self.maximum_length = maximum_length
        self.device = device
        self.dimension = int(getattr(self.model.config, "hidden_size",
                                     getattr(self.model.config, "d_model", 0)))
        expected = 960 if kind == "esmc" else 480
        if self.dimension != expected:
            raise ValueError(f"{kind} checkpoint width is not the requested 35M/300M variant")
        expected_layers = 30 if kind == "esmc" else 12
        if int(self.model.config.num_hidden_layers) != expected_layers:
            raise ValueError(f"{kind} checkpoint depth is not the requested 35M/300M variant")
        self.metadata = {"extractor": kind, "identifier": identifier, "revision": revision,
            "extractor_version": "1.1", "input_type": "AA+3Di" if kind == "saprot" else "AA",
            "vocabulary_sha256": hashlib.sha256(json.dumps(self.tokenizer.get_vocab(),
                sort_keys=True).encode()).hexdigest(),
            "checkpoint_sha256": fingerprint, "checkpoint_inventory": entries,
            "layer": layer, "maximum_length": maximum_length,
            "dimension": self.dimension, "dtype": "float32", "frozen": True,
            "chunk_policy": "independent-disjoint-fragments-v1",
            "tokenizer_class": type(self.tokenizer).__name__,
            "transformers_version": transformers.__version__,
            "license_reference": f"https://huggingface.co/{identifier}/blob/{revision}/README.md",
            "license": "verify the staged checkpoint model card; code and weight terms can differ"}

    def encode(self, tokens: list[str]) -> np.ndarray:
        """Return contextual vectors in exactly the caller's residue-token order.

        Args:
            tokens: One amino acid per token for ESM, or official AA+lowercase-3Di pairs for SaProt.

        Returns:
            Float32 [R,d] final/selected hidden states; no global pooling or gradient graph.

        Raises:
            ValueError: A tokenizer splits/merges residues or uses unknown tokens unexpectedly.
        """
        chunks = []
        vocabulary = self.tokenizer.get_vocab()
        with torch.inference_mode():
            for start in range(0, len(tokens), self.maximum_length):
                subset = tokens[start:start + self.maximum_length]
                if any(token not in vocabulary for token in subset):
                    raise ValueError("external tokenizer lacks the exact residue/3Di token")
                ids = self.tokenizer.convert_tokens_to_ids(subset)
                if len(ids) != len(subset):
                    raise ValueError("external tokenization changes residue correspondence")
                bos, eos = self.tokenizer.cls_token_id, self.tokenizer.eos_token_id
                if bos is None or eos is None:
                    raise ValueError("external tokenizer lacks the verified CLS/EOS convention")
                ids = [bos, *ids, eos]
                inputs = torch.tensor([ids], dtype=torch.long, device=self.device)
                output = self.model(input_ids=inputs, output_hidden_states=True)
                hidden = output.hidden_states[self.layer]
                if hidden.ndim != 3 or hidden.shape[1] != len(subset) + 2:
                    raise ValueError("external model BOS/EOS or hidden-state contract mismatch")
                chunks.append(hidden[0, 1:-1].float().cpu().numpy())
        return np.concatenate(chunks) if chunks else np.empty((0, self.dimension), np.float32)

    def static_table(self) -> tuple[np.ndarray, list[str]]:
        """Export actual input weights, independent of contextual outputs or WISDOM projection.

        Returns:
            Float32 [20,d] and canonical amino-acid order. SaProt prototypes average every
            present official 3Di-state input embedding equally, excluding unknown/mask states.

        Raises:
            ValueError: The checkpoint does not expose compatible token input embeddings.
        """
        alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        states = list("acdefghiklmnpqrstvwy")
        weights = self.model.get_input_embeddings().weight.detach().float().cpu()
        vocabulary = self.tokenizer.get_vocab()
        rows = []
        for amino in alphabet:
            tokens = [amino + state for state in states] if self.kind == "saprot" else [amino]
            if any(token not in vocabulary for token in tokens):
                raise ValueError("checkpoint lacks the required static prototype vocabulary")
            rows.append(weights[[vocabulary[token] for token in tokens]].mean(dim=0).numpy())
        return np.stack(rows), alphabet
