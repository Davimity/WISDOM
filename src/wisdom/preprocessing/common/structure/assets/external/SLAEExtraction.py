"""Standalone SLAE entrypoint staged by LF; never imports WISDOM or LambdaForge."""

import json
import torch
import hashlib
import argparse
import importlib
import numpy as np

from pathlib import Path
from scipy.spatial import cKDTree
from collections.abc import Mapping


class SLAEExtraction:
    """Load frozen official weights and map observed physical atoms to scalar representations."""

    CHECKPOINT_SHA256 = "ac5bf9613209fc310145e58f1aa7ddc9111d833a131488fe30ea738d51f7bb99"
    ATOM_NAMES = ("N", "CA", "C", "O", "CB", "OG", "CG", "CD1", "CD2", "CE1", "CE2",
                  "CZ", "OD1", "ND2", "CG1", "CG2", "CD", "CE", "NZ", "OD2", "OE1",
                  "NE2", "OE2", "OH", "NE", "NH1", "NH2", "OG1", "SD", "ND1", "SG",
                  "NE1", "CE3", "CZ2", "CZ3", "CH2", "OXT")

    def __init__(self, checkpoint: Path, device: str, maximum_edges: int = 2000000,
                 radius: float = 8.0) -> None:
        """Verify immutable weights and initialize only the official frozen scalar encoder.

        Args:
            checkpoint: Reviewed base autoencoder file, never modified or downloaded here.
            device: cpu or a logical CUDA device within the inherited provider grant.
            maximum_edges: Hard directed radius-graph budget; never truncates evidence.
            radius: Atom-pair cutoff in Å; the official encoder keeps its own inner cutoff.

        Raises:
            ValueError: Wrong checkpoint, atom vocabulary, device or missing GPU grant.
            ImportError: The isolated environment lacks its declared SLAE/PyG dependencies.
            RuntimeError: An encoder weight is missing, unexpected or has an incompatible shape.
        """
        digest = hashlib.sha256()
        with checkpoint.open("rb") as stream:
            for part in iter(lambda: stream.read(4 * 1024**2), b""):
                digest.update(part)
        if digest.hexdigest() != self.CHECKPOINT_SHA256:
            raise ValueError("SLAE checkpoint is not the reviewed, immutable base autoencoder")
        selected = torch.device(device)
        if selected.type not in {"cpu", "cuda"}:
            raise ValueError("SLAE extraction supports only explicit cpu or logical cuda devices")
        if selected.type == "cuda" and (not torch.cuda.is_available() or
                (selected.index or 0) >= torch.cuda.device_count()):
            raise ValueError("SLAE requires a granted logical CUDA device; no CPU fallback")

        # Imports happen inside this independent environment, not the parent Work interpreter.

        api       = importlib.import_module("SLAE")
        constants = importlib.import_module("SLAE.util.constants")
        graph     = importlib.import_module("torch_geometric.data")

        self.names         = tuple(constants.PROTEIN_ATOMS)
        self.residue_names = frozenset(constants.STANDARD_RESIDUE_ATOMS)
        if self.names != self.ATOM_NAMES:
            raise ValueError("SLAE runtime atom37 alphabet differs from the reviewed revision")

        # The byte-verified base checkpoint contains two retired, non-learned buffers. The pinned
        # upstream encoder recomputes normalization from each graph's average neighbor count;
        # neither buffer exists in its current state. Exclude only these exact names, then require
        # every actual encoder weight to match strictly. Never relax all state-dictionary checks.
        # Loading pickle is limited to the trusted bytes verified above, not arbitrary inputs.

        stored          = torch.load(checkpoint, map_location="cpu", weights_only=False)
        retired_buffers = {"allegro.env_sum_normalizations", "edge_eng_sum.per_edge_scales"}
        state           = {name: value for name, value in stored["encoder"].items()
                           if name not in retired_buffers}
        self.model = api.ProteinEncoder(**api.default_encoder_config()).to(selected)
        self.model.load_state_dict(state, strict=True)
        self.model.eval()
        del stored, state

        # Only the encoder is needed: do not construct or transfer the unused coordinate decoder.
        # Uninitialized lazy parameters cannot accept requires_grad_ before the first forward.
        # Inference mode below disables all gradients without modifying upstream lazy layers.

        self.encode_api    = api.encode
        self.data_type     = graph.Data
        self.device        = selected
        self.maximum_edges = maximum_edges
        self.radius        = radius

    def encode(self, arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Encode supported atoms in sorted physical order and scatter to exact parent indices.

        Args:
            arrays: Parent atom_positions float32 [N,3] in Å, residue_indices integer [N],
                atom_names/residue_names Unicode [N], and atomic_numbers integer [N].

        Returns:
            Float32 embedding [N,128] and residue_embedding [R,128], boolean valid [N] and
            residue_valid [R], int32 atom_parent [N] and int16 atom_slot [N]. Unsupported rows
            remain zero with false masks. Assembly copies retain distinct physical parents.

        Raises:
            ValueError: Ambiguous atom37 slots, unsupported neighbor count or output shape.
            MemoryError: More than maximum_edges directed pairs; no silent neighbor truncation.
        """
        parent        = arrays["residue_indices"].astype(int)
        names         = arrays["atom_names"].astype(str)
        residue_names = arrays["residue_names"].astype(str)
        slots         = np.asarray([self.names.index(n) if n in self.names else -1 for n in names])
        valid         = (slots >= 0) & (arrays["atomic_numbers"] > 1) & np.isin(
            residue_names, list(self.residue_names))
        ids = np.flatnonzero(valid)
        ids = ids[np.lexsort((slots[ids], parent[ids]))]
        if len(set(zip(parent[ids], slots[ids], strict=True))) != len(ids):
            raise ValueError("SLAE observed physical atom37 correspondence is ambiguous")

        embedding         = np.zeros((len(parent), 128), np.float32)
        residue_embedding = np.zeros((int(parent.max()) + 1, 128), np.float32)
        residue_valid     = np.zeros(len(residue_embedding), bool)
        features = {
            "embedding":         embedding,
            "valid":             valid,
            "residue_embedding": residue_embedding,
            "residue_valid":     residue_valid,
            "atom_parent":       parent.astype(np.int32),
            "atom_slot":         slots.astype(np.int16),
        }
        if not len(ids):
            return features

        # Every undirected physical pair becomes two directed messages. Count before allocating
        # the pair table; a preventive budget failure must not itself allocate an unsafe graph.

        represented, compact = np.unique(parent[ids], return_inverse=True)
        positions = arrays["atom_positions"][ids].astype(np.float32)
        tree      = cKDTree(positions)
        degrees   = tree.query_ball_point(positions, self.radius, return_length=True) - 1
        if int(degrees.sum()) > self.maximum_edges:
            raise MemoryError("SLAE radius graph exceeds the declared extraction budget")
        if degrees.max() > 1000:
            raise ValueError("SLAE graph exceeds the official 1000-neighbor support; no truncation")
        pairs = tree.query_pairs(self.radius, output_type="ndarray").T
        edges = np.concatenate((pairs, pairs[::-1]), axis=1)
        data  = self.data_type(
            pos=torch.from_numpy(positions),
            atom_type=torch.from_numpy(slots[ids]).long(),
            atom37_type=torch.from_numpy(slots[ids]).long(),
            residue_index=torch.from_numpy(compact).long(),
            batch=torch.zeros(len(ids), dtype=torch.long),
            residue_batch=torch.zeros(len(represented), dtype=torch.long),
            edge_index=torch.from_numpy(edges).long(),
        ).to(self.device)
        with torch.inference_mode():
            result = self.encode_api(self.model, data)
        atom    = result["node_embedding"].float().cpu().numpy()
        residue = result["residue_embedding"].float().cpu().numpy()
        if atom.shape != (len(ids), 128) or residue.shape != (len(represented), 128):
            raise ValueError("official SLAE changed its scalar atom/residue output contract")
        embedding[ids]                 = atom
        residue_embedding[represented] = residue
        residue_valid[represented]     = True
        return features

    @staticmethod
    def main() -> None:
        """Execute a probe or one explicit NPZ-to-NPZ extraction without framework imports.

        Command arguments select checkpoint/device, radius in Å, directed-edge budget, and
        optional input/output paths. A probe loads weights but creates no scientific file.
        Extraction reads only universal atomic fields, never targets or split membership.

        Raises:
            ValueError: Missing extraction paths or non-finite/object result arrays.
            RuntimeError: Upstream loading or inference failed; the native tool logs the traceback.
        """
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--checkpoint", type=Path, required=True)
        parser.add_argument("--device", required=True)
        parser.add_argument("--maximum-edges", type=int, default=2000000)
        parser.add_argument("--radius", type=float, default=8.0)
        parser.add_argument("--input", type=Path)
        parser.add_argument("--output", type=Path)
        parser.add_argument("--probe", action="store_true")
        args = parser.parse_args()
        extractor = SLAEExtraction(args.checkpoint, args.device, args.maximum_edges, args.radius)
        if args.probe:
            print(json.dumps({"slae": "ready", "device": args.device}), flush=True)
            return
        if args.input is None or args.output is None:
            raise ValueError("SLAE extraction requires --input and --output or --probe")
        print(f"SLAE encoding {args.input.name} on {args.device}", flush=True)
        with np.load(args.input, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in ("atom_positions", "residue_indices",
                      "atom_names", "residue_names", "atomic_numbers")}
        features = extractor.encode(arrays)
        if any(value.dtype.kind == "O" or (value.dtype.kind in "fc" and not
               np.isfinite(value).all()) for value in features.values()):
            raise ValueError("SLAE produced non-finite or object arrays")
        with args.output.open("wb") as stream:
            np.savez(stream, **features)  # type: ignore[arg-type]
        print(f"SLAE encoded {int(features['valid'].sum())}/{len(features['valid'])} atoms",
              flush=True)


if __name__ == "__main__":
    SLAEExtraction.main()
