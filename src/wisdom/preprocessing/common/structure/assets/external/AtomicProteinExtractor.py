"""Optional SLAE adapter with exact observed-atom identity and explicit coverage masks."""

import numpy as np
import lambdaforge as lf

from pathlib import Path
from tempfile import TemporaryDirectory
from wisdom.preprocessing.common.structure.assets.external.SLAEExtraction import SLAEExtraction
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.chemistry.context_chemistry import EXACT_ATOMS


class AtomicProteinExtractor:
    """Freeze one official scalar atom encoder; never infer alignment from equal row counts."""

    FAMILY = "external_atomic_slae"
    REVISION = "078b94e436b8eceb02251a254c3c224358ec6ee2"
    CHECKPOINT_SHA256 = "ac5bf9613209fc310145e58f1aa7ddc9111d833a131488fe30ea738d51f7bb99"
    ATOM_NAMES = SLAEExtraction.ATOM_NAMES

    def __init__(self, checkpoint: Path, device: str = "cpu", maximum_edges: int = 2000000,
                 radius: float = 8.0, *, work: lf.Work) -> None:
        """Preflight the native isolated SLAE runtime and verify trusted weights before mapping.

        Args:
            checkpoint: Exact official base autoencoder.ckpt; the PDB-finetuned variant is not
                substituted and would require a separate overlap/licensing decision.
            device: cpu or a granted logical CUDA device; never enumerates physical GPUs.
            maximum_edges: Hard directed radius-graph bound, default two million.
            radius: Official featurizer radius in Å, default eight; model retains its own cutoff.
            work: Bound Work owning the public tool runtime, logs, CPU/GPU grant and scratch.

        Raises:
            RuntimeError: Bootstrap has not prepared the declared slae tool runtime.
            ValueError: Wrong checkpoint or CPU request against a CUDA-only declaration.
            ToolExecutionError: Runtime imports, granted CUDA device or weight loading fail.
        """
        if ProteinArchive.sha256(checkpoint) != self.CHECKPOINT_SHA256:
            raise ValueError("SLAE checkpoint is not the reviewed, immutable base autoencoder")
        self.work = work
        self.tool = work.tools.runtime("slae")
        self.tool.preflight()
        cuda_runtime = self.tool.binding["policy"]["torch_cuda"] != "cpu"
        if cuda_runtime != device.startswith("cuda"):
            raise ValueError("atomic_device must match tool_runtimes.slae.torch_cuda; "
                             "review pyproject.toml and bootstrap, never fall back silently")
        self.command = ["python", str(self.tool.entrypoint("extract")),
                        "--checkpoint", str(checkpoint), "--device", device,
                        "--maximum-edges", str(maximum_edges), "--radius", str(radius)]
        work.log("Checking isolated SLAE imports, granted device and reviewed checkpoint")
        self.tool.run([*self.command, "--probe"], cwd=work.temp_dir, timeout=600)
        self.parameters = {"extractor": "slae", "revision": self.REVISION,
            "checkpoint_sha256": self.CHECKPOINT_SHA256, "radius": radius,
            "maximum_edges": maximum_edges, "maximum_neighbors": 1000, "dimension": 128,
            "alignment": "physical-parent-and-observed-atom37-name-v1"}

    def write(self, base: Path, output: Path) -> None:
        """Persist independently contracted frozen features, never rewrite the universal NPZ.

        Args:
            base: Immutable base defining observed atom positions and parent identities.
            output: Native LF-managed temporary asset path.
        """
        with np.load(base, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in ("atom_positions", "residue_indices",
                "atom_names", "residue_names", "atomic_numbers", "chain_indices")}
        # Scratch contains only disposable inter-process exchange. Durable resume stays at the
        # existing native checkpoints.file boundary, with full scientific validation on reuse.

        self.work.log(f"SLAE extracting {base.name}")
        with TemporaryDirectory(prefix="slae-", dir=self.work.temp_dir) as temporary:
            exchange = Path(temporary) / "features.npz"
            self.tool.run([*self.command, "--input", str(base), "--output", str(exchange)],
                          cwd=temporary, timeout=3600)
            with np.load(exchange, allow_pickle=False) as stored:
                features = {name: stored[name] for name in stored.files}
        # Coverage remains audit metadata, never an input identifier. Grouping by physical chain
        # distinguishes assembly copies; grouping by names exposes chemically selective dropout.

        coverage = {field: {str(value): {"total": int((values == value).sum()),
            "supported": int(features["valid"][values == value].sum())}
            for value in np.unique(values)} for field, values in (
                ("chain", arrays["chain_indices"]), ("atom_name", arrays["atom_names"]),
                ("residue_name", arrays["residue_names"]))}
        StructuralAsset.write(output, base, features, self.FAMILY, self.parameters,
            {"license": "Apache-2.0 upstream code; reviewed base checkpoint distribution",
             "license_reference": f"https://github.com/ProteinDesignLab/SLAE/tree/{self.REVISION}",
             "corpus_overlap": "unknown; no claim of benchmark-independent pretraining",
             "frozen": True, "units": "dimensionless scalar learned representation",
             "atom_coverage": coverage})
        self.validate(output, base)

    @staticmethod
    def validate(path: Path, base: Path) -> bool:
        """Verify point-independent physical alignment and masked scalar representation arrays.

        Args:
            path: Model-qualified frozen asset with recorded revision/checkpoint identity.
            base: Exact original physical atom/residue array namespace.

        Returns:
            True for finite [N,128]/[R,128] scalar values, correctly zeroed invalid rows and exact
            parent/name correspondence. This does not certify pretrained scientific utility.
        """
        arrays, header = StructuralAsset.read(path, base, AtomicProteinExtractor.FAMILY)
        if (header["parameters"]["checkpoint_sha256"] != AtomicProteinExtractor.CHECKPOINT_SHA256
            or header["parameters"].get("revision") != AtomicProteinExtractor.REVISION):
            raise ValueError("SLAE asset checkpoint provenance mismatch")
        with np.load(base, allow_pickle=False) as stored:
            parent = stored["residue_indices"]
            count = len(stored["atomic_numbers"])
            names = stored["atom_names"].astype(str)
            residue_names = stored["residue_names"].astype(str)
            heavy = stored["atomic_numbers"] > 1
        vocabulary = AtomicProteinExtractor.ATOM_NAMES
        slots = np.asarray([vocabulary.index(name) if name in vocabulary else -1 for name in names])
        standard = {residue for residue, _ in EXACT_ATOMS}
        valid = (slots >= 0) & heavy & np.isin(residue_names, list(standard))
        residue_valid = np.zeros(int(parent.max())+1, bool)
        residue_valid[parent[valid]] = True
        if arrays["embedding"].shape != (count, 128) or arrays["valid"].shape != (count,) or (
            not np.array_equal(arrays["atom_parent"], parent)
        ) or arrays["residue_embedding"].shape != (int(parent.max())+1, 128) or (
            arrays["embedding"].dtype.kind != "f"
        ) or arrays["residue_embedding"].dtype.kind != "f":
            raise ValueError("SLAE physical entity correspondence mismatch")
        if arrays["valid"].dtype.kind != "b" or arrays["residue_valid"].dtype.kind != "b" or (
            not np.array_equal(arrays["valid"], valid)
        ) or not np.array_equal(arrays["residue_valid"], residue_valid) or (
            not np.array_equal(arrays["atom_slot"], slots)
        ):
            raise ValueError("SLAE observed atom-name slots or physical coverage masks disagree")
        if np.any(arrays["embedding"][~arrays["valid"]]) or np.any(
            arrays["residue_embedding"][~arrays["residue_valid"]]
        ):
            raise ValueError("missing SLAE physical entities must remain explicitly masked")
        return True
