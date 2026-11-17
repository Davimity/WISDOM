"""Run-scoped, label-free structural codes for the explicitly bounded V9g global encoder."""

import torch
import inspect
import hashlib
import numpy as np
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from zipfile import BadZipFile
from collections.abc import Iterable, Mapping
from lambdaforge.nn import GraphStructuralEncoding


class AtomicStructuralInputs:
    """Prepare fixed graph codes through LF files, never inside a trainable forward."""

    RELATIONS = ("covalent", "spatial")
    FIELDS = ("codes", "in_degree", "out_degree")

    def __init__(self, spatial_k: int, maximum_bytes: int = 134_217_728) -> None:
        """Bind the exact nested topology and its approved per-tensor resource bound.

        Args:
            spatial_k: Positive active atom-neighbor rank; all covalent pairs remain active.
            maximum_bytes: Int64 path/two-head FP32 score limit per protein, not total VRAM.
        """
        self.spatial_k = spatial_k
        self.maximum_bytes = maximum_bytes
        self.implementation = hashlib.sha256(
            inspect.getsource(GraphStructuralEncoding.from_edges).encode()
        ).hexdigest()

    def prepare(self, work: lf.Work, sources: Iterable[Path]) -> dict[Path, Path]:
        """Restore or build one atomic-code archive per unique base before any neural work.

        Args:
            work: Current LF Work; owns locking, checksums, atomic publication and cache lifetime.
            sources: Universal archives from precisely the requested splits/views; labels unused.

        Returns:
            Base-path to managed-file-path mapping. The cache is Run-identity scoped, not a
            promise of cross-seed sharing. Compatible Attempts and evaluation views reuse files.

        Raises:
            MemoryError: Any member exceeds the bound, before preparing any quadratic tensors.
            OSError: Input/cache cannot be read or written; LF retains already completed files.
        """
        jobs = []
        for source in dict.fromkeys(sources):
            nodes, pairs, covalent, spatial = self.topology(source)
            required = nodes * nodes * 8
            if required > self.maximum_bytes:
                raise MemoryError(
                    f"global atomic codes for {source.name}: atoms={nodes}, needs {required:,} "
                    f"bytes per principal tensor; limit={self.maximum_bytes:,}; not a CUDA OOM"
                )
            digest = hashlib.sha256(
                f"atomic-paths-v1:{self.spatial_k}:{nodes}:{self.implementation}".encode()
            )
            for value in (pairs, covalent, spatial):
                digest.update(value.numpy().tobytes())
            jobs.append((source, nodes, digest.hexdigest()))

        work.log(f"Preparing/restoring global atomic codes for {len(jobs)} proteins; "
                 "CPU work happens once per Run, before training")
        result = {}
        for index, (source, nodes, signature) in enumerate(jobs, 1):
            work.log(f"Atomic codes {index}/{len(jobs)}: "
                     f"{source.parent.name}/{source.name}, atoms={nodes}")
            managed = work.cache.file(
                f"atomic-structure/{signature}.npz",
                build=partial(self.write, source),
                validate=partial(self.valid, nodes=nodes),
            )
            result[source] = Path(managed)
            work.log(f"Atomic codes {index}/{len(jobs)} ready")
        return result

    def topology(self, source: Path) -> tuple[int, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Read only the immutable arrays needed to reproduce the collator's two relations.

        Args:
            source: Existing universal NPZ, read without pickle or task annotations.

        Returns:
            Atom count, long undirected pairs [2,E], covalent mask [E] and spatial rank mask [E].
            Overlapping covalent/spatial pairs remain in both streams, exactly as during training.
        """
        with np.load(source, allow_pickle=False) as stored:
            nodes = len(stored["atomic_numbers"])
            pairs = torch.from_numpy(stored["atom_edge_index"].astype(np.int64))
            covalent = torch.from_numpy(stored["atom_edge_is_covalent"].astype(np.bool_))
            ranks = stored["atom_edge_spatial_rank"]
            spatial = torch.from_numpy((ranks > 0) & (ranks <= self.spatial_k))
        return nodes, pairs, covalent, spatial

    def write(self, source: Path, destination: Path) -> None:
        """Compute unchanged LF shortest-path codes once and store their exact compact buckets.

        Args:
            source: Immutable base archive; node order and relation topology are never changed.
            destination: Temporary file supplied and atomically published by LF, not WISDOM.

        Raises:
            MemoryError: The public path builder rejects the declared bound.
            OSError: Reading the base or writing the temporary archive fails.
        """
        nodes, pairs, covalent, spatial = self.topology(source)
        values: dict[str, Any] = {"spatial_k": np.asarray(self.spatial_k, dtype=np.int64)}
        for relation, mask in zip(self.RELATIONS, (covalent, spatial), strict=True):
            edges = pairs[:, mask]
            directed = torch.cat((edges, edges.flip(0)), dim=1)
            encoding = GraphStructuralEncoding.from_edges(
                directed, nodes, num_spatial_codes=32, max_pairwise_bytes=self.maximum_bytes,
            )
            # LF's codebook is exactly 0..31. uint8 is a lossless storage/host-transfer format;
            # the public attention layer still receives its original int64 indices on device.

            values[f"{relation}_codes"] = encoding.spatial_indices.numpy().astype(np.uint8)
            values[f"{relation}_in_degree"] = encoding.in_degree.numpy()
            values[f"{relation}_out_degree"] = encoding.out_degree.numpy()
        with destination.open("wb") as stream:
            np.savez_compressed(stream, **values)

    def valid(self, path: Path, nodes: int) -> bool:
        """Validate the compact contract while LF independently verifies exact file integrity.

        Args:
            path: Candidate or restored managed file supplied by LF.
            nodes: Exact atom count from the topology-bound source key.

        Returns:
            True only for complete, lossless codebooks and nonnegative incidence degrees.
            Malformed cache files return False so LF may reconstruct them safely.
        """
        try:
            with np.load(path, allow_pickle=False) as values:
                if int(values["spatial_k"]) != self.spatial_k:
                    return False
                for relation in self.RELATIONS:
                    codes = values[f"{relation}_codes"]
                    if codes.shape != (nodes, nodes) or codes.dtype != np.uint8:
                        return False
                    if np.any(codes >= 32):
                        return False
                    for field in self.FIELDS[1:]:
                        degree = values[f"{relation}_{field}"]
                        if degree.shape != (nodes,) or degree.dtype != np.int64:
                            return False
                        if np.any(degree < 0):
                            return False
        except (OSError, ValueError, KeyError, EOFError, BadZipFile):
            return False
        return True

    @staticmethod
    def read(path: Path) -> dict[str, torch.Tensor | str]:
        """Load one prepared code pack without recomputing paths or expanding its host bytes.

        Args:
            path: Complete managed NPZ validated during preparation.

        Returns:
            Flat prefixed tensors for the dataset/collator; uint8 codes and int64 degrees/K.
        """
        with np.load(path, allow_pickle=False) as stored:
            return {f"atomic_structure__{name}": torch.from_numpy(stored[name])
                    for name in stored.files}

    @staticmethod
    def pack(sample: Mapping[str, torch.Tensor | str], spatial_k: int) -> dict[str, torch.Tensor]:
        """Keep one molecule's prepared operators local while disjoint batches shuffle members.

        Args:
            sample: Dataset tensors for one protein, including the prepared prefix fields.
            spatial_k: Active collator K; using codes prepared for another topology must fail.

        Returns:
            Protein-local relation tensors, without global atom offsets or dense batch padding.

        Raises:
            ValueError: Prepared K or array shape disagrees with the active molecule.
        """
        if int(sample["atomic_structure__spatial_k"]) != spatial_k:
            raise ValueError("global atomic codes were prepared for another spatial K")
        numbers = sample["atomic_numbers"]
        assert isinstance(numbers, torch.Tensor)
        nodes = len(numbers)
        result = {}
        for relation in AtomicStructuralInputs.RELATIONS:
            for field in AtomicStructuralInputs.FIELDS:
                name = f"{relation}_{field}"
                value = sample[f"atomic_structure__{name}"]
                shape = (nodes, nodes) if field == "codes" else (nodes,)
                if not isinstance(value, torch.Tensor) or value.shape != shape:
                    raise ValueError("global atomic code shape disagrees with its protein")
                result[name] = value
        return result
