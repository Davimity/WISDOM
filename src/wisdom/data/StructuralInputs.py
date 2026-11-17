"""Selective immutable sidecar ingestion and explicit protein-local index namespaces."""

import json
import torch
import numpy as np

from typing import Any
from pathlib import Path
from torch import Tensor
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.structure.assets.graphs.ExtendedSurfaceAtom import (
    ExtendedSurfaceAtom,
)
from wisdom.preprocessing.common.structure.assets.StructuralSidecars import StructuralSidecars


class StructuralInputs:
    """Validate requested families once, never reopen unrequested sidecars per epoch."""

    FAMILIES = StructuralSidecars.FAMILIES

    def __init__(
        self, requested: tuple[str, ...] = (), extended_radius: float | None = None,
        extended_k: int | None = None,
    ) -> None:
        """Select independent optional families; defaults leave historical loaders unchanged.

        Args:
            requested: Unique public asset names; each must exist on every selected member.
            extended_radius: Optional runtime radius in Å, bounded by precomputed CSR support.
            extended_k: Optional requested stored candidate limit, bounded by asset maximum.

        Raises:
            ValueError: Unknown or repeated family names.
        """
        if len(set(requested)) != len(requested) or set(requested) - self.FAMILIES:
            raise ValueError("structural_assets contains unknown or repeated asset families")
        self.requested = requested
        self.extended_radius = extended_radius
        self.extended_k = extended_k
        self.paths: dict[str, dict[str, Path]] = {}

    def register(self, identifier: str, base: Path, paths: Mapping[str, Path]) -> None:
        """Bind and scientifically validate only explicitly requested immutable member assets.

        Args:
            identifier: Physical member ID, independent of labels or filename conventions.
            base: Managed universal NPZ.
            paths: Available native member asset paths.

        Raises:
            ValueError: Requested asset absent or scientific validator rejects its binding.
        """
        if not self.requested:
            return
        missing = set(self.requested) - paths.keys()
        if missing:
            raise ValueError(f"member {identifier!r} lacks requested structural assets: "
                             f"{sorted(missing)}")
        selected = {key: paths[key] for key in self.requested}
        StructuralSidecars.validate_assets(base, selected)
        if "extended_surface_atom" in selected:
            with np.load(selected["extended_surface_atom"], allow_pickle=False) as stored:
                metadata = json.loads(str(stored["metadata_json"].item()))
            if (self.extended_radius is not None and not (
                0 < self.extended_radius <= metadata["radius_angstrom"]
            )) or (self.extended_k is not None and not (
                0 < self.extended_k <= metadata["maximum"]
            )):
                raise ValueError("requested extended radius/K exceeds stored candidate support")
        self.paths[identifier] = selected

    def read(self, identifier: str) -> dict[str, Tensor | str]:
        """Load detached numeric tensors/JSON strings without per-epoch base rehashing.

        Args:
            identifier: Member already verified at immutable dataset initialization.

        Returns:
            Family-prefixed tensors; Unicode identity fields become JSON strings, not artificial
            amino-acid categories. Metadata stays explicit and is not a learned feature.
        """
        result: dict[str, Tensor | str] = {}
        for family, path in self.paths.get(identifier, {}).items():
            with np.load(path, allow_pickle=False) as stored:
                for key in stored.files:
                    if key == "base_npz_sha256":
                        continue
                    value = stored[key]
                    name = f"{family}__{key}"
                    if value.dtype.kind in "US":
                        result[name] = str(value.item()) if value.shape == () else json.dumps(
                            value.tolist())
                    else:
                        result[name] = torch.from_numpy(value.astype(np.int64)
                            if value.dtype.kind in "iu" else value)
            if family == "extended_surface_atom":
                if self.extended_radius is not None:
                    result[family + "__active_radius"] = torch.tensor(self.extended_radius)
                if self.extended_k is not None:
                    result[family + "__active_k"] = torch.tensor(self.extended_k)
        return result

    @staticmethod
    def collate(samples: Sequence[Mapping[str, Tensor | str]]) -> dict[str, Any]:
        """Keep heterogeneous sidecar index domains explicitly separated by protein.

        Args:
            samples: Ordered datasets samples, each with independently validated local indices.

        Returns:
            Protein-local member dictionaries per family, with ownership boundaries. Extended
            atom candidates additionally expose the existing global disjoint CSR representation.
            Views and their sparse operators stay local, avoiding dense padding or accidentally
            mixing base-point, alternate-point, residue and sequence-token namespaces.

        Raises:
            ValueError: Missing family/field in a batch; no silently dropped optional input.
        """
        keys = {key for sample in samples for key in sample if "__" in key}
        if not keys:
            return {}
        if any(not keys <= sample.keys() for sample in samples):
            raise ValueError("optional structural asset fields differ across batch members")
        families = sorted({key.split("__", 1)[0] for key in keys})
        result: dict[str, Any] = {}
        for family in families:
            prefix = family + "__"
            members = [{key[len(prefix):]: sample[key] for key in sorted(keys)
                        if key.startswith(prefix)} for sample in samples]
            result[family] = {"namespace": "protein-local", "members": members}
            if family == "extended_surface_atom":
                numpy_members = [{key: value.numpy() for key, value in member.items()
                                  if isinstance(value, Tensor)} for member in members]
                atom_counts = [len(sample["atomic_numbers"]) for sample in samples]
                result[family]["disjoint_csr"] = {key: torch.from_numpy(value) for key, value in
                    ExtendedSurfaceAtom.batch(numpy_members, atom_counts).items()}
        return {"structural_assets": result}
