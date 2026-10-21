"""Publish optional structural context for an existing dataset, without rebuilding geometry."""

import lambdaforge as lf

from typing import Any
from pathlib import Path
from lambdaforge.data import DatasetIndex
from wisdom.preprocessing.common.structure.StructuralSidecars import StructuralSidecars


class StructuralEnrichment(lf.Work):
    """Reuse immutable base and task assets and attach common independently validated sidecars."""

    def run(
        self, dataset: Path, dataset_name: str, dataset_version: str,
        atomic_context: bool = True, source_quality: bool = True,
        extended_surface_atom: bool = True, extended_radius: float = 12.0,
        extended_k_max: int = 96, workers: int = 4,
    ) -> dict[str, Any]:
        """Enrich a typed Registry placement without selecting members or recomputing surfaces.

        Args:
            dataset: Native immutable dataset root containing index.jsonl and base assets.
            dataset_name: Destination Registry family (existing or new).
            dataset_version: New immutable version, never overwrite the source version.
            atomic_context: Include context, default true.
            source_quality: Include observed metadata if exact original source remains available.
            extended_surface_atom: Include expanded CSR candidates, default true.
            extended_radius: Positive physical radius in Å, default 12.
            extended_k_max: Positive stored per-point cap, default 96; truncation audited.
            workers: Bounded native file-map concurrency, default 4; does not affect content.

        Returns:
            Native DatasetVersion record. Member IDs, targets, partitions, metadata and all
            existing task/base assets are preserved, with optional structural assets added.

        Raises:
            ValueError: Invalid base or sidecar; already published destination; bad member index.
        """
        self.outputs.dataset_preflight(name=dataset_name, version=dataset_version)
        members = list(DatasetIndex(dataset / "index.jsonl"))
        sidecars = StructuralSidecars(atomic_context, source_quality, extended_surface_atom,
                                     extended_radius, extended_k_max)
        bases = {m.member_id: dataset / m.assets["universal_npz"].path for m in members}
        additions = sidecars.generate(self, bases, workers, existing={member.member_id:
            {name: dataset / asset.path for name, asset in member.assets.items()}
            for member in members})

        # Import original assets once through the public ownership API. Publication retains byte
        # identity, not necessarily a zero-copy physical placement. No Registry shim is introduced.

        source = self.outputs.artifact("source-dataset", dataset)
        published = []
        present: set[str] = set()
        for member in members:
            extra = additions[member.member_id]
            assets = {name: source / asset.path for name, asset in member.assets.items()}
            assets.update({name: self.outputs.artifact(f"{name}-{member.member_id}", path)
                           for name, path in extra.items()})
            StructuralSidecars.validate_assets(assets["universal_npz"], assets)
            present.update(name for name in assets if name in (
                "atomic_context", "extended_surface_atom"))
            published.append({"id": member.member_id, "assets": assets,
                "targets": dict(member.targets), "partitions": dict(member.partitions),
                "metadata": dict(member.metadata), "display": dict(member.display)})
        return dict(self.outputs.dataset(name=dataset_name, version=dataset_version,
            members=published, metadata={"structural_schema": "3.0",
                "atomic_context_schema": "1.0" if "atomic_context" in present else None,
                "extended_surface_atom_schema": (
                    "1.0" if "extended_surface_atom" in present else None),
                "base_geometry_recomputed": False}))
