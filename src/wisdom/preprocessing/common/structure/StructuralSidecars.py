"""Shared managed enrichment for DNA, zinc, and already published universal archives."""

import json
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping
from wisdom.utils.structure.models.Protein import Protein
from wisdom.preprocessing.common.structure.AtomicContext import AtomicContext
from wisdom.preprocessing.common.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.StructureSource import StructureSource
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.ExtendedSurfaceAtom import ExtendedSurfaceAtom


class StructuralSidecars:
    """Coordinate optional independently fingerprinted files using LambdaForge checkpoints."""

    def __init__(
        self, atomic_context: bool = False, source_quality: bool = True,
        extended_surface_atom: bool = False, extended_radius: float = 12.0,
        extended_k_max: int = 96,
    ) -> None:
        """Keep enrichment choices outside the unchanged base-geometry configuration hash.

        Args:
            atomic_context: Generate canonical identity, conformation and covalent context.
            source_quality: Include observed quality fields when an exact source is available.
            extended_surface_atom: Generate an independently versioned CSR candidate asset.
            extended_radius: Positive candidate radius in Å, default 12.
            extended_k_max: Positive maximum stored candidates per point, default 96.
        """
        self.context = AtomicContext(source_quality) if atomic_context else None
        self.extended = ExtendedSurfaceAtom(extended_radius, extended_k_max) if (
            extended_surface_atom) else None

    def generate(
        self, work: lf.Work, bases: Mapping[str, Path], workers: int,
        existing: Mapping[str, Mapping[str, Path]] | None = None,
    ) -> dict[str, dict[str, Path]]:
        """Restore/build validated sidecars without recomputing surfaces or spectral operators.

        Args:
            work: Active native Work supplying managed files, map checkpoints and logs.
            bases: Exact universal NPZ path per immutable member ID.
            workers: Bounded independent thread workers, excluded from scientific keys.
            existing: Optional already published assets per member. Compatible validated assets
                are reused, preserving source-only information after the original source is gone.

        Returns:
            Optional asset-name to checkpoint-path mapping per member. Disabled families produce
            no file and no source reads. Every reused base undergoes full schema validation.
        """
        if self.context is None and self.extended is None:
            return {identifier: {} for identifier in bases}
        work.log(f"Enriching {len(bases)} immutable archives; base geometry will not be rebuilt")

        def enrich(row: dict[str, str]) -> dict[str, str]:
            """Build one record behind the native managed scientific validation boundary.

            Args:
                row: Member ID and exact base path; task targets are deliberately absent.

            Returns:
                Managed optional file paths, serialized for the native resume map.
            """
            base = Path(row["base"])
            metadata = ProteinArchive(PreprocessConfig()).read_metadata(base)
            if metadata is None:
                raise ValueError("structural enrichment requires readable base metadata")
            archive = ProteinArchive(PreprocessConfig(**metadata["config"]))
            if not archive.can_resume(base, str(metadata["source_hash"])):
                raise ValueError(f"invalid universal NPZ before enrichment: {row['id']}")
            digest = ProteinArchive.sha256(base)
            result: dict[str, str] = {}

            # Re-enriching a placement must not erase verified author/quality fields merely
            # because an old source path is gone. Reuse compatible immutable input sidecars.

            for name in ("atomic_context", "extended_surface_atom"):
                if name not in row:
                    continue
                path = Path(row[name])
                stored = archive.read_metadata(path)
                if stored is None:
                    raise ValueError(f"unreadable existing structural asset: {name}")
                if name == "atomic_context" and self.context is not None and stored.get(
                    "source_quality") == self.context.source_quality:
                    self.context.validate(path, base)
                    result[name] = str(path)
                if name == "extended_surface_atom" and self.extended is not None and (
                    stored.get("radius_angstrom"), stored.get("maximum")) == (
                        self.extended.radius, self.extended.maximum):
                    self.extended.validate(path, base)
                    result[name] = str(path)
            protein = self.source_protein(metadata, self.context.source_quality) if (
                self.context is not None and "atomic_context" not in result) else None
            if self.context is not None and "atomic_context" not in result:
                settings = {"base": digest, "schema": AtomicContext.VERSION,
                            "quality": self.context.source_quality,
                            "source_aligned": protein is not None}
                key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                path = work.checkpoints.file(f"structural-sidecars/{key}/atomic-context.npz",
                    build=partial(self.context.write, base, protein=protein),
                    validate=partial(self.context.validate, base=base, protein=protein))
                result["atomic_context"] = str(path)
            if self.extended is not None and "extended_surface_atom" not in result:
                settings = {"base": digest, "schema": ExtendedSurfaceAtom.VERSION,
                            "radius": self.extended.radius, "maximum": self.extended.maximum}
                key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                path = work.checkpoints.file(f"structural-sidecars/{key}/extended-surface-atom.npz",
                    build=partial(self.extended.write, base),
                    validate=partial(self.extended.validate, base=base))
                result["extended_surface_atom"] = str(path)
            return result

        # Every mapped invocation reaches scientific file validation. The file checkpoint API owns
        # concurrent access, temporary files, replacement and recovery; WISDOM adds no cache locks.

        rows = [{"id": key, "base": str(path), **{name: str(asset) for name, asset in
            (existing or {}).get(key, {}).items() if name in (
                "atomic_context", "extended_surface_atom")}} for key, path in bases.items()]
        result = work.map(rows, enrich, workers=workers, executor="thread",
                          name="structural-sidecars")
        work.log("Optional structural sidecars validated; historical base archives unchanged")
        return {row["id"]: {name: Path(path) for name, path in value.items()}
                for row, value in zip(rows, result, strict=True)}

    @staticmethod
    def source_protein(
        metadata: Mapping[str, Any], preserve_quality: bool = True,
    ) -> Protein | None:
        """Reopen a source only when its exact deposited bytes still match base provenance.

        Args:
            metadata: Schema-3 source digest, model/filter settings and selected chains.
            preserve_quality: Read explicit mmCIF quality columns, default true. False still
                recovers author residue/chain identity but avoids the extra quality-table parse.

        Returns:
            Identically selected hierarchy, or None if the original exact source is unavailable.
            A portable uncompressed copy is not silently treated as the original gzip container.
        """
        path = Path(str(metadata.get("source_path", "")))
        if not path.is_file() or ProteinArchive.sha256(path) != metadata.get(
            "source_hash"):
            return None
        source = StructureSource(str(metadata["source_identifier"]), str(metadata["protein_id"]),
            tuple(metadata["selected_chains"]), path, str(metadata["source_hash"]),
            str(metadata["source_format"]), True)
        reader = ProteinReader(PreprocessConfig(**metadata["config"]), preserve_quality)
        protein, _ = reader.read(source)
        return protein

    @staticmethod
    def validate_assets(base: Path, assets: Mapping[str, Path]) -> None:
        """Audit independently versioned optional assets during either task's final validation.

        Args:
            base: Immutable universal archive.
            assets: Native member asset paths; absent enrichment assets are legal legacy input.

        Raises:
            ValueError: Wrong binding, schema, source alignment, numerical fields or coverage.
        """
        for name in ("atomic_context", "extended_surface_atom"):
            if name not in assets:
                continue
            path = assets[name]
            import_metadata = ProteinArchive(PreprocessConfig()).read_metadata(path)
            if import_metadata is None:
                raise ValueError(f"missing optional {name} metadata")
            if name == "atomic_context":
                context = AtomicContext(bool(import_metadata["source_quality"]))
                base_metadata = ProteinArchive(PreprocessConfig()).read_metadata(base)
                protein = StructuralSidecars.source_protein(base_metadata or {},
                    context.source_quality) if (
                    import_metadata["source_aligned"]) else None
                context.validate(path, base, protein)
            else:
                ExtendedSurfaceAtom(float(import_metadata["radius_angstrom"]),
                    int(import_metadata["maximum"])).validate(path, base)
