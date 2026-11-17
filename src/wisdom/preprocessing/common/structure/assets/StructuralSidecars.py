"""Shared managed enrichment for DNA, zinc, and already published universal archives."""

import json
import time
import hashlib
import numpy as np
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping
from wisdom.utils.structure.models.Protein import Protein
from wisdom.preprocessing.common.structure.io.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.io.StructureSource import StructureSource
from wisdom.preprocessing.common.structure.assets.context.AtomicSASA import AtomicSASA
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.surface.SurfaceView import SurfaceView
from wisdom.preprocessing.common.structure.assets.graphs.ResidueGraph import ResidueGraph
from wisdom.preprocessing.common.structure.assets.context.AtomicContext import AtomicContext
from wisdom.preprocessing.common.structure.assets.physics.PhysicalFields import PhysicalFields
from wisdom.preprocessing.common.structure.assets.context.SequenceContext import SequenceContext
from wisdom.preprocessing.common.structure.assets.graphs.SurfaceLocalGraph import SurfaceLocalGraph
from wisdom.preprocessing.common.structure.assets.StructuralAssetFamily import StructuralAssetFamily
from wisdom.preprocessing.common.structure.assets.surface.SpectralExtension import SpectralExtension
from wisdom.preprocessing.common.structure.io.VerifiedStructureSource import VerifiedStructureSource
from wisdom.preprocessing.common.structure.assets.external.StructureTokens import (
    StructureTokens,
)
from wisdom.preprocessing.common.structure.assets.surface.SurfaceEnvironment import (
    SurfaceEnvironment,
)
from wisdom.preprocessing.common.structure.assets.graphs.ExtendedSurfaceAtom import (
    ExtendedSurfaceAtom,
)
from wisdom.preprocessing.common.structure.assets.context.CovalentEnvironment import (
    CovalentEnvironment,
)
from wisdom.preprocessing.common.structure.assets.external.AtomicProteinExtractor import (
    AtomicProteinExtractor,
)
from wisdom.preprocessing.common.structure.assets.external.ExternalProteinRepresentation import (
    ExternalProteinRepresentation,
)


class StructuralSidecars:
    """Coordinate optional independently fingerprinted files using LambdaForge checkpoints."""

    FAMILIES = frozenset(family.value for family in StructuralAssetFamily)

    def __init__(
        self, atomic_context: bool = False, source_quality: bool = True,
        extended_surface_atom: bool = False, extended_radius: float = 12.0,
        extended_k_max: int = 96,
        *, sequence_context: bool = False, atomic_sasa: bool = False,
        sasa_probe: float = 1.4, sasa_samples: int = 960,
        residue_graph: bool = False, residue_contact_radius: float = 8.0,
        residue_surface_radius: float = 12.0, surface_local_graph: bool = False,
        surface_graph_resolution: float = 1.0, surface_view: bool = False,
        view_id: int = 1, view_resolution: float | None = None,
        external: ExternalProteinRepresentation | None = None,
        structure_tokens: bool = False, static_embedding_table: bool = False,
        covalent_environment: bool = False, spectral_modes: int | None = None,
        residue_ca_radius: float | None = None, residue_ca_k_max: int = 32,
        surface_environment: bool = False, environment_options: Mapping[str, Any] | None = None,
        atomic_external: AtomicProteinExtractor | None = None,
    ) -> None:
        """Keep enrichment choices outside the unchanged base-geometry configuration hash.

        Args:
            atomic_context: Generate canonical identity, conformation and covalent context.
            source_quality: Include observed quality fields when an exact source is available.
            extended_surface_atom: Generate an independently versioned CSR candidate asset.
            extended_radius: Positive candidate radius in Å, default 12.
            extended_k_max: Positive maximum stored candidates per point, default 96.
            sequence_context: Add residue/observed/deposited sequence correspondence, default false.
            atomic_sasa: Add deterministic atomic solvent exposure, default false.
            sasa_probe: Positive probe radius in Å, default 1.4.
            sasa_samples: Quadrature directions per atom, at least 24, default 960.
            residue_graph: Add residue contacts and untruncated surface candidates, default false.
            residue_contact_radius: Minimum heavy-atom residue contact cutoff, default 8 Å.
            residue_surface_radius: Point-to-residue cutoff, default 12 Å.
            surface_local_graph: Publish existing filtered surface topology, default false.
            surface_graph_resolution: Existing graph rule scale, default 1 Å.
            surface_view: Generate one independent surface discretization, default false.
            view_id: Nonnegative deterministic view ID, default 1.
            view_resolution: Optional spacing in Å; None preserves base resolution.
            external: Optional explicitly loaded frozen residue extractor; never constructed by
                default or from task-specific evidence.
            structure_tokens: Request official Foldseek 3Di, default false.
            static_embedding_table: Export the loaded model's actual input vectors, default false.
            covalent_environment: Compact neighbour element/order statistics, default false.
            spectral_modes: Optional independent fixed spectral budget, default None.
            residue_ca_radius: Optional separately constructed CA graph radius in Å.
            residue_ca_k_max: Directed CA neighbor storage budget, default 32.
            surface_environment: Build masked free-space geometry, default false.
            environment_options: SurfaceEnvironment physical parameters; None uses its defaults.
            atomic_external: Optional explicit frozen atom extractor loaded before geometry;
                None does not import/download/execute external atomic models.
        """
        self.context = AtomicContext(source_quality) if atomic_context else None
        self.extended = ExtendedSurfaceAtom(extended_radius, extended_k_max) if (
            extended_surface_atom) else None
        self.sequence = SequenceContext() if (sequence_context or external or structure_tokens
                                              ) else None
        self.external = external
        self.structure_tokens = structure_tokens or (external is not None
                                                     and external.extractor.kind == "saprot")
        self.static_table = static_embedding_table
        self.report: dict[str, Any] = {}
        if residue_ca_radius is not None and not residue_graph:
            raise ValueError("residue_ca_radius requires residue_graph=true")
        if static_embedding_table and external is None:
            raise ValueError("static_embedding_table requires an explicit external model")
        self.producers: dict[str, AtomicSASA | ResidueGraph | SurfaceLocalGraph | SurfaceView |
                             CovalentEnvironment | SpectralExtension | SurfaceEnvironment |
                             AtomicProteinExtractor] = {}
        if atomic_external is not None:
            self.producers[atomic_external.FAMILY] = atomic_external
        if atomic_sasa:
            self.producers["atomic_sasa"] = AtomicSASA(sasa_probe, sasa_samples)
        if residue_graph:
            self.producers["residue_graph"] = ResidueGraph(
                residue_contact_radius, residue_surface_radius, residue_ca_radius, residue_ca_k_max)
        if surface_environment:
            self.producers["surface_environment"] = SurfaceEnvironment(
                **(environment_options or {}))
        if surface_local_graph:
            self.producers["surface_local_graph"] = SurfaceLocalGraph(surface_graph_resolution)
        if surface_view:
            self.producers["surface_view"] = SurfaceView(view_id, view_resolution)
        if covalent_environment:
            self.producers["covalent_environment"] = CovalentEnvironment()
        if spectral_modes is not None:
            self.producers["spectral_extension"] = SpectralExtension(spectral_modes)

    @property
    def requested_families(self) -> frozenset[str]:
        """Return the exact family inventory required before any dataset may be published.

        Returns:
            Enabled producer names plus automatic sequence/3Di dependencies. This is an asset
            completeness contract, not a claim that every residue has observable source data.
        """
        names = set(self.producers)
        if self.context is not None:
            names.add("atomic_context")
        if self.extended is not None:
            names.add("extended_surface_atom")
        if self.sequence is not None:
            names.add("sequence_context")
        if self.structure_tokens:
            names.add("structure_tokens")
        if self.external is not None:
            names.add("external_embeddings")
        if self.static_table:
            names.add("static_embedding_table")
        return frozenset(names)

    def generate(
        self, work: lf.Work, bases: Mapping[str, Path], workers: int,
        existing: Mapping[str, Mapping[str, Path]] | None = None,
        source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        source_hashes: Mapping[str, str] | None = None,
    ) -> dict[str, dict[str, Path]]:
        """Restore/build validated sidecars without rebuilding the historical base geometry.

        Requested alternate views compute their own surfaces/operators; spectral extensions
        solve only missing fixed modes. These independent files never modify base arrays.

        Args:
            work: Active native Work supplying managed files, map checkpoints and logs.
            bases: Exact universal NPZ path per immutable member ID.
            workers: Bounded independent thread workers, excluded from scientific keys.
            existing: Optional already published assets per member. Compatible validated assets
                are reused, preserving source-only information after the original source is gone.
            source_metadata: Optional explicit universal assembly/copy selection per member;
                functional evidence and labels must not be supplied or consumed here.
            source_hashes: Native source_structure asset checksums, when available.

        Returns:
            Optional asset-name to checkpoint-path mapping per member. Disabled families produce
            no file and no source reads. Every reused base undergoes full schema validation.
        """
        began = time.perf_counter()
        self.report = {"total": len(bases), "status": "incomplete", "records": [],
                       "base_geometry_recomputed": False}
        if self.context is None and self.extended is None and self.sequence is None \
                and not self.producers:
            self.report["status"] = "no-enrichment-requested"
            return {identifier: {} for identifier in bases}
        work.log(f"Enriching {len(bases)} immutable archives; base geometry will not be rebuilt")
        work.log("Existing sidecars undergo scientific validation, including SASA quadrature "
                 "and deterministic surface resampling; reuse still requires CPU computation")
        self.preflight_sequences(work, bases, workers, existing, source_metadata, source_hashes)
        tokenizer = StructureTokens(work) if self.structure_tokens else None

        def enrich(row: dict[str, str]) -> dict[str, str]:
            """Build one record behind the native managed scientific validation boundary.

            Args:
                row: Member ID and exact base path; task targets are deliberately absent.

            Returns:
                Managed optional file paths, serialized for the native resume map.
            """
            base = Path(row["base"])
            started = time.perf_counter()
            work.log(f"Validating/restoring structural assets for {row['id']}")
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
            published = Path(row["source_structure"]) if "source_structure" in row else None
            protein, source, quality = (None, None, "unavailable")
            if (self.context is not None and "atomic_context" not in result) or (
                self.sequence is not None and "sequence_context" not in row
            ) or (tokenizer is not None and "structure_tokens" not in row):
                protein, source, quality = VerifiedStructureSource.recover(
                    base, published, (source_hashes or {}).get(row["id"]),
                    quality=self.context.source_quality if self.context else False,
                    assembly=(source_metadata or {}).get(row["id"]))
            if self.context is not None and "atomic_context" not in result:
                settings = {"base": digest, "schema": AtomicContext.VERSION,
                            "quality": self.context.source_quality,
                            "source_aligned": protein is not None}
                if quality == "equivalent":
                    settings["source_quality_level"] = quality
                    settings["verified_source"] = ProteinArchive.sha256(source) if source else None
                    settings["source_selection"] = (source_metadata or {}).get(row["id"])
                key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                path = work.checkpoints.file(f"structural-sidecars/{key}/atomic-context.npz",
                    build=partial(self.context.write, base, protein=protein,
                                  source_level=quality,
                                  source_selection=(source_metadata or {}).get(row["id"])),
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
            if self.sequence is not None:
                stored_context = None
                context_path = result.get("atomic_context", row.get("atomic_context"))
                if context_path is not None:
                    context_metadata = archive.read_metadata(Path(context_path))
                    if context_metadata is None:
                        raise ValueError("unreadable historical atomic context")
                    if "atomic_context" not in result:
                        AtomicContext(bool(context_metadata["source_quality"])).validate(
                            Path(context_path), base)
                    with np.load(context_path, allow_pickle=False) as stored:
                        stored_context = {key: stored[key] for key in stored.files}
                settings = {"base": digest, "schema": "1.0", "quality": quality,
                            "source_selection": (source_metadata or {}).get(row["id"]),
                            "verified_source": ProteinArchive.sha256(source) if source else None,
                            "context": self.sequence.author_evidence_hash(stored_context)}
                key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                existing_sequence = row.get("sequence_context")
                if existing_sequence:
                    self.sequence.validate(Path(existing_sequence), base)
                    result["sequence_context"] = existing_sequence
                else:
                    path = work.checkpoints.file(f"structural-sidecars/{key}/sequence-context.npz",
                        build=partial(self.sequence.write, base, protein=protein, source=source,
                                      context=stored_context, quality=quality,
                                      source_selection=(source_metadata or {}).get(row["id"])),
                        validate=partial(self.sequence.validate, base=base))
                    result["sequence_context"] = str(path)
            if tokenizer is not None:
                if "structure_tokens" in row:
                    tokenizer.validate(Path(row["structure_tokens"]), base)
                    result["structure_tokens"] = row["structure_tokens"]
                elif protein is None:
                    raise ValueError("official 3Di extraction requires a verified portable source")
                else:
                    sequence_path = Path(result["sequence_context"])
                    key = hashlib.sha256(json.dumps({"base": digest,
                        "sequence": ProteinArchive.sha256(sequence_path),
                        "tool": tokenizer.tool.name, "version": tokenizer.tool.version},
                        sort_keys=True).encode()).hexdigest()
                    path = work.checkpoints.file(f"structural-sidecars/{key}/structure-tokens.npz",
                        build=partial(tokenizer.write, base, sequence=sequence_path,
                                      protein=protein),
                        validate=partial(tokenizer.validate, base=base))
                    result["structure_tokens"] = str(path)
            if self.external is not None:
                sequence_path = Path(result["sequence_context"])
                token_path = (Path(result["structure_tokens"])
                              if "structure_tokens" in result else None)
                for static in ((False, True) if self.static_table else (False,)):
                    name = "static_embedding_table" if static else "external_embeddings"
                    settings = {"base": digest, "family": name,
                        "sequence": ProteinArchive.sha256(sequence_path),
                        "structure_tokens": (ProteinArchive.sha256(token_path)
                                             if token_path else None),
                        "extractor": self.external.parameters}
                    if name in row:
                        _, header = StructuralAsset.read(Path(row[name]), base, name)
                        if header["parameters"] == self.external.parameters and header.get(
                            "sequence_sha256") == settings["sequence"] and header.get(
                                "structure_tokens_sha256") == settings["structure_tokens"]:
                            self.external.validate(Path(row[name]), base, static=static)
                            result[name] = row[name]
                            continue
                    key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                    path = work.checkpoints.file(f"structural-sidecars/{key}/{name}.npz",
                        build=partial(self.external.write, base, sequence=sequence_path,
                                      structure_tokens=token_path, static=static),
                        validate=partial(self.external.validate, base=base, static=static))
                    result[name] = str(path)
            for name, producer in self.producers.items():
                if name in row:
                    _, header = StructuralAsset.read(Path(row[name]), base, name)
                    stored_parameters = header["parameters"]
                    parameters = producer.parameters
                    if isinstance(producer, SurfaceEnvironment):
                        stored_parameters = {k: v for k, v in stored_parameters.items()
                                             if k != "maximum_voxels"}
                        parameters = producer.scientific_parameters()
                    if stored_parameters == parameters:
                        producer.validate(Path(row[name]), base)
                        result[name] = row[name]
                        continue
                parameters = (producer.scientific_parameters()
                              if isinstance(producer, SurfaceEnvironment) else producer.parameters)
                settings = {"base": digest, "family": name, "schema": "1.0",
                            "parameters": parameters}
                key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
                checkpoint = f"structural-sidecars/{key}/{name}.npz"

                # Earlier environment keys included a pure allocation guard. Restore their exact
                # native files after a guard increase, but retain every physical/schema/hash check.
                # This is a bounded legacy-key migration, not a directory scan or copied cache.

                if isinstance(producer, SurfaceEnvironment):
                    for ceiling in dict.fromkeys((2000000, producer.parameters["maximum_voxels"])):
                        legacy_settings = {**settings, "parameters": {
                            **producer.parameters, "maximum_voxels": ceiling}}
                        legacy_key = hashlib.sha256(json.dumps(
                            legacy_settings, sort_keys=True).encode()).hexdigest()
                        legacy = f"structural-sidecars/{legacy_key}/{name}.npz"
                        if work.checkpoints.exists(legacy):
                            checkpoint = legacy
                            work.log(
                                f"Restoring historical environment checkpoint for {row['id']}; "
                                "allocation ceiling changed, physical settings unchanged")
                            break
                path = work.checkpoints.file(checkpoint,
                    build=partial(producer.write, base),
                    validate=partial(producer.validate, base=base))
                result[name] = str(path)
            work.log(f"Structural assets {row['id']}: {len(result)} families validated in "
                     f"{time.perf_counter() - started:.2f}s")
            return result

        # Every mapped invocation reaches scientific file validation. The file checkpoint API owns
        # concurrent access, temporary files, replacement and recovery; WISDOM adds no cache locks.

        rows = [{"id": key, "base": str(path), **{name: str(asset) for name, asset in
            (existing or {}).get(key, {}).items() if name in (
                "atomic_context", "extended_surface_atom", "source_structure", "sequence_context",
                "structure_tokens", "external_embeddings", "static_embedding_table",
                *self.producers)}} for key, path in bases.items()]
        def checked(row: dict[str, str]) -> dict[str, str]:
            """Retain other proteins' validated checkpoints when one requested asset fails.

            Args:
                row: One member's immutable structural inputs, never task labels.

            Returns:
                Validated asset paths or an explicit failure. The complete Work fails after all
                records finish; an incomplete mandatory enrichment is never published as valid.
            """
            try:
                return enrich(row)
            except Exception as error:
                message = f"{type(error).__name__}: {error}"
                work.log(f"Structural enrichment failed for {row['id']}: {message}")
                return {"_failure": message}

        result = work.map(rows, checked, workers=workers, executor="thread",
                          name="structural-sidecars")
        failed = [{"identifier": row["id"], "error": value["_failure"]}
                  for row, value in zip(rows, result, strict=True) if "_failure" in value]
        records = []
        reused = 0
        for row, value in zip(rows, result, strict=True):
            if "_failure" in value:
                records.append({"identifier": row["id"], "status": "failed",
                                "error": value["_failure"]})
                continue
            files = {}
            for name, path in value.items():
                reused += int(path == row.get(name))
                metadata = ProteinArchive(PreprocessConfig()).read_metadata(Path(path)) or {}
                files[name] = {"bytes": Path(path).stat().st_size,
                    "schema": metadata.get("schema"), "arrays": metadata.get("arrays"),
                    "coverage": metadata.get("coverage", metadata.get("boolean_counts")),
                    "source_aligned": metadata.get("source_aligned"),
                    "source_quality_level": metadata.get("source_quality_level")}
            records.append({"identifier": row["id"], "status": "validated", "assets": files})
        self.report.update(status="failed" if failed else "validated", records=records,
            failed=len(failed), completed=len(rows) - len(failed),
            reused_published_assets=reused, seconds=time.perf_counter() - began)
        if failed:
            summary = {"total": len(rows), "failed": len(failed), "records": failed,
                       "completed": len(rows) - len(failed), "published": False}
            key = hashlib.sha256(json.dumps(summary, sort_keys=True).encode()).hexdigest()
            work.checkpoints.file(f"structural-sidecars/{key}/failures.json",
                                  build=partial(ProteinArchive.write_report, value=summary))
            raise RuntimeError(f"structural enrichment failed for {len(failed)}/{len(rows)} "
                               f"members; successful checkpoints retained: {json.dumps(failed)}")
        work.log("Optional structural sidecars validated; historical base archives unchanged")
        return {row["id"]: {name: Path(path) for name, path in value.items()}
                for row, value in zip(rows, result, strict=True)}

    def preflight_sequences(
        self, work: lf.Work, bases: Mapping[str, Path], workers: int,
        existing: Mapping[str, Mapping[str, Path]] | None = None,
        source_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        source_hashes: Mapping[str, str] | None = None,
    ) -> None:
        """Check every physical sequence join before any costly structural family is built.

        Args:
            work: Native Work providing bounded maps, logging and managed failure artifacts.
            bases: Immutable universal NPZ per member; base geometry is never recomputed.
            workers: Bounded concurrent lightweight source/sequence checks, not scientific identity.
            existing: Optional source_structure and already published sequence_context assets.
            source_metadata: Exact assembly/copy provenance, never functional labels or targets.
            source_hashes: Optional expected source_structure byte hashes.

        Returns:
            None after every requested sequence mapping passes. No surface, spectral solve,
            environment field or neural inference is performed. Existing file checkpoints remain
            untouched when a different member fails; ordinary enrichment revalidates its assets.

        Raises:
            RuntimeError: One or more source/sequence joins fail; all identifiers/errors are saved
                in sequence-preflight/failures.json before expensive enrichment begins.
        """
        if self.sequence is None:
            return
        sequence = self.sequence
        work.log(f"Preflighting physical sequence correspondence for {len(bases)} proteins "
                 "before expensive structural assets")

        def check(identifier: str) -> dict[str, str]:
            """Read only exact source evidence and arrays consumed by the sequence contract.

            Args:
                identifier: Exact member identity in bases and optional source/asset mappings.

            Returns:
                Member identity and optional diagnostic error; no data is repaired or published.
            """
            try:
                base = bases[identifier]
                assets = (existing or {}).get(identifier, {})
                if "sequence_context" in assets:
                    sequence.validate(assets["sequence_context"], base)
                else:
                    protein, source, _ = VerifiedStructureSource.recover(
                        base, assets.get("source_structure"),
                        (source_hashes or {}).get(identifier), quality=False,
                        assembly=(source_metadata or {}).get(identifier))
                    context = None
                    if "atomic_context" in assets:
                        with np.load(assets["atomic_context"], allow_pickle=False) as stored:
                            context = {key: stored[key] for key in stored.files}
                    with np.load(base, allow_pickle=False) as stored:
                        keys = ("residue_indices", "chain_indices", "residue_names",
                                "atom_role_ids", "atom_edge_index", "atom_edge_is_covalent",
                                "atom_names")
                        arrays = {key: stored[key] for key in keys}
                        metadata = json.loads(str(stored["metadata_json"].item()))
                    sequence.build(arrays, protein, source, context,
                                   int(metadata.get("model_index", 0)))
                return {"identifier": identifier}
            except Exception as error:
                return {"identifier": identifier, "error": f"{type(error).__name__}: {error}"}

        results = work.map(list(bases), check, workers=workers, executor="thread",
                           name="sequence-correspondence-preflight")
        failed = [result for result in results if "error" in result]
        if failed:
            self.report.update(status="sequence-preflight-failed", records=failed,
                               failed=len(failed), completed=len(bases) - len(failed))
            output = Path(work.outputs.directory("sequence-preflight", role="report"))
            (output / "failures.json").write_text(json.dumps(failed, indent=2))
            raise RuntimeError(f"sequence preflight failed for {len(failed)}/{len(bases)} "
                "members before expensive enrichment; prior checkpoints retained: "
                + json.dumps(failed))
        work.log(f"Physical sequence preflight passed for all {len(bases)} proteins")

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
        if "sequence_context" in assets:
            SequenceContext().validate(assets["sequence_context"], base)
        if "structure_tokens" in assets:
            StructureTokens.validate(assets["structure_tokens"], base)
        for name in ("external_embeddings", "static_embedding_table"):
            if name in assets:
                ExternalProteinRepresentation.validate(assets[name], base,
                                                       static=name == "static_embedding_table")
            for kind in ("esm2", "esmc", "saprot"):
                qualified = f"{name}_{kind}"
                if qualified in assets:
                    ExternalProteinRepresentation.validate(assets[qualified], base,
                        static=name == "static_embedding_table", family=qualified)
        if AtomicProteinExtractor.FAMILY in assets:
            AtomicProteinExtractor.validate(assets[AtomicProteinExtractor.FAMILY], base)
        if PhysicalFields.REFERENCE in assets:
            PhysicalFields.validate_reference(assets[PhysicalFields.REFERENCE], base,
                sequence_context=assets.get("sequence_context"))
        if PhysicalFields.QUERIES in assets:
            PhysicalFields.validate_queries(assets[PhysicalFields.QUERIES], base)
        if {PhysicalFields.REFERENCE, PhysicalFields.QUERIES} <= assets.keys():
            _, query_metadata = StructuralAsset.read(assets[PhysicalFields.QUERIES], base,
                                                     PhysicalFields.QUERIES)
            if query_metadata["reference_sha256"] != ProteinArchive.sha256(
                assets[PhysicalFields.REFERENCE]
            ):
                raise ValueError("VF query bank and physical reference snapshots differ")
        for name in ("atomic_sasa", "residue_graph", "surface_local_graph", "surface_view",
                     "covalent_environment", "spectral_extension", "surface_environment"):
            if name not in assets:
                continue
            _, header = StructuralAsset.read(assets[name], base, name)
            parameters = header["parameters"]
            producer = {"atomic_sasa": AtomicSASA, "residue_graph": ResidueGraph,
                        "surface_local_graph": SurfaceLocalGraph, "surface_view": SurfaceView,
                        "covalent_environment": CovalentEnvironment,
                        "spectral_extension": SpectralExtension,
                        "surface_environment": SurfaceEnvironment,
                        }[name](**parameters)
            producer.validate(assets[name], base)
