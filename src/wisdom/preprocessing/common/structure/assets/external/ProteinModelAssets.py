"""Stage selected reviewed frozen protein models as separate, resumable residue assets."""

import json
import hashlib
import lambdaforge as lf

from pathlib import Path
from functools import partial
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.structure.assets.external.ExternalModelKind import (
    ExternalModelKind,
)
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.external.ExternalProteinRepresentation import (
    ExternalProteinRepresentation,
)
from wisdom.preprocessing.common.structure.assets.external.ProteinRepresentationExtractor import (
    ProteinRepresentationExtractor,
)


class ProteinModelAssets:
    """Load one selected checkpoint at a time; retain explicit model-qualified asset names."""

    def __init__(self, root: Path, maximum_length: int = 1022, layer: int = -1,
                 device: str = "cpu", *, kinds: Sequence[str] | None = None,
                 static_tables: bool = True) -> None:
        """Bind explicit local checkpoints and verify their staged bytes before geometry starts.

        Args:
            root: Typed directory with checkpoints.json and the selected model subdirectories.
            maximum_length: Positive disjoint residue chunk length, excluding CLS/EOS, default 1022.
            layer: Hidden-state layer index, default -1 (last).
            device: Explicit extraction device, default cpu; no automatic GPU reservation.
            kinds: Model kinds to generate: esm2, esmc and/or saprot. None preserves the historical
                all-model bundle. Unselected weights are neither checked nor loaded.
            static_tables: Export each selected model's input table, default true for historical
                bundles. False generates contextual residue vectors only.

        Raises:
            ValueError: Missing model, changed files, unsupported identity, or invalid local layout.
            KeyError: A selected model is absent from the manifest.
            OSError: A selected checkpoint or its manifest cannot be read.
        """
        self.root = root
        self.maximum_length, self.layer, self.device = maximum_length, layer, device
        manifest = json.loads((root / "checkpoints.json").read_text())
        if kinds is None and set(manifest["models"]) != {kind.value for kind in ExternalModelKind}:
            raise ValueError("protein_models requires exactly esm2, esmc and saprot checkpoints")
        selected = tuple(ExternalModelKind(kind).value for kind in (
            manifest["models"] if kinds is None else kinds))
        if not selected or len(set(selected)) != len(selected):
            raise ValueError("protein model selection must be nonempty and contain no duplicates")
        self.models = {kind: manifest["models"][kind] for kind in selected}
        self.static_tables = static_tables
        for kind, spec in self.models.items():
            directory = root / Path(spec["checkpoint"]).name
            if {path.name for path in directory.iterdir() if path.is_file()} != set(spec["files"]):
                raise ValueError(f"checkpoint file inventory changed for {kind}")
            for name, entry in spec["files"].items():
                path = directory / name
                if path.stat().st_size != entry["bytes"] or ProteinArchive.sha256(path) != (
                    entry["sha256"]
                ):
                    raise ValueError(f"checkpoint byte integrity mismatch for {kind}/{name}")
            spec["resolved_checkpoint"] = directory

    @property
    def requires_structure_tokens(self) -> bool:
        """Report whether selected residue extraction needs official Foldseek 3Di.

        Returns:
            True only when SaProt is selected. ESM-2/ESM-C alone need no structural tokens.
        """
        return "saprot" in self.models

    @property
    def requested_families(self) -> frozenset[str]:
        """Return only the explicitly selected model-qualified mandatory assets.

        Returns:
            One contextual embedding per selected model, plus its input table when enabled.
        """
        families = (("external_embeddings", "static_embedding_table") if self.static_tables
                    else ("external_embeddings",))
        return frozenset(f"{family}_{kind}" for kind in self.models for family in families)

    def extractor(self, kind: str) -> ProteinRepresentationExtractor:
        """Load exactly one locally verified frozen checkpoint through public Transformers APIs.

        Args:
            kind: One reviewed manifest model kind: esm2, esmc or saprot.

        Returns:
            Frozen extractor; no downloads, remote code, training or task labels are involved.
        """
        spec = self.models[kind]
        return ProteinRepresentationExtractor(kind, spec["resolved_checkpoint"], spec["identifier"],
            spec["revision"], self.layer, self.maximum_length, self.device)

    def preflight(self) -> None:
        """Check actual tokenizer/model loading before expensive geometry, without inference.

        Raises:
            ImportError: Required public model/tokenizer implementation is absent.
            ValueError: A staged checkpoint has incompatible architecture or vocabulary.
        """
        for kind in self.models:
            extractor = self.extractor(kind)
            if self.static_tables:
                extractor.static_table()
            del extractor

    def generate(self, work: lf.Work, bases: Mapping[str, Path],
                 assets: Mapping[str, Mapping[str, Path]]) -> dict[str, dict[str, Path]]:
        """Extract selected models sequentially using native per-protein file checkpoints.

        Args:
            work: Active LF Work supplying managed files, bounded map, logs and reports.
            bases: Exact universal archive per immutable member ID.
            assets: Validated sequence_context per member, plus structure_tokens for SaProt only.

        Returns:
            Selected model-qualified file paths per member. A failed record stops publication after
            retaining every successful checkpoint; no model may overwrite another model's files.

        Raises:
            RuntimeError: At least one member/model cannot produce its mandatory assets.
        """
        result: dict[str, dict[str, Path]] = {identifier: {} for identifier in bases}
        failures: list[dict[str, str]] = []
        for kind in self.models:
            content = "embeddings and static input weights" if self.static_tables else "embeddings"
            work.log(f"Frozen {kind}: extracting {content} for "
                     f"{len(bases)} proteins (one CPU/GPU inference at a time)")
            producer = ExternalProteinRepresentation(self.extractor(kind))

            def extract(identifier: str, kind: str = kind,
                        producer: ExternalProteinRepresentation = producer) -> dict[str, str]:
                """Checkpoint one model/member without retaining other members' neural states.

                Args:
                    identifier: Exact member ID in bases and the aligned structural asset mapping.
                    kind: Model kind bound to this completed sequential map.
                    producer: Frozen producer bound before scheduling; never shared across models.

                Returns:
                    Qualified validated files, or an explicit error retained in the batch report.
                """
                try:
                    base = bases[identifier]
                    sequence = assets[identifier]["sequence_context"]
                    tokens = assets[identifier]["structure_tokens"] if kind == "saprot" else None
                    paths = {}
                    families = [(False, "external_embeddings")]
                    if self.static_tables:
                        families.append((True, "static_embedding_table"))
                    for static, prefix in families:
                        family = f"{prefix}_{kind}"
                        dependencies = {"base": ProteinArchive.sha256(base),
                            "sequence": ProteinArchive.sha256(sequence),
                            "structure_tokens": ProteinArchive.sha256(tokens) if tokens else None,
                            "parameters": producer.parameters, "family": family}
                        key = hashlib.sha256(json.dumps(dependencies, sort_keys=True).encode()
                                             ).hexdigest()
                        paths[family] = str(work.checkpoints.file(
                            f"protein-model-assets/{key}/{family}.npz",
                            build=partial(producer.write, base, sequence=sequence,
                                structure_tokens=tokens, static=static, family=family),
                            validate=partial(producer.validate, base=base, static=static,
                                             family=family)))
                    return paths
                except Exception as error:
                    return {"error": f"{type(error).__name__}: {error}"}

            outputs = work.map(list(bases), extract, workers=1, executor="thread",
                               name=f"frozen-{kind}")
            for identifier, output in zip(bases, outputs, strict=True):
                if "error" in output:
                    failures.append({"identifier": identifier, "model": kind,
                                     "error": output["error"]})
                else:
                    result[identifier].update({name: Path(path) for name, path in output.items()})
            model_failures = sum("error" in output for output in outputs)
            work.log(f"Frozen {kind}: {len(outputs) - model_failures}/{len(bases)} members "
                     f"complete; {model_failures} failures; valid file checkpoints retained")
            del extract, producer
        report = {"members": len(bases), "models": list(self.models), "failures": failures,
                  "required_assets": sorted(self.requested_families), "complete": not failures}
        root = Path(work.outputs.directory("protein-model-report", role="report"))
        ProteinArchive.write_report(root / "report.json", report)
        if failures:
            raise RuntimeError(f"frozen protein model extraction failed: {json.dumps(failures)}")
        return result
