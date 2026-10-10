"""Independent scientific audit of immutable zinc member assets and split provenance."""

import gzip
import json
import math
import hashlib
import numpy as np
import lambdaforge as lf

from typing import Any
from pathlib import Path
from contextlib import suppress
from lambdaforge.data import DatasetIndex
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.zinc.evidence import load_evidence
from wisdom.preprocessing.zinc.audit import write_diversity_report
from wisdom.preprocessing.zinc.ZincAnnotation import ZincAnnotation
from wisdom.utils.structure.ProteinStructure import ProteinStructure
from wisdom.features.SurfaceFeatureSidecar import SurfaceFeatureSidecar
from wisdom.preprocessing.zinc.ZincCoordination import ZincCoordination
from wisdom.preprocessing.common.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig


class ZincValidation(lf.Work):
    """Keep byte integrity distinct from evidence, topology and reference-label validation."""

    def run(self, dataset: Path, fail_on_error: bool = True) -> dict[str, Any]:
        """Publish a human verdict and detailed JSON before raising on hard failures.

        Args:
            dataset: Native typed managed dataset placement.
            fail_on_error: Raise after report registration when the audit fails, default true.

        Returns:
            Counts, verdict and ordered per-protein failures.

        Raises:
            RuntimeError: Scientific failures with fail_on_error enabled.
            OSError: Missing/unreadable member index or report destination.
        """
        output = Path(self.outputs.directory("zinc-validation", role="report"))
        result = self.audit(dataset, output)
        self.metrics.log("validation_failures", len(result["failures"]))
        if fail_on_error and result["verdict"] != "PASS":
            raise RuntimeError("Zn scientific validation failed; inspect zinc-validation report")
        return result

    def audit(self, dataset: Path, output: Path) -> dict[str, Any]:
        """Audit every universal archive, sidecar, source digest, negative and leakage group.

        Args:
            dataset: Index root, including exact member assets and selection evidence.
            output: Writable audit destination, outside immutable dataset assets.

        Returns:
            PASS/FAIL and exact per-protein errors. Unavailable local metrics remain unavailable.

        Raises:
            OSError: Unreadable index/output; member-specific failures are recorded, not hidden.
        """
        members = [{"id": member.member_id, "partitions": dict(member.partitions),
                    "targets": dict(member.targets), "metadata": dict(member.metadata),
                    "assets": {key: dataset / asset.path for key, asset in member.assets.items()},
                    "checksums": {key: asset.sha256 for key, asset in member.assets.items()
                                  if asset.kind == "file" and asset.sha256}}
                   for member in DatasetIndex(dataset / "index.jsonl")]
        return self.audit_members(members, output)

    def audit_members(
        self, members: Sequence[Mapping[str, Any]], output: Path,
    ) -> dict[str, Any]:
        """Validate resolved native member mappings before or after Registry publication.

        Args:
            members: Native publication-style mappings with id, assets as Paths, partitions,
                targets, metadata and optional per-file checksums. No physical paths are persisted.
            output: Human/machine report directory outside immutable source assets.

        Returns:
            Ordered verdict, split counts and per-protein errors without copying large NPZ files.

        Raises:
            OSError: Unwritable report destination; individual asset failures are recorded.
        """
        output.mkdir(parents=True, exist_ok=True)
        failures: list[dict[str, str]] = []
        groups: dict[str, set[str]] = defaultdict(set)
        counts: dict[str, Counter[int]] = defaultdict(Counter)
        identities: set[str] = set()
        sequence_groups: dict[str, set[str]] = defaultdict(set)
        views: dict[str, set[str]] = defaultdict(set)
        train_groups: dict[str, set[str]] = defaultdict(set)
        train_sources: dict[str, str] = {}
        normalization_populations: dict[str, dict[str, str]] = {"full": train_sources}
        verified_rows: list[dict[str, Any]] = []
        local_support: list[dict[str, Any]] = []
        if any(member["metadata"].get("optional_surface_feature_assets") for member in members):
            for member in members:
                base = member["assets"].get("universal_npz")
                if member["partitions"]["split"] == "train" and base is not None:
                    # Missing inputs are recorded by the normal ordered per-member audit.

                    with suppress(OSError):
                        train_sources[member["id"]] = hashlib.sha256(base.read_bytes()).hexdigest()
                        for view in member["metadata"].get("dilutions", ()):
                            normalization_populations.setdefault(str(view), {})[member["id"]] = (
                                train_sources[member["id"]]
                            )
        for member in members:
            try:
                if member["id"] in identities:
                    raise ValueError("duplicate member identity")
                identities.add(member["id"])
                split = str(member["partitions"]["split"])
                label = int(member["targets"]["zinc_binding"])
                if label not in (0, 1) or split not in ("train", "validation", "test"):
                    raise ValueError("invalid binary target or supervised split")
                groups[str(member["partitions"]["leakage_group"])].add(split)
                counts[split][label] += 1
                group = str(member["partitions"]["leakage_group"])
                if split == "train":
                    train_groups[group].add(member["id"])
                for view in member["metadata"].get("dilutions", ()):
                    if split != "train":
                        raise ValueError("train dilution contains an evaluation protein")
                    replicate, population = str(view).split("/", 1)
                    percentage = float(population.removeprefix("train-"))
                    if not replicate.startswith("replicate-") or not population.startswith(
                        "train-"
                    ) or not math.isfinite(percentage) or not 0 < percentage <= 100:
                        raise ValueError("invalid train dilution name or percentage")
                    views[str(view)].add(member["id"])
                paths = member["assets"]
                for name, checksum in member.get("checksums", {}).items():
                    actual = hashlib.sha256(paths[name].read_bytes()).hexdigest()
                    if actual != checksum.removeprefix("sha256:"):
                        raise ValueError(f"asset checksum mismatch: {name}")
                base, annotation = paths["universal_npz"], paths["zinc_annotation"]
                with np.load(base, allow_pickle=False) as archive:
                    arrays = {name: archive[name] for name in archive.files}
                metadata = json.loads(str(arrays["metadata_json"].item()))
                if metadata["preprocessing_schema_version"] != "3.0":
                    raise ValueError("Zn requires universal structural schema 3.0")
                ProteinArchive(PreprocessConfig(**metadata["config"])).validate(arrays)
                if np.any(arrays["atomic_numbers"] == 30):
                    raise ValueError("Zn atom leaked into universal inputs")
                ZincAnnotation.validate(annotation, base)
                # Optional fields carry their own point-order binding and exactly the published
                # training population. An asset claiming train-only without those IDs is unsafe.

                feature_names: list[str] = []
                for name in member["metadata"].get("optional_surface_feature_assets", ()):
                    with np.load(paths[name], allow_pickle=False) as fields:
                        names = fields["feature_names"].tolist()
                    SurfaceFeatureSidecar.read(paths[name], base, names)
                    if set(feature_names) & set(names):
                        raise ValueError("duplicate optional surface feature channel")
                    feature_names.extend(names)
                if feature_names:
                    statistics = json.loads(paths["surface_feature_statistics"].read_text())
                    SurfaceFeatureSidecar.validate_statistics(
                        statistics, feature_names, normalization_populations)
                row = member["metadata"]["zinc_evidence"]
                with np.load(annotation, allow_pickle=False) as sidecar:
                    encoded_metadata = str(sidecar["annotation_metadata_json"].item())
                    annotation_metadata = json.loads(encoded_metadata)
                for key in ("identifier", "label", "assembly_id", "protein_chain", "protein_copy"):
                    if annotation_metadata[key] != row[key]:
                        raise ValueError(f"Zn annotation evidence disagrees with selection: {key}")
                if annotation_metadata["source_structure_sha256"] != row["structure_sha256"]:
                    raise ValueError("Zn annotation refers to another source deposition")
                # Reuse the explicit evidence parser rather than trusting a binary index target.

                evidence_path = output / "checked-evidence.jsonl"
                evidence_path.write_text(json.dumps(row) + "\n")
                load_evidence(evidence_path)
                if int(row["label"]) != label or row["split"] != split:
                    raise ValueError("selection label/split disagrees with member targets")
                source = paths["source_structure"].read_bytes()
                # Registry assets can lose filename suffixes. Detect only the gzip container
                # signature, then let Gemmi parse the exact uncompressed scientific payload.

                if source.startswith(b"\x1f\x8b"):
                    source = gzip.decompress(source)
                if hashlib.sha256(source).hexdigest() != row["structure_sha256"]:
                    raise ValueError("exact selection structure snapshot disagrees with evidence")
                structure = ProteinStructure(paths["source_structure"], mmcif_bytes=source)
                assembly = structure.assembly(str(row["assembly_id"]))
                deposited, _ = assembly.protein_copy(row["protein_chain"], int(row["protein_copy"]))
                sequence = structure.sequence(deposited)
                sequence_groups[hashlib.sha256(sequence.encode()).hexdigest()].add(group)
                if sequence != row["sequence"]:
                    raise ValueError("source protein sequence differs from selection evidence")
                if label == 0 and hashlib.sha256(sequence.encode()).hexdigest() != (
                    row["label_evidence"]["sequence_sha256"]
                ):
                    raise ValueError("negative evidence is bound to another protein sequence")
                sites = ZincCoordination(**row["coordination_parameters"]).analyse(
                    assembly, row["protein_chain"], int(row["protein_copy"]),
                )
                if not ZincCoordination.matches(sites, row["sites"]):
                    raise ValueError("assembly coordination evidence does not reproduce")
                if label == 0 and any(site["selected_donor_count"] for site in sites):
                    raise ValueError("negative evidence contradicts structural Zn contacts")
                if label == 1 and not any(site["accepted"] for site in sites):
                    raise ValueError("positive protein lacks verified selected-copy coordination")
                with np.load(annotation, allow_pickle=False) as sidecar:
                    available = bool(sidecar["local_gt_available"].item())
                    if bool(member["targets"]["local_ground_truth"]) != available:
                        raise ValueError("index local GT availability disagrees with sidecar")
                    if label == 1 and split != "train" and (
                        not available or not sidecar["surface_target_hard"].any()
                    ):
                        raise ValueError("positive evaluation member has no positive local GT")
                    if label == 0 and sidecar["surface_target_hard"].any():
                        raise ValueError("negative protein has a positive surface reference")
                    local_support.append({
                        "identifier": member["id"], "available": available,
                        "point_count": len(sidecar["surface_target_hard"]),
                        "positive_point_fraction": (
                            float(sidecar["surface_target_hard"].mean()) if available else None
                        ),
                    })
                verified_rows.append({**row, "leakage_group": group})
            except (KeyError, ValueError, TypeError, OSError, RuntimeError) as error:
                failures.append({"identifier": member["id"], "failure": str(error)})
        for group, splits in sorted(groups.items()):
            if len(splits) > 1:
                failures.append({"identifier": group, "failure": "leakage group crosses splits"})
        for split in ("train", "validation", "test"):
            if set(counts[split]) != {0, 1}:
                failures.append({"identifier": split, "failure": "split requires both classes"})
        for digest, assigned in sorted(sequence_groups.items()):
            if len(assigned) > 1:
                failures.append({"identifier": digest,
                                 "failure": "exact sequence spans leakage groups"})
        previous_by_replicate: dict[str, set[str]] = {}
        for view, identifiers in sorted(views.items(), key=lambda item: (
            item[0].split("/")[0], float(item[0].rsplit("-", 1)[-1]),
        )):
            replicate = view.split("/", 1)[0]
            if not previous_by_replicate.get(replicate, set()) <= identifiers:
                failures.append({"identifier": view, "failure": "train dilution is not nested"})
            if any(ids & identifiers and not ids <= identifiers for ids in train_groups.values()):
                failures.append({"identifier": view,
                                 "failure": "dilution fragments a leakage group"})
            previous_by_replicate[replicate] = identifiers
        # Post-geometry size/prevalence and class covariates are review signals, never exclusion
        # rules. Only verified members enter the descriptive report; its support is explicit.

        diversity = (write_diversity_report(output, verified_rows, verified_rows)
                     if verified_rows else {"warnings": [], "covariates": {}, "coverage": {}})
        warnings = list(diversity["warnings"])
        sizes = np.asarray([row["point_count"] for row in local_support], dtype=float)
        if len(sizes) >= 4:
            lower, upper = np.quantile(sizes, [0.25, 0.75])
            spread = upper - lower
            if spread > 0:
                for row in local_support:
                    if not lower - 1.5 * spread <= row["point_count"] <= upper + 1.5 * spread:
                        warnings.append(f"{row['identifier']}: surface size lies outside the "
                                        "1.5-IQR fence; inspect, do not discard automatically.")
        for row in local_support:
            fraction = row["positive_point_fraction"]
            if fraction is not None and fraction >= 0.5:
                warnings.append(f"{row['identifier']}: at least half of sampled points are "
                                "local-positive; inspect thresholds and site extent.")
        result = {"verdict": "FAIL" if failures else "PASS", "member_count": len(members),
                  "failures": failures,
                  "warnings": warnings, "descriptive_member_count": len(verified_rows),
                  "local_support": local_support, "diversity": diversity,
                  "splits": {name: dict(count) for name, count in counts.items()}}
        (output / "verdict.txt").write_text(f"Zn scientific validation: {result['verdict']}\n"
                                            f"Members: {len(members)}; failures: {len(failures)}\n")
        (output / "report.json").write_text(json.dumps(result, indent=2))
        return result
