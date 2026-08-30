"""Assemble, validate, and publish the immutable WISDOM-DNA DatasetVersion."""

import csv
import json
import shutil
import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections.abc import Iterable, Mapping, Sequence

from wisdom.preprocessing.dna.DNAValidation import DNAValidation
from lambdaforge.data import DatasetAsset, DatasetIndex, DatasetMember


def publish_dataset(
    work               : lf.Work,
    rows               : Sequence[Mapping[str, Any]],
    annotation_root    : Path,
    geometry_report    : Path,
    annotation_report  : Path,
    dataset_name       : str,
    dataset_version    : str,
) -> dict[str, Any]:
    """Create the portable root, run scientific validation, and publish one dataset.

    Args:
        work: Active Work providing managed outputs, metrics, and Dataset publication.
        rows: Complete fixed split records.
        annotation_root: Self-contained base-NPZ, sidecar, and structure directory.
        geometry_report: Ordered universal geometry report.
        annotation_report: Ordered DNA sidecar report.
        dataset_name: Stable LambdaForge Registry family.
        dataset_version: Immutable version selected by the researcher.

    Returns:
        Dataset record, member/group counts, and scientific validation verdict.

    Raises:
        RuntimeError: If scientific validation fails before publication.
    """
    final_root = work.run_dir / "dataset"
    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.copytree(annotation_root, final_root)

    # Reconstruct only the compact audit views derivable from the three authoritative manifests.
    # Pairwise MMseqs2/Foldseek evidence stays with Selection; the final dataset carries the fixed
    # transitive leakage group on every member and rechecks that no group crosses a split.

    evidence = final_root / "design"
    _write_evidence(
        evidence,
        rows,
    )
    _write_index(final_root, rows, annotation_report)

    work.log("Running final checksum, NPZ, sidecar, split, group, and dilution validation")
    validation = DNAValidation().audit(final_root, work.run_dir / "dna-validation")
    if validation["verdict"] != "PASS":
        failed = [check for check in validation["checks"] if not check["passed"]]
        summary = ", ".join(
            f"{check['name']}={check['failure_count']}" for check in failed
        )
        work.log(f"Final scientific validation failed: {summary}", level="error")

        for detail in validation["member_failures"][:10]:
            work.log(
                f"Invalid member {detail['identifier']}: {detail['failure']}",
                level="error",
            )

        raise RuntimeError(
            "final WISDOM-DNA scientific validation failed before publication: " + summary
        )

    members = tuple(_published_members(final_root))
    record  = work.outputs.dataset(
        name          = dataset_name,
        version       = dataset_version,
        members       = members,
        output        = "dataset",
        metadata      = {
            "description":       "Leakage-safe DNA proteins with universal WISDOM surfaces",
            "structural_schema": "2.1",
            "annotation_schema": "1.3",
            "manifest_schema":   "1.0",
            "supervision":       "protein-level-only",
        },
        target_schema = {
            "type": "object",
            "properties": {
                "dna_binding":       {"type": "integer", "enum": [0, 1]},
                "local_ground_truth": {"type": "boolean"},
            },
            "required": ["dna_binding", "local_ground_truth"],
            "additionalProperties": False,
        },
    )
    work.outputs.artifact("preprocessing-report", geometry_report, role="report")
    work.outputs.artifact("annotation-report", annotation_report, role="report")
    work.outputs.artifact(
        "validation-report",
        work.run_dir / "dna-validation",
        role="report",
    )
    work.metrics.log("published_members", len(members))
    work.log(f"Published {dataset_name}@{dataset_version} with {len(members)} members")
    return {
        **record,
        "published_members":  len(members),
        "leakage_groups":     len({str(row["leakage_group"]) for row in rows}),
        "validation_verdict": validation["verdict"],
    }


def _write_evidence(
    root: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Materialize compact, human-readable views from the three exact manifests.

    Args:
        root: Final dataset evidence directory.
        rows: Decoded manifest records.
    """
    root.mkdir(parents=True)
    preprocessing = root / "preprocessing"
    preprocessing.mkdir()
    for split, name in (
        ("train",      "train.jsonl"),
        ("validation", "val.jsonl"),
        ("test",       "test.jsonl"),
    ):
        selected = sorted(
            (row for row in rows if str(row["split"]) == split),
            key=lambda row: str(row["identifier"]),
        )
        (preprocessing / name).write_text(
            "".join(
                json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
                for row in selected
            ),
            encoding="utf-8",
        )

    _write_csv(root / "catalog.csv", rows)
    _write_manifest_views(root, "proteins", rows)
    for split in ("train", "validation", "test"):
        _write_manifest_views(
            root,
            split,
            [row for row in rows if row["split"] == split],
        )

    dilution_rows: dict[str, list[Mapping[str, Any]]] = {}
    for row in rows:
        for view in row.get("dilutions", ()):
            dilution_rows.setdefault(str(view), []).append(row)
    for view, selected in dilution_rows.items():
        replicate, subset = view.split("/", 1)
        directory         = root / "dilutions" / replicate
        directory.mkdir(parents=True, exist_ok=True)
        _write_manifest_views(directory, subset, selected)

    (root / "provenance.json").write_text(
        json.dumps(
            {
                "design_schema_version": "1.3",
                "manifest_schema_version": "1.0",
                "selection_evidence": "verified_by_upstream_selection_work",
                "published_inputs": ["train.jsonl", "val.jsonl", "test.jsonl"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "REPORT.md").write_text(
        "# WISDOM-DNA preprocessing manifest\n\n"
        "The three JSONL files preserve the exact normalized population used for geometry "
        "generation, whether preprocessing received complete JSONL or labelled TXT plus its "
        "catalog. Each line contains the binary label, split, selected biological assembly and "
        "chain copy, local DNA evidence, transitive leakage group, physical phenotypes, and "
        "training-dilution memberships. MMseqs2/Foldseek pair evidence and selection statistics "
        "remain in the separately managed Selection output.\n",
        encoding="utf-8",
    )


def _write_manifest_views(
    root: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write deterministic ID-only and two-column labelled views.

    Args:
        root: Destination directory.
        name: Common output basename.
        rows: Protein records included in the view.
    """
    ordered = sorted(rows, key=lambda row: str(row["identifier"]))
    (root / f"{name}.txt").write_text(
        "".join(f"{row['identifier']}\n" for row in ordered),
        encoding="utf-8",
    )
    (root / f"{name}-labelled.txt").write_text(
        "".join(f"{row['identifier']}\t{int(row['label'])}\n" for row in ordered),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write a deterministic CSV compatibility view of JSONL records.

    Args:
        path: Destination catalog path.
        rows: Protein mappings with a shared preprocessing schema.
    """
    fields = sorted({str(field) for row in rows for field in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: str(value["identifier"])):
            writer.writerow(
                {
                    field: json.dumps(row.get(field), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple))
                    else row.get(field, "")
                    for field in fields
                }
            )


def _write_index(
    root             : Path,
    rows             : Sequence[Mapping[str, Any]],
    annotation_report: Path,
) -> None:
    """Join fixed manifest metadata to validated NPZ and structure assets.

    Args:
        root: Self-contained pre-publication dataset root.
        rows: Decoded split records.
        annotation_report: Sidecar report containing portable asset paths and hashes.

    Raises:
        ValueError: If annotation and manifest membership differ.
    """
    report      = json.loads(annotation_report.read_text(encoding="utf-8"))
    annotations = {str(row["identifier"]): row for row in report["records"]}
    if set(annotations) != {str(row["identifier"]) for row in rows}:
        raise ValueError("annotation report and preprocessing manifests have different members")

    members = []
    for row in rows:
        identifier = str(row["identifier"])
        annotation = annotations[identifier]
        base        = root / str(annotation["portable_base_path"])
        sidecar     = root / str(annotation["output"])
        structure   = root / "structures" / f"{annotation['source_structure_sha256']}.cif"
        members.append(
            DatasetMember(
                member_id = identifier,
                partitions = {
                    "split":               str(row["split"]),
                    "leakage_group":       str(row["leakage_group"]),
                    "global_phenotype":    str(row["global_phenotype"]),
                    "interface_phenotype": str(row["interface_phenotype"]),
                },
                targets = {
                    "dna_binding":       int(row["label"]),
                    "local_ground_truth": bool(annotation["local_gt_available"]),
                },
                metadata = {
                    "origin":        str(row["origin"]),
                    "pdb_id":        str(row["pdb_id"]),
                    "protein_chain": str(row["protein_chain"]),
                    "assembly_id":   str(row["assembly_id"]),
                    "protein_copy":  int(row["protein_copy"]),
                    "dilutions":     list(row.get("dilutions", ())),
                },
                assets = {
                    "universal_npz": DatasetAsset(
                        path        = base.relative_to(root).as_posix(),
                        sha256      = f"sha256:{annotation['base_npz_sha256']}",
                        size_bytes  = base.stat().st_size,
                        media_type  = "application/x-npz",
                    ),
                    "dna_annotation": DatasetAsset(
                        path        = sidecar.relative_to(root).as_posix(),
                        sha256      = f"sha256:{annotation['sidecar_sha256']}",
                        size_bytes  = sidecar.stat().st_size,
                        media_type  = "application/x-npz",
                    ),
                    "source_structure": DatasetAsset(
                        path        = structure.relative_to(root).as_posix(),
                        sha256      = f"sha256:{annotation['source_structure_sha256']}",
                        size_bytes  = structure.stat().st_size,
                        media_type  = "chemical/x-mmcif",
                    ),
                },
            )
        )
    DatasetIndex.write(root / "members.jsonl", members)


def _published_members(root: Path) -> Iterable[Mapping[str, Any]]:
    """Translate the validated index into LambdaForge streaming members.

    Args:
        root: Self-contained pre-publication root with member index and assets.

    Yields:
        Member mappings accepted by ``Work.outputs.dataset``.
    """
    for index, member in enumerate(DatasetIndex(root / "members.jsonl")):
        assets = {name: root / asset.path for name, asset in member.assets.items()}
        if index == 0:
            assets["dataset_design"] = root / "design"
        yield {
            "id":         member.member_id,
            "partitions": dict(member.partitions),
            "targets":    dict(member.targets),
            "metadata":   dict(member.metadata),
            "display":    dict(member.display),
            "assets":     assets,
        }
