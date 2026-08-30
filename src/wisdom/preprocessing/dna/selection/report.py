"""Write the portable dataset-design contract and its human report."""

import csv
import json
import shutil

from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence


def write_design(
    root           : Path,
    raw            : Sequence[Mapping[str, Any]],
    selected       : Sequence[Mapping[str, Any]],
    leakage        : Mapping[str, Any],
    phenotypes     : Mapping[str, Any],
    dilutions      : Mapping[str, Any],
    selection_audit: Mapping[str, Any],
    split_audit    : Mapping[str, Any],
    audit          : Mapping[str, Any],
    similarity     : Mapping[str, Any],
    parameters     : Mapping[str, Any],
) -> None:
    """Serialize every selection result needed by preprocessing or independent review.

    Args:
        root: LambdaForge-managed output directory, including the coordinate snapshot when present.
        raw: Full structurally analysed evidence population.
        selected: Canonical balanced population with fixed splits.
        leakage: Exact pairs, specialist edges, and connected components.
        phenotypes: Global and positive-interface clustering diagnostics.
        dilutions: Nested training-view memberships.
        selection_audit: Canonical balancing decisions and omissions.
        split_audit: Whole-group split decisions and counts.
        audit: Final hard-invariant verdict and descriptive statistics.
        similarity: Managed raw MMseqs2 and Foldseek table paths.
        parameters: Scientific thresholds needed to reproduce leakage decisions.

    Raises:
        OSError: If a portable output cannot be written.
    """
    root.mkdir(parents=True, exist_ok=True)
    selected_by_id = {str(row["identifier"]): dict(row) for row in selected}

    # catalog-all.csv retains every candidate. catalog.csv is the smaller authoritative contract
    # consumed by structural preprocessing, so no downstream step has to repeat dataset selection.

    raw_catalog: list[dict[str, Any]] = []
    for original in raw:
        identifier = str(original["identifier"])
        row = _portable_row(original)
        row["selected"] = identifier in selected_by_id
        row["split"]    = selected_by_id.get(identifier, {}).get("split", "")
        raw_catalog.append(row)
    canonical = [_portable_row({**row, "selected": True}) for row in selected]
    _write_csv(root / "catalog-all.csv", raw_catalog)
    _write_csv(root / "catalog.csv", canonical)
    _write_fasta(root / "selected.fasta", selected)

    # Paired manifests make manual use convenient while the catalog retains assembly, contact,
    # provenance, grouping, and phenotype fields that cannot fit safely into two columns.

    _write_manifests(root, "proteins", selected)
    for split in ("train", "validation", "test"):
        _write_manifests(
            root,
            split,
            [row for row in selected if row["split"] == split],
        )

    # The three downstream manifests are self-contained. A two-column TXT remains convenient for
    # people, but it cannot preserve the selected biological assembly, chain copy, DNA contacts,
    # leakage group, or phenotype. JSONL keeps one complete protein record per readable line and
    # lets Preprocessing consume exactly three files rather than the complete design directory.

    _write_preprocessing_manifests(root, selected, dilutions)

    clusters = root / "clusters"
    clusters.mkdir()
    shutil.copyfile(Path(similarity["sequence_path"]), clusters / "sequence-pairs.tsv")
    shutil.copyfile(Path(similarity["structure_path"]), clusters / "structure-pairs.tsv")
    _write_pairs(clusters / "sequence-edges.csv", leakage["sequence_edges"])
    _write_pairs(clusters / "structure-edges.csv", leakage["structure_edges"])
    _write_csv(
        clusters / "exact-pairs.csv",
        leakage["exact_pairs"],
        fields=("left", "right", "reasons"),
    )
    _write_csv(
        clusters / "leakage-groups.csv",
        [
            {"identifier": row["identifier"], "leakage_group": row["leakage_group"]}
            for row in raw
        ],
    )
    _write_csv(
        clusters / "global-phenotypes.csv",
        [
            {
                "identifier":       row["identifier"],
                "global_phenotype": row["global_phenotype"],
                "probability":      row["global_phenotype_probability"],
            }
            for row in raw
        ],
    )
    _write_csv(
        clusters / "positive-interface-phenotypes.csv",
        [
            {
                "identifier":          row["identifier"],
                "interface_phenotype": row["interface_phenotype"],
                "probability":         row["interface_phenotype_probability"],
            }
            for row in raw
            if int(row["label"]) == 1
        ],
    )

    # Dilutions contain complete train leakage groups. Their prefixes are nested, while the fixed
    # validation/test hashes prove that learning-curve experiments use the same evaluation sets.

    by_id = {str(row["identifier"]): row for row in selected}
    for replicate, subsets in dilutions["replicates"].items():
        directory = root / "dilutions" / replicate
        directory.mkdir(parents=True)
        for name, subset in subsets.items():
            subset_rows = [by_id[identifier] for identifier in subset["identifiers"]]
            _write_manifests(directory, name, subset_rows)

    selection_record = {
        **selection_audit,
        "quality_exclusions": [
            {
                "identifier": row["identifier"],
                "reason":     row["quality_exclusion_reason"],
            }
            for row in raw
            if not bool(row["quality_eligible"])
        ],
    }
    _write_json(root / "selection-audit.json", selection_record)
    _write_json(root / "split-audit.json", split_audit)
    _write_json(root / "dilution-audit.json", dilutions)
    _write_json(root / "phenotype-audit.json", phenotypes)
    _write_json(root / "design-summary.json", audit)

    leakage_criteria = {
        key: value
        for key, value in parameters.items()
        if key.startswith("sequence_") or key.startswith("foldseek_")
    }

    fixed_evaluation = {
        "test_sha256":       dilutions["test_sha256"],
        "validation_sha256": dilutions["validation_sha256"],
    }

    _write_json(
        root / "provenance.json",
        {
            "design_schema_version": "1.3",
            "raw_records_format":    "jsonl-1.0",
            "seed":                  parameters["seed"],
            "clustering":            phenotypes,
            "dilutions":             fixed_evaluation,
            "parameters":            dict(parameters),
            "leakage_criteria":      leakage_criteria,
        },
    )

    omitted = selection_audit.get("omitted", [])
    _write_lines(
        root / "omitted-positives.txt",
        [str(row["identifier"]) for row in omitted if int(row["label"]) == 1],
    )
    exclusions = [
        f"{row['identifier']}\t{row['quality_exclusion_reason']}"
        for row in raw
        if not bool(row["quality_eligible"])
    ]
    _write_lines(root / "quality-exclusions.txt", exclusions)
    (root / "REPORT.md").write_text(
        _markdown_report(audit, selection_audit, split_audit, phenotypes, dilutions),
        encoding="utf-8",
    )


def _portable_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Remove attempt-local objects while preserving scientific values.

    Args:
        row: In-memory structural row, possibly carrying a managed Foldseek path.

    Returns:
        CSV-safe mapping with no machine-specific working path.
    """
    transient_fields = {"foldseek_structure", "source_structure"}
    return {key: value for key, value in row.items() if key not in transient_fields}


def _write_manifests(
    root: Path,
    name: str,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write ID-only, labelled, and FASTA views of one population.

    Args:
        root: Destination directory.
        name: Common manifest basename.
        rows: Protein rows to serialize in identifier order.
    """
    ordered = sorted(rows, key=lambda row: str(row["identifier"]))
    _write_lines(root / f"{name}.txt", [str(row["identifier"]) for row in ordered])
    _write_lines(
        root / f"{name}-labelled.txt",
        [f"{row['identifier']}\t{int(row['label'])}" for row in ordered],
    )
    _write_fasta(root / f"{name}.fasta", ordered)


def _write_preprocessing_manifests(
    root     : Path,
    rows     : Sequence[Mapping[str, Any]],
    dilutions: Mapping[str, Any],
) -> None:
    """Write the three complete records consumed by structural preprocessing.

    Args:
        root: Selection output directory.
        rows: Canonical proteins with fixed labels, splits, and design metadata.
        dilutions: Nested training memberships keyed by replicate and fraction.
    """
    directory = root / "preprocessing"
    directory.mkdir()

    dilution_memberships: dict[str, list[str]] = {}
    for replicate, subsets in dilutions["replicates"].items():
        for subset_name, subset in subsets.items():
            view = f"{replicate}/{subset_name}"
            for identifier in subset["identifiers"]:
                dilution_memberships.setdefault(str(identifier), []).append(view)

    fields = (
        "identifier",
        "label",
        "split",
        "leakage_group",
        "global_phenotype",
        "interface_phenotype",
        "origin",
        "label_evidence",
        "pdb_id",
        "protein_chain",
        "assembly_id",
        "protein_copy",
        "structure_sha256",
        "dna_chains",
        "binding_residue_indices",
        "local_gt_expected",
        "local_gt_method",
        "assembly_rotation",
        "assembly_translation",
    )
    filenames = {
        "train":      "train.jsonl",
        "validation": "val.jsonl",
        "test":       "test.jsonl",
    }
    for split, filename in filenames.items():
        records = []
        for row in sorted(rows, key=lambda value: str(value["identifier"])):
            if row["split"] != split:
                continue
            record              = {field: row[field] for field in fields}
            record["dilutions"] = sorted(dilution_memberships.get(str(row["identifier"]), ()))
            records.append(record)
        _write_jsonl(directory / filename, records)


def _write_fasta(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write one conventional two-line FASTA record per protein.

    Args:
        path: Destination FASTA file.
        rows: Protein identifiers and sequences.
    """
    path.write_text(
        "".join(f">{row['identifier']}\n{row['sequence']}\n" for row in rows),
        encoding="utf-8",
    )


def _write_pairs(path: Path, pairs: Sequence[Sequence[str]]) -> None:
    """Write canonical undirected edges with an explicit header.

    Args:
        path: Destination CSV file.
        pairs: Two-identifier edge sequence.
    """
    _write_csv(
        path,
        [{"left": left, "right": right} for left, right in pairs],
        fields=("left", "right"),
    )


def _write_csv(
    path  : Path,
    rows  : Sequence[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    """Write a stable union-schema CSV and JSON-encode nested values.

    Args:
        path: Destination CSV path.
        rows: Record mappings whose complete key union forms the header.
        fields: Explicit header for a table that may legitimately have zero rows.
    """
    inferred = {str(key) for row in rows for key in row}
    header   = list(fields) if fields is not None else sorted(inferred)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=header)
        writer.writeheader()
        for row in sorted(rows, key=lambda value: str(value.get("identifier", ""))):
            writer.writerow(
                {
                    field: json.dumps(row.get(field), sort_keys=True)
                    if isinstance(row.get(field), (dict, list, tuple, set))
                    else row.get(field, "")
                    for field in header
                }
            )


def _write_json(path: Path, value: Any) -> None:
    """Write deterministic, readable JSON.

    Args:
        path: Destination JSON path.
        value: JSON-compatible value.
    """
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    """Write deterministic one-record-per-line JSON.

    Args:
        path: Destination JSONL path.
        rows: Ordered JSON-compatible protein records.
    """
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _write_lines(path: Path, lines: Sequence[str]) -> None:
    """Write newline-terminated text records.

    Args:
        path: Destination text path.
        lines: Logical records without trailing newlines.
    """
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")


def _markdown_report(
    audit          : Mapping[str, Any],
    selection_audit: Mapping[str, Any],
    split_audit    : Mapping[str, Any],
    phenotypes     : Mapping[str, Any],
    dilutions      : Mapping[str, Any],
) -> str:
    """Explain the generated design and interpret its main statistics.

    Args:
        audit: Final invariant and composition audit.
        selection_audit: Population-balancing audit.
        split_audit: Whole-group split audit.
        phenotypes: HDBSCAN diagnostics.
        dilutions: Nested learning-curve memberships.

    Returns:
        Self-contained English Markdown report.
    """
    counts = audit["selected_counts"]
    lines = [
        "# WISDOM-DNA dataset-design report",
        "",
        "## 1. Verdict",
        "",
        "**PASS.** Every selected protein has one explicit evidence label, every transitive "
        "sequence/structure leakage group stays inside one split, and every training dilution "
        "contains complete groups only.",
        "",
        f"The canonical dataset contains **{counts['total']} proteins**: "
        f"{counts['positive']} DNA-binding positives and {counts['negative']} curated negatives. "
        "A positive has verified heavy-atom contact with DNA in its declared biological assembly. "
        "A negative comes from explicit experimental benchmark evidence; absence of DNA in a PDB "
        "entry was never treated as a negative label.",
        "",
        "## 2. What the selection did",
        "",
        "1. It verified the frozen label, chain, assembly, sequence, and structural coordinates.",
        "2. It compared all RAW candidates with MMseqs2 (sequence) and Foldseek (3D structure).",
        "3. It formed transitive leakage groups: if A resembles B and B resembles C, all three "
        "remain together even when A and C are not directly linked.",
        "4. It clustered label-free physical descriptors with HDBSCAN. HDBSCAN may mark unusual "
        "proteins as noise instead of forcing them into a misleading phenotype.",
        "5. It balanced the two classes and assigned complete groups to train, validation, "
        "and test.",
        "6. It created nested train-only dilutions; validation and test never change.",
        "",
        "## 3. Canonical splits",
        "",
        "| Split | Total | Positive | Negative | Leakage groups |",
        "|---|---:|---:|---:|---:|",
    ]
    for split in ("train", "validation", "test"):
        values = audit["splits"][split]
        lines.append(
            f"| {split} | {values['total']} | {values['positive']} | "
            f"{values['negative']} | {values['leakage_groups']} |"
        )
    lines.extend(
        [
            "",
            "Exact 50/50 balance inside every split can be impossible because a leakage group is "
            "indivisible. The important hard result is zero cross-split groups; small count "
            "deviations are preferable to homologous leakage.",
            "",
            "## 4. Physical phenotypes",
            "",
            f"Global clustering used {phenotypes['global']['eligible']} complete records and found "
            f"{phenotypes['global']['clusters']} non-noise clusters; its noise fraction is "
            f"{phenotypes['global']['noise_fraction']:.1%}. Interface clustering used "
            f"{phenotypes['interface']['eligible']} verified positives and found "
            f"{phenotypes['interface']['clusters']} clusters, with "
            f"{phenotypes['interface']['noise_fraction']:.1%} noise.",
            "",
            "Here, *noise* does not mean corrupt data. It means that HDBSCAN did not find enough "
            "nearby proteins to claim a stable density-based family. Phenotypes are used only to "
            "preserve physical variety; they do not define DNA-binding labels.",
            "",
            "## 5. Training dilutions",
            "",
        ]
    )
    first_replicate = next(iter(dilutions["replicates"].values()))
    for name, subset in sorted(first_replicate.items(), key=lambda item: item[1]["fraction"]):
        values = subset["counts"]
        lines.append(
            f"- `{name}` contains {values['total']} training proteins "
            f"({values['positive']} positive, {values['negative']} negative) in "
            f"{len(subset['leakage_groups'])} complete leakage groups."
        )
    lines.extend(
        [
            "",
            "The subsets are nested: a protein present at a smaller fraction remains present at "
            "every larger fraction. Their fixed validation and test SHA-256 values are recorded in "
            "`dilution-audit.json`.",
            "",
            "## 6. Warnings and file guide",
            "",
        ]
    )
    warnings = audit.get("warnings", [])
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- No descriptive warning crossed its reporting threshold.")
    lines.extend(
        [
            "",
            "`catalog.csv` is the authoritative canonical table. The `*-labelled.txt` files are "
            "compact `RCSB_CHAIN<TAB>0|1` views. The three files below `preprocessing/` add the "
            "exact assembly, copy, contact, leakage, phenotype, and dilution metadata required "
            "to reproduce NPZ files without staging the complete design. `catalog-all.csv` "
            "preserves excluded RAW evidence. "
            "`clusters/` contains raw and thresholded similarity evidence. JSON audit files carry "
            "machine-readable versions of the counts summarized here.",
            "",
            f"Selection retained {selection_audit['selected']['total']} proteins. The split method "
            f"was `{split_audit['method']}`.",
            "",
        ]
    )
    return "\n".join(lines)
