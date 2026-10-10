"""Compute full-population sequence and structural similarity evidence."""

import math

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Mapping, Sequence
from lambdaforge.work import ManagedFile, Tool


def compute_similarity(
    work                 : Any,
    rows                 : Sequence[Mapping[str, Any]],
    mmseqs               : Tool,
    foldseek             : Tool,
    workers              : int,
    sequence_identity    : float,
    sequence_coverage    : float,
    sequence_evalue      : float,
    foldseek_probability : float,
    foldseek_tmscore     : float,
    foldseek_coverage    : float,
    foldseek_evalue      : float,
) -> dict[str, Any]:
    """Run or restore MMseqs2/Foldseek and threshold their auditable raw tables.

    Args:
        work: Active LambdaForge Work providing tools, temporary storage, and checkpoints.
        rows: Complete RAW structural rows.
        mmseqs: LambdaForge-resolved MMseqs2 executable.
        foldseek: LambdaForge-resolved Foldseek executable.
        workers: Threads assigned to each external program.
        sequence_identity: Minimum aligned sequence identity.
        sequence_coverage: Minimum query and target sequence coverage.
        sequence_evalue: Largest accepted MMseqs2 E-value.
        foldseek_probability: Minimum Foldseek probability.
        foldseek_tmscore: Minimum query and target normalized TM-score.
        foldseek_coverage: Minimum query and target structural coverage.
        foldseek_evalue: Largest accepted Foldseek E-value.

    Returns:
        Raw managed tables plus canonical sequence and structure edge sets.
    """
    parameters = {
        "sequence_evalue":      sequence_evalue,
        "foldseek_evalue":      foldseek_evalue,
        "sequence_identity":    sequence_identity,
        "sequence_coverage":    sequence_coverage,
        "foldseek_tmscore":     foldseek_tmscore,
        "foldseek_coverage":    foldseek_coverage,
        "foldseek_probability": foldseek_probability,
    }

    work.log("Running or restoring full-RAW MMseqs2 similarity")

    sequence_path = work.checkpoints.file(
        "similarity/sequence-pairs.tsv",
        build=partial(_run_mmseqs, work, rows, workers, parameters, mmseqs),
        validate=partial(_valid_table, columns=7),
    )

    work.log("Running or restoring full-RAW Foldseek similarity")

    structure_path = work.checkpoints.file(
        "similarity/structure-pairs.tsv",
        build=partial(_run_foldseek, work, rows, workers, parameters, foldseek),
        validate=partial(_valid_table, columns=8),
    )

    sequence_edges  = _sequence_edges(Path(sequence_path), rows, parameters)
    structure_edges = _structure_edges(Path(structure_path), rows, parameters)
    work.log(
        f"Similarity ready: {len(sequence_edges)} sequence edges and "
        f"{len(structure_edges)} structure edges"
    )
    return {
        "sequence_path":   sequence_path,
        "structure_path":  structure_path,
        "sequence_edges":  sequence_edges,
        "structure_edges": structure_edges,
    }


def _run_mmseqs(
    work      : Any,
    rows      : Sequence[Mapping[str, Any]],
    workers   : int,
    parameters: Mapping[str, float],
    tool      : Tool,
    output    : Path,
) -> None:
    """Run MMseqs2 all-vs-all into a seven-column checkpoint target.

    Args:
        work: Active LambdaForge Work.
        rows: RAW identifiers and sequences.
        workers: MMseqs2 threads.
        parameters: Sequence thresholds used for tool-side pruning.
        tool: Resolved MMseqs2 executable.
        output: LambdaForge checkpoint build target.
    """
    root  = work.temp_dir / "mmseqs"
    fasta = root / "proteins.fasta"
    root.mkdir(parents=True, exist_ok=True)
    fasta.write_text(
        "".join(f">{row['identifier']}\n{row['sequence']}\n" for row in rows),
        encoding="utf-8",
    )
    work.tools.run(
        [
            tool,
            "easy-search",
            fasta,
            fasta,
            output,
            root / "tmp",
            "--min-seq-id",
            str(parameters["sequence_identity"]),
            "-c",
            str(parameters["sequence_coverage"]),
            "--cov-mode",
            "0",
            "--alignment-mode",
            "3",
            "-e",
            str(parameters["sequence_evalue"]),
            "--max-seqs",
            str(max(10000, len(rows) + 1)),
            "--threads",
            str(workers),
            "--format-output",
            "query,target,fident,qcov,tcov,evalue,bits",
        ],
        name="MMseqs2",
        threads=workers,
    )


def _run_foldseek(
    work      : Any,
    rows      : Sequence[Mapping[str, Any]],
    workers   : int,
    parameters: Mapping[str, float],
    tool      : Tool,
    output    : Path,
) -> None:
    """Run Foldseek all-vs-all into an eight-column checkpoint target.

    Args:
        work: Active LambdaForge Work.
        rows: RAW rows carrying managed one-chain Foldseek inputs.
        workers: Foldseek threads.
        parameters: Structural thresholds used for tool-side pruning.
        tool: Resolved Foldseek executable.
        output: LambdaForge checkpoint build target.
    """
    root  = work.temp_dir / "foldseek"
    inputs = Path(rows[0]["foldseek_structure"]).parent
    root.mkdir(parents=True, exist_ok=True)
    work.tools.run(
        [
            tool,
            "easy-search",
            inputs,
            inputs,
            output,
            root / "tmp",
            "-e",
            str(parameters["foldseek_evalue"]),
            "--max-seqs",
            str(max(10000, len(rows) + 1)),
            "--threads",
            str(workers),
            "--format-output",
            "query,target,prob,evalue,qtmscore,ttmscore,qcov,tcov",
        ],
        name="Foldseek",
        threads=workers,
    )


def _sequence_edges(
    path      : Path,
    rows      : Sequence[Mapping[str, Any]],
    parameters: Mapping[str, float],
) -> set[tuple[str, str]]:
    """Threshold MMseqs2 identity, bilateral coverage, and E-value.

    Args:
        path: Seven-column MMseqs2 table.
        rows: Complete legal identifier population.
        parameters: Sequence edge thresholds.

    Returns:
        Canonical undirected qualifying pairs.
    """
    identifiers = {str(row["identifier"]) for row in rows}
    edges: set[tuple[str, str]] = set()
    for fields in _table(path, 7):
        query, target = fields[:2]
        if query not in identifiers or target not in identifiers:
            raise ValueError(f"MMseqs2 returned unknown identifiers: {query}, {target}")
        identity, query_coverage, target_coverage, evalue = map(float, fields[2:6])
        identity        = identity / 100.0 if identity > 1.0 else identity
        query_coverage  = query_coverage / 100.0 if query_coverage > 1.0 else query_coverage
        target_coverage = target_coverage / 100.0 if target_coverage > 1.0 else target_coverage
        values = (identity, query_coverage, target_coverage, evalue)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("MMseqs2 returned a non-finite value")
        if (
            query != target
            and identity >= parameters["sequence_identity"]
            and min(query_coverage, target_coverage) >= parameters["sequence_coverage"]
            and evalue <= parameters["sequence_evalue"]
        ):
            left, right = sorted((query, target))
            edges.add((left, right))
    return edges


def _structure_edges(
    path      : Path,
    rows      : Sequence[Mapping[str, Any]],
    parameters: Mapping[str, float],
) -> set[tuple[str, str]]:
    """Threshold Foldseek probability, two TM-scores, coverage, and E-value.

    Args:
        path: Eight-column Foldseek table.
        rows: Complete legal identifier population.
        parameters: Structural edge thresholds.

    Returns:
        Canonical undirected qualifying pairs.
    """
    identifiers = {str(row["identifier"]) for row in rows}
    edges: set[tuple[str, str]] = set()
    for fields in _table(path, 8):
        query, target = Path(fields[0]).stem, Path(fields[1]).stem
        if query not in identifiers or target not in identifiers:
            raise ValueError(f"Foldseek returned unknown identifiers: {query}, {target}")
        probability, evalue, query_tm, target_tm, query_coverage, target_coverage = map(
            float, fields[2:8]
        )
        probability     = probability / 100.0 if probability > 1.0 else probability
        query_coverage  = query_coverage / 100.0 if query_coverage > 1.0 else query_coverage
        target_coverage = target_coverage / 100.0 if target_coverage > 1.0 else target_coverage
        if (
            query != target
            and probability >= parameters["foldseek_probability"]
            and min(query_tm, target_tm) >= parameters["foldseek_tmscore"]
            and min(query_coverage, target_coverage) >= parameters["foldseek_coverage"]
            and evalue <= parameters["foldseek_evalue"]
        ):
            left, right = sorted((query, target))
            edges.add((left, right))
    return edges


def _table(path: Path, columns: int) -> list[list[str]]:
    """Read one exact-width tab-separated specialist table.

    Args:
        path: Table path.
        columns: Required number of columns per non-empty row.

    Returns:
        Parsed non-empty rows.
    """
    rows = [line.rstrip("\n").split("\t") for line in path.read_text().splitlines() if line]
    if any(len(row) != columns for row in rows):
        raise ValueError(f"{path.name} must contain exactly {columns} columns")
    return rows


def _valid_table(file: ManagedFile, columns: int) -> bool:
    """Return whether managed ``file`` has exactly ``columns`` tab-separated fields."""
    try:
        _table(Path(file), columns)
        return Path(file).stat().st_size > 0
    except (OSError, ValueError):
        return False
