"""Build explicit WISDOM-DNA raw evidence from BTD and contact-verified RCSB structures.

The script never creates a negative from missing DNA or an observed non-contact. BTD negatives are
retained only as exclusion-derived benchmark evidence after exact-sequence structural mapping,
complete technical audit, and rejection of direct DNA-contact contradictions. RCSB discovery adds
positive candidates only after a heavy-atom contact in the selected biological assembly. The
canonical output is one typed JSON object per line in ``data/dna/raw/raw.jsonl``; an equivalent
FASTA is kept solely for specialist sequence-tool interoperability.
"""

from __future__ import annotations

# Prevent nested BLAS/OpenMP parallelism inside structural worker processes.
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import csv
import gzip
import hashlib
import json
import math
import multiprocessing
import re
import shutil
import threading
import time
import urllib.request
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TextIO

import gemmi
import numpy as np
from scipy.spatial import cKDTree

# =============================================================================
# Configuration
# =============================================================================

# Every path is anchored to the repository rather than the caller's working directory. Final
# researcher-facing records live in ``data/dna/raw``; bulky intermediate evidence remains in its
# ``build`` child so the input selected in LambdaForge is always obvious.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_ROOT  = Path(__file__).resolve().parent
RAW_ROOT     = PROJECT_ROOT / "data" / "dna" / "raw"
BUILD_ROOT   = RAW_ROOT / "build"

# Stage 1: strict BTD Core. If CORE_FASTA already exists, it is reused.
BTD_INPUT_FASTA = SCRIPT_ROOT / "btd_combo.fasta"
CORE_FASTA      = BUILD_ROOT / "btd_combo_raw.fasta"
CORE_EVIDENCE   = BUILD_ROOT / "btd_combo_raw_evidence.csv"
CORE_SUMMARY    = BUILD_ROOT / "btd_combo_raw_summary.json"

# Stage 2: direct structural positives from RCSB.
#
# This date is deliberately fixed. Entries initially released after this date
# are excluded, so rerunning the query in the future does not silently add new
# PDB entries to this benchmark generation.
RCSB_CUTOFF = "2026-08-24"
RCSB_SEARCH_URL = "https://search.rcsb.org/rcsbsearch/v2/query"
RCSB_QUERY_CACHE = BUILD_ROOT / "rcsb-query-cache"

RCSB_POS_FASTA    = BUILD_ROOT / "rcsb_structural_positive_raw.fasta"
RCSB_POS_EVIDENCE = BUILD_ROOT / "rcsb_structural_positive_raw_evidence.csv"
RCSB_POS_SUMMARY  = BUILD_ROOT / "rcsb_structural_positive_raw_summary.json"

# Final canonical candidate records and compatibility FASTA.
EXPANDED_FULL_JSONL   = RAW_ROOT / "raw.jsonl"
EXPANDED_FULL_FASTA   = RAW_ROOT / "raw.fasta"
EXPANDED_FULL_EVIDENCE = BUILD_ROOT / "expanded_raw_evidence.csv"
EXPANDED_FULL_SUMMARY = RAW_ROOT / "raw-summary.json"

# Convenience 1:1 view. This is NOT the canonical source dataset.
BALANCED_FASTA   = BUILD_ROOT / "balanced_raw.fasta"
BALANCED_SUMMARY = BUILD_ROOT / "balanced_raw_summary.json"

# RCSB sequence database used only for strict BTD sequence -> PDB mapping.
PDB_SEQRES    = BUILD_ROOT / "pdb_seqres.txt"
PDB_SEQRES_GZ = BUILD_ROOT / "pdb_seqres.txt.gz"
PDB_SEQRES_URL = (
    "https://files.rcsb.org/pub/pdb/derived_data/pdb_seqres.txt.gz"
)

# Shared mmCIF cache. Existing structures from the BTD-Core stage are reused.
STRUCTURE_CACHE = BUILD_ROOT / "structure-cache"
STRUCTURE_URL = "https://files.rcsb.org/download/{pdb_id}.cif.gz"

# Scientific filters.
#
# Sequence transfer for BTD is always exact 100% full deposited sequence.
#
# This threshold is different: it is the fraction of the deposited protein
# sequence that has resolved heavy-atom coordinates in the experimental model.
MIN_COORDINATE_COVERAGE = 0.90

# Minimum length for new RCSB structural-positive proteins.
# 30 aa is deliberately conservative and matches the receptor-chain lower
# bound commonly used by BioLiP/Q-BioLiP-style curation.
MIN_PROTEIN_LENGTH = 30

# BioLiP/Q-BioLiP-style protein-DNA contact criterion:
# d < vdW(protein atom) + vdW(DNA atom) + 0.5 Å
CONTACT_TOLERANCE = 0.5

# B/Z/J/X are ambiguous and are rejected. U/O are unambiguous amino acids.
VALID_AMINO_ACIDS = frozenset("ACDEFGHIKLMNPQRSTVWYOU")

# Parallelism.
AUTO_WORKER_CAP = 36
MAX_DOWNLOAD_WORKERS = 8
DOWNLOAD_RETRIES = 3
DOWNLOAD_TIMEOUT = 180
EXPECTED_BTDCOMBO_RECORDS = 66737

# Reproducible hash-based balancing view.
BALANCE_SALT = "WISDOM-DNA-balanced-v1"
RAW_SCHEMA_VERSION = "1.0"


# =============================================================================
# Data classes
# =============================================================================

@dataclass(frozen=True)
class InputRecord:
    source_id: str
    original_header: str
    sequence: str
    label: int


@dataclass(frozen=True, order=True)
class PdbHit:
    pdb_id: str
    chain_id: str


@dataclass(frozen=True)
class PdbTarget:
    sequence: str
    header_chain_ids: tuple[str, ...]


@dataclass(frozen=True)
class Candidate:
    pdb_id: str
    chain_id: str
    assembly_id: str
    protein_copy: int
    assembly_source: str
    coordinate_coverage: float
    resolution: float
    dna_chains: tuple[str, ...]
    binding_positions: tuple[int, ...]

    @property
    def has_dna_contact(self) -> bool:
        return bool(self.binding_positions)


@dataclass
class PdbResult:
    pdb_id: str
    candidates: dict[str, list[Candidate]]
    notes: dict[str, list[str]]


@dataclass
class DnaIndex:
    xyz: np.ndarray
    radii: np.ndarray
    chain_names: np.ndarray
    tree: cKDTree
    max_radius: float


@dataclass(frozen=True)
class FinalRecord:
    header: str
    sequence: str
    label: int
    origin: str


@dataclass
class RcsbPositiveEntryResult:
    pdb_id: str
    candidates: dict[str, list[Candidate]]
    eligible_sequences: tuple[str, ...]
    noncontact_sequences: tuple[str, ...]
    notes: tuple[str, ...]


# =============================================================================
# CLI / CPU detection
# =============================================================================

def detected_cpus() -> int:
    slurm = os.getenv("SLURM_CPUS_PER_TASK")
    if slurm:
        try:
            value = int(slurm)
            if value > 0:
                return value
        except ValueError:
            pass

    if hasattr(os, "sched_getaffinity"):
        try:
            return max(1, len(os.sched_getaffinity(0)))
        except OSError:
            pass

    return max(1, os.cpu_count() or 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build WISDOM-DNA Core, exhaustive RCSB structural positives, "
            "the full Expanded dataset, and a reproducible balanced view."
        )
    )
    parser.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help=f"CPU processes for structural analysis (default: auto, cap {AUTO_WORKER_CAP}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild every stage, even when completed outputs already exist.",
    )
    parser.add_argument(
        "--force-core",
        action="store_true",
        help="Rebuild only the strict BTD Core stage.",
    )
    parser.add_argument(
        "--refresh-rcsb-query",
        action="store_true",
        help="Repeat the frozen RCSB assembly search instead of reusing its cached ID snapshot.",
    )
    parser.add_argument(
        "--force-rcsb",
        action="store_true",
        help="Rebuild the RCSB structural-positive FASTA from the cached/fresh query snapshot.",
    )
    parser.add_argument(
        "--force-expanded",
        action="store_true",
        help="Rebuild Expanded and balanced outputs from existing upstream FASTAs.",
    )
    args = parser.parse_args()

    available = detected_cpus()
    if args.workers is None:
        args.workers = min(available, AUTO_WORKER_CAP)
    elif args.workers < 1:
        parser.error("--workers must be >= 1")
    elif args.workers > available:
        print(
            f"WARNING: requested {args.workers} workers but only {available} CPUs "
            f"are visible. Using {available}."
        )
        args.workers = available

    args.available_cpus = available
    return args


# =============================================================================
# Shared helpers
# =============================================================================

def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalize_sequence(sequence: str) -> str:
    return "".join(sequence.split()).upper()


def atomic_temp(path: Path) -> Path:
    return path.with_name(f".{path.name}.{os.getpid()}.part")


def safe_token(value: str) -> str:
    return re.sub(r"\s+", "_", value).replace("|", "%7C")


def atomic_write_json(path: Path, value: dict) -> None:
    temporary = atomic_temp(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def iter_fasta_lines(lines: Iterable[str]):
    header: str | None = None
    parts: list[str] = []

    for raw_line in lines:
        line = raw_line.strip()
        if not line:
            continue

        if line.startswith(">"):
            if header is not None:
                yield header, "".join(parts)
            header = line[1:]
            parts = []
        else:
            if header is not None:
                parts.append(line)

    if header is not None:
        yield header, "".join(parts)


def iter_fasta(path: Path):
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        yield from iter_fasta_lines(handle)


def extract_label(header: str) -> int:
    match = re.search(r"(?:^|\|)label[_=]?([01])(?:\||$)", header)
    if match is None:
        # BTD source headers are commonly seq_N_label_1 rather than pipe-delimited.
        match = re.search(r"label[_=]?([01])", header)
    if match is None:
        raise ValueError(f"Could not find label_0 or label_1 in FASTA header: {header}")
    return int(match.group(1))


def read_final_fasta(path: Path, origin: str) -> list[FinalRecord]:
    output = []
    for header, raw_sequence in iter_fasta(path):
        output.append(
            FinalRecord(
                header=header,
                sequence=normalize_sequence(raw_sequence),
                label=extract_label(header),
                origin=origin,
            )
        )
    return output


def read_btd_input():
    valid: list[InputRecord] = []
    rejected: list[tuple[InputRecord, str, str]] = []

    for index, (header, raw_sequence) in enumerate(iter_fasta(BTD_INPUT_FASTA)):
        sequence = normalize_sequence(raw_sequence)
        record = InputRecord(
            source_id=f"btdc_{index:06d}",
            original_header=header,
            sequence=sequence,
            label=extract_label(header),
        )

        if not sequence:
            rejected.append((record, "empty_sequence", ""))
            continue

        invalid = sorted(set(sequence) - VALID_AMINO_ACIDS)
        if invalid:
            rejected.append((record, "ambiguous_sequence", ",".join(invalid)))
            continue

        valid.append(record)

    return valid, rejected


def deduplicate_btd(records: list[InputRecord]):
    groups: dict[str, list[InputRecord]] = defaultdict(list)
    for record in records:
        groups[record.sequence].append(record)

    clean: list[InputRecord] = []
    rejected: list[tuple[InputRecord, str, str]] = []

    for group in groups.values():
        labels = {record.label for record in group}
        if len(labels) != 1:
            for record in group:
                rejected.append((record, "conflicting_labels_for_exact_sequence", ""))
            continue

        canonical = min(group, key=lambda record: record.source_id)
        clean.append(canonical)
        for record in group:
            if record.source_id != canonical.source_id:
                rejected.append((record, "exact_duplicate", canonical.source_id))

    clean.sort(key=lambda record: record.source_id)
    return clean, rejected


def valid_gzip(path: Path) -> bool:
    try:
        if not path.exists() or path.stat().st_size == 0:
            return False
        with gzip.open(path, "rb") as handle:
            handle.read(1024)
        return True
    except (OSError, EOFError):
        return False


def download_atomic(url: str, final_path: Path, validate=None) -> Path:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        temporary = final_path.with_name(
            f".{final_path.name}.{os.getpid()}.{threading.get_ident()}.{attempt}.part"
        )
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "WISDOM-curation"})
            with (
                urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as source,
                temporary.open("wb") as destination,
            ):
                shutil.copyfileobj(source, destination)

            if not temporary.exists() or temporary.stat().st_size == 0:
                raise RuntimeError("Downloaded file is empty")
            if validate is not None and not validate(temporary):
                raise RuntimeError("Downloaded file failed validation")

            os.replace(temporary, final_path)
            return final_path
        except Exception as error:
            last_error = error
            temporary.unlink(missing_ok=True)
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(1.5 * attempt)

    raise RuntimeError(f"Download failed: {url}") from last_error


def ensure_pdb_seqres() -> Path:
    if PDB_SEQRES.exists():
        return PDB_SEQRES
    if valid_gzip(PDB_SEQRES_GZ):
        return PDB_SEQRES_GZ

    print("Downloading RCSB PDB SEQRES...")
    return download_atomic(PDB_SEQRES_URL, PDB_SEQRES_GZ, validate=valid_gzip)


def open_seqres(path: Path) -> TextIO:
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def parse_seqres_header(header: str) -> PdbHit | None:
    token = header.split(maxsplit=1)[0]
    if "_" not in token:
        return None

    pdb_id, chain_id = token.split("_", 1)
    pdb_id = pdb_id.upper()
    if len(pdb_id) != 4 or not pdb_id.isalnum() or not chain_id:
        return None

    return PdbHit(pdb_id=pdb_id, chain_id=chain_id)


def exact_pdb_index(path: Path, wanted: set[str]) -> dict[str, list[PdbHit]]:
    matches: dict[str, set[PdbHit]] = defaultdict(set)
    header: str | None = None
    parts: list[str] = []

    def consume(current_header: str | None, sequence_parts: list[str]) -> None:
        if current_header is None:
            return
        if "mol:protein" not in current_header.lower():
            return

        sequence = normalize_sequence("".join(sequence_parts))
        if sequence not in wanted:
            return

        hit = parse_seqres_header(current_header)
        if hit is not None:
            matches[sequence].add(hit)

    with open_seqres(path) as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line.startswith(">"):
                consume(header, parts)
                header = line[1:]
                parts = []
            else:
                parts.append(line)
        consume(header, parts)

    return {sequence: sorted(hits) for sequence, hits in matches.items()}


def build_jobs(sequences: Iterable[str], exact_index: dict[str, list[PdbHit]]):
    jobs: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for sequence in sequences:
        for hit in exact_index.get(sequence, []):
            jobs[hit.pdb_id][sequence].add(hit.chain_id)

    output = {}
    for pdb_id, sequence_map in sorted(jobs.items()):
        targets = [
            PdbTarget(sequence=sequence, header_chain_ids=tuple(sorted(chain_ids)))
            for sequence, chain_ids in sequence_map.items()
        ]
        targets.sort(key=lambda target: sha256_text(target.sequence))
        output[pdb_id] = tuple(targets)

    return output


def structure_path(pdb_id: str) -> Path:
    return STRUCTURE_CACHE / f"{pdb_id}.cif.gz"


def download_one_structure(pdb_id: str) -> Path:
    STRUCTURE_CACHE.mkdir(parents=True, exist_ok=True)
    path = structure_path(pdb_id)
    if valid_gzip(path):
        return path

    path.unlink(missing_ok=True)
    return download_atomic(
        STRUCTURE_URL.format(pdb_id=pdb_id),
        path,
        validate=valid_gzip,
    )


def download_structures(pdb_ids: Iterable[str], workers: int):
    pdb_ids = sorted(set(pdb_ids))
    if not pdb_ids:
        return set(), {}

    io_workers = max(1, min(MAX_DOWNLOAD_WORKERS, workers, len(pdb_ids)))
    print(f"Downloading/checking {len(pdb_ids)} PDBs with {io_workers} I/O threads...")

    successful: set[str] = set()
    errors: dict[str, str] = {}

    with ThreadPoolExecutor(max_workers=io_workers) as executor:
        futures = {executor.submit(download_one_structure, pdb_id): pdb_id for pdb_id in pdb_ids}
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            pdb_id = futures[future]
            try:
                future.result()
                successful.add(pdb_id)
            except Exception as error:
                errors[pdb_id] = f"{type(error).__name__}: {error}"

            if completed % 100 == 0 or completed == total:
                print(f"  structures ready: {completed}/{total}")

    return successful, errors


def is_protein(chain: gemmi.Chain) -> bool:
    polymer = chain.get_polymer()
    return len(polymer) > 0 and polymer.check_polymer_type() in {
        gemmi.PolymerType.PeptideL,
        gemmi.PolymerType.PeptideD,
    }


def is_dna(chain: gemmi.Chain) -> bool:
    polymer = chain.get_polymer()
    return len(polymer) > 0 and polymer.check_polymer_type() == gemmi.PolymerType.Dna


def full_chain_sequence(structure: gemmi.Structure, chain: gemmi.Chain) -> str | None:
    polymer = chain.get_polymer()
    if len(polymer) == 0:
        return None

    entity = structure.get_entity_of(polymer)
    if entity is None or not entity.full_sequence:
        return None

    return gemmi.one_letter_code(entity.full_sequence).upper()


def has_resolved_heavy_atom(residue: gemmi.Residue) -> bool:
    return any(
        atom.element.atomic_number != 1 and atom.occ > 0
        for atom in residue.first_conformer()
    )


def coordinate_coverage(chain: gemmi.Chain, sequence_length: int) -> float:
    if sequence_length <= 0:
        return 0.0

    positions: set[int] = set()
    for residue in chain.get_polymer().first_conformer():
        if residue.label_seq is None or not has_resolved_heavy_atom(residue):
            continue
        position = int(residue.label_seq)
        if 1 <= position <= sequence_length:
            positions.add(position)

    return len(positions) / sequence_length


def assembly_source(assembly: gemmi.Assembly) -> str:
    if assembly.author_determined:
        return "author"
    if assembly.software_determined:
        return "software"
    return "unspecified"


def eligible_assemblies(structure: gemmi.Structure):
    assemblies = list(structure.assemblies)
    if not assemblies:
        return []

    author = [assembly for assembly in assemblies if assembly.author_determined]
    return author if author else assemblies


def make_dna_index(chains: list[gemmi.Chain]) -> DnaIndex | None:
    xyz = []
    radii = []
    chain_names = []

    for chain in chains:
        for residue in chain.get_polymer().first_conformer():
            for atom in residue.first_conformer():
                if atom.element.atomic_number == 1 or atom.occ <= 0:
                    continue
                radius = float(atom.element.vdw_r)
                if not math.isfinite(radius) or radius <= 0:
                    continue

                xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
                radii.append(radius)
                chain_names.append(chain.name)

    if not xyz:
        return None

    xyz_array = np.asarray(xyz, dtype=np.float64)
    radii_array = np.asarray(radii, dtype=np.float64)
    return DnaIndex(
        xyz=xyz_array,
        radii=radii_array,
        chain_names=np.asarray(chain_names, dtype=object),
        tree=cKDTree(xyz_array),
        max_radius=float(radii_array.max()),
    )


def contacts(protein: gemmi.Chain, dna: DnaIndex | None):
    if dna is None:
        return set(), set()

    xyz = []
    radii = []
    positions = []

    for residue in protein.get_polymer().first_conformer():
        if residue.label_seq is None:
            continue
        position = int(residue.label_seq)

        for atom in residue.first_conformer():
            if atom.element.atomic_number == 1 or atom.occ <= 0:
                continue
            radius = float(atom.element.vdw_r)
            if not math.isfinite(radius) or radius <= 0:
                continue

            xyz.append((atom.pos.x, atom.pos.y, atom.pos.z))
            radii.append(radius)
            positions.append(position)

    if not xyz:
        return set(), set()

    protein_xyz = np.asarray(xyz, dtype=np.float64)
    protein_radii = np.asarray(radii, dtype=np.float64)

    broad_radius = float(protein_radii.max() + dna.max_radius + CONTACT_TOLERANCE)
    neighbour_lists = dna.tree.query_ball_point(protein_xyz, broad_radius)

    binding_positions: set[int] = set()
    contacting_dna_chains: set[str] = set()

    for protein_index, neighbours in enumerate(neighbour_lists):
        if not neighbours:
            continue

        indices = np.asarray(neighbours, dtype=np.int64)
        delta = dna.xyz[indices] - protein_xyz[protein_index]
        squared_distance = np.einsum("ij,ij->i", delta, delta)
        cutoffs = protein_radii[protein_index] + dna.radii[indices] + CONTACT_TOLERANCE
        contacting = squared_distance < cutoffs * cutoffs

        if np.any(contacting):
            binding_positions.add(positions[protein_index])
            for chain_name in dna.chain_names[indices[contacting]]:
                contacting_dna_chains.add(str(chain_name))

    return binding_positions, contacting_dna_chains


def analyse_pdb(pdb_id: str, targets: tuple[PdbTarget, ...]) -> PdbResult:
    structure = gemmi.read_structure(str(structure_path(pdb_id)))
    if len(structure) == 0:
        raise RuntimeError("Structure has no coordinate model")

    structure.setup_entities()
    structure.assign_label_seq_id()
    model = structure[0]
    assemblies = eligible_assemblies(structure)

    resolution = float(structure.resolution)
    if not math.isfinite(resolution) or resolution <= 0:
        resolution = math.nan

    candidates = {target.sequence: [] for target in targets}
    notes = {target.sequence: [] for target in targets}

    if not assemblies:
        for target in targets:
            notes[target.sequence].append("no_declared_biological_assembly")
        return PdbResult(pdb_id, candidates, notes)

    wanted = {target.sequence for target in targets}
    chains_by_sequence: dict[str, list[gemmi.Chain]] = defaultdict(list)

    # Revalidate exact sequence inside the mmCIF itself.
    for chain in model:
        if not is_protein(chain):
            continue
        sequence = full_chain_sequence(structure, chain)
        if sequence in wanted:
            chains_by_sequence[sequence].append(chain)

    usable: dict[tuple[str, str], float] = {}
    for target in targets:
        base_chains = chains_by_sequence.get(target.sequence, [])
        if not base_chains:
            notes[target.sequence].append("no_exact_chain_in_mmcif")
            continue

        for chain in base_chains:
            coverage = coordinate_coverage(chain, len(target.sequence))
            if coverage >= MIN_COORDINATE_COVERAGE:
                usable[(target.sequence, chain.name)] = coverage
            else:
                notes[target.sequence].append(
                    f"low_coordinate_coverage:{chain.name}:{coverage:.4f}"
                )

    if not usable:
        return PdbResult(pdb_id, candidates, notes)

    for assembly in assemblies:
        assembled = gemmi.make_assembly(
            assembly,
            model,
            gemmi.HowToNameCopiedChain.Dup,
        )
        dna_chains = [chain for chain in assembled if is_dna(chain)]
        dna_index = make_dna_index(dna_chains)
        source = assembly_source(assembly)

        for target in targets:
            source_chain_names = sorted(
                chain_name
                for sequence, chain_name in usable
                if sequence == target.sequence
            )

            for chain_name in source_chain_names:
                copies = [
                    chain
                    for chain in assembled
                    if chain.name == chain_name and is_protein(chain)
                ]

                # Each transformed protein copy is evaluated independently.
                for copy_number, protein_copy in enumerate(copies, start=1):
                    binding_positions, dna_names = contacts(protein_copy, dna_index)
                    candidates[target.sequence].append(
                        Candidate(
                            pdb_id=pdb_id,
                            chain_id=chain_name,
                            assembly_id=str(assembly.name),
                            protein_copy=copy_number,
                            assembly_source=source,
                            coordinate_coverage=usable[(target.sequence, chain_name)],
                            resolution=resolution,
                            dna_chains=tuple(sorted(dna_names)),
                            binding_positions=tuple(sorted(binding_positions)),
                        )
                    )

    for target in targets:
        if not candidates[target.sequence]:
            notes[target.sequence].append("no_usable_chain_in_biological_assembly")

    return PdbResult(pdb_id, candidates, notes)


def analyse_all(jobs, successful_downloads: set[str], workers: int):
    runnable = {
        pdb_id: targets
        for pdb_id, targets in jobs.items()
        if pdb_id in successful_downloads
    }

    print(f"Analysing {len(runnable)} PDBs with {workers} CPU process(es)...")
    candidate_index = {}
    notes_index = {}
    errors: dict[str, str] = {}

    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(analyse_pdb, pdb_id, targets): pdb_id
            for pdb_id, targets in runnable.items()
        }
        total = len(futures)

        for completed, future in enumerate(as_completed(futures), start=1):
            pdb_id = futures[future]
            try:
                result = future.result()
                for sequence, values in result.candidates.items():
                    candidate_index[(sequence, pdb_id)] = values
                for sequence, values in result.notes.items():
                    notes_index[(sequence, pdb_id)] = values
            except Exception as error:
                errors[pdb_id] = f"{type(error).__name__}: {error}"

            if completed % 25 == 0 or completed == total:
                print(f"  PDBs analysed: {completed}/{total}")

    return candidate_index, notes_index, errors


def candidate_rank(candidate: Candidate):
    source_rank = {"author": 0, "software": 1, "unspecified": 2}.get(
        candidate.assembly_source, 3
    )
    resolution_rank = (
        candidate.resolution
        if math.isfinite(candidate.resolution) and candidate.resolution > 0
        else math.inf
    )

    # Never rank positives by interface size/contact count: that would bias GT.
    return (
        source_rank,
        -candidate.coordinate_coverage,
        resolution_rank,
        candidate.pdb_id,
        candidate.chain_id,
        candidate.assembly_id,
        candidate.protein_copy,
    )


def structural_scan(sequences: Iterable[str], workers: int):
    sequences = sorted(set(sequences), key=sha256_text)
    seqres = ensure_pdb_seqres()

    print(f"Scanning RCSB for exact matches to {len(sequences)} unique sequences...")
    exact_index = exact_pdb_index(seqres, set(sequences))
    jobs = build_jobs(sequences, exact_index)

    exact_count = sum(sequence in exact_index for sequence in sequences)
    print(f"  exact sequence matches: {exact_count}/{len(sequences)}")
    print(f"  unique PDB entries to inspect: {len(jobs)}")

    downloaded, download_errors = download_structures(jobs.keys(), workers)
    candidate_index, notes_index, analysis_errors = analyse_all(
        jobs, downloaded, workers
    )

    return (
        exact_index,
        candidate_index,
        notes_index,
        download_errors,
        analysis_errors,
        seqres,
        len(jobs),
    )


def collect_sequence_results(
    sequence: str,
    exact_index: dict[str, list[PdbHit]],
    candidate_index,
    notes_index,
    download_errors,
    analysis_errors,
):
    hits = exact_index.get(sequence, [])
    pdb_ids = sorted({hit.pdb_id for hit in hits})

    candidates: list[Candidate] = []
    notes: list[str] = []
    technical_errors: list[str] = []

    for pdb_id in pdb_ids:
        candidates.extend(candidate_index.get((sequence, pdb_id), []))
        notes.extend(notes_index.get((sequence, pdb_id), []))

        if pdb_id in download_errors:
            technical_errors.append(f"{pdb_id}:download:{download_errors[pdb_id]}")
        if pdb_id in analysis_errors:
            technical_errors.append(f"{pdb_id}:analysis:{analysis_errors[pdb_id]}")

    return hits, candidates, notes, technical_errors


# =============================================================================
# Core evidence fields
# =============================================================================

CORE_FIELDS = [
    "source_id",
    "original_header",
    "label",
    "sequence_sha256",
    "status",
    "reason",
    "exact_pdb_hits",
    "pdb_id",
    "chain_id",
    "assembly_id",
    "protein_copy",
    "assembly_source",
    "coordinate_coverage",
    "resolution",
    "dna_chains",
    "binding_positions",
    "binding_residue_count",
    "structure_sha256",
    "technical_errors",
    "structural_notes",
]


def core_row(
    record: InputRecord,
    status: str,
    reason: str,
    hits=None,
    candidate: Candidate | None = None,
    errors=None,
    notes=None,
):
    row = {
        "source_id": record.source_id,
        "original_header": record.original_header,
        "label": record.label,
        "sequence_sha256": sha256_text(record.sequence),
        "status": status,
        "reason": reason,
        "exact_pdb_hits": ";".join(f"{hit.pdb_id}_{hit.chain_id}" for hit in (hits or [])),
        "pdb_id": "",
        "chain_id": "",
        "assembly_id": "",
        "protein_copy": "",
        "assembly_source": "",
        "coordinate_coverage": "",
        "resolution": "",
        "dna_chains": "",
        "binding_positions": "",
        "binding_residue_count": "",
        "structure_sha256": "",
        "technical_errors": " | ".join(sorted(set(errors or []))),
        "structural_notes": " | ".join(sorted(set(notes or []))),
    }

    if candidate is not None:
        row.update(
            {
                "pdb_id": candidate.pdb_id,
                "chain_id": candidate.chain_id,
                "assembly_id": candidate.assembly_id,
                "protein_copy": candidate.protein_copy,
                "assembly_source": candidate.assembly_source,
                "coordinate_coverage": f"{candidate.coordinate_coverage:.6f}",
                "resolution": (
                    f"{candidate.resolution:.4f}"
                    if math.isfinite(candidate.resolution)
                    else ""
                ),
                "dna_chains": ";".join(candidate.dna_chains),
                "binding_positions": ";".join(map(str, candidate.binding_positions)),
                "binding_residue_count": len(candidate.binding_positions),
            }
        )

    return row


def build_core(workers: int) -> None:
    if not BTD_INPUT_FASTA.exists():
        raise FileNotFoundError(
            f"Core output is missing and BTD source FASTA was not found: {BTD_INPUT_FASTA}"
        )

    print("\n=== Stage 1: building WISDOM-DNA Core from BTD-Combo ===")
    records_before_dedup, invalid_records = read_btd_input()
    original_count = len(records_before_dedup) + len(invalid_records)
    records, duplicate_rejections = deduplicate_btd(records_before_dedup)

    if original_count != EXPECTED_BTDCOMBO_RECORDS:
        print(
            f"WARNING: BTD-Combo has {original_count} records; expected "
            f"{EXPECTED_BTDCOMBO_RECORDS}."
        )

    (
        exact_index,
        candidate_index,
        notes_index,
        download_errors,
        analysis_errors,
        seqres_path,
        pdb_count,
    ) = structural_scan((record.sequence for record in records), workers)

    accepted: list[tuple[InputRecord, Candidate]] = []
    rows = []
    counts = defaultdict(int)

    for record, reason, detail in invalid_records + duplicate_rejections:
        rows.append(
            core_row(
                record,
                "rejected",
                f"{reason}:{detail}" if detail else reason,
            )
        )

    for record in records:
        hits, candidates, notes, technical_errors = collect_sequence_results(
            record.sequence,
            exact_index,
            candidate_index,
            notes_index,
            download_errors,
            analysis_errors,
        )

        if not hits:
            counts["no_exact_pdb_match"] += 1
            rows.append(core_row(record, "rejected", "no_exact_pdb_match"))
            continue

        if record.label == 1:
            contact_candidates = [c for c in candidates if c.has_dna_contact]
            if contact_candidates:
                best = min(contact_candidates, key=candidate_rank)
                accepted.append((record, best))
                counts["accepted_positive"] += 1
                rows.append(
                    core_row(
                        record,
                        "accepted",
                        "positive_exact_sequence_with_dna_contact",
                        hits,
                        best,
                        technical_errors,
                        notes,
                    )
                )
                continue

            if technical_errors:
                reason = "positive_incomplete_structural_audit"
            elif candidates:
                reason = "positive_without_experimental_dna_contact"
            else:
                reason = "positive_without_usable_structure"

            counts[reason] += 1
            rows.append(
                core_row(
                    record,
                    "rejected",
                    reason,
                    hits,
                    errors=technical_errors,
                    notes=notes,
                )
            )
            continue

        conflicts = [c for c in candidates if c.has_dna_contact]
        if conflicts:
            conflict = min(conflicts, key=candidate_rank)
            reason = "negative_label_conflict_with_pdb_dna_contact"
            counts[reason] += 1
            rows.append(
                core_row(
                    record,
                    "rejected",
                    reason,
                    hits,
                    conflict,
                    technical_errors,
                    notes,
                )
            )
            continue

        if technical_errors:
            reason = "negative_incomplete_structural_audit"
            counts[reason] += 1
            rows.append(
                core_row(
                    record,
                    "rejected",
                    reason,
                    hits,
                    errors=technical_errors,
                    notes=notes,
                )
            )
            continue

        if not candidates:
            reason = "negative_without_usable_structure"
            counts[reason] += 1
            rows.append(core_row(record, "rejected", reason, hits, notes=notes))
            continue

        best = min(candidates, key=candidate_rank)
        accepted.append((record, best))
        counts["accepted_negative"] += 1
        rows.append(
            core_row(
                record,
                "accepted",
                "btd_exclusion_derived_negative_without_structural_contradiction",
                hits,
                best,
                notes=notes,
            )
        )

    # Hash only structures actually referenced in the evidence table.
    structure_hashes = {}
    for row in rows:
        pdb_id = str(row["pdb_id"])
        if not pdb_id:
            continue
        path = structure_path(pdb_id)
        if pdb_id not in structure_hashes and path.exists():
            structure_hashes[pdb_id] = sha256_file(path)
        row["structure_sha256"] = structure_hashes.get(pdb_id, "")

    accepted.sort(key=lambda item: item[0].source_id)
    rows.sort(key=lambda row: str(row["source_id"]))

    fasta_tmp = atomic_temp(CORE_FASTA)
    evidence_tmp = atomic_temp(CORE_EVIDENCE)
    try:
        with fasta_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for record, candidate in accepted:
                header = (
                    f"{candidate.pdb_id}_{safe_token(candidate.chain_id)}"
                    f"|assembly_{safe_token(candidate.assembly_id)}"
                    f"|copy_{candidate.protein_copy}"
                    f"|label_{record.label}"
                    f"|source_{record.source_id}"
                )
                handle.write(f">{header}\n{record.sequence}\n")

        with evidence_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=CORE_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(fasta_tmp, CORE_FASTA)
        os.replace(evidence_tmp, CORE_EVIDENCE)
    finally:
        fasta_tmp.unlink(missing_ok=True)
        evidence_tmp.unlink(missing_ok=True)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "core",
        "source": "BTD-Combo",
        "input_fasta": str(BTD_INPUT_FASTA),
        "input_fasta_sha256": sha256_file(BTD_INPUT_FASTA),
        "pdb_seqres_sha256": sha256_file(seqres_path),
        "sequence_mapping": "exact full deposited sequence only",
        "min_coordinate_coverage": MIN_COORDINATE_COVERAGE,
        "contact_definition": "distance < vdW(protein)+vdW(DNA)+0.5 Angstrom",
        "negative_evidence": (
            "BTD exclusion-derived benchmark negative mapped by exact full sequence. Direct DNA "
            "contact contradictions and incomplete structural audits are rejected. This tier is "
            "not described as experimental proof of universal non-binding."
        ),
        "accepted_total": len(accepted),
        "unique_pdb_entries_inspected": pdb_count,
        "counts": dict(sorted(counts.items())),
    }
    atomic_write_json(CORE_SUMMARY, summary)

    print(
        f"Core complete: {len(accepted)} proteins "
        f"({counts['accepted_positive']} positive, {counts['accepted_negative']} negative)."
    )


# =============================================================================
# RCSB frozen assembly discovery
# =============================================================================

def rcsb_snapshot_path(cutoff: str = RCSB_CUTOFF) -> Path:
    safe = cutoff.replace(":", "-")
    return RCSB_QUERY_CACHE / f"protein_dna_assemblies_before_{safe}.json"


def rcsb_search_payload(cutoff: str = RCSB_CUTOFF) -> dict:
    """
    Search biological assemblies that:
      - are experimental RCSB PDB content,
      - contain >=1 protein instance,
      - contain >=1 DNA instance,
      - belong to entries initially released on/before the frozen cutoff.

    This is only candidate discovery. A protein is NOT labelled positive until
    direct atomic protein-DNA contact is independently re-established from the
    downloaded biological assembly.
    """
    return {
        "query": {
            "type": "group",
            "logical_operator": "and",
            "nodes": [
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_assembly_info.polymer_entity_instance_count_protein",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_assembly_info.polymer_entity_instance_count_DNA",
                        "operator": "greater_or_equal",
                        "value": 1,
                    },
                },
                {
                    "type": "terminal",
                    "service": "text",
                    "parameters": {
                        "attribute": "rcsb_accession_info.initial_release_date",
                        "operator": "less_or_equal",
                        "value": cutoff,
                    },
                },
            ],
        },
        "return_type": "assembly",
        "request_options": {
            "results_content_type": ["experimental"],
            "return_all_hits": True,
            "results_verbosity": "compact",
        },
    }


def post_json(url: str, payload: dict) -> dict:
    encoded = json.dumps(payload).encode("utf-8")
    last_error: Exception | None = None

    for attempt in range(1, DOWNLOAD_RETRIES + 1):
        try:
            request = urllib.request.Request(
                url,
                data=encoded,
                headers={
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "WISDOM-curation",
                },
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=DOWNLOAD_TIMEOUT) as response:
                return json.load(response)
        except Exception as error:
            last_error = error
            if attempt < DOWNLOAD_RETRIES:
                time.sleep(2.0 * attempt)

    raise RuntimeError(
        f"RCSB Search API request failed after {DOWNLOAD_RETRIES} attempts"
    ) from last_error


def fetch_rcsb_assembly_snapshot(refresh: bool = False) -> dict:
    path = rcsb_snapshot_path()
    if path.exists() and not refresh:
        with path.open("r", encoding="utf-8") as handle:
            snapshot = json.load(handle)

        if snapshot.get("cutoff") != RCSB_CUTOFF:
            raise RuntimeError(
                f"Cached RCSB snapshot cutoff is {snapshot.get('cutoff')!r}, "
                f"expected {RCSB_CUTOFF!r}"
            )
        return snapshot

    print(f"Querying RCSB for protein+DNA biological assemblies released <= {RCSB_CUTOFF}...")
    payload = rcsb_search_payload()
    response = post_json(RCSB_SEARCH_URL, payload)

    identifiers = []
    for item in response.get("result_set", []):
        identifier = item if isinstance(item, str) else item.get("identifier", "")

        identifier = str(identifier).strip()
        if not identifier:
            continue

        # Search/Data API biological assembly format is ENTRY-ASSEMBLY, e.g. 4HHB-1.
        if "-" not in identifier:
            continue

        entry_id, assembly_id = identifier.split("-", 1)
        entry_id = entry_id.upper()
        if len(entry_id) != 4 or not entry_id.isalnum() or not assembly_id:
            continue

        identifiers.append(f"{entry_id}-{assembly_id}")

    identifiers = sorted(set(identifiers))
    if not identifiers:
        raise RuntimeError("RCSB Search API returned no valid protein+DNA assembly identifiers")

    snapshot = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "cutoff": RCSB_CUTOFF,
        "search_url": RCSB_SEARCH_URL,
        "query": payload,
        "assembly_count": len(identifiers),
        "assembly_ids": identifiers,
    }

    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(path, snapshot)

    print(f"  frozen assembly candidates: {len(identifiers)}")
    print(f"  snapshot: {path}")
    return snapshot


def assembly_jobs_from_snapshot(snapshot: dict) -> dict[str, tuple[str, ...]]:
    jobs: dict[str, set[str]] = defaultdict(set)

    for identifier in snapshot["assembly_ids"]:
        entry_id, assembly_id = identifier.split("-", 1)
        jobs[entry_id.upper()].add(assembly_id)

    return {
        pdb_id: tuple(sorted(assembly_ids))
        for pdb_id, assembly_ids in sorted(jobs.items())
    }


# =============================================================================
# Direct RCSB structural-positive analysis
# =============================================================================

def analyse_rcsb_positive_entry(
    pdb_id: str,
    requested_assembly_ids: tuple[str, ...],
) -> RcsbPositiveEntryResult:
    """
    Analyse one RCSB PDB entry.

    A new positive is created only from a protein chain that:
      1. has an unambiguous deposited full protein sequence >=30 aa,
      2. has >= MIN_COORDINATE_COVERAGE resolved positions,
      3. occurs in an eligible declared biological assembly selected by RCSB
         as containing protein + DNA,
      4. directly contacts pure DNA by the VdW + 0.5 Å atomic criterion.

    Importantly, protein copies are evaluated independently.
    """
    structure = gemmi.read_structure(str(structure_path(pdb_id)))
    if len(structure) == 0:
        raise RuntimeError("Structure has no coordinate model")

    structure.setup_entities()
    structure.assign_label_seq_id()
    model = structure[0]

    resolution = float(structure.resolution)
    if not math.isfinite(resolution) or resolution <= 0:
        resolution = math.nan

    requested = set(map(str, requested_assembly_ids))
    eligible = [
        assembly
        for assembly in eligible_assemblies(structure)
        if str(assembly.name) in requested
    ]

    if not eligible:
        return RcsbPositiveEntryResult(
            pdb_id=pdb_id,
            candidates={},
            eligible_sequences=(),
            noncontact_sequences=(),
            notes=("no_requested_eligible_biological_assembly",),
        )

    # Source-chain information from the asymmetric unit.
    base: dict[str, tuple[str, float]] = {}
    notes: list[str] = []

    for chain in model:
        if not is_protein(chain):
            continue

        sequence = full_chain_sequence(structure, chain)
        if sequence is None:
            continue

        sequence = normalize_sequence(sequence)
        if len(sequence) < MIN_PROTEIN_LENGTH:
            continue

        invalid = set(sequence) - VALID_AMINO_ACIDS
        if invalid:
            notes.append(
                f"ambiguous_protein_sequence:{chain.name}:{','.join(sorted(invalid))}"
            )
            continue

        coverage = coordinate_coverage(chain, len(sequence))
        if coverage < MIN_COORDINATE_COVERAGE:
            notes.append(f"low_coordinate_coverage:{chain.name}:{coverage:.4f}")
            continue

        base[chain.name] = (sequence, coverage)

    if not base:
        return RcsbPositiveEntryResult(
            pdb_id=pdb_id,
            candidates={},
            eligible_sequences=(),
            noncontact_sequences=(),
            notes=tuple(sorted(set([*notes, "no_usable_protein_chain"]))),
        )

    candidates: dict[str, list[Candidate]] = defaultdict(list)
    eligible_sequences: set[str] = set()
    noncontact_sequences: set[str] = set()

    for assembly in eligible:
        assembled = gemmi.make_assembly(
            assembly,
            model,
            gemmi.HowToNameCopiedChain.Dup,
        )

        dna_chains = [chain for chain in assembled if is_dna(chain)]
        dna_index = make_dna_index(dna_chains)
        if dna_index is None:
            notes.append(f"assembly_without_resolved_pure_dna:{assembly.name}")
            continue

        source = assembly_source(assembly)
        copy_counter: dict[str, int] = defaultdict(int)

        for protein_copy in assembled:
            chain_name = protein_copy.name
            if chain_name not in base or not is_protein(protein_copy):
                continue

            sequence, coverage = base[chain_name]
            eligible_sequences.add(sequence)
            copy_counter[chain_name] += 1
            copy_number = copy_counter[chain_name]

            binding_positions, dna_names = contacts(protein_copy, dna_index)

            if binding_positions:
                candidates[sequence].append(
                    Candidate(
                        pdb_id=pdb_id,
                        chain_id=chain_name,
                        assembly_id=str(assembly.name),
                        protein_copy=copy_number,
                        assembly_source=source,
                        coordinate_coverage=coverage,
                        resolution=resolution,
                        dna_chains=tuple(sorted(dna_names)),
                        binding_positions=tuple(sorted(binding_positions)),
                    )
                )
            else:
                noncontact_sequences.add(sequence)

    return RcsbPositiveEntryResult(
        pdb_id=pdb_id,
        candidates=dict(candidates),
        eligible_sequences=tuple(sorted(eligible_sequences, key=sha256_text)),
        noncontact_sequences=tuple(sorted(noncontact_sequences, key=sha256_text)),
        notes=tuple(sorted(set(notes))),
    )


RCSB_POS_FIELDS = [
    "source_id",
    "sequence_sha256",
    "status",
    "reason",
    "pdb_id",
    "chain_id",
    "assembly_id",
    "protein_copy",
    "assembly_source",
    "coordinate_coverage",
    "resolution",
    "dna_chains",
    "binding_positions",
    "binding_residue_count",
    "supporting_pdb_count",
    "supporting_pdb_ids",
    "structure_sha256",
]


def build_rcsb_structural_positives(
    workers: int,
    refresh_query: bool = False,
) -> None:
    print("\n=== Stage 2: discovering direct RCSB structural positives ===")

    snapshot = fetch_rcsb_assembly_snapshot(refresh=refresh_query)
    jobs = assembly_jobs_from_snapshot(snapshot)

    print(
        f"RCSB candidate set: {len(snapshot['assembly_ids'])} biological assemblies "
        f"across {len(jobs)} PDB entries."
    )

    downloaded, download_errors = download_structures(jobs.keys(), workers)

    runnable = {
        pdb_id: assembly_ids
        for pdb_id, assembly_ids in jobs.items()
        if pdb_id in downloaded
    }

    context = multiprocessing.get_context("spawn")

    best_by_sequence: dict[str, Candidate] = {}
    support_pdbs: dict[str, set[str]] = defaultdict(set)
    eligible_sequences: set[str] = set()
    noncontact_observations: set[str] = set()
    analysis_errors: dict[str, str] = {}
    worker_notes = defaultdict(int)
    candidate_observation_count = 0

    print(f"Analysing {len(runnable)} RCSB entries with {workers} CPU process(es)...")

    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                analyse_rcsb_positive_entry,
                pdb_id,
                assembly_ids,
            ): pdb_id
            for pdb_id, assembly_ids in runnable.items()
        }

        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            pdb_id = futures[future]

            try:
                result = future.result()
            except Exception as error:
                analysis_errors[pdb_id] = f"{type(error).__name__}: {error}"
            else:
                eligible_sequences.update(result.eligible_sequences)
                noncontact_observations.update(result.noncontact_sequences)

                for note in result.notes:
                    worker_notes[note.split(":", 1)[0]] += 1

                for sequence, candidates in result.candidates.items():
                    for candidate in candidates:
                        candidate_observation_count += 1
                        support_pdbs[sequence].add(candidate.pdb_id)

                        previous = best_by_sequence.get(sequence)
                        if previous is None or candidate_rank(candidate) < candidate_rank(previous):
                            best_by_sequence[sequence] = candidate

            if completed % 25 == 0 or completed == total:
                print(f"  RCSB entries analysed: {completed}/{total}")

    # A non-contact observation is deliberately NOT a negative label.
    # Remove sequences that also have positive evidence from this count.
    noncontact_only = noncontact_observations - set(best_by_sequence)

    accepted = sorted(best_by_sequence.items(), key=lambda item: sha256_text(item[0]))

    rows = []
    structure_hashes: dict[str, str] = {}

    for sequence, candidate in accepted:
        pdb_id = candidate.pdb_id
        path = structure_path(pdb_id)
        if pdb_id not in structure_hashes and path.exists():
            structure_hashes[pdb_id] = sha256_file(path)

        rows.append(
            {
                "source_id": f"rcsb_{sha256_text(sequence)[:16]}",
                "sequence_sha256": sha256_text(sequence),
                "status": "accepted",
                "reason": "direct_protein_dna_contact_in_declared_biological_assembly",
                "pdb_id": candidate.pdb_id,
                "chain_id": candidate.chain_id,
                "assembly_id": candidate.assembly_id,
                "protein_copy": candidate.protein_copy,
                "assembly_source": candidate.assembly_source,
                "coordinate_coverage": f"{candidate.coordinate_coverage:.6f}",
                "resolution": (
                    f"{candidate.resolution:.4f}"
                    if math.isfinite(candidate.resolution)
                    else ""
                ),
                "dna_chains": ";".join(candidate.dna_chains),
                "binding_positions": ";".join(map(str, candidate.binding_positions)),
                "binding_residue_count": len(candidate.binding_positions),
                "supporting_pdb_count": len(support_pdbs[sequence]),
                "supporting_pdb_ids": ";".join(sorted(support_pdbs[sequence])),
                "structure_sha256": structure_hashes.get(pdb_id, ""),
            }
        )

    fasta_tmp = atomic_temp(RCSB_POS_FASTA)
    evidence_tmp = atomic_temp(RCSB_POS_EVIDENCE)

    try:
        with fasta_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for sequence, candidate in accepted:
                source_id = f"rcsb_{sha256_text(sequence)[:16]}"
                header = (
                    f"{candidate.pdb_id}_{safe_token(candidate.chain_id)}"
                    f"|assembly_{safe_token(candidate.assembly_id)}"
                    f"|copy_{candidate.protein_copy}"
                    f"|label_1"
                    f"|origin_rcsb_structural"
                    f"|source_{source_id}"
                )
                handle.write(f">{header}\n{sequence}\n")

        with evidence_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=RCSB_POS_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(fasta_tmp, RCSB_POS_FASTA)
        os.replace(evidence_tmp, RCSB_POS_EVIDENCE)
    finally:
        fasta_tmp.unlink(missing_ok=True)
        evidence_tmp.unlink(missing_ok=True)

    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "rcsb-direct-structural-positive",
        "source": "RCSB PDB experimental biological assemblies",
        "release_cutoff": RCSB_CUTOFF,
        "query_snapshot": str(rcsb_snapshot_path()),
        "query_snapshot_sha256": sha256_file(rcsb_snapshot_path()),
        "candidate_assembly_count": len(snapshot["assembly_ids"]),
        "candidate_pdb_entry_count": len(jobs),
        "downloaded_or_cached_entries": len(downloaded),
        "download_error_count": len(download_errors),
        "analysis_error_count": len(analysis_errors),
        "eligible_unique_protein_sequences_observed": len(eligible_sequences),
        "noncontact_only_unique_sequences_observed": len(noncontact_only),
        "noncontact_policy": (
            "Non-contact in a DNA-containing assembly is NOT considered a negative label. "
            "Such proteins remain unlabeled and are excluded from the dataset unless supported "
            "elsewhere as positives."
        ),
        "positive_contact_observations": candidate_observation_count,
        "accepted_unique_positive_sequences": len(accepted),
        "minimum_protein_length": MIN_PROTEIN_LENGTH,
        "min_coordinate_coverage": MIN_COORDINATE_COVERAGE,
        "contact_definition": "distance < vdW(protein)+vdW(DNA)+0.5 Angstrom",
        "dna_policy": "pure DNA polymer chains only; RNA and nucleic-acid hybrids excluded",
        "assembly_policy": (
            "Use author-determined biological assemblies when the entry provides them; "
            "otherwise use declared biological assemblies. Protein copies are evaluated "
            "independently."
        ),
        "candidate_ranking_policy": (
            "Prefer author assembly, higher coordinate coverage, then better numeric resolution; "
            "never rank by interface size/contact count."
        ),
        "download_errors": dict(sorted(download_errors.items())),
        "analysis_errors": dict(sorted(analysis_errors.items())),
        "worker_note_counts": dict(sorted(worker_notes.items())),
    }
    atomic_write_json(RCSB_POS_SUMMARY, summary)

    print(
        f"RCSB structural positives complete: {len(accepted)} unique positive sequences."
    )
    print(
        f"Observed {len(noncontact_only)} unique protein sequences with no direct contact "
        "in the inspected context; they were NOT labelled negative."
    )


# =============================================================================
# Stage 3: Core + direct RCSB positives
# =============================================================================

EXPANDED_FIELDS = [
    "sequence_sha256",
    "label",
    "origin",
    "status",
    "reason",
    "core_header",
    "rcsb_header",
    "final_header",
]


def upstream_hashes() -> dict[str, str]:
    return {
        "core_fasta_sha256": sha256_file(CORE_FASTA),
        "rcsb_positive_fasta_sha256": sha256_file(RCSB_POS_FASTA),
    }


def expanded_is_fresh() -> bool:
    if not (
        EXPANDED_FULL_JSONL.exists()
        and
        EXPANDED_FULL_FASTA.exists()
        and EXPANDED_FULL_EVIDENCE.exists()
        and EXPANDED_FULL_SUMMARY.exists()
        and BALANCED_FASTA.exists()
        and BALANCED_SUMMARY.exists()
    ):
        return False

    try:
        with EXPANDED_FULL_SUMMARY.open("r", encoding="utf-8") as handle:
            summary = json.load(handle)
    except Exception:
        return False

    expected = {
        **upstream_hashes(),
        "canonical_jsonl_sha256": sha256_file(EXPANDED_FULL_JSONL),
        "compatibility_fasta_sha256": sha256_file(EXPANDED_FULL_FASTA),
        "expanded_evidence_sha256": sha256_file(EXPANDED_FULL_EVIDENCE),
        "balanced_fasta_sha256": sha256_file(BALANCED_FASTA),
    }
    return all(summary.get(key) == value for key, value in expected.items())


def balanced_key(record: FinalRecord) -> str:
    return sha256_text(f"{BALANCE_SALT}\n{record.label}\n{record.sequence}")


def raw_json_record(record: FinalRecord) -> dict[str, object]:
    """Convert one final FASTA record into the canonical explicit JSONL schema.

    The negative evidence category deliberately describes BTD's exclusion-derived benchmark
    protocol. It does not claim that absence of an observed structural contact proves non-binding.
    Positive categories record whether the selected chain has direct RCSB support or a BTD label
    whose DNA contact was revalidated during the strict Core stage.
    """
    fields = record.header.split("|")
    metadata = {
        name: value
        for name, value in (field.split("_", 1) for field in fields[1:])
    }
    identifier = fields[0]
    pdb_id, protein_chain = identifier.split("_", 1)
    if record.label == 0:
        label_evidence = "benchmark_exclusion_derived_negative"
    elif "rcsb" in record.origin:
        label_evidence = "direct_structural_dna_contact"
    else:
        label_evidence = "benchmark_positive_with_revalidated_dna_contact"
    known = {"assembly", "copy", "label", "origin", "source"}
    return {
        "schema_version": RAW_SCHEMA_VERSION,
        "identifier": identifier,
        "pdb_id": pdb_id.upper(),
        "protein_chain": protein_chain,
        "assembly_id": metadata["assembly"],
        "protein_copy": int(metadata["copy"]),
        "label": record.label,
        "label_evidence": label_evidence,
        "origin": record.origin,
        "source": metadata["source"],
        "sequence": record.sequence,
        "sequence_sha256": sha256_text(record.sequence),
        "original_header": record.header,
        "header_flags": sorted(
            field for field in fields[1:] if field.split("_", 1)[0] not in known
        ),
    }


def build_expanded() -> None:
    print("\n=== Stage 3: building full Expanded + reproducible 1:1 view ===")

    core = read_final_fasta(CORE_FASTA, "btd_core")
    rcsb = read_final_fasta(RCSB_POS_FASTA, "rcsb_structural")

    core_by_sequence = {record.sequence: record for record in core}
    rcsb_by_sequence = {record.sequence: record for record in rcsb}

    if len(core_by_sequence) != len(core):
        raise RuntimeError("Core FASTA contains duplicate exact sequences")
    if len(rcsb_by_sequence) != len(rcsb):
        raise RuntimeError("RCSB positive FASTA contains duplicate exact sequences")

    final_records: list[FinalRecord] = []
    rows = []
    counts = defaultdict(int)

    all_sequences = sorted(
        set(core_by_sequence) | set(rcsb_by_sequence),
        key=sha256_text,
    )

    for sequence in all_sequences:
        core_record = core_by_sequence.get(sequence)
        rcsb_record = rcsb_by_sequence.get(sequence)
        seq_hash = sha256_text(sequence)

        if core_record is not None and rcsb_record is not None:
            if core_record.label == 1:
                final_header = (
                    core_record.header
                    + "|origin_btd_core|rcsb_structural_supported_1"
                )
                final_records.append(
                    FinalRecord(
                        final_header,
                        sequence,
                        1,
                        "btd_core+rcsb_support",
                    )
                )
                counts["core_positive_also_supported_by_rcsb"] += 1
                rows.append(
                    {
                        "sequence_sha256": seq_hash,
                        "label": 1,
                        "origin": "btd_core+rcsb_support",
                        "status": "included",
                        "reason": "exact_sequence_present_in_core_positive_and_rcsb_contact_set",
                        "core_header": core_record.header,
                        "rcsb_header": rcsb_record.header,
                        "final_header": final_header,
                    }
                )
                continue

            # Strong structural positive evidence contradicts the BTD negative.
            # Do not silently relabel; quarantine this exact sequence.
            counts["cross_source_exact_label_conflict"] += 1
            rows.append(
                {
                    "sequence_sha256": seq_hash,
                    "label": "",
                    "origin": "conflict",
                    "status": "excluded",
                    "reason": "btd_core_negative_vs_direct_rcsb_dna_contact",
                    "core_header": core_record.header,
                    "rcsb_header": rcsb_record.header,
                    "final_header": "",
                }
            )
            continue

        if core_record is not None:
            final_header = core_record.header + "|origin_btd_core"
            final_records.append(
                FinalRecord(
                    final_header,
                    sequence,
                    core_record.label,
                    "btd_core",
                )
            )
            counts[f"included_core_label_{core_record.label}"] += 1
            rows.append(
                {
                    "sequence_sha256": seq_hash,
                    "label": core_record.label,
                    "origin": "btd_core",
                    "status": "included",
                    "reason": "core_only",
                    "core_header": core_record.header,
                    "rcsb_header": "",
                    "final_header": final_header,
                }
            )
            continue

        assert rcsb_record is not None
        final_header = rcsb_record.header
        final_records.append(
            FinalRecord(
                final_header,
                sequence,
                1,
                "rcsb_structural",
            )
        )
        counts["included_rcsb_structural_addition"] += 1
        rows.append(
            {
                "sequence_sha256": seq_hash,
                "label": 1,
                "origin": "rcsb_structural",
                "status": "included",
                "reason": "new_direct_rcsb_structural_positive",
                "core_header": "",
                "rcsb_header": rcsb_record.header,
                "final_header": final_header,
            }
        )

    final_records.sort(key=lambda record: (record.label, sha256_text(record.sequence)))
    rows.sort(key=lambda row: str(row["sequence_sha256"]))

    # Write the canonical FULL dataset first.
    jsonl_tmp   = atomic_temp(EXPANDED_FULL_JSONL)
    fasta_tmp   = atomic_temp(EXPANDED_FULL_FASTA)
    evidence_tmp = atomic_temp(EXPANDED_FULL_EVIDENCE)

    try:
        with jsonl_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in final_records:
                handle.write(json.dumps(raw_json_record(record), sort_keys=True) + "\n")

        with fasta_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in final_records:
                handle.write(f">{record.header}\n{record.sequence}\n")

        with evidence_tmp.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=EXPANDED_FIELDS)
            writer.writeheader()
            writer.writerows(rows)

        os.replace(jsonl_tmp, EXPANDED_FULL_JSONL)
        os.replace(fasta_tmp, EXPANDED_FULL_FASTA)
        os.replace(evidence_tmp, EXPANDED_FULL_EVIDENCE)
    finally:
        jsonl_tmp.unlink(missing_ok=True)
        fasta_tmp.unlink(missing_ok=True)
        evidence_tmp.unlink(missing_ok=True)

    positives = [record for record in final_records if record.label == 1]
    negatives = [record for record in final_records if record.label == 0]

    hashes = upstream_hashes()
    full_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "expanded-full",
        "canonical_format": (
            f"JSON Lines schema {RAW_SCHEMA_VERSION}; one explicit evidence record per line"
        ),
        "canonical_jsonl": str(EXPANDED_FULL_JSONL),
        "canonical_jsonl_sha256": sha256_file(EXPANDED_FULL_JSONL),
        "compatibility_fasta": str(EXPANDED_FULL_FASTA),
        "compatibility_fasta_sha256": sha256_file(EXPANDED_FULL_FASTA),
        "expanded_evidence_sha256": sha256_file(EXPANDED_FULL_EVIDENCE),
        **hashes,
        "final_total": len(final_records),
        "final_positive": len(positives),
        "final_negative": len(negatives),
        "positive_fraction": (
            len(positives) / len(final_records)
            if final_records
            else 0.0
        ),
        "counts": dict(sorted(counts.items())),
        "negative_policy": (
            "No new RCSB negatives are created. Negative labels come only from BTD's "
            "exclusion-derived benchmark class after exact-sequence mapping and contradiction "
            "auditing. This is benchmark evidence, not universal experimental proof of "
            "non-binding. Absence of observed DNA contact is never a negative label."
        ),
        "policy": (
            "Exact-sequence duplicates are represented once. Core positives take provenance "
            "precedence when independently supported by RCSB. Exact Core-negative/direct-RCSB-"
            "positive conflicts are quarantined rather than silently relabelled."
        ),
    }

    # -------------------------------------------------------------------------
    # Reproducible 1:1 convenience view
    #
    # We keep the canonical full dataset untouched. The balanced view is a
    # deterministic hash sample of EACH class to the smaller class size.
    #
    # This avoids choosing examples by resolution, interface size, morphology,
    # or another biological feature that could introduce selection bias.
    #
    # Final train/val/test balancing must still be done later at homology-group
    # level; this file is only a convenient raw view.
    # -------------------------------------------------------------------------

    target = min(len(positives), len(negatives))

    selected_positive = sorted(positives, key=balanced_key)[:target]
    selected_negative = sorted(negatives, key=balanced_key)[:target]
    balanced = selected_positive + selected_negative
    balanced.sort(key=lambda record: (record.label, balanced_key(record)))

    balanced_tmp = atomic_temp(BALANCED_FASTA)
    try:
        with balanced_tmp.open("w", encoding="utf-8", newline="\n") as handle:
            for record in balanced:
                handle.write(f">{record.header}|balanced_view_1\n{record.sequence}\n")
        os.replace(balanced_tmp, BALANCED_FASTA)
    finally:
        balanced_tmp.unlink(missing_ok=True)

    balanced_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "tier": "expanded-balanced-view",
        "source_full_fasta": str(EXPANDED_FULL_FASTA),
        "source_full_fasta_sha256": sha256_file(EXPANDED_FULL_FASTA),
        "balanced_fasta_sha256": sha256_file(BALANCED_FASTA),
        "selection": (
            "Deterministic SHA-256 hash sampling with a fixed salt; no biological/"
            "structural feature is used for selection."
        ),
        "balance_salt": BALANCE_SALT,
        "target_per_class": target,
        "final_total": len(balanced),
        "final_positive": len(selected_positive),
        "final_negative": len(selected_negative),
        "warning": (
            "This is not the final train/validation/test split. Final balancing must "
            "respect MMseqs2/Foldseek leakage groups and morphology constraints."
        ),
    }
    atomic_write_json(BALANCED_SUMMARY, balanced_summary)

    # The full summary is written last because its freshness contract covers the canonical JSONL,
    # compatibility/evidence views, and the derived balanced FASTA. An interrupted earlier write
    # can therefore never make the complete stage look reusable.
    full_summary["balanced_fasta_sha256"] = sha256_file(BALANCED_FASTA)
    atomic_write_json(EXPANDED_FULL_SUMMARY, full_summary)

    print(
        f"Expanded full: {len(final_records)} proteins "
        f"({len(positives)} positive, {len(negatives)} negative)."
    )
    print(
        f"Balanced view: {len(balanced)} proteins "
        f"({len(selected_positive)} positive, {len(selected_negative)} negative)."
    )
    if counts["cross_source_exact_label_conflict"]:
        print(
            "WARNING: "
            f"{counts['cross_source_exact_label_conflict']} exact Core-negative/RCSB-positive "
            "conflicts were quarantined. Inspect the Expanded evidence CSV."
        )


# =============================================================================
# Stage orchestration
# =============================================================================

def main() -> None:
    args = parse_args()

    # Keep canonical inputs and bulky reproducible build evidence in predictable project paths.
    RAW_ROOT.mkdir(parents=True, exist_ok=True)
    BUILD_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n============================================")
    print("      WISDOM DNA RCSB dataset builder")
    print("============================================")
    print(f"CPUs visible:       {args.available_cpus}")
    print(f"CPU workers:        {args.workers}")
    print(f"RCSB release cutoff:{RCSB_CUTOFF}")
    print("BTD mapping:        exact full sequence only")
    print(f"Coord. coverage:    >= {MIN_COORDINATE_COVERAGE:.0%}")
    print(f"New RCSB min length:{MIN_PROTEIN_LENGTH} aa")
    print("New negatives:      NONE inferred from RCSB absence/non-contact")

    # Stage 1: strict BTD Core.
    rebuild_core = args.force or args.force_core or not CORE_FASTA.exists()
    if rebuild_core:
        build_core(args.workers)
    else:
        print(f"\n=== Stage 1: Core already exists -> reuse {CORE_FASTA} ===")
        if not CORE_EVIDENCE.exists() or not CORE_SUMMARY.exists():
            print(
                "WARNING: Core FASTA exists but one or more audit sidecars are missing. "
                "The FASTA will still be reused because it is the requested completion marker."
            )

    if not CORE_FASTA.exists():
        raise RuntimeError("Core stage did not produce the expected FASTA")

    # Stage 2: exhaustive structural-positive expansion.
    rebuild_rcsb = args.force or args.force_rcsb or not RCSB_POS_FASTA.exists()
    if rebuild_rcsb:
        build_rcsb_structural_positives(
            args.workers,
            refresh_query=(args.force or args.refresh_rcsb_query),
        )
    else:
        print(
            f"\n=== Stage 2: RCSB structural positives already exist -> reuse "
            f"{RCSB_POS_FASTA} ==="
        )

    if not RCSB_POS_FASTA.exists():
        raise RuntimeError("RCSB positive stage did not produce the expected FASTA")

    # Stage 3: combine + balance. Rebuild automatically if upstream FASTAs changed.
    rebuild_expanded = (
        args.force
        or args.force_expanded
        or not expanded_is_fresh()
    )

    if rebuild_expanded:
        build_expanded()
    else:
        print(
            f"\n=== Stage 3: Expanded outputs are fresh -> reuse "
            f"{EXPANDED_FULL_FASTA} and {BALANCED_FASTA} ==="
        )

    print("\n============================================")
    print("                 DONE")
    print("============================================")
    print(f"Core:             {CORE_FASTA}")
    print(f"RCSB positives:   {RCSB_POS_FASTA}")
    print(f"Expanded JSONL:   {EXPANDED_FULL_JSONL}")
    print(f"Compatibility FASTA: {EXPANDED_FULL_FASTA}")
    print(f"Balanced view:    {BALANCED_FASTA}")
    print()
    print(
        "Use Expanded FULL as the canonical raw pool. The balanced file is only "
        "a convenience view; final split balancing must happen after homology/"
        "structure leakage grouping."
    )


if __name__ == "__main__":
    main()
