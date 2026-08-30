"""Focused tests for the simple WISDOM-DNA selection stages."""

import ast
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from wisdom.preprocessing.dna.selection.audit import audit_dataset
from wisdom.preprocessing.dna.selection.dilutions import create_dilutions
from wisdom.preprocessing.dna.selection.evidence import load_evidence
from wisdom.preprocessing.dna.selection.leakage import assign_leakage_groups
from wisdom.preprocessing.dna.selection.population import select_population
from wisdom.preprocessing.dna.selection.report import write_design
from wisdom.preprocessing.dna.selection.similarity import _sequence_edges, _structure_edges
from wisdom.preprocessing.dna.selection.splits import assign_splits
from wisdom.preprocessing.dna.selection.structures import _contacts, snapshot_structures


class Log:
    """Collect evidence-loader messages without constructing a LambdaForge runner."""

    def __init__(self) -> None:
        """Create an empty message list."""
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        """Store one researcher-facing message.

        Args:
            message: Message emitted by the selection stage.
        """
        self.messages.append(message)


def test_pipeline_imports_follow_project_visual_order() -> None:
    """Selection/preprocessing imports stay global and ordered by statement length."""
    parent = Path(__file__).parents[1] / "src" / "wisdom" / "preprocessing" / "dna"
    paths  = [
        path
        for directory in (parent / "selection", parent / "preprocessing")
        for path in directory.glob("*.py")
    ]
    for path in paths:
        source = path.read_text(encoding="utf-8")
        tree   = ast.parse(source)

        imports = [
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.Import, ast.ImportFrom))
        ]
        assert all(node in tree.body for node in imports), f"{path.name} has a local import"

        lines      = source.splitlines()
        plain      = [lines[node.lineno - 1] for node in imports if isinstance(node, ast.Import)]
        from_lines = [
            lines[node.lineno - 1]
            for node in imports
            if isinstance(node, ast.ImportFrom)
        ]

        assert list(map(len, plain)) == sorted(map(len, plain))
        assert list(map(len, from_lines)) == sorted(map(len, from_lines))
        if plain and from_lines:
            assert max(node.lineno for node in imports if isinstance(node, ast.Import)) < min(
                node.lineno for node in imports if isinstance(node, ast.ImportFrom)
            )


def test_evidence_preserves_multi_character_chain_name(tmp_path: Path) -> None:
    """The text after the first underscore is one chain name, not a character list."""
    path = tmp_path / "raw.jsonl"
    path.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "identifier": "1ABC_AQ",
                "sequence": "ACDE",
                "label": 0,
                "origin": "fixture",
                "source": "fixture",
                "label_evidence": "experimental_negative",
                "assembly_id": "1",
                "protein_copy": 1,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_evidence(Log(), path, False)

    assert rows[0]["pdb_id"] == "1ABC"
    assert rows[0]["protein_chain"] == "AQ"


def test_similarity_thresholds_create_canonical_edges(tmp_path: Path) -> None:
    """Both specialist parsers require bilateral coverage and canonicalize edge direction."""
    rows = [{"identifier": "A"}, {"identifier": "B"}]
    parameters = {
        "sequence_identity": 0.30,
        "sequence_coverage": 0.80,
        "sequence_evalue": 1e-3,
        "foldseek_probability": 0.90,
        "foldseek_tmscore": 0.75,
        "foldseek_coverage": 0.80,
        "foldseek_evalue": 1e-3,
    }
    sequence = tmp_path / "sequence.tsv"
    sequence.write_text("B\tA\t35\t90\t85\t1e-8\t100\n", encoding="utf-8")
    structure = tmp_path / "structure.tsv"
    structure.write_text("B.cif\tA.cif\t95\t1e-8\t0.8\t0.9\t0.85\t0.90\n", encoding="utf-8")

    assert _sequence_edges(sequence, rows, parameters) == {("A", "B")}
    assert _structure_edges(structure, rows, parameters) == {("A", "B")}


def test_leakage_balance_splits_and_dilutions_are_consistent() -> None:
    """The complete in-memory selection preserves balance and indivisible groups."""
    rows = []
    for index in range(60):
        label = index % 2
        rows.append(
            {
                "identifier": f"P{index:03d}_A",
                "pdb_id": f"P{index:03d}",
                "sequence_sha256": f"sequence-{index}",
                "label": label,
                "quality_eligible": True,
                "origin": "btd_core" if label else "btd_combo",
                "global_phenotype": f"G{index % 3}",
                "interface_phenotype": f"I{index % 2}" if label else "not_applicable",
            }
        )
    similarity = {"sequence_edges": {("P000_A", "P002_A")}, "structure_edges": set()}
    rows, leakage = assign_leakage_groups(rows, similarity, True)
    selected, _ = select_population(rows, 1.0, True, True, 7)
    selected, _ = assign_splits(selected, 0.70, 0.15, 0.15, 7)
    dilutions = create_dilutions(selected, (1.0, 0.5, 0.1), 2, 7)
    audit = audit_dataset(rows, selected, dilutions)

    assert audit["valid"] is True
    assert audit["selected_counts"] == {"total": 60, "positive": 30, "negative": 30}
    assert leakage["largest_component"] == 2
    group_splits: dict[str, set[str]] = {}
    for row in selected:
        group_splits.setdefault(str(row["leakage_group"]), set()).add(str(row["split"]))
    assert all(len(splits) == 1 for splits in group_splits.values())
    for subsets in dilutions["replicates"].values():
        ten = set(subsets["train-10"]["identifiers"])
        half = set(subsets["train-50"]["identifiers"])
        full = set(subsets["train-100"]["identifiers"])
        assert ten <= half <= full


def test_contact_cutoff_uses_van_der_waals_gap() -> None:
    """Sparse contact logic accepts a near atom and rejects a distant atom."""
    protein = {
        "atom_positions": np.asarray([[0.0, 0.0, 0.0]]),
        "atom_radii": np.asarray([1.7]),
        "atom_owners": np.asarray([4]),
    }
    near = {
        "positions": np.asarray([[3.5, 0.0, 0.0]]),
        "radii": np.asarray([1.7]),
        "owners": np.asarray([0]),
    }
    far = {**near, "positions": np.asarray([[4.0, 0.0, 0.0]])}

    assert _contacts(protein, near)["pair_count"] == 1
    assert _contacts(protein, far)["pair_count"] == 0


def test_structure_snapshot_is_exact_and_deterministic(tmp_path: Path) -> None:
    """Selection preserves exact source bytes in a reproducible gzip snapshot.

    Args:
        tmp_path: Isolated directories receiving two independently written snapshots.
    """
    source = Path(__file__).parent / "data" / "tiny.pdb"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    rows   = [
        {
            "pdb_id":            "1ABC",
            "source_structure":  source,
            "structure_sha256":  digest,
        }
    ]

    first  = snapshot_structures(tmp_path / "first", rows)
    second = snapshot_structures(tmp_path / "second", rows)

    assert gzip.decompress((first / "1abc.cif.gz").read_bytes()) == source.read_bytes()
    assert (first / "1abc.cif.gz").read_bytes() == (second / "1abc.cif.gz").read_bytes()

    index = json.loads((first / "index.json").read_text(encoding="utf-8"))
    assert index["structures"][0]["uncompressed_sha256"] == digest


def test_writer_produces_preprocessing_contract(tmp_path: Path) -> None:
    """The final writer produces three complete, self-contained JSONL manifests."""
    selected = [_catalog_row("1AAA_A", 0, "train"), _catalog_row("2AAA_A", 1, "validation")]
    selected.append(_catalog_row("3AAA_A", 0, "test"))
    raw = [{**row, "quality_eligible": True} for row in selected]
    sequence = tmp_path / "sequence.tsv"
    structure = tmp_path / "structure.tsv"
    sequence.write_text("1AAA_A\t1AAA_A\t100\t100\t100\t0\t10\n", encoding="utf-8")
    structure.write_text(
        "1AAA_A.cif\t1AAA_A.cif\t100\t0\t1\t1\t1\t1\n",
        encoding="utf-8",
    )
    dilutions = {
        "validation_sha256": "a",
        "test_sha256": "b",
        "replicates": {
            "replicate-00": {
                "train-100": {
                    "fraction": 1.0,
                    "identifiers": ["1AAA_A"],
                    "counts": {"total": 1, "positive": 0, "negative": 1},
                    "leakage_groups": ["L1"],
                }
            }
        },
    }
    audit = {
        "selected_counts": {"total": 3, "positive": 1, "negative": 2},
        "splits": {
            split: {
                "total": 1,
                "positive": int(split == "validation"),
                "negative": int(split != "validation"),
                "leakage_groups": 1,
            }
            for split in ("train", "validation", "test")
        },
        "warnings": [],
    }
    phenotype = {
        "global": {"eligible": 3, "clusters": 0, "noise_fraction": 1.0},
        "interface": {"eligible": 1, "clusters": 0, "noise_fraction": 1.0},
    }
    selection_audit = {"selected": {"total": 3}, "omitted": []}
    split_audit = {"method": "fixture"}
    leakage = {
        "sequence_edges": [],
        "structure_edges": [],
        "exact_pairs": [],
    }
    parameters = {
        "seed": 7,
        "sequence_identity": 0.3,
        "sequence_coverage": 0.8,
        "sequence_evalue": 1e-3,
        "foldseek_probability": 0.9,
        "foldseek_tmscore": 0.75,
        "foldseek_coverage": 0.8,
        "foldseek_evalue": 1e-3,
    }

    write_design(
        tmp_path,
        raw,
        selected,
        leakage,
        phenotype,
        dilutions,
        selection_audit,
        split_audit,
        audit,
        {"sequence_path": sequence, "structure_path": structure},
        parameters,
    )

    assert (tmp_path / "REPORT.md").is_file()
    assert (tmp_path / "train-labelled.txt").read_text() == "1AAA_A\t0\n"
    manifests = {
        name: [
            json.loads(line)
            for line in (tmp_path / "preprocessing" / name).read_text().splitlines()
        ]
        for name in ("train.jsonl", "val.jsonl", "test.jsonl")
    }
    assert [row["identifier"] for row in manifests["train.jsonl"]] == ["1AAA_A"]
    assert [row["identifier"] for row in manifests["val.jsonl"]] == ["2AAA_A"]
    assert [row["identifier"] for row in manifests["test.jsonl"]] == ["3AAA_A"]
    assert manifests["train.jsonl"][0]["dilutions"] == ["replicate-00/train-100"]
    assert manifests["val.jsonl"][0]["assembly_rotation"] == np.eye(3).tolist()


def _catalog_row(identifier: str, label: int, split: str) -> dict[str, object]:
    """Create one minimal canonical output row for writer tests.

    Args:
        identifier: PDB-chain identity.
        label: Binary DNA-binding label.
        split: Supervised role.

    Returns:
        Complete row required by structural preprocessing.
    """
    pdb_id, chain = identifier.split("_", 1)
    return {
        "identifier": identifier,
        "base_identifier": identifier,
        "sequence": "ACDE",
        "label": label,
        "split": split,
        "selected": True,
        "leakage_group": f"L-{identifier}",
        "global_phenotype": "G_NOISE",
        "global_phenotype_probability": 0.0,
        "interface_phenotype": "I_NOISE" if label else "not_applicable",
        "interface_phenotype_probability": 0.0,
        "origin": "fixture",
        "label_evidence": "fixture",
        "pdb_id": pdb_id,
        "protein_chain": chain,
        "assembly_id": "1",
        "protein_copy": 1,
        "structure_sha256": "a" * 64,
        "dna_chains": [],
        "binding_residue_indices": [],
        "local_gt_expected": True,
        "local_gt_method": "dna_distance" if label else "global_negative",
        "assembly_rotation": np.eye(3).tolist(),
        "assembly_translation": [0.0, 0.0, 0.0],
        "quality_eligible": True,
        "quality_exclusion_reason": "",
    }
