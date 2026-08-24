"""Focused deterministic tests for the single-file WISDOM-DNA design Work."""

from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wisdom.preprocessing.DatasetDesign import DatasetDesign
from wisdom.preprocessing.dna.DNAValidation import DNAValidation


def row(
    identifier: str,
    label: int,
    group: str,
    origin: str = "fixture",
) -> dict[str, Any]:
    """Create one compact synthetic descriptor row for pure design algorithms."""
    return {
        "identifier": identifier,
        "label": label,
        "leakage_group": group,
        "global_phenotype": f"G{label + 1:03d}",
        "interface_phenotype": "I001" if label else "not_applicable",
        "origin": origin,
        "sequence_length": 100.0,
        "coordinate_coverage": 0.95,
        "resolution": 2.0,
        "release_year": 2020.0,
        "pdb_id": identifier[:4],
        "sequence_sha256": hashlib.sha256(identifier.encode()).hexdigest(),
    }


def test_fasta_parser_preserves_explicit_identity(tmp_path: Path) -> None:
    """The fixed two-line FASTA contract must become one record per header/sequence pair."""
    source = tmp_path / "raw.fasta"
    source.write_text(
        ">1ABC_A|assembly_2|copy_1|label_1|origin_btd_core|source_fixture\nACDEFG\n"
        ">2ABC_B|assembly_1|copy_3|label_0|source_fixture|origin_btd_core\nHIKLMN\n",
        encoding="utf-8",
    )
    parsed = DatasetDesign()._read_fasta(source)

    assert len(parsed) == 2
    assert parsed[0]["identifier"] == "1ABC_A"
    assert parsed[0]["assembly_id"] == "2"
    assert parsed[0]["protein_copy"] == 1
    assert parsed[0]["label"] == 1
    assert parsed[1]["identifier"] == "2ABC_B"
    assert parsed[1]["protein_copy"] == 3
    assert parsed[1]["sequence"] == "HIKLMN"
    assert parsed[1]["label"] == 0


def test_jsonl_parser_preserves_typed_label_evidence(tmp_path: Path) -> None:
    """Canonical JSONL must preserve explicit evidence without parsing overloaded headers."""
    source = tmp_path / "raw.jsonl"
    source.write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "identifier": "1ABC_A",
                "assembly_id": "2",
                "protein_copy": 1,
                "label": 0,
                "label_evidence": "benchmark_exclusion_derived_negative",
                "origin": "btd_core",
                "source": "fixture",
                "sequence": "ACDEFG",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    parsed = DatasetDesign()._read_records(source)

    assert parsed[0]["identifier"] == "1ABC_A"
    assert parsed[0]["label_evidence"] == "benchmark_exclusion_derived_negative"
    assert parsed[0]["sequence_sha256"] == hashlib.sha256(b"ACDEFG").hexdigest()


def test_labelled_manifest_is_sorted_and_tab_delimited() -> None:
    """Human label views must remain deterministic projections of catalog rows."""
    content = DatasetDesign()._labelled_manifest(
        [{"identifier": "2ABC_B", "label": 1}, {"identifier": "1ABC_A", "label": 0}]
    )
    assert content == "1ABC_A\t0\n2ABC_B\t1\n"


def test_manifest_audit_detects_label_drift(tmp_path: Path) -> None:
    """Validation must reject a labelled convenience view that contradicts the catalog."""
    catalog = {
        "1ABC_A": {"label": "0", "split": "train"},
        "2ABC_B": {"label": "1", "split": "validation"},
        "3ABC_C": {"label": "0", "split": "test"},
    }
    for name, identifiers in {
        "proteins": tuple(catalog),
        "train": ("1ABC_A",),
        "validation": ("2ABC_B",),
        "test": ("3ABC_C",),
    }.items():
        (tmp_path / f"{name}.txt").write_text(
            "".join(f"{identifier}\n" for identifier in identifiers), encoding="utf-8"
        )
        (tmp_path / f"{name}-labelled.txt").write_text(
            "".join(
                f"{identifier}\t{catalog[identifier]['label']}\n"
                for identifier in identifiers
            ),
            encoding="utf-8",
        )

    assert DNAValidation._manifest_audit(tmp_path, catalog) == []
    (tmp_path / "test-labelled.txt").write_text("3ABC_C\t1\n", encoding="utf-8")
    assert DNAValidation._manifest_audit(tmp_path, catalog) == [
        "test: labelled manifest differs from catalog"
    ]


def test_mmseqs_thresholds_require_bilateral_coverage(tmp_path: Path) -> None:
    """High identity with low target coverage is not a sequence leakage edge."""
    evidence = tmp_path / "pairs.tsv"
    evidence.write_text("A\tB\t0.9\t0.9\t0.7\t1e-9\t100\n", encoding="utf-8")
    edges = DatasetDesign()._sequence_edges(
        evidence,
        [row("A", 0, "a"), row("B", 1, "b")],
        {"sequence_identity": 0.3, "sequence_coverage": 0.8, "sequence_evalue": 1e-3},
    )
    assert edges == set()


def test_mmseqs_thresholds_accept_qualifying_pair(tmp_path: Path) -> None:
    """Identity, two coverages, and E-value jointly create one canonical edge."""
    evidence = tmp_path / "pairs.tsv"
    evidence.write_text("B\tA\t35\t90\t85\t1e-9\t100\n", encoding="utf-8")
    edges = DatasetDesign()._sequence_edges(
        evidence,
        [row("A", 0, "a"), row("B", 1, "b")],
        {"sequence_identity": 0.3, "sequence_coverage": 0.8, "sequence_evalue": 1e-3},
    )
    assert edges == {("A", "B")}


def test_foldseek_thresholds_require_every_symmetric_measure(tmp_path: Path) -> None:
    """One low target-normalized TM-score must reject an otherwise strong pair."""
    evidence = tmp_path / "pairs.tsv"
    evidence.write_text("A.pdb\tB.pdb\t95\t1e-9\t0.9\t0.7\t0.9\t0.9\n", encoding="utf-8")
    edges = DatasetDesign()._structure_edges(
        evidence,
        [row("A", 0, "a"), row("B", 1, "b")],
        {
            "foldseek_probability": 0.9,
            "foldseek_tmscore": 0.75,
            "foldseek_coverage": 0.8,
            "foldseek_evalue": 1e-3,
        },
    )
    assert edges == set()


def test_foldseek_thresholds_accept_qualifying_pair(tmp_path: Path) -> None:
    """A pair passing probability, E-value, both TM-scores, and coverage is retained."""
    evidence = tmp_path / "pairs.tsv"
    evidence.write_text("A.pdb\tB.pdb\t95\t1e-9\t0.8\t0.76\t0.9\t0.8\n", encoding="utf-8")
    edges = DatasetDesign()._structure_edges(
        evidence,
        [row("A", 0, "a"), row("B", 1, "b")],
        {
            "foldseek_probability": 0.9,
            "foldseek_tmscore": 0.75,
            "foldseek_coverage": 0.8,
            "foldseek_evalue": 1e-3,
        },
    )
    assert edges == {("A", "B")}


def test_mmseqs_command_requests_auditable_bilateral_columns(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MMseqs2 construction must expose identity, both coverages, E-value, and bits."""
    commands: list[list[str]] = []

    class Tools:
        """Minimal LambdaForge tool-service fixture that records one command."""

        def run(self, command: list[str], **options: Any) -> None:
            """Capture the command and emulate its final TSV without running MMseqs2."""
            assert options == {"name": "MMseqs2", "threads": 7}
            commands.append(command)
            Path(command[4]).write_text("A\tA\t1\t1\t1\t0\t100\n", encoding="utf-8")

    monkeypatch.setattr(DatasetDesign, "temp_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(DatasetDesign, "tools", property(lambda self: Tools()))
    design = DatasetDesign()
    output = tmp_path / "sequence.tsv"
    design._run_mmseqs(
        [{"identifier": "A", "sequence": "ACDE"}],
        output,
        7,
        {
            "mmseqs_executable": "mmseqs",
            "sequence_identity": 0.3,
            "sequence_coverage": 0.8,
            "sequence_evalue": 1e-3,
        },
        "mmseqs",
    )
    assert output.is_file()
    assert commands[0][commands[0].index("--threads") + 1] == "7"
    assert commands[0][-1] == "query,target,fident,qcov,tcov,evalue,bits"


def test_foldseek_command_requests_two_tmscores_and_coverages(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Foldseek construction must retain every symmetric threshold input in its TSV."""
    commands: list[list[str]] = []

    class Tools:
        """Minimal LambdaForge tool-service fixture that records one command."""

        def run(self, command: list[str], **options: Any) -> None:
            """Capture the command and emulate its final TSV without running Foldseek."""
            assert options == {"name": "Foldseek", "threads": 9}
            commands.append(command)
            Path(command[4]).write_text("A\tA\t1\t0\t1\t1\t1\t1\n", encoding="utf-8")

    monkeypatch.setattr(DatasetDesign, "temp_dir", property(lambda self: tmp_path))
    monkeypatch.setattr(DatasetDesign, "tools", property(lambda self: Tools()))
    design = DatasetDesign()
    foldseek_input = tmp_path / "foldseek-input"
    foldseek_input.mkdir()
    structure = foldseek_input / "A.cif"
    structure.write_text("data_A\n", encoding="utf-8")
    output = tmp_path / "structure.tsv"
    design._run_foldseek(
        [{"identifier": "A", "foldseek_structure": structure}],
        output,
        9,
        {"foldseek_executable": "foldseek", "foldseek_evalue": 1e-3},
        "foldseek",
    )
    assert output.is_file()
    assert commands[0][commands[0].index("--threads") + 1] == "9"
    assert commands[0][-1] == "query,target,prob,evalue,qtmscore,ttmscore,qcov,tcov"


def test_connected_components_preserve_transitive_bridge() -> None:
    """An omitted middle candidate still connects two selected endpoints full-raw."""
    components = DatasetDesign()._components(["A", "B", "C"], {("A", "B"), ("B", "C")})
    assert components == [["A", "B", "C"]]


def test_same_pdb_creates_provenance_edge() -> None:
    """Different chains/copies of one deposition become an auditable hard dependency."""
    left = row("1abc_A", 0, "a")
    right = row("1abc_B", 1, "b")
    left["pdb_id"] = right["pdb_id"] = "1abc"
    pairs = DatasetDesign()._exact_pairs([left, right], True)
    assert pairs == [
        {"left": "1abc_A", "right": "1abc_B", "reasons": ["same_pdb_deposition"]}
    ]


def test_too_small_hdbscan_population_becomes_noise() -> None:
    """Unsupported physical clusters remain explicit noise instead of fake phenotypes."""
    rows = [{"identifier": "A", "sequence_length": 100.0, "x": 1.0}]
    result = DatasetDesign()._phenotypes(rows, ("x",), "G", 2, 1, 0.6, 1)
    assert result["labels"] == {"A": "G_NOISE"}
    assert result["diagnostics"]["robust"] is False


def test_worker_count_does_not_change_noise_clustering() -> None:
    """Operational HDBSCAN threads cannot alter an unsupported scientific result."""
    rows = [{"identifier": "A", "sequence_length": 100.0, "x": 1.0}]
    one = DatasetDesign()._phenotypes(rows, ("x",), "G", 2, 1, 0.6, 1)
    many = DatasetDesign()._phenotypes(rows, ("x",), "G", 2, 1, 0.6, 36)
    assert one == many


def test_hdbscan_copy_transition_warning_is_not_emitted() -> None:
    """The sklearn copy-default transition must not flood one warning per stability fit."""
    rows = [
        {
            "identifier": f"P{index:03d}",
            "sequence_length": 100.0,
            "x": float(index // 20) * 20.0 + float(index % 4) * 0.05,
            "y": float(index % 5) * 0.05,
        }
        for index in range(40)
    ]

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        DatasetDesign()._phenotypes(rows, ("x", "y"), "G", 5, 2, 0.0, 1)

    assert not any("default value of `copy`" in str(item.message) for item in captured)


def test_interface_aspect_ratio_measures_the_principal_surface_plane() -> None:
    """A planar interface must report finite elongation rather than inverse thickness."""
    design = DatasetDesign()
    design._interface_region_distance = 8.0
    protein = {
        "residue_indices": [1, 2, 3, 4],
        "residue_positions": [
            [-2.0, -1.0, 0.0],
            [2.0, -1.0, 0.0],
            [-2.0, 1.0, 0.0],
            [2.0, 1.0, 0.0],
        ],
        "residue_letters": ["K", "D", "A", "Y"],
        "observed_residue_count": 4,
        "atom_positions": [[0.0, 0.0, 0.0]],
    }
    contacts = {
        "binding_residues": [1, 2, 3, 4],
        "contacting_atoms": [0],
        "pair_count": 1,
        "contacted_dna_chains": ["D"],
    }

    result = design._interface_features(protein, contacts, 1)

    assert result["interface_aspect_ratio"] == pytest.approx(2.0)


def test_selection_delegates_runtime_infrastructure_to_lambdaforge() -> None:
    """Selection must use managed framework services instead of private cache/process machinery."""
    source = inspect.getsource(importlib.import_module("wisdom.preprocessing.DatasetDesign"))
    for forbidden in (
        "cache.path(",
        "checkpoints.path(",
        "self.map(",
        "subprocess.run(",
        "urllib.request",
        "os.replace(",
        "sklearn.cluster",
    ):
        assert forbidden not in source
    for managed_api in (
        "self.resume_map(",
        "self.cache.fetch(",
        "self.cache.file(",
        "self.checkpoints.file(",
        "self.tools.run(",
        "lf.clustering.HDBSCAN(",
        "lf.clustering.stability(",
        "publish_to = output_directory",
    ):
        assert managed_api in source


def test_balancing_keeps_all_negatives_and_exact_positive_target() -> None:
    """Default selection keeps negatives and returns the requested 1:1 class count."""
    rows = [row(f"N{i}", 0, f"n{i}") for i in range(3)] + [
        row(f"P{i}", 1, f"p{i}") for i in range(6)
    ]
    selected, audit = DatasetDesign()._select_population(rows, True, 1.0, True, False, 7)
    assert DatasetDesign()._class_counts(selected) == {"total": 6, "positive": 3, "negative": 3}
    assert audit["omitted_positive_count"] == 3


def test_balancing_prioritizes_core_positives() -> None:
    """Core positives are retained whenever the requested quota can contain them."""
    rows = [row("N", 0, "n"), row("CORE", 1, "p", "btd_core"), row("OTHER", 1, "q")]
    selected, _ = DatasetDesign()._select_population(rows, True, 1.0, True, True, 7)
    assert {value["identifier"] for value in selected} == {"N", "CORE"}


def test_balancing_is_order_deterministic() -> None:
    """Input iteration order does not change seeded scientific membership."""
    rows = [row(f"N{i}", 0, f"n{i}") for i in range(2)] + [
        row(f"P{i}", 1, f"p{i}") for i in range(5)
    ]
    first, _ = DatasetDesign()._select_population(rows, True, 1.0, True, False, 19)
    second, _ = DatasetDesign()._select_population(list(reversed(rows)), True, 1.0, True, False, 19)
    assert [value["identifier"] for value in first] == [value["identifier"] for value in second]


def test_split_validation_rejects_group_leakage() -> None:
    """One full-raw component may never appear in two final roles."""
    rows = [
        {**row("A", 0, "shared"), "split": "validation"},
        {**row("B", 1, "shared"), "split": "test"},
    ]
    with pytest.raises(RuntimeError, match="group crosses"):
        DatasetDesign()._validate_splits(rows, set(), set(), [])


def test_split_validation_accepts_complete_disjoint_roles() -> None:
    """Disjoint groups with both evaluation labels satisfy the direct invariant audit."""
    rows = [
        {**row("T0", 0, "g0"), "split": "train"},
        {**row("V0", 0, "g1"), "split": "validation"},
        {**row("V1", 1, "g2"), "split": "validation"},
        {**row("E0", 0, "g3"), "split": "test"},
        {**row("E1", 1, "g4"), "split": "test"},
    ]
    DatasetDesign()._validate_splits(rows, set(), set(), [])


def test_split_assignment_is_seed_and_order_deterministic() -> None:
    """The same seed yields identical whole-group assignments for reversed input rows."""
    rows = [row(f"N{i}", 0, f"n{i}") for i in range(8)] + [
        row(f"P{i}", 1, f"p{i}") for i in range(8)
    ]
    parameters = {
        "train_fraction": 0.5,
        "validation_fraction": 0.25,
        "test_fraction": 0.25,
        "split_size_weight": 1.0,
        "split_class_weight": 2.0,
        "split_global_phenotype_weight": 0.5,
        "split_interface_phenotype_weight": 0.5,
        "split_source_weight": 0.25,
        "split_nuisance_weight": 0.25,
        "split_refinement_steps": 20,
    }
    first, _ = DatasetDesign()._assign_splits(rows, parameters, 23)
    second, _ = DatasetDesign()._assign_splits(list(reversed(rows)), parameters, 23)
    assert first == second


def test_dilutions_are_nested_and_group_complete(tmp_path: Path) -> None:
    """Every smaller training view is a prefix of complete leakage groups."""
    rows = [
        {**row(f"T{i}", i % 2, f"g{i // 2}"), "split": "train"} for i in range(8)
    ] + [
        {**row("V", 0, "v"), "split": "validation"},
        {**row("E", 1, "e"), "split": "test"},
    ]
    audit = DatasetDesign()._write_dilutions(rows, tmp_path, (1.0, 0.5, 0.1), 1, 5)
    subsets = audit["replicates"]["replicate-00"]["subsets"]
    assert set(subsets["train-10"]["identifiers"]).issubset(subsets["train-50"]["identifiers"])
    assert set(subsets["train-50"]["identifiers"]).issubset(subsets["train-100"]["identifiers"])
    assert len(subsets["train-10"]["identifiers"]) > 0


def test_dilution_evaluation_hashes_ignore_replicate_count(tmp_path: Path) -> None:
    """Validation/test membership fingerprints stay fixed across dilution rankings."""
    rows = [
        {**row(f"T{i}", i % 2, f"g{i}"), "split": "train"} for i in range(4)
    ] + [
        {**row("V", 0, "v"), "split": "validation"},
        {**row("E", 1, "e"), "split": "test"},
    ]
    one = DatasetDesign()._write_dilutions(rows, tmp_path / "one", (1.0, 0.5), 1, 3)
    two = DatasetDesign()._write_dilutions(rows, tmp_path / "two", (1.0, 0.5), 2, 3)
    assert one["validation_sha256"] == two["validation_sha256"]
    assert one["test_sha256"] == two["test_sha256"]


def test_continuous_statistics_report_known_smd_and_ks() -> None:
    """Basic effect/distribution statistics reproduce a separated synthetic example."""
    result = DatasetDesign()._compare_continuous(
        np.asarray([0.0, 1.0, 2.0]), np.asarray([3.0, 4.0, 5.0])
    )
    assert result["smd"] == pytest.approx(-3.0)
    assert result["ks_statistic"] == pytest.approx(1.0)


def test_categorical_statistics_detect_perfect_association() -> None:
    """Cramer's V approaches one when a category exactly separates binary labels."""
    rows = ([{"label": 0, "source": "A"}] * 20) + ([{"label": 1, "source": "B"}] * 20)
    result = DatasetDesign()._label_category_table(rows, "source")
    assert result["cramers_v"] > 0.9


def test_shortcut_cross_validation_receives_leakage_groups(monkeypatch: pytest.MonkeyPatch) -> None:
    """Technical baselines pass full-raw group IDs to group-aware cross-validation."""
    observed: list[list[str]] = []

    class SplitSpy:
        """Capture groups supplied by the diagnostic baseline without fitting models."""

        def __init__(self, n_splits: int, shuffle: bool, random_state: int) -> None:
            """Accept the same constructor arguments as StratifiedGroupKFold."""
            assert n_splits == 2
            assert shuffle is True
            assert random_state == 31

        def split(
            self,
            matrix: np.ndarray,
            labels: np.ndarray,
            groups: np.ndarray,
        ) -> list[tuple[np.ndarray, np.ndarray]]:
            """Record exact group IDs and return no folds to avoid model fitting."""
            assert len(matrix) == len(labels) == len(groups)
            observed.append(groups.tolist())
            return []

    module = importlib.import_module("wisdom.preprocessing.DatasetDesign")
    monkeypatch.setattr(module, "StratifiedGroupKFold", SplitSpy)
    rows = [
        row("N0", 0, "negative-a"),
        row("N1", 0, "negative-b"),
        row("P0", 1, "positive-a"),
        row("P1", 1, "positive-b"),
    ]
    DatasetDesign()._shortcut_baselines(rows, 31)
    assert observed
    assert observed[0] == [value["leakage_group"] for value in rows]
    assert "groups" in inspect.getsource(DatasetDesign._shortcut_baselines)


def test_managed_output_text_replaces_complete_content(tmp_path: Path) -> None:
    """Writing within an attempt-owned output leaves the complete newest payload."""
    target = tmp_path / "nested" / "result.txt"
    DatasetDesign()._write_text(target, "first\n")
    DatasetDesign()._write_text(target, "second\n")
    assert target.read_text(encoding="utf-8") == "second\n"


def test_contact_validation_uses_element_radii() -> None:
    """A close protein/DNA heavy-atom pair is a contact while a distant pair is not."""
    protein = {
        "atom_positions": np.asarray([[0.0, 0.0, 0.0]]),
        "atom_radii": np.asarray([1.7]),
        "atom_owners": np.asarray([0]),
    }
    near = {
        "positions": np.asarray([[3.5, 0.0, 0.0]]),
        "radii": np.asarray([1.7]),
        "owners": np.asarray([0]),
    }
    far = {
        "positions": np.asarray([[5.0, 0.0, 0.0]]),
        "radii": np.asarray([1.7]),
        "owners": np.asarray([0]),
    }
    assert DatasetDesign()._contacts(protein, near)["pair_count"] == 1
    assert DatasetDesign()._contacts(protein, far)["pair_count"] == 0
