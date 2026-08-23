"""Offline tests for leakage, phenotype, partition, and dilution invariants."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import gemmi
import numpy as np
import pytest
from lambdaforge.data import DatasetIndex, DatasetMember

from wisdom.preprocessing.dna.DNAPartitionTask import DNAPartitionTask
from wisdom.preprocessing.dna.DNAValidation import DNAValidation


def _partition_row(
    identifier : str,
    label      : int,
    group      : str,
    phenotype  : str,
) -> dict[str, object]:
    """Create one complete row for deterministic split and dilution tests.

    Args:
        identifier: Unique logical protein identifier.
        label: Binary global DNA-binding target.
        group: Indivisible sequence/structure leakage component.
        phenotype: Stable class-specific physical phenotype.

    Returns:
        Minimal row accepted by partition-only methods.
    """
    return {
        "identifier": identifier,
        "label": label,
        "local_gt_available": True,
        "leakage_group": group,
        "phenotype_cluster": phenotype,
        "source_dataset": "fixture",
    }


def test_similarity_pairs_use_declared_thresholds_and_transitive_components(
    tmp_path: Path,
) -> None:
    """Retain only bilateral sequence matches and close their graph transitively."""
    pairs = tmp_path / "sequence.tsv"
    pairs.write_text(
        "A\tB\t0.31\t0.90\t0.85\t1e-20\t100\n"
        "B\tC\t0.45\t0.90\t0.90\t1e-10\t80\n"
        "A\tC\t0.80\t0.79\t0.95\t1e-30\t120\n",
        encoding="utf-8",
    )
    task  = DNAPartitionTask(phenotype_min_cluster_size=2)
    edges = task._sequence_edges(pairs, {"A", "B", "C"})

    assert edges == {("A", "B"), ("B", "C")}
    assert task._components(["A", "B", "C"], edges) == [["A", "B", "C"]]


def test_foldseek_probability_is_not_confused_with_tm_score(tmp_path: Path) -> None:
    """Apply Foldseek's homology probability and e-value, not its descriptive TM-score."""
    pairs = tmp_path / "structure.tsv"
    pairs.write_text(
        "A.cif\tB.cif\t70\t1e-8\t0.20\t0.9\t0.9\n"
        "A.cif\tC.cif\t40\t1e-20\t0.95\t0.9\t0.9\n",
        encoding="utf-8",
    )
    edges = DNAPartitionTask()._structure_edges(pairs, {"A", "B", "C"})

    assert edges == {("A", "B")}


def test_foldseek_input_contains_only_the_selected_protein_chain(tmp_path: Path) -> None:
    """Prevent DNA or unrelated deposited chains from defining structural leakage edges."""
    source = Path(__file__).parent / "data" / "dna_complex.pdb"
    target = tmp_path / "selected.cif"

    DNAPartitionTask._selected_chain_structure(source, "A", target)
    structure = gemmi.read_structure(str(target))

    assert len(structure) == 1
    assert [chain.name for chain in structure[0]] == ["A"]


def test_evaluation_rejects_positive_without_local_ground_truth() -> None:
    """Keep global-only positives trainable but forbid them from validation and test."""
    task = DNAPartitionTask()
    rows = [
        {
            "identifier": "A",
            "label": 1,
            "local_gt_available": False,
            "leakage_group": "L00001",
            "split": "validation",
        },
        {
            "identifier": "B",
            "label": 0,
            "local_gt_available": True,
            "leakage_group": "L00002",
            "split": "test",
        },
    ]

    with pytest.raises(RuntimeError, match="local ground truth"):
        task._validate_partitions(rows, set())


def test_hdbscan_reports_noise_when_sample_support_is_insufficient() -> None:
    """Do not manufacture phenotype classes from fewer than two minimum-size groups."""
    task = DNAPartitionTask(phenotype_min_cluster_size=5, phenotype_min_samples=2)
    features = {
        f"P{index}": {"area": float(index), "curvature": float(index % 2)}
        for index in range(8)
    }

    result = task._phenotypes(features, "P")

    assert result["diagnostics"]["robust"] is False
    assert set(result["labels"].values()) == {"P_NOISE"}


def test_hdbscan_reports_only_parameter_stable_fixture_clusters() -> None:
    """Retain two well-separated fixture phenotypes when every nearby fit agrees."""
    features = {
        **{
            f"A{index}": {"compactness": index * 0.01, "curvature": index * 0.02}
            for index in range(20)
        },
        **{
            f"B{index}": {
                "compactness": 10.0 + index * 0.01,
                "curvature": 10.0 + index * 0.02,
            }
            for index in range(20)
        },
    }
    result = DNAPartitionTask(
        phenotype_min_cluster_size=5,
        phenotype_min_samples=2,
    )._phenotypes(features, "P")

    assert result["diagnostics"]["robust"] is True
    assert result["diagnostics"]["cluster_count"] == 2
    assert result["diagnostics"]["median_adjusted_rand"] == pytest.approx(1.0)
    assert result["diagnostics"]["noise_fraction"] == pytest.approx(0.0)


def test_balancing_reduces_only_the_majority_after_groups_and_phenotypes_exist() -> None:
    """Create an equal benchmark while retaining diverse majority groups deterministically."""
    task = DNAPartitionTask(seed=17)
    rows = [
        {
            "identifier": f"P{index}",
            "label": 1,
            "local_gt_available": True,
            "leakage_group": f"LP{index}",
            "phenotype_cluster": f"P{index % 2:03d}",
            "source_dataset": "positive",
        }
        for index in range(5)
    ] + [
        {
            "identifier": f"N{index}",
            "label": 0,
            "local_gt_available": True,
            "leakage_group": f"LN{index}",
            "phenotype_cluster": f"N{index:03d}",
            "source_dataset": "negative",
        }
        for index in range(3)
    ]

    balanced, audit = task._balance_population(rows)

    assert task._class_counts(balanced) == {"total": 6, "positive": 3, "negative": 3}
    assert audit["input_counts"] == {"total": 8, "positive": 5, "negative": 3}
    assert audit["omitted_count"] == 2
    assert {row["phenotype_cluster"] for row in balanced if row["label"] == 1} == {
        "P000",
        "P001",
    }


def test_local_surface_descriptors_use_nearest_residue_and_connected_regions() -> None:
    """Map local chemistry through sparse atom edges and retain isolated positive patches."""
    base = {
        "surface_positions": np.zeros((4, 3), dtype=np.float32),
        "surface_atom_edge_index": np.asarray(
            [[0, 0, 1, 2, 3], [0, 1, 1, 2, 3]], dtype=np.int32
        ),
        "surface_atom_distance": np.asarray([2.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float32),
        "residue_type_ids": np.asarray([1, 2, 3, 4], dtype=np.uint8),
        "residue_indices": np.asarray([0, 1, 2, 3], dtype=np.int32),
    }
    residue_types, residue_indices = DNAPartitionTask._surface_residues(base)
    hard = np.asarray([True, True, False, True])
    edges = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
    region_count, largest_fraction = DNAPartitionTask._positive_regions(
        hard, edges, np.ones(4)
    )

    assert residue_types.tolist() == [2, 2, 3, 4]
    assert residue_indices.tolist() == [1, 1, 2, 3]
    assert region_count == 2
    assert largest_fraction == pytest.approx(2.0 / 3.0)


def test_split_is_deterministic_and_preserves_supported_rare_phenotypes() -> None:
    """Place every phenotype backed by three independent groups in all three splits."""
    rows = [
        _partition_row(f"P{index}", 1, f"LP{index}", "P001")
        for index in range(3)
    ] + [
        _partition_row(f"N{index}", 0, f"LN{index}", "N001")
        for index in range(3)
    ]
    first, audit = DNAPartitionTask(seed=7)._assign_splits(rows)
    second, _ = DNAPartitionTask(seed=7)._assign_splits(list(reversed(rows)))

    assert first == second
    assert {first[f"LP{index}"] for index in range(3)} == {
        "train",
        "validation",
        "test",
    }
    assert audit["phenotype_feasibility"]["P001"]["representable_in_all_splits"] is True


def test_split_reports_when_a_phenotype_has_too_few_independent_groups() -> None:
    """Describe mathematical non-representability instead of fragmenting leakage groups."""
    rows = [
        _partition_row(f"P{index}", 1, f"LP{index}", "P001")
        for index in range(3)
    ] + [
        _partition_row(f"N{index}", 0, f"LN{index}", "N001")
        for index in range(3)
    ] + [
        _partition_row(f"R{index}", 1, f"LR{index}", "P_RARE")
        for index in range(2)
    ]
    assignments, audit = DNAPartitionTask(seed=17)._assign_splits(rows)
    final_rows = [dict(row, split=assignments[str(row["leakage_group"])]) for row in rows]

    DNAPartitionTask(seed=17)._validate_partitions(final_rows)
    rare = audit["phenotype_feasibility"]["P_RARE"]
    assert rare["leakage_group_count"] == 2
    assert rare["representable_in_all_splits"] is False


def test_dilutions_are_nested_balanced_and_training_only(tmp_path: Path) -> None:
    """Keep exact nested sizes while covering labels, phenotypes, and distinct train groups."""
    rows = [
        dict(_partition_row("P1", 1, "L1", "P001"), split="train"),
        dict(_partition_row("N1", 0, "L2", "N001"), split="train"),
        dict(_partition_row("P2", 1, "L3", "P002"), split="train"),
        dict(_partition_row("N2", 0, "L4", "N002"), split="train"),
        dict(_partition_row("P3", 1, "L5", "P001"), split="train"),
        dict(_partition_row("N3", 0, "L6", "N001"), split="train"),
        dict(_partition_row("PV", 1, "LV", "P001"), split="validation"),
        dict(_partition_row("NT", 0, "LT", "N001"), split="test"),
    ]
    report = DNAPartitionTask(dilution_sizes=(2, 4, 6), seed=27)._dilutions(
        rows, tmp_path
    )
    subsets = {
        size: set(
            (tmp_path / "canonical" / f"train-{size}.txt")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for size in (2, 4, 6)
    }

    assert subsets[2] < subsets[4] < subsets[6]
    assert not ({"PV", "NT"} & subsets[6])
    assert report["train-6"]["class_counts"] == {"total": 6, "positive": 3, "negative": 3}
    assert set(report["train-4"]["phenotype_counts"]) == {"P001", "P002", "N001", "N002"}


def test_read_only_validation_fails_on_an_injected_cross_split_pair(
    tmp_path   : Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Emit a report and fail the Work when retained MMseqs2 evidence crosses splits."""
    members = (
        DatasetMember(
            member_id="POS_A",
            partitions={"split": "train", "leakage_group": "L1", "phenotype": "P_NOISE"},
            targets={"dna_binding": 1, "local_ground_truth": True},
            assets={},
        ),
        DatasetMember(
            member_id="NEG_A",
            partitions={"split": "test", "leakage_group": "L2", "phenotype": "N_NOISE"},
            targets={"dna_binding": 0, "local_ground_truth": True},
            assets={},
        ),
    )
    DatasetIndex.write(tmp_path / "index.jsonl", members)
    with (tmp_path / "catalog.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("identifier", "label", "split", "leakage_group", "logical_protein_id"),
        )
        writer.writeheader()
        writer.writerow(
            {"identifier": "POS_A", "label": 1, "split": "train", "leakage_group": "L1"}
        )
        writer.writerow(
            {"identifier": "NEG_A", "label": 0, "split": "test", "leakage_group": "L2"}
        )
    clusters = tmp_path / "clusters"
    clusters.mkdir()
    for name in ("sequence-pairs.csv", "structure-pairs.csv", "exact-identity-pairs.csv"):
        (clusters / name).write_text("left,right\n", encoding="utf-8")
    (clusters / "sequence-pairs.tsv").write_text(
        "POS_A\tNEG_A\t0.75\t0.95\t0.95\t1e-20\t100\n", encoding="utf-8"
    )
    (clusters / "structure-pairs.tsv").write_text("", encoding="utf-8")
    dilutions = tmp_path / "dilutions" / "canonical"
    dilutions.mkdir(parents=True)
    (dilutions / "train-1.txt").write_text("POS_A\n", encoding="utf-8")
    (tmp_path / "partition-report.json").write_text(
        json.dumps(
            {
                "parameters": {
                    "sequence_identity": 0.30,
                    "sequence_coverage": 0.80,
                    "sequence_evalue": 1e-3,
                    "structure_probability": 0.50,
                    "structure_evalue": 1e-3,
                }
            }
        )
        + "\n",
        encoding="utf-8",
    )

    run_dir = tmp_path / "run"
    monkeypatch.setattr(DNAValidation, "_validate_member", lambda *args: None)
    monkeypatch.setattr(DNAValidation, "_plot", lambda *args: None)
    monkeypatch.setattr(DNAValidation, "_plot_phenotype_pca", lambda *args: None)

    payload = DNAValidation().audit(tmp_path, run_dir / "dna-validation")
    assert payload["verdict"] == "FAIL"
    report = json.loads(
        (run_dir / "dna-validation" / "dna-validation-report.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["verdict"] == "FAIL"
    assert report["pair_failures"]["sequence-pairs.csv"] == 0
    assert report["raw_pair_failures"]["mmseqs2"] == 1
