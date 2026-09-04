from __future__ import annotations

import ast
import csv
import gzip
import hashlib
import inspect
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from lambdaforge.controlplane import NativeEnvironmentSpecification
from lambdaforge.data import DatasetRegistry
from lambdaforge.work import Work, WorkConfig, WorkRunner
from yaml import safe_load

from wisdom.preprocessing.dna.preprocessing.DatasetManifests import DatasetManifests
from wisdom.preprocessing.dna.preprocessing.geometry import _process_geometry, generate_geometry
from wisdom.preprocessing.dna.preprocessing.Preprocessing import Preprocessing
from wisdom.preprocessing.dna.preprocessing.structures import (
    _validate_structure,
    validate_structure_snapshot,
)
from wisdom.preprocessing.dna.selection.Selection import Selection
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.ProteinPreprocessor import ProteinPreprocessor
from wisdom.preprocessing.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource
from wisdom.preprocessing.structure.StructureResolver import StructureResolver
from wisdom.visualization.Visualization import Visualization


def test_project_declares_bootstrap_native_environment() -> None:
    """Managed bootstrap must install and verify both specialist command-line tools."""
    project_root  = Path(__file__).parents[1]
    specification = NativeEnvironmentSpecification.discover(project_root)

    assert specification is not None
    assert specification.manager == "conda"
    assert specification.source == project_root / "environment.yml"
    assert specification.required_executables == ("mmseqs", "foldseek")
    assert {"foldseek", "mmseqs2"}.issubset(specification.dependencies)


def test_structure_snapshot_validation_reports_exact_digest_mismatch(tmp_path: Path) -> None:
    """A changed snapshot reports both digests instead of consulting current RCSB bytes.

    Args:
        tmp_path: Isolated directory receiving the gzip validation candidate.
    """
    source = Path(__file__).parent / "data" / "tiny.pdb"
    archive = tmp_path / "1abc.pdb.gz"
    with source.open("rb") as input_stream, gzip.open(archive, "wb") as output_stream:
        output_stream.write(input_stream.read())

    observed = hashlib.sha256(source.read_bytes()).hexdigest()
    job = {
        "pdb_id":              "1abc",
        "file":                archive.name,
        "compressed_sha256":   hashlib.sha256(archive.read_bytes()).hexdigest(),
        "uncompressed_sha256": observed,
    }

    assert _validate_structure(job, tmp_path, False)["pdb_id"] == "1abc"
    job["uncompressed_sha256"] = "0" * 64
    with pytest.raises(ValueError, match=f"expected {'0' * 64}") as error:
        _validate_structure(job, tmp_path, False)

    assert "uncompressed structure snapshot changed for 1ABC" in str(error.value)
    assert f"observed {observed}" in str(error.value)


def test_selective_preprocessing_accepts_a_complete_structure_snapshot(
    tmp_path: Path,
) -> None:
    """A train dilution validates only its PDBs while retaining the complete design snapshot.

    Args:
        tmp_path: Isolated complete-snapshot fixture and index root.
    """

    class SnapshotWork:
        """Execute the bounded validation map synchronously for this focused contract test."""

        def __init__(self) -> None:
            """Create an empty collection of emitted progress messages."""
            self.messages: list[str] = []

        def log(self, message: str) -> None:
            """Retain one user-facing validation message.

            Args:
                message: Progress or selected-subset explanation.
            """
            self.messages.append(message)

        def resume_map(
            self,
            items   : list[dict[str, Any]],
            function: Any,
            **options: Any,
        ) -> list[dict[str, str]]:
            """Run selected validation jobs without emulating LambdaForge persistence.

            Args:
                items: Selected PDB validation records.
                function: Bound archive validator.
                options: LambdaForge map settings unused by the synchronous fixture.

            Returns:
                Validator results in selected PDB order.
            """
            del options
            return [function(item) for item in items]

    source  = Path(__file__).parent / "data" / "tiny.pdb"
    archive = tmp_path / "1abc.pdb.gz"
    with source.open("rb") as input_stream, gzip.open(archive, "wb") as output_stream:
        output_stream.write(input_stream.read())

    uncompressed_digest = hashlib.sha256(source.read_bytes()).hexdigest()
    index = {
        "schema_version": "1.0",
        "structures": [
            {
                "pdb_id":              "1abc",
                "file":                archive.name,
                "compressed_sha256":   hashlib.sha256(archive.read_bytes()).hexdigest(),
                "uncompressed_sha256": uncompressed_digest,
            },
            {
                "pdb_id":              "2xyz",
                "file":                "2xyz.cif.gz",
                "compressed_sha256":   "0" * 64,
                "uncompressed_sha256": "1" * 64,
            },
        ],
    }
    (tmp_path / "index.json").write_text(json.dumps(index), encoding="utf-8")
    work = SnapshotWork()

    result = validate_structure_snapshot(
        work,  # type: ignore[arg-type]
        ({"pdb_id": "1abc", "structure_sha256": uncompressed_digest},),
        tmp_path,
        workers=1,
        progress_log_seconds=60.0,
        verbose=False,
    )

    assert result == tmp_path
    assert any("1 unused PDB archives" in message for message in work.messages)


def test_one_class_per_source_file_and_reader_public_api() -> None:
    source_dir = Path(__file__).parents[1] / "src"
    entrypoints = {
        source_dir / "wisdom" / "Training.py",
        source_dir
        / "wisdom"
        / "preprocessing"
        / "dna"
        / "preprocessing"
        / "Preprocessing.py",
    }
    simple_stage_directories = {
        source_dir / "wisdom" / "preprocessing" / "dna" / "selection",
        source_dir / "wisdom" / "preprocessing" / "dna" / "preprocessing",
    }
    for path in source_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        functions = [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert len(classes) <= 1, f"{path.name} contains {len(classes)} classes"
        if functions:
            is_simple_pipeline_stage = path.parent in simple_stage_directories
            assert path in entrypoints or is_simple_pipeline_stage, (
                f"{path.name} contains unexpected free module functions"
            )
        if classes and path.name != "__init__.py":
            assert classes[0].name == path.stem
    public_methods = [
        name
        for name, value in inspect.getmembers(ProteinReader, inspect.isfunction)
        if not name.startswith("_")
    ]
    assert public_methods == ["read"]


def test_every_source_method_documents_each_parameter() -> None:
    source_dir = Path(__file__).parents[1] / "src"
    for path in source_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            docstring = ast.get_docstring(node)
            assert docstring, f"{path.name}:{node.lineno} {node.name} has no docstring"
            parameters = (
                *node.args.posonlyargs,
                *node.args.args,
                *node.args.kwonlyargs,
            )
            for parameter in parameters:
                if parameter.arg not in {"self", "cls"}:
                    assert parameter.arg in docstring, (
                        f"{path.name}:{node.lineno} {node.name} does not document "
                        f"parameter {parameter.arg}"
                    )


def test_bilingual_readmes_have_parallel_sections_and_scientific_detail() -> None:
    project_root = Path(__file__).parents[1]
    english = (project_root / "README.md").read_text(encoding="utf-8")
    spanish = (project_root / "README.es.md").read_text(encoding="utf-8")
    assert "[Español](README.es.md)" in english
    assert "[English](README.md)" in spanish

    for document in (english, spanish):
        assert "./install.sh" in document
        assert "conda activate wisdom" in document
        assert "python -m venv" not in document
        assert "\\[" not in document
        assert "\\]" not in document

    assert "Benchmark de unión a ADN" in spanish

    heading_pattern = re.compile(r"^(#{2,6}) (\d+(?:\.\d+)*)\. ", re.MULTILINE)
    english_headings = heading_pattern.findall(english)
    spanish_headings = heading_pattern.findall(spanish)
    assert len(english_headings) == len(spanish_headings)
    assert [number for _, number in english_headings] == [
        number for _, number in spanish_headings
    ]
    assert len(english_headings) == len(re.findall(r"^#{2,6} ", english, re.MULTILINE))
    assert len(spanish_headings) == len(re.findall(r"^#{2,6} ", spanish, re.MULTILINE))
    for marks, number in (*english_headings, *spanish_headings):
        assert len(number.split(".")) == len(marks) - 1
    assert [number for marks, number in english_headings if marks == "##"] == [
        "0",
        "1",
        "2",
        "3",
            "4",
            "5",
            "6",
        ]
    assert not re.search(r"^#{4,6} ", english, re.MULTILINE)
    assert not re.search(r"^#{4,6} ", spanish, re.MULTILINE)
    for number in (number for _, number in english_headings if number != "0"):
        assert f"[{number}. " in english
        assert f"[{number}. " in spanish

    numbered_title_pattern = re.compile(
        r"^#{2,6} ((\d+(?:\.\d+)*)\. .+)$", re.MULTILINE
    )
    for document in (english, spanish):
        toc_targets = set(re.findall(r"\]\(#([^)]+)\)", document))
        for title, number in numbered_title_pattern.findall(document):
            if number == "0":
                continue
            anchor = re.sub(r"[^\w\- ]", "", title.lower()).replace(" ", "-")
            assert anchor in toc_targets

    assert english.count("```math") == spanish.count("```math")
    assert english.count("| `") == spanish.count("| `")
    assert english.count("doi.org/") == spanish.count("doi.org/")
    for required_expression in (
        "g_i(\\mathbf{x})",
        "n_i = \\max",
        "surface_curvatures",
        "E_{atom}",
        "allow_pickle=False",
    ):
        assert required_expression in english
        assert required_expression in spanish


def test_dataset_splits_are_disjoint_and_cover_master_manifest() -> None:
    data_dir = Path(__file__).parents[1] / "data"
    required = tuple(data_dir / f"{name}.txt" for name in ("proteins", "train", "val", "test"))
    if not all(path.is_file() for path in required):
        pytest.skip("the generated production benchmark is not present in a fresh checkout")
    manifests = {
        name: tuple(
            line
            for raw_line in (data_dir / f"{name}.txt").read_text(encoding="utf-8").splitlines()
            if (line := raw_line.strip()) and not line.startswith("#")
        )
        for name in ("proteins", "train", "val", "test")
    }

    for name, identifiers in manifests.items():
        assert len(identifiers) == len(set(identifiers)), f"{name}.txt contains duplicates"

    train = set(manifests["train"])
    val   = set(manifests["val"])
    test  = set(manifests["test"])

    assert train.isdisjoint(val)
    assert train.isdisjoint(test)
    assert val.isdisjoint(test)
    assert train | val | test == set(manifests["proteins"])


def test_npz_schema_is_sparse_compact_and_has_no_learned_features(
    tmp_path: Path, pdb_path: Path
) -> None:
    config = PreprocessConfig(
        chains=["A"],
        surface_resolution=1.2,
    )
    manifest = tmp_path / "proteins.txt"
    manifest.write_text(f"{pdb_path}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True)
    output_root = run_dir / "processed"
    source      = tuple(ProteinSource().records(manifest))
    result      = ProteinPreprocessor(config).process(
        source[0],
        manifest,
        pdb_path.parent,
        output_root,
    )
    sink   = ProteinSink()
    sink.records = {str(result["key"]): dict(result["value"])}
    sink.finalize(manifest, output_root, run_dir / "preprocessing-report.json")
    archive_path = output_root / f"{pdb_path.stem}.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        forbidden_fragments = ("embedding", "one_hot", "rbf", "relative_vector", "message")
        assert not any(
            fragment in name for name in archive.files for fragment in forbidden_fragments
        )
        assert all(archive[name].dtype != object for name in archive.files)
        atom_count = len(archive["atom_positions"])
        assert archive["atom_edge_index"].dtype == np.int32
        assert archive["atom_edge_spatial_rank"].dtype == np.uint16
        assert archive["surface_atom_neighbors"].dtype == np.int32
        assert archive["diffusion_gradient_index"].dtype == np.int32
        assert archive["atom_edge_index"].shape[1] < atom_count * atom_count
        assert archive["surface_atom_neighbors"].shape[1] == config.surface_atom_k_max
        assert archive["surface_neighbors"].shape[1] == config.surface_neighbor_k_max
        assert "surface_edge_index" not in archive.files
        assert "surface_atom_edge_index" not in archive.files
        assert archive["atomic_numbers"].dtype == np.uint8
        assert archive["surface_positions"].dtype == np.float32
        assert {
            "atom_hybridization_ids",
            "atom_aromaticity",
            "atom_hbond_donor",
            "atom_hbond_acceptor",
            "residue_hydropathy",
            "residue_polarity",
        } <= set(archive.files)
        assert archive["atom_hybridization_ids"].shape == (atom_count,)
        assert archive["atom_aromaticity"].dtype == np.bool_
        assert archive["residue_hydropathy"].dtype == np.float32


def test_geometry_worker_returns_a_non_reusable_failure_record(tmp_path: Path) -> None:
    """A protein error must remain attributable without aborting sibling map records.

    Args:
        tmp_path: Isolated placeholder paths passed to the worker boundary.
    """

    class FailingPipeline:
        """Provide the smallest pipeline double that raises one scientific error."""

        def process(
            self,
            record        : object,
            manifest      : Path,
            structure_root: Path,
            output_root   : Path,
        ) -> object:
            """Raise the representative archive-ordering error.

            Args:
                record: Unused protein record.
                manifest: Unused manifest path.
                structure_root: Unused coordinate directory.
                output_root: Unused archive directory.

            Raises:
                ValueError: Always, to exercise the map failure boundary.
            """
            raise ValueError("surface atom neighbors must be sorted by distance then ID")

    result = _process_geometry(
        {"key": "1ABC_A"},
        FailingPipeline(),  # type: ignore[arg-type]
        tmp_path / "proteins.txt",
        tmp_path / "structures",
        tmp_path / "processed",
        False,
    )

    assert result["value"]["identifier"] == "1ABC_A"
    assert result["value"]["status"] == "failed"
    assert result["value"]["output"] is None
    assert result["value"]["error_type"] == "ValueError"
    assert "sorted by distance then ID" in result["value"]["error"]


def test_preprocessing_configuration_reuses_complete_design() -> None:
    project_root = Path(__file__).parents[1]
    values = safe_load(
        (project_root / "experiments" / "dna_preprocess.yaml").read_text(encoding="utf-8")
    )
    design_values, preprocess_values, visualization_values = values["steps"]

    assert tuple(inspect.signature(Selection.run).parameters)[1] == "skip"
    assert tuple(inspect.signature(Preprocessing.run).parameters)[1] == "skip"
    assert tuple(inspect.signature(Visualization.run).parameters)[1] == "skip"

    assert design_values["run"] == (
        "wisdom.preprocessing.dna.selection.Selection.Selection"
    )
    assert design_values["resources"] == {
        "cpu": 36,
        "memory": "120GiB",
        "storage": "100GiB",
        "time": "24h",
    }
    assert design_values["with"]["skip"] is True
    assert design_values["with"]["existing_design"] == {"file": "../data/dna/design"}
    assert design_values["with"]["raw_path"] is None
    assert design_values["with"]["output_directory"] == "../data/dna/design"
    assert design_values["with"]["overwrite_output"] is True
    assert design_values["with"]["maximum_resolution"] == 4.0
    assert design_values["with"]["dilution_fractions"] == [1.0, 0.75, 0.5, 0.25, 0.1]
    assert preprocess_values["run"] == (
        "wisdom.preprocessing.dna.preprocessing.Preprocessing.Preprocessing"
    )
    assert preprocess_values["resources"] == {
        "cpu": 36,
        "memory": "120GiB",
        "storage": "100GiB",
        "time": "24h",
    }
    assert preprocess_values["with"]["skip"] is False
    assert preprocess_values["with"]["workers"] == 36
    assert preprocess_values["with"]["progress_log_seconds"] == 120.0
    assert preprocess_values["with"]["train"] == {"from": "select.train"}
    assert preprocess_values["with"]["validation"] == {"from": "select.validation"}
    assert preprocess_values["with"]["test"] == {"from": "select.test"}
    assert preprocess_values["with"]["catalog"] == {"from": "select.catalog"}
    assert preprocess_values["with"]["dilutions"] == {"from": "select.dilutions"}
    assert preprocess_values["with"]["structures"] == {"from": "select.structures"}
    assert preprocess_values["with"]["dataset_name"] == "wisdom-dna-reduced"
    assert preprocess_values["with"]["dataset_version"] == "6"
    assert preprocess_values["with"]["include_full_train"] is False
    assert preprocess_values["with"]["train_dilutions"] == ["replicate-00/train-25"]
    assert preprocess_values["with"]["include_validation"] is True
    assert preprocess_values["with"]["include_test"] is False
    assert preprocess_values["with"]["curvature_scales"] == [1.5, 2.5, 5.0, 7.5, 10.0]
    assert visualization_values["run"] == (
        "wisdom.visualization.Visualization.Visualization"
    )
    assert visualization_values["resources"] == {
        "cpu": 1,
        "memory": "8GiB",
        "storage": "10GiB",
        "time": "2h",
    }
    assert visualization_values["with"]["skip"] is True
    assert visualization_values["with"]["dataset"] == {"from": "preprocess.dataset"}
    assert visualization_values["with"]["maximum_vdw_atoms"] == 1500
    assert any(name == "map" for name, _ in inspect.getmembers(Work, inspect.isfunction))

    preprocessing_source = inspect.getsource(Preprocessing)
    geometry_source = inspect.getsource(generate_geometry)
    structure_resolver_source = inspect.getsource(StructureResolver)
    assert "DatasetManifests(" in preprocessing_source
    assert "work.resume_map(" in geometry_source
    assert "work.cache.fetch(" not in preprocessing_source
    assert "urllib" not in structure_resolver_source
    assert "CrossProcessFileLock" not in structure_resolver_source

    assert not any(
        hasattr(PreprocessConfig(), name)
        for name in ("workers", "processes_per_cpu", "resume", "fail_fast", "raw_dir")
    )


def test_preprocessing_reads_only_three_self_contained_manifests(tmp_path: Path) -> None:
    """Preprocessing needs no complete Selection directory or derived structure path."""
    paths = {}
    for filename, identifier, label, split in (
        ("train.jsonl", "1ABC_A", 0, "train"),
        ("val.jsonl", "2ABC_B", 1, "validation"),
        ("test.jsonl", "3ABC_C", 0, "test"),
    ):
        row = {
            "identifier": identifier,
            "label": label,
            "split": split,
            "leakage_group": f"group-{identifier}",
            "global_phenotype": "G_NOISE",
            "interface_phenotype": "I_NOISE",
            "origin": "fixture",
            "label_evidence": "fixture",
            "pdb_id": identifier[:4],
            "protein_chain": identifier.split("_", 1)[1],
            "assembly_id": "1",
            "protein_copy": 1,
            "structure_sha256": "a" * 64,
            "dna_chains": [],
            "binding_residue_indices": [],
            "local_gt_expected": True,
            "local_gt_method": "global_negative" if label == 0 else "dna_distance",
            "assembly_rotation": np.eye(3).tolist(),
            "assembly_translation": [0.0, 0.0, 0.0],
            "dilutions": ["replicate-00/train-100"] if split == "train" else [],
        }
        path = tmp_path / filename
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        paths[filename] = path

    class Log:
        """Collect Work messages for the manifest reader fixture."""

        def log(self, message: str, level: str = "info") -> None:
            """Accept one fixture log line.

            Args:
                message: Human-readable manifest progress.
                level: LambdaForge-compatible severity.
            """

    rows = DatasetManifests(
        paths["train.jsonl"],
        paths["val.jsonl"],
        paths["test.jsonl"],
    ).load(Log())

    assert [row["identifier"] for row in rows] == ["1ABC_A", "2ABC_B", "3ABC_C"]
    assert all("structure_path" not in row for row in rows)


def test_preprocessing_joins_existing_labelled_views_to_catalog(tmp_path: Path) -> None:
    """Existing two-column split files retain the catalog's scientific identity fields."""
    catalog = tmp_path / "catalog.csv"
    fields  = sorted(DatasetManifests.REQUIRED_FIELDS)
    rows    = []
    for identifier, label, split in (
        ("1ABC_A", 0, "train"),
        ("2ABC_B", 1, "validation"),
        ("3ABC_C", 0, "test"),
    ):
        row = {
            field: "fixture"
            for field in fields
        }
        row.update(
            {
                "identifier":              identifier,
                "label":                   label,
                "split":                   split,
                "protein_copy":            1,
                "local_gt_expected":       True,
                "dna_chains":              json.dumps([]),
                "binding_residue_indices": json.dumps([]),
                "assembly_rotation":       json.dumps(np.eye(3).tolist()),
                "assembly_translation":    json.dumps([0.0, 0.0, 0.0]),
            }
        )
        rows.append(row)

    with catalog.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    paths = {}
    for filename, identifier, label in (
        ("train-labelled.txt", "1ABC_A", 0),
        ("validation-labelled.txt", "2ABC_B", 1),
        ("test-labelled.txt", "3ABC_C", 0),
    ):
        path = tmp_path / filename
        path.write_text(f"{identifier}\t{label}\n", encoding="utf-8")
        paths[filename] = path

    class Log:
        """Accept manifest reader messages for the compatibility fixture."""

        def log(self, message: str, level: str = "info") -> None:
            """Accept one fixture message.

            Args:
                message: Human-readable manifest progress.
                level: LambdaForge-compatible severity.
            """

    loaded = DatasetManifests(
        paths["train-labelled.txt"],
        paths["validation-labelled.txt"],
        paths["test-labelled.txt"],
        catalog,
    ).load(Log())

    assert [row["identifier"] for row in loaded] == ["1ABC_A", "2ABC_B", "3ABC_C"]
    assert loaded[1]["assembly_rotation"] == np.eye(3).tolist()


def test_preprocessing_selects_train_dilution_before_geometry(tmp_path: Path) -> None:
    """Train25 plus validation excludes larger-only train members and every test member."""
    def record(identifier: str, label: int, split: str, views: list[str]) -> dict[str, Any]:
        """Create one complete manifest row for selective-loading verification.

        Args:
            identifier: Unique synthetic PDB-chain identifier.
            label: Binary protein target.
            split: Canonical supervised split.
            views: Nested training-view memberships.

        Returns:
            Complete JSON-compatible preprocessing record.
        """
        return {
            "identifier": identifier,
            "label": label,
            "split": split,
            "leakage_group": f"group-{identifier}",
            "global_phenotype": "G_NOISE",
            "interface_phenotype": "I_NOISE",
            "origin": "fixture",
            "label_evidence": "fixture",
            "pdb_id": identifier[:4],
            "protein_chain": identifier.split("_", 1)[1],
            "assembly_id": "1",
            "protein_copy": 1,
            "structure_sha256": "a" * 64,
            "dna_chains": [],
            "binding_residue_indices": [],
            "local_gt_expected": True,
            "local_gt_method": "fixture",
            "assembly_rotation": np.eye(3).tolist(),
            "assembly_translation": [0.0, 0.0, 0.0],
            "dilutions": views,
        }

    split_rows = {
        "train.jsonl": [
            record("1AAA_A", 0, "train", ["replicate-00/train-25"]),
            record("2AAA_A", 1, "train", ["replicate-00/train-25"]),
            record("3AAA_A", 0, "train", ["replicate-00/train-50"]),
            record("4AAA_A", 1, "train", []),
        ],
        "val.jsonl": [record("5AAA_A", 1, "validation", [])],
        "test.jsonl": [record("6AAA_A", 0, "test", [])],
    }
    paths: dict[str, Path] = {}
    for name, records in split_rows.items():
        path = tmp_path / name
        path.write_text(
            "".join(json.dumps(value) + "\n" for value in records),
            encoding="utf-8",
        )
        paths[name] = path

    class Log:
        """Accept selective-loader progress messages."""

        def log(self, message: str, level: str = "info") -> None:
            """Accept one fixture message.

            Args:
                message: Human-readable progress line.
                level: LambdaForge-compatible severity.
            """

    selected = DatasetManifests(
        paths["train.jsonl"],
        paths["val.jsonl"],
        paths["test.jsonl"],
    ).load(
        Log(),
        include_full_train=False,
        train_dilutions=("replicate-00/train-25",),
        include_validation=True,
        include_test=False,
    )

    assert {row["identifier"] for row in selected} == {"1AAA_A", "2AAA_A", "5AAA_A"}
    assert all(row["identifier"] != "6AAA_A" for row in selected)


@pytest.mark.parametrize(
    "filename",
    ("wisdom_v1.yaml", "wisdom_v2.yaml", "wisdom_v3.yaml"),
)
def test_training_catalog_resolves_before_every_dry_run(
    filename : str,
) -> None:
    project_root = Path(__file__).parents[1]
    try:
        record = DatasetRegistry().get("wisdom-dna-reduced@6")
    except KeyError:
        pytest.skip("wisdom-dna-reduced@6 is not published in the LambdaForge DatasetRegistry")
    local = next(
        (placement for placement in record.placements if placement.cluster == "local"),
        None,
    )
    if local is None:
        pytest.skip("wisdom-dna-reduced@6 has no local placement")
    expected = Path(local.root).resolve()
    assert expected.is_dir()
    config = WorkConfig.from_yaml(project_root / "experiments" / filename)
    plan   = WorkRunner().plan(config)

    # Every expanded LambdaForge 0.14 Run resolves the same exact DatasetVersion marker at launch.
    expected_runs = {
        "wisdom_v1.yaml": 1003,
        "wisdom_v2.yaml": 63,
        "wisdom_v3.yaml": 53,
    }
    assert len(plan.levels[0]) == expected_runs[filename]
    assert config.raw["objective"] == {
        "metrics": {
            "val_auprc": {
                "mode": "max",
                "weight": 0.35,
                "range": [0.0, 1.0],
            },
            "val_auroc": {
                "mode": "max",
                "weight": 0.20,
                "range": [0.0, 1.0],
            },
            "val_balanced_accuracy": {
                "mode": "max",
                "weight": 0.25,
                "range": [0.0, 1.0],
            },
            "val_mcc_objective": {
                "mode": "max",
                "weight": 0.20,
                "range": [0.0, 1.0],
            },
        },
        "aggregation": "geometric",
    }


@pytest.mark.parametrize(
    "scales",
    [(), (2.5, 2.5), (2.5, 0.0), (2.5, -1.0), (5.0, 2.5)],
)
def test_curvature_scales_reject_nonpositive_duplicate_or_unordered_values(
    scales: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="curvature_scales"):
        PreprocessConfig(curvature_scales=scales)
