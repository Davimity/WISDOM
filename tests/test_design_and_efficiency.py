from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from lambdaforge.data import DatasetRegistry
from lambdaforge.work import Work, WorkConfig, WorkRunner
from yaml import safe_load

from wisdom.preprocessing.Preprocessing import Preprocessing
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessPipeline import PreprocessPipeline
from wisdom.preprocessing.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource
from wisdom.preprocessing.structure.StructureCache import StructureCache


def test_one_class_per_source_file_and_reader_public_api() -> None:
    source_dir = Path(__file__).parents[1] / "src"
    entrypoints = {
        source_dir / "wisdom" / "Training.py",
        source_dir / "wisdom" / "preprocessing" / "DatasetDesign.py",
        source_dir / "wisdom" / "preprocessing" / "Preprocessing.py",
    }
    for path in source_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        functions = [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert len(classes) <= 1, f"{path.name} contains {len(classes)} classes"
        if functions:
            assert path in entrypoints, f"{path.name} contains unexpected free module functions"
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
    context = ProcessingWorkspace(
        run_dir,
        inputs={"protein_identifiers": manifest, "structures": pdb_path.parent},
        outputs={
            "processed": run_dir / "processed",
            "report": run_dir / "preprocessing-report.json",
        },
    )
    run_dir.mkdir(parents=True)
    source = tuple(ProteinSource().records(context))
    result = PreprocessPipeline(config).process(source[0], context)
    sink   = ProteinSink()
    sink.records = {result.key: dict(result.value)}
    sink.finalize(context)
    archive_path = context.run_dir / "processed" / f"{pdb_path.stem}.npz"
    with np.load(archive_path, allow_pickle=False) as archive:
        forbidden_fragments = ("embedding", "one_hot", "rbf", "relative_vector", "message")
        assert not any(
            fragment in name for name in archive.files for fragment in forbidden_fragments
        )
        assert all(archive[name].dtype != object for name in archive.files)
        atom_count = len(archive["atom_positions"])
        surface_count = len(archive["surface_positions"])
        assert archive["atom_edge_index"].dtype == np.int32
        assert archive["surface_edge_index"].dtype == np.int32
        assert archive["atom_edge_index"].shape[1] < atom_count * atom_count
        assert archive["surface_edge_index"].shape[1] < surface_count * surface_count // 4
        assert archive["atomic_numbers"].dtype == np.uint8
        assert archive["surface_positions"].dtype == np.float32


def test_lambdaforge_owns_preprocessing_workers_resources_and_resume() -> None:
    project_root = Path(__file__).parents[1]
    design_values = safe_load(
        (project_root / "experiments" / "dna_design.yaml").read_text(encoding="utf-8")
    )
    preprocess_values = safe_load(
        (project_root / "experiments" / "dna_preprocess.yaml").read_text(encoding="utf-8")
    )

    assert design_values["run"] == "wisdom.preprocessing.DatasetDesign.DatasetDesign"
    assert design_values["resources"] == {
        "cpu": 36,
        "memory": "96GiB",
        "storage": "100GiB",
        "time": "24h",
    }
    assert design_values["with"]["workers"] == 36
    assert design_values["with"]["output_directory"] == "data/dna/design"
    assert design_values["with"]["overwrite_output"] is True
    assert design_values["with"]["maximum_resolution"] == 4.0
    assert design_values["with"]["dilution_fractions"] == [1.0, 0.75, 0.5, 0.25, 0.1]
    assert [step["name"] for step in preprocess_values["steps"]] == ["design", "preprocess"]
    assert preprocess_values["steps"][0]["with"]["output_directory"] == "data/dna/design"
    preprocess_step = preprocess_values["steps"][1]
    assert preprocess_step["run"] == "wisdom.preprocessing.Preprocessing.Preprocessing"
    assert preprocess_step["resources"] == {
        "cpu": 36,
        "memory": "128GiB",
        "storage": "150GiB",
        "time": "24h",
    }
    assert preprocess_step["with"]["workers"] == 36
    assert preprocess_step["with"]["requests_per_second"] == 4.0
    assert preprocess_step["with"]["retries"] == 5
    assert preprocess_step["with"]["design"] == {"from": "design.dataset-design"}
    assert any(name == "map" for name, _ in inspect.getmembers(Work, inspect.isfunction))

    preprocessing_source = inspect.getsource(Preprocessing)
    structure_cache_source = inspect.getsource(StructureCache)
    assert "self.resume_map(" in preprocessing_source
    assert "self.cache.fetch(" in preprocessing_source
    assert "urllib" not in structure_cache_source
    assert "CrossProcessFileLock" not in structure_cache_source

    assert not any(
        hasattr(PreprocessConfig(), name)
        for name in ("workers", "processes_per_cpu", "resume", "fail_fast", "raw_dir")
    )


@pytest.mark.parametrize(
    "filename",
    ("wisdom_v1.yaml", "wisdom_v2.yaml"),
)
def test_training_catalog_resolves_before_every_dry_run(
    filename : str,
) -> None:
    project_root = Path(__file__).parents[1]
    try:
        record = DatasetRegistry().get("wisdom-dna@4")
    except KeyError:
        pytest.skip("wisdom-dna@4 is not published in the LambdaForge DatasetRegistry")
    local = next(
        (placement for placement in record.placements if placement.cluster == "local"),
        None,
    )
    if local is None:
        pytest.skip("wisdom-dna@4 has no local placement")
    expected = Path(local.root).resolve()
    assert expected.is_dir()
    config = WorkConfig.from_yaml(project_root / "experiments" / filename)
    plan   = WorkRunner().plan(config)

    # Every expanded LambdaForge 0.12 Run resolves the same exact DatasetVersion marker at launch.
    assert len(plan.levels[0]) == (120 if filename == "wisdom_v1.yaml" else 18)
    assert config.raw["objective"] == {
        "metric": "val_auprc",
        "mode": "max",
    }


@pytest.mark.parametrize("scales", [(), (2.5, 2.5), (2.5, 0.0), (2.5, -1.0)])
def test_curvature_scales_reject_empty_duplicate_or_nonpositive_values(
    scales: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="curvature_scales"):
        PreprocessConfig(curvature_scales=scales)
