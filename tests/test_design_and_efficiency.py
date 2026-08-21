from __future__ import annotations

import ast
import inspect
import re
from pathlib import Path

import numpy as np
import pytest
from lambdaforge import Experiment
from lambdaforge.data import DatasetRecipeConfig, DatasetRegistry
from lambdaforge.preprocessing import PreprocessingTask
from lambdaforge.tasks import TaskConfig, TaskContext

from preprocess.PreprocessConfig import PreprocessConfig
from preprocess.PreprocessPipeline import PreprocessPipeline
from preprocess.ProteinReader import ProteinReader
from preprocess.ProteinSink import ProteinSink
from preprocess.ProteinSource import ProteinSource


def test_one_class_per_source_file_and_reader_public_api() -> None:
    source_dir = Path(__file__).parents[1] / "src"
    for path in source_dir.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
        functions = [
            node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]
        assert len(classes) <= 1, f"{path.name} contains {len(classes)} classes"
        assert not functions, f"{path.name} contains free module functions"
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
    context = TaskContext(
        name="schema-test",
        run_dir=tmp_path / "run",
        source_dir=tmp_path,
        attempt_id="schema-test-attempt",
        config_fingerprint="schema-test-fingerprint",
        resume=True,
        inputs=(
            {
                "name": "protein_identifiers",
                "path": str(manifest),
                "resolved_path": str(manifest),
                "sha256": "schema-manifest",
                "size_bytes": manifest.stat().st_size,
            },
            {
                "name": "local_structures",
                "path": str(pdb_path.parent),
                "resolved_path": str(pdb_path.parent),
                "sha256": "schema-structures",
                "size_bytes": 0,
            },
        ),
        outputs={
            "downloads": "raw",
            "processed": "processed",
            "report": "preprocessing-report.json",
        },
    )
    context.run_dir.mkdir(parents=True)
    PreprocessingTask(
        source=ProteinSource(),
        transforms=(PreprocessPipeline(config, download=False),),
        sink=ProteinSink(),
        workers=1,
        workload="cpu",
        on_error="skip",
    ).run(context)
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
    selection    = TaskConfig.from_yaml(project_root / "experiments" / "dna_select.yaml")
    recipe       = DatasetRecipeConfig.from_yaml(
        project_root / "experiments" / "dna_preprocess.yaml"
    )
    stage_configs = {
        stage.name: TaskConfig(
            stage.task,
            source=project_root / "experiments" / f".embedded-{stage.name}.yaml",
        )
        for stage in recipe.stages
        if not isinstance(stage.task, Path)
    }
    geometry     = next(stage for stage in recipe.stages if stage.name == "geometry")
    assert not isinstance(geometry.task, Path)
    config       = stage_configs["geometry"]
    task_params  = config["task"]["params"]

    assert recipe.selector == "wisdom-dna@2"
    assert selection.resources.cpu_cores == 36
    assert selection.resources.ram_bytes == 64 * 1024**3
    assert selection.resources.gpu_count == 0
    assert selection.resources.runtime_seconds == 24 * 60 * 60
    selection_params = selection["task"]["params"]
    assert selection_params["workers"] == 72
    assert selection_params["workload"] == "io"
    assert selection_params["sink"]["params"]["dilutions"] == [0.10, 0.25, 0.50, 0.75]
    assert recipe.resource_override == {
        "cpus": 36,
        "memory": "128GiB",
        "gpus": 0,
        "time": "24h",
    }
    assert [stage.name for stage in recipe.stages] == ["geometry", "annotate"]
    assert {
        name: stage_config["task"]["params"]["workers"]
        for name, stage_config in stage_configs.items()
    } == {"geometry": 36, "annotate": 36}
    assert config.resume and not config.rerun_completed
    assert task_params["workers"] == 36
    assert task_params["workload"] == "cpu"
    assert task_params["on_error"] == "fail"
    assert "progress_interval_seconds" not in task_params
    assert inspect.signature(PreprocessingTask).parameters[
        "progress_interval_seconds"
    ].default == 10.0
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
        record = DatasetRegistry().get("wisdom-dna@2")
    except KeyError:
        pytest.skip("wisdom-dna@2 is not published in the LambdaForge DatasetRegistry")
    local = next(
        (placement for placement in record.placements if placement.cluster == "local"),
        None,
    )
    if local is None:
        pytest.skip("wisdom-dna@2 has no local placement")
    experiment = Experiment.from_yaml(project_root / "experiments" / filename)
    expected   = (Path(local.root) / "manifest.csv").resolve()

    # Every expanded seed/ablation must receive the same logical dataset at its real local mount.
    expanded = experiment.expand()
    assert len(expanded) == 3
    for run in expanded:
        manifest = Path(run["data"]["train"]["params"]["manifest"])
        assert manifest == expected

    # Exercise the runner's second configuration materialization without launching optimization.
    results = experiment.run(dry_run=True)
    plan = results.values
    assert plan["mode"] == "adaptive_hpo"
    assert plan["objective"]["metric"] == "val_auprc"


@pytest.mark.parametrize("scales", [(), (2.5, 2.5), (2.5, 0.0), (2.5, -1.0)])
def test_curvature_scales_reject_empty_duplicate_or_nonpositive_values(
    scales: tuple[float, ...],
) -> None:
    with pytest.raises(ValueError, match="curvature_scales"):
        PreprocessConfig(curvature_scales=scales)
