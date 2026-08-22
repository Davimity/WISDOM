from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest
from lambdaforge.preprocessing import PreprocessingDebugService, PreprocessingTask
from lambdaforge.tasks import TaskContext

from wisdom.preprocessing.structure.DatasetValidator import DatasetValidator
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessPipeline import PreprocessPipeline
from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource
from wisdom.preprocessing.structure.ProteinVisualizer import ProteinVisualizer
from wisdom.preprocessing.structure.StorageManager import StorageManager


def _context(run_dir: Path, id_file: Path, config: PreprocessConfig) -> TaskContext:
    run_dir.mkdir(parents=True, exist_ok=True)
    return TaskContext(
        name="test",
        run_dir=run_dir,
        source_dir=id_file.parent,
        attempt_id="test-attempt",
        config_fingerprint=StorageManager(config).config_hash,
        resume=True,
        inputs=(
            {
                "name": "protein_identifiers",
                "path": str(id_file),
                "resolved_path": str(id_file),
                "sha256": "test-identifiers",
                "size_bytes": id_file.stat().st_size,
            },
            {
                "name": "local_structures",
                "path": str(id_file.parent),
                "resolved_path": str(id_file.parent),
                "sha256": "test-structures",
                "size_bytes": 0,
            },
        ),
        outputs={
            "downloads": "raw",
            "processed": "processed",
            "report": "preprocessing-report.json",
        },
    )


def _run(
    run_dir: Path,
    id_file: Path,
    *,
    workers: int,
    resolution: float = 1.2,
    chains: tuple[str, ...] = (),
) -> dict:
    config = PreprocessConfig(chains=chains, surface_resolution=resolution)
    task   = PreprocessingTask(
        source=ProteinSource(),
        transforms=(PreprocessPipeline(config, download=False),),
        sink=ProteinSink(),
        workers=workers,
        workload="cpu",
        on_error="skip",
        checkpoint_interval=1,
        dataset_name="test-proteins",
    )
    task.run(_context(run_dir, id_file, config))
    return json.loads((run_dir / "preprocessing-report.json").read_text(encoding="utf-8"))


def test_txt_to_valid_pickle_free_npz_roundtrip(tmp_path: Path, pdb_path: Path) -> None:
    source = tmp_path / "tiny.pdb"
    shutil.copyfile(pdb_path, source)
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"# comment\n\n{source}\n{source}\n", encoding="utf-8")
    outputs = _run(tmp_path / "run", id_file, workers=1, chains=("A",))
    assert {key: outputs[key] for key in ("total", "processed", "skipped", "failed")} == {
        "total": 1,
        "processed": 1,
        "skipped": 0,
        "failed": 0,
    }
    npz_path = tmp_path / "run" / "processed" / "tiny.npz"
    metadata = StorageManager(PreprocessConfig()).read_metadata(npz_path)
    assert metadata is not None
    storage = StorageManager(PreprocessConfig(**metadata["config"]))
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
        storage.validate(arrays)
        assert all(array.dtype != object for array in arrays.values())
        assert "atom_edge_index" in arrays and "surface_atom_edge_index" in arrays
    assert metadata["atom_count"] == 17
    assert metadata["selected_chains"] == ["A"]
    assert metadata["surface_point_count"] > 0

    dataset_artifact = json.loads(
        (tmp_path / "run" / "dataset-artifact.json").read_text(encoding="utf-8")
    )
    assert dataset_artifact["sample_count"] == 1
    assert dataset_artifact["name"] == "test-proteins"
    assert dataset_artifact["dataset_id"].startswith("sha256:")

    visualizer = ProteinVisualizer(["tiny"], max_surface_points=50)
    diagnostics = visualizer.visualize(npz_path, tmp_path / "tiny.html", "tiny")
    html = (tmp_path / "tiny.html").read_text(encoding="utf-8")
    assert diagnostics["status"] == "PASS"
    assert "surface_positions" in html
    assert "C-alpha backbone cartoon" in html


def test_lambdaforge_debugs_one_protein_without_calling_the_sink(
    tmp_path: Path,
    pdb_path: Path,
) -> None:
    source = tmp_path / "tiny.pdb"
    shutil.copyfile(pdb_path, source)
    identifiers = tmp_path / "proteins.txt"
    identifiers.write_text(f"{source}\n", encoding="utf-8")
    config = tmp_path / "debug.yaml"
    config.write_text(
        f"""name: wisdom-debug-test
inputs:
  protein_identifiers: {identifiers}
  local_structures: {tmp_path}
outputs:
  downloads: raw
  processed: processed
  report: preprocessing-report.json
task:
  target: lambdaforge.preprocessing.PreprocessingTask
  params:
    source:
      target: wisdom.preprocessing.structure.ProteinSource.ProteinSource
    transforms:
      - target: wisdom.preprocessing.structure.PreprocessPipeline.PreprocessPipeline
        params:
          download: false
          config:
            target: wisdom.preprocessing.structure.PreprocessConfig.PreprocessConfig
            params: {{surface_resolution: 1.2}}
    sink:
      target: wisdom.preprocessing.structure.ProteinSink.ProteinSink
    workers: 1
    workload: cpu
""",
        encoding="utf-8",
    )

    result = PreprocessingDebugService().debug(config, records=1)

    assert result.ok
    assert result.records[0]["source_key"] == str(source)
    assert len(result.records[0]["transform_stages"]) == 1
    assert not (tmp_path / "processed").exists()


def test_parallel_and_single_worker_arrays_are_equivalent(tmp_path: Path, pdb_path: Path) -> None:
    first_source = tmp_path / "tiny1.pdb"
    second_source = tmp_path / "tiny2.pdb"
    shutil.copyfile(pdb_path, first_source)
    shutil.copyfile(pdb_path, second_source)
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{first_source}\n{second_source}\n", encoding="utf-8")
    _run(tmp_path / "single", id_file, workers=1)
    _run(tmp_path / "parallel", id_file, workers=2)
    for name in ("tiny1.npz", "tiny2.npz"):
        with (
            np.load(tmp_path / "single" / "processed" / name, allow_pickle=False) as left,
            np.load(tmp_path / "parallel" / "processed" / name, allow_pickle=False) as right,
        ):
            assert left.files == right.files
            for key in left.files:
                if key == "metadata_json":
                    continue
                assert np.array_equal(left[key], right[key])


def test_resume_config_and_source_invalidation(tmp_path: Path, pdb_path: Path) -> None:
    source = tmp_path / "source.pdb"
    shutil.copyfile(pdb_path, source)
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{source}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    assert _run(run_dir, id_file, workers=1)["processed"] == 1
    assert _run(run_dir, id_file, workers=1)["skipped"] == 1
    assert _run(run_dir, id_file, workers=1, resolution=1.1)["processed"] == 1
    with source.open("a", encoding="utf-8") as handle:
        handle.write("REMARK source hash changed\n")
    assert _run(run_dir, id_file, workers=1, resolution=1.1)["processed"] == 1


def test_failure_is_reported_without_losing_successes(tmp_path: Path, pdb_path: Path) -> None:
    source = tmp_path / "tiny.pdb"
    shutil.copyfile(pdb_path, source)
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{tmp_path / 'missing.pdb'}\n{source}\n", encoding="utf-8")
    outputs = _run(tmp_path / "run", id_file, workers=1)
    assert outputs["processed"] == 1
    assert outputs["failed"] == 1
    report = json.loads((tmp_path / "run" / "preprocessing-report.json").read_text())
    failure = next(record for record in report["records"] if record["status"] == "failed")
    assert failure["error_type"] == "FileNotFoundError"
    assert "missing.pdb" in failure["message"]


def test_pipeline_rejects_a_dataset_with_no_usable_proteins(tmp_path: Path) -> None:
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{tmp_path / 'missing.pdb'}\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="no usable proteins"):
        _run(run_dir, id_file, workers=1)

    report = json.loads((run_dir / "preprocessing-report.json").read_text())
    assert report["processed"] == 0
    assert report["skipped"] == 0
    assert report["failed"] == 1


def test_dataset_validator_audits_coverage_arrays_metadata_and_report(
    tmp_path: Path, pdb_path: Path
) -> None:
    source = tmp_path / "tiny.pdb"
    shutil.copyfile(pdb_path, source)
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{source}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    _run(run_dir, id_file, workers=1)

    validator = DatasetValidator()
    report = validator.validate(
        run_dir / "processed",
        run_dir / "preprocessing-report.json",
        id_file,
    )

    assert report["status"] == "valid"
    assert report["summary"]["valid_proteins"] == 1
    assert report["summary"]["invalid_proteins"] == 0
    assert report["records"][0]["sha256"]
    assert "Status: VALID" in validator.format_summary(report)


def test_dataset_validator_reports_corrupt_geometry_clearly(
    tmp_path: Path, pdb_path: Path
) -> None:
    source = tmp_path / "tiny.pdb"
    shutil.copyfile(pdb_path, source)
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{source}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    _run(run_dir, id_file, workers=1)

    npz_path = run_dir / "processed" / "tiny.npz"
    with np.load(npz_path, allow_pickle=False) as archive:
        payload = {name: archive[name] for name in archive.files}
    payload["atom_edge_distance"][0] += 1.0
    np.savez_compressed(npz_path, **payload)

    validator = DatasetValidator()
    report = validator.validate(
        run_dir / "processed",
        run_dir / "preprocessing-report.json",
        id_file,
    )

    assert report["status"] == "invalid"
    assert report["summary"]["invalid_proteins"] == 1
    assert any(
        "distances disagree with positions" in error for error in report["records"][0]["errors"]
    )
    assert "tiny.pdb" in validator.format_summary(report)

    # Resume must repair corrupt arrays even when their metadata still looks compatible.
    assert _run(run_dir, id_file, workers=1)["processed"] == 1


def test_surface_diagnostics_reject_flying_points_normals_and_curvature(
    tmp_path: Path, pdb_path: Path
) -> None:
    source = tmp_path / "tiny.pdb"
    shutil.copyfile(pdb_path, source)
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{source}\n", encoding="utf-8")
    run_dir = tmp_path / "run"
    _run(run_dir, id_file, workers=1)

    npz_path = run_dir / "processed" / "tiny.npz"
    metadata = StorageManager(PreprocessConfig()).read_metadata(npz_path)
    assert metadata is not None
    storage = StorageManager(PreprocessConfig(**metadata["config"]))
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name].copy() for name in archive.files if name != "metadata_json"}

    diagnostics = storage.surface_diagnostics(arrays)
    assert abs(diagnostics["minimum_signed_gap"]) <= 0.03
    assert abs(diagnostics["maximum_signed_gap"]) <= 0.03
    assert diagnostics["minimum_normal_cosine"] >= 0.99

    flying = dict(arrays)
    flying["surface_positions"] = arrays["surface_positions"].copy()
    flying["surface_positions"][0] += 100.0
    with pytest.raises(ValueError, match="floats outside the expanded-sphere"):
        storage.surface_diagnostics(flying)

    reversed_normal = dict(arrays)
    reversed_normal["surface_normals"] = arrays["surface_normals"].copy()
    reversed_normal["surface_normals"][0] *= -1.0
    with pytest.raises(ValueError, match="outward molecular envelope"):
        storage.surface_diagnostics(reversed_normal)

    inconsistent_curvature = dict(arrays)
    inconsistent_curvature["surface_curvatures"] = arrays["surface_curvatures"].copy()
    inconsistent_curvature["surface_curvatures"][0, 0, 2] += 1.0
    with pytest.raises(ValueError, match="curvature channels violate"):
        storage.surface_diagnostics(inconsistent_curvature)
