from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from wisdom.preprocessing.ProcessingRecord import ProcessingRecord
from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace
from wisdom.preprocessing.structure.DatasetValidator import DatasetValidator
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.PreprocessPipeline import PreprocessPipeline
from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource
from wisdom.preprocessing.structure.ProteinVisualizer import ProteinVisualizer
from wisdom.preprocessing.structure.StorageManager import StorageManager


def _context(run_dir: Path, id_file: Path, config: PreprocessConfig) -> ProcessingWorkspace:
    """Create explicit paths for one local scientific integration fixture.

    Args:
        run_dir: Temporary root receiving coordinate, NPZ, and report outputs.
        id_file: Manifest containing the fixture structure paths.
        config: Scientific configuration retained for signature parity with callers.

    Returns:
        Path-only preprocessing workspace; LambdaForge runtime behavior is tested upstream.
    """
    del config
    run_dir.mkdir(parents=True, exist_ok=True)
    return ProcessingWorkspace(
        run_dir,
        inputs={"protein_identifiers": id_file},
        outputs={
            "downloads": run_dir / "raw",
            "processed": run_dir / "processed",
            "report": run_dir / "preprocessing-report.json",
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
    context  = _context(run_dir, id_file, config)
    records  = tuple(ProteinSource().records(context))
    pipeline = PreprocessPipeline(config, download=False)
    operation = partial(pipeline.process, context=context)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = tuple(pool.map(operation, records))

    sink = ProteinSink()
    sink.records = {
        record.key: dict(record.value)
        for value in results
        for record in (ProcessingRecord.restore(value),)
    }
    sink.finalize(context)
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

    visualizer = ProteinVisualizer(["tiny"], max_surface_points=50)
    diagnostics = visualizer.visualize(npz_path, tmp_path / "tiny.html", "tiny")
    html = (tmp_path / "tiny.html").read_text(encoding="utf-8")
    assert diagnostics["status"] == "PASS"
    assert "surface_positions" in html
    assert "C-alpha backbone cartoon" in html


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


def test_missing_geometry_input_fails_the_complete_dataset(tmp_path: Path, pdb_path: Path) -> None:
    source = tmp_path / "tiny.pdb"
    shutil.copyfile(pdb_path, source)
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{tmp_path / 'missing.pdb'}\n{source}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match=r"missing\.pdb"):
        _run(tmp_path / "run", id_file, workers=1)


def test_pipeline_rejects_a_dataset_with_no_usable_proteins(tmp_path: Path) -> None:
    id_file = tmp_path / "proteins.txt"
    id_file.write_text(f"{tmp_path / 'missing.pdb'}\n", encoding="utf-8")
    run_dir = tmp_path / "run"

    with pytest.raises(FileNotFoundError, match=r"missing\.pdb"):
        _run(run_dir, id_file, workers=1)


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
