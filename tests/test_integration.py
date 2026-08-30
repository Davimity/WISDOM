from __future__ import annotations

import hashlib
import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np
import pytest

from wisdom.preprocessing.structure.DatasetValidator import DatasetValidator
from wisdom.preprocessing.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.structure.ProteinPreprocessor import ProteinPreprocessor
from wisdom.preprocessing.structure.ProteinSink import ProteinSink
from wisdom.preprocessing.structure.ProteinSource import ProteinSource
from wisdom.preprocessing.structure.ProteinVisualizer import ProteinVisualizer


def _run(
    run_dir: Path,
    id_file: Path,
    *,
    workers: int,
    resolution: float = 1.2,
    chains: tuple[str, ...] = (),
) -> dict:
    run_dir.mkdir(parents=True, exist_ok=True)

    config      = PreprocessConfig(chains=chains, surface_resolution=resolution)
    records     = tuple(ProteinSource().records(id_file))
    pipeline    = ProteinPreprocessor(config)
    output_root = run_dir / "processed"
    operation   = partial(
        pipeline.process,
        manifest       = id_file,
        structure_root = id_file.parent,
        output_root    = output_root,
    )
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = tuple(pool.map(operation, records))

    sink = ProteinSink()
    sink.records = {
        str(value["key"]): dict(value["value"])
        for value in results
    }
    sink.finalize(id_file, output_root, run_dir / "preprocessing-report.json")
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
    metadata = ProteinArchive(PreprocessConfig()).read_metadata(npz_path)
    assert metadata is not None
    storage = ProteinArchive(PreprocessConfig(**metadata["config"]))
    with np.load(npz_path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files if name != "metadata_json"}
        storage.validate(arrays)
        assert all(array.dtype != object for array in arrays.values())
        assert "atom_edge_index" in arrays and "surface_atom_neighbors" in arrays
        assert "surface_edge_index" not in arrays
        assert "surface_atom_edge_index" not in arrays
        assert "diffusion_eigenvectors" in arrays
    assert metadata["atom_count"] == 17
    assert metadata["selected_chains"] == ["A"]
    assert metadata["surface_point_count"] > 0

    visualizer = ProteinVisualizer(max_surface_points=50)
    diagnostics = visualizer.visualize(
        npz_path,
        tmp_path / "tiny.html",
        "tiny",
        plotly_script="../plotly.min.js",
    )
    html = (tmp_path / "tiny.html").read_text(encoding="utf-8")
    assert diagnostics["status"] == "PASS"
    assert "surface_positions" in html
    assert "C-alpha backbone trace" in html
    assert "Van der Waals envelopes" in html
    assert "Distance measurement" in html
    assert 'id="controls-panel"' in html
    assert 'id="details-panel"' in html
    assert "alphahull" not in html
    assert "Plotly.restyle(gd,{visible:values},indices)" in html

    base_arrays, base_metadata, empty_sidecar = visualizer._load(npz_path, None)
    figure, controls = visualizer._figure(
        base_arrays,
        visualizer._surface_channels(base_arrays, base_metadata, empty_sidecar),
    )
    expected_layers = {
        "surface", "mesh", "atoms", "vdw", "spatial", "covalent", "both",
        "surface_edges", "normals", "cartoon", "measurement",
    }
    assert set(controls["traces"]) == expected_layers
    assert sorted(controls["traces"].values()) == list(range(len(figure.data)))
    assert controls["meshTriangles"] > 0
    assert controls["meshMethod"] in {"alpha-complex", "convex-hull fallback"}
    assert controls["meshVertexCount"] > 0
    assert controls["vdwAtoms"] == controls["atomCount"]

    surface_trace = figure.data[controls["traces"]["surface"]]
    mesh_trace    = figure.data[controls["traces"]["mesh"]]
    assert surface_trace.marker.opacity == 1.0
    assert mesh_trace.opacity == 1.0
    assert mesh_trace.colorscale[0][1] == "#59c3d1"

    # DNA sidecar channels retain exact point order and preserve unavailable distances as null in
    # strict browser JSON rather than emitting non-standard NaN tokens.

    with np.load(npz_path, allow_pickle=False) as archive:
        surface_count = len(archive["surface_positions"])
    distance = np.linspace(0.5, 5.0, surface_count, dtype=np.float32)
    distance[0] = np.nan
    distance_valid = np.ones(surface_count, dtype=np.bool_)
    distance_valid[0] = False
    hard = (np.arange(surface_count) % 2).astype(np.uint8)
    sidecar = tmp_path / "tiny.dna.npz"
    np.savez_compressed(
        sidecar,
        surface_target_hard             = hard,
        surface_valid_mask              = np.ones(surface_count, dtype=np.bool_),
        surface_target_soft             = hard.astype(np.float32),
        surface_distance_to_dna         = distance,
        surface_distance_valid          = distance_valid,
        surface_target_hard_sensitivity=np.column_stack((hard, hard)),
        sensitivity_gaps                = np.asarray([1.0, 1.4], dtype=np.float32),
        base_npz_sha256                 = np.asarray(
            hashlib.sha256(npz_path.read_bytes()).hexdigest()
        ),
        annotation_metadata_json        = np.asarray('{"local_gt_available":true}'),
    )

    channels = visualizer.surface_channels(npz_path, sidecar)
    assert {"dna_target_hard", "dna_target_soft", "dna_distance"}.issubset(channels)
    visualizer.visualize(
        npz_path,
        tmp_path / "tiny-annotated.html",
        "tiny",
        annotation=sidecar,
        protein_label=1,
        partitions={"split": "test"},
        plotly_script="../plotly.min.js",
    )
    annotated_html = (tmp_path / "tiny-annotated.html").read_text(encoding="utf-8")
    assert "dna_target_hard" in annotated_html
    assert "dna_distance" in annotated_html


def test_visual_mesh_and_van_der_waals_geometry_are_precomputed() -> None:
    """The browser receives explicit triangles and physically scaled atom envelopes."""
    cube = np.asarray(
        [
            (x, y, z)
            for x in (0.0, 1.0)
            for y in (0.0, 1.0)
            for z in (0.0, 1.0)
        ],
        dtype=np.float64,
    )
    outward       = cube - np.mean(cube, axis=0)
    visualizer    = ProteinVisualizer(mesh_alpha=4.0, max_vdw_atoms=1)
    faces, method = visualizer._alpha_faces(cube, outward)

    assert method == "alpha-complex"
    assert faces.ndim == 2 and faces.shape[1] == 3
    assert np.all((faces >= 0) & (faces < len(cube)))

    triangles     = cube[faces]
    face_normals  = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    mean_outwards = np.mean(outward[faces], axis=1)
    assert np.all(np.einsum("ij,ij->i", face_normals, mean_outwards) >= 0.0)

    atoms     = np.asarray(((0.0, 0.0, 0.0), (5.0, 0.0, 0.0)))
    radii     = np.asarray((1.5, 2.0))
    positions, sphere_faces, atom_ids = visualizer._van_der_waals_mesh(atoms, radii)

    assert positions.shape == (12, 3)
    assert sphere_faces.shape == (20, 3)
    assert np.array_equal(atom_ids, np.zeros(12, dtype=np.int64))
    assert np.allclose(np.linalg.norm(positions, axis=1), 1.5)


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
    metadata = ProteinArchive(PreprocessConfig()).read_metadata(npz_path)
    assert metadata is not None
    storage = ProteinArchive(PreprocessConfig(**metadata["config"]))
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
