"""Measure optional candidate costs on existing NPZs or an explicitly bounded structure pilot."""

import argparse
import csv
import json
import resource
import tempfile
import time
import tracemalloc
from pathlib import Path

import numpy as np

from wisdom.preprocessing.common.structure.AtomicStructureBuilder import AtomicStructureBuilder
from wisdom.preprocessing.common.structure.ExtendedSurfaceAtom import ExtendedSurfaceAtom
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.ProteinReader import ProteinReader
from wisdom.preprocessing.common.structure.StructureResolver import StructureResolver
from wisdom.preprocessing.common.structure.SurfaceBuilder import SurfaceBuilder


def measure(arrays: dict[str, np.ndarray], identifier: str, maximum: int) -> list[dict]:
    """Measure sparse builds/serialization and dense-prefix conversion for six physical cutoffs.

    Args:
        arrays: Fixed atom coordinates, surface coordinates and normals for one protein.
        identifier: Human pilot identifier, not an inferred task label.
        maximum: Explicit persisted candidate cap to audit, normally 96.

    Returns:
        JSON records with candidate quantiles, clipping, compressed/raw bytes, wall times,
        Python/NumPy-tracked peak allocation, process cumulative peak RSS and an explicitly
        estimated float32 gather activation size at hidden width 128.
    """
    rows = []
    for radius in (6, 8, 10, 12, 14, 16):
        builder = ExtendedSurfaceAtom(radius, maximum)
        tracemalloc.start()
        started = time.perf_counter()
        values = builder.build(arrays)
        elapsed = time.perf_counter() - started
        _, tracked_peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        count = len(values["atom_index"])
        with tempfile.TemporaryDirectory(prefix="wisdom-candidate-benchmark-") as directory:
            path = Path(directory) / "csr.npz"
            started = time.perf_counter()
            np.savez_compressed(path, **values)
            write_seconds = time.perf_counter() - started
            started = time.perf_counter()
            with np.load(path, allow_pickle=False) as archive:
                reopened = {key: archive[key] for key in archive.files}
            load_seconds = time.perf_counter() - started
            started = time.perf_counter()
            dense = builder.dense(reopened, maximum)
            convert_seconds = time.perf_counter() - started
            started = time.perf_counter()
            builder.batch([reopened, reopened], [len(arrays["atom_positions"])] * 2)
            batch_seconds = time.perf_counter() - started
            dense_path = Path(directory) / "dense.npz"
            np.savez_compressed(dense_path, **dense)
            rows.append(
                {
                    "identifier": identifier,
                    "radius_angstrom": radius,
                    "maximum": maximum,
                    "atoms": len(arrays["atom_positions"]),
                    "points": len(arrays["surface_positions"]),
                    "coverage": builder.coverage(values["candidate_count"]),
                    "discarded": int(values["discarded_count"].sum()),
                    "stored_pairs": count,
                    "csr_raw_bytes": sum(value.nbytes for value in values.values()),
                    "csr_compressed_bytes": path.stat().st_size,
                    "dense_raw_bytes": sum(value.nbytes for value in dense.values()),
                    "dense_compressed_bytes": dense_path.stat().st_size,
                    "build_seconds": elapsed,
                    "write_seconds": write_seconds,
                    "load_seconds": load_seconds,
                    "dense_conversion_seconds": convert_seconds,
                    "csr_two_copy_cpu_batch_seconds": batch_seconds,
                    "tracked_peak_bytes": tracked_peak,
                    "process_cumulative_peak_rss_kib": resource.getrusage(
                        resource.RUSAGE_SELF
                    ).ru_maxrss,
                    "estimated_gather_float32_h128_bytes": count * 128 * 4,
                }
            )
    return rows


def main() -> None:
    """Run an explicit local diagnostic, never submit a Work or publish a dataset.

    Existing NPZ inputs measure their exact fixed surfaces. The optional design pilot chooses
    three retained chains near 300/2000/10000 heavy atoms, computes only their point geometry
    (no Laplacian, task targets, or publication), and reports its limited sampling scope.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archives", type=Path, nargs="*")
    parser.add_argument("--design", type=Path)
    parser.add_argument("--maximum", type=int, default=96)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {
        "scope": "local diagnostic; no model or dataset publication",
        "memory_note": "tracked peak is not total RAM; RSS is cumulative process high-water",
        "measurements": [],
    }
    for base in args.archives:
        with np.load(base, allow_pickle=False) as archive:
            arrays = {
                key: archive[key]
                for key in ("atom_positions", "surface_positions", "surface_normals")
            }
            operator_bytes = sum(
                archive[key].nbytes for key in archive.files if key.startswith("diffusion_")
            )
        for row in measure(arrays, base.stem, args.maximum):
            row.update(base_npz_bytes=base.stat().st_size, diffusion_array_bytes=operator_bytes)
            report["measurements"].append(row)
    if args.design:
        with (args.design / "catalog.csv").open() as stream:
            candidates = [
                row
                for row in csv.DictReader(stream)
                if row.get("selected") in ("True", "true", "1")
            ]
        selected = {
            min(candidates, key=lambda row: abs(int(float(row["heavy_atom_count"])) - n))[
                "identifier"
            ]
            for n in (300, 2000, 10000)
        }
        for identifier in sorted(selected):
            started = time.perf_counter()
            source = StructureResolver(args.design / "structures").resolve(identifier, args.design)
            protein, _ = ProteinReader(PreprocessConfig()).read(source)
            arrays = AtomicStructureBuilder(6).build(protein)
            surface, _ = SurfaceBuilder(1.0, 1.4, (2.5, 5.0)).build(
                arrays["atom_positions"], arrays["vdw_radii"]
            )
            arrays.update(surface)
            geometry_seconds = time.perf_counter() - started
            for row in measure(arrays, identifier, args.maximum):
                row.update(
                    pilot_geometry_seconds=geometry_seconds,
                    base_npz_bytes=None,
                    diffusion_array_bytes=None,
                    audit_label=int(
                        next(r["label"] for r in candidates if r["identifier"] == identifier)
                    ),
                )
                report["measurements"].append(row)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Measured {len(report['measurements'])} protein/radius combinations: {args.output}")


if __name__ == "__main__":
    main()
