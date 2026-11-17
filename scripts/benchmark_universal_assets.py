"""Offline voxel equivalence microbenchmark and optional per-stage single-structure profile."""

import argparse
import json
import platform
import tempfile
import time
from itertools import pairwise
from pathlib import Path

import numpy as np

from wisdom.preprocessing.common.structure.geometry.SurfaceBuilder import SurfaceBuilder
from wisdom.preprocessing.common.structure.PreprocessConfig import PreprocessConfig
from wisdom.preprocessing.common.structure.ProteinPreprocessor import ProteinPreprocessor


def reference_voxels(points: np.ndarray, owners: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Retain the former h=1 per-voxel implementation as an independent benchmark oracle.

    Args:
        points: Synthetic finite candidate positions [P,3], in Å.
        owners: Candidate-owner IDs [P].

    Returns:
        Selected points and owners using the original distance/first-index tie rule.
    """
    origin = points.min(axis=0)
    cells = np.floor(points - origin).astype(np.int64)
    order = np.lexsort((np.arange(len(points)), cells[:, 2], cells[:, 1], cells[:, 0]))
    ordered = cells[order]
    boundaries = np.r_[0, np.flatnonzero(np.any(np.diff(ordered, axis=0), axis=1)) + 1, len(order)]
    selected = []
    for start, stop in pairwise(boundaries):
        group = order[start:stop]
        center = origin + (cells[group[0]] + 0.5)
        distances = np.sum((points[group] - center) ** 2, axis=1)
        selected.append(group[np.argmin(distances)])
    return points[selected], owners[selected]


def benchmark(repeats: int, structure: Path | None = None) -> dict[str, object]:
    """Measure actual component timings without submitting a Work or occupying a GPU.

    Args:
        repeats: Positive repeated timing samples per synthetic candidate cloud.
        structure: Optional local coordinate fixture to profile once, not a dataset scan.

    Returns:
        Machine-readable samples/medians/dispersion and optional scientific transform stage times.
        Synthetic candidate clouds are not production proteins or end-to-end speedup evidence.
    """
    rng = np.random.default_rng(6)
    result: dict[str, object] = {
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scope": "CPU voxel microbenchmark; no dataset rebuild or training",
        "clouds": [],
    }
    rows = []
    for size in (1000, 10000, 100000):
        points = rng.normal(size=(size, 3)) * 8
        owners = np.arange(size, dtype=np.int32)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            expected = reference_voxels(points, owners)
            original = time.perf_counter() - started
            started = time.perf_counter()
            actual = SurfaceBuilder(1)._voxel_select(points, owners)
            optimized = time.perf_counter() - started
            if not all(np.array_equal(a, b) for a, b in zip(actual, expected, strict=True)):
                raise AssertionError("vectorized voxel selection changes the scientific result")
            samples.append([original, optimized])
        rows.append(
            {
                "candidates": size,
                "exact_equal": True,
                "samples_seconds": samples,
                "median_seconds": np.median(samples, axis=0).tolist(),
                "sample_sd_seconds": np.std(samples, axis=0, ddof=1).tolist()
                if repeats > 1
                else None,
            }
        )
    result["clouds"] = rows
    if structure is not None:
        with tempfile.TemporaryDirectory(prefix="wisdom-stage-profile-") as directory:
            path = structure.resolve()
            record = {"key": str(path), "identifier": str(path), "output_name": "profile.npz"}
            transformed = ProteinPreprocessor(
                PreprocessConfig(diffusion_spectral_modes_max=8)
            ).process(record, path, Path(directory), Path(directory))
            transformed["value"]["identifier"] = str(structure)
            result["single_structure_profile"] = transformed["value"]
    return result


def main() -> None:
    """Parse an explicit small CPU benchmark request and print/save its measurements.

    Raises:
        ValueError: Repetitions are not positive or a supplied structure cannot be processed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--structure", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.repeats < 1:
        raise ValueError("benchmark repeats must be positive")
    result = benchmark(args.repeats, args.structure)
    encoded = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded + "\n")
    print(encoded)


if __name__ == "__main__":
    main()
