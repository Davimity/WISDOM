"""Pickle-free storage shared by independent, base-bound structural assets."""

import json
import numpy as np

from typing import Any
from pathlib import Path
from collections.abc import Mapping
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive


class StructuralAsset:
    """Persist scientific arrays without changing the universal archive or owning a cache."""

    @staticmethod
    def write(
        output: Path, base: Path, arrays: Mapping[str, np.ndarray],
        family: str, parameters: Mapping[str, Any], metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Write one new-family schema-1 asset to an LF-owned temporary file.

        Args:
            output: Native checkpoint/output temporary destination.
            base: Exact immutable schema-3 NPZ defining every physical index.
            arrays: Named numeric, boolean or Unicode arrays; object/pickle data are forbidden.
            family: Public asset name identifying its independent scientific contract.
            parameters: Scientific settings only, excluding workers and downstream model choices.
            metadata: Algorithm, units, conventions, evidence and coverage descriptions.

        Raises:
            ValueError: Non-finite numeric fields or object arrays would be persisted.
        """
        for name, value in arrays.items():
            if value.dtype.kind == "O" or (value.dtype.kind in "fc" and not
                    np.isfinite(value).all()):
                raise ValueError(f"invalid scientific asset array: {name}")
        header = {**dict(metadata or {}), "family": family, "schema": "1.0",
                  "parameters": dict(parameters),
                  "arrays": {key: {"shape": list(value.shape), "dtype": value.dtype.str}
                             for key, value in arrays.items()},
                  "boolean_counts": {key: {"true": int(value.sum()), "total": int(value.size)}
                                     for key, value in arrays.items() if value.dtype.kind == "b"}}
        with output.open("wb") as stream:
            np.savez_compressed(stream, **arrays,  # type: ignore[arg-type]
                base_npz_sha256=np.asarray(ProteinArchive.sha256(base)),
                metadata_json=np.asarray(json.dumps(header, sort_keys=True)))

    @staticmethod
    def read(
        path: Path, base: Path, family: str, verify_hash: bool = True,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Verify binding and storage schema before returning detached arrays and metadata.

        Args:
            path: Asset created by this contract; partially written ZIP files fail naturally.
            base: Immutable structural NPZ whose physical index spaces are authoritative.
            family: Required public asset family, never inferred from a filename.
            verify_hash: Check exact base bytes at publication/resume; immutable managed loaders
                may skip this repeated checksum after dataset integrity verification.

        Returns:
            Arrays and schema metadata. Scientific producers additionally validate their own
            correspondence, equations and masks; a valid ZIP alone is not scientific validation.

        Raises:
            ValueError: Wrong family/schema/binding, shape, dtype or non-finite values.
        """
        with np.load(path, allow_pickle=False) as stored:
            header = json.loads(str(stored["metadata_json"].item()))
            digest = str(stored["base_npz_sha256"].item())
            arrays = {key: stored[key] for key in stored.files
                      if key not in {"metadata_json", "base_npz_sha256"}}
        if header.get("family") != family or header.get("schema") != "1.0":
            raise ValueError(f"unsupported {family} asset family/schema")
        if verify_hash and digest != ProteinArchive.sha256(base):
            raise ValueError(f"{family} base NPZ hash mismatch")
        signature = {key: {"shape": list(value.shape), "dtype": value.dtype.str}
                     for key, value in arrays.items()}
        if signature != header.get("arrays"):
            raise ValueError(f"{family} storage schema mismatch")
        counts = {key: {"true": int(value.sum()), "total": int(value.size)}
                  for key, value in arrays.items() if value.dtype.kind == "b"}
        if header.get("boolean_counts") != counts:
            raise ValueError(f"{family} mask coverage report mismatch")
        if any(value.dtype.kind == "O" or (value.dtype.kind in "fc" and not
               np.isfinite(value).all()) for value in arrays.values()):
            raise ValueError(f"{family} contains invalid numerical/object data")
        return arrays, header

    @staticmethod
    def compare(actual: Mapping[str, np.ndarray], expected: Mapping[str, np.ndarray]) -> None:
        """Require exact deterministic reconstruction of a fixed scientific asset.

        Args:
            actual: Stored detached arrays.
            expected: Arrays reconstructed from immutable inputs and declared parameters.

        Raises:
            ValueError: Missing/extra fields, wrong dtypes or unequal values.
        """
        if actual.keys() != expected.keys() or any(
            actual[key].dtype != value.dtype or not np.array_equal(actual[key], value)
            for key, value in expected.items()
        ):
            raise ValueError("structural asset does not match scientific reconstruction")
