"""Standard multichannel PLY/NPZ export for evaluated protein surfaces."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from uuid import uuid4

import numpy as np


class PointCloudExporter:
    """Write aligned coordinates and scalar channels without visualization dependencies."""

    def export(
        self,
        path            : Path,
        positions       : np.ndarray,
        channels        : Mapping[str, np.ndarray],
        latent_channels : Sequence[int] = (),
    ) -> tuple[Path, Path]:
        """Write an ASCII PLY and lossless companion NPZ over identical point order.

        Args:
            path: Final PLY path.
            positions: Finite Cartesian coordinates ``float [M,3]`` in ångströms.
            channels: Named scalar ``[M]`` arrays or a ``surface_embeddings [M,H]`` matrix.
            latent_channels: Explicit embedding dimensions to export. Empty means no latent values;
                indices are never selected or projected silently.

        Returns:
            Paths to the PLY and companion NPZ.

        Raises:
            ValueError: If coordinates/channels are misaligned, non-scalar, or names unsafe.
        """
        coordinates = np.asarray(positions, dtype=np.float32)
        if coordinates.ndim != 2 or coordinates.shape[1] != 3 or not np.isfinite(coordinates).all():
            raise ValueError("point-cloud positions must have finite shape [M,3]")
        scalar_channels: dict[str, np.ndarray] = {}
        for name, values in channels.items():
            if name == "surface_embeddings":
                matrix = np.asarray(values)
                if matrix.ndim != 2 or matrix.shape[0] != len(coordinates):
                    raise ValueError("surface_embeddings must have shape [M,H]")
                for index in latent_channels:
                    if index < 0 or index >= matrix.shape[1]:
                        raise ValueError(f"latent channel {index} is outside embedding width")
                    scalar_channels[f"latent_chemical_channel_{index}"] = matrix[:, index]
                continue
            if not name.replace("_", "").isalnum():
                raise ValueError(f"PLY scalar channel has an unsafe name: {name!r}")
            array = np.asarray(values).reshape(-1)
            if array.shape != (len(coordinates),):
                raise ValueError(f"PLY scalar channel {name!r} must have shape [M]")
            scalar_channels[name] = array

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        header = [
            "ply",
            "format ascii 1.0",
            "comment WISDOM evaluated surface point cloud",
            f"element vertex {len(coordinates)}",
            "property float x",
            "property float y",
            "property float z",
            *(f"property float {name}" for name in scalar_channels),
            "end_header",
        ]
        with temporary.open("w", encoding="ascii") as stream:
            stream.write("\n".join(header) + "\n")
            arrays = [coordinates[:, index] for index in range(3)] + [
                np.asarray(values, dtype=np.float32) for values in scalar_channels.values()
            ]
            for row in zip(*arrays, strict=True):
                stream.write(" ".join(f"{float(value):.8g}" for value in row) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)

        companion = path.with_suffix(".npz")
        temporary_npz = companion.with_name(
            f".{companion.name}.{os.getpid()}.{uuid4().hex}.tmp.npz"
        )
        try:
            payload = {
                "positions": coordinates,
                **{name: np.asarray(values) for name, values in scalar_channels.items()},
            }
            np.savez_compressed(temporary_npz, **payload)  # type: ignore[arg-type]
            os.replace(temporary_npz, companion)
        finally:
            temporary_npz.unlink(missing_ok=True)
        return path, companion
