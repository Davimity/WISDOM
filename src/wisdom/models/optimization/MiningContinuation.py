"""Native bounded continuation checkpoint for VN01's epoch-dependent selector state."""

import torch
import random
import numpy as np

from pathlib import Path
from typing import Any, cast
from collections.abc import Mapping


class MiningContinuation:
    """Save one replaceable LF-owned epoch checkpoint, not one large archive per epoch."""

    NAME = "negative-mining/continuation.pt"

    def __init__(self, work: Any) -> None:
        """Bind the native checkpoint service of the current compatible Run.

        Args:
            work: Executing LF Work; framework recovery owns Run and Attempt identity.
        """
        self.work = work

    def save(self, epoch: int, state: Mapping[str, Any], generator: torch.Generator) -> None:
        """Commit weights/optimizer/miner/counters and all optimizer RNG streams atomically.

        Args:
            epoch: Completed optimizer epoch; its review is already complete or explicitly final.
            state: Trusted tensor/JSON-shaped training state including the best checkpoint bundle.
            generator: Training sampler's private CPU generator, separate from loader worker seeds.
        """
        numpy_state = cast(Any, np.random.get_state())
        payload = {**state, "epoch": epoch, "python_rng": random.getstate(),
            "numpy_rng": [numpy_state[0], numpy_state[1].tolist(), *numpy_state[2:]],
            "torch_rng": torch.get_rng_state(), "loader_rng": generator.get_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}

        def build(path: Path) -> None:
            """Serialize this complete epoch into LF's private unpublished build target.

            Args:
                path: Native temporary checkpoint target; LF owns replacement and integrity.
            """
            torch.save(payload, path)

        self.work.checkpoints.file(self.NAME, build=build, step=epoch,
            validate=lambda path: torch.load(path, map_location="cpu", weights_only=True).get(
                "epoch") == epoch)

    def restore(self, generator: torch.Generator) -> dict[str, Any] | None:
        """Restore a committed epoch without reconstructing priorities using other weights.

        Args:
            generator: Training sampler generator whose order must continue identically.

        Returns:
            Trusted weights/optimizer/miner/counters, or None before the first committed epoch.

        Raises:
            ValueError: Native checkpoint integrity or visible GPU topology is incompatible.
        """
        if not self.work.checkpoints.exists(self.NAME):
            return None
        path = self.work.checkpoints.file(self.NAME)
        state: dict[str, Any] = torch.load(path, map_location="cpu", weights_only=True)
        random.setstate(state["python_rng"])
        numpy = state["numpy_rng"]
        np.random.set_state((numpy[0], np.asarray(numpy[1], dtype=np.uint32),
                             numpy[2], numpy[3], numpy[4]))
        torch.set_rng_state(state["torch_rng"])
        generator.set_state(state["loader_rng"])
        if state["cuda_rng"]:
            if len(state["cuda_rng"]) != torch.cuda.device_count():
                raise ValueError("VN01 continuation requires the same granted logical GPU count")
            torch.cuda.set_rng_state_all(state["cuda_rng"])
        return state
