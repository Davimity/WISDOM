"""Explicit filesystem paths shared by cohesive WISDOM preprocessing components."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


class ProcessingWorkspace:
    """Resolve named scientific inputs and outputs without implementing an execution lifecycle.

    LambdaForge owns execution, checkpoints, retries, progress, and parallelism through
    :class:`lambdaforge.Work`. This class only gives lower-level scientific components explicit,
    immutable path bindings so they do not depend on removed framework context classes.
    """

    def __init__(
        self,
        run_dir: Path,
        inputs : Mapping[str, Path],
        outputs: Mapping[str, Path],
    ) -> None:
        """Bind run-owned paths used by one scientific stage.

        Args:
            run_dir: Root owned by the active LambdaForge Work attempt.
            inputs: Read-only logical input paths already resolved by LambdaForge.
            outputs: Run-owned or checkpoint-owned logical output paths.

        Raises:
            ValueError: If a logical name is empty or an output escapes ``run_dir``.
        """
        if any(not str(name).strip() for name in (*inputs, *outputs)):
            raise ValueError("workspace path names cannot be empty")

        self.run_dir  = run_dir.resolve()
        self._inputs  = {str(name): Path(path).resolve() for name, path in inputs.items()}
        self._outputs = {str(name): Path(path).resolve() for name, path in outputs.items()}

    def input(self, name: str) -> Path:
        """Return one explicitly bound immutable input path.

        Args:
            name: Logical input name.

        Returns:
            Resolved input path.

        Raises:
            KeyError: If the component requests an undeclared input.
        """
        return self._inputs[name]

    def output(self, name: str, create: bool = False) -> Path:
        """Return one explicitly bound output path and optionally create its parent.

        Args:
            name: Logical output name.
            create: Create the parent directory when true; the target itself may be a file.

        Returns:
            Resolved output path.

        Raises:
            KeyError: If the component requests an undeclared output.
        """
        path = self._outputs[name]
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def output_path(self, name: str) -> Path:
        """Return a run-root-relative auxiliary output path.

        Args:
            name: Safe relative auxiliary filename.

        Returns:
            Path below ``run_dir``.

        Raises:
            ValueError: If ``name`` escapes the run root.
        """
        path = (self.run_dir / name).resolve()
        if not path.is_relative_to(self.run_dir):
            raise ValueError(f"workspace output escapes the run root: {name}")
        return path

    def declared_input_path(self, path: Path) -> Path:
        """Validate and return a local path declared inside an input manifest.

        Args:
            path: Existing local coordinate path selected from the manifest.

        Returns:
            Resolved existing path.

        Raises:
            FileNotFoundError: If the manifest-selected path does not exist.
        """
        selected = path.resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"manifest-selected structure does not exist: {selected}")
        return selected
