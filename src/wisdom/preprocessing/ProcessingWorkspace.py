"""Explicit path bindings shared by retained geometry and annotation components."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path


class ProcessingWorkspace:
    """Resolve named scientific paths without implementing an execution lifecycle.

    LambdaForge owns execution, checkpoints, progress, resume, and parallelism. This small value
    object only lets retained geometry and annotation classes receive explicit paths rather than a
    framework context or hidden global state.
    """

    def __init__(
        self,
        run_dir: Path,
        inputs : Mapping[str, Path],
        outputs: Mapping[str, Path],
    ) -> None:
        """Bind immutable inputs and owned outputs for one internal scientific phase.

        Args:
            run_dir: Root owned by the active LambdaForge Work Attempt.
            inputs: Logical names mapped to already-resolved immutable paths.
            outputs: Logical names mapped to run-, checkpoint-, or cache-owned paths.

        Raises:
            ValueError: If a logical name is empty.
        """
        if any(not str(name).strip() for name in (*inputs, *outputs)):
            raise ValueError("workspace path names cannot be empty")
        self.run_dir  = run_dir.resolve()
        self._inputs  = {str(name): Path(path).resolve() for name, path in inputs.items()}
        self._outputs = {str(name): Path(path).resolve() for name, path in outputs.items()}

    def input(self, name: str) -> Path:
        """Return one explicitly bound immutable input.

        Args:
            name: Declared logical input name.

        Returns:
            Resolved input path.

        Raises:
            KeyError: If ``name`` was not bound.
        """
        return self._inputs[name]

    def output(self, name: str, create: bool = False) -> Path:
        """Return one bound output and optionally create its parent directory.

        Args:
            name: Declared logical output name.
            create: Create the parent directory when true.

        Returns:
            Resolved output path.

        Raises:
            KeyError: If ``name`` was not bound.
        """
        path = self._outputs[name]
        if create:
            path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def output_path(self, name: str) -> Path:
        """Resolve a safe auxiliary path below the Work run directory.

        Args:
            name: Relative auxiliary path.

        Returns:
            Resolved path below ``run_dir``.

        Raises:
            ValueError: If the relative name escapes ``run_dir``.
        """
        path = (self.run_dir / name).resolve()
        if not path.is_relative_to(self.run_dir):
            raise ValueError(f"workspace output escapes the run root: {name}")
        return path

    def declared_input_path(self, path: Path) -> Path:
        """Validate a local coordinate path explicitly present in an input manifest.

        Args:
            path: Manifest-selected local coordinate file.

        Returns:
            Resolved existing file.

        Raises:
            FileNotFoundError: If the selected file does not exist.
        """
        selected = path.resolve()
        if not selected.is_file():
            raise FileNotFoundError(f"manifest-selected structure does not exist: {selected}")
        return selected
