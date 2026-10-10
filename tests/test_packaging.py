"""Verify deployed wheel contents, including the final training report's shared renderer."""

import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path


def test_wheel_uses_current_source_despite_newer_stale_build_files(tmp_path):
    """Build offline with future-dated obsolete modules and compare every shipped source byte."""
    root = Path(__file__).resolve().parents[1]
    project = tmp_path / "project"
    project.mkdir()
    for filename in ("pyproject.toml", "README.md", "setup.cfg"):
        shutil.copy2(root / filename, project / filename)
    shutil.copytree(root / "src/wisdom", project / "src/wisdom")
    shutil.copytree(project / "src/wisdom", project / "build/lib/wisdom")

    # Reproduce the deployed failure: an obsolete cached viewer looks newer than its source.

    viewer = project / "build/lib/wisdom/preprocessing/common/structure/ProteinVisualizer.py"
    viewer.write_text(viewer.read_text().replace("def render(", "def retired_render("))
    future = time.time() + 3600
    for cached in (project / "build/lib/wisdom").rglob("*.py"):
        os.utime(cached, (future, future))

    wheels = tmp_path / "wheels"
    wheels.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from setuptools.build_meta import build_wheel; "
            "import sys; build_wheel(sys.argv[1])",
            str(wheels),
        ],
        cwd=project,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    wheel, = wheels.glob("*.whl")
    with zipfile.ZipFile(wheel) as archive:
        for source in (project / "src/wisdom").rglob("*.py"):
            name = source.relative_to(project / "src").as_posix()
            assert archive.read(name) == source.read_bytes(), name
        assert b"def render(" in archive.read(
            "wisdom/preprocessing/common/structure/ProteinVisualizer.py"
        )
