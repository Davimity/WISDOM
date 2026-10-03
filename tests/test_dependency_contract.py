"""Prevent WISDOM packaging from contradicting its required LambdaForge report extra."""

# ruff: noqa: I001

import re
import tomli

from pathlib import Path
from packaging.version import Version
from importlib.metadata import distribution
from packaging.requirements import Requirement


def test_report_extra_and_wisdom_share_a_supported_plotly_domain() -> None:
    """Check declared Plotly 7 compatibility without requiring a rewritten installed environment.

    The actual framework report extra must accept the same supported major-version domain as
    the protein viewer. Installed WISDOM metadata may be stale until the researcher reinstalls;
    therefore its authoritative requirement is read from the project's TOML, not site-packages.
    """
    project_root = Path(__file__).parents[1]
    project = tomli.loads((project_root / "pyproject.toml").read_text())["project"]
    requirements = {r.name: r for r in map(Requirement, project["dependencies"])}
    assert "analysis-report" in requirements["lambdaforge"].extras
    report_dependencies = [
        r
        for r in map(Requirement, distribution("lambdaforge").requires or [])
        if r.name == "plotly"
        and r.marker is not None
        and r.marker.evaluate({"extra": "analysis-report"})
    ]
    assert report_dependencies, "The required framework report extra must declare Plotly."
    supported_versions = (Version("7.0.0"), Version("7.1.0"))
    for version in supported_versions:
        assert version in requirements["plotly"].specifier
    assert any(
        all(version in r.specifier for r in report_dependencies) for version in supported_versions
    )
    assert Version("6.9.0") not in requirements["plotly"].specifier


def test_installer_minimum_matches_the_project_framework_requirement() -> None:
    """Ensure the installer rejects the same obsolete framework releases as package resolution."""
    project_root = Path(__file__).parents[1]
    project = tomli.loads((project_root / "pyproject.toml").read_text())["project"]
    framework = next(
        r for r in map(Requirement, project["dependencies"]) if r.name == "lambdaforge"
    )
    minimum = re.search(
        r'^LAMBDAFORGE_MIN_VERSION="([^"]+)"', (project_root / "install.sh").read_text(), re.M
    )
    assert minimum is not None
    assert str(framework.specifier) == f">={minimum.group(1)}"
