"""Audit fixed V9f contrasts from LF metadata; never train, change decisions or read weights."""

# ruff: noqa: I001

import argparse

from pathlib import Path
from lambdaforge.work.ResultStore import ResultStore
from wisdom.analysis.SchNetCampaignAudit import SchNetCampaignAudit


def main() -> None:
    """Read one selected registered/imported Execution and author a standalone descriptive audit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("execution", help="exact native registered or imported Execution ID")
    parser.add_argument("--root", type=Path, help="ResultStore root, or the project default")
    parser.add_argument("--output", type=Path, required=True, help="new standalone audit directory")
    args = parser.parse_args()
    execution = ResultStore(args.root).select(args.execution)
    report = SchNetCampaignAudit(execution).write(args.output)
    completed = sum(len(cell["completed_seeds"]) for cell in report["coverage"].values())
    print(f"V9f metadata audit: {completed}/52 completed cells/seeds; output={args.output}")
    print("Descriptive contrasts only. Native LF owns Study decisions and curve reports.")


if __name__ == "__main__":
    main()
