"""Author a native V9f cell/block configuration; never submit or execute scientific work."""

# ruff: noqa: I001

import re
import yaml
import argparse

from pathlib import Path
from lambdaforge.work import WorkConfig


def configure(source: Path, output: Path, cells: list[int]) -> int:
    """Retain explicit candidate rows and original comments in a new LF-native YAML.

    Args:
        source: Complete thirteen-cell V9f YAML; never modified.
        output: New authoring file, outside retained evidence and distinct from source.
        cells: One-based declared cell numbers. The caller includes cell 01 for a matched control.

    Returns:
        Native planned Run count, including all four mandatory seeds.

    Raises:
        ValueError: Cells are invalid/duplicated or destination is the source/evidence directory.
        FileExistsError: Destination exists; authoring never overwrites user files.
    """
    if not cells or len(set(cells)) != len(cells) or not all(1 <= cell <= 13 for cell in cells):
        raise ValueError("select unique cell numbers in 1..13")
    if source.resolve() == output.resolve() or any(
        part in {"runs", "attempts", "artifacts"} for part in output.parts
    ):
        raise ValueError("author a new configuration, never overwrite source or retained evidence")
    source_text = source.read_text()
    start = source_text.index("    # Cell 01,")
    end = source_text.index("  reference:", start)
    rows = re.findall(r"    # Cell (\d+),.*?(?=    # Cell |\Z)", source_text[start:end], re.S)
    chunks = re.split(r"(?=    # Cell \d+,)", source_text[start:end])[1:]
    assert len(rows) == len(chunks) == 13
    selected = "".join(chunks[cell - 1] for cell in sorted(cells))
    result = source_text[:start] + selected + source_text[end:]
    if 1 not in cells:
        result = re.sub(r"^  reference:.*\n", "", result, flags=re.M)
    suffix = "-".join(f"{cell:02d}" for cell in sorted(cells))
    result = result.replace(
        "name: wisdom-v9f-schnet  #", f"name: wisdom-v9f-schnet-cells-{suffix}  #")
    result = result.replace("Thirteen exact configurations x four seeds = 52 mandatory Runs",
                            f"{len(cells)} selected configurations x four seeds "
                            f"= {len(cells)*4} Runs")
    config = WorkConfig.from_mapping(yaml.safe_load(result), source=output.resolve())
    errors = config.validation_errors(check_inputs=False)
    if errors:
        raise ValueError("\n".join(errors))
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as stream:
        stream.write(result)
    return config.planned_runs


def main() -> None:
    """Parse cell/block choices, validate native planning and write a configuration only."""
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--cells", type=int, nargs="+", help="one-based exact cells, e.g. 2 4")
    selection.add_argument("--block", choices=("A", "B", "C"), help="add cell 01 as block control")
    parser.add_argument("--output", type=Path, required=True, help="new YAML; never overwritten")
    args = parser.parse_args()
    blocks = {"A": list(range(1, 6)), "B": [1, *range(6, 10)], "C": [1, *range(10, 14)]}
    cells = blocks[args.block] if args.block else args.cells
    source = Path(__file__).resolve().parents[1] / "experiments/v9/wisdom_v9f.yaml"
    count = configure(source, args.output, cells)
    print(f"Authored {args.output}: {count} mandatory Runs. Nothing was submitted.")


if __name__ == "__main__":
    main()
