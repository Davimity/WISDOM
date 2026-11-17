"""Copy the externally maintained scientific report into the repository, or verify its freshness."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def content_digest(path: Path) -> str:
    """Hash a report incrementally, including on the minimum supported Python 3.10.

    Args:
        path: Readable PDF or other file whose exact bytes must be identified.

    Returns:
        SHA-256 hexadecimal digest without retaining the complete document in memory.
    """
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def synchronize(source: Path, destination: Path, check: bool = False) -> dict[str, str | int]:
    """Publish an exact PDF snapshot and its digest without editing or compiling the report.

    Args:
        source: Author-maintained PDF, outside or inside this repository.
        destination: Repository PDF path; its sibling JSON records the content digest and size.
        check: Verify both the copy and its metadata instead of replacing either file.

    Returns:
        Portable metadata containing the source filename, SHA-256 digest and byte count.

    Raises:
        ValueError: The source is not a PDF, or the checked copy/metadata is stale.
        OSError: A source cannot be read or a destination cannot be written.
    """
    with source.open("rb") as stream:
        if stream.read(5) != b"%PDF-":
            raise ValueError("the report source must be a PDF")
    digest = content_digest(source)
    metadata: dict[str, str | int] = {
        "source_filename": source.name,
        "sha256": digest,
        "bytes": source.stat().st_size,
    }
    manifest = destination.with_suffix(".json")

    # Byte identity makes an external report update explicit; timestamps alone prove nothing.

    if check:
        if not destination.is_file() or not manifest.is_file():
            raise ValueError(
                "the repository report or its metadata is missing; synchronize it first"
            )
        copied_digest = content_digest(destination)
        if copied_digest != digest or json.loads(manifest.read_text()) != metadata:
            raise ValueError("the repository report is stale; run sync_report.py without --check")
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copyfile(source, destination)
        manifest.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    return metadata


def main() -> None:
    """Synchronize an explicit local PDF or exit unsuccessfully when its checked copy is stale."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="author-maintained compiled PDF")
    parser.add_argument(
        "--check", action="store_true", help="verify freshness without modifying files"
    )
    args = parser.parse_args()
    target = Path(__file__).resolve().parents[1] / "docs" / "WISDOM-memoria.pdf"
    metadata = synchronize(args.source, target, args.check)
    action = "verified" if args.check else "synchronized"
    print(f"Report {action}: {target} ({metadata['bytes']} bytes)")


if __name__ == "__main__":
    main()
