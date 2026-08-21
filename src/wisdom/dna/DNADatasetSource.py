"""Automatic discovery from pinned public DNA-binding benchmark releases."""

from __future__ import annotations

import gzip
import hashlib
import json
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any

from lambdaforge.preprocessing import PreprocessingRecord, PreprocessingSource
from lambdaforge.tasks import TaskContext

from wisdom.dna.DiscoveryMode import DiscoveryMode
from wisdom.dna.EvidenceKind import EvidenceKind
from wisdom.dna.PublicDataClient import PublicDataClient


class DNADatasetSource(PreprocessingSource):
    """Read DyProL positives and BTD-Combo negatives without private input files."""

    DYPROL_RECORD       = "zenodo:19547616"
    DYPROL_VERSION      = "v1-2026-04-13"
    DYPROL_URL          = "https://zenodo.org/api/records/19547616/files/DyProL.zip/content"
    DYPROL_SIZE         = 14_466_259_823
    DYPROL_ARCHIVE_MD5  = "0ee550e020a1d9a8149e7f84c504d924"
    DYPROL_TRAIN_MEMBER = "DyProL/DNA/DNA-1022_Train.txt"
    DYPROL_TEST_MEMBER  = "DyProL/DNA/DNA-256_Test.txt"
    DYPROL_TRAIN_SHA256 = "101bd43951e4213085f926289a1c2fe8fb3760d88994c13648c044efa8aa9d55"
    DYPROL_TEST_SHA256  = "6abdceec8496b0edc6c802e29e49d29db77ed0313fe5ab9331c35a2ebc769c6e"

    BTD_COMMIT       = "714756450e537cebbc3d9814a1fc059758fee58b"
    BTD_VERSION      = "BTD-Combo-train-test-2024-09-01"
    BTD_ROOT         = f"https://raw.githubusercontent.com/Rafeed-bot/DNA_BP_Benchmarking/{BTD_COMMIT}/Dataset"
    BTD_TRAIN_SHA256 = "f6b23bed7a88f611dea3fca774f8507ceac4e7b2c7e4bed5a558450b485741a6"
    BTD_TEST_SHA256  = "a866192cf598085e5953b16b34b4c2844c7e279ba412ed507d2b08d323de672c"

    BIOLIP_URL     = "https://seq2fun.dcmb.med.umich.edu/BioLiP/download/BioLiP.txt.gz"
    BIOLIP_VERSION = "snapshot-2026-03-29"
    BIOLIP_SHA256  = "a6b172e7d56c70ea4c25762045d6dc606f51728ec43af6b2369b71a49e1d8f0c"

    def __init__(
        self,
        mode                    : str   = DiscoveryMode.LIVE,
        source_manifest_input   : str   = "public_sources",
        fixture_input           : str   = "candidate_fixtures",
        cache_output            : str   = "discovery-cache",
        positive_source         : str   = "dyprol-v1",
        negative_source         : str   = "btd-combo-2024",
        max_positive_candidates : int   = 0,
        max_negative_candidates : int   = 0,
        requests_per_second     : float = 2.0,
        retries                 : int   = 4,
    ) -> None:
        """Configure pinned benchmark sources and optional smoke-test limits.

        ``max_*_candidates`` limits source records before expensive structure mapping. Zero means
        every record in the pinned release. Limits preserve both development and external-test
        partitions and use the same downstream code path as a production build.

        Args:
            mode: ``live`` for public releases or ``fixture`` for an offline JSONL test input.
            source_manifest_input: Named immutable public-source definition fingerprinted by
                LambdaForge.
            fixture_input: Named LambdaForge JSONL input used only in fixture mode.
            cache_output: Named run-relative output containing exact downloaded source bytes.
            positive_source: Closed source name; currently ``dyprol-v1``.
            negative_source: Closed source name; currently ``btd-combo-2024``.
            max_positive_candidates: Non-negative positive record limit; zero means unlimited.
            max_negative_candidates: Non-negative negative record limit; zero means unlimited.
            requests_per_second: Positive mean request-rate limit.
            retries: Positive bounded HTTP attempt count.

        Raises:
            ValueError: If a source name, mode, or operational bound is invalid.
        """
        try:
            self.mode = DiscoveryMode(mode)
        except ValueError as error:
            raise ValueError("mode must be 'fixture' or 'live'") from error
        if positive_source != "dyprol-v1":
            raise ValueError("positive_source must be 'dyprol-v1'")
        if negative_source != "btd-combo-2024":
            raise ValueError("negative_source must be 'btd-combo-2024'")
        if max_positive_candidates < 0 or max_negative_candidates < 0:
            raise ValueError("candidate limits cannot be negative")
        if not source_manifest_input.strip():
            raise ValueError("source_manifest_input cannot be empty")

        self.source_manifest_input    = source_manifest_input
        self.fixture_input           = fixture_input
        self.cache_output            = cache_output
        self.positive_source         = positive_source
        self.negative_source         = negative_source
        self.max_positive_candidates = int(max_positive_candidates)
        self.max_negative_candidates = int(max_negative_candidates)
        self.client                  = PublicDataClient(requests_per_second, retries)

    def records(self, context: TaskContext) -> Iterable[PreprocessingRecord]:
        """Yield interleaved, stable candidates from pinned public source manifests.

        Live mode range-fetches only DyProL's two DNA manifests, downloads the two small BTD-Combo
        FASTA files, and materializes a compact BioLiP DNA-UniProt conflict index. Positive and
        negative records are interleaved so ``lambdaforge debug --records N`` exercises both
        classes, but source interleaving does not imply equal accepted class counts. Exact balance
        is enforced only after evidence filtering, structure mapping, and homology-safe splitting
        by :class:`DNASelectionSink`. This source performs no structural preprocessing and assigns
        no WISDOM surface.

        Args:
            context: LambdaForge context resolving fixture inputs and the source cache output.

        Yields:
            Candidate records containing source labels, sequences, published partition, and exact
            release/checksum provenance. Expensive structure mapping belongs to the transform.

        Raises:
            OSError: If cached source bytes cannot be read or atomically published.
            RuntimeError: If a pinned public release changes or contains malformed records.
            ValueError: If an offline fixture row is not an object.
        """
        if self.mode is DiscoveryMode.FIXTURE:
            yield from self._unique_records(self._json_lines(context.input(self.fixture_input)))
            return

        self._validate_source_manifest(context.input(self.source_manifest_input))

        cache_root = context.output(self.cache_output)
        cache_root.mkdir(parents=True, exist_ok=True)
        positive = self._bounded(
            self._dyprol_candidates(cache_root),
            self.max_positive_candidates,
        )
        negative = self._bounded(
            self._btd_candidates(cache_root),
            self.max_negative_candidates,
        )

        candidates: list[dict[str, Any]] = []
        for left, right in zip_longest(positive, negative):
            for value in (left, right):
                if value is None:
                    continue
                candidates.append(value)

        # Reject ambiguous source identity before building indexes or mapping remote structures.
        records = self._unique_records(candidates)

        # The compact index lets each negative transform reject known BioLiP DNA binders locally.
        biolip_index = self._biolip_index(cache_root)
        query_date   = datetime.now(timezone.utc).isoformat()
        for record in records:
            value = dict(record.value)
            value["biolip_dna_uniprot_path"] = str(biolip_index)
            value["query_date_utc"]          = query_date
            yield record.with_value(value)

    def _validate_source_manifest(self, path: Path) -> None:
        """Require the declared public-source manifest to match code-level parsers exactly.

        Args:
            path: LambdaForge-bound JSON definition whose bytes participate in task identity.

        Raises:
            ValueError: If the schema, versions, URLs, or pinned checksums disagree with this
                implementation.
            OSError: If the manifest cannot be read.
        """
        payload = json.loads(path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": "1.0",
            "dyprol": {
                "record": self.DYPROL_RECORD,
                "version": self.DYPROL_VERSION,
                "url": self.DYPROL_URL,
                "archive_md5": self.DYPROL_ARCHIVE_MD5,
                "train_sha256": self.DYPROL_TRAIN_SHA256,
                "test_sha256": self.DYPROL_TEST_SHA256,
            },
            "btd_combo": {
                "version": self.BTD_VERSION,
                "commit": self.BTD_COMMIT,
                "train_sha256": self.BTD_TRAIN_SHA256,
                "test_sha256": self.BTD_TEST_SHA256,
            },
            "biolip2": {
                "version": self.BIOLIP_VERSION,
                "url": self.BIOLIP_URL,
                "sha256": self.BIOLIP_SHA256,
            },
        }
        if payload != expected:
            raise ValueError(
                "public source manifest does not match the pinned parser contract; update both "
                "deliberately instead of accepting mutable source drift"
            )

    def _dyprol_candidates(self, cache_root: Path) -> list[dict[str, Any]]:
        """Extract and parse the pinned DyProL DNA train/test manifests.

        Args:
            cache_root: Run-relative public-source cache directory.

        Returns:
            Positive candidate mappings with PDB chain IDs and residue-level binding masks.

        Raises:
            RuntimeError: If member bytes, three-line records, sequences, or masks are invalid.
        """
        specifications = (
            (
                self.DYPROL_TRAIN_MEMBER,
                cache_root / "sources" / "dyprol-dna-train.txt",
                self.DYPROL_TRAIN_SHA256,
                "development",
            ),
            (
                self.DYPROL_TEST_MEMBER,
                cache_root / "sources" / "dyprol-dna-test.txt",
                self.DYPROL_TEST_SHA256,
                "external_test",
            ),
        )
        output: list[dict[str, Any]] = []
        for member, path, digest, partition in specifications:
            source_path = self.client.zip_member(
                self.DYPROL_URL,
                self.DYPROL_SIZE,
                member,
                path,
            )
            if hashlib.sha256(source_path.read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"DyProL member digest changed: {member}")
            lines = [line.strip() for line in source_path.read_text().splitlines() if line.strip()]
            if len(lines) % 3:
                raise RuntimeError(f"DyProL member has an incomplete three-line record: {member}")
            for offset in range(0, len(lines), 3):
                header, sequence, mask = lines[offset : offset + 3]
                source_identifier = header.removeprefix(">")
                pdb_text, separator, chain = source_identifier.partition("_")
                pdb_id     = pdb_text.upper()
                identifier = f"{pdb_id}_{chain}"
                if (
                    not header.startswith(">")
                    or not separator
                    or len(pdb_id) != 4
                    or len(chain) != 1
                    or not pdb_id.isalnum()
                    or not chain.isalnum()
                    or not sequence.isalpha()
                    or set(mask) - {"0", "1"}
                    or len(mask) != len(sequence)
                ):
                    raise RuntimeError(f"malformed DyProL DNA record {header!r}")
                output.append(
                    {
                        "source_class": "positive",
                        "published_partition": partition,
                        "source_database": "DyProL",
                        "source_record": identifier,
                        "source_version": self.DYPROL_VERSION,
                        "source_url": self.DYPROL_URL,
                        "source_checksum": f"md5:{self.DYPROL_ARCHIVE_MD5}",
                        "pdb_id": pdb_id,
                        "protein_chain": chain,
                        "sequence": sequence.upper(),
                        "binding_site_mask": mask,
                        "local_gt_expected": "1" in mask,
                        "evidence": [EvidenceKind.CURATED_DNA_BINDING.value],
                        "evidence_sources": [self.DYPROL_RECORD],
                    }
                )
        return output

    def _btd_candidates(self, cache_root: Path) -> list[dict[str, Any]]:
        """Download and parse negative records from the published BTD-Combo split.

        Args:
            cache_root: Run-relative public-source cache directory.

        Returns:
            Negative sequence mappings retaining the authors' development/external-test partition.

        Raises:
            RuntimeError: If pinned FASTA bytes or labels are malformed.
        """
        specifications = (
            ("Train.fasta", self.BTD_TRAIN_SHA256, "development"),
            ("Test.fasta", self.BTD_TEST_SHA256, "external_test"),
        )
        output: list[dict[str, Any]] = []
        for filename, digest, partition in specifications:
            url  = f"{self.BTD_ROOT}/{filename}"
            path = self.client.download(url, cache_root / "sources" / filename, digest)
            header: str | None = None
            sequence: list[str] = []
            for line in (*path.read_text().splitlines(), ">END_label_1"):
                if line.startswith(">"):
                    if header is not None and header.endswith("_label_0"):
                        joined = "".join(sequence).upper()
                        if not joined.isalpha():
                            raise RuntimeError(f"BTD record {header!r} has invalid residues")
                        output.append(
                            {
                                "source_class": "negative",
                                "published_partition": partition,
                                "source_database": "BTD-Combo",
                                "source_record": header.removeprefix(">"),
                                "source_version": self.BTD_VERSION,
                                "source_url": url,
                                "source_checksum": f"sha256:{digest}",
                                "sequence": joined,
                                "local_gt_expected": True,
                                "evidence": [EvidenceKind.CURATED_NOT_DNA_BINDING.value],
                                "evidence_sources": [
                                    "doi:10.1093/bib/bbae634",
                                    f"github:{self.BTD_COMMIT}",
                                ],
                                "negative_source": "BTD-Combo",
                                "negative_source_label": "non_DBP",
                            }
                        )
                    header   = line.strip()
                    sequence = []
                elif header is not None:
                    sequence.append(line.strip())
        return output

    def _biolip_index(self, cache_root: Path) -> Path:
        """Build a checksum-pinned set of UniProt IDs with BioLiP DNA interactions.

        Args:
            cache_root: Run-relative public-source cache directory.

        Returns:
            Absolute path to a sorted newline-delimited UniProt index.

        Raises:
            RuntimeError: If the BioLiP snapshot contains malformed tabular rows.
        """
        index_path = cache_root / "sources" / "biolip-dna-uniprot.txt"
        if index_path.is_file():
            return index_path.resolve()
        archive = self.client.download(
            self.BIOLIP_URL,
            cache_root / "sources" / "BioLiP-2026-03-29.txt.gz",
            self.BIOLIP_SHA256,
        )
        identifiers: set[str] = set()
        with gzip.open(archive, "rt", encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, start=1):
                columns = line.rstrip("\n").split("\t")
                if len(columns) < 21:
                    raise RuntimeError(f"BioLiP row {line_number} has fewer than 21 columns")
                if columns[4] == "dna" and columns[17] and columns[17] != "----":
                    identifiers.update(value.strip() for value in columns[17].split(","))
        content = "".join(f"{value}\n" for value in sorted(identifiers)).encode("utf-8")
        self.client.atomic_write(index_path, content)
        return index_path.resolve()

    @staticmethod
    def _bounded(values: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
        """Apply a deterministic class limit while retaining both published partitions.

        Args:
            values: Candidate mappings from one source class.
            limit: Maximum returned records; zero means every record.

        Returns:
            Stable records selected by SHA-256 rank and alternated by source partition.
        """
        if limit == 0 or len(values) <= limit:
            return values
        groups = {
            partition: sorted(
                (value for value in values if value["published_partition"] == partition),
                key=lambda value: hashlib.sha256(
                    f"2026:{value['source_record']}".encode()
                ).hexdigest(),
            )
            for partition in ("development", "external_test")
        }
        selected: list[dict[str, Any]] = []
        for left, right in zip_longest(groups["development"], groups["external_test"]):
            for value in (left, right):
                if value is not None:
                    selected.append(value)
                    if len(selected) == limit:
                        return selected
        return selected

    @staticmethod
    def _json_lines(path: Path) -> tuple[dict[str, Any], ...]:
        """Read object-valued JSONL fixture records without network access.

        Args:
            path: Declared LambdaForge fixture input.

        Returns:
            Ordered object mappings, ignoring blank lines and comments.

        Raises:
            ValueError: If a data line is not a JSON object.
        """
        output: list[dict[str, Any]] = []
        for line_number, line in enumerate(path.read_text().splitlines(), start=1):
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} must contain a JSON object")
            output.append(value)
        return tuple(output)

    @staticmethod
    def _unique_records(values: Iterable[Mapping[str, Any]]) -> tuple[PreprocessingRecord, ...]:
        """Collapse exact repetitions and reject ambiguous source-key collisions up front.

        LambdaForge requires every source record key to be unique. The key includes source database,
        source version, published partition, and the source's native record identifier, so the same
        biological protein may legitimately occur in different sources without colliding. An exact
        repeated row within one release is harmless and retains its first occurrence. Two
        different rows producing the same key are ambiguous scientific input and fail before
        structure mapping, downloads, or curation consume substantial time.

        Args:
            values: Ordered candidate mappings after source parsing and provenance enrichment.

        Returns:
            Stable unique LambdaForge records in first-occurrence order.

        Raises:
            RuntimeError: If one candidate key denotes two non-identical source rows.
            ValueError: If a candidate lacks fields required by :meth:`_record`.
        """
        records  : list[PreprocessingRecord]      = []
        previous : dict[str, PreprocessingRecord] = {}

        for value in values:
            record = DNADatasetSource._record(value)
            prior  = previous.get(record.key)
            if prior is None:
                previous[record.key] = record
                records.append(record)
                continue
            if prior.value != record.value:
                raise RuntimeError(
                    "public source rows produce one conflicting candidate key: "
                    f"{record.key!r}; source record identifiers, including chain case, must be "
                    "unique within a release partition"
                )

        return tuple(records)

    @staticmethod
    def _record(value: Mapping[str, Any]) -> PreprocessingRecord:
        """Validate source identity and construct a stable LambdaForge record.

        Args:
            value: Public-source or fixture candidate mapping.

        Returns:
            Stable record keyed by source/version/partition/record identifier.

        Raises:
            ValueError: If essential source identity or sequence evidence is missing.
        """
        if value.get("structure_path"):
            key = str(value.get("candidate_key") or value.get("source_record") or "fixture")
            return PreprocessingRecord(key=key, value=dict(value))
        required = ("source_database", "source_record", "source_version", "sequence", "evidence")
        missing  = [name for name in required if not value.get(name)]
        if missing:
            raise ValueError(f"DNA public candidate is missing fields: {missing}")
        key = ":".join(
            (
                str(value["source_database"]),
                str(value["source_version"]),
                str(value.get("published_partition", "fixture")),
                str(value["source_record"]),
            )
        )
        return PreprocessingRecord(key=key, value=dict(value), metadata={"candidate_key": key})
