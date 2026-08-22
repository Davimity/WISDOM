"""Rate-limited, cached access to immutable public benchmark resources."""

from __future__ import annotations

import binascii
import hashlib
import json
import os
import struct
import threading
import time
import urllib.error
import urllib.request
import zlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from uuid import uuid4


class PublicDataClient:
    """Download exact public bytes with retries, caching, and atomic publication."""

    def __init__(
        self,
        requests_per_second: float = 2.0,
        retries            : int   = 4,
    ) -> None:
        """Configure bounded public-service access.

        Args:
            requests_per_second: Maximum mean HTTP request rate for this process.
            retries: Positive number of attempts before a request fails.

        Raises:
            ValueError: If the rate or retry count is not positive.
        """
        if requests_per_second <= 0.0 or retries < 1:
            raise ValueError("public-data request bounds must be positive")

        self.request_interval = 1.0 / float(requests_per_second)
        self.retries          = int(retries)
        self.last_request     = 0.0
        self.lock             = threading.Lock()

    def download(
        self,
        url            : str,
        path           : Path,
        expected_sha256: str | None = None,
    ) -> Path:
        """Cache one complete remote object and verify its optional SHA-256 digest.

        Existing bytes are reused only when the configured digest agrees. New bytes are written to
        a sibling temporary file, flushed, synchronized, and atomically renamed so interrupted
        downloads never masquerade as complete source evidence.

        Args:
            url: Official HTTPS download endpoint.
            path: Cache destination below a LambdaForge output directory.
            expected_sha256: Optional lowercase hexadecimal digest pin.

        Returns:
            Absolute path to verified cached bytes.

        Raises:
            RuntimeError: If downloaded or cached bytes disagree with the digest pin.
            OSError: If the request or atomic publication cannot complete.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file() and self._digest(path.read_bytes(), expected_sha256):
            return path.resolve()

        content = self.request(url)
        if not self._digest(content, expected_sha256):
            raise RuntimeError(f"downloaded source digest does not match its pin: {url}")
        self.atomic_write(path, content)
        return path.resolve()

    def json(
        self,
        url    : str,
        payload: Mapping[str, Any] | None = None,
    ) -> Mapping[str, Any]:
        """Request and decode one JSON object from an official API.

        Args:
            url: HTTPS endpoint, including encoded query parameters for GET requests.
            payload: Optional JSON mapping; when present, the request uses POST.

        Returns:
            Decoded object-valued JSON response.

        Raises:
            RuntimeError: If the response is not valid object-valued JSON.
        """
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        content_type = None if payload is None else "application/json"
        content = self.request(url, data=data, content_type=content_type)
        if not content:
            return {}
        try:
            value = json.loads(content)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(f"public API returned invalid JSON: {url}") from error
        if not isinstance(value, Mapping):
            raise RuntimeError(f"public API returned a non-object JSON value: {url}")
        return value

    def zip_member(
        self,
        url         : str,
        archive_size: int,
        member_name : str,
        path        : Path,
    ) -> Path:
        """Fetch one ZIP member through HTTP byte ranges without downloading the full archive.

        The method reads the ZIP64 end record, the complete central directory, and exactly one
        local member. It verifies the member's uncompressed size and CRC-32 from the archive index.
        This is used for DyProL because its two text manifests occupy less than one megabyte inside
        a multi-gigabyte archive dominated by generated conformational ensembles.

        Args:
            url: Immutable archive URL.
            archive_size: Pinned byte length reported by the source repository.
            member_name: Exact POSIX member path inside the ZIP.
            path: Cache destination for the extracted member bytes.

        Returns:
            Absolute path to the validated extracted member.

        Raises:
            RuntimeError: If ZIP records, compression, size, or CRC are invalid.
            OSError: If range access or atomic publication fails.
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.is_file():
            return path.resolve()

        tail_start = max(0, archive_size - 65_536)
        tail       = self.request(url, byte_range=(tail_start, archive_size - 1))
        zip64_at   = tail.rfind(b"PK\x06\x06")
        if zip64_at < 0:
            raise RuntimeError("DyProL archive has no ZIP64 end-of-central-directory record")
        try:
            zip64 = struct.unpack_from("<4sQ2H2L4Q", tail, zip64_at)
        except struct.error as error:
            raise RuntimeError("DyProL ZIP64 end record is truncated") from error
        directory_size   = int(zip64[-2])
        directory_offset = int(zip64[-1])
        directory = self.request(
            url,
            byte_range=(directory_offset, directory_offset + directory_size - 1),
        )
        member = self._zip_index(directory).get(member_name)
        if member is None:
            raise RuntimeError(f"DyProL archive lacks required member {member_name!r}")
        compressed_size, uncompressed_size, local_offset, crc32, method = member

        header = self.request(url, byte_range=(local_offset, local_offset + 4095))
        if header[:4] != b"PK\x03\x04":
            raise RuntimeError(f"DyProL member {member_name!r} has no local ZIP header")
        try:
            local = struct.unpack_from("<4s5H3L2H", header)
        except struct.error as error:
            raise RuntimeError(f"DyProL member {member_name!r} has a truncated header") from error
        data_offset = local_offset + 30 + int(local[-2]) + int(local[-1])
        compressed  = self.request(
            url,
            byte_range=(data_offset, data_offset + compressed_size - 1),
        )
        if method == 0:
            content = compressed
        elif method == 8:
            content = zlib.decompress(compressed, -15)
        else:
            raise RuntimeError(f"unsupported ZIP compression method {method}")
        if len(content) != uncompressed_size:
            raise RuntimeError(f"DyProL member {member_name!r} has the wrong uncompressed size")
        if binascii.crc32(content) & 0xFFFFFFFF != crc32:
            raise RuntimeError(f"DyProL member {member_name!r} failed CRC-32 validation")

        self.atomic_write(path, content)
        return path.resolve()

    def request(
        self,
        url         : str,
        data        : bytes | None = None,
        content_type: str | None   = None,
        byte_range  : tuple[int, int] | None = None,
    ) -> bytes:
        """Perform one bounded HTTP request with serialized rate limiting and backoff.

        Args:
            url: HTTPS endpoint.
            data: Optional request body; its presence selects POST.
            content_type: Optional MIME type for the request body.
            byte_range: Optional inclusive byte interval for an HTTP Range request.

        Returns:
            Exact response body bytes.

        Raises:
            RuntimeError: If every attempt fails or a range server returns incomplete bytes.
        """
        headers = {"Accept": "application/json", "User-Agent": "WISDOM-DNA-dataset/0.6"}
        if content_type is not None:
            headers["Content-Type"] = content_type
        if byte_range is not None:
            headers["Range"] = f"bytes={byte_range[0]}-{byte_range[1]}"

        for attempt in range(self.retries):
            with self.lock:
                remaining = self.request_interval - (time.monotonic() - self.last_request)
                if remaining > 0.0:
                    time.sleep(remaining)
                self.last_request = time.monotonic()
            try:
                request = urllib.request.Request(url, data=data, headers=headers)
                with urllib.request.urlopen(request, timeout=90.0) as response:
                    content = response.read()
                if byte_range is not None and len(content) != byte_range[1] - byte_range[0] + 1:
                    raise RuntimeError("public server returned incomplete range bytes")
                return content
            except (OSError, RuntimeError, urllib.error.HTTPError) as error:
                if attempt + 1 == self.retries:
                    raise RuntimeError(
                        f"request failed after {self.retries} attempts: {url}"
                    ) from error
                time.sleep(min(30.0, 2.0**attempt))
        raise RuntimeError("unreachable HTTP retry state")

    @staticmethod
    def atomic_write(path: Path, content: bytes) -> None:
        """Publish complete bytes using flush, fsync, and atomic replacement.

        Args:
            path: Final cache path.
            content: Exact validated bytes.

        Raises:
            OSError: If writing, synchronization, or replacement fails.
        """
        temporary = path.with_name(f".{path.name}.{os.getpid()}.{uuid4().hex}.tmp")
        try:
            with temporary.open("wb") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    @staticmethod
    def _digest(content: bytes, expected_sha256: str | None) -> bool:
        """Compare bytes with an optional SHA-256 pin.

        Args:
            content: Candidate source bytes.
            expected_sha256: Optional lowercase digest.

        Returns:
            True when no pin was supplied or the digest agrees exactly.
        """
        return expected_sha256 is None or hashlib.sha256(content).hexdigest() == expected_sha256

    @staticmethod
    def _zip_index(
        directory: bytes,
    ) -> dict[str, tuple[int, int, int, int, int]]:
        """Parse a complete ZIP central directory including ZIP64 size/offset extensions.

        Args:
            directory: Exact central-directory bytes.

        Returns:
            Member name to compressed size, uncompressed size, local offset, CRC-32, and method.

        Raises:
            RuntimeError: If any directory header or ZIP64 extension is malformed.
        """
        output: dict[str, tuple[int, int, int, int, int]] = {}
        offset = 0
        while offset < len(directory):
            if directory[offset : offset + 4] != b"PK\x01\x02":
                raise RuntimeError("ZIP central directory contains an invalid member header")
            try:
                fields = struct.unpack_from("<4s6H3L5H2L", directory, offset)
            except struct.error as error:
                raise RuntimeError("ZIP central directory member is truncated") from error
            method                          = int(fields[4])
            crc32                           = int(fields[7])
            compressed_size, uncompressed_size = int(fields[8]), int(fields[9])
            name_length, extra_length, comment_length = map(int, fields[10:13])
            local_offset = int(fields[-1])
            name_start   = offset + 46
            extra_start  = name_start + name_length
            name = directory[name_start:extra_start].decode("utf-8")
            extra = directory[extra_start : extra_start + extra_length]
            if 0xFFFFFFFF in (compressed_size, uncompressed_size, local_offset):
                values: list[int] = []
                cursor = 0
                while cursor + 4 <= len(extra):
                    kind, size = struct.unpack_from("<HH", extra, cursor)
                    body       = extra[cursor + 4 : cursor + 4 + size]
                    if kind == 1:
                        values = list(
                            struct.unpack(f"<{len(body) // 8}Q", body[: len(body) // 8 * 8])
                        )
                        break
                    cursor += 4 + size
                value_index = 0
                if uncompressed_size == 0xFFFFFFFF:
                    uncompressed_size = values[value_index]
                    value_index += 1
                if compressed_size == 0xFFFFFFFF:
                    compressed_size = values[value_index]
                    value_index += 1
                if local_offset == 0xFFFFFFFF:
                    local_offset = values[value_index]
            output[name] = (
                compressed_size,
                uncompressed_size,
                local_offset,
                crc32,
                method,
            )
            offset += 46 + name_length + extra_length + comment_length
        return output
