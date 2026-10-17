"""Portable protein galleries for LambdaForge's isolated project-report sections."""

# ruff: noqa: E501 -- browser-native presentation remains a readable embedded template.

from __future__ import annotations

import re
import gzip
import json
import base64
import argparse
import webbrowser

from typing import Any
from pathlib import Path
from collections import defaultdict
from plotly.offline import get_plotlyjs
from collections.abc import Mapping, Sequence
from wisdom.preprocessing.common.structure.ProteinVisualizer import ProteinVisualizer


class ProteinReportPage:
    """Package existing inspectors with one shared library and a bounded offline navigator."""

    def __init__(self, maximum_mib: float = 4.0) -> None:
        """Set the presentation budget without changing scientific evaluation coverage.

        Args:
            maximum_mib: Maximum UTF-8 document size in MiB, positive and at most LambdaForge's
                16 MiB/document limit. The framework separately enforces 64 MiB/report.

        Raises:
            ValueError: If the budget exceeds the framework's document limit or is non-positive.
        """
        if not 0.0 < maximum_mib <= 16.0:
            raise ValueError("protein report maximum_mib must lie in (0,16]")
        self.maximum_bytes = int(maximum_mib * 1024 * 1024)

    @staticmethod
    def _encode(text: str) -> str:
        """Encode deterministic gzip data for a modern browser's DecompressionStream.

        Args:
            text: UTF-8 document or JavaScript owned by the shared WISDOM renderer.

        Returns:
            ASCII base64 containing gzip bytes; no files or network requests are needed.
        """
        return base64.b64encode(gzip.compress(text.encode("utf-8"), mtime=0)).decode("ascii")

    @staticmethod
    def _serialize(value: Any) -> str:
        """Serialize presentation data exactly as embedded in the script-safe document.

        Args:
            value: JSON-compatible gallery payload, entry, or metadata.

        Returns:
            Compact ASCII JSON with less-than signs escaped to prevent closing the
            enclosing script. The same representation measures and publishes bytes.
        """
        return json.dumps(value, ensure_ascii=True, separators=(",", ":")).replace(
            "<", "\\u003c"
        )

    def render(
        self,
        documents: Sequence[Mapping[str, Any]],
        context  : Mapping[str, Any],
        reason   : str = "No protein visualizations were generated for this Run.",
    ) -> tuple[str, dict[str, Any]]:
        """Build a searchable gallery reusing exactly the preprocessing inspector.

        Documents contain ``identifier``, ``split``, ``label`` and ``html``. The HTML must come
        from ProteinVisualizer.render(plotly_script=False), so the library is packaged only once.
        A deterministic round-robin over model/seed/split/label strata bounds presentation without
        preferring one class or checkpoint. Recorded pooling families are interleaved in pairs so
        a large early family cannot consume the budget before another family is offered a slot.
        Omitted viewers remain explicit; metrics are not filtered.
        The budget includes the complete HTML shell, escaped context, all omission identifiers,
        shared library and accepted entries; metadata never relies on a fixed-size allowance.

        Args:
            documents: Generated, already point-aligned inspectors; no NPZ is read here.
            context: JSON-compatible Run/checkpoint metadata displayed without HTML interpolation.
            reason: Empty-state explanation, including disabled or pruned visualization.

        Returns:
            Self-contained HTML and a JSON-compatible included/omitted/byte-count summary.

        Raises:
            ValueError: If a document lacks its renderer-owned HTML or the shell cannot fit.
        """
        # One compressed Plotly bundle serves whichever protein is currently open. Data and
        # templates are decompressed lazily; no nested iframe or relative asset escapes LF's CSP.

        library = self._encode(get_plotlyjs()) if documents else ""
        configurations = [document.get("configuration", {}) for document in documents]
        varying_keys = [
            key for key in sorted(set.intersection(*(set(row) for row in configurations)))
            if len({self._serialize(row[key]) for row in configurations}) > 1
            and key not in {"seed", "init_seed", "training_seed", "batch_size", "data_workers"}
            and not key.startswith((
                "pooling_", "topk_", "attention_", "log_sum_exp_", "autopool_", "gem_",
                "max_mean_", "regional_", "multiscale_regional_", "surface_visualization",
            ))
        ] if configurations else []
        buckets : dict[tuple[str, str, str, int], list[Mapping[str, Any]]] = defaultdict(list)
        for document in documents:
            buckets[(str(document.get("variant", "")), str(document.get("seed", "")),
                     str(document["split"]), int(document["label"]))].append(document)
        for bucket in buckets.values():
            bucket.sort(key=lambda item: str(item["identifier"]))

        ordered: list[Mapping[str, Any]] = []
        row = 0
        while any(row < len(bucket) for bucket in buckets.values()):
            for key in sorted(buckets):
                bucket = buckets[key]
                if row >= len(bucket):
                    continue
                ordered.append(bucket[row])
            row += 1

        # Family coverage matters for comparisons, not numerical ranking. Preserve paired
        # presentation order within each family; metadata-free legacy documents keep their order.

        if ordered and all(document.get("configuration", {}).get("pooling_type") for document in ordered):
            families: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
            for document in ordered:
                families[str(document["configuration"]["pooling_type"])].append(document)
            ordered = []
            for offset in range(0, max(map(len, families.values())), 2):
                for family in sorted(families):
                    ordered.extend(families[family][offset:offset + 2])

        # Reserve the actual shell, context and ALL omission identifiers, not a fixed
        # allowance. Large reviews can have thousands of omissions. Keeping this full
        # list in the reservation is conservative: accepted viewers remove their IDs
        # from it at publication, so final metadata can only become smaller.

        identifiers = [
            f"{document.get('variant', '')} seed={document.get('seed')} "
            f"{document['split']}/{document['identifier']}"
            for document in ordered
        ]
        empty_reason = (
            "The configured report byte budget cannot fit a protein viewer. Use "
            "prediction-only content, reduce the display point count, "
            "or increase the report byte budget."
            if documents else reason
        )
        # Reserve the larger of both explanations because the final empty/nonempty
        # state is unknown here. ASCII escaping also counts Unicode and script markers.

        reserved_reason = max((reason, empty_reason), key=lambda value: len(self._serialize(value)))
        reserved = {
            "documents": [],
            "plotly":    library,
            "context":   dict(context),
            "omitted":   identifiers,
            "reason":    reserved_reason,
        }
        used = len(self._html(self._serialize(reserved)).encode("utf-8"))
        entries: list[dict[str, Any]] = []
        omitted: list[str] = []
        for document, identifier in zip(ordered, identifiers, strict=True):
            configuration = document.get("configuration", {})
            caption = (
                self.configuration_label(configuration) if configuration
                else str(document.get("variant", "Source model"))
            )
            differences = ", ".join(f"{key}={configuration[key]}" for key in varying_keys)
            if differences:
                caption += f" · {differences}"
            entry = {
                "identifier": str(document["identifier"]),
                "split":      str(document["split"]),
                "label":      int(document["label"]),
                "content":    self._encode(str(document["html"])),
                "variant":    str(document.get("variant", "")),
                "seed":       document.get("seed"),
                "details":    document.get("details", {}),
                "configuration": document.get("configuration", {}),
                "configuration_label": caption,
                "run_id": document.get("run_id"),
            }
            # One extra byte safely reserves the comma between accepted entries.

            size = len(self._serialize(entry).encode("utf-8")) + 1
            if used + size <= self.maximum_bytes:
                entries.append(entry)
                used += size
            else:
                omitted.append(identifier)
        if not entries:
            library = ""
            reason = empty_reason

        payload = self._serialize({
            "documents": entries,
            "plotly":    library,
            "context":   dict(context),
            "omitted":   omitted,
            "reason":    reason,
        })
        page = self._html(payload)
        page_size = len(page.encode("utf-8"))
        if page_size > self.maximum_bytes:
            raise ValueError(
                f"protein report shell exceeds its configured byte budget: "
                f"shell and metadata require {page_size} bytes, "
                f"configured limit is {self.maximum_bytes} bytes"
            )
        return page, {
            "included_proteins": len(entries),
            "omitted_proteins":  omitted,
            "size_bytes":        page_size,
        }

    @staticmethod
    def configuration_label(parameters: Mapping[str, Any]) -> str:
        """Describe the pooling and its active controls without exposing arbitrary Trial numbers.

        Args:
            parameters: Recorded scientific Run parameters, not inferred filenames.

        Returns:
            A readable pooling caption including configured pooling-specific parameters.
        """
        pooling = str(parameters.get("pooling_type", "source model"))
        relevant = {
            "topk": ("topk_",),
            "log_sum_exp": ("log_sum_exp_", "pooling_curriculum_", "lse_curriculum_"),
            "attention": ("attention_",),
            "autopool": ("autopool_",),
            "gem": ("gem_",),
            "max_mean": ("max_mean_",),
            "local_mean_max": ("regional_",),
            "multiscale_regional_max": ("regional_", "multiscale_regional_"),
        }
        keys = relevant.get(pooling, ())
        # Frozen checkpoints may record defaults for inactive families as well. Do not
        # describe those as scientific controls of the currently active pooling family.
        selected = {
            key: value for key, value in parameters.items()
            if key.startswith(keys) or (key == "pooling_area_mode" and pooling not in {
                "max", "local_mean_max", "multiscale_regional_max",
            })
        }
        if parameters.get("log_sum_exp_mode") == "fixed":
            selected = {key: value for key, value in selected.items() if "curriculum" not in key}
        elif parameters.get("log_sum_exp_mode") == "curriculum":
            selected.pop("log_sum_exp_beta", None)
        controls = ", ".join(f"{key.removeprefix('pooling_')}={value}" for key, value in selected.items())
        return pooling.replace("_", " ").title() + (f" · {controls}" if controls else "")

    @staticmethod
    def upgrade(source: str, audits: Sequence[Mapping[str, Any]] = ()) -> list[dict[str, Any]]:
        """Recover existing maps for a derived report without inference or artifact mutation.

        Args:
            source: An existing WISDOM gallery HTML document containing its embedded payload.
            audits: Optional exact source review audit records with Run parameters and metrics.

        Returns:
            Decoded inspectors with refreshed shared UI and authoritative configuration metadata.

        Raises:
            ValueError: If the gallery or shared inspector cannot be read as a supported document.
        """
        match = re.search(
            r'<script id="wisdom-gallery-data" type="application/json">(.*?)</script>',
            source, re.DOTALL,
        )
        if match is None:
            raise ValueError("source is not a WISDOM protein gallery")
        payload = json.loads(match.group(1))
        rows = []
        for entry in payload["documents"]:
            row = dict(entry)
            row["html"] = ProteinVisualizer.refresh(
                gzip.decompress(base64.b64decode(row.pop("content"))).decode("utf-8")
            )
            # Older captions carried a one-based Trial index; use it only as an audit join,
            # never as the new scientific configuration caption.
            trial = re.search(r"Trial (\d+)", row.get("variant", ""))
            for audit in audits:
                if (
                    audit.get("seed") == row.get("seed")
                    and audit.get("split") == row["split"]
                    and (audit.get("run_id") == row.get("run_id") if row.get("run_id")
                         else trial is not None and audit.get("trial") == int(trial.group(1)))
                ):
                    row["configuration"] = audit["parameters"]
                    row["details"] = {
                        **audit.get("proteins", {}).get(row["identifier"], {}),
                        **row.get("details", {}),
                    }
                    row["run_id"] = audit["run_id"]
                    break
            rows.append(row)
        return rows

    @staticmethod
    def _html(payload: str) -> str:
        """Embed the shared offline catalogue assets and script-safe data.

        Args:
            payload: Compact escaped JSON containing compressed inspectors and metadata.

        Returns:
            A complete offline protein-first gallery; only the opened map owns WebGL.
        """
        assets = Path(__file__).parent
        style = (assets / "protein_gallery.css").read_text(encoding="utf-8")
        script = (assets / "protein_gallery.js").read_text(encoding="utf-8").replace(
            "@@CASE_DESCRIPTIONS@@", ProteinReportPage._serialize(ProteinVisualizer.CASE_DESCRIPTIONS)
        )
        return (
            '<!doctype html><html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>WISDOM proteins</title><style id="wisdom-gallery-style">'
            + style + '</style></head><body class="wisdom-gallery">'
            '<p>Loading WISDOM gallery…</p>'
            '<script id="wisdom-gallery-data" type="application/json">'
            + payload + '</script><script>' + script + '</script></body></html>'
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Refresh saved WISDOM maps without inference.")
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--open", action="store_true")
    arguments = parser.parse_args()
    if arguments.output.resolve() == arguments.source.resolve():
        parser.error("output must be a new derived document, not the immutable source artifact")
    source = arguments.source.read_text(encoding="utf-8")
    audits = json.loads(arguments.audit.read_text()) if arguments.audit else []
    documents = ProteinReportPage.upgrade(source, audits)
    payload_match = re.search(
        r'<script id="wisdom-gallery-data" type="application/json">(.*?)</script>',
        source, re.DOTALL,
    )
    original = json.loads(payload_match.group(1)) if payload_match else {}
    page, summary = ProteinReportPage(maximum_mib=16).render(
        documents, {
            **original.get("context", {}),
            "source": arguments.source.name, "derived_without_inference": True,
            "source_omitted": [
                *original.get("context", {}).get("source_omitted", []),
                *original.get("omitted", []),
            ],
        },
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(page, encoding="utf-8")
    print(f"Updated gallery: {arguments.output.resolve()} ({summary['included_proteins']} maps)")
    if arguments.open:
        webbrowser.open(arguments.output.resolve().as_uri())
