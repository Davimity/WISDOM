"""Persistent numerical and interactive reports for trained WISDOM surface maps."""

# ruff: noqa: E501 -- embedded HTML/CSS/JavaScript remains readable as one report template.

from __future__ import annotations

import os
import re
import html
import json
import torch
import numpy as np

from typing import Any
from pathlib import Path
from torch import Tensor
from collections.abc import Mapping
from plotly.offline import get_plotlyjs

from wisdom.data.WisdomDataset import WisdomDataset
from wisdom.evaluation.PointCloudExporter import PointCloudExporter
from wisdom.preprocessing.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.structure.ProteinVisualizer import ProteinVisualizer


class SurfacePredictionReport:
    """Collect one evaluated split and publish point-aligned predictions and shared viewers."""

    def __init__(
        self,
        dataset               : WisdomDataset,
        output_root           : Path,
        split                 : str,
        prediction_threshold  : float = 0.5,
        maximum_visualizations: int = 12,
        save_predictions      : bool = False,
    ) -> None:
        """Bind predictions to the exact ordered proteins used by one evaluation loader.

        Args:
            dataset: Sidecar-enabled WISDOM dataset whose record order defines expected proteins.
            output_root: Managed Run directory receiving all split reports.
            split: Human-facing ``validation`` or ``test`` partition name.
            prediction_threshold: Initial probability cutoff for hard prediction arrays and HTML.
            maximum_visualizations: Maximum balanced HTML/PLY sample. Zero renders every protein;
                prediction arrays remain in memory long enough to build those viewers.
            save_predictions: Also write one compact point-aligned NPZ per protein. False avoids
                duplicating values already embedded in selected HTML viewers.

        Raises:
            ValueError: If the threshold, limit, split, or dataset records are invalid.
        """
        if not 0.0 <= prediction_threshold <= 1.0:
            raise ValueError("surface prediction threshold must lie in [0,1]")
        if maximum_visualizations < 0:
            raise ValueError("maximum surface visualizations cannot be negative")
        if not split.strip():
            raise ValueError("surface prediction split cannot be empty")

        records = {
            str(record[3]): (
                Path(record[0]),
                Path(record[1]) if record[1] is not None else None,
                int(record[2]),
            )
            for record in dataset.records
        }
        if len(records) != len(dataset.records):
            raise ValueError("surface prediction dataset contains duplicate identifiers")

        self.records                = records
        self.output_root            = output_root
        self.split                  = split
        self.prediction_threshold   = prediction_threshold
        self.maximum_visualizations = maximum_visualizations
        self.save_predictions       = save_predictions
        self.predictions            : dict[str, np.ndarray] = {}

    def collect(
        self,
        batch : Mapping[str, Any],
        output: Mapping[str, Tensor],
    ) -> None:
        """Separate one disjoint model batch back into immutable per-protein point order.

        Args:
            batch: Collated evaluation batch containing identifiers and ``surface_ptr[B+1]``.
            output: Model outputs containing unnormalized ``surface_logits[M]``.

        Raises:
            ValueError: If identifiers, boundaries, logits, or repeated records are inconsistent.
        """
        identifiers = batch.get("identifier")
        boundaries  = batch.get("surface_ptr")
        logits      = output.get("surface_logits")
        if not isinstance(identifiers, list) or not isinstance(boundaries, Tensor):
            raise ValueError("surface prediction batches require identifiers and surface_ptr")
        if not isinstance(logits, Tensor) or logits.ndim != 1:
            raise ValueError("surface prediction output requires surface_logits[M]")

        pointers      = boundaries.detach().cpu().tolist()
        probabilities = torch.sigmoid(logits.detach()).float().cpu().numpy()
        if len(pointers) != len(identifiers) + 1 or pointers[0] != 0 or pointers[-1] != len(logits):
            raise ValueError("surface prediction boundaries disagree with the evaluated batch")

        for index, identifier in enumerate(identifiers):
            name = str(identifier)
            if name not in self.records:
                raise ValueError(f"surface prediction returned unexpected protein {name!r}")
            if name in self.predictions:
                raise ValueError(f"surface prediction repeated protein {name!r}")

            start = int(pointers[index])
            stop  = int(pointers[index + 1])
            self.predictions[name] = probabilities[start:stop].copy()

    def publish(
        self,
        metrics   : Mapping[str, float | None],
        best_epoch: int,
    ) -> dict[str, Any]:
        """Write an optional numerical map set and a bounded gallery using ProteinVisualizer.

        Args:
            metrics: Aggregate evaluation-only surface metrics for this exact prediction pass.
            best_epoch: Validation-selected checkpoint epoch that produced the predictions.

        Returns:
            JSON-compatible split summary with counts and relative report paths.

        Raises:
            ValueError: If evaluation omitted a protein or a prediction does not match its base
                surface length.
            OSError: If an NPZ, HTML, PLY, or report cannot be written.
        """
        missing = sorted(self.records.keys() - self.predictions.keys())
        extra   = sorted(self.predictions.keys() - self.records.keys())
        if missing or extra:
            raise ValueError(
                f"surface prediction coverage differs from {self.split}: "
                f"missing={missing[:10]}, extra={extra[:10]}"
            )

        split_root   = self.output_root / self.split
        protein_root = split_root / "proteins"
        selected     = set(self._selected_identifiers())
        visualizer   = ProteinVisualizer()
        exporter     = PointCloudExporter()
        reports      : list[dict[str, Any]] = []

        self.output_root.mkdir(parents=True, exist_ok=True)
        plotly = self.output_root / "plotly.min.js"
        if not plotly.is_file():
            ProteinArchive.write_text(plotly, get_plotlyjs())

        # Viewer mode writes only the selected HTML/PLY files. Full mode additionally retains one
        # compact prediction NPZ per protein for downstream numerical analysis.

        for identifier in sorted(self.records):
            base, annotation, label = self.records[identifier]
            probabilities          = self.predictions[identifier]
            with np.load(base, allow_pickle=False) as archive:
                positions = archive["surface_positions"]
            if probabilities.shape != (len(positions),):
                raise ValueError(
                    f"surface prediction for {identifier!r} has {len(probabilities)} values but "
                    f"the immutable surface has {len(positions)} points"
                )

            safe_name       = re.sub(r"[^A-Za-z0-9_.-]+", "-", identifier).strip(".-")
            prediction_path = split_root / "predictions" / f"{safe_name}.npz"
            hard_prediction = probabilities >= self.prediction_threshold
            if self.save_predictions:
                self._write_prediction(
                    prediction_path,
                    identifier,
                    probabilities,
                    hard_prediction,
                    best_epoch,
                )

            report: dict[str, Any] = {
                "identifier":        identifier,
                "split":             self.split,
                "label":             int(label),
                "surface_points":    len(probabilities),
                "prediction":        (
                    prediction_path.relative_to(self.output_root).as_posix()
                    if self.save_predictions
                    else None
                ),
                "html":              None,
                "ply":               None,
                "diagnostics":       None,
            }
            if identifier in selected:
                channels = {
                    "model_prediction_probability": probabilities,
                    "model_prediction_hard":        hard_prediction.astype(np.uint8),
                }
                html_path = protein_root / f"{safe_name}.html"
                ply_path  = protein_root / f"{safe_name}.ply"
                diagnostics = visualizer.visualize(
                    base,
                    html_path,
                    identifier,
                    annotation           = annotation,
                    protein_label        = int(label),
                    partitions           = {"split": self.split},
                    plotly_script        = "../../plotly.min.js",
                    additional_channels  = channels,
                    prediction_threshold = self.prediction_threshold,
                )
                surface_channels = visualizer.surface_channels(
                    base,
                    annotation,
                    channels,
                )
                exporter.export(ply_path, positions, surface_channels)
                report.update(
                    html        = html_path.relative_to(self.output_root).as_posix(),
                    ply         = ply_path.relative_to(self.output_root).as_posix(),
                    diagnostics = dict(diagnostics),
                )
            reports.append(report)

        summary = {
            "split":                   self.split,
            "best_epoch":              best_epoch,
            "prediction_threshold":    self.prediction_threshold,
            "prediction_npz_enabled":  self.save_predictions,
            "predicted_proteins":      len(reports),
            "visualized_proteins":     len(selected),
            "surface_metrics":         dict(metrics),
            "members":                 reports,
        }
        ProteinArchive.write_text(
            split_root / "surface-predictions.json",
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
        )
        ProteinArchive.write_text(split_root / "index.html", self._index_html(reports, metrics))
        return {
            "split":               self.split,
            "predicted_proteins":  len(reports),
            "visualized_proteins": len(selected),
            "index":               (split_root / "index.html").relative_to(self.output_root).as_posix(),
            "manifest":            (split_root / "surface-predictions.json").relative_to(
                self.output_root
            ).as_posix(),
        }

    def _selected_identifiers(self) -> tuple[str, ...]:
        """Choose a deterministic class-balanced HTML sample without limiting raw predictions.

        Returns:
            Ordered protein identifiers alternating between available global target classes.
        """
        buckets = {
            label: sorted(
                identifier
                for identifier, (_base, _annotation, record_label) in self.records.items()
                if int(record_label) == label
            )
            for label in (0, 1)
        }
        limit    = self.maximum_visualizations or len(self.records)
        selected: list[str] = []
        row      = 0
        while len(selected) < limit:
            added = False
            for label in (0, 1):
                if row < len(buckets[label]):
                    selected.append(buckets[label][row])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            row += 1
        return tuple(selected)

    @staticmethod
    def publish_index(
        output_root: Path,
        reports    : list[dict[str, Any]],
    ) -> Path:
        """Write the artifact landing page after all requested evaluation splits finish.

        Args:
            output_root: Managed surface-prediction artifact directory.
            reports: Ordered validation/test summaries returned by :meth:`publish`.

        Returns:
            Path to the resulting top-level ``index.html``.

        Raises:
            ValueError: If no completed split report is available.
            OSError: If the landing page cannot be written.
        """
        if not reports:
            raise ValueError("surface prediction index requires at least one split report")

        cards = "".join(
            f"<a class='card' href='{html.escape(str(report['index']))}'><strong>"
            f"{html.escape(str(report['split']).title())}</strong><span>"
            f"{int(report['predicted_proteins'])} evaluated proteins · "
            f"{int(report['visualized_proteins'])} interactive viewers</span></a>"
            for report in reports
        )
        page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>WISDOM surface predictions</title><style>:root{{color-scheme:dark;font:15px Inter,system-ui,sans-serif;background:#080c16;color:#e7edf7}}body{{max-width:900px;margin:auto;padding:48px 24px}}p{{color:#a9b7ca;line-height:1.55}}.grid{{display:grid;gap:12px;margin-top:24px}}.card{{display:flex;justify-content:space-between;gap:18px;padding:18px;background:#111827;border:1px solid #2d3b50;border-radius:12px;color:#70e5f0;text-decoration:none}}.card span{{color:#a9b7ca}}@media(max-width:600px){{.card{{display:grid}}}}</style></head><body><h1>WISDOM surface predictions</h1><p>These maps come from the checkpoint selected using protein-level validation only. Surface ground truth never contributes to the loss or checkpoint choice; during architecture development, its aggregate validation metrics contribute to the declared WISDOM HPO score.</p><div class="grid">{cards}</div></body></html>"""
        path = output_root / "index.html"
        ProteinArchive.write_text(path, page)
        return path

    def _write_prediction(
        self,
        path            : Path,
        identifier      : str,
        probabilities   : np.ndarray,
        hard_prediction : np.ndarray,
        best_epoch      : int,
    ) -> None:
        """Atomically persist one pickle-free point-aligned prediction sidecar.

        Args:
            path: Final compressed NPZ path.
            identifier: Dataset member ID represented by the arrays.
            probabilities: Finite sigmoid probabilities with shape ``[M]``.
            hard_prediction: Boolean threshold decisions with shape ``[M]``.
            best_epoch: Checkpoint epoch that produced the map.

        Raises:
            ValueError: If the arrays are misaligned, non-finite, or outside probability bounds.
            OSError: If the atomic file write fails.
        """
        values = np.asarray(probabilities, dtype=np.float32)
        hard   = np.asarray(hard_prediction, dtype=np.bool_)
        if values.ndim != 1 or hard.shape != values.shape:
            raise ValueError("surface probability and hard prediction must share shape [M]")
        if not np.isfinite(values).all() or np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("surface prediction probabilities must be finite values in [0,1]")

        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
        try:
            np.savez_compressed(
                temporary,
                identifier=np.asarray(identifier),
                surface_prediction_probability=values,
                surface_prediction_hard=hard,
                prediction_threshold=np.asarray(self.prediction_threshold, dtype=np.float32),
                best_epoch=np.asarray(best_epoch, dtype=np.int32),
            )
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _index_html(
        self,
        reports: list[dict[str, Any]],
        metrics: Mapping[str, float | None],
    ) -> str:
        """Build a searchable split landing page linking raw and interactive surface results.

        Args:
            reports: Ordered prediction records for every evaluated protein.
            metrics: Aggregate surface metrics computed from the identical model pass.

        Returns:
            Complete UTF-8 HTML document with metric summary and artifact links.
        """
        metric_rows = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{'unavailable' if value is None else f'{value:.6g}'}</td></tr>"
            for name, value in sorted(metrics.items())
        )
        member_rows: list[str] = []
        for report in reports:
            prediction_path = (
                Path(str(report["prediction"])).relative_to(self.split).as_posix()
                if report["prediction"]
                else None
            )
            html_path = (
                Path(str(report["html"])).relative_to(self.split).as_posix()
                if report["html"]
                else None
            )
            member_rows.append(
                "<tr data-id='{}' data-label='{}'><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
                    html.escape(str(report["identifier"]).lower()),
                    report["label"],
                    html.escape(str(report["identifier"])),
                    report["label"],
                    f"<a href='{html.escape(html_path)}'>Interactive HTML</a>"
                    if html_path is not None
                    else "Numerical map only",
                    f"<a href='{html.escape(prediction_path)}'>Prediction NPZ</a>"
                    if prediction_path is not None
                    else "NPZ disabled",
                )
            )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>WISDOM {html.escape(self.split)} surface predictions</title><style>
:root{{color-scheme:dark;font:15px Inter,system-ui,sans-serif;background:#080c16;color:#e7edf7}}
body{{max-width:1180px;margin:auto;padding:38px 22px}}h1{{margin-bottom:6px}}
p{{color:#a9b7ca;line-height:1.5}}.grid{{display:grid;grid-template-columns:minmax(260px,.65fr) 1.35fr;gap:18px}}
.card{{background:#111827;border:1px solid #2d3b50;border-radius:12px;padding:15px;overflow:auto}}
input,select{{padding:8px;background:#1b2638;color:#e7edf7;border:1px solid #3b4a61;border-radius:8px;margin:0 8px 12px 0}}
table{{border-collapse:collapse;width:100%}}th,td{{padding:8px 10px;border-bottom:1px solid #27364b;text-align:left}}
a{{color:#6fe7f2;text-decoration:none}}@media(max-width:820px){{.grid{{grid-template-columns:1fr}}}}
</style></head><body><h1>WISDOM {html.escape(self.split)} surface predictions</h1>
<p>Interactive pages use the same structural viewer as preprocessing and are limited to a
deterministic class-balanced sample. Companion prediction NPZ files are optional. The
probability threshold inside each page changes the hard prediction instantly; it does not alter
the stored probabilities or metrics.</p><div class="grid"><section class="card">
<h2>Aggregate surface metrics</h2><table><tbody>{metric_rows}</tbody></table></section>
<section class="card"><h2>Proteins</h2><input id="search" type="search" placeholder="Search protein ID">
<select id="label"><option value="">Both labels</option><option value="0">Negative</option>
<option value="1">Positive</option></select><table><thead><tr><th>Protein</th><th>Label</th>
<th>Viewer</th><th>Numerical output</th></tr></thead><tbody>{''.join(member_rows)}</tbody></table></section></div>
<script>const rows=[...document.querySelectorAll('tbody tr[data-id]')],search=document.getElementById('search'),label=document.getElementById('label');function filter(){{const query=search.value.trim().toLowerCase();for(const row of rows)row.hidden=Boolean((query&&!row.dataset.id.includes(query))||(label.value&&row.dataset.label!==label.value))}}search.addEventListener('input',filter);label.addEventListener('change',filter);</script>
</body></html>"""
