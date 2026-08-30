"""LambdaForge Work for dataset-native interactive WISDOM protein inspection."""

# ruff: noqa: E501 -- embedded HTML/CSS/JavaScript remains readable as a complete page template.

from __future__ import annotations

import re
import json
import numpy as np
import lambdaforge as lf

from typing import Any
from pathlib import Path
from collections import defaultdict
from collections.abc import Sequence
from plotly.offline import get_plotlyjs

from lambdaforge.data import DatasetIndex, DatasetMember
from wisdom.evaluation.PointCloudExporter import PointCloudExporter
from wisdom.preprocessing.structure.ProteinArchive import ProteinArchive
from wisdom.preprocessing.structure.ProteinVisualizer import ProteinVisualizer


class Visualization(lf.Work):
    """Render a bounded, stratified sample from one immutable WISDOM DatasetVersion."""

    def run(
        self,
        skip                  : bool            = True,
        dataset               : Path | None     = None,
        output_directory      : str             = "../data/dna/visualizations",
        overwrite_output      : bool            = True,
        identifiers           : Sequence[str]   = (),
        splits                : Sequence[str]   = ("train", "validation", "test"),
        labels                : Sequence[int]   = (0, 1),
        maximum_proteins      : int             = 12,
        maximum_surface_points: int             = 6000,
        maximum_mesh_points   : int             = 2500,
        maximum_edges         : int             = 5000,
        normal_stride         : int             = 25,
        normal_length         : float           = 1.5,
        mesh_alpha            : float           = 4.0,
        maximum_vdw_atoms     : int             = 1500,
        verbose               : bool            = False,
    ) -> dict[str, Any]:
        """Create interactive HTML and portable PLY views without changing the dataset.

        Explicit identifiers are rendered in their requested order. Otherwise, members are sorted
        deterministically inside ``(split, label)`` strata and selected round-robin, so a small
        gallery does not accidentally show only one class or partition. Each HTML uses a shared
        local Plotly WebGL library, exposes point/mesh/atom/edge/normal/backbone layers, and offers
        every supported surface and atom scalar as a colour channel. Mesh faces are computed once
        before serialization, keeping browser-side layer changes responsive. The companion PLY
        preserves complete surface point order and scalar fields for external scientific viewers.

        Args:
            skip: Return immediately without resolving or rendering dataset members when true.
            dataset: LambdaForge-resolved immutable DatasetVersion placement containing
                ``index.jsonl`` and member assets.
            output_directory: Conventional project-relative directory receiving an atomic copy of
                the managed visualization artifact.
            overwrite_output: Replace a different existing visualization directory after success.
            identifiers: Optional exact member IDs to render; empty selects a stratified sample.
            splits: Dataset ``split`` partition values eligible for automatic sampling.
            labels: Protein-level ``dna_binding`` targets eligible for automatic sampling.
            maximum_proteins: Largest automatic sample size; zero renders every eligible member
                and explicit identifiers are never truncated.
            maximum_surface_points: Largest deterministic point subset embedded in each HTML.
            maximum_mesh_points: Largest point subset supplied to diagnostic alpha-complex meshing.
            maximum_edges: Largest atomic or bounded surface-neighbour subset drawn per HTML.
            normal_stride: Draw one normal for every this many displayed surface points.
            normal_length: Display length of each normal vector in ångströms.
            mesh_alpha: Largest retained alpha-complex tetrahedron radius in ångströms.
            maximum_vdw_atoms: Largest deterministic atom subset rendered as physical-radius
                icosahedra; every atom remains available in the ordinary marker layer.
            verbose: Log every rendered member in addition to normal progress summaries.

        Returns:
            JSON-compatible skip state or rendered member count, IDs, output name, and diagnostic
            failure count.

        Raises:
            ValueError: If the dataset/index, filters, requested IDs, assets, or visualization
                parameters are missing or inconsistent.
            OSError: If dataset assets cannot be read or managed outputs cannot be written.
        """
        if skip:
            self.log("Visualization skipped; no dataset members or output files were read")
            return {"skipped": True}

        if dataset is None:
            raise ValueError("dataset is required when visualization runs")
        if maximum_proteins < 0:
            raise ValueError("maximum_proteins cannot be negative")

        dataset_root = Path(dataset).resolve()
        index_path   = dataset_root / "index.jsonl"
        if not index_path.is_file():
            raise ValueError("managed WISDOM dataset root must contain index.jsonl")

        # Select exact members from the immutable logical index; physical dataset paths never enter
        # YAML and the Work neither modifies nor republishes the DatasetVersion.

        members  = tuple(DatasetIndex(index_path))
        selected = self._select_members(
            members,
            identifiers      = identifiers,
            splits           = splits,
            labels           = labels,
            maximum_proteins = maximum_proteins,
        )
        self.log(
            f"Rendering {len(selected)} of {len(members)} dataset members as HTML and PLY"
        )

        # One shared local Plotly bundle keeps every protein page offline-capable without embedding
        # several megabytes of JavaScript repeatedly.

        output = self.outputs.directory(
            "protein-visualizations",
            role       = "visualization",
            publish_to = output_directory,
            overwrite  = overwrite_output,
        )
        output_root = Path(output)
        (output_root / "plotly.min.js").write_text(get_plotlyjs(), encoding="utf-8")

        visualizer = ProteinVisualizer(
            max_surface_points = maximum_surface_points,
            max_mesh_points    = maximum_mesh_points,
            max_edges          = maximum_edges,
            normal_stride      = normal_stride,
            normal_length      = normal_length,
            mesh_alpha         = mesh_alpha,
            max_vdw_atoms      = maximum_vdw_atoms,
        )
        exporter = PointCloudExporter()

        reports: list[dict[str, Any]] = []
        for member_index, member in enumerate(selected, start=1):
            report = self._render_member(
                dataset_root,
                output_root,
                member,
                visualizer,
                exporter,
            )
            reports.append(report)

            if verbose or member_index == len(selected) or member_index % 10 == 0:
                self.log(
                    f"Rendered {member_index}/{len(selected)} proteins: {member.member_id} "
                    f"({report['diagnostics']['status']})"
                )

        # The landing page and JSON manifest expose exactly which immutable members were rendered
        # and link every browser/external-viewer representation.

        ProteinArchive.write_text(
            output_root / "visualizations.json",
            json.dumps(
                {
                    "dataset_root_name": dataset_root.name,
                    "rendered_members":  reports,
                    "mesh_notice": (
                        "Alpha-complex meshes are diagnostic reconstructions, not stored molecular "
                        "surfaces and not scientific model inputs."
                    ),
                },
                indent=2,
                sort_keys=True,
            ),
        )
        ProteinArchive.write_text(output_root / "index.html", self._index_html(reports))

        failures = sum(report["diagnostics"]["status"] != "PASS" for report in reports)
        self.metrics.log("visualized_proteins", len(reports))
        self.metrics.log("visual_diagnostic_failures", failures)
        self.log(
            f"Visualization gallery complete: {len(reports)} proteins, "
            f"{failures} automatic diagnostic failures"
        )
        return {
            "skipped":             False,
            "output":              "protein-visualizations",
            "rendered_proteins":   len(reports),
            "identifiers":         [report["identifier"] for report in reports],
            "diagnostic_failures": failures,
        }

    @staticmethod
    def _select_members(
        members         : Sequence[DatasetMember],
        identifiers     : Sequence[str],
        splits          : Sequence[str],
        labels          : Sequence[int],
        maximum_proteins: int,
    ) -> tuple[DatasetMember, ...]:
        """Resolve explicit IDs or create a deterministic split/class round-robin sample.

        Args:
            members: Complete immutable DatasetIndex membership in stored order.
            identifiers: Optional exact member IDs whose order must be preserved.
            splits: Eligible canonical split values for automatic sampling.
            labels: Eligible binary protein targets for automatic sampling.
            maximum_proteins: Largest automatic sample size; zero retains all eligible members.

        Returns:
            Ordered members to render.

        Raises:
            ValueError: If an explicit ID is absent or automatic filters select no members.
        """
        by_identifier = {member.member_id: member for member in members}
        if identifiers:
            missing = [identifier for identifier in identifiers if identifier not in by_identifier]
            if missing:
                raise ValueError(f"visualization requested unknown dataset IDs: {missing}")
            return tuple(by_identifier[identifier] for identifier in identifiers)

        eligible_splits = {str(value) for value in splits}
        eligible_labels = {int(value) for value in labels}
        buckets: dict[tuple[str, int], list[DatasetMember]] = defaultdict(list)
        for member in sorted(members, key=lambda value: value.member_id):
            split = str(member.partitions.get("split", ""))
            label = int(member.targets.get("dna_binding", -1))
            if split in eligible_splits and label in eligible_labels:
                buckets[(split, label)].append(member)
        if not buckets:
            raise ValueError("visualization split/label filters selected no dataset members")

        # Taking one member from each non-empty stratum before taking the next keeps a bounded
        # gallery representative whenever indivisible stratum sizes permit it.

        limit        = maximum_proteins or sum(len(bucket) for bucket in buckets.values())
        selected: list[DatasetMember] = []
        ordered_keys = sorted(buckets)
        row          = 0
        while len(selected) < limit:
            added = False
            for key in ordered_keys:
                if row < len(buckets[key]):
                    selected.append(buckets[key][row])
                    added = True
                    if len(selected) == limit:
                        break
            if not added:
                break
            row += 1
        return tuple(selected)

    @staticmethod
    def _render_member(
        dataset_root: Path,
        output_root : Path,
        member      : DatasetMember,
        visualizer  : ProteinVisualizer,
        exporter    : PointCloudExporter,
    ) -> dict[str, Any]:
        """Render one indexed member from its universal and DNA sidecar assets.

        Args:
            dataset_root: Resolved immutable DatasetVersion placement.
            output_root: Managed visualization directory owned by the current Work.
            member: Canonical DatasetIndex member with targets, partitions, and assets.
            visualizer: Configured interactive HTML renderer.
            exporter: Lossless-order PLY/NPZ point-cloud exporter.

        Returns:
            JSON-compatible paths, member metadata, channel names, and diagnostics.

        Raises:
            ValueError: If required member assets or aligned arrays are missing.
            OSError: If NPZ/HTML/PLY files cannot be read or written.
        """
        try:
            base       = dataset_root / member.assets["universal_npz"].path
            annotation = dataset_root / member.assets["dna_annotation"].path
        except KeyError as error:
            raise ValueError(
                f"dataset member {member.member_id!r} lacks WISDOM visualization assets"
            ) from error

        safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "-", member.member_id).strip(".-")
        html_path = output_root / "proteins" / f"{safe_name}.html"
        ply_path  = output_root / "proteins" / f"{safe_name}.ply"

        surface_channels = visualizer.surface_channels(base, annotation)
        diagnostics = visualizer.visualize(
            base,
            html_path,
            member.member_id,
            annotation      = annotation,
            protein_label   = int(member.targets["dna_binding"]),
            partitions      = dict(member.partitions),
            plotly_script   = "../plotly.min.js",
        )
        with np.load(base, allow_pickle=False) as archive:
            positions = archive["surface_positions"]
        exporter.export(ply_path, positions, surface_channels)

        return {
            "identifier":  member.member_id,
            "split":       str(member.partitions.get("split", "")),
            "label":       int(member.targets["dna_binding"]),
            "html":        html_path.relative_to(output_root).as_posix(),
            "ply":         ply_path.relative_to(output_root).as_posix(),
            "channels":    sorted(surface_channels),
            "diagnostics": dict(diagnostics),
        }

    @staticmethod
    def _index_html(reports: Sequence[dict[str, Any]]) -> str:
        """Build a compact offline landing page for rendered protein reports.

        Args:
            reports: Ordered rendered-member mappings with links, labels, and diagnostics.

        Returns:
            Complete UTF-8 HTML document containing a searchable report table.
        """
        rows = "".join(
            f"<tr data-id='{report['identifier'].lower()}' data-split='{report['split']}' "
            f"data-label='{report['label']}' data-status='{report['diagnostics']['status']}'>"
            f"<td><a href='{report['html']}'>{report['identifier']}</a></td>"
            f"<td>{report['split']}</td><td>{report['label']}</td>"
            f"<td><span class='status {str(report['diagnostics']['status']).lower()}'>"
            f"{report['diagnostics']['status']}</span></td>"
            f"<td><a class='secondary' href='{report['ply']}'>PLY</a></td>"
            "</tr>"
            for report in reports
        )
        return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>WISDOM dataset visualizations</title><style>
:root{{color-scheme:dark;font:15px Inter,system-ui,sans-serif;background:#080c16;color:#e7edf7}}
*{{box-sizing:border-box}}body{{max-width:1180px;margin:0 auto;padding:42px 24px 70px}}h1{{font-size:30px;margin:0 0 8px}}.lead{{color:#a9b7ca;max-width:780px;line-height:1.55}}.count{{display:inline-block;margin:12px 0 24px;padding:5px 10px;border-radius:999px;background:#173e49;color:#baf6ff}}.filters{{display:grid;grid-template-columns:2fr repeat(3,1fr);gap:10px;padding:14px;background:#111827;border:1px solid #2d3b50;border-radius:12px;margin:18px 0}}input,select{{width:100%;padding:9px 10px;background:#1b2638;color:#e7edf7;border:1px solid #3b4a61;border-radius:8px}}.table-wrap{{overflow:auto;border:1px solid #2d3b50;border-radius:12px;background:#101827}}table{{border-collapse:collapse;width:100%}}th,td{{padding:11px 13px;border-bottom:1px solid #27364b;text-align:left}}th{{font-size:12px;color:#98a2b3;text-transform:uppercase;letter-spacing:.04em}}a{{color:#6fe7f2;text-decoration:none;font-weight:650}}a:hover{{text-decoration:underline}}a.secondary{{color:#b9c5d6;font-weight:500}}.notice{{padding:13px;background:#2d2415;border-left:4px solid #f79009;border-radius:7px;line-height:1.5}}.status{{font-size:11px;padding:4px 7px;border-radius:999px}}.status.pass{{background:#123923;color:#8ce6a7}}.status.fail{{background:#482020;color:#ff9b9b}}#empty{{display:none;text-align:center;padding:30px;color:#98a2b3}}@media(max-width:760px){{.filters{{grid-template-columns:1fr 1fr}}}}
</style></head><body><h1>WISDOM protein gallery</h1>
<p class="lead">Open a protein to inspect geometric layers, scalar channels, atoms, bonds, van der
Waals envelopes, and point-to-point distances. PLY files preserve the complete surface cloud and
scalar fields for external viewers.</p><span class="count">{len(reports)} proteins rendered</span>
<p class="notice"><strong>Mesh limitation.</strong> The alpha-complex mesh is a diagnostic visual
reconstruction. The immutable point cloud is authoritative; the mesh is not a molecular surface,
is not consumed by WISDOM, and may bridge narrow pockets or omit poorly sampled regions.</p>
<div class="filters"><input id="search" type="search" placeholder="Search protein ID">
<select id="split"><option value="">All splits</option><option>train</option><option>validation</option><option>test</option></select>
<select id="label"><option value="">Both labels</option><option value="0">Negative · 0</option><option value="1">Positive · 1</option></select>
<select id="status"><option value="">Any audit</option><option>PASS</option><option>FAIL</option></select></div>
<div class="table-wrap"><table><thead><tr><th>Protein</th><th>Split</th><th>Label</th>
<th>Automatic audit</th><th>External</th></tr></thead><tbody>{rows}</tbody></table>
<div id="empty">No protein matches these filters.</div></div>
<script>const rows=[...document.querySelectorAll('tbody tr')],byId=id=>document.getElementById(id);function filter(){{const query=byId('search').value.trim().toLowerCase(),split=byId('split').value,label=byId('label').value,status=byId('status').value;let visible=0;for(const row of rows){{const show=(!query||row.dataset.id.includes(query))&&(!split||row.dataset.split===split)&&(!label||row.dataset.label===label)&&(!status||row.dataset.status===status);row.hidden=!show;if(show)visible++}}byId('empty').style.display=visible?'none':'block'}}for(const id of ['search','split','label','status'])byId(id).addEventListener('input',filter);</script></body></html>"""
