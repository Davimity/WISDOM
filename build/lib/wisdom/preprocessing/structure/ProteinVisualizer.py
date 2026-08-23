"""Dependency-free interactive HTML inspection of WISDOM NPZ proteins."""

# ruff: noqa: E501 -- the embedded dependency-free HTML/JavaScript remains readable as source lines.

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from itertools import pairwise
from pathlib import Path
from typing import Any, cast

import numpy as np

from wisdom.preprocessing.ProcessingWorkspace import ProcessingWorkspace
from wisdom.preprocessing.structure.StorageManager import StorageManager


class ProteinVisualizer:
    """Render atoms, backbone, surface points, normals, diagnostics, and NPZ array contents."""

    TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>WISDOM protein inspector</title><style>
:root{color-scheme:dark;background:#0b1020;color:#e7edf7;font:14px system-ui,sans-serif}
body{margin:0;display:grid;grid-template-columns:minmax(540px,2fr) minmax(360px,1fr);height:100vh}
#viewer{position:relative;min-height:600px;background:radial-gradient(circle,#172444,#080c17)}
canvas{width:100%;height:100%;display:block}.panel{overflow:auto;padding:18px;background:#111827}
.controls{position:absolute;left:12px;top:12px;background:#0b1020dd;padding:10px;border-radius:8px}
label{display:block;margin:5px 0}select,input{accent-color:#4dd0e1}h1{font-size:20px;margin:0 0 8px}
h2{font-size:16px;margin-top:22px;color:#80deea}.ok{color:#8ee59b}.bad{color:#ff8a80}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #344054;padding:5px;text-align:left;vertical-align:top}
code{white-space:pre-wrap;word-break:break-word;color:#c5e1ff}.hint{position:absolute;bottom:10px;left:12px;color:#b7c6dc}
</style></head><body><div id="viewer"><canvas id="canvas"></canvas><div class="controls">
<strong id="title"></strong><label><input id="surface" type="checkbox" checked> Surface points</label>
<label><input id="atoms" type="checkbox" checked> Atomic spheres</label>
<label><input id="cartoon" type="checkbox" checked> C-alpha backbone cartoon</label>
<label><input id="normals" type="checkbox"> Surface normals</label>
<label>Surface colour <select id="colour"><option value="gap">Envelope gap</option>
<option value="curvature">Curvedness</option><option value="component">Component</option></select></label>
<label>Point size <input id="size" type="range" min="1" max="5" value="2"></label></div>
<div class="hint">Drag to rotate · wheel to zoom · click a surface point for values</div></div>
<aside class="panel"><h1>WISDOM NPZ inspector</h1><div id="diagnostics"></div>
<h2>Selected point</h2><code id="selected">Click a visible surface point.</code>
<h2>Arrays</h2><table><thead><tr><th>Name</th><th>Shape</th><th>Dtype</th><th>Range / sample</th></tr></thead><tbody id="arrays"></tbody></table>
<h2>Metadata</h2><code id="metadata"></code></aside><script>
const D=__WISDOM_DATA__; const canvas=document.getElementById('canvas'),ctx=canvas.getContext('2d');
let yaw=.45,pitch=-.35,zoom=1,drag=false,last=[0,0],projected=[];
const el=id=>document.getElementById(id); el('title').textContent=D.identifier;
function esc(v){return String(v).replace(/[&<>]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;'}[c]));}
el('diagnostics').innerHTML=`<div class="${D.diagnostics.status==='PASS'?'ok':'bad'}"><b>${D.diagnostics.status}</b></div>`+
Object.entries(D.diagnostics).filter(([k])=>k!=='status').map(([k,v])=>`${esc(k)}: <b>${esc(v)}</b>`).join('<br>');
el('arrays').innerHTML=D.arrays.map(a=>`<tr><td><code>${esc(a.name)}</code></td><td>${esc(a.shape)}</td><td>${esc(a.dtype)}</td><td>${esc(a.summary)}</td></tr>`).join('');
el('metadata').textContent=JSON.stringify(D.metadata,null,2);
function resize(){const r=canvas.getBoundingClientRect(),d=devicePixelRatio||1;canvas.width=r.width*d;canvas.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);draw();}
function rotate(p){let x=p[0]-D.center[0],y=p[1]-D.center[1],z=p[2]-D.center[2];
 let c=Math.cos(yaw),s=Math.sin(yaw),x1=c*x-s*z,z1=s*x+c*z;c=Math.cos(pitch);s=Math.sin(pitch);
 return [x1,c*y-s*z1,s*y+c*z1];}
function screen(p){const q=rotate(p),scale=Math.min(canvas.clientWidth,canvas.clientHeight)*.42*zoom/D.radius;
 return [canvas.clientWidth/2+q[0]*scale,canvas.clientHeight/2-q[1]*scale,q[2],scale];}
function colour(i){if(el('colour').value==='component'){const h=(D.surface.component[i]*67)%360;return `hsl(${h} 75% 58%)`;}
 const v=el('colour').value==='curvature'?D.surface.curvature[i]:D.surface.gap[i], range=el('colour').value==='curvature'?D.ranges.curvature:D.ranges.gap;
 const t=Math.max(0,Math.min(1,(v-range[0])/Math.max(range[1]-range[0],1e-9)));return `rgb(${Math.round(255*t)},${Math.round(210*(1-Math.abs(2*t-1)))},${Math.round(255*(1-t))})`;}
function draw(){ctx.clearRect(0,0,canvas.clientWidth,canvas.clientHeight);projected=[];
 if(el('cartoon').checked){ctx.strokeStyle='#f8fafc';ctx.lineWidth=2;for(const e of D.cartoon){const a=screen(e[0]),b=screen(e[1]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}}
 if(el('atoms').checked){const rows=D.atoms.position.map((p,i)=>[screen(p),i]).sort((a,b)=>a[0][2]-b[0][2]);for(const [q,i] of rows){ctx.globalAlpha=.52;ctx.fillStyle=D.atoms.colour[i];ctx.beginPath();ctx.arc(q[0],q[1],Math.max(2,D.atoms.radius[i]*q[3]*.24),0,Math.PI*2);ctx.fill();}ctx.globalAlpha=1;}
 if(el('surface').checked){const size=+el('size').value;for(let i=0;i<D.surface.position.length;i++){const q=screen(D.surface.position[i]);projected.push([q[0],q[1],q[2],i]);ctx.fillStyle=colour(i);ctx.fillRect(q[0]-size/2,q[1]-size/2,size,size);}}
 if(el('normals').checked){ctx.strokeStyle='#fff8';ctx.lineWidth=1;for(let i=0;i<D.normals.position.length;i++){const a=screen(D.normals.position[i]),b=screen(D.normals.end[i]);ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();}}
}
canvas.addEventListener('pointerdown',e=>{drag=true;last=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId)});
canvas.addEventListener('pointermove',e=>{if(!drag)return;yaw+=(e.clientX-last[0])*.008;pitch+=(e.clientY-last[1])*.008;last=[e.clientX,e.clientY];draw()});
canvas.addEventListener('pointerup',()=>drag=false);canvas.addEventListener('wheel',e=>{e.preventDefault();zoom*=Math.exp(-e.deltaY*.001);zoom=Math.max(.15,Math.min(8,zoom));draw()},{passive:false});
canvas.addEventListener('click',e=>{if(Math.hypot(e.clientX-last[0],e.clientY-last[1])>4)return;const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top;
 let best=null;for(const p of projected){const d=(p[0]-x)**2+(p[1]-y)**2;if(d<64&&(!best||d<best[0]))best=[d,p[3]];}if(best){const i=best[1];el('selected').textContent=JSON.stringify({display_index:i,position:D.surface.position[i],normal:D.surface.normal[i],signed_gap:D.surface.gap[i],curvedness:D.surface.curvature[i],component:D.surface.component[i]},null,2);}});
for(const id of ['surface','atoms','cartoon','normals','colour','size'])el(id).addEventListener('input',draw);window.addEventListener('resize',resize);resize();
</script></body></html>"""

    def __init__(
        self,
        identifiers         : Sequence[str],
        processed_input     : str   = "processed_dataset",
        report_input        : str   = "preprocessing_report",
        visualization_output: str   = "visualizations",
        index_output        : str   = "visualization_index",
        max_surface_points  : int   = 6000,
        normal_stride       : int   = 25,
        normal_length       : float = 1.5,
    ) -> None:
        """Configure immutable inputs and bounded browser payload sizes.

        Args:
            identifiers: Non-empty identifiers to render, in requested index order.
            processed_input: Named input containing WISDOM NPZ files.
            report_input: Named input mapping exact identifiers to NPZ output names.
            visualization_output: Named output directory receiving standalone protein viewers.
            index_output: Named output file linking every generated viewer.
            max_surface_points: Maximum deterministic surface subset embedded per HTML file.
            normal_stride: Draw one normal per this many displayed points.
            normal_length: Displayed normal-vector length in ångströms.

        Raises:
            ValueError: If identifiers/logical names are empty or a display limit is not positive.
        """
        if not identifiers:
            raise ValueError("identifiers must contain at least one protein")
        logical_names = (processed_input, report_input, visualization_output, index_output)
        if any(not name.strip() for name in logical_names):
            raise ValueError("logical input and output names cannot be empty")
        if max_surface_points < 1 or normal_stride < 1 or normal_length <= 0:
            raise ValueError("visualization limits and normal length must be positive")

        self.identifiers          = tuple(str(value) for value in identifiers)
        self.processed_input      = processed_input
        self.report_input         = report_input
        self.visualization_output = visualization_output
        self.index_output         = index_output
        self.max_surface_points   = max_surface_points
        self.normal_stride        = normal_stride
        self.normal_length        = normal_length

    def run(self, context: ProcessingWorkspace) -> dict[str, Any]:
        """Create one standalone interactive HTML inspector per requested protein.

        Args:
            context: LambdaForge task context resolving fingerprinted inputs and safe output paths.

        Returns:
            JSON-compatible count and run-relative HTML index path.

        Raises:
            ValueError: If the report is malformed, an identifier is absent/failed, or an output
                path is unsafe.
            OSError: If input archives cannot be read or HTML artifacts cannot be written.
        """
        processed_dir = context.input(self.processed_input)
        report_path   = context.input(self.report_input)
        report        = json.loads(report_path.read_text(encoding="utf-8"))
        records       = {
            str(record.get("identifier")): record
            for record in cast(list[dict[str, Any]], report.get("records", []))
        }

        output_dir = context.output(self.visualization_output)
        output_dir.mkdir(parents=True, exist_ok=True)
        links: list[tuple[str, str, Mapping[str, Any]]] = []
        for identifier in self.identifiers:
            record = records.get(identifier)
            if record is None or record.get("status") not in {"processed", "skipped"}:
                raise ValueError(f"identifier {identifier!r} has no successful preprocessing record")
            output_name = record.get("output")
            if not isinstance(output_name, str) or Path(output_name).name != output_name:
                raise ValueError(f"identifier {identifier!r} has an unsafe output filename")

            html_name  = f"{Path(output_name).stem}.html"
            diagnostics = self.visualize(
                processed_dir / output_name,
                output_dir / html_name,
                identifier,
            )
            links.append((identifier, html_name, diagnostics))

        # The index surfaces the automatic verdict before opening the interactive 3D view.
        visualization_path = output_dir.relative_to(context.run_dir).as_posix()
        items = "\n".join(
            f'<li><a href="{visualization_path}/{name}">{identifier}</a> — '
            f'{diagnostics["status"]}</li>'
            for identifier, name, diagnostics in links
        )
        index = (
            "<!doctype html><meta charset=\"utf-8\"><title>WISDOM visualizations</title>"
            "<h1>WISDOM preprocessing visualizations</h1><ul>"
            + items
            + "</ul>"
        )
        index_path = context.output(self.index_output, create=True)
        StorageManager.write_text(index_path, index)
        return {
            "count": len(links),
            "index": index_path.relative_to(context.run_dir).as_posix(),
        }

    def visualize(
        self,
        path      : Path,
        output    : Path,
        identifier: str,
    ) -> Mapping[str, Any]:
        """Inspect one NPZ and write a self-contained offline canvas-based 3D report.

        The view deterministically subsamples very large surfaces, projects atomic van der Waals
        spheres as depth-scaled circles, joins consecutive C-alpha atoms as a backbone cartoon,
        draws optional outward normals, and colors points by signed envelope gap, curvedness, or
        component. A complete array inventory reports every name, shape, dtype, numeric range, and
        sample value even when the 3D point cloud is subsampled.

        Args:
            path: Existing pickle-free WISDOM NPZ archive.
            output: Final standalone HTML file written atomically.
            identifier: Human-facing source identifier displayed in the report.

        Returns:
            Automatic visual-quality diagnostic mapping also shown at the top of the HTML report.

        Raises:
            ValueError: If geometry arrays or metadata required for rendering are absent/malformed.
            OSError: If the archive cannot be read or the HTML cannot be published.
        """
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        required = {
            "atom_positions",
            "atomic_numbers",
            "vdw_radii",
            "atom_names",
            "residue_indices",
            "chain_indices",
            "surface_positions",
            "surface_normals",
            "surface_curvatures",
            "surface_component_ids",
            "surface_edge_index",
            "surface_atom_edge_index",
            "surface_atom_distance",
            StorageManager.METADATA_NAME,
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"visualization archive is missing arrays: {sorted(missing)}")

        metadata = json.loads(str(arrays[StorageManager.METADATA_NAME].item()))
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json root must be an object")

        atom_positions    = arrays["atom_positions"].astype(np.float64)
        surface_positions = arrays["surface_positions"].astype(np.float64)
        surface_normals   = arrays["surface_normals"].astype(np.float64)
        bipartite         = arrays["surface_atom_edge_index"]
        config_value = cast(dict[str, Any], metadata.get("config", {}))
        probe_radius = float(config_value.get("probe_radius", 1.4))

        # Signed gaps and envelope-gradient cosines directly expose buried/floating/reversed points.
        expanded = arrays["vdw_radii"].astype(np.float64) + probe_radius
        gaps     = arrays["surface_atom_distance"].astype(np.float64) - expanded[bipartite[1]]
        minimum_gaps = np.full(len(surface_positions), np.inf)
        np.minimum.at(minimum_gaps, bipartite[0], gaps)

        resolution = float(config_value.get("surface_resolution", 1.0))
        tolerance = max(5.0e-4, 0.025 * resolution)
        interior  = int(np.count_nonzero(minimum_gaps < -tolerance))
        floating  = int(np.count_nonzero(minimum_gaps > tolerance))

        # Reconstruct the soft-min sphere gradient to expose reversed or inconsistent normals.
        offsets   = surface_positions[bipartite[0]] - atom_positions[bipartite[1]]
        distances = arrays["surface_atom_distance"].astype(np.float64)
        radial    = offsets / np.maximum(distances[:, None], 1.0e-12)

        smoothness = max(0.25 * resolution, 1.0e-3)
        active      = gaps <= minimum_gaps[bipartite[0]] + 2.5 * smoothness
        weights     = np.exp(
            -(gaps[active] - minimum_gaps[bipartite[0, active]]) / smoothness
        )
        expected_normals = np.zeros_like(surface_positions)
        np.add.at(
            expected_normals,
            bipartite[0, active],
            weights[:, None] * radial[active],
        )
        expected_normals /= np.maximum(
            np.linalg.norm(expected_normals, axis=1, keepdims=True),
            1.0e-12,
        )
        normal_cosines = np.sum(expected_normals * surface_normals, axis=1)
        normal_errors  = int(np.count_nonzero(normal_cosines < 0.99))

        # Algebraic identities and scale-normalized magnitude expose unstable curvature fits.
        curvatures = arrays["surface_curvatures"].astype(np.float64)
        mean       = curvatures[:, :, 0]
        gaussian   = curvatures[:, :, 1]
        curvedness_matrix = curvatures[:, :, 2]
        algebra_errors = int(
            np.size(mean)
            - np.count_nonzero(
                np.isclose(
                    curvedness_matrix * curvedness_matrix,
                    2.0 * mean * mean - gaussian,
                    rtol=2.0e-4,
                    atol=2.0e-6,
                )
            )
        )
        scale_values = config_value.get("curvature_scales", [2.5, 5.0])
        scales       = np.asarray(scale_values, dtype=np.float64)
        if scales.shape != (curvatures.shape[1],):
            scales = np.arange(1, curvatures.shape[1] + 1, dtype=np.float64)
        dimensionless_curvature = curvedness_matrix * (resolution * scales[None, :])
        unstable_curvatures = int(np.count_nonzero(dimensionless_curvature > 25.0))

        surface_edges = arrays["surface_edge_index"]
        degree        = np.zeros(len(surface_positions), dtype=np.int64)
        np.add.at(degree, surface_edges[0], 1)
        np.add.at(degree, surface_edges[1], 1)
        isolated = int(np.count_nonzero(degree == 0))

        valid = not any(
            (interior, floating, normal_errors, algebra_errors, unstable_curvatures)
        )
        diagnostics: dict[str, Any] = {
            "status": "PASS" if valid else "FAIL",
            "surface points": len(surface_positions),
            "interior points": interior,
            "floating points": floating,
            "normal orientation errors": normal_errors,
            "minimum normal cosine": f"{normal_cosines.min():.6g}",
            "curvature identity errors": algebra_errors,
            "unstable curvature values": unstable_curvatures,
            "maximum dimensionless curvature": f"{dimensionless_curvature.max():.6g}",
            "isolated graph points": isolated,
            "signed gap range (Å)": (
                f"{minimum_gaps.min():.5g} .. {minimum_gaps.max():.5g}"
            ),
            "surface components": len(np.unique(arrays["surface_component_ids"])),
        }

        # Deterministic equal-index sampling preserves coverage without random visual differences.
        display_count = min(self.max_surface_points, len(surface_positions))
        display_ids   = np.linspace(0, len(surface_positions) - 1, display_count, dtype=np.int64)
        normal_ids    = display_ids[:: self.normal_stride]

        curvedness = curvedness_matrix[:, 0]
        components = arrays["surface_component_ids"]

        # C-alpha segments provide a recognizable structural reference without a second parser.
        atom_names      = arrays["atom_names"].astype(str)
        residue_indices = arrays["residue_indices"]
        chain_indices   = arrays["chain_indices"]
        alpha_ids       = np.flatnonzero(atom_names == "CA")
        cartoon: list[list[list[float]]] = []
        for left, right in pairwise(alpha_ids):
            same_chain = chain_indices[left] == chain_indices[right]
            adjacent   = residue_indices[right] - residue_indices[left] == 1
            close      = np.linalg.norm(atom_positions[right] - atom_positions[left]) < 5.0
            if same_chain and adjacent and close:
                cartoon.append(
                    [atom_positions[left].tolist(), atom_positions[right].tolist()]
                )

        # A complete table makes the archive inspectable beyond the tensors used in the 3D view.
        inventory: list[dict[str, str]] = []
        for name in sorted(arrays):
            array = arrays[name]
            flattened = array.reshape(-1)
            sample = flattened[: min(8, len(flattened))].tolist()
            if np.issubdtype(array.dtype, np.number) and len(flattened):
                summary = (
                    f"min={np.min(flattened):.6g}, max={np.max(flattened):.6g}, "
                    f"mean={np.mean(flattened):.6g}; sample={sample}"
                )
            else:
                summary = f"sample={sample}"
            inventory.append(
                {
                    "name": name,
                    "shape": str(array.shape),
                    "dtype": str(array.dtype),
                    "summary": summary,
                }
            )

        all_positions = np.concatenate((atom_positions, surface_positions))
        center        = (all_positions.min(axis=0) + all_positions.max(axis=0)) / 2.0
        radius        = float(np.linalg.norm(all_positions - center, axis=1).max())
        atom_colours  = {
            6: "#5b6678",
            7: "#4f8cff",
            8: "#ff5d73",
            15: "#ff9f43",
            16: "#ffd84d",
        }
        atomic_numbers = arrays["atomic_numbers"].astype(int)

        payload = {
            "identifier": identifier,
            "center": center.tolist(),
            "radius": max(radius, 1.0),
            "diagnostics": diagnostics,
            "arrays": inventory,
            "metadata": metadata,
            "atoms": {
                "position": atom_positions.tolist(),
                "radius": arrays["vdw_radii"].astype(float).tolist(),
                "colour": [atom_colours.get(int(value), "#d8dee9") for value in atomic_numbers],
            },
            "cartoon": cartoon,
            "surface": {
                "position": surface_positions[display_ids].tolist(),
                "normal": surface_normals[display_ids].tolist(),
                "gap": minimum_gaps[display_ids].tolist(),
                "curvature": curvedness[display_ids].tolist(),
                "component": components[display_ids].astype(int).tolist(),
            },
            "normals": {
                "position": surface_positions[normal_ids].tolist(),
                "end": (
                    surface_positions[normal_ids]
                    + self.normal_length * surface_normals[normal_ids]
                ).tolist(),
            },
            "ranges": {
                "gap": [float(minimum_gaps.min()), float(minimum_gaps.max())],
                "curvature": [
                    float(np.quantile(curvedness, 0.01)),
                    float(np.quantile(curvedness, 0.99)),
                ],
            },
        }
        data = json.dumps(payload, separators=(",", ":")).replace("<", "\\u003c")
        StorageManager.write_text(output, self.TEMPLATE.replace("__WISDOM_DATA__", data))
        return diagnostics
