"""Interactive WebGL inspection of WISDOM NPZ protein representations."""

# ruff: noqa: E501 -- embedded HTML/JavaScript and Plotly hover templates remain readable intact.

from __future__ import annotations

import json
import hashlib
import numpy as np
import plotly.graph_objects as go

from pathlib import Path
from typing import Any, cast
from itertools import pairwise
from collections.abc import Mapping
from scipy.spatial import Delaunay, ConvexHull, QhullError

from wisdom.preprocessing.structure.ProteinArchive import ProteinArchive


class ProteinVisualizer:
    """Render scientific arrays as switchable point, mesh, atom, graph, and normal layers."""

    def __init__(
        self,
        max_surface_points: int   = 6000,
        max_mesh_points   : int   = 2500,
        max_edges         : int   = 5000,
        normal_stride     : int   = 25,
        normal_length     : float = 1.5,
        mesh_alpha        : float = 4.0,
        max_vdw_atoms     : int   = 1500,
    ) -> None:
        """Configure bounded browser payloads and diagnostic mesh reconstruction.

        Args:
            max_surface_points: Maximum deterministic surface subset embedded in the HTML cloud.
            max_mesh_points: Maximum subset supplied to alpha-complex meshing.
            max_edges: Maximum edges displayed from each sparse graph layer.
            normal_stride: Draw one normal per this many displayed surface points.
            normal_length: Displayed normal-vector length in ångströms.
            mesh_alpha: Largest alpha-complex tetrahedron radius in ångströms.
            max_vdw_atoms: Maximum deterministic atom subset represented by physical-radius
                icosahedra. Ordinary atom markers remain available for every atom.

        Raises:
            ValueError: If a limit, length, or alpha parameter is not positive.
        """
        if (
            max_surface_points < 1
            or max_mesh_points < 4
            or max_edges < 1
            or normal_stride < 1
            or normal_length <= 0.0
            or mesh_alpha <= 0.0
            or max_vdw_atoms < 1
        ):
            raise ValueError("visualization limits, lengths, and mesh alpha must be positive")

        self.max_surface_points = max_surface_points
        self.max_mesh_points    = max_mesh_points
        self.max_edges          = max_edges
        self.normal_stride      = normal_stride
        self.normal_length      = normal_length
        self.mesh_alpha         = mesh_alpha
        self.max_vdw_atoms      = max_vdw_atoms

    def surface_channels(
        self,
        path      : Path,
        annotation: Path | None = None,
    ) -> Mapping[str, np.ndarray]:
        """Return full-order scalar surface fields for HTML and PLY colouring.

        Curvature tensors are expanded by invariant and physical radius. DNA targets are included
        only after their base digest and point count match the universal archive.

        Args:
            path: Pickle-free universal WISDOM NPZ archive.
            annotation: Optional point-aligned DNA sidecar NPZ.

        Returns:
            PLY-safe channel names mapped to numeric arrays with shape ``[M]``.

        Raises:
            ValueError: If required arrays, shapes, or fingerprints disagree.
            OSError: If either NPZ cannot be read.
        """
        arrays, metadata, sidecar = self._load(path, annotation)
        return self._surface_channels(arrays, metadata, sidecar)

    def visualize(
        self,
        path          : Path,
        output        : Path,
        identifier    : str,
        annotation    : Path | None       = None,
        protein_label : int | None        = None,
        partitions    : Mapping[str, Any] | None = None,
        plotly_script : str | bool        = True,
    ) -> Mapping[str, Any]:
        """Write one interactive report with all structural and annotation views.

        The point cloud is authoritative. The hidden alpha-complex mesh is a diagnostic derivative
        because sampled points alone cannot guarantee molecular topology. Atoms, backbone, outward
        normals, surface edges, and the three atomic relation types are independently toggleable.

        Args:
            path: Existing universal WISDOM NPZ archive.
            output: Final HTML file written atomically.
            identifier: Human-facing immutable dataset member ID.
            annotation: Optional aligned DNA sidecar.
            protein_label: Optional global binary DNA-binding target.
            partitions: Optional dataset split/group/phenotype provenance.
            plotly_script: ``True`` embeds Plotly; a string references a shared local script.

        Returns:
            Automatic geometric diagnostics shown in the report.

        Raises:
            ValueError: If geometry, metadata, or annotation alignment is malformed.
            OSError: If NPZ inputs or the HTML cannot be read or written.
        """
        arrays, metadata, sidecar = self._load(path, annotation)
        channels                 = self._surface_channels(arrays, metadata, sidecar)
        diagnostics              = self._diagnostics(arrays, metadata)
        figure, controls         = self._figure(arrays, channels)

        plot = figure.to_html(
            full_html        = False,
            include_plotlyjs = plotly_script,
            div_id           = "wisdom-plot",
            config           = {"displaylogo": False, "responsive": True, "scrollZoom": True},
        )
        inventory_rows = "".join(
            f"<tr><td><code>{self._escape(item['name'])}</code></td>"
            f"<td>{item['shape']}</td><td>{item['dtype']}</td>"
            f"<td>{self._escape(item['summary'])}</td></tr>"
            for item in self._inventory(arrays, sidecar)
        )
        diagnostic_rows = "".join(
            f"<tr><td>{self._escape(name)}</td><td>{self._escape(value)}</td></tr>"
            for name, value in diagnostics.items()
            if name != "status"
        )
        annotation_metadata = None
        if "annotation_metadata_json" in sidecar:
            annotation_metadata = json.loads(
                str(sidecar["annotation_metadata_json"].item())
            )
        provenance = self._escape(
            json.dumps(
                {
                    "identifier":          identifier,
                    "protein_label":       protein_label,
                    "partitions":          dict(partitions or {}),
                    "base_metadata":       metadata,
                    "annotation_metadata": annotation_metadata,
                },
                indent=2,
                sort_keys=True,
            )
        )
        control_data = json.dumps(controls, separators=(",", ":"), allow_nan=False).replace(
            "<", "\\u003c"
        )
        html = self._html(
            identifier      = self._escape(identifier),
            status          = str(diagnostics["status"]),
            diagnostic_rows = diagnostic_rows,
            inventory_rows  = inventory_rows,
            provenance      = provenance,
            plot            = plot,
            controls        = control_data,
        )
        ProteinArchive.write_text(output, html)
        return diagnostics

    def _load(
        self,
        path      : Path,
        annotation: Path | None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
        """Load base arrays and verify an optional sidecar against exact base bytes.

        Args:
            path: Universal NPZ archive path.
            annotation: Optional DNA sidecar path.

        Returns:
            Base arrays, decoded metadata, and optional sidecar arrays.

        Raises:
            ValueError: If required arrays, metadata, counts, or SHA-256 disagree.
            OSError: If either archive cannot be opened.
        """
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        required = {
            "atom_positions", "atomic_numbers", "residue_type_ids", "atom_role_ids",
            "formal_charges", "vdw_radii", "atom_names", "residue_names",
            "residue_indices", "chain_indices", "atom_edge_index",
            "atom_edge_is_covalent", "atom_edge_spatial_rank", "surface_positions",
            "surface_normals", "surface_curvatures", "surface_area_weights",
            "surface_neighbors", "surface_neighbor_mask", "surface_atom_neighbors",
            "surface_atom_distances", "surface_atom_mask",
            ProteinArchive.METADATA_NAME,
        }
        missing = required - arrays.keys()
        if missing:
            raise ValueError(f"visualization archive is missing arrays: {sorted(missing)}")

        metadata = json.loads(str(arrays[ProteinArchive.METADATA_NAME].item()))
        if not isinstance(metadata, dict):
            raise ValueError("metadata_json root must be an object")

        sidecar: dict[str, np.ndarray] = {}
        if annotation is not None:
            with np.load(annotation, allow_pickle=False) as archive:
                sidecar = {name: archive[name] for name in archive.files}
            required_sidecar = {
                "surface_target_hard", "surface_valid_mask", "surface_target_soft",
                "surface_distance_to_dna", "surface_distance_valid",
                "surface_target_hard_sensitivity", "sensitivity_gaps", "base_npz_sha256",
            }
            missing_sidecar = required_sidecar - sidecar.keys()
            if missing_sidecar:
                raise ValueError(
                    f"visualization sidecar is missing arrays: {sorted(missing_sidecar)}"
                )
            surface_count = len(arrays["surface_positions"])
            for name in (
                "surface_target_hard", "surface_valid_mask", "surface_target_soft",
                "surface_distance_to_dna", "surface_distance_valid",
            ):
                if sidecar[name].shape != (surface_count,):
                    raise ValueError(f"visualization sidecar {name} must have shape [M]")
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            if str(sidecar["base_npz_sha256"].item()) != digest:
                raise ValueError("visualization sidecar does not match universal NPZ bytes")
        return arrays, metadata, sidecar

    @staticmethod
    def _surface_channels(
        arrays  : Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
        sidecar : Mapping[str, np.ndarray],
    ) -> dict[str, np.ndarray]:
        """Expand structural and DNA tensors into aligned scalar point fields.

        Args:
            arrays: Universal NPZ arrays.
            metadata: Base scientific configuration and provenance.
            sidecar: Optional validated DNA annotation arrays.

        Returns:
            Full-order scalar arrays with PLY-safe names.
        """
        count         = len(arrays["surface_positions"])
        config        = cast(Mapping[str, Any], metadata.get("config", {}))
        probe_radius  = float(config.get("probe_radius", 1.4))
        resolution    = float(config.get("surface_resolution", 1.0))
        neighbors     = arrays["surface_atom_neighbors"]
        mask          = arrays["surface_atom_mask"]
        expanded      = arrays["vdw_radii"].astype(np.float64) + probe_radius
        gaps          = arrays["surface_atom_distances"][mask].astype(np.float64) - expanded[
            neighbors[mask]
        ]
        minimum_gaps  = np.full(count, np.inf)
        surface_ids   = np.repeat(np.arange(count), mask.sum(axis=1))
        np.minimum.at(minimum_gaps, surface_ids, gaps)

        channels: dict[str, np.ndarray] = {}
        if sidecar:
            channels["dna_target_hard"]    = sidecar["surface_target_hard"]
            channels["dna_target_soft"]    = sidecar["surface_target_soft"]
            channels["dna_target_valid"]   = sidecar["surface_valid_mask"].astype(np.uint8)
            channels["dna_distance"]       = sidecar["surface_distance_to_dna"]
            channels["dna_distance_valid"] = sidecar["surface_distance_valid"].astype(np.uint8)
            sensitivity = sidecar["surface_target_hard_sensitivity"]
            for index, cutoff in enumerate(sidecar["sensitivity_gaps"].tolist()):
                suffix = f"{float(cutoff):g}A".replace(".", "p")
                channels[f"dna_target_hard_gap_{suffix}"] = sensitivity[:, index]

        channels["signed_envelope_gap"] = minimum_gaps.astype(np.float32)
        channels["surface_area_weight"]  = arrays["surface_area_weights"]
        channels["normal_x"]             = arrays["surface_normals"][:, 0]
        channels["normal_y"]             = arrays["surface_normals"][:, 1]
        channels["normal_z"]             = arrays["surface_normals"][:, 2]

        curvatures = arrays["surface_curvatures"]
        scales     = config.get("curvature_scales", tuple(range(1, curvatures.shape[1] + 1)))
        for scale_index, scale in enumerate(scales):
            suffix = f"{resolution * float(scale):g}A".replace(".", "p")
            for channel_index, name in enumerate(
                ("mean_curvature", "gaussian_curvature", "curvedness")
            ):
                channels[f"{name}_{suffix}"] = curvatures[:, scale_index, channel_index]
        return channels

    @staticmethod
    def _diagnostics(
        arrays  : Mapping[str, np.ndarray],
        metadata: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Recompute checks for buried, flying, reversed, or unstable surface samples.

        Args:
            arrays: Universal atom, incidence, surface, and curvature arrays.
            metadata: Base configuration defining resolution, probe, and curvature scales.

        Returns:
            Human-readable counts, extrema, and PASS/FAIL verdict.
        """
        atoms      = arrays["atom_positions"].astype(np.float64)
        points     = arrays["surface_positions"].astype(np.float64)
        normals    = arrays["surface_normals"].astype(np.float64)
        neighbors  = arrays["surface_atom_neighbors"]
        neighbor_mask = arrays["surface_atom_mask"]
        config     = cast(Mapping[str, Any], metadata.get("config", {}))
        probe      = float(config.get("probe_radius", 1.4))
        resolution = float(config.get("surface_resolution", 1.0))
        expanded   = arrays["vdw_radii"].astype(np.float64) + probe
        surface_ids = np.repeat(np.arange(len(points)), neighbor_mask.sum(axis=1))
        atom_ids    = neighbors[neighbor_mask]
        distances   = arrays["surface_atom_distances"][neighbor_mask].astype(np.float64)
        gaps        = distances - expanded[atom_ids]
        minimum    = np.full(len(points), np.inf)
        np.minimum.at(minimum, surface_ids, gaps)
        tolerance = max(5.0e-4, 0.025 * resolution)
        interior  = int(np.count_nonzero(minimum < -tolerance))
        floating  = int(np.count_nonzero(minimum > tolerance))

        offsets   = points[surface_ids] - atoms[atom_ids]
        radial    = offsets / np.maximum(distances[:, None], 1.0e-12)
        smoothness = max(0.25 * resolution, 1.0e-3)
        active     = gaps <= minimum[surface_ids] + 2.5 * smoothness
        weights    = np.exp(-(gaps[active] - minimum[surface_ids[active]]) / smoothness)
        expected = np.zeros_like(points)
        np.add.at(expected, surface_ids[active], weights[:, None] * radial[active])
        expected /= np.maximum(np.linalg.norm(expected, axis=1, keepdims=True), 1.0e-12)
        cosines       = np.sum(expected * normals, axis=1)
        normal_errors = int(np.count_nonzero(cosines < 0.99))

        curvature = arrays["surface_curvatures"].astype(np.float64)
        mean      = curvature[:, :, 0]
        gaussian  = curvature[:, :, 1]
        curved    = curvature[:, :, 2]
        algebra_errors = int(
            np.size(mean)
            - np.count_nonzero(
                np.isclose(curved * curved, 2.0 * mean * mean - gaussian, rtol=2e-4, atol=2e-6)
            )
        )
        scales = np.asarray(
            config.get("curvature_scales", tuple(range(1, curvature.shape[1] + 1))),
            dtype=np.float64,
        )
        dimensionless = curved * (resolution * scales[None, :])
        unstable      = int(np.count_nonzero(dimensionless > 25.0))
        surface_mask = arrays["surface_neighbor_mask"]
        degree       = surface_mask.sum(axis=1)
        isolated = int(np.count_nonzero(degree == 0))
        valid    = not any((interior, floating, normal_errors, algebra_errors, unstable))
        return {
            "status":                          "PASS" if valid else "FAIL",
            "surface points":                  len(points),
            "interior points":                 interior,
            "floating points":                 floating,
            "normal orientation errors":       normal_errors,
            "minimum normal cosine":           f"{cosines.min():.6g}",
            "curvature identity errors":        algebra_errors,
            "unstable curvature values":       unstable,
            "maximum dimensionless curvature": f"{dimensionless.max():.6g}",
            "isolated bounded-neighbor points": isolated,
            "signed gap range (Å)":             f"{minimum.min():.5g} .. {minimum.max():.5g}",
            "surface components":               "see preprocessing validation report",
        }

    def _figure(
        self,
        arrays  : Mapping[str, np.ndarray],
        channels: Mapping[str, np.ndarray],
    ) -> tuple[go.Figure, dict[str, Any]]:
        """Build bounded WebGL traces and compact browser-side interaction data.

        Args:
            arrays: Universal atom, graph, and surface arrays.
            channels: Full-order scalar surface fields.

        Returns:
            Plotly figure and JSON-compatible trace, scalar, and inspector data.
        """
        atoms        = arrays["atom_positions"].astype(np.float64)
        points       = arrays["surface_positions"].astype(np.float64)
        normals      = arrays["surface_normals"].astype(np.float64)
        display_ids  = np.linspace(
            0, len(points) - 1, min(self.max_surface_points, len(points)), dtype=np.int64
        )
        mesh_ids     = np.linspace(
            0, len(points) - 1, min(self.max_mesh_points, len(points)), dtype=np.int64
        )
        normal_ids   = display_ids[:: self.normal_stride]
        default_surface = (
            "dna_target_hard"
            if "dna_target_hard" in channels
            else next(iter(channels))
        )

        # Atom scalars remain full-order. The van der Waals layer reuses these arrays through an
        # atom-to-icosahedron lookup rather than duplicating every channel in the HTML payload.

        atom_channels = {
            "atomic_number":   arrays["atomic_numbers"],
            "residue_type":    arrays["residue_type_ids"],
            "atom_role":       arrays["atom_role_ids"],
            "formal_charge":   arrays["formal_charges"],
            "chain_index":     arrays["chain_indices"],
            "residue_index":   arrays["residue_indices"],
            "van_der_waals_A": arrays["vdw_radii"],
        }
        surface_values = np.asarray(channels[default_surface])[display_ids]
        surface_range  = self._colour_range(surface_values)
        atom_values    = np.asarray(atom_channels["atomic_number"])
        atom_range     = self._colour_range(atom_values)
        mesh_faces, mesh_method = self._alpha_faces(points[mesh_ids], normals[mesh_ids])
        figure         = go.Figure()
        traces: dict[str, int] = {}

        traces["surface"] = len(figure.data)
        figure.add_trace(go.Scatter3d(
            x=points[display_ids, 0], y=points[display_ids, 1], z=points[display_ids, 2],
            mode="markers", name="Surface point cloud (authoritative)", customdata=display_ids,
            marker={"size": 4.0, "color": surface_values, "colorscale": "Turbo",
                    "cmin": surface_range[0], "cmax": surface_range[1], "opacity": 1.0,
                    "colorbar": {"title": {"text": default_surface.replace("_", " ")},
                                 "thickness": 14, "len": 0.62, "x": 1.01}},
            hovertemplate="surface %{customdata}<br>x=%{x:.3f} Å<br>y=%{y:.3f} Å<br>z=%{z:.3f} Å<br>value=%{marker.color:.5g}<extra></extra>",
        ))

        traces["mesh"] = len(figure.data)
        figure.add_trace(go.Mesh3d(
            x=points[mesh_ids, 0], y=points[mesh_ids, 1], z=points[mesh_ids, 2],
            i=mesh_faces[:, 0], j=mesh_faces[:, 1], k=mesh_faces[:, 2],
            intensity=np.zeros(len(mesh_ids)), intensitymode="vertex",
            colorscale=((0.0, "#59c3d1"), (1.0, "#59c3d1")), cmin=0.0, cmax=1.0,
            opacity=1.0,
            visible=False, showscale=False, name="Diagnostic alpha-complex mesh (derived)",
            hoverinfo="skip", flatshading=False,
            lighting={"ambient": 0.48, "diffuse": 0.78, "specular": 0.22,
                      "roughness": 0.58, "fresnel": 0.06},
        ))

        # Atom hover records expose stable atom IDs and chemical names. Marker radii are optimized
        # for selection; the separate van der Waals trace has physical radii in scene coordinates.

        atom_hover = np.column_stack((
            np.arange(len(atoms)).astype(str), arrays["atom_names"].astype(str),
            arrays["residue_names"].astype(str),
            arrays["residue_indices"].astype(str), arrays["chain_indices"].astype(str),
        ))
        traces["atoms"] = len(figure.data)
        figure.add_trace(go.Scatter3d(
            x=atoms[:, 0], y=atoms[:, 1], z=atoms[:, 2], mode="markers", name="Atoms",
            customdata=atom_hover,
            marker={"size": 4.5,
                    "color": atom_values, "colorscale": "Viridis",
                    "cmin": atom_range[0], "cmax": atom_range[1], "opacity": 0.88},
            hovertemplate="atom %{customdata[0]} · %{customdata[1]}<br>%{customdata[2]} %{customdata[3]} · chain index %{customdata[4]}<br>x=%{x:.3f} Å<br>y=%{y:.3f} Å<br>z=%{z:.3f} Å<br>value=%{marker.color:.5g}<extra></extra>",
        ))

        sphere_positions, sphere_faces, sphere_atom_ids = self._van_der_waals_mesh(
            atoms,
            arrays["vdw_radii"].astype(np.float64),
        )
        sphere_values = atom_values[sphere_atom_ids]
        traces["vdw"] = len(figure.data)
        figure.add_trace(go.Mesh3d(
            x=sphere_positions[:, 0], y=sphere_positions[:, 1], z=sphere_positions[:, 2],
            i=sphere_faces[:, 0], j=sphere_faces[:, 1], k=sphere_faces[:, 2],
            intensity=sphere_values, intensitymode="vertex", colorscale="Viridis",
            cmin=atom_range[0], cmax=atom_range[1], opacity=0.48, visible=False,
            showscale=False, name="Van der Waals envelopes", hoverinfo="skip",
            flatshading=False,
            lighting={"ambient": 0.42, "diffuse": 0.75, "specular": 0.22,
                      "roughness": 0.55, "fresnel": 0.12},
        ))

        relation_styles = {
            1: ("spatial",  "Atomic spatial edges",          "#98a2b3"),
            2: ("covalent", "Atomic covalent edges",         "#f04438"),
            3: ("both",      "Atomic spatial+covalent edges", "#fdb022"),
        }
        atom_edges = arrays["atom_edge_index"]
        covalent   = arrays["atom_edge_is_covalent"]
        spatial    = arrays["atom_edge_spatial_rank"] > 0
        relations  = spatial.astype(np.uint8) + 2 * covalent.astype(np.uint8)
        for relation, (key, name, colour) in relation_styles.items():
            x, y, z = self._edge_coordinates(atoms, atom_edges[:, relations == relation])
            traces[key] = len(figure.data)
            figure.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="lines", name=name,
                line={"color": colour, "width": 2}, visible=False, hoverinfo="skip",
            ))

        surface_mask      = arrays["surface_neighbor_mask"]
        surface_neighbors = arrays["surface_neighbors"]
        sources = np.repeat(np.arange(len(points)), surface_mask.sum(axis=1))
        targets = surface_neighbors[surface_mask]
        surface_edges = np.asarray((sources, targets), dtype=np.int32)
        x, y, z = self._edge_coordinates(points, surface_edges)
        traces["surface_edges"] = len(figure.data)
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name="Bounded surface neighbours",
            line={"color": "#53b1fd", "width": 1}, visible=False, hoverinfo="skip",
        ))

        normal_end       = points[normal_ids] + self.normal_length * normals[normal_ids]
        normal_positions = np.concatenate((points[normal_ids], normal_end))
        normal_edges     = np.vstack((
            np.arange(len(normal_ids)), np.arange(len(normal_ids)) + len(normal_ids)
        ))
        x, y, z = self._edge_coordinates(normal_positions, normal_edges)
        traces["normals"] = len(figure.data)
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name="Outward surface normals",
            line={"color": "#f5f7ff", "width": 2}, visible=False, hoverinfo="skip",
        ))

        x, y, z = self._cartoon_coordinates(arrays, atoms)
        traces["cartoon"] = len(figure.data)
        figure.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="lines", name="C-alpha backbone trace",
            line={"color": "#f8fafc", "width": 7}, hoverinfo="skip",
        ))

        traces["measurement"] = len(figure.data)
        figure.add_trace(go.Scatter3d(
            x=[], y=[], z=[], mode="lines+markers", name="Distance measurement",
            line={"color": "#fdb022", "width": 5}, marker={"color": "#fdb022", "size": 7},
            hoverinfo="skip", showlegend=False,
        ))

        figure.update_layout(
            template="plotly_dark", paper_bgcolor="#080c16", plot_bgcolor="#080c16",
            margin={"l": 0, "r": 0, "t": 0, "b": 0}, showlegend=False,
            uirevision="wisdom-protein-view",
            scene={
                "aspectmode": "data",
                "xaxis": {"title": "x (Å)", "showbackground": False, "gridcolor": "#283548"},
                "yaxis": {"title": "y (Å)", "showbackground": False, "gridcolor": "#283548"},
                "zaxis": {"title": "z (Å)", "showbackground": False, "gridcolor": "#283548"},
                "camera": {"eye": {"x": 1.5, "y": 1.5, "z": 1.1}},
                "dragmode": "orbit",
            },
        )

        return figure, {
            "surface": {
                name: {"cloud": self._json_values(np.asarray(values)[display_ids]),
                       "mesh": self._json_values(np.asarray(values)[mesh_ids]),
                       "range": list(self._colour_range(np.asarray(values)))}
                for name, values in channels.items()
            },
            "atoms": {
                name: {"values": self._json_values(np.asarray(values)),
                       "range": list(self._colour_range(np.asarray(values)))}
                for name, values in atom_channels.items()
            },
            "defaultSurface": default_surface,
            "defaultAtom":    "atomic_number",
            "traces":          traces,
            "traceCount":      len(figure.data),
            "sphereAtomIds":   sphere_atom_ids.tolist(),
            "surfaceIndices":  display_ids.tolist(),
            "meshVertexCount": len(mesh_ids),
            "meshTriangles":   len(mesh_faces),
            "meshMethod":      mesh_method,
            "vdwAtoms":        len(np.unique(sphere_atom_ids)),
            "atomCount":       len(atoms),
        }

    def _alpha_faces(
        self,
        points  : np.ndarray,
        normals : np.ndarray | None = None,
    ) -> tuple[np.ndarray, str]:
        """Construct one bounded 3-D alpha-complex boundary before browser rendering.

        Delaunay tetrahedra are retained when their circumsphere radius ``r`` satisfies
        ``r <= mesh_alpha``. A triangle belongs to the boundary exactly when it occurs in one
        retained tetrahedron. This makes layer toggling a visibility change instead of asking the
        browser to recompute a hull. If the alpha complex is empty or numerically degenerate, the
        convex hull is returned as an explicitly less detailed diagnostic fallback.

        Args:
            points: Deterministic surface subset with shape ``[M,3]`` in ångströms.
            normals: Optional outward unit normals with shape ``[M,3]``. When present, every
                boundary triangle is wound so that its cross-product normal agrees with the mean
                stored normal at its vertices. A centroid direction is used only when that mean is
                numerically undefined.

        Returns:
            Triangle vertex indices with shape ``[F,3]`` and dtype ``int32`` plus either
            ``alpha-complex`` or the explicit ``convex-hull fallback`` method label.

        Raises:
            ValueError: If fewer than four non-coplanar points prevent any diagnostic mesh.
        """
        if len(points) < 4:
            raise ValueError("at least four surface points are required for a diagnostic mesh")

        try:
            tetrahedra = Delaunay(points, qhull_options="QJ").simplices
            vertices   = points[tetrahedra]
            matrices   = 2.0 * (vertices[:, 1:] - vertices[:, :1])
            right      = np.sum(vertices[:, 1:] ** 2, axis=2) - np.sum(
                vertices[:, :1] ** 2,
                axis=2,
            )
            determinant = np.abs(np.linalg.det(matrices))
            scale       = max(1.0, float(np.max(np.abs(matrices))))
            solvable    = determinant > 1.0e-12 * scale**3
            centres     = np.full((len(tetrahedra), 3), np.nan)
            centres[solvable] = np.linalg.solve(
                matrices[solvable],
                right[solvable, ..., None],
            )[..., 0]
            radii = np.linalg.norm(centres - vertices[:, 0], axis=1)
            kept    = tetrahedra[np.isfinite(radii) & (radii <= self.mesh_alpha)]
        except (QhullError, np.linalg.LinAlgError):
            kept = np.empty((0, 4), dtype=np.int64)

        if len(kept):
            faces = np.concatenate((
                kept[:, (0, 1, 2)], kept[:, (0, 1, 3)],
                kept[:, (0, 2, 3)], kept[:, (1, 2, 3)],
            ))
            canonical       = np.sort(faces, axis=1)
            _, first, count = np.unique(canonical, axis=0, return_index=True, return_counts=True)
            boundary        = faces[first[count == 1]]
            if len(boundary):
                oriented = self._orient_faces(points, boundary, normals)
                if len(oriented):
                    return oriented, "alpha-complex"

        try:
            faces = ConvexHull(points, qhull_options="QJ").simplices.astype(np.int32)
            oriented = self._orient_faces(points, faces, normals)
            return oriented, "convex-hull fallback"
        except QhullError as error:
            raise ValueError("surface points cannot form a diagnostic 3-D mesh") from error

    @staticmethod
    def _orient_faces(
        points  : np.ndarray,
        faces   : np.ndarray,
        normals : np.ndarray | None,
    ) -> np.ndarray:
        """Remove degenerate triangles and orient the remaining faces outwards.

        For a triangle ``(a,b,c)``, its geometric normal is
        ``n_f = (b-a) x (c-a)``. The face is reversed to ``(a,c,b)`` whenever
        ``n_f . n_ref < 0``, where ``n_ref`` is the mean stored surface normal at the three
        vertices. This consistent winding lets WebGL apply lighting and back/front depth tests
        without adjacent triangles behaving as opposite-facing sheets.

        Args:
            points: Mesh vertices with shape ``[M,3]`` in ångströms.
            faces: Candidate triangle indices with shape ``[F,3]``.
            normals: Optional outward vertex normals with shape ``[M,3]``.

        Returns:
            Non-degenerate, consistently wound triangle indices with shape ``[F',3]`` and dtype
            ``int32``. If stored normals cancel at one face, its direction from the point-cloud
            centroid supplies the reference orientation.
        """
        triangles = points[faces]
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        areas = np.linalg.norm(face_normals, axis=1)
        valid = areas > 1.0e-12

        oriented     = faces[valid].astype(np.int32, copy=True)
        face_normals = face_normals[valid]
        triangles    = triangles[valid]

        if normals is None:
            references = np.mean(triangles, axis=1) - np.mean(points, axis=0)
        else:
            references = np.mean(normals[oriented], axis=1)
            weak       = np.linalg.norm(references, axis=1) <= 1.0e-12
            references[weak] = (
                np.mean(triangles[weak], axis=1) - np.mean(points, axis=0)
            )

        reverse = np.einsum("ij,ij->i", face_normals, references) < 0.0
        second  = oriented[reverse, 1].copy()
        oriented[reverse, 1] = oriented[reverse, 2]
        oriented[reverse, 2] = second
        return oriented

    def _van_der_waals_mesh(
        self,
        atoms: np.ndarray,
        radii: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Approximate physical van der Waals envelopes with one efficient icosahedral mesh.

        Each retained atom receives a regular icosahedron whose vertices lie exactly one stored
        van der Waals radius from the atomic centre. A deterministic subset is used only when the
        configured limit is exceeded; the ordinary atom layer always retains every atom.

        Args:
            atoms: Atomic Cartesian coordinates with shape ``[N,3]`` in ångströms.
            radii: Stored van der Waals radii with shape ``[N]`` in ångströms.

        Returns:
            Vertex coordinates ``[12K,3]``, faces ``[20K,3]``, and one source-atom index per
            vertex ``[12K]``.
        """
        atom_ids = np.linspace(
            0,
            len(atoms) - 1,
            min(self.max_vdw_atoms, len(atoms)),
            dtype=np.int64,
        )
        phi = (1.0 + np.sqrt(5.0)) / 2.0
        unit_vertices = np.asarray([
            (-1, phi, 0), (1, phi, 0), (-1, -phi, 0), (1, -phi, 0),
            (0, -1, phi), (0, 1, phi), (0, -1, -phi), (0, 1, -phi),
            (phi, 0, -1), (phi, 0, 1), (-phi, 0, -1), (-phi, 0, 1),
        ], dtype=np.float64)
        unit_vertices /= np.linalg.norm(unit_vertices, axis=1, keepdims=True)
        unit_faces = np.asarray([
            (0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11),
            (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8),
            (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9),
            (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1),
        ], dtype=np.int32)

        positions = (
            atoms[atom_ids, None, :]
            + radii[atom_ids, None, None] * unit_vertices[None, :, :]
        ).reshape(-1, 3)
        offsets = 12 * np.arange(len(atom_ids), dtype=np.int32)
        faces   = (unit_faces[None, :, :] + offsets[:, None, None]).reshape(-1, 3)
        sources = np.repeat(atom_ids, 12)
        return positions, faces, sources

    def _edge_coordinates(
        self,
        positions: np.ndarray,
        edges    : np.ndarray,
    ) -> tuple[list[float | None], list[float | None], list[float | None]]:
        """Convert a bounded edge subset into separated Plotly line coordinates.

        Args:
            positions: Cartesian vertex coordinates with shape ``[N,3]`` in ångströms.
            edges: Integer endpoints with shape ``[2,E]``.

        Returns:
            x, y, and z lists where ``None`` separates segments.
        """
        if edges.shape[1] > self.max_edges:
            indices = np.linspace(0, edges.shape[1] - 1, self.max_edges, dtype=np.int64)
            edges   = edges[:, indices]
        coordinates: list[list[float | None]] = [[], [], []]
        for source, target in edges.T:
            for axis in range(3):
                coordinates[axis].extend(
                    (float(positions[source, axis]), float(positions[target, axis]), None)
                )
        return coordinates[0], coordinates[1], coordinates[2]

    @staticmethod
    def _cartoon_coordinates(
        arrays: Mapping[str, np.ndarray],
        atoms : np.ndarray,
    ) -> tuple[list[float | None], list[float | None], list[float | None]]:
        """Join consecutive close C-alpha atoms into chain-respecting backbone segments.

        Args:
            arrays: Atom names, residue indices, and chain indices.
            atoms: Cartesian atom coordinates with shape ``[N,3]`` in ångströms.

        Returns:
            x, y, and z lists separated between independent segments.
        """
        alpha_ids = np.flatnonzero(arrays["atom_names"].astype(str) == "CA")
        coordinates: list[list[float | None]] = [[], [], []]
        for left, right in pairwise(alpha_ids):
            if (
                arrays["chain_indices"][left] == arrays["chain_indices"][right]
                and arrays["residue_indices"][right] - arrays["residue_indices"][left] == 1
                and np.linalg.norm(atoms[right] - atoms[left]) < 5.0
            ):
                for axis in range(3):
                    coordinates[axis].extend(
                        (float(atoms[left, axis]), float(atoms[right, axis]), None)
                    )
        return coordinates[0], coordinates[1], coordinates[2]

    @staticmethod
    def _colour_range(values: np.ndarray) -> tuple[float, float]:
        """Return robust finite colour limits while preserving constant channels.

        Args:
            values: Numeric scalar field; NaNs may mark unavailable measurements.

        Returns:
            First/99th-percentile bounds or a unit-width interval for constants/no data.
        """
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        if not len(finite):
            return 0.0, 1.0
        lower, upper = np.quantile(finite, (0.01, 0.99)).tolist()
        if lower == upper:
            return float(lower - 0.5), float(upper + 0.5)
        return float(lower), float(upper)

    @staticmethod
    def _json_values(values: np.ndarray) -> list[float | None]:
        """Convert a numeric channel to strict JSON while retaining missing-value positions.

        Args:
            values: Numeric scalar field whose NaNs encode unavailable measurements.

        Returns:
            Float values with every non-finite entry represented by JSON ``null``.
        """
        flattened = np.asarray(values, dtype=np.float64).reshape(-1)
        return [float(value) if np.isfinite(value) else None for value in flattened]

    @staticmethod
    def _inventory(
        arrays : Mapping[str, np.ndarray],
        sidecar: Mapping[str, np.ndarray],
    ) -> list[dict[str, str]]:
        """Summarize every base and annotation array for the HTML sidebar.

        Args:
            arrays: Complete universal NPZ arrays.
            sidecar: Optional DNA annotation arrays.

        Returns:
            Ordered names, shapes, dtypes, ranges, and bounded samples.
        """
        inventory: list[dict[str, str]] = []
        combined = {
            **{f"base/{name}": value for name, value in arrays.items()},
            **{f"dna/{name}": value for name, value in sidecar.items()},
        }
        for name in sorted(combined):
            array     = combined[name]
            flattened = array.reshape(-1)
            sample    = flattened[: min(8, len(flattened))].tolist()
            if np.issubdtype(array.dtype, np.number) and len(flattened):
                finite = flattened[np.isfinite(flattened)]
                summary = (
                    f"min={np.min(finite):.6g}, max={np.max(finite):.6g}, "
                    f"mean={np.mean(finite):.6g}; sample={sample}"
                    if len(finite)
                    else f"no finite values; sample={sample}"
                )
            else:
                summary = f"sample={sample}"
            inventory.append({
                "name": name, "shape": str(array.shape),
                "dtype": str(array.dtype), "summary": summary,
            })
        return inventory

    @staticmethod
    def _escape(value: object) -> str:
        """Escape display text before insertion into static HTML.

        Args:
            value: Object converted to report text.

        Returns:
            HTML-safe string.
        """
        return (
            str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&#39;")
        )

    @staticmethod
    def _html(
        identifier      : str,
        status          : str,
        diagnostic_rows : str,
        inventory_rows  : str,
        provenance      : str,
        plot            : str,
        controls        : str,
    ) -> str:
        """Compose the final report shell and browser channel/view interactions.

        Args:
            identifier: Escaped member ID.
            status: Automatic diagnostic verdict.
            diagnostic_rows: Escaped diagnostic table rows.
            inventory_rows: Escaped array inventory rows.
            provenance: Escaped formatted provenance JSON.
            plot: Plotly WebGL div and loader script.
            controls: Safe compact JSON containing scalar channels.

        Returns:
            Complete HTML document.
        """
        status_class = "ok" if status == "PASS" else "bad"
        template = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>WISDOM protein inspector · @@IDENTIFIER@@</title>
<style>
:root{color-scheme:dark;--bg:#080c16;--panel:#111827;--panel2:#182230;--line:#2d3b50;--text:#e7edf7;--muted:#98a2b3;--accent:#45c5d5;--accent2:#7f8cff;--warn:#fdb022;--bad:#ff6b6b;--ok:#65d68a;font:14px Inter,ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
*{box-sizing:border-box}body{margin:0;height:100vh;overflow:hidden;background:var(--bg);color:var(--text);display:grid;grid-template-columns:320px minmax(420px,1fr) 390px;transition:grid-template-columns .2s ease}body.left-wide{grid-template-columns:460px minmax(420px,1fr) 390px}body.right-wide{grid-template-columns:320px minmax(420px,1fr) 560px}body.left-wide.right-wide{grid-template-columns:460px minmax(420px,1fr) 560px}body.left-closed{grid-template-columns:0 minmax(420px,1fr) 390px}body.right-closed{grid-template-columns:320px minmax(420px,1fr) 0}body.left-closed.right-closed{grid-template-columns:0 minmax(420px,1fr) 0}body.left-closed.right-wide{grid-template-columns:0 minmax(420px,1fr) 560px}body.right-closed.left-wide{grid-template-columns:460px minmax(420px,1fr) 0}
button,select,input{font:inherit}button,select,input[type=number],input[type=color]{border:1px solid #3b4a61;background:#1b2638;color:var(--text);border-radius:8px;padding:7px 9px}input[type=range]{width:100%;accent-color:var(--accent)}input[type=color]{width:100%;height:36px;padding:4px;cursor:pointer}input:disabled{opacity:.42;cursor:not-allowed}output{color:#d9faff;font-variant-numeric:tabular-nums}button{cursor:pointer;transition:border-color .15s,background .15s,transform .08s}button:hover{border-color:var(--accent);background:#24344d}button:active{transform:translateY(1px)}button.active{background:#173e49;border-color:var(--accent);color:#dffcff}button.icon{padding:5px 8px;min-width:32px}.sidebar{min-width:0;overflow:hidden;background:linear-gradient(180deg,#111827,#0e1625);border-color:var(--line);transition:opacity .16s,transform .2s}.sidebar-inner{height:100%;overflow:auto;padding:14px}.left-panel{border-right:1px solid var(--line)}.right-panel{border-left:1px solid var(--line)}body.left-closed .left-panel{opacity:0;pointer-events:none;transform:translateX(-100%)}body.right-closed .right-panel{opacity:0;pointer-events:none;transform:translateX(100%)}
.panel-head,.viewer-head,.row,.control-head{display:flex;align-items:center;gap:8px}.panel-head{position:sticky;top:-14px;z-index:4;margin:-14px -14px 12px;padding:13px 14px;background:rgba(17,24,39,.96);border-bottom:1px solid var(--line)}.panel-head h1{font-size:15px;margin:0;flex:1}.viewer-head{height:48px;padding:0 14px;border-bottom:1px solid var(--line);background:rgba(12,18,30,.96)}.viewer-head strong{font-size:15px}.badge{font-size:11px;padding:3px 7px;border-radius:999px;background:#243044;color:#cbd5e1}.badge.ok{background:#123923;color:#8ce6a7}.badge.bad{background:#482020;color:#ff9b9b}.viewer-state{margin-left:auto;color:var(--muted);font-size:12px}.viewer-state.busy{color:var(--warn)}
#viewer{min-width:0;min-height:0;display:flex;flex-direction:column;position:relative;background:radial-gradient(circle at 50% 42%,#121d2e 0,#080c16 70%)}#wisdom-plot{flex:1;min-height:0}.floating-open{display:none;position:absolute;z-index:10;top:58px;box-shadow:0 8px 24px #0008}.floating-open.left{left:10px}.floating-open.right{right:10px}body.left-closed .floating-open.left,body.right-closed .floating-open.right{display:block}.loading-shade{position:absolute;inset:48px 0 0;z-index:9;background:#080c1688;display:none;place-items:center;pointer-events:none}.loading-shade.visible{display:grid}.loading-card{padding:10px 14px;background:#101a2b;border:1px solid var(--line);border-radius:10px;color:#d9e2f2}
.section{border:1px solid var(--line);border-radius:11px;background:#121c2c;margin:10px 0;overflow:hidden}.section>summary{cursor:pointer;list-style:none;padding:11px 12px;font-weight:650;display:flex;align-items:center;justify-content:space-between}.section>summary::-webkit-details-marker{display:none}.section>summary::after{content:'⌄';color:var(--muted)}.section[open]>summary::after{content:'⌃'}.section-body{padding:0 12px 12px}.preset-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.layer-list{display:grid;gap:7px}.check{display:flex;align-items:center;gap:8px;padding:5px 0;color:#d8e0ec}.check input{accent-color:var(--accent);width:16px;height:16px}.field{display:grid;gap:5px;margin:9px 0}.field>span,.label{font-size:12px;color:var(--muted)}.range-grid{display:grid;grid-template-columns:1fr 1fr auto;gap:6px;align-items:end}.range-grid input{min-width:0;width:100%}.inline-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px}.camera-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px}.small{font-size:12px;color:var(--muted);line-height:1.45}.notice{padding:10px;border-left:3px solid var(--warn);background:#2d2415;border-radius:6px;font-size:12px;line-height:1.45}.measure-result{margin-top:8px;padding:9px;background:#172236;border-radius:8px;min-height:37px;font-variant-numeric:tabular-nums}.measure-result strong{color:#ffd27a}
table{border-collapse:collapse;width:100%;font-size:12px}th,td{border-bottom:1px solid #28374c;padding:6px 5px;text-align:left;vertical-align:top}th{color:#aebbd0;font-weight:600}.array-table{min-width:680px}.table-scroll{overflow:auto;max-height:54vh}code,pre{white-space:pre-wrap;word-break:break-word;color:#c5e1ff;font:12px ui-monospace,SFMono-Regular,Consolas,monospace}.selection-card{padding:10px;border-radius:8px;background:#172236}.selection-card h3{font-size:13px;margin:0 0 8px}.selection-table td:first-child{color:var(--muted);width:48%}.footer-note{font-size:11px;color:#7f8da3;margin:10px 0 2px}
@media(max-width:1100px){body,body.left-wide,body.right-wide,body.left-wide.right-wide{grid-template-columns:minmax(0,1fr)}.sidebar{position:fixed;top:0;bottom:0;z-index:30;width:min(420px,92vw);box-shadow:0 0 32px #000a}.left-panel{left:0}.right-panel{right:0}.right-panel,.left-panel{transform:translateX(0)}body.left-closed .left-panel{transform:translateX(-105%)}body.right-closed .right-panel{transform:translateX(105%)}.floating-open{position:fixed}.panel-wide{display:none}}
</style></head><body>
<aside id="controls-panel" class="sidebar left-panel"><div class="sidebar-inner">
  <header class="panel-head"><h1>Representations</h1><button class="icon panel-wide" data-panel="left" data-action="wide" title="Expand or contract controls">↔</button><button class="icon" data-panel="left" data-action="close" title="Close controls">&times;</button></header>
  <details class="section" open><summary>View presets</summary><div class="section-body preset-grid">
    <button data-preset="surface">Surface</button><button data-preset="mesh">Solid mesh</button><button data-preset="atoms">Atoms</button><button data-preset="vdw">Van der Waals</button><button data-preset="cartoon">Backbone</button><button data-preset="combined" class="active">Combined</button><button data-preset="bonds">Atomic bonds</button><button data-preset="graphs">All graphs</button>
  </div></details>
  <details class="section" open><summary>Layers</summary><div class="section-body layer-list">
    <label class="check"><input type="checkbox" data-layer="surface" checked>Authoritative surface points</label>
    <label class="check"><input type="checkbox" data-layer="mesh">Derived alpha-complex mesh</label>
    <label class="check"><input type="checkbox" data-layer="atoms">Atoms</label>
    <label class="check"><input type="checkbox" data-layer="vdw">Van der Waals envelopes</label>
    <label class="check"><input type="checkbox" data-layer="cartoon" checked>C-alpha backbone trace</label>
    <label class="check"><input type="checkbox" data-layer="covalent">Covalent atom edges</label>
    <label class="check"><input type="checkbox" data-layer="spatial">Spatial atom edges</label>
    <label class="check"><input type="checkbox" data-layer="both">Combined-relation atom edges</label>
    <label class="check"><input type="checkbox" data-layer="surface_edges">Bounded surface neighbours</label>
    <label class="check"><input type="checkbox" data-layer="normals">Outward normals</label>
  </div></details>
  <details class="section" open><summary>Surface colouring</summary><div class="section-body">
    <label class="field"><span>Scalar channel</span><select id="surface-channel"></select></label>
    <div class="inline-grid"><label class="field"><span>Gradient</span><select id="surface-palette"></select></label><label class="check"><input id="surface-reverse" type="checkbox">Reverse gradient</label></div>
    <div class="range-grid"><label class="field"><span>Minimum</span><input id="surface-min" type="number" step="any"></label><label class="field"><span>Maximum</span><input id="surface-max" type="number" step="any"></label><button id="surface-auto">Auto</button></div>
    <div class="inline-grid"><label class="field"><span>Point size · <output id="surface-size-value">4.0 px</output></span><input id="surface-size" type="range" min="1" max="10" step="0.5" value="4"></label><label class="field"><span>Point opacity · <output id="surface-opacity-value">1.00</output></span><input id="surface-opacity" type="range" min="0.1" max="1" step="0.05" value="1"></label></div>
    <label class="field"><span>Mesh colouring</span><select id="mesh-colour-mode"><option value="uniform" selected>Uniform material</option><option value="channel">Selected surface channel</option></select></label>
    <div class="inline-grid"><label class="field"><span>Mesh material</span><input id="mesh-colour" type="color" value="#59c3d1"></label><label class="field"><span>Mesh opacity · <output id="mesh-opacity-value">1.00</output></span><input id="mesh-opacity" type="range" min="0.1" max="1" step="0.05" value="1"></label></div>
    <p class="small">Points and mesh are fully opaque by default. Rear points can still be seen through gaps between markers; increase point size or use the solid mesh for continuous occlusion. Choose channel colouring only when triangle colour variation is scientifically useful.</p>
  </div></details>
  <details class="section"><summary>Atom colouring</summary><div class="section-body">
    <label class="field"><span>Scalar channel</span><select id="atom-channel"></select></label>
    <div class="inline-grid"><label class="field"><span>Gradient</span><select id="atom-palette"></select></label><label class="check"><input id="atom-reverse" type="checkbox">Reverse gradient</label></div>
    <div class="range-grid"><label class="field"><span>Minimum</span><input id="atom-min" type="number" step="any"></label><label class="field"><span>Maximum</span><input id="atom-max" type="number" step="any"></label><button id="atom-auto">Auto</button></div>
  </div></details>
  <details class="section"><summary>Camera and scene</summary><div class="section-body">
    <div class="inline-grid"><label class="field"><span>Projection</span><select id="projection"><option value="perspective">Perspective</option><option value="orthographic">Orthographic</option></select></label><label class="check"><input id="show-axes" type="checkbox" checked>Show axes and grid</label></div>
    <div class="camera-grid"><button data-camera="front">Front</button><button data-camera="side">Side</button><button data-camera="top">Top</button><button data-camera="reset">Reset</button><button id="fit-view">Fit</button><button id="reset-scene">Defaults</button></div>
  </div></details>
  <details class="section" open><summary>Distance measurement</summary><div class="section-body">
    <button id="measure-toggle">Start measuring</button><button id="measure-clear">Clear</button>
    <div id="measure-result" class="measure-result">Select measurement mode, then click two atoms or surface points.</div>
    <p class="small">Distances are Euclidean centre-to-centre distances in ångströms. Orbiting remains available outside measurement clicks.</p>
  </div></details>
  <p class="footer-note">Mesh: <span id="mesh-method"></span>, <span id="mesh-count"></span> triangles · van der Waals: <span id="vdw-count"></span>/<span id="atom-count"></span> atoms.</p>
</div></aside>
<main id="viewer">
  <header class="viewer-head"><strong>@@IDENTIFIER@@</strong><span class="badge @@STATUS_CLASS@@">@@STATUS@@</span><span id="viewer-state" class="viewer-state">Ready</span></header>
  <button class="floating-open left" data-panel="left" data-action="open">Controls</button><button class="floating-open right" data-panel="right" data-action="open">Details</button>
  <div id="loading-shade" class="loading-shade"><div class="loading-card">Updating WebGL scene…</div></div>
  @@PLOT@@
</main>
<aside id="details-panel" class="sidebar right-panel"><div class="sidebar-inner">
  <header class="panel-head"><h1>Scientific inspection</h1><button class="icon panel-wide" data-panel="right" data-action="wide" title="Expand or contract details">↔</button><button class="icon" data-panel="right" data-action="close" title="Close details">&times;</button></header>
  <details class="section" open><summary>Selected element</summary><div class="section-body"><div id="selection-card" class="selection-card">Click an atom or authoritative surface point to inspect all its available attributes.</div></div></details>
  <details class="section" open><summary>Automatic geometry audit</summary><div class="section-body"><table><tbody>@@DIAGNOSTICS@@</tbody></table></div></details>
  <details class="section"><summary>Mesh interpretation</summary><div class="section-body"><p class="notice"><strong>Diagnostic only.</strong> The mesh is derived from a bounded point subset before the page is written. It can bridge pockets or omit regions and is never a model input. The point cloud and numerical audit remain authoritative.</p></div></details>
  <details class="section"><summary>NPZ arrays</summary><div class="section-body table-scroll"><table class="array-table"><thead><tr><th>Name</th><th>Shape</th><th>Dtype</th><th>Range / sample</th></tr></thead><tbody>@@INVENTORY@@</tbody></table></div></details>
  <details class="section"><summary>Dataset and provenance</summary><div class="section-body"><pre>@@PROVENANCE@@</pre></div></details>
</div></aside>
<script>
const C=@@CONTROLS@@;
const gd=document.getElementById('wisdom-plot');
const state={surface:C.defaultSurface,atom:C.defaultAtom,measure:false,measurePoints:[],selected:null,pending:0,queue:Promise.resolve()};
const palettes=['Turbo','Viridis','Cividis','Plasma','Magma','Inferno','RdBu','Portland','Jet','Greys'];
const presets={surface:['surface'],mesh:['mesh'],atoms:['atoms'],vdw:['vdw'],cartoon:['cartoon'],combined:['surface','cartoon'],bonds:['atoms','covalent','both'],graphs:['atoms','covalent','spatial','both','surface_edges']};
const uniformMesh=Array(C.meshVertexCount).fill(0);
const byId=id=>document.getElementById(id);
const human=value=>value.replaceAll('_',' ');
const finite=value=>Number.isFinite(Number(value));
const layer=name=>document.querySelector(`[data-layer="${name}"]`);
function fillOptions(select,names,selected){for(const name of names){const option=document.createElement('option');option.value=name;option.textContent=human(name);option.selected=name===selected;select.appendChild(option)}}
function setBusy(active,message='Updating'){state.pending+=active?1:-1;state.pending=Math.max(0,state.pending);byId('loading-shade').classList.toggle('visible',state.pending>0);byId('viewer-state').classList.toggle('busy',state.pending>0);byId('viewer-state').textContent=state.pending>0?message:'Ready'}
function schedule(operation,message){setBusy(true,message);state.queue=state.queue.then(operation,operation).catch(error=>{console.error(error);byId('viewer-state').textContent='Update failed — inspect console'}).finally(()=>setBusy(false));return state.queue}
function setRange(kind,range){byId(`${kind}-min`).value=Number(range[0]).toPrecision(6);byId(`${kind}-max`).value=Number(range[1]).toPrecision(6)}
function readRange(kind,fallback){const lower=Number(byId(`${kind}-min`).value),upper=Number(byId(`${kind}-max`).value);return finite(lower)&&finite(upper)&&lower<upper?[lower,upper]:fallback}
function updateStyleLabels(){byId('surface-size-value').textContent=`${Number(byId('surface-size').value).toFixed(1)} px`;byId('surface-opacity-value').textContent=Number(byId('surface-opacity').value).toFixed(2);byId('mesh-opacity-value').textContent=Number(byId('mesh-opacity').value).toFixed(2)}
fillOptions(byId('surface-channel'),Object.keys(C.surface),C.defaultSurface);fillOptions(byId('atom-channel'),Object.keys(C.atoms),C.defaultAtom);fillOptions(byId('surface-palette'),palettes,'Turbo');fillOptions(byId('atom-palette'),palettes,'Viridis');setRange('surface',C.surface[C.defaultSurface].range);setRange('atom',C.atoms[C.defaultAtom].range);
updateStyleLabels();
byId('mesh-method').textContent=C.meshMethod;byId('mesh-count').textContent=C.meshTriangles.toLocaleString();byId('vdw-count').textContent=C.vdwAtoms.toLocaleString();byId('atom-count').textContent=C.atomCount.toLocaleString();
function applyLayers(){const indices=[],values=[];document.querySelectorAll('[data-layer]').forEach(input=>{indices.push(C.traces[input.dataset.layer]);values.push(input.checked)});return Plotly.restyle(gd,{visible:values},indices).then(updateSurface)}
function choosePreset(name){const enabled=new Set(presets[name]);document.querySelectorAll('[data-layer]').forEach(input=>input.checked=enabled.has(input.dataset.layer));document.querySelectorAll('[data-preset]').forEach(button=>button.classList.toggle('active',button.dataset.preset===name));schedule(applyLayers,`Applying ${name} view`)}
function updateSurface(){const channel=C.surface[state.surface],range=readRange('surface',channel.range),scale=byId('surface-palette').value,reversed=byId('surface-reverse').checked,uniform=byId('mesh-colour-mode').value==='uniform',meshColour=byId('mesh-colour').value,surfaceVisible=layer('surface').checked,meshVisible=layer('mesh').checked;byId('mesh-colour').disabled=!uniform;updateStyleLabels();return Plotly.restyle(gd,{'marker.color':[channel.cloud],'marker.cmin':[range[0]],'marker.cmax':[range[1]],'marker.colorscale':[scale],'marker.reversescale':[reversed],'marker.size':[Number(byId('surface-size').value)],'marker.opacity':[Number(byId('surface-opacity').value)],'marker.showscale':[surfaceVisible],'marker.colorbar.title.text':[human(state.surface)]},[C.traces.surface]).then(()=>Plotly.restyle(gd,{intensity:[uniform?uniformMesh:channel.mesh],cmin:[uniform?0:range[0]],cmax:[uniform?1:range[1]],colorscale:[uniform?[[0,meshColour],[1,meshColour]]:scale],reversescale:[uniform?false:reversed],opacity:[Number(byId('mesh-opacity').value)],showscale:[!uniform&&meshVisible&&!surfaceVisible]},[C.traces.mesh]))}
function updateAtoms(){const channel=C.atoms[state.atom],range=readRange('atom',channel.range),scale=byId('atom-palette').value,reversed=byId('atom-reverse').checked,sphere=C.sphereAtomIds.map(index=>channel.values[index]);return Plotly.restyle(gd,{'marker.color':[channel.values],'marker.cmin':[range[0]],'marker.cmax':[range[1]],'marker.colorscale':[scale],'marker.reversescale':[reversed]},[C.traces.atoms]).then(()=>Plotly.restyle(gd,{intensity:[sphere],cmin:[range[0]],cmax:[range[1]],colorscale:[scale],reversescale:[reversed]},[C.traces.vdw]))}
function appendRow(table,name,value){const row=table.insertRow(),key=row.insertCell(),cell=row.insertCell();key.textContent=name;cell.textContent=value===null?'unavailable':String(value)}
function inspectPoint(point){const card=byId('selection-card'),table=document.createElement('table');table.className='selection-table';card.replaceChildren();const title=document.createElement('h3');if(point.curveNumber===C.traces.surface){const local=point.pointNumber,index=Number(point.customdata);title.textContent=`Surface point ${index}`;appendRow(table,'position',`${Number(point.x).toFixed(4)}, ${Number(point.y).toFixed(4)}, ${Number(point.z).toFixed(4)} Å`);for(const [name,channel] of Object.entries(C.surface))appendRow(table,human(name),channel.cloud[local])}else if(point.curveNumber===C.traces.atoms){const local=point.pointNumber,data=point.customdata;title.textContent=`Atom ${data[0]} · ${data[1]}`;appendRow(table,'residue',`${data[2]} ${data[3]}`);appendRow(table,'chain index',data[4]);appendRow(table,'position',`${Number(point.x).toFixed(4)}, ${Number(point.y).toFixed(4)}, ${Number(point.z).toFixed(4)} Å`);for(const [name,channel] of Object.entries(C.atoms))appendRow(table,human(name),channel.values[local])}else{return false}card.append(title,table);state.selected=point;return true}
function updateMeasurementTrace(){const coordinates=axis=>state.measurePoints.map(point=>point[axis]);return Plotly.restyle(gd,{x:[coordinates('x')],y:[coordinates('y')],z:[coordinates('z')],visible:[state.measurePoints.length>0]},[C.traces.measurement])}
function measurementClick(point){if(!state.measure||!inspectPoint(point))return;if(state.measurePoints.length===2)state.measurePoints=[];state.measurePoints.push({x:Number(point.x),y:Number(point.y),z:Number(point.z),label:point.curveNumber===C.traces.atoms?`atom ${point.customdata[0]}`:`surface ${point.customdata}`});if(state.measurePoints.length===1){byId('measure-result').textContent=`First point: ${state.measurePoints[0].label}. Select the second point.`}else{const [a,b]=state.measurePoints,distance=Math.hypot(a.x-b.x,a.y-b.y,a.z-b.z);byId('measure-result').replaceChildren();const strong=document.createElement('strong');strong.textContent=`${distance.toFixed(4)} Å`;byId('measure-result').append(strong,document.createTextNode(` · ${a.label} ↔ ${b.label}`))}schedule(updateMeasurementTrace,'Updating measurement')}
gd.on('plotly_click',event=>{const point=event.points[0];if(state.measure)measurementClick(point);else inspectPoint(point)});
gd.on('plotly_webglcontextlost',()=>{byId('viewer-state').textContent='WebGL context lost — reload this page';byId('viewer-state').classList.add('busy')});
document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>choosePreset(button.dataset.preset)));document.querySelectorAll('[data-layer]').forEach(input=>input.addEventListener('change',()=>{document.querySelectorAll('[data-preset]').forEach(button=>button.classList.remove('active'));schedule(applyLayers,'Updating layers')}));
byId('surface-channel').addEventListener('change',event=>{state.surface=event.target.value;setRange('surface',C.surface[state.surface].range);schedule(updateSurface,'Updating surface channel')});byId('atom-channel').addEventListener('change',event=>{state.atom=event.target.value;setRange('atom',C.atoms[state.atom].range);schedule(updateAtoms,'Updating atom channel')});
for(const id of ['surface-palette','surface-reverse','surface-min','surface-max','surface-size','surface-opacity','mesh-colour-mode','mesh-colour','mesh-opacity'])byId(id).addEventListener('change',()=>schedule(updateSurface,'Updating surface style'));for(const id of ['atom-palette','atom-reverse','atom-min','atom-max'])byId(id).addEventListener('change',()=>schedule(updateAtoms,'Updating atom colours'));
for(const id of ['surface-size','surface-opacity','mesh-opacity'])byId(id).addEventListener('input',updateStyleLabels);
byId('surface-auto').addEventListener('click',()=>{setRange('surface',C.surface[state.surface].range);schedule(updateSurface,'Resetting surface range')});byId('atom-auto').addEventListener('click',()=>{setRange('atom',C.atoms[state.atom].range);schedule(updateAtoms,'Resetting atom range')});
byId('measure-toggle').addEventListener('click',event=>{state.measure=!state.measure;event.currentTarget.classList.toggle('active',state.measure);event.currentTarget.textContent=state.measure?'Stop measuring':'Start measuring';byId('viewer-state').textContent=state.measure?'Measurement mode':'Ready'});byId('measure-clear').addEventListener('click',()=>{state.measurePoints=[];byId('measure-result').textContent='Select measurement mode, then click two atoms or surface points.';schedule(updateMeasurementTrace,'Clearing measurement')});
byId('projection').addEventListener('change',event=>schedule(()=>Plotly.relayout(gd,{'scene.camera.projection.type':event.target.value}),'Changing projection'));byId('show-axes').addEventListener('change',event=>{const visible=event.target.checked;const update={};for(const axis of ['xaxis','yaxis','zaxis']){update[`scene.${axis}.visible`]=visible}schedule(()=>Plotly.relayout(gd,update),'Updating axes')});
const cameras={front:{x:0,y:2.2,z:0},side:{x:2.2,y:0,z:0},top:{x:0,y:0.01,z:2.2},reset:{x:1.5,y:1.5,z:1.1}};document.querySelectorAll('[data-camera]').forEach(button=>button.addEventListener('click',()=>schedule(()=>Plotly.relayout(gd,{'scene.camera.eye':cameras[button.dataset.camera]}),'Moving camera')));byId('fit-view').addEventListener('click',()=>schedule(()=>Plotly.relayout(gd,{'scene.aspectmode':'data','scene.camera.eye':cameras.reset,'scene.camera.center':{x:0,y:0,z:0}}),'Fitting scene'));byId('reset-scene').addEventListener('click',()=>{byId('projection').value='perspective';byId('show-axes').checked=true;schedule(()=>Plotly.relayout(gd,{'scene.camera.projection.type':'perspective','scene.camera.eye':cameras.reset,'scene.xaxis.visible':true,'scene.yaxis.visible':true,'scene.zaxis.visible':true}),'Resetting scene')});
document.querySelectorAll('[data-panel]').forEach(button=>button.addEventListener('click',()=>{const side=button.dataset.panel,action=button.dataset.action;if(action==='close')document.body.classList.add(`${side}-closed`);if(action==='open')document.body.classList.remove(`${side}-closed`);if(action==='wide')document.body.classList.toggle(`${side}-wide`);requestAnimationFrame(()=>Plotly.Plots.resize(gd));setTimeout(()=>Plotly.Plots.resize(gd),240)}));
if(window.innerWidth<1100){document.body.classList.add('left-closed','right-closed')}
window.addEventListener('resize',()=>Plotly.Plots.resize(gd));
</script></body></html>"""
        replacements = {
            "@@IDENTIFIER@@":   identifier,
            "@@STATUS@@":       status,
            "@@STATUS_CLASS@@": status_class,
            "@@DIAGNOSTICS@@":  diagnostic_rows,
            "@@INVENTORY@@":    inventory_rows,
            "@@PROVENANCE@@":   provenance,
            "@@PLOT@@":         plot,
            "@@CONTROLS@@":     controls,
        }
        for token, value in replacements.items():
            template = template.replace(token, value)
        return template
