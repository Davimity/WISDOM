"""Native resumable offline VF references and base-bound, label-free query banks."""

import json
import time
import gemmi
import shutil
import hashlib
import tempfile
import importlib
import numpy as np

from typing import Any
from pathlib import Path
from functools import partial
from scipy.spatial import cKDTree
from collections.abc import Mapping
from scipy.interpolate import RegularGridInterpolator
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset
from wisdom.preprocessing.common.structure.assets.physics.FieldQueries import FieldQueries
from wisdom.preprocessing.common.structure.assets.physics.PhysicsConfig import PhysicsConfig
from wisdom.preprocessing.common.structure.assets.physics.HeavyAtomPolicy import HeavyAtomPolicy
from wisdom.preprocessing.common.structure.assets.physics.ProbeInteraction import ProbeInteraction
from wisdom.preprocessing.common.structure.assets.physics.PreparationAdmission import (
    PreparationAdmission,
)
from wisdom.preprocessing.common.structure.assets.physics.MolecularPreparation import (
    MolecularPreparation,
)
from wisdom.preprocessing.common.structure.assets.physics.PhysicalReferenceUnavailable import (
    PhysicalReferenceUnavailable,
)


class PhysicalFields:
    """Publish three computed fields independently of the base NPZ, labels and preserved SLAE."""

    REFERENCE = "physical_reference"
    QUERIES   = "physical_queries"
    ALGORITHM = "vf-protein-only-amber14-apbs-lj-v1"

    def __init__(self, work: Any, options: Mapping[str, Any],
                 maximum_voxels: int = 2000000) -> None:
        """Resolve declared scientific options and tools before entering a protein map.

        Args:
            work: Active LF Work; owns phase files, dependency recovery, tools and logs.
            options: PhysicsConfig keyword values; no execution fields or arbitrary tool hooks.
            maximum_voxels: Operational hard allocation ceiling per APBS grid, not a scientific
                approximation. An oversized grid fails; its spacing is never silently coarsened.
        """
        self.work = work
        self.config = PhysicsConfig(**options)
        self.reference_parameters = self.config.parameters()
        self.reference_parameters.pop("bank_size")
        self.preparation = MolecularPreparation(work, self.config)
        self.grid_module = importlib.import_module("gridData")
        self.maximum_voxels = maximum_voxels
        self.report: dict[str, Any] = {}

    def generate(self, bases: Mapping[str, Path], workers: int,
                 existing: Mapping[str, Mapping[str, Path]] | None = None
                 ) -> dict[str, dict[str, Path]]:
        """Restore complete physical phase files and compute only missing/invalid references.

        Args:
            bases: Exact universal NPZ per member, never labels or task-bound ligand structures.
            workers: Bounded native thread-map concurrency; each solver receives one CPU thread.
            existing: Parent assets, including optional exact sequence evidence and physical
                references. Bounded completion can re-audit unchanged strict fields without solves.

        Returns:
            Reference/query assets per original member, including explicit unavailable chemistry.
            Unknown/operational failures still block publication; valid native files are retained.

        Raises:
            RuntimeError: Execution, storage or independent physical validation fails.
        """
        self.work.log(f"VF: preparing/restoring three physical fields for {len(bases)} proteins; "
                      "base geometry and all inherited assets, including SLAE, remain unchanged")

        def process(identifier: str) -> dict[str, Any]:
            """Keep each immutable reference and its regenerable query bank separately resumable.

            Args:
                identifier: Exact structural member ID, not a source of scientific labels.

            Returns:
                Native validated paths and elapsed cost, or a recorded field/preparation failure.
            """
            began = time.perf_counter()
            base = bases[identifier]
            settings: dict[str, Any] = {"base": ProteinArchive.sha256(base),
                        "algorithm": self.ALGORITHM,
                        "parameters": self.reference_parameters,
                        "tools": self.preparation.provenance}
            assets = (existing or {}).get(identifier, {})
            bounded = HeavyAtomPolicy(self.config.heavy_atom_policy) == HeavyAtomPolicy.BOUNDED
            sequence = assets.get("sequence_context") if bounded else None
            if bounded:
                settings["sequence_context"] = (ProteinArchive.sha256(sequence)
                                                  if sequence is not None else None)
            key = hashlib.sha256(json.dumps(settings, sort_keys=True).encode()).hexdigest()
            self.work.log(f"VF {identifier}: checking physical reference and query checkpoints")
            build_options: dict[str, Any] = {"sequence_context": sequence,
                "inherited": assets.get(self.REFERENCE)} if bounded else {}
            validation_options: dict[str, Any] = {"sequence_context": sequence} if bounded else {}
            try:
                reference = self.work.checkpoints.file(f"physics/{key}/reference.npz",
                    build=partial(self.restore_reference, base, key=key, **build_options),
                    validate=partial(self.validate_reference, base=base,
                        parameters=self.reference_parameters, **validation_options))
                query_settings = {"reference": ProteinArchive.sha256(Path(reference)),
                                  "parameters": self.config.parameters()}
                query_key = hashlib.sha256(json.dumps(query_settings,
                    sort_keys=True).encode()).hexdigest()
                queries = self.work.checkpoints.file(f"physics/queries/{query_key}.npz",
                    build=partial(self.write_queries, base, reference=Path(reference)),
                    validate=partial(self.validate_queries, base=base,
                                     parameters=self.config.parameters()))
                with np.load(queries, allow_pickle=False) as stored:
                    support = stored["valid"].sum(axis=0).tolist()
                    header = json.loads(str(stored["metadata_json"].item()))
                self.work.log(f"VF {identifier}: complete; valid query counts "
                              f"potential/LJ1/LJ2={support}")
                return {"id": identifier, "reference": str(reference), "queries": str(queries),
                        "valid_queries": support, "availability": header.get("availability",
                            "available"), "reason": header.get("reason"),
                        "preparation_audit": header.get("preparation_audit", {}),
                        "potential_convergence": header.get("potential_convergence", {}),
                        "seconds": time.perf_counter() - began}
            except Exception as error:
                self.work.log(f"VF {identifier} failed: {type(error).__name__}: {error}")
                return {"id": identifier, "error": f"{type(error).__name__}: {error}",
                        "seconds": time.perf_counter() - began}

        rows = self.work.map(list(bases), process, workers=workers, executor="thread",
                             name="physical-fields")
        failed = [row for row in rows if "error" in row]
        self.report = {"total": len(bases), "completed": len(rows) - len(failed),
                       "failed": len(failed), "records": rows,
                       "parameters": self.config.parameters(),
                       "provenance": self.preparation.provenance,
                       "base_geometry_recomputed": False}
        self.report["unavailable"] = sum(row.get("availability") == "unavailable" for row in rows)
        self.report["field_proteins"] = {name: sum(row.get("valid_queries", [0, 0, 0])[i] > 0
            for row in rows) for i, name in enumerate(("potential", "lj1", "lj2"))}
        # Attempt artifacts may be cleaned after failure. Keep the diagnostic as a durable native
        # checkpoint too; no failure cache or project-owned recovery infrastructure is needed.

        self.work.checkpoints.save_json("physics/last-report.json", self.report)
        if failed:
            raise RuntimeError(f"VF references failed for {len(failed)}/{len(rows)} proteins; "
                               "completed native checkpoints were retained. Inspect the "
                               "physical-fields report; no partial dataset was published.")
        return {row["id"]: {self.REFERENCE: Path(row["reference"]),
                            self.QUERIES: Path(row["queries"])} for row in rows}

    def restore_reference(self, base: Path, output: Path, key: str,
                          sequence_context: Path | None = None,
                          inherited: Path | None = None) -> None:
        """Back durable per-Run checkpoints with LF's validated reconstructible file cache.

        Args:
            base: Exact immutable parent NPZ.
            output: Native checkpoint's unpublished destination.
            key: Scientific base/configuration/tool identity, unchanged for earlier valid results.
            sequence_context: Optional base-bound sequence evidence; included in bounded keys.
            inherited: Optional parent reference; reuse requires full validation and re-admission.

        Raises:
            Exception: Native restoration, reference computation or validation fails. Existing
                checkpoints are checked before this builder, so compatible recovery reuses them.
        """
        bounded = self.config.heavy_atom_policy == HeavyAtomPolicy.BOUNDED
        build_options: dict[str, Any] = {"sequence_context": sequence_context,
            "inherited": inherited} if bounded else {}
        validation_options: dict[str, Any] = ({"sequence_context": sequence_context}
                                               if bounded else {})
        cached = self.work.cache.file(f"physics/{key}/reference.npz",
            build=partial(self.write_reference, base, **build_options),
            validate=partial(self.validate_reference, base=base,
                parameters=self.reference_parameters, **validation_options))
        shutil.copyfile(cached, output)

    def solve(self, scratch: Path, positions: np.ndarray, spacing: float,
              stem: str, margin: float | None = None,
              fixed_delta: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
        """Run the declared APBS focusing solve and read its actual OpenDX grid geometry.

        Args:
            scratch: Native disposable directory containing prepared.pqr.
            positions: Prepared protein coordinates [N,3], Å; no probe charge is added.
            spacing: Requested fine-grid spacing in Å; actual spacing is stored from OpenDX.
            stem: Distinct reference/refinement filename stem inside scratch.
            margin: Optional box-margin contrast in Å; None uses the configured production box.
            fixed_delta: Optional exact recorded axis spacings [3], Å. Enlarged-box contrasts
                preserve these spacings and extend by whole multigrid blocks; otherwise box
                enlargement would accidentally test a coarser discretization as well.

        Returns:
            Potential [nx,ny,nz] in kBT/e, actual origin [3] Å, spacings [3] Å, and exact
            APBS input text. Finite-grid interpolation is not a higher-resolution measurement.

        Raises:
            ValueError: The selected nominal/refined/enlarged grid exceeds the operational voxel
                ceiling, or the solver output is ambiguous, invalid or explicitly nonconverged.
                Raising the ceiling admits the same grid; it never changes scientific settings.
        """
        length = np.ptp(positions, axis=0) + 2 * (
            self.config.margin if margin is None else margin)
        center = 0.5 * (positions.min(0) + positions.max(0))
        dimensions = (32 * np.ceil(length / spacing / 32) + 1).astype(int)
        if fixed_delta is not None:
            dimensions = (32 * np.ceil(length / fixed_delta / 32) + 1).astype(int)
            length = (dimensions - 1) * fixed_delta
        voxels = int(np.prod(dimensions))
        if voxels > self.maximum_voxels:
            raise ValueError(f"VF APBS {stem} grid {dimensions.tolist()} requires {voxels:,} "
                f"voxels, exceeding physics_maximum_voxels={self.maximum_voxels:,}; increase "
                "this operational guard explicitly, allowing for refinement/box audits too. "
                "Recover the same Work with --accept-operational-changes to reuse completed "
                "protein checkpoints; grid spacing is unchanged.")
        vectors = [" ".join(str(float(value)) for value in vector)
                   for vector in (length * self.config.coarse_factor, length, center)]
        equation = "npbe" if self.config.nonlinear else "lpbe"
        text = (
            "read\n mol pqr prepared.pqr\nend\nelec\n mg-auto\n"
            f" dime {' '.join(map(str, dimensions))}\n"
            f" cglen {vectors[0]}\n fglen {vectors[1]}\n"
            f" cgcent {vectors[2]}\n fgcent {vectors[2]}\n mol 1\n {equation}\n"
            f" bcfl {self.config.boundary_condition}\n"
            " chgm spl2\n srfm smol\n sdens 10\n swin 0.3\n"
            f" pdie {self.config.protein_dielectric}\n sdie {self.config.solvent_dielectric}\n"
            f" srad {self.config.solvent_radius}\n temp {self.config.temperature}\n"
            f" ion charge 1 conc {self.config.salt} radius 2\n"
            f" ion charge -1 conc {self.config.salt} radius 2\n"
            f" calcenergy no\n calcforce no\n write pot dx {stem}\nend\nquit\n"
        )
        source = scratch / f"{stem}.in"
        source.write_text(text)
        self.work.log(f"VF APBS {stem}: grid={dimensions.tolist()}, voxels={voxels:,}, "
                      f"operational ceiling={self.maximum_voxels:,}")
        result = self.work.tools.run((self.preparation.apbs, source), cwd=scratch,
            threads=1, name=f"vf-apbs-{stem}")
        diagnostic = (result.stdout + result.stderr).lower()
        if any(message in diagnostic for message in ("not converged", "failed to converge",
                                                     "error while", "error tolerance exceeded")):
            raise ValueError("VF APBS reported a failed/nonconverged solve; inspect tool log")
        paths = list(scratch.glob(f"{stem}*.dx"))
        if len(paths) != 1:
            raise ValueError("VF requires one serial APBS OpenDX potential, not guessed chunks")
        grid = self.grid_module.Grid(str(paths[0]))
        values = np.asarray(grid.grid, dtype=np.float64)
        if values.ndim != 3 or not np.isfinite(values).all():
            raise ValueError("VF APBS potential has invalid dimensions or non-finite values")
        return values, np.asarray(grid.origin), np.asarray(grid.delta), text

    def write_reference(self, base: Path, output: Path, sequence_context: Path | None = None,
                        inherited: Path | None = None) -> None:
        """Persist raw physical fields, chemistry and audit evidence outside the structural NPZ.

        Args:
            base: Exact immutable heavy-atom structural archive.
            output: LF private unpublished file; native checkpoints own atomic publication.
            sequence_context: Base-bound complete-chain evidence for bounded terminal repair.
            inherited: Optional validated strict reference whose field equations are unchanged.

        Raises:
            ValueError: LJ/OpenMM agreement, strict APBS convergence or structural integrity fails.
                Known unsupported chemistry instead writes an explicitly unavailable reference.
                Mask policy retains audited LJ when potential coverage is inadequate. Unknown
                tool failures propagate; no synthetic electrostatic fallback is used.
        """
        began = time.perf_counter()
        with tempfile.TemporaryDirectory(dir=self.work.temp_dir, prefix="vf-") as directory:
            scratch = Path(directory)
            self.work.log(f"VF {base.name}: preparing protein-only protonation and "
                          "force-field types")
            try:
                if inherited is not None and self.reuse_strict_reference(
                    base, output, inherited, sequence_context):
                    return
                if sequence_context is None:
                    prepared = self.preparation.prepare(base, scratch)
                else:
                    prepared = self.preparation.prepare(base, scratch,
                                                         sequence_context=sequence_context)
            except PhysicalReferenceUnavailable as error:
                # Keep the protein, labels, SLAE and base geometry. An empty explicitly marked
                # physical domain has no fabricated queries and cannot contribute a VF loss.

                StructuralAsset.write(output, base, {"valid": np.zeros((0, 3), dtype=bool)},
                    self.REFERENCE, self.reference_parameters,
                    {"algorithm": self.ALGORITHM, "provenance": self.preparation.provenance,
                     "units": ["kBT/e", "kJ/mol", "kJ/mol"],
                     "availability": "unavailable", "reason": str(error),
                     "preparation_audit": error.audit,
                     "sequence_context_sha256": (ProteinArchive.sha256(sequence_context)
                         if sequence_context is not None else None),
                     "reason_type": "unsupported_observed_chemistry"})
                self.work.log(f"VF {base.name}: all three fields unavailable; {error}")
                return
            self.work.log(f"VF {base.name}: solving APBS potential")
            potential, origin, delta, apbs_input = self.solve(
                scratch, prepared["positions"], self.config.spacing, "potential")
            shape = potential.shape
            axes = [origin[i] + np.arange(shape[i]) * delta[i] for i in range(3)]
            points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
            with np.load(base, allow_pickle=False) as stored:
                geometry = FieldQueries.domains(points, stored["atom_positions"],
                    stored["vdw_radii"], shape, self.config.solvent_radius,
                    self.config.connectivity, self.config.boundary_width,
                    self.config.exterior_width)
            interaction = ProbeInteraction(prepared["positions"], prepared["sigma"],
                                           prepared["epsilon"])
            targets = np.zeros((len(points), 3), dtype=np.float64)
            valid   = np.zeros_like(targets, dtype=bool)
            targets[:, 0] = potential.ravel()
            valid[:, 0], potential_audit, potential_error = self.audit_potential(scratch,
                prepared["positions"], potential, origin, delta, points, geometry)
            audits: list[dict[str, Any]] = []
            raw_lj = np.zeros((len(points), 2), dtype=np.float64)
            overlap_valid = np.zeros((len(points), 2), dtype=bool)
            for channel, probe in enumerate(self.preparation.probes, start=1):
                self.work.log(f"VF {base.name}: evaluating LJ{channel} at {len(points):,} grid "
                              "positions and auditing against OpenMM")
                pair = (probe["sigma_nm"], probe["epsilon_kj_mol"])
                values, available = interaction.evaluate(points, pair,
                    self.config.overlap_fraction, self.config.lj_cutoff)
                raw_lj[:, channel - 1] = values
                overlap_valid[:, channel - 1] = available
                available &= (geometry["distance"] >= 0) & (
                    np.abs(values) <= self.config.repulsion_ceiling)
                targets[:, channel] = values
                valid[:, channel] = available
                ids = FieldQueries.sample(geometry["region"], available,
                    self.config.audit_queries, self.config.seed + channel)
                if not len(ids):
                    raise ValueError(f"VF probe {channel} has no admissible audit positions")
                reference = interaction.reference(points[ids], pair, self.config.lj_cutoff)
                if not np.allclose(values[ids], reference,
                    atol=self.config.lj_absolute_tolerance, rtol=self.config.lj_relative_tolerance):
                    raise ValueError(f"VF LJ{channel} disagrees with OpenMM interaction groups")
                full, _ = interaction.evaluate(points[ids], pair, self.config.overlap_fraction)
                local, _ = interaction.evaluate(points[ids], pair, self.config.overlap_fraction,
                                                 self.config.decoder_radius)
                audits.append({"field": f"lj{channel}", "queries": len(ids),
                    "openmm_maximum_absolute_error": float(np.max(np.abs(values[ids] - reference))),
                    "configured_cutoff_maximum_error": float(np.max(np.abs(values[ids] - full))),
                    "decoder_radius_truncation_maximum_error": float(np.max(np.abs(local - full)))})
            targets[~valid] = 0.0
            arrays = {"potential": potential.astype(np.float32), "origin": origin, "delta": delta,
                "targets": targets.reshape((*shape, 3)).astype(np.float32),
                "valid": valid.reshape((*shape, 3)),
                "potential_numerical_error": potential_error.reshape(shape),
                "lj_energy": raw_lj.reshape((*shape, 2)),
                "lj_overlap_valid": overlap_valid.reshape((*shape, 2)),
                **{key: array.reshape(shape) for key, array in geometry.items()},
                "prepared_positions": prepared["positions"], "prepared_sigma": prepared["sigma"],
                "prepared_epsilon": prepared["epsilon"],
                "partial_charges": prepared["partial_charges"],
                "heavy_mapping": prepared["heavy_mapping"], "pqr_radii": prepared["pqr_radii"],
                "pqr_text": np.asarray((scratch / "prepared.pqr").read_text()),
                "prepared_pdb": np.asarray((scratch / "prepared.pdb").read_text()),
                "apbs_input": np.asarray(apbs_input),
                "open_dx": np.asarray(next(scratch.glob("potential*.dx")).read_text())}
            if self.config.heavy_atom_policy == HeavyAtomPolicy.BOUNDED:
                arrays["prepared_atomic_numbers"] = prepared["elements"]
                arrays["prepared_bonds"] = prepared["bonds"]
            StructuralAsset.write(output, base, arrays, self.REFERENCE, self.reference_parameters,
                {"algorithm": self.ALGORITHM, "provenance": self.preparation.provenance,
                 "units": ["kBT/e", "kJ/mol", "kJ/mol"],
                 "potential_to_volts": 8.617333262145e-5 * self.config.temperature,
                 "potential_to_kj_mol_for_plus_e": .00831446261815324 * self.config.temperature,
                 "heavy_identity_keys": prepared["original_heavy_keys"],
                 "prepared_identity_keys": prepared["prepared_keys"],
                 "coordinate_origin": prepared["coordinate_origin"],
                 "hydrogen_count": prepared["hydrogen_count"],
                 "net_charge_e": prepared["net_charge_e"],
                 "preparation_audit": prepared.get("preparation_audit", {}),
                 "sequence_context_sha256": (ProteinArchive.sha256(sequence_context)
                     if sequence_context is not None else None),
                 "lj_audits": audits, "potential_convergence": potential_audit,
                 "physical_accuracy_claim": "none; fixed-structure pilot with numerical audits"})
            self.work.log(f"VF {base.name}: reference validated in "
                          f"{time.perf_counter() - began:.1f}s")

    def reuse_strict_reference(self, base: Path, output: Path, inherited: Path,
                               sequence_context: Path | None) -> bool:
        """Re-admit valid strict parent fields through a typed dataset, without recomputation.

        Args:
            base: Exact unchanged universal NPZ.
            output: Native unpublished reference destination under the new bounded protocol.
            inherited: Parent's strict reference, not a guessed checkpoint path.
            sequence_context: Parent's base-bound deposited sequence evidence, if available.

        Returns:
            True after full source validation, unchanged equations/tool provenance and bounded
            admission. Unavailable or scientifically different references return False to rebuild.

        Raises:
            ValueError: An inherited archive is corrupt or loses observed correspondence.
            PhysicalReferenceUnavailable: Observed backbone fails the new admission policy.
        """
        values, header = StructuralAsset.read(inherited, base, self.REFERENCE)
        source_config = PhysicsConfig(**header["parameters"])
        if source_config.heavy_atom_policy != "strict" or (
            header.get("availability") == "unavailable"):
            return False
        unchanged = {key: value for key, value in self.reference_parameters.items()
                     if key != "heavy_atom_policy" and not key.startswith("repair_")}
        if header["parameters"] != unchanged or header.get("provenance") != (
            self.preparation.provenance):
            return False
        self.validate_reference(inherited, base)
        model = gemmi.read_structure_string(str(values["prepared_pdb"].item()),
                                            format=gemmi.CoorFormat.Pdb)[0]
        flat = [(c, r, a) for c in model for r in c for a in r]
        keys = [(c.name, int(r.seqid.num or 0), a.name) for c, r, a in flat]
        elements = np.asarray([a.element.atomic_number for _, _, a in flat], dtype=np.int16)
        with np.load(base, allow_pickle=False) as stored:
            arrays = {name: stored[name] for name in ("atom_positions", "atomic_numbers",
                      "atom_names", "residue_indices", "chain_indices")}
        sequence = StructuralAsset.read(sequence_context, base, "sequence_context")[0] if (
            sequence_context is not None) else None
        audit = PreparationAdmission(self.config).assess(arrays, keys,
            values["prepared_positions"], elements, sequence)
        if audit["rebuilt_indices"]:
            return False
        values["prepared_atomic_numbers"] = elements
        values["prepared_bonds"] = np.empty((0, 2), dtype=np.int32)
        audit["geometry"] = {"performed": False, "reason": "no modeled heavy atoms"}
        header.update({"preparation_audit": audit,
            "sequence_context_sha256": (ProteinArchive.sha256(sequence_context)
                if sequence_context is not None else None),
            "source_reference_sha256": ProteinArchive.sha256(inherited)})
        StructuralAsset.write(output, base, values, self.REFERENCE,
                              self.reference_parameters, header)
        self.work.log(f"VF {base.name}: reused validated strict parent fields; no physical solves")
        return True

    def audit_potential(self, scratch: Path, positions: np.ndarray, potential: np.ndarray,
                        origin: np.ndarray, delta: np.ndarray, points: np.ndarray,
                        geometry: Mapping[str, np.ndarray]
                        ) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
        """Compare actual identical positions and exclude numerically unstable supervision.

        Args:
            scratch: Native disposable directory containing the prepared PQR.
            positions: Prepared protein coordinates [N,3], Å.
            potential: Nominal APBS grid, native kBT/e.
            origin: Actual nominal grid origin [3], Å.
            delta: Actual nominal axis spacings [3], Å.
            points: C-order nominal grid positions [Q,3], Å, never moved to fit another grid.
            geometry: Heavy-sphere distance and region arrays [Q].

        Returns:
            Potential-valid mask [Q], complete-volume convergence/coverage facts and maximum
            absolute comparison errors [Q] in kBT/e. Outside common domains errors are stored
            as zero only with validity false; raw APBS values are always retained separately.

        Raises:
            ValueError: Strict policy encounters instability or insufficient coverage. Mask policy
                instead disables the whole potential when overall/per-region coverage fails.
                Numerical agreement is not physical-model accuracy.
        """
        domain = geometry["distance"] >= 0
        valid = domain.copy()
        errors = np.zeros(len(points), dtype=np.float32)
        audit: dict[str, Any] = {"performed": self.config.convergence, "comparisons": []}
        if not self.config.convergence:
            return valid, audit, errors
        if self.config.potential_failure_policy not in {"fail", "mask"}:
            raise ValueError("VF potential_failure_policy must be fail or mask")
        self.work.log("VF: checking refinement and box sensitivity over the complete physical grid")
        contrasts = [("refined", self.config.spacing * self.config.refinement_factor,
                      self.config.margin)]
        if self.config.margin_convergence:
            contrasts.append(("larger-box", self.config.spacing,
                              self.config.margin * self.config.margin_factor))
        for stem, spacing, margin in contrasts:
            field, other_origin, other_delta, input_text = self.solve(scratch, positions,
                spacing, stem, margin, fixed_delta=delta if stem == "larger-box" else None)
            axes = [other_origin[i] + np.arange(field.shape[i]) * other_delta[i]
                    for i in range(3)]
            # OpenDX rounds spacings. Points outside the ACTUAL common box are unavailable,
            # not extrapolated or shifted onto a different voxel to hide a boundary mismatch.

            compared = RegularGridInterpolator(axes, field, bounds_error=False,
                                               fill_value=np.nan)(points)
            common = domain & np.isfinite(compared)
            difference = np.abs(compared[common] - potential.ravel()[common])
            tolerance = self.config.potential_absolute_tolerance + (
                self.config.potential_relative_tolerance * np.abs(compared[common]))
            admitted = np.zeros(len(points), dtype=bool)
            admitted[common] = difference <= tolerance
            valid &= admitted
            errors[common] = np.maximum(errors[common], difference)
            audit["comparisons"].append({"name": stem, "input": input_text,
                "grid_shape": list(field.shape), "origin": other_origin.tolist(),
                "delta": other_delta.tolist(), "common_queries": int(common.sum()),
                "outside_common_domain": int((domain & ~common).sum()),
                "maximum_absolute_error_kbt_e": float(difference.max()),
                "mae_kbt_e": float(difference.mean()),
                "unstable_queries": int((common & ~admitted).sum())})
        coverage = float(valid.sum() / max(1, domain.sum()))
        region_coverage = {str(region): float(valid[geometry["region"] == region].mean())
                           for region in range(4) if np.any(geometry["region"] == region)}
        passed = coverage >= self.config.minimum_potential_coverage and all(
            value >= self.config.minimum_region_coverage for value in region_coverage.values())
        if self.config.potential_failure_policy == "fail":
            passed = passed and np.array_equal(domain, valid)
        audit.update({"passed": bool(passed), "policy": self.config.potential_failure_policy,
            "domain_queries": int(domain.sum()), "valid_queries": int(valid.sum()),
            "coverage": coverage, "region_coverage": region_coverage,
            "nominal_origin": origin.tolist(), "nominal_delta": delta.tolist()})
        self.work.log(f"VF potential supervision coverage={coverage:.3%}; "
                      f"per-region={region_coverage}; admitted={passed}")
        if not passed and self.config.potential_failure_policy == "fail":
            raise ValueError(f"VF potential failed its declared numerical coverage policy: {audit}")
        if not passed:
            # Mask the entire unqualified potential, not merely the difficult region. Independently
            # audited LJ fields remain usable, and raw APBS values/errors stay available to audit.

            audit["candidate_coverage"] = coverage
            audit["candidate_region_coverage"] = region_coverage
            audit["candidate_valid_queries"] = int(valid.sum())
            audit["valid_queries"] = 0
            audit["coverage"] = 0.0
            audit["region_coverage"] = {region: 0.0 for region in region_coverage}
            valid[:] = False
        return valid, audit, errors

    def write_queries(self, base: Path, output: Path, reference: Path) -> None:
        """Create a compact stratified bank with untruncated sparse local atom candidates.

        Args:
            base: Exact immutable universal atom domain and coordinate frame.
            output: LF-owned unpublished bank NPZ.
            reference: Validated reference NPZ; raw physical values are never model inputs.
        """
        from_reference, header = StructuralAsset.read(reference, base, self.REFERENCE)
        if header.get("availability") == "unavailable":
            StructuralAsset.write(output, base,
                {"positions": np.empty((0, 3), dtype=np.float32),
                 "targets": np.empty((0, 3), dtype=np.float32),
                 "valid": np.zeros((0, 3), dtype=bool), "region": np.empty(0, dtype=np.int8),
                 "distance": np.empty(0, dtype=np.float32),
                 "accessible": np.empty(0, dtype=bool), "grid_index": np.empty(0, dtype=np.int64),
                 "offsets": np.zeros(1, dtype=np.int64), "atom_ids": np.empty(0, dtype=np.int32)},
                self.QUERIES, self.config.parameters(),
                {"algorithm": self.ALGORITHM, "reference_sha256": ProteinArchive.sha256(reference),
                 "units": header["units"], "queries": 0, "availability": "unavailable",
                 "reason": header["reason"], "reason_type": header["reason_type"],
                 "preparation_audit": header.get("preparation_audit", {}),
                 "physical_sources_are_targets_only": True})
            return
        shape = from_reference["potential"].shape
        origin, delta = from_reference["origin"], from_reference["delta"]
        available = np.asarray(from_reference["valid"].reshape(-1, 3).any(axis=1), dtype=bool)
        rebuilt = header.get("preparation_audit", {}).get("rebuilt_indices", [])
        repair_positions = (from_reference["prepared_positions"][rebuilt] if rebuilt
                            else np.empty((0, 3)))
        if len(repair_positions):
            # Keep raw computed volumes, but avoid teaching exact near-field values around an
            # unobserved modeled atom. This does not prove that distant electrostatics is exact.

            tree = cKDTree(repair_positions)
            for start in range(0, len(available), 65536):
                addresses = np.arange(start, min(start + 65536, len(available)))
                index = np.stack(np.unravel_index(addresses, shape), axis=1)
                points = (origin + index * delta).astype(np.float32)
                available[addresses] &= tree.query(points)[0] >= self.config.repair_exclusion_radius
        ids = FieldQueries.sample(from_reference["region"].ravel(),
            available, self.config.bank_size, self.config.seed)
        index = np.stack(np.unravel_index(ids, shape), axis=1)
        points = (origin + index * delta).astype(np.float32)
        with np.load(base, allow_pickle=False) as stored:
            atoms = stored["atom_positions"]
        tree = cKDTree(atoms)
        nearby = tree.query_ball_point(points, self.config.decoder_radius, return_sorted=True)
        counts = np.asarray([len(row) for row in nearby], dtype=np.int64)
        values = {"positions": points.astype(np.float32),
            "targets": from_reference["targets"].reshape(-1, 3)[ids],
            "valid": from_reference["valid"].reshape(-1, 3)[ids],
            "region": from_reference["region"].ravel()[ids],
            "distance": from_reference["distance"].ravel()[ids],
            "accessible": from_reference["accessible"].ravel()[ids],
            "grid_index": ids.astype(np.int64), "offsets": np.r_[0, np.cumsum(counts)],
            "atom_ids": np.concatenate(nearby).astype(np.int32) if counts.sum() else
                        np.empty(0, dtype=np.int32)}
        StructuralAsset.write(output, base, values, self.QUERIES, self.config.parameters(),
            {"algorithm": self.ALGORITHM, "reference_sha256": ProteinArchive.sha256(reference),
             "units": header["units"], "queries": len(ids),
             "grid_shape": list(shape), "grid_origin": origin.tolist(),
             "grid_delta": delta.tolist(),
             "repair_exclusion_positions": repair_positions.tolist(),
             "potential_convergence": {key: header["potential_convergence"].get(key)
                 for key in ("passed", "coverage", "region_coverage", "candidate_coverage",
                             "candidate_region_coverage")},
             "region_counts": {str(i): int((values["region"] == i).sum()) for i in range(4)},
             "physical_sources_are_targets_only": True})

    @staticmethod
    def validate_reference(path: Path, base: Path,
                           parameters: Mapping[str, Any] | None = None,
                           sequence_context: Path | None = None) -> bool:
        """Audit exact base binding, physical units, masks and reference evidence on every reuse.

        Args:
            path: Independent physical reference NPZ, never the universal archive.
            base: Unchanged structural NPZ.
            parameters: Exact requested scientific settings; None audits the recorded settings.
            sequence_context: Optional exact sequence asset to recheck modeled terminal evidence.

        Returns:
            True only after complete storage checks and explicit physical/numerical audit facts.

        Raises:
            ValueError: Incomplete, non-finite, wrong-unit, wrong-frame or failed-audit reference.
        """
        values, header = StructuralAsset.read(path, base, PhysicalFields.REFERENCE)
        config = PhysicsConfig(**header["parameters"])
        if header.get("algorithm") != PhysicalFields.ALGORITHM or (
            parameters is not None and header["parameters"] != parameters
        ) or header.get("units") != ["kBT/e", "kJ/mol", "kJ/mol"]:
            raise ValueError("VF physical reference identity or unit convention mismatch")
        if config.heavy_atom_policy == HeavyAtomPolicy.BOUNDED and (
            sequence_context is not None) and header.get("sequence_context_sha256") != (
            ProteinArchive.sha256(sequence_context)):
            raise ValueError("VF physical reference sequence evidence changed")
        if header.get("availability") == "unavailable":
            if set(values) != {"valid"} or values["valid"].shape != (0, 3) or (
                values["valid"].dtype != np.bool_ or not header.get("reason") or
                header.get("reason_type") != "unsupported_observed_chemistry"
            ):
                raise ValueError("VF unavailable chemistry has fabricated or unqualified targets")
            return True
        shape = values["potential"].shape
        if len(shape) != 3 or values["targets"].shape != (*shape, 3) or (
            values["valid"].shape != (*shape, 3) or values["valid"].dtype != np.bool_
        ):
            raise ValueError("VF physical volume/mask domains disagree")
        if np.any(values["targets"][~values["valid"]]) or np.any(values["delta"] <= 0):
            raise ValueError("VF masked values or grid spacing are invalid")
        potential_valid = values["valid"][..., 0]
        if not np.array_equal(values["targets"][..., 0][potential_valid],
                              values["potential"][potential_valid]):
            raise ValueError("VF potential targets differ from the retained raw APBS volume")
        domain = values["distance"] >= 0
        if np.any(potential_valid & ~domain):
            raise ValueError("VF potential supervision includes an excluded heavy-atom interior")
        for channel in range(2):
            expected_mask = (values["lj_overlap_valid"][..., channel] & domain &
                (np.abs(values["lj_energy"][..., channel]) <= config.repulsion_ceiling))
            selected = values["valid"][..., channel + 1]
            if not np.array_equal(expected_mask, selected) or not np.array_equal(
                values["targets"][..., channel + 1][selected],
                values["lj_energy"][..., channel][selected].astype(np.float32)
            ):
                raise ValueError("VF LJ targets/masks differ from the retained raw probe field")
        audit = header.get("potential_convergence", {})
        unavailable_potential = (config.convergence and audit.get("passed") is False and
            config.potential_failure_policy == "mask" and not potential_valid.any() and
            (audit.get("candidate_coverage", 1.0) < config.minimum_potential_coverage or
             any(value < config.minimum_region_coverage
                 for value in audit.get("candidate_region_coverage", {}).values())))
        if config.convergence and audit.get("passed") is not True and not unavailable_potential:
            raise ValueError("VF lacks its required potential convergence audit")
        if config.convergence:
            audit = header["potential_convergence"]
            coverage = float(potential_valid.sum() / max(1, domain.sum()))
            if audit["valid_queries"] != int(potential_valid.sum()) or not np.isclose(
                coverage, audit["coverage"], rtol=0, atol=1e-12
            ) or (coverage < config.minimum_potential_coverage and not unavailable_potential):
                raise ValueError("VF potential coverage disagrees with its numerical audit")
            for region, recorded in audit["region_coverage"].items():
                actual = float(potential_valid[values["region"] == int(region)].mean())
                if not np.isclose(actual, recorded, rtol=0, atol=1e-12) or (
                    actual < config.minimum_region_coverage and not unavailable_potential
                ):
                    raise ValueError("VF regional potential confidence is inconsistent")
        comparisons = header.get("potential_convergence", {}).get("comparisons", [])
        if config.convergence and {row["name"] for row in comparisons} != (
            {"refined", "larger-box"} if config.margin_convergence else {"refined"}
        ):
            raise ValueError("VF lacks its declared grid/box sensitivity comparisons")
        if len(header.get("lj_audits", [])) != 2:
            raise ValueError("VF lacks both independent OpenMM probe audits")
        with np.load(base, allow_pickle=False) as stored:
            atoms = stored["atom_positions"]
            observed = {"atom_positions": atoms, **{key: stored[key] for key in (
                "atom_names", "residue_indices", "chain_indices")}}

        # Reusing fields must not bypass known source contradictions. Qualified unavailable
        # references returned above contain no physical targets and remain valid checkpoints.

        PreparationAdmission(config).validate_observed_termini(observed)
        mapping = values["heavy_mapping"]
        if mapping.shape != (len(atoms),) or len(np.unique(mapping)) != len(atoms) or (
            np.any(mapping < 0) or np.any(mapping >= len(values["prepared_positions"]))
        ) or not np.allclose(values["prepared_positions"][mapping], atoms, rtol=0, atol=.002):
            raise ValueError("VF reference lost exact retained-heavy-atom correspondence")
        if config.heavy_atom_policy == HeavyAtomPolicy.BOUNDED:
            with np.load(base, allow_pickle=False) as stored:
                arrays = {name: stored[name] for name in ("atom_positions", "atomic_numbers",
                          "atom_names", "residue_indices", "chain_indices")}
            keys = [(str(c), int(r), str(n)) for c, r, n in header["prepared_identity_keys"]]
            rebuilt = header["preparation_audit"].get("rebuilt_indices", [])
            if rebuilt and not len(values["prepared_bonds"]):
                raise ValueError("VF rebuilt physical atoms lack their force-field topology")
            sequence = StructuralAsset.read(sequence_context, base, "sequence_context")[0] if (
                sequence_context is not None) else None
            if sequence is None and header["preparation_audit"].get("verified_modeled_c_termini"):
                raise ValueError("VF modeled terminal validation requires its "
                                 "sequence_context asset")
            admission = PreparationAdmission(config)
            expected = admission.assess(arrays, keys, values["prepared_positions"],
                                         values["prepared_atomic_numbers"], sequence)
            expected["geometry"] = admission.geometry(values["prepared_positions"],
                values["prepared_atomic_numbers"], values["prepared_sigma"],
                values["prepared_bonds"].tolist(), expected["rebuilt_indices"])
            if not np.array_equal(values["heavy_mapping"], expected["heavy_mapping"]):
                raise ValueError("VF heavy-atom mapping differs from audited chemical identities")
            if json.dumps(expected, sort_keys=True) != json.dumps(header["preparation_audit"],
                                                                 sort_keys=True):
                raise ValueError("VF bounded preparation audit disagrees with its physical sources")
        return True

    @staticmethod
    def validate_queries(path: Path, base: Path,
                         parameters: Mapping[str, Any] | None = None) -> bool:
        """Check the model-query namespace without importing physics executables in training.

        Args:
            path: Immutable bank NPZ.
            base: Exact parent NPZ defining retained atom IDs and centered coordinates.
            parameters: Optional requested complete PhysicsConfig mapping.

        Returns:
            True after storage, coordinate, units, region, mask and sparse-index checks.

        Raises:
            ValueError: Unqualified empty/invalid banks, wrong settings or cross-domain indices.
                Explicitly unavailable chemistry must have a reason and an entirely empty bank.
        """
        values, header = StructuralAsset.read(path, base, PhysicalFields.QUERIES)
        config = PhysicsConfig(**header["parameters"])
        if header.get("algorithm") != PhysicalFields.ALGORITHM or (
            parameters is not None and header["parameters"] != parameters
        ) or header.get("physical_sources_are_targets_only") is not True or header.get(
            "units") != ["kBT/e", "kJ/mol", "kJ/mol"]:
            raise ValueError("VF query-bank scientific contract mismatch")
        count = len(values["positions"])
        if header.get("availability") == "unavailable":
            empty_shapes = {"positions": (0, 3), "targets": (0, 3), "valid": (0, 3),
                "region": (0,), "distance": (0,), "accessible": (0,), "grid_index": (0,),
                "offsets": (1,), "atom_ids": (0,)}
            if set(values) != set(empty_shapes) or any(
                values[key].shape != shape for key, shape in empty_shapes.items()
            ) or values["valid"].dtype != np.bool_ or values["offsets"][0] != 0 or (
                not header.get("reason") or header.get("queries") != 0 or
                header.get("reason_type") != "unsupported_observed_chemistry"
            ):
                raise ValueError("VF unavailable bank must be empty and explicitly qualified")
            return True
        if not count or values["positions"].shape != (count, 3) or (
            values["targets"].shape != (count, 3) or values["valid"].shape != (count, 3)
        ) or values["valid"].dtype != np.bool_ or np.any(values["targets"][~values["valid"]]):
            raise ValueError("VF query domains or field masks are invalid")
        if not values["valid"].any() or np.any(values["region"] < 0) or (
            np.any(values["region"] > 3) or len(np.unique(values["grid_index"])) != count
        ):
            raise ValueError("VF query regions or canonical grid addresses are invalid")
        offsets, ids = values["offsets"], values["atom_ids"]
        with np.load(base, allow_pickle=False) as stored:
            atoms = stored["atom_positions"]
        if offsets.shape != (count + 1,) or offsets[0] != 0 or offsets[-1] != len(ids) or (
            np.any(np.diff(offsets) < 0) or np.any(ids < 0) or np.any(ids >= len(atoms))
        ):
            raise ValueError("VF sparse query/atom ownership is invalid")
        shape = tuple(header["grid_shape"])
        addresses = values["grid_index"]
        if np.any(addresses < 0) or np.any(addresses >= np.prod(shape)):
            raise ValueError("VF query addresses lie outside the physical solver grid")
        expected = (np.asarray(header["grid_origin"]) + np.stack(
            np.unravel_index(addresses, shape), axis=1) * np.asarray(header["grid_delta"]))
        if not np.array_equal(expected.astype(np.float32), values["positions"]):
            raise ValueError("VF query coordinates disagree with their stored grid addresses")
        repair_positions = np.asarray(header.get("repair_exclusion_positions", []), dtype=float)
        if len(repair_positions) and np.any(cKDTree(repair_positions).query(
            values["positions"])[0] < config.repair_exclusion_radius):
            raise ValueError("VF query bank includes the excluded modeled-atom near field")
        neighbors = cKDTree(atoms).query_ball_point(values["positions"], config.decoder_radius,
                                                   return_sorted=True)
        for row, candidates in enumerate(neighbors):
            if not np.array_equal(ids[offsets[row]:offsets[row + 1]], candidates):
                raise ValueError("VF query candidates are truncated or disagree with their radius")
        return True
