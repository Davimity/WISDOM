"""Evaluation-only zinc and coordinating-atom surface references."""

import json
import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from scipy.spatial import cKDTree
from collections.abc import Mapping, Sequence


class ZincAnnotation:
    """Project verified coordination sites onto immutable protein-only surface point order."""

    def __init__(
        self, positive_gap: float = 1.4, negative_gap: float = 3.0,
        sensitivity_gaps: Sequence[float] = (1.0, 1.4, 2.0),
    ) -> None:
        """Set evaluation thresholds independently of model inputs and losses.

        Args:
            positive_gap: Maximum gap to a selected-copy coordinating atom, in ångströms.
            negative_gap: Minimum confidently negative gap in ångströms, greater than positive_gap.
            sensitivity_gaps: Alternative positive gap cutoffs in ångströms.

        Raises:
            ValueError: Unordered/non-finite thresholds or empty sensitivity cutoffs.
        """
        if not 0 <= positive_gap < negative_gap or not sensitivity_gaps:
            raise ValueError("Zn annotation needs ordered positive/negative gaps and sensitivities")
        if not np.isfinite([positive_gap, negative_gap, *sensitivity_gaps]).all():
            raise ValueError("Zn annotation thresholds must be finite")
        self.positive_gap = positive_gap
        self.negative_gap = negative_gap
        self.sensitivity_gaps = tuple(sensitivity_gaps)

    def write(self, base: Path, output: Path, row: Mapping[str, Any]) -> None:
        """Compute the union of verified coordinating-atom neighborhoods, not a Zn input.

        Centered surface points are restored to deposited coordinates, then transformed into
        the selected assembly copy. For point p, donor gap=min_a(||p-a||-r_vdw(a)) across
        accepted sites' selected-copy N/O/S atoms. All candidate distances within the nearest
        center distance plus maximum radius are checked, so differing radii cannot invalidate
        the minimum. Zn-center minus 1.39 Å is retained separately as a diagnostic, not the
        reference label. Hard=gap<=positive_gap; valid=hard or gap>=negative_gap. Soft uses
        a cosine transition. Empty positive references are unavailable, never all-negative GT.

        Args:
            base: Immutable universal protein-only NPZ with point coordinates and source origin.
            output: Native checkpoint build destination for pickle-free annotation NPZ.
            row: Verified selected record, sites, label, assembly transform and split provenance.

        Raises:
            ValueError: Missing accepted site, invalid transform, or observed Zn in base inputs.
        """
        with np.load(base, allow_pickle=False) as archive:
            metadata = json.loads(str(archive["metadata_json"].item()))
            surface = archive["surface_positions"].astype(np.float64)
            if np.any(archive["atomic_numbers"] == 30):
                raise ValueError("Zn reference positions must not enter universal protein inputs")
        origin = np.asarray(metadata["coordinate_origin"], dtype=np.float64)
        rotation = np.asarray(row["assembly_rotation"], dtype=np.float64)
        translation = np.asarray(row["assembly_translation"], dtype=np.float64)
        if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-5):
            raise ValueError("Zn annotation assembly transform must be a rigid rotation")
        points = (surface + origin) @ rotation.T + translation
        count = len(points)
        label = int(row["label"])
        sites = [site for site in row["sites"] if site["accepted"]]

        # Coordinating atoms belong to the selected copy, not an adjacent binding partner.
        # Their positions are GT-only evidence; no projection/classifier reads this sidecar.

        if label == 1:
            if not sites:
                raise ValueError("positive Zn annotation requires verified assembly sites")
            donors = {tuple(atom["position"]): atom for site in sites for atom in site["partners"]
                      if atom["residue"] in site["selected_residues"]
                      and atom["chain"] == str(row["protein_chain"])
                      and atom["chain_index"] == site["selected_chain_index"]}
            coords = np.asarray(list(donors), dtype=np.float64).reshape(-1, 3)
            radii = np.asarray([atom["vdw_radius"] for atom in donors.values()], dtype=np.float64)
            tree = cKDTree(coords)
            nearest = tree.query(points)[0]
            candidates = tree.query_ball_point(points, nearest + radii.max())
            # Reference decisions use persisted float32 gaps, avoiding a boundary disagreement
            # between a float64 construction and a float32 validation after serialization.

            gap = np.asarray([np.min(np.linalg.norm(coords[ids] - point, axis=1) - radii[ids])
                              for point, ids in zip(points, candidates, strict=True)],
                             dtype=np.float32)
            zinc_tree = cKDTree(np.asarray([site["position"] for site in sites]))
            zinc_gap = zinc_tree.query(points)[0] - 1.39
            hard = gap <= self.positive_gap
            available = bool(hard.any())
            valid = ((gap <= self.positive_gap) | (gap >= self.negative_gap)) & available
            fraction = np.clip((gap - self.positive_gap) /
                               (self.negative_gap - self.positive_gap), 0, 1)
            soft = (1 + np.cos(np.pi * fraction)) / 2
            sensitivity = gap[:, None] <= np.asarray(self.sensitivity_gaps, dtype=np.float32)
            distance_valid = np.ones(count, dtype=np.bool_)
        else:
            if any(site["selected_donor_count"] for site in row["sites"]):
                raise ValueError("explicit negative conflicts with observed Zn contact evidence")
            gap = np.zeros(count, dtype=np.float32)
            zinc_gap = np.zeros(count)
            hard = np.zeros(count, dtype=np.bool_)
            soft = np.zeros(count)
            valid = np.ones(count, dtype=np.bool_)
            sensitivity = np.zeros((count, len(self.sensitivity_gaps)), dtype=np.bool_)
            distance_valid = np.zeros(count, dtype=np.bool_)
            available = True
        if label and not available and row["split"] in ("validation", "test"):
            raise ValueError(f"{row['identifier']}: positive evaluation member has no usable GT; "
                             "revise Selection explicitly; never create all-negative positive GT")
        audit = {"schema_version": "1.0", "task_name": "zinc_binding",
                 "identifier": row["identifier"], "label": label,
                 "source_structure_sha256": row["structure_sha256"],
                 "assembly_id": row["assembly_id"], "protein_chain": row["protein_chain"],
                 "protein_copy": row["protein_copy"], "positive_gap": self.positive_gap,
                 "negative_gap": self.negative_gap,
                 "distance_definition": "minimum coordinating-atom center distance minus donor vdw",
                 "zinc_diagnostic_radius_angstrom": 1.39, "sites": sites,
                 "label_evidence": row["label_evidence"]}
        with output.open("wb") as stream:
            np.savez_compressed(
                stream, surface_target_hard=hard, surface_target_soft=soft.astype(np.float32),
                surface_valid_mask=valid, surface_distance_to_target=gap.astype(np.float32),
                coordinating_atom_gap=gap.astype(np.float32), zinc_gap=zinc_gap.astype(np.float32),
                surface_distance_valid=distance_valid, surface_target_hard_sensitivity=sensitivity,
                sensitivity_gaps=np.asarray(self.sensitivity_gaps, dtype=np.float32),
                local_gt_available=np.asarray(available),
                base_npz_sha256=np.asarray(hashlib.sha256(base.read_bytes()).hexdigest()),
                annotation_metadata_json=np.asarray(json.dumps(audit, sort_keys=True)),
            )

    @staticmethod
    def validate(path: Path, base: Path) -> bool:
        """Audit complete sidecar alignment and basic numerical target invariants.

        Args:
            path: Candidate annotation NPZ.
            base: Corresponding immutable protein-only NPZ.

        Returns:
            True after schema, point-order fingerprint and numerical checks.

        Raises:
            ValueError: Invalid hash, dtype, count, probability, mask, or sensitivity shape.
        """
        with np.load(base, allow_pickle=False) as archive:
            count = len(archive["surface_positions"])
        with np.load(path, allow_pickle=False) as archive:
            digest = hashlib.sha256(base.read_bytes()).hexdigest()
            if str(archive["base_npz_sha256"].item()) != digest:
                raise ValueError("Zn annotation base hash mismatch")
            for name in ("surface_target_hard", "surface_target_soft", "surface_valid_mask",
                         "surface_distance_to_target", "coordinating_atom_gap",
                         "surface_distance_valid", "zinc_gap"):
                values = archive[name]
                if (values.shape != (count,) or values.dtype == object
                    or not np.isfinite(values).all()):
                    raise ValueError(f"Zn annotation {name}: invalid point count/dtype/numbers")
            hard, soft = archive["surface_target_hard"], archive["surface_target_soft"]
            valid = archive["surface_valid_mask"]
            sensitivity = archive["surface_target_hard_sensitivity"]
            if hard.dtype != np.bool_ or valid.dtype != np.bool_:
                raise ValueError("Zn hard targets and masks must be Boolean")
            if np.any((soft < 0) | (soft > 1)) or sensitivity.shape != (
                count, len(archive["sensitivity_gaps"])
            ):
                raise ValueError("Zn soft targets/sensitivity dimensions are invalid")
            if not bool(archive["local_gt_available"].item()) and valid.any():
                raise ValueError("unavailable Zn GT must not provide supervised points")
            audit = json.loads(str(archive["annotation_metadata_json"].item()))
            if audit["task_name"] != "zinc_binding" or audit["label"] not in (0, 1):
                raise ValueError("Zn annotation task/label metadata is invalid")
            gap = archive["surface_distance_to_target"]
            if not np.array_equal(gap, archive["coordinating_atom_gap"]):
                raise ValueError("Zn reference and coordinating-atom gaps disagree")
            if sensitivity.dtype != np.bool_ or archive["surface_distance_valid"].dtype != np.bool_:
                raise ValueError("Zn sensitivity/distance masks must be Boolean")
            if audit["label"]:
                expected_hard = gap <= audit["positive_gap"]
                expected_valid = (expected_hard | (gap >= audit["negative_gap"])) & bool(
                    archive["local_gt_available"].item())
                fraction = np.clip((gap - audit["positive_gap"]) /
                                   (audit["negative_gap"] - audit["positive_gap"]), 0, 1)
                expected_soft = (1 + np.cos(np.pi * fraction)) / 2
                if (not np.array_equal(hard, expected_hard)
                    or not np.array_equal(valid, expected_valid)
                    or not np.allclose(soft, expected_soft, atol=1e-6, rtol=1e-6)
                    or not np.array_equal(sensitivity, gap[:, None] <= archive["sensitivity_gaps"])
                    or not archive["surface_distance_valid"].all()):
                    raise ValueError("Zn targets disagree with documented signed-gap thresholds")
            elif (hard.any() or soft.any() or sensitivity.any() or not valid.all()
                  or archive["surface_distance_valid"].any()):
                raise ValueError("explicit negative Zn annotation has inconsistent targets/masks")
        return True
