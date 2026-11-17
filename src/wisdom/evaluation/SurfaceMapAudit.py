"""Frozen prediction audits across independently sampled, physically compatible surfaces."""

import json
import torch
import hashlib
import numpy as np
import lambdaforge as lf

from typing import Any
from pathlib import Path
from torch import Tensor, nn
from functools import partial
from scipy.spatial import cKDTree
from scipy.stats import spearmanr
from collections.abc import Sequence
from scipy.spatial.transform import Rotation
from wisdom.data.WisdomCollator import WisdomCollator
from wisdom.Training import _model_inputs, _device_batch
from wisdom.preprocessing.common.structure.storage.ProteinArchive import ProteinArchive
from wisdom.preprocessing.common.structure.assets.surface.SurfaceView import SurfaceView
from wisdom.preprocessing.common.structure.assets.StructuralAsset import StructuralAsset


class SurfaceMapAudit:
    """Separate tensor invariance from discretization sensitivity without using local targets."""

    def __init__(
        self,
        maximum_distance: float = 1.5,
        minimum_normal_cosine: float = 0.5,
        query_neighbors: int = 32,
        interpolation_neighbors: int = 3,
        protocols: Sequence[str] = ("published",),
        view_seeds: Sequence[int] = (2026,),
    ) -> None:
        """Bind physical correspondence, not a prediction-dependent matching rule.

        Args:
            maximum_distance: Maximum compatible point separation in Å.
            minimum_normal_cosine: Minimum outward normal dot product in [-1,1].
            query_neighbors: Bounded KD-tree candidates searched before filtering normals.
            interpolation_neighbors: Compatible neighbors retained, default three.
            protocols: published (legacy), identity, rigid_tensor, rigid_rebuild, phase, grid,
                resolution_0.8 or resolution_1.25. Regenerated views use native managed files.
            view_seeds: Predeclared physical-view seeds, not new training replications.

        Raises:
            ValueError: Unsupported distance, normal threshold or neighbor budgets.
        """
        if maximum_distance <= 0 or not -1 <= minimum_normal_cosine <= 1:
            raise ValueError("invalid physical surface correspondence thresholds")
        if not 1 <= interpolation_neighbors <= query_neighbors:
            raise ValueError("interpolation budget must be positive and <= query budget")
        self.radius = maximum_distance
        self.cosine = minimum_normal_cosine
        self.query = query_neighbors
        self.neighbors = interpolation_neighbors
        self.protocols, self.view_seeds = tuple(protocols), tuple(view_seeds)
        if set(protocols) - {"published", "identity", "rigid_tensor", "rigid_rebuild", "phase",
                             "grid", "resolution_0.8", "resolution_1.25"}:
            raise ValueError("unsupported frozen geometry audit protocol")

    def correspondence(
        self, positions: Tensor, normals: Tensor, source_positions: Tensor, source_normals: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Construct sparse inverse-square interpolation after physical filtering.

        Args:
            positions: Destination points [M,3] in Å, same physical frame as source.
            normals: Destination unit outward normals [M,3].
            source_positions: Source points [N,3] in Å.
            source_normals: Source unit outward normals [N,3].

        Returns:
            CPU neighbor IDs [M,K], weights [M,K], and supported rows [M]. Search first,
            filter distance/normals next, then retain K. Exact coincidences receive all mass;
            otherwise w_j = d_j^-2 / sum_k d_k^-2. Unsupported rows have zero weights.
            Equal distances are resolved by source index; storage is O(M query_neighbors).
        """
        a, b = positions.double().cpu(), source_positions.double().cpu()
        na = torch.nn.functional.normalize(normals.double().cpu(), dim=1)
        nb = torch.nn.functional.normalize(source_normals.double().cpu(), dim=1)
        distance, ids = cKDTree(b.numpy()).query(
            a.numpy(), k=min(self.query, len(b)), distance_upper_bound=self.radius
        )
        distance = np.asarray(distance).reshape(len(a), -1)
        ids = np.asarray(ids).reshape(len(a), -1)
        valid = np.isfinite(distance) & (ids < len(b))
        safe = np.minimum(ids, len(b) - 1)
        valid &= np.einsum("mi,mki->mk", na.numpy(), nb.numpy()[safe]) >= self.cosine
        distance = np.where(valid, distance, np.inf)
        order = np.lexsort((ids, distance), axis=1)[:, : self.neighbors]
        distance = np.take_along_axis(distance, order, axis=1)
        ids = np.take_along_axis(safe, order, axis=1)
        valid = np.isfinite(distance)
        exact = (distance == 0) & valid
        weights = np.where(valid, np.maximum(distance, 1e-12) ** -2, 0.0)
        weights = np.where(exact.any(1, keepdims=True), exact.astype(float), weights)
        weights /= np.maximum(weights.sum(1, keepdims=True), 1e-300)
        return torch.from_numpy(ids), torch.from_numpy(weights), torch.from_numpy(valid.any(1))

    @staticmethod
    def top_area(values: Tensor, area: Tensor, fraction: float) -> Tensor:
        """Represent the highest-scoring fraction of physical area with a fractional boundary.

        Args:
            values: Scores [M] on the supported comparison domain.
            area: Positive represented area [M] in Å².
            fraction: Requested area fraction in (0,1].

        Returns:
            Selection mass [M] in Å². Ties use original index; the last point may be partial.
        """
        order = torch.argsort(values, descending=True, stable=True)
        remaining = fraction * area.sum() - (area[order].cumsum(0) - area[order])
        chosen = remaining.clamp_min(0).minimum(area[order])
        result = torch.zeros_like(area)
        result[order] = chosen
        return result

    def compare(
        self,
        reference: dict[str, Tensor],
        view: dict[str, Tensor],
        reference_logits: Tensor,
        view_logits: Tensor,
    ) -> dict[str, float | None]:
        """Compare logits/probabilities in both directions and expose coverage separately.

        Args:
            reference: surface_positions/normals [M,3], surface_area_weights [M].
            view: Same fields on the independently sampled [N] domain.
            reference_logits: Frozen logits [M], never thresholded or subsampled.
            view_logits: Frozen logits [N] from operators matching the view point order.

        Returns:
            Direction-qualified coverage, area-weighted logit/probability MAE and RMSE,
            unweighted Spearman rank correlation, and area top-5/10% IoU. Zero support and
            constant-map correlations remain unavailable rather than becoming artificial zeros.
            This measures stability, not accuracy; no GT is accepted.
        """
        report: dict[str, float | None] = {}
        for prefix, destination, source, first, second in (
            ("reference", reference, view, reference_logits, view_logits),
            ("view", view, reference, view_logits, reference_logits),
        ):
            ids, weights, valid = self.correspondence(
                destination["surface_positions"],
                destination["surface_normals"],
                source["surface_positions"],
                source["surface_normals"],
            )
            area = destination["surface_area_weights"].detach().double().cpu()
            report[prefix + "_area_coverage"] = float(area[valid].sum() / area.sum())
            report[prefix + "_supported_points"] = float(valid.sum())
            aligned = (second.detach().double().cpu()[ids] * weights).sum(1)[valid]
            original = first.detach().double().cpu()[valid]
            supported_area = area[valid]
            aligned_probability = (second.detach().double().cpu().sigmoid()[ids] * weights).sum(1)[
                valid
            ]
            for space, left, right in (
                ("logit", original, aligned),
                ("probability", original.sigmoid(), aligned_probability),
            ):
                delta = left - right
                report[f"{prefix}_{space}_mae"] = (
                    float((supported_area * delta.abs()).sum() / supported_area.sum())
                    if valid.any()
                    else None
                )
                report[f"{prefix}_{space}_rmse"] = (
                    float(((supported_area * delta.square()).sum() / supported_area.sum()).sqrt())
                    if valid.any()
                    else None
                )
            nonconstant = len(original) > 1 and original.std() > 0 and aligned.std() > 0
            report[prefix + "_spearman"] = (
                float(spearmanr(original, aligned).statistic) if nonconstant else None
            )
            for fraction in (0.05, 0.1):
                left = self.top_area(original, supported_area, fraction)
                right = self.top_area(aligned, supported_area, fraction)
                union = torch.maximum(left, right).sum()
                report[f"{prefix}_top_{int(100 * fraction)}_area_iou"] = (
                    float(torch.minimum(left, right).sum() / union) if union > 0 else None
                )
        return report

    def evaluate(
        self,
        model: nn.Module,
        sample: dict[str, Any],
        collator: WisdomCollator,
        device: torch.device,
        work: lf.Work | None = None,
        base: Path | None = None,
    ) -> dict[str, Any]:
        """Run frozen FP32 controls on an existing published alternate view and rigid tensors.

        Args:
            model: Restored best checkpoint in evaluation mode with its recorded gate policy.
            sample: One validated dataset member, including its surface_view sidecar and geometry.
            collator: Exact original base-source collator; incompatible sidecars fail explicitly.
            device: Available CPU/CUDA device. No fitting, backward, or model selection occurs.
            work: Native managed-file service required for regenerated protocol views.
            base: Immutable universal archive bound to the evaluated member.

        Returns:
            Rigid-tensor maximum logit error, bidirectional view metrics, and constant/smooth/
            localized analytic-field interpolation controls. The published view combines phase
            and grid changes: this is not an isolated phase, resolution, or rotated-regeneration
            experiment. No original point target is copied onto the alternate domain.

        Raises:
            ValueError: Extended transfer/spectrum/local-graph inputs would refer to old points.
        """
        if self.protocols != ("published",):
            if work is None or base is None:
                raise ValueError("regenerated V12 protocols need a Work and exact parent archive")
            return self.protocol_audit(model, sample, collator, device, work, base)
        if (
            collator.transfer_source != "base"
            or collator.spectral_source != "base"
            or collator.surface_graph_source != "base"
        ):
            raise ValueError(
                "view audit requires base point-bound inputs; regenerate new sidecars first"
            )
        tensors = {key: value for key, value in sample.items() if isinstance(value, Tensor)}
        geometry = {
            key: tensors[key]
            for key in ("surface_positions", "surface_normals", "surface_area_weights")
        }
        alternate = {
            key.split("__", 1)[1]: value
            for key, value in tensors.items()
            if key.startswith("surface_view__")
        }
        point_targets = {
            "surface_target_hard",
            "surface_target_soft",
            "surface_valid_mask",
            "surface_distance_to_target",
            "surface_target_hard_sensitivity",
        }
        clean = {key: value for key, value in sample.items() if key not in point_targets}
        view = {**clean, **alternate}
        arrays = {key: value.numpy() for key, value in tensors.items() if "__" not in key}
        rotation = np.asarray([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        transformed = SurfaceView.rigid_transform(arrays, rotation, np.asarray([2.0, -3.0, 1.0]))
        rigid = {**clean, **{key: torch.from_numpy(value) for key, value in transformed.items()}}
        predictions = []
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            for member in (clean, rigid, view):
                inputs = _model_inputs(_device_batch(collator([member]), device))
                predictions.append(model(**inputs)["surface_logits"].float().cpu())
        result: dict[str, Any] = self.compare(geometry, alternate, predictions[0], predictions[2])
        result["rigid_tensor_max_logit_error"] = float(
            (predictions[0] - predictions[1]).abs().max()
        )
        result["protocol"] = "published-combined-phase-grid-view; fixed-model; no-GT"
        center = geometry["surface_positions"].mean(0)
        for name in ("constant", "smooth", "patch"):
            fields = []
            for domain in (geometry, alternate):
                relative = domain["surface_positions"] - center
                fields.append(
                    torch.ones(len(relative))
                    if name == "constant"
                    else (
                        relative[:, 0] / 10
                        if name == "smooth"
                        else torch.exp(-relative.square().sum(1) / 8)
                    )
                )
            result[name + "_calibration"] = self.compare(geometry, alternate, *fields)
        return result

    def protocol_audit(self, model: nn.Module, sample: dict[str, Any],
                       collator: WisdomCollator, device: torch.device,
                       work: lf.Work, base: Path) -> dict[str, Any]:
        """Evaluate isolated views with frozen weights and dependency-bound geometry checkpoints.

        Args:
            model: Existing evaluation-mode predictor; weights and gates are never updated.
            sample: Exact original physical member and its unchanged atomic/context sources.
            collator: Saved runtime budgets and physical source choices.
            device: Granted logical PyTorch device; CPU is also valid.
            work: Native cache, logging and checkpoint ownership.
            base: Exact immutable universal parent whose bytes bind every regenerated view.

        Returns:
            Protocol/seed-qualified stability, coverage, calibration and global-probability
            differences. Views are not averaged into a supposedly improved prediction.

        Raises:
            ValueError: Point-dependent model inputs cannot be regenerated coherently.
        """
        if collator.transfer_source != "base" or collator.spectral_source != "base" or (
            collator.surface_graph_source != "base"
        ) or sample.get("surface_explicit_features") is not None:
            raise ValueError("V12 views require fresh point-bound extended/explicit input assets")
        if getattr(model, "surface_environment", None) is not None:
            raise ValueError("V12 environment models require view-bound environment regeneration")
        representation = getattr(model, "atomic_representation", None)
        residue_encoder = getattr(representation, "residue_encoder", None)
        if residue_encoder is not None and residue_encoder.topology == "parallel":
            raise ValueError("V12 parallel residue models require view-bound residue candidates")
        with np.load(base, allow_pickle=False) as stored:
            import_header = str(stored["metadata_json"].item())
        # JSON parsing belongs to the base format, not to a task-specific label catalog.

        metadata = json.loads(import_header)
        spacing = float(metadata["config"]["surface_resolution"])
        parent_digest = ProteinArchive.sha256(base)
        clean = {key: value for key, value in sample.items() if not key.startswith(
            ("surface_target", "surface_distance", "surface_view__")) and key not in {
                "surface_valid_mask", "sensitivity_gaps"}}
        reference = {key: clean[key] for key in (
            "surface_positions", "surface_normals", "surface_area_weights")}
        report: dict[str, Any] = {}
        with torch.inference_mode(), torch.autocast(device_type=device.type, enabled=False):
            original = model(**_model_inputs(_device_batch(collator([clean]), device)))
            for protocol in self.protocols:
                if protocol == "published":
                    raise ValueError("choose published audit separately from isolated protocols")
                for seed in self.view_seeds:
                    if protocol == "rigid_tensor":
                        arrays = {key: value.numpy() for key, value in clean.items()
                                  if isinstance(value, Tensor) and "__" not in key}
                        rotation = Rotation.random(random_state=seed).as_matrix()
                        moved = SurfaceView.rigid_transform(arrays, rotation,
                                                            np.array([2., -3., 1.]))
                        view = {**clean, **{key: torch.from_numpy(value)
                                           for key, value in moved.items()}}
                    else:
                        ratio = float(protocol.split("_")[1]) if protocol.startswith(
                            "resolution_") else 1.
                        kind = ("resolution" if protocol.startswith("resolution_") else
                                "rigid" if protocol == "rigid_rebuild" else protocol)
                        producer = SurfaceView(1, spacing*ratio, kind, seed)
                        settings = {"base": parent_digest, "view": producer.parameters}
                        serialized = json.dumps(settings, sort_keys=True).encode()
                        key = hashlib.sha256(serialized).hexdigest()
                        path = work.cache.file(
                            f"geometry-audit/{key}.npz",
                            build=partial(producer.write, base),
                            validate=partial(producer.validate, base=base))
                        arrays, _ = StructuralAsset.read(Path(path), base, "surface_view")
                        view = {**clean, **{key: torch.from_numpy(value)
                                           for key, value in arrays.items()}}
                    changed = model(**_model_inputs(_device_batch(collator([view]), device)))
                    prefix = f"{protocol}_{seed}_"
                    before = original["surface_logits"].float().cpu()
                    after = changed["surface_logits"].float().cpu()
                    if protocol in {"identity", "rigid_tensor"}:
                        report[prefix + "index_max_logit_error"] = float((before-after).abs().max())
                    aligned_view = reference if protocol == "rigid_tensor" else view
                    report.update({prefix + key: value for key, value in self.compare(
                        reference, aligned_view, before, after).items()})
                    raw_first = original.get("raw_surface_logits", original["surface_logits"])
                    raw_second = changed.get("raw_surface_logits", changed["surface_logits"])
                    report.update({prefix + "raw_" + key: value for key, value in self.compare(
                        reference, aligned_view, raw_first.float().cpu(),
                        raw_second.float().cpu()).items()})
                    report[prefix + "protein_probability_error"] = float((
                        original["logits"].sigmoid()-changed["logits"].sigmoid()).abs().max())
                    # Calibrate correspondence with fields defined by a physical atom anchor,
                    # rather than a point index or a prediction-selected region.

                    anchor = clean["atom_positions"][0].cpu()
                    for field in ("constant", "smooth", "patch"):
                        fields = []
                        for domain in (reference, aligned_view):
                            radius = (domain["surface_positions"].cpu()-anchor).norm(dim=1)
                            fields.append(torch.ones_like(radius) if field == "constant" else
                                radius/10 if field == "smooth" else torch.exp(-radius.square()/8))
                        report.update({prefix + field + "_" + key: value for key, value in
                                       self.compare(reference, aligned_view, *fields).items()})
        report["protocol"] = "isolated-numerical-views; frozen-checkpoint; no-local-GT"
        return report
