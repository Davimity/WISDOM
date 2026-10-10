"""Publish optional label-free fields without replacing universal geometry or annotations."""

import json
import hashlib
import lambdaforge as lf

from typing import Any
from pathlib import Path
from functools import partial
from collections.abc import Sequence
from lambdaforge.data import DatasetIndex
from wisdom.features.SurfaceFeatureSchema import SurfaceFeatureSchema
from wisdom.features.SurfaceFeatureSidecar import SurfaceFeatureSidecar


class SurfaceFeatures(lf.Work):
    """Augment immutable geometry with fixed fields and frozen training statistics."""

    def run(
        self,
        dataset: Path,
        dataset_name: str,
        dataset_version: str,
        feature_names: Sequence[str] = (),
        feature_group: str | None = "generic_basic",
        exclude: Sequence[str] = (),
        sigma: float = 2.0,
        normalization_weighting: str = "pooled_points",
        workers: int = 4,
    ) -> dict[str, Any]:
        """Project protein-only chemistry, fit train normalization, and publish new members.

        Args:
            dataset: Typed managed input root with index.jsonl and immutable universal assets.
            dataset_name: New Registry family, or the existing family with a new version.
            dataset_version: New immutable release; an existing version is never overwritten.
            feature_names: Additional ordered channel names from SurfaceFeatureSchema.
            feature_group: Named default group, or None for explicit feature_names only.
            exclude: Known channels to omit after group expansion.
            sigma: Positive Gaussian distance width in ångströms, default 2.
            normalization_weighting: pooled_points (default), pooled_area or equal_protein.
                Each published train dilution receives its own frozen fit.
            workers: Bounded concurrent independent projections, default 4.

        Returns:
            Native publication record and exact selected field order. Existing targets, splits,
            geometry, and annotations remain unchanged; no training or label inference occurs.

        Raises:
            ValueError: Missing train members, invalid channels, or invalid sidecar alignment.
            OSError: Unreadable input or failed managed file publication.
        """
        self.outputs.dataset_preflight(name=dataset_name, version=dataset_version)
        names = SurfaceFeatureSchema.resolve(feature_names, feature_group, exclude)
        if not names:
            raise ValueError("surface features require at least one selected channel")
        members = list(DatasetIndex(dataset / "index.jsonl"))

        # Cache only reconstructible, label-free projections. The key depends on exact base
        # bytes, ordered channels and scientific kernel choices, never worker count or labels.

        def project(member: dict[str, str]) -> str:
            """Restore one projection through the managed file boundary.

            Args:
                member: JSON-compatible ID and immutable base path record.

            Returns:
                Path to a completely validated managed feature sidecar.
            """
            base = Path(member["base"])
            signature = json.dumps({"base": hashlib.sha256(base.read_bytes()).hexdigest(),
                                    "names": names, "sigma": sigma,
                                    "schema": SurfaceFeatureSchema.VERSION}, sort_keys=True)
            key = hashlib.sha256(signature.encode()).hexdigest()
            result = self.cache.file(
                f"surface-fields/{key}.npz",
                build=partial(SurfaceFeatureSidecar.write, base, names=names, sigma=sigma),
                validate=partial(SurfaceFeatureSidecar.validate, base=base, names=names),
            )
            return str(result)

        self.log(f"Projecting {len(names)} fixed fields for {len(members)} proteins")
        jobs = [{"id": member.member_id,
                 "base": str(dataset / member.assets["universal_npz"].path)} for member in members]
        paths = self.resume_map(jobs, project, key=lambda row: row["id"], workers=workers,
                                executor="thread", name="surface-fields")

        # Validation and test never contribute even a mean or variance. The frozen transform
        # records the exact training IDs/digests for a separate scientific leakage audit.

        train = [(member.member_id, dataset / member.assets["universal_npz"].path, Path(path))
                 for member, path in zip(members, paths, strict=True)
                 if member.partitions.get("split") == "train"]
        populations = {"full": train}
        for member, path in zip(members, paths, strict=True):
            if member.partitions.get("split") == "train":
                for view in member.metadata.get("dilutions", ()):
                    populations.setdefault(str(view), []).append(
                        (member.member_id,
                         dataset / member.assets["universal_npz"].path, Path(path))
                    )
        statistics: dict[str, Any] = {
            "schema_version": "2.0",
            "populations": {
                view: SurfaceFeatureSidecar.fit_statistics(records, names,
                    weighting=normalization_weighting, population=view)
                for view, records in sorted(populations.items())
            },
        }
        stats_path = Path(self.outputs.file("surface-feature-statistics.json", role="report"))
        stats_path.write_text(json.dumps(statistics, indent=2), encoding="utf-8")
        audit_path = Path(self.outputs.file("surface-feature-audit.md", role="report"))
        sections = ["# Surface feature audit", "",
                    "Only the named training population contributes to each fit. Validation and "
                    "test reuse that frozen transform. No channel is removed automatically.", ""]
        for view, fitted in statistics["populations"].items():
            audit = fitted["audit"]
            sections.extend([f"## {view}", "", f"Proteins: {len(fitted['sources'])}; "
                f"points: {fitted['count']}; weighting: {normalization_weighting}.", "",
                "| Channel | Mean | Standard deviation | Minimum | Maximum | Zero fraction |",
                "| --- | ---: | ---: | ---: | ---: | ---: |"])
            for i, name in enumerate(names):
                sections.append(f"| {name} | {fitted['mean'][i]:.6g} | {fitted['std'][i]:.6g} | "
                    f"{audit['minimum'][i]:.6g} | {audit['maximum'][i]:.6g} | "
                    f"{audit['zero_fraction'][i]:.3f} |")
            sections.extend(["", "Constant channels cannot discriminate within this training "
                "population. Zero-only channels have no measured support. Correlations near "
                "±1 flag possible redundancy; they do not establish scientific irrelevance.",
                "", f"Flags: `{json.dumps(audit['highly_correlated_pairs'])}`", ""])
            sections.extend([
                "Mean and standard deviation use the chosen statistical weights. Zero/nonzero "
                "and finite fractions describe unweighted point support, not physical area.", "",
                "| Channel | Nonzero fraction | Finite fraction |",
                "| --- | ---: | ---: |",
            ])
            for i, name in enumerate(names):
                sections.append(f"| {name} | {audit['nonzero_fraction'][i]:.3f} | "
                                f"{audit['finite_fraction'][i]:.3f} |")
            sections.extend(["", "Pairwise correlation describes co-variation on this training "
                "population. A constant channel has undefined correlation (unavailable), "
                "not evidence of independence. No validation/test point contributes.", "",
                "| Channel | " + " | ".join(names) + " |",
                "| --- | " + " | ".join("---:" for _ in names) + " |"])
            for name, correlations in zip(names, audit["correlation"], strict=True):
                rendered = ["unavailable" if value is None else f"{value:.3f}"
                            for value in correlations]
                sections.append(f"| {name} | " + " | ".join(rendered) + " |")
            sections.append("")
        audit_path.write_text("\n".join(sections), encoding="utf-8")

        # Native publication accepts only Run-owned assets. Import the immutable input once;
        # LF owns copying, hashing and containment checks, including reconstructible cache files.

        source = self.outputs.artifact("source-dataset", dataset)
        published = []
        for member, path in zip(members, paths, strict=True):
            assets = {name: source / asset.path for name, asset in member.assets.items()}
            fields = self.outputs.artifact(f"surface-fields-{member.member_id}", Path(path))
            assets.update(surface_features=fields, surface_feature_statistics=stats_path)
            published.append({"id": member.member_id, "assets": assets,
                              "targets": dict(member.targets),
                              "partitions": dict(member.partitions),
                              "metadata": {**dict(member.metadata),
                                           "optional_surface_feature_assets": ["surface_features"]},
                              })
        return dict(self.outputs.dataset(
            name=dataset_name, version=dataset_version, members=published,
            metadata={"optional_surface_fields": list(names),
                      "field_schema": SurfaceFeatureSchema.VERSION,
                      "normalization": normalization_weighting,
                      "normalization_populations": sorted(populations)},
        ))
