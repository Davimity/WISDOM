"""Training-only VF normalization, sampled-query loss and original-unit reconstruction metrics."""

import torch
import hashlib
import numpy as np

from typing import Any
from pathlib import Path
from torch import Tensor
from collections.abc import Iterable, Mapping
from wisdom.preprocessing.common.structure.assets.physics.FieldQueries import FieldQueries


class PhysicalObjective:
    """Keep references on the CPU and pass only sampled geometry into the neural decoder."""

    FIELDS = ("potential", "lj1", "lj2")

    def __init__(self, options: Mapping[str, Any]) -> None:
        """Configure optimization scales without allowing field targets into input features.

        Args:
            options: fields defaults to all three; weight=0.1, weights=null, queries=512,
                sampling=stratified, loss=huber, beta=1, seed=2026. Explicit weights supply
                individual lambda_s and cannot coexist with a global weight. LJ transforms are
                asinh(U/u0), u0 fitted on valid physical train only; potential stays signed.
        """
        self.options = dict(options)
        self.fields = tuple(options.get("fields", self.FIELDS))
        if set(self.fields) - set(self.FIELDS) or not self.fields:
            raise ValueError("VF fields must be a nonempty subset of potential/lj1/lj2")
        if "weights" in options and "weight" in options:
            raise ValueError("choose one VF global weight or individual field weights, not both")
        individual = options.get("weights")
        if individual is not None and set(individual) - set(self.FIELDS):
            raise ValueError("VF individual weights contain an unknown physical field")
        self.weights = torch.tensor([float(individual.get(name, 0.0) if individual is not None
            else options.get("weight", .1)) if name in self.fields else 0.0
            for name in self.FIELDS])
        if not torch.isfinite(self.weights).all() or torch.any(self.weights < 0):
            raise ValueError("VF field loss weights must be finite and nonnegative")
        self.scale = torch.ones(3)
        self.mean  = torch.zeros(3)
        self.std   = torch.ones(3)
        self.loss_kind = str(options.get("loss", "huber"))
        if self.loss_kind not in {"huber", "mse"}:
            raise ValueError("VF loss must be huber or mse")
        if float(options.get("beta", 1)) <= 0:
            raise ValueError("VF Huber beta must be positive")

    def fit(self, samples: Iterable[Mapping[str, Any] | Path]) -> dict[str, Any]:
        """Fit signed transformations and scales exclusively from valid physical train rows.

        Args:
            samples: Training-only query-bank Paths, or unbatched members containing
                physical_queries__targets [Q,3]/physical_queries__valid [Q,3]. File input reads
                just targets/masks, avoiding geometry, spectral operators and sparse candidates.
                Validation/test must never enter this iterable.

        Returns:
            Serializable fixed scales/means/standard deviations and valid support counts. Missing
            a selected field on every training protein is a hard error, never a zero loss claim.
        """
        rows: list[list[Tensor]] = [[], [], []]
        for sample in samples:
            if isinstance(sample, Path):
                with np.load(sample, allow_pickle=False) as stored:
                    target = torch.from_numpy(stored["targets"]).float()
                    valid = torch.from_numpy(stored["valid"]).bool()
            else:
                target = sample["physical_queries__targets"].float()
                valid = sample["physical_queries__valid"].bool()
            for channel in range(3):
                rows[channel].append(target[:, channel][valid[:, channel]])
        counts = []
        for channel, values in enumerate(rows):
            joined = torch.cat(values) if values else torch.empty(0)
            counts.append(len(joined))
            if not len(joined):
                if self.FIELDS[channel] in self.fields:
                    raise ValueError(f"VF training has no valid {self.FIELDS[channel]} references")
                continue
            if channel:
                self.scale[channel] = joined.abs().median().clamp_min(1e-6)
                joined = torch.asinh(joined / self.scale[channel])
            self.mean[channel] = joined.mean()
            self.std[channel] = joined.std(unbiased=False).clamp_min(1e-6)
        return {"scale": self.scale.tolist(), "mean": self.mean.tolist(),
                "std": self.std.tolist(), "valid_train_queries": counts,
                "valid_train_proteins": [sum(bool(len(value)) for value in field)
                                         for field in rows],
                "fit_split": "physical_train_only", "units": ["kBT/e", "kJ/mol", "kJ/mol"]}

    def queries(self, batch: Mapping[str, Any], device: torch.device, epoch: int = 0,
                full: bool = False, maximum: int | None = None
                ) -> tuple[dict[str, Tensor], dict[str, Tensor]]:
        """Sample detached references and sparse geometry with explicit protein ownership.

        Args:
            batch: Collated CPU graph with physical_queries protein-local member dictionaries.
            device: Granted logical model device, never a physical GPU ordinal.
            epoch: Training sampling epoch; validation uses zero for repeatable fixed queries.
            full: Evaluate every bank row instead of a sampled optimizer batch.
            maximum: Optional evaluation count override; None uses the training query count.

        Returns:
            Decoder geometry only and separate targets/masks/regions/owners. No target, partial
            charge, force-field sigma or epsilon is placed in the decoder mapping.
        """
        members = batch["structural_assets"]["physical_queries"]["members"]
        positions, targets, masks, regions, owners, pair_rows, pair_atoms = ([] for _ in range(7))
        atom_offset = query_offset = 0
        atom_counts = torch.bincount(batch["atom_batch"], minlength=len(members)).tolist()
        for owner, member in enumerate(members):
            region = member["region"].numpy()
            valid = member["valid"].numpy()
            identity = str(batch["identifier"][owner]).encode()
            member_seed = int.from_bytes(hashlib.sha256(identity).digest()[:4], "little")
            seed = int(self.options.get("seed", 2026)) + epoch * 104729 + member_seed
            ids = np.arange(len(region)) if full else FieldQueries.sample(region, valid,
                int(self.options.get("queries", 512)) if maximum is None else maximum, seed,
                str(self.options.get("sampling", "stratified")))
            selected = torch.as_tensor(ids, dtype=torch.long)
            positions.append(member["positions"][selected])
            targets.append(member["targets"][selected])
            masks.append(member["valid"][selected])
            regions.append(member["region"][selected])
            owners.append(torch.full((len(ids),), owner, dtype=torch.long))
            offsets = member["offsets"]
            for row, index in enumerate(ids):
                atoms = member["atom_ids"][int(offsets[index]):int(offsets[index + 1])]
                pair_atoms.append(atoms + atom_offset)
                pair_rows.append(torch.full((len(atoms),), query_offset + row, dtype=torch.long))
            atom_offset += atom_counts[owner]
            query_offset += len(ids)
        geometry = {"positions": torch.cat(positions).to(device),
            "owners": torch.cat(owners).to(device),
            "pair_query": (torch.cat(pair_rows) if pair_rows else
                           torch.empty(0, dtype=torch.long)).to(device),
            "pair_atom": (torch.cat(pair_atoms) if pair_atoms else
                          torch.empty(0, dtype=torch.long)).to(device),
            "protein_count": torch.tensor(len(members))}
        references = {"targets": torch.cat(targets).to(device),
                      "valid": torch.cat(masks).to(device),
                      "region": torch.cat(regions).to(device), "owners": geometry["owners"]}
        return geometry, references

    def loss(self, prediction: Tensor, references: Mapping[str, Tensor]) -> dict[str, Tensor]:
        """Compute each masked field loss; never interpret unsupported proteins as zero error.

        Args:
            prediction: Normalized decoder outputs [Q,3].
            references: Original-unit targets [Q,3], per-field valid masks and protein ownership.

        Returns:
            Available per-field losses and their explicitly weighted total. The denominator is
            valid query count per field, not all points or an invented common validity mask.
        """
        target = references["targets"].float().clone()
        scale, mean, std = (value.to(prediction.device) for value in (
            self.scale, self.mean, self.std))
        target[:, 1:] = torch.asinh(target[:, 1:] / scale[1:])
        target = (target - mean) / std
        errors = (prediction.float() - target)
        if self.loss_kind == "huber":
            beta = float(self.options.get("beta", 1.0))
            absolute = errors.abs()
            errors = torch.where(absolute < beta, .5 * errors.square() / beta,
                                 absolute - .5 * beta)
        else:
            errors = errors.square()
        result = {}
        total = prediction.sum() * 0.0
        for channel, name in enumerate(self.FIELDS):
            valid = references["valid"][:, channel]
            if name in self.fields and bool(valid.any()):
                value = errors[valid, channel].mean()
                result[name] = value
                total = total + self.weights[channel].to(prediction.device) * value
        result["total"] = total
        return result

    def decode(self, prediction: Tensor) -> Tensor:
        """Invert the fixed train-only transform for honest original-unit evaluation.

        Args:
            prediction: Transformed/standardized field predictions [Q,3].

        Returns:
            Potential in kBT/e and two LJ energies in kJ/mol, preserving sign. Overflow is not
            silently clipped into an apparently accurate finite physical prediction.
        """
        result = (prediction.float() * self.std.to(prediction.device)
                  + self.mean.to(prediction.device))
        result = torch.cat((result[:, :1], torch.sinh(result[:, 1:]) *
                            self.scale[1:].to(prediction.device)), dim=1)
        return result
