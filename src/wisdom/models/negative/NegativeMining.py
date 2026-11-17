"""VN01: current negative components, deterministic tracking, and differentiable selective loss."""

import math
import numpy as np

from typing import Any
from torch import Tensor
from torch.nn import functional as F
from collections.abc import Mapping, Sequence
from wisdom.models.negative.SurfaceRegions import SurfaceRegions
from wisdom.models.negative.RegionDiversity import RegionDiversity
from wisdom.models.negative.NegativeMiningMode import NegativeMiningMode


class NegativeMining:
    """Keep run-local supports; never consume local targets or archived logits as labels."""

    def __init__(self, options: Mapping[str, Any],
                 geometry: Mapping[str, SurfaceRegions], negative_fraction: float) -> None:
        """Configure the global-per-Run budget and optional N2 selector.

        Args:
            options: mode=regional (none|regional|point|distributed|audit), lower=.25, upper=.40,
                regions=3, strong_weight=.1, memory=false, weak_ratio=.25, link_overlap=.5,
                memory_budget=4*regions, memory_epochs=5, history_budget=10*(K+B),
                per_protein=0 (unlimited), diversity=null or RegionDiversity options.
                strong_weight is an uncalibrated pilot default, not a scientific optimum.
            geometry: Fixed negative TRAIN-only geometry keyed by immutable identifier.
            negative_fraction: N_negative/N_train under uniform protein sampling, in (0,1].

        Raises:
            ValueError: Invalid probability/weight/budget, no negatives, or an incoherent control.
        """
        self.options = dict(options)
        self.geometry = dict(geometry)
        self.mode = NegativeMiningMode(options.get("mode", "regional"))
        self.lower = float(options.get("lower", .25))
        self.upper = float(options.get("upper", .40))
        self.k = int(options.get("regions", 3))
        self.strong = float(options.get("strong_weight", .1))
        self.ratio = float(options.get("weak_ratio", .25))
        self.memory = bool(options.get("memory", False))
        self.link = float(options.get("link_overlap", .5))
        self.budget = int(options.get("memory_budget", 4 * self.k))
        self.age = int(options.get("memory_epochs", 5))
        self.history_budget = int(options.get("history_budget", 10 * (self.k + self.budget)))
        self.cap = int(options.get("per_protein", 0))
        self.fraction = negative_fraction
        if not (0 < self.lower < self.upper < 1 and 0 < negative_fraction <= 1):
            raise ValueError("VN01 needs 0 < lower < upper < 1 and training negatives")
        if (self.k < 1 or not math.isfinite(self.strong) or self.strong < 0
                or not 0 <= self.ratio < 1
                or not 0 < self.link <= 1 or min(self.budget, self.age, self.cap) < 0
                or self.history_budget < 1):
            raise ValueError("invalid negative-mining mode, coefficient, tracking or region budget")
        if (self.mode is not NegativeMiningMode.REGIONAL and
                (self.memory or options.get("diversity"))):
            raise ValueError("point/distributed/audit controls cannot silently add memory or FGW")
        self.diversity = (RegionDiversity(geometry, options["diversity"])
                          if options.get("diversity") else None)
        self.active: list[dict[str, Any]] = []
        self.history: list[dict[str, Any]] = []
        self.review_epoch = 0
        self.next_id = 0
        self.masks: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        self.last_audit: dict[str, Any] = {}

    @staticmethod
    def priority(region: Mapping[str, Any]) -> tuple[float, str, int]:
        """Return descending difficulty with stable protein/point-index tie breaks.

        Args:
            region: A current component with score, protein identifier and sorted support.

        Returns:
            Comparable tuple independent of review-loader or batch order.
        """
        return -float(region["score"]), str(region["protein"]), int(region["support"][0])

    def review(self, probabilities: Mapping[str, np.ndarray], epoch: int) -> dict[str, Any]:
        """Replace supports with this epoch's components and prepare next-epoch masks.

        Args:
            probabilities: Deterministic current operational probabilities for EVERY train negative.
            epoch: Completed optimizer epoch, strictly later than the saved review epoch.

        Returns:
            JSON-compatible coverage, lineage, memory, numerical and family diagnostics. Splits
            and merges keep current supports once; no historical mask receives its own loss.

        Raises:
            ValueError: Missing negatives, repeated reviews, or changed point discretization.
        """
        if set(probabilities) != set(self.geometry) or epoch <= self.review_epoch:
            raise ValueError("VN01 review requires every train negative and a new completed epoch")
        current = [{**region, "protein": protein} for protein in sorted(probabilities)
            for region in self.geometry[protein].components(probabilities[protein], self.lower)]
        current.sort(key=self.priority)
        links: list[list[dict[str, Any]]] = [[] for _ in current]
        children: dict[int, list[int]] = {r["id"]: [] for r in self.active}
        for index, region in enumerate(current):
            weights = self.geometry[region["protein"]].weights
            support = np.asarray(region["support"])
            for old in self.active:
                if old["protein"] != region["protein"]:
                    continue
                overlap = np.intersect1d(support, old["support"], assume_unique=True)
                mass = float(weights[overlap].sum())
                denominator = min(float(weights[support].sum()),
                                  float(weights[old["support"]].sum()))
                if (mass / denominator >= self.link and
                        (old["anchor"] in overlap or region["anchor"] in overlap)):
                    links[index].append(old)
                    children[old["id"]].append(index)

        events = []
        for region, parents in zip(current, links, strict=True):
            same = len(parents) == 1 and len(children[parents[0]["id"]]) == 1
            if same:
                region["id"] = parents[0]["id"]
            else:
                region["id"] = self.next_id
                self.next_id += 1
            region["parents"] = [p["id"] for p in parents]
            region["last_priority"] = max((p["last_priority"] for p in parents), default=-1)
            region["activation_epoch"] = min((p["activation_epoch"] for p in parents),
                                              default=epoch)
            region["recurrences"] = max((p.get("recurrences", 0) for p in parents), default=0)
            if len(parents) > 1:
                events.append({"kind": "merge", "id": region["id"], "parents": region["parents"]})
        for old in self.active:
            if len(children[old["id"]]) > 1:
                events.append({"kind": "split", "id": old["id"]})
            elif not children[old["id"]]:
                # The old support is never reused. Remaining high points without a trusted
                # correspondence mean lost tracking, not proven correction.

                values = probabilities[old["protein"]][old["support"]]
                nearby = self.geometry[old["protein"]].graph[old["support"]].indices
                unresolved = (np.any(values >= self.lower) or
                    np.any(probabilities[old["protein"]][nearby] >= self.lower))
                events.append({"kind": "tracking_lost" if unresolved else "corrected",
                               "id": old["id"]})

        remembered = []
        if self.memory:
            for region in current:
                if region["last_priority"] < 0:
                    continue
                if epoch - region["last_priority"] > self.age:
                    events.append({"kind": "expired", "id": region["id"]})
                else:
                    remembered.append(region)
            remembered.sort(key=lambda r: (-r["last_priority"], *self.priority(r), r["id"]))
            events.extend({"kind": "evicted", "id": r["id"]} for r in remembered[self.budget:])
            remembered = remembered[:self.budget]
        eligible = [r for r in current if r["score"] >= self.upper]
        family_audit: dict[str, Any] = {}
        if self.diversity and (eligible or remembered):
            universe = sorted({r["id"]: r for r in [*eligible, *remembered]}.values(),
                              key=self.priority)
            labels, family_audit = self.diversity.cluster(universe, self.geometry)
            families = {r["id"]: label for r, label in zip(universe, labels, strict=True)}
            eligible = self.diversity.order(eligible, [families[r["id"]] for r in eligible],
                int(self.diversity.options.get("noise_quota", 1)))
        selected: list[dict[str, Any]] = []
        counts: dict[str, int] = {}
        for region in eligible:
            protein = region["protein"]
            if self.cap and counts.get(protein, 0) >= self.cap:
                continue
            selected.append(region)
            counts[protein] = counts.get(protein, 0) + 1
            if len(selected) == self.k:
                break
        chosen = {r["id"] for r in selected}
        weak = [r for r in remembered if r["id"] not in chosen]
        for region in selected:
            events.append({"kind": "priority" if region["last_priority"] >= 0 else "activation",
                           "id": region["id"], "protein": region["protein"]})
            region["recurrences"] += int(region["last_priority"] >= 0)
            region["last_priority"] = epoch
            region["strength"] = 1.0
        for region in weak:
            region["strength"] = self.ratio
        self.active = [*selected, *weak]
        self.review_epoch = epoch
        self.history.extend({**event, "epoch": epoch} for event in events)
        self.history = self.history[-self.history_budget:]
        self._masks(probabilities)
        self.last_audit = {"epoch": epoch, "negative_proteins": len(self.geometry),
            "components": len(current), "candidates": len(eligible), "selected": len(selected),
            "remembered": len(weak), "selected_proteins": len(counts), "events": events,
            "selected_regions": self.active, "families": family_audit,
            "selected_area": sum(r["area"] for r in selected),
            "candidate_proteins": len({r["protein"] for r in eligible}),
            "catalog_records": len(self.history)}
        self.last_audit.update({"events_" + kind: sum(e["kind"] == kind for e in events)
            for kind in ("activation", "priority", "merge", "split", "corrected",
                         "tracking_lost", "expired", "evicted")})
        self.last_audit["weighted_support_area"] = sum(float(
            self.geometry[p].weights @ (strong + weak))
            for p, (strong, weak) in self.masks.items())
        return self.last_audit

    def _masks(self, probabilities: Mapping[str, np.ndarray] | None = None) -> None:
        """Construct disjoint strong/weak masks from current supports, never detached targets.

        Args:
            probabilities: Current snapshot used only for the area-matched point control.
                None restores already persisted point-control masks without recomputing a field.
        """
        self.masks = {p: (np.zeros(len(g.weights)), np.zeros(len(g.weights)))
                      for p, g in self.geometry.items()}
        for region in self.active:
            strong, weak = self.masks[region["protein"]]
            target = strong if region["strength"] == 1.0 else weak
            target[region["support"]] = region["strength"]
        if self.mode is NegativeMiningMode.POINT and probabilities is not None:
            for protein, (strong, _) in self.masks.items():
                weights = self.geometry[protein].weights
                budget = float(weights @ strong)
                strong[:] = 0
                order = np.lexsort((np.arange(len(weights)), -probabilities[protein]))
                for point in order:
                    if budget <= 1e-15:
                        break
                    strong[point] = min(1.0, budget / weights[point])
                    budget -= strong[point] * weights[point]

    def loss(self, logits: Tensor, weights: Tensor, pointers: Tensor,
             identifiers: Sequence[str], targets: Tensor) -> Tensor:
        """Apply current-logit excess on selected supports with whole-protein proxy area.

        Args:
            logits: Operational, differentiable post-refinement logits [M].
            weights: Positive unnormalized surface proxy areas [M].
            pointers: CPU surface prefix boundaries [B+1].
            identifiers: Immutable protein IDs in the same B-member order.
            targets: Global labels [B]; no local targets are accepted by this interface.

        Returns:
            FP32 sum_b J_b/(B*pi_negative). Selected q=relu(z-logit(lower))^2;
            distributed control uses softplus(z). Empty batches/supports yield connected zero.
        """
        result = logits.float().sum() * 0
        if self.mode in {NegativeMiningMode.NONE, NegativeMiningMode.AUDIT}:
            return result
        labels = targets.detach().cpu().tolist()
        for index, identifier in enumerate(identifiers):
            if labels[index] != 0:
                continue
            start, stop = int(pointers[index]), int(pointers[index + 1])
            current = logits[start:stop].float()
            area = weights[start:stop].float()
            area = area / area.sum()
            if self.mode is NegativeMiningMode.DISTRIBUTED:
                penalty = F.softplus(current)
            else:
                strong, weak = self.masks.get(identifier, (np.zeros(stop-start),
                                                           np.zeros(stop-start)))
                mask = current.new_tensor(strong + weak)
                penalty = (current - math.log(self.lower / (1 - self.lower))).relu().square() * mask
            result = result + self.strong * (area * penalty).sum()
        return result / (len(identifiers) * self.fraction)

    def state_dict(self) -> dict[str, Any]:
        """Return bounded JSON-shaped tracking state with exact geometry/configuration identity."""
        return {"options": self.options,
            "geometry": {p: g.identity for p, g in self.geometry.items()},
            "attribute_center": self.diversity.center.tolist() if self.diversity else None,
            "attribute_scale": self.diversity.spread.tolist() if self.diversity else None,
            "active": self.active, "history": self.history, "review_epoch": self.review_epoch,
            "next_id": self.next_id, "masks": {p: [a.tolist(), b.tolist()]
                                              for p, (a, b) in self.masks.items()}}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore masks only if geometry, point order and scientific options are unchanged.

        Args:
            state: Native continuation state generated by state_dict, not a new model review.

        Raises:
            ValueError: Different point discretization or scientific mining settings.
        """
        if state["options"] != self.options or state["geometry"] != {
                p: g.identity for p, g in self.geometry.items()}:
            raise ValueError("cannot restore VN01 memory on different geometry or configuration")
        self.active = list(state["active"])
        self.history = list(state["history"])
        self.review_epoch = int(state["review_epoch"])
        self.next_id = int(state["next_id"])
        self.masks = {p: (np.asarray(values[0]), np.asarray(values[1]))
                      for p, values in state["masks"].items()}
        if self.diversity:
            self.diversity.center = np.asarray(state["attribute_center"])
            self.diversity.spread = np.asarray(state["attribute_scale"])
