"""Scientific metric meaning; all profiling, discovery, and ranking remain in LambdaForge."""

# Catalog descriptions remain contiguous literals so their full scientific meaning is reviewable.
# ruff: noqa: E501

from typing import Any, ClassVar
from collections.abc import Sequence


class WisdomAnalysisProfile:
    """Build one versioned declaration consumed by ``Work.analysis_profile``.

    This is a catalog, not an evaluator: it never reads targets, chooses checkpoints, fits
    relationships, or changes search policy. Numeric priorities focus the framework's bounded
    exploration on global quality, localization, and their agreement before instrumented costs.
    Optional questions preserve genuinely unavailable attention, intervention, and view evidence.
    """

    VERSION = 1
    BINARY: ClassVar[dict[str, str]] = {
        "accuracy":          "Fraction of proteins correctly classified at probability 0.5.",
        "balanced_accuracy": "Mean sensitivity and specificity at probability 0.5; both classes are required.",
        "precision":         "Fraction of predicted positives that are truly positive; undefined without predicted positives.",
        "recall":            "Fraction of true positives recovered at probability 0.5.",
        "specificity":       "Fraction of true negatives rejected at probability 0.5.",
        "f1":                "Harmonic mean of precision and recall at probability 0.5.",
        "mcc":               "Matthews correlation of binary decisions and labels; undefined for a zero confusion-matrix denominator.",
        "cohen_kappa":       "Agreement of thresholded decisions with labels after correcting for chance agreement.",
        "auroc":             "Probability that a positive protein outranks a negative protein, with half credit for ties.",
        "auprc":             "Precision-recall area for protein ranking; its random-ranking baseline depends on positive prevalence.",
    }
    SUBGROUP: ClassVar[dict[str, str]] = {
        "count":                       "Number of proteins supporting this subgroup.",
        "positive_count":              "Number of globally positive proteins in this subgroup.",
        "surface_count":               "Number of positive proteins with usable, nonconstant local ground truth.",
        "protein_auprc":               "Protein ranking AUPRC inside this subgroup, not the full validation population.",
        "protein_auroc":               "Protein ranking AUROC inside this subgroup; unavailable without both classes.",
        "global_score":                "Subgroup G=0.70*AUPRC+0.30*AUROC at the evaluated checkpoint.",
        "surface_auprc":               "Mean point AUPRC across locally evaluable positive proteins in this subgroup.",
        "global_at_selection":         "Largest observed subgroup G; the subgroup's own best epoch need not be the full population's selected epoch.",
        "surface_at_global_selection": "Subgroup surface AUPRC at the subgroup's own G-selected epoch.",
        "surface_regret":              "Best observed subgroup surface AUPRC minus its value at the subgroup's G-selected epoch.",
        "coupling":                    "Subgroup curve summary combining normalized regret and late-trajectory rank agreement.",
        "wisdom_score":                "Subgroup W=0.35*G+0.45*S+0.20*C; diagnostic only, never a checkpoint selector.",
    }

    def __init__(self) -> None:
        """Create fresh declaration containers; no scientific or framework state is accessed."""
        self.metrics  : dict[str, Any] = {}
        self.families : dict[str, Any] = {}
        self.defaults : list[dict[str, Any]] = []
        self.questions: list[dict[str, Any]] = []

    def build(
        self,
        global_phenotypes   : Sequence[str] = (),
        local_phenotypes: Sequence[str] = (),
    ) -> dict[str, Any]:
        """Return native class-profile declarations, ready for LambdaForge validation.

        Args:
            global_phenotypes: Optional known sanitized strata, e.g. ``g001`` and ``g_noise``.
                Unspecified dataset-dependent strata receive pattern metadata, not invented members.
            local_phenotypes: Optional known sanitized interface strata, e.g. ``i001``.

        Returns:
            Native metrics/defaults/families/questions mapping. The framework merges YAML
            overrides and freezes its own version and semantic identity before any Run starts.
            Calling again produces fresh data rather than accumulating duplicate questions.
        """
        self.metrics   = {}
        self.families  = {}
        self.defaults  = []
        self.questions = []

        # Unknown future metrics remain inspectable, but cannot dominate exploratory findings.
        # Split rules follow the actual keys produced by metrics.log(..., split=...).

        self.defaults.append({"pattern": "*", "metadata": {
            "description": "Unclassified metric; inspect its producer before drawing a scientific conclusion.",
            "visibility":  "hidden",
            "discovery":   False,
            "notes":       f"WISDOM research semantics v{self.VERSION}.",
        }})
        for prefix, split in (("train", "train"), ("val", "validation"), ("test", "test")):
            self.defaults.append({"pattern": f"{prefix}_*", "metadata": {"split": split}})
        self.defaults.append({"pattern": "resource.*", "metadata": {
            "description": "Framework-observed intrinsic cost per comparable screening Run; not controller spend or an epoch sample.",
            "category":    "runtime/framework",
            "role":        "resource",
            "direction":   "min",
            "visibility":  "advanced",
            "priority":    20,
            "discovery":   True,
            "aggregation": "terminal",
        }})

        for prefix, split in (("val", "validation"), ("test", "test")):
            self._global(prefix, split)
            self._surface(prefix, split)
            self._faithfulness(prefix, split)
            self._subgroups(prefix, split, global_phenotypes, local_phenotypes)
            self._views(prefix, split)
        self._coupling()
        self._operational()
        self._questions()
        return {
            "metrics":  self.metrics,
            "defaults": self.defaults,
            "families": self.families,
            "questions": self.questions,
        }

    def _metric(
        self,
        name       : str,
        description: str,
        category   : str,
        direction  : str = "unknown",
        bounds     : tuple[float, float] | None = None,
        priority   : int = 40,
        **metadata : Any,
    ) -> None:
        """Declare one stable metric, with actual units and evidence aggregation.

        Args:
            name: Exact logged key, including its split prefix where present.
            description: Scientific meaning, missing-value behavior, and limiting interpretation.
            category: Slash-separated domain hierarchy, never a substitute for split semantics.
            direction: ``max``, ``min``, or ``unknown``; no good/bad ordering is invented.
            bounds: Known mathematical bounds, or None for an unbounded statistic.
            priority: Relative browsing/discovery interest, not a scientific confidence threshold.
            **metadata: Native LambdaForge metadata fields overriding the common defaults.
        """
        entry = {
            "label":       name.removeprefix("val_").removeprefix("test_").replace("_", " ").capitalize(),
            "description": description,
            "category":    category,
            "tags":        category.split("/"),
            "direction":   direction,
            "unit":        "fraction" if bounds == (0, 1) else "dimensionless",
            "role":        "quality",
            "priority":    priority,
            "visibility":  "primary" if priority >= 80 else "normal",
            "discovery":   True,
            "aggregation": "terminal",
            **metadata,
        }
        if bounds is not None:
            entry["range"] = list(bounds)
        self.metrics[name] = entry

    def _global(self, prefix: str, split: str) -> None:
        """Describe binary rankings and thresholded outcomes at the G-selected checkpoint.

        Args:
            prefix: ``val`` or ``test``, matching persisted metric keys.
            split: Explicit native ``validation`` or ``test`` evidence partition.
        """
        for name, description in self.BINARY.items():
            priority = 85 if name in {"auprc", "auroc", "mcc"} else 45
            if name in {"accuracy", "cohen_kappa"}:
                priority = 20
            category = "quality" if name in {"auprc", "auroc"} else "thresholded"
            bounds = (-1, 1) if name in {"mcc", "cohen_kappa"} else (0, 1)
            self._metric(f"{prefix}_{name}", description, f"{split}/global/{category}",
                         "max", bounds, priority, split=split, aggregation="selected_epoch")
            self._metric(f"{prefix}_surface_derived_{name}",
                         "The surface-derived protein head only. " + description,
                         f"{split}/global/surface-head", "max", bounds, split=split,
                         aggregation="selected_epoch")
        for name in ("loss", "surface_derived_loss"):
            self._metric(f"{prefix}_{name}",
                         "Mean protein binary cross entropy at the globally selected checkpoint; not a surface-target loss.",
                         f"{split}/global/calibration", "min", priority=25, split=split,
                         role="diagnostic", aggregation="selected_epoch")
        self._metric(f"{prefix}_head_disagreement",
                     "Mean absolute probability gap between the direct and surface-derived protein heads; disagreement is not proof of error.",
                     f"{split}/surface/calibration", bounds=(0, 1), split=split,
                     aggregation="selected_epoch")

    def _surface(self, prefix: str, split: str) -> None:
        """Keep macro localization, pooled points, ranking, and negative controls distinct.

        Args:
            prefix: Actual persisted split prefix.
            split: Native partition name used independently of the category path.
        """
        for aggregation in ("micro", "positive_macro"):
            weighting = (
                "Pool all valid points, including negatives; larger point clouds contribute more observations."
                if aggregation == "micro" else
                "Compute per-positive-protein scores, then average equally over evaluable proteins; invalid/ambiguous points are excluded."
            )
            for name in ("auprc", "auroc", "balanced_accuracy", "f1"):
                meaning = self.BINARY[name].replace("protein", "point")
                priority = 90 if aggregation == "positive_macro" and name in {"auprc", "auroc"} else 45
                self._metric(f"{prefix}_surface_{aggregation}_{name}", meaning + " " + weighting,
                             f"{split}/surface/quality", "max", (0, 1), priority,
                             split=split, aggregation="selected_epoch")
        self.metrics[f"{prefix}_surface_positive_macro_auprc"].update(
            label="Surface macro AUPRC",
            aliases=[f"{split} surface auprc", f"{split} localization quality", f"{split} local auprc"],
        )
        for suffix, description, direction, bounds in (
            ("positive_macro_auprc", "Equal-protein macro AUPRC before evidence refinement.",
             "max", (0, 1)),
            ("positive_macro_normalized_auprc", "Pre-refinement macro (AP-prevalence)/(1-prevalence); zero is random ranking.",
             "max", None),
            ("negative_positive_mass", "Pre-refinement mean area-weighted probability on negative proteins.",
             "min", (0, 1)),
            ("negative_peak", "Pre-refinement mean peak probability on negative proteins.",
             "min", (0, 1)),
        ):
            self._metric(f"{prefix}_surface_raw_{suffix}", description,
                         f"{split}/surface/raw", direction, bounds,
                         split=split, aggregation="selected_epoch")
        self._metric(f"{prefix}_surface_refinement_gain",
                     "Refined minus raw positive-protein macro AUPRC on identical valid points; negative means the refiner worsened ranking. Evaluation only.",
                     f"{split}/surface/refinement", "max", (-1, 1), 80,
                     split=split, aggregation="selected_epoch")
        self._metric(f"{prefix}_surface_positive_macro_normalized_auprc",
                     "Mean (AUPRC-prevalence)/(1-prevalence) per positive protein. Zero is the random-ranking baseline; negative values are possible without a universal finite lower bound.",
                     f"{split}/surface/ranking", "max", split=split, aggregation="selected_epoch")

        # Numeric fractions use explicit members because persisted names contain percentages.

        for statistic in ("recall", "enrichment"):
            members = {}
            for fraction in (0.05, 0.10, 0.25):
                name = f"{prefix}_surface_positive_macro_top_{round(100 * fraction)}_{statistic}"
                description = (
                    f"Mean fraction of the true interface recovered among the highest-scored {100*fraction:g}% of valid points."
                    if statistic == "recall" else
                    f"Mean selected positive-point rate divided by baseline prevalence among the top {100*fraction:g}% of points; one is random ranking."
                )
                self._metric(name, description, f"{split}/surface/ranking", "max",
                             (0, 1) if statistic == "recall" else None,
                             split=split, aggregation="selected_epoch")
                members[name] = {"fraction": fraction}
            self.families[f"{prefix}_surface_top_{statistic}"] = {
                "label": f"{split.capitalize()} surface top-fraction {statistic}",
                "dimensions": {"fraction": {"kind": "numeric", "values": [0.05, 0.10, 0.25]}},
                "members": members,
            }
        for name, description in {
            "surface_negative_positive_mass": "Mean per-negative-protein area-weighted local probability: false activation over the whole negative surface.",
            "surface_negative_peak": "Mean maximum local probability per negative protein: the strongest false hotspot.",
        }.items():
            self._metric(f"{prefix}_{name}", description, f"{split}/surface/negative-control",
                         "min", (0, 1), split=split, aggregation="selected_epoch")
        for name, description, direction, bounds in (
            ("surface_attention_positive_macro_auprc", "Mean interface ranking AUPRC of pooling attention, not of local evidence logits.", "max", (0, 1)),
            ("surface_attention_logit_spearman", "Mean per-positive-protein rank agreement between attention weights and local sigmoid scores (same ranks as logits).", "max", (-1, 1)),
            ("surface_attention_entropy", "Mean normalized categorical attention entropy: zero is concentrated, one is uniform; neither extreme is universally preferable.", "unknown", (0, 1)),
        ):
            self._metric(f"{prefix}_{name}", description, f"{split}/surface/attention",
                         direction, bounds, split=split, aggregation="selected_epoch")
        self._metric(f"{prefix}_surface_confidence_quality_spearman",
                     "Across positive proteins, rank agreement of global confidence with each protein's local AUPRC; distinct from epoch-wise G/S coupling.",
                     f"{split}/surface/calibration", "max", (-1, 1), split=split, unit="correlation",
                     aggregation="selected_epoch")
        for signal in ("map_entropy", "head_disagreement"):
            description = (
                "Area-weighted binary probability entropy on positive surfaces"
                if signal == "map_entropy" else "absolute direct-versus-surface protein-head probability gap"
            )
            self._metric(f"{prefix}_surface_{signal}_error_spearman",
                         f"Rank agreement between {description} and 1-local AUPRC; positive values support its use as an uncertainty proxy, not calibrated error probabilities.",
                         f"{split}/surface/calibration", "max", (-1, 1), split=split,
                         unit="correlation", aggregation="selected_epoch")
            members = {}
            for coverage in (70, 80, 90):
                name = f"{prefix}_surface_{signal}_auprc_at_coverage_{coverage}"
                self._metric(name, f"Mean local AUPRC after retaining the {coverage}% least-uncertain positive proteins ranked by {description}; not a new checkpoint objective.",
                             f"{split}/surface/calibration", "max", (0, 1), split=split,
                             aggregation="selected_epoch")
                members[name] = {"coverage": coverage / 100}
            self.families[f"{prefix}_{signal}_coverage"] = {
                "label": f"{split.capitalize()} localization retained by {signal.replace('_', ' ')}",
                "dimensions": {"coverage": {"kind": "numeric", "values": [0.7, 0.8, 0.9]}},
                "members": members,
            }
        for name in ("surface_valid_points", "surface_positive_proteins", "surface_negative_proteins",
                     "surface_prediction_proteins", "surface_visualized_proteins"):
            self._metric(f"{prefix}_{name}", "Number of " + name.removeprefix("surface_").replace("_", " ") + " supporting evaluation or exported reports; not prediction quality.",
                         "diagnostics/counts", priority=0, split=split, role="support", unit="count",
                         visibility="hidden", discovery=False)

        # Raw and refined evidence share evaluation support and metric definitions, not values.
        # Keep their names separate so analysis can study the operator without duplicating outcomes.

        for name, declaration in list(self.metrics.items()):
            if name.startswith(f"{prefix}_surface_positive_macro_") and name.endswith(
                ("top_5_recall", "top_10_recall", "top_25_recall",
                 "top_5_enrichment", "top_10_enrichment", "top_25_enrichment",
                 "auroc", "balanced_accuracy", "f1")
            ):
                raw_name = name.replace("_surface_", "_surface_raw_", 1)
                self.metrics[raw_name] = {
                    **declaration,
                    "description": "Before evidence refinement: " + declaration["description"],
                    "category": f"{split}/surface/raw",
                }
        for name in ("valid_points", "positive_proteins", "negative_proteins"):
            self._metric(f"{prefix}_surface_raw_{name}",
                         "Evaluation support for raw evidence; same valid points/proteins as refined evidence.",
                         "diagnostics/counts", priority=0, split=split, role="support", unit="count",
                         visibility="hidden", discovery=False)

    def _coupling(self) -> None:
        """Declare the frozen G/S/C/W formulas and their structural, non-novel dependencies."""
        definitions = {
            "wisdom_hpo_global": ("Global quality G", "G=0.70*protein AUPRC+0.30*protein AUROC at the G-selected observed epoch.", "global", []),
            "wisdom_hpo_surface": ("Surface quality S", "Positive-protein macro surface AUPRC at the G-selected observed epoch.", "surface", []),
            "wisdom_hpo_coupling": ("Global-surface coupling C", "C=0.70*regret_component+0.30*(rho+1)/2. rho excludes the first 30% of observed epochs; undefined rho contributes the neutral value zero only to this utility, not to scientific Spearman.", "coupling", ["surface_regret_component", "global_surface_spearman"]),
            "wisdom_hpo_score": ("WISDOM HPO score W", "W=0.35*G+0.45*S+0.20*C. G selects checkpoints without local targets; the architecture-level HPO score also measures localization. Associations with components are structural, not new discoveries.", "", ["wisdom_hpo_global", "wisdom_hpo_surface", "wisdom_hpo_coupling"]),
        }
        for name, (label, description, suffix, parents) in definitions.items():
            self._metric(f"val_{name}", description, "validation/selection" + (f"/{suffix}" if suffix else ""),
                         "max", (0, 1), 100, label=label, split="validation",
                         role="selection", derived_from=[f"val_{parent}" for parent in parents],
                         aggregation="terminal", notes="Terminal curve summary, not the last plateau epoch. Missing local or rank support remains unavailable.")
        self.metrics["val_wisdom_hpo_global"]["derived_from"] = ["val_auprc", "val_auroc"]
        self.metrics["val_wisdom_hpo_surface"]["derived_from"] = ["val_surface_positive_macro_auprc"]
        for name, transformation in {
            "global":   "0.70*val_auprc+0.30*val_auroc",
            "surface":  "identity at the globally selected epoch",
            "coupling": "0.70*regret_component+0.30*((defined(rho)?rho:0)+1)/2",
            "score":    "0.35*G+0.45*S+0.20*C",
        }.items():
            self.metrics[f"val_wisdom_hpo_{name}"]["transformation"] = transformation
        for name, canonical in {
            "protein_global_score": "wisdom_hpo_global",
            "surface_selected_score": "wisdom_hpo_surface",
            "surface_coupling_score": "wisdom_hpo_coupling",
        }.items():
            self._metric(f"val_{name}", f"Identity duplicate of val_{canonical}, retained for historical raw evidence.",
                         "validation/selection/aliases", "max", (0, 1), 0, split="validation",
                         derived_from=[f"val_{canonical}"], transformation="identity",
                         role="alias", visibility="hidden", discovery=False)
        self._metric("val_selection_utility", "G=0.70*AUPRC+0.30*AUROC. Epoch values select the checkpoint; the terminal value is the selected G, identical to final wisdom_hpo_global when local evidence exists.",
                     "validation/selection/global", "max", (0, 1), 20, split="validation",
                     derived_from=["val_auprc", "val_auroc"], role="selection",
                     aggregation="selected_epoch", visibility="advanced", discovery=False)
        for name, description, priority in (
            ("global_surface_spearman", "Rank correlation of epoch-wise G and S after discarding the first 30% of observed pairs. Positive agreement is desirable; constant/short trajectories remain undefined.", 95),
            ("global_surface_spearman_all", "Rank correlation of G/S over the entire observed trajectory, including early transient epochs; not cross-candidate or cross-protein correlation.", 45),
        ):
            self._metric(f"val_{name}", description, "validation/coupling", "max", (-1, 1), priority,
                         split="validation", unit="correlation", phase="curve-summary")
        self.metrics["val_global_surface_spearman"].update(
            label="Global-surface Spearman",
            aliases=["global surface correlation", "surface correlation", "G S correlation", "coupling correlation"],
        )
        self._metric("val_surface_selection_regret", "Best S observed over the trajectory minus S at the G-selected epoch. Zero means global selection preserved the best observed localization; incomplete schedules limit the comparison.",
                     "validation/coupling", "min", (0, 1), 95, split="validation", phase="curve-summary",
                     derived_from=["val_surface_best_score", "val_wisdom_hpo_surface"])
        self._metric("val_surface_best_score", "Maximum surface AUPRC among observed epochs; descriptive oracle, never used to select weights.",
                     "validation/coupling", "max", (0, 1), split="validation", phase="curve-summary")
        self._metric("val_surface_regret_component", "1-min(surface_selection_regret/0.20,1); bounded utility transformation of regret, not another independent outcome.",
                     "validation/coupling", "max", (0, 1), split="validation",
                     derived_from=["val_surface_selection_regret"], transformation="1-min(regret/0.20,1)",
                     role="derived", discovery=False, phase="curve-summary")
        for name in ("global_selected_epoch", "surface_best_epoch"):
            self._metric(f"val_{name}", "Observed epoch selected by " + ("G." if name.startswith("global") else "S as a descriptive oracle, not for optimization."),
                         "diagnostics/integrity", priority=0, split="validation", unit="epoch",
                         role="status", visibility="hidden", discovery=False)
        self._metric("val_mcc_defined", "One only when selected-checkpoint scientific MCC has a nonzero denominator. A degenerate predictor does not receive a fabricated MCC.",
                     "diagnostics/integrity", bounds=(0, 1), priority=0, split="validation", role="status",
                     visibility="hidden", discovery=False)
        self._metric("val_mcc_objective", "Defined MCC is transformed to (MCC+1)/2; undefined MCC receives worst utility zero. This is a technical utility, not scientific MCC.",
                     "validation/selection/technical", "max", (0, 1), 0, split="validation",
                     derived_from=["val_mcc", "val_mcc_defined"], transformation="defined?(mcc+1)/2:0",
                     role="derived", visibility="hidden", discovery=False, aggregation="selected_epoch")

    def _faithfulness(self, prefix: str, split: str) -> None:
        """Describe equal-area interventions on the unchanged pooling boundary.

        Args:
            prefix: Persisted split prefix.
            split: Native validation/test partition. No surface target is used by this audit.
        """
        fractions = [0.01, 0.02, 0.05, 0.10, 0.20]
        for intervention in ("deletion", "insertion"):
            members = {}
            for strategy in ("top", "random", "bottom", "top_minus_random"):
                for fraction in fractions:
                    suffix = round(100 * fraction)
                    name = f"{prefix}_faithfulness_{intervention}_{strategy}_{suffix}"
                    meaning = (
                        "baseline probability minus probability after removing the region"
                        if intervention == "deletion" else "positive probability after retaining only the region"
                    )
                    difference = strategy == "top_minus_random"
                    description = f"Positive-protein mean {meaning}, selecting approximately {100*fraction:g}% represented area by {strategy.replace('_', ' ')} ordering; an intervention on the model, not proof of a biological cause."
                    parents = [f"{prefix}_faithfulness_{intervention}_{mode}_{suffix}" for mode in ("top", "random")] if difference else []
                    if difference:
                        description = "Top intervention minus its equal-area deterministic-random control. " + description
                    bounds = (-1, 1) if intervention == "deletion" or difference else (0, 1)
                    self._metric(name, description, f"{split}/faithfulness/{intervention}",
                                 "max" if difference or strategy == "top" else "unknown", bounds,
                                 55 if difference else 35, split=split, derived_from=parents,
                                 role="derived" if difference else "quality", aggregation="selected_epoch",
                                 transformation="top-random" if difference else "pooling intervention")
                    members[name] = {"strategy": strategy, "fraction": fraction}
            self.families[f"{prefix}_faithfulness_{intervention}"] = {
                "label": f"{split.capitalize()} surface {intervention}",
                "description": "Regional diffusion audits are unavailable: removing vertices would invalidate their stored operator.",
                "dimensions": {
                    "strategy": {"kind": "categorical", "values": ["top", "random", "bottom", "top_minus_random"]},
                    "fraction": {"kind": "numeric", "values": fractions},
                },
                "members": members,
            }

    def _subgroups(
        self, prefix: str, split: str,
        global_phenotypes: Sequence[str], local_phenotypes: Sequence[str],
    ) -> None:
        """Describe stratum support and family coordinates without inventing dataset clusters.

        Args:
            prefix: Persisted metric prefix.
            split: Native evidence partition.
            global_phenotypes: Known sanitized global phenotype labels, if supplied.
            local_phenotypes: Known sanitized interface phenotype labels, if supplied.

        Dynamic phenotype strata receive meaningful patterns even without a declared family.
        Native families cannot discover new coordinate values from logged metric names; explicit
        YAML member overrides can supply them once the dataset's actual labels are known.
        """
        axes = {
            "atom_count": ("atom-count", ["q1", "q2", "q3", "q4"]),
            "surface_count": ("surface-count", ["q1", "q2", "q3", "q4"]),
            "surface_prevalence": ("surface-prevalence", ["q1", "q2", "q3", "q4"]),
            "tier": ("difficulty", ["core", "challenge"]),
            "global_phenotype": ("global-phenotype", list(global_phenotypes)),
            "local_phenotype": ("interface-phenotype", list(local_phenotypes)),
        }
        for axis, (category, strata) in axes.items():
            members = {}
            for statistic, meaning in self.SUBGROUP.items():
                count = statistic.endswith("count")
                curve = statistic in {"global_at_selection", "surface_at_global_selection", "surface_regret", "coupling", "wisdom_score"}
                metadata: dict[str, Any] = {
                    "description": meaning + " Phenotypes require >=8 proteins or >=4 usable positives; tied quartile boundaries remain together. Small groups are descriptive, not evidence of equality or biological function.",
                    "category": f"{split}/subgroups/{category}",
                    "split": split, "unit": "count" if count else "fraction",
                    "role": "support" if count else "quality",
                    "direction": "unknown" if count else "min" if statistic == "surface_regret" else "max",
                    "priority": 0 if count else 35,
                    "visibility": "hidden" if count else "normal",
                    "discovery": not count,
                    "aggregation": "terminal" if curve else "selected_epoch",
                }
                if not count:
                    metadata["range"] = [0, 1]
                self.defaults.append({"pattern": f"{prefix}_subgroup_{axis}_*_{statistic}", "metadata": metadata})
                for stratum in strata:
                    name = f"{prefix}_subgroup_{axis}_{stratum}_{statistic}"
                    self.metrics[name] = dict(metadata)
                    self.metrics[name]["label"] = f"{axis.replace('_', ' ')} {stratum}: {statistic.replace('_', ' ')}"
                    members[name] = {"stratum": stratum, "metric": statistic}
            for stratum in strata:
                stem = f"{prefix}_subgroup_{axis}_{stratum}_"
                for statistic, parents, transformation in (
                    ("global_score", ["protein_auprc", "protein_auroc"], "0.70*AUPRC+0.30*AUROC"),
                    ("global_at_selection", ["global_score"], "max(subgroup G history)"),
                    ("surface_at_global_selection", ["global_score", "surface_auprc"], "subgroup S at argmax(subgroup G history)"),
                    ("surface_regret", ["surface_auprc", "surface_at_global_selection"], "max(subgroup S history)-S at subgroup G optimum"),
                    ("coupling", ["global_score", "surface_auprc", "surface_regret"], "0.70*normalized subgroup regret utility+0.30*late subgroup rank utility"),
                    ("wisdom_score", ["global_at_selection", "surface_at_global_selection", "coupling"], "0.35*G+0.45*S+0.20*C"),
                ):
                    self.metrics[f"{stem}{statistic}"].update(
                        derived_from=[f"{stem}{parent}" for parent in parents],
                        transformation=transformation,
                    )
            if members:
                self.families[f"{prefix}_subgroup_{axis}"] = {
                    "label": f"{split.capitalize()} {category}",
                    "dimensions": {
                        "stratum": {"kind": "ordered" if axis.endswith("count") or axis == "surface_prevalence" else "categorical", "values": strata},
                        "metric": {"kind": "categorical", "values": list(self.SUBGROUP)},
                    },
                    "members": members,
                }

    def _views(self, prefix: str, split: str) -> None:
        """Catalog optional physically matched views; agreement alone is not correctness.

        Args:
            prefix: Stable split prefix.
            split: Native evidence partition.
        """
        definitions = {
            "view_logit_spearman": ("Rank agreement of local logits on matched physical points.", "max", (-1, 1)),
            "view_probability_pearson": ("Linear agreement of sigmoid probabilities on matched points.", "max", (-1, 1)),
            "view_map_js_divergence": ("Jensen-Shannon divergence of probability-normalized maps, divided by log(2).", "min", (0, 1)),
            "view_map_l1": ("L1 distance between probability-normalized maps; twice their total-variation distance.", "min", (0, 2)),
            "view_map_total_variation": ("Half the L1 distance between probability-normalized maps.", "min", (0, 1)),
            "view_top_05_overlap": ("Intersection fraction of the highest-scored 5% matched points.", "max", (0, 1)),
            "view_top_10_overlap": ("Intersection fraction of the highest-scored 10% matched points.", "max", (0, 1)),
            "view_regional_probability_variance": ("Mean matched-point population variance across the two view probabilities; not a spatially diffused statistic despite its historical name.", "min", (0, 0.25)),
            "view_surface_consistency": ("Mean bounded agreement forms of divergence, variation, variance, overlaps, and defined correlations; consistency is not localization accuracy.", "max", (0, 1)),
        }
        for name, (description, direction, bounds) in definitions.items():
            self._metric(f"{prefix}_{name}", description, f"{split}/view-consistency", direction,
                         bounds, split=split, aggregation="terminal")
        self.metrics[f"{prefix}_view_map_l1"].update(
            derived_from=[f"{prefix}_view_map_total_variation"], transformation="2*total_variation", discovery=False)
        self.metrics[f"{prefix}_view_surface_consistency"]["derived_from"] = [
            f"{prefix}_{name}" for name in definitions if name not in {"view_surface_consistency", "view_map_l1"}
        ]
        self._metric(f"{prefix}_view_matched_points", "Number of physical point correspondences supporting view comparisons.",
                     "diagnostics/counts", priority=0, split=split, unit="count", role="support",
                     visibility="hidden", discovery=False)
        self.families[f"{prefix}_view_consistency_components"] = {
            "label": f"{split.capitalize()} view agreement components",
            "dimensions": {"component": {"kind": "categorical", "values": list(definitions)}},
            "members": {f"{prefix}_{name}": {"component": name} for name in definitions},
        }

    def _operational(self) -> None:
        """Separate costs, optimization, schedules, capacity, and integrity from outcomes."""
        for name, description in {
            "loss": "Total minibatch objective, including weighted weak terms and gate regularization.",
            "total_loss": "Identity duplicate of train_loss, retained for compatibility.",
            "task_loss": "Protein binary cross entropy for the active main prediction head.",
            "direct_task_loss": "Protein binary cross entropy of the direct global head.",
            "surface_task_loss": "Protein binary cross entropy of the surface-derived global head; no local ground truth is optimized.",
            "negative_surface_loss": "Weak negative-bag penalty from protein labels, never from surface targets.",
            "gate_regularization": "Mean expected semantic gate activity before multiplying by effective_gate_lambda.",
        }.items():
            self._metric(f"train_{name}", description, "training/loss", "min", priority=15,
                         split="train", role="optimization", aggregation="latest", visibility="advanced",
                         discovery=False)
        self.metrics["train_total_loss"].update(derived_from=["train_loss"], transformation="identity", role="alias")
        for pattern, category, description in (
            ("train_weak_surface_*", "training/loss", "Epoch mean weakly supervised surface penalty from protein-level labels; not a local-GT loss."),
            ("train_gradient_*", "training/gradients", "Arithmetic epoch mean of a pre-clipping component gradient L2 norm; neither larger nor smaller is universally preferable."),
            ("train_activation_*", "training/optimization", "Arithmetic epoch mean activation distribution summary; a stability diagnostic, not validation quality."),
            ("train_surface_logit_*", "training/optimization", "Local evidence-logit distribution summary over training batches; not prediction quality or calibration evidence by itself."),
            ("train_sampled_gate_*", "training/gating", "Arithmetic epoch mean fraction of sampled gates at a clipping boundary; a saturation diagnostic."),
            ("pooling_*", "model/pooling", "Epoch trajectory of the active fixed, learned, or scheduled pooling scalar or scale-bank weight; not an outcome metric."),
            ("surface_refiner_*", "model/refinement", "Learned heat length or last evaluated batch's learned conductance mean/std; not epoch-averaged conductance or validation quality."),
            ("final_surface_refiner_*", "model/refinement", "Refiner length restored from the global-selected checkpoint, or last evaluated batch's conductance moments; not a quality metric."),
            ("final_pooling_*", "model/pooling", "Active pooling parameter restored from the globally selected checkpoint; absent for pruned Runs."),
        ):
            self.defaults.append({"pattern": pattern, "metadata": {
                "description": description, "category": category, "role": "control" if "pooling" in pattern else "diagnostic",
                "direction": "unknown", "visibility": "hidden", "discovery": False,
                "aggregation": "terminal" if pattern.startswith("final_") else "latest",
            }})
        for name, description, unit in (
            ("learning_rate", "Effective AdamW learning rate after any warm-up schedule.", "dimensionless"),
            ("effective_gate_lambda", "Effective normalized L0 pressure after gate warm-up/ramp.", "dimensionless"),
            ("initial_local_head_bias", "Initial local-head bias after optional train-label-only calibration.", "logit"),
        ):
            self._metric(name, description, "training/optimization", priority=0, unit=unit,
                         role="control", visibility="hidden", discovery=False, aggregation="latest")
        for name in ("gate_expected_active_fraction", "gate_deterministic_active_fraction"):
            self._metric(f"train_{name}", "Expected L0 activity fraction." if "expected" in name else "Fraction of semantic gates with a deterministic clipped value greater than zero.",
                         "model/gating", bounds=(0, 1), priority=20, split="train", role="diagnostic",
                         visibility="advanced", aggregation="latest")
        for name in ("parameter_count", "gate_parameter_count", "semantic_gate_count",
                     "evidence_head_parameter_count", "evidence_head_added_parameter_count", "evidence_input_width"):
            self._metric(name, "Static capacity: " + name.replace("_", " ") + "; may vary with HPO architecture, not an accuracy measure.",
                         "model/capacity", priority=10, role="model-size", unit="count", visibility="hidden",
                         discovery=name == "parameter_count")
        self._metric("preprocessing_bytes", "Sum of base structural archive sizes consumed by the training population; input storage, not GPU memory.",
                     "runtime/data", "min", priority=5, role="resource", unit="bytes", visibility="hidden")
        for name in ("maximum_surface_points", "maximum_atoms", "maximum_active_atomic_edges",
                     "maximum_atomic_degree", "mean_atomic_degree", "mean_atoms_per_batch",
                     "mean_surface_points_per_batch", "active_atom_spatial_k", "active_surface_atom_k",
                     "mean_spectral_modes"):
            echo = name.startswith("active_")
            self._metric(f"train_{name}", "First-epoch representation profile: " + name.replace("_", " ") + "; batch composition and authored topology can change it.",
                         "model/architecture", priority=0, split="train", role="parameter-echo" if echo else "diagnostic",
                         unit="count", aggregation="latest", visibility="hidden", discovery=False)
        for name in ("patience_used", "patience_remaining", "surface_metrics_computed", "averaged_validation_weights"):
            self._metric(f"val_{name}", {
                "patience_used": "Consecutive epochs without improvement exceeding minimum_delta in G.",
                "patience_remaining": "Remaining global-validation patience, not an HPO pruning probability.",
                "surface_metrics_computed": "One when local validation was scheduled for this epoch, otherwise zero.",
                "averaged_validation_weights": "One when EMA/SWA weights were used for this epoch's validation.",
            }[name], "diagnostics/integrity", priority=0, split="validation", role="status",
                         aggregation="latest", visibility="hidden", discovery=False)

        # Pure training throughput and end-to-end epoch throughput are NOT aliases. Split
        # prefixing produces train_train_seconds and train_train_proteins_per_second.

        resources = {
            "resource.duration_seconds": ("Framework median duration across comparable screening Runs, not the final epoch duration.", "seconds", "min", "time"),
            "resource.gpu_seconds": ("Framework median allocated GPU-seconds per comparable screening Run; not a CUDA utilization measurement.", "seconds", "min", "time"),
            "resource.cpu_seconds": ("Framework median CPU cost per comparable screening Run as recorded by its resource evidence.", "seconds", "min", "time"),
            "resource.peak_vram": ("Framework median whole-Run peak VRAM across comparable screening Runs; measurement provenance remains in framework resource evidence.", "bytes", "min", "gpu-memory"),
            "resource.peak_ram": ("Framework median whole-Run peak host RAM across comparable screening Runs.", "bytes", "min", "host-memory"),
            "train_epoch_seconds": ("End-to-end epoch wall time including validation and reporting.", "seconds", "min", "time"),
            "train_train_seconds": ("Training-loop wall time, including data wait, forward, backward, and optimizer work; excludes validation.", "seconds", "min", "time"),
            "train_train_proteins_per_second": ("Training examples divided by training-loop wall time; excludes validation.", "proteins/s", "max", "throughput"),
            "train_proteins_per_second": ("Training examples divided by end-to-end epoch wall time; includes validation/reporting overhead.", "proteins/s", "max", "throughput"),
            "train_data_wait_seconds": ("Time spent waiting for the next training DataLoader batch during this epoch.", "seconds", "min", "data"),
            "val_validation_seconds": ("Whole per-epoch validation wall time, including metric computation.", "seconds", "min", "time"),
            "train_cuda_allocated_gib": ("PyTorch live allocated device memory sampled at epoch end, not whole-process physical VRAM.", "GiB", "min", "gpu-memory"),
            "train_cuda_reserved_gib": ("PyTorch caching-allocator reserved device memory sampled at epoch end.", "GiB", "min", "gpu-memory"),
            "train_cuda_peak_gib": ("PyTorch peak allocated device memory for this epoch; terminal fallback is the last recorded epoch, not a whole-Run maximum.", "GiB", "min", "gpu-memory"),
        }
        for name, (description, unit, direction, category) in resources.items():
            self._metric(name, description, f"runtime/{category}", direction, priority=15,
                         unit=unit, role="resource", aggregation="latest", visibility="advanced")
        self.metrics["train_cuda_peak_gib"].update(label="Epoch peak GPU memory", aliases=["gpu memory", "peak vram"])
        for name in resources:
            if name.startswith("resource."):
                self.metrics[name]["aggregation"] = "terminal"
        for prefix, split in (("val", "validation"), ("test", "test")):
            for suffix, description, unit, direction in (
                ("forward_seconds", "Model forward-only time over the evaluated split; excludes DataLoader, transfer, metrics, and audits.", "seconds", "min"),
                ("forward_proteins_per_second", "Evaluated proteins per forward-only second; not end-to-end pipeline throughput.", "proteins/s", "max"),
                ("forward_milliseconds_per_protein", "Forward-only milliseconds per evaluated protein.", "milliseconds/protein", "min"),
            ):
                self._metric(f"{prefix}_{suffix}", description, "runtime/inference", direction, priority=15,
                             unit=unit, role="resource", split=split, aggregation="selected_epoch",
                             visibility="advanced")
        members = {}
        for stage in ("initial", "final"):
            for statistic in ("minimum", "mean", "median", "maximum"):
                name = f"{stage}_diffusion_time_{statistic}"
                self._metric(name, f"{statistic.capitalize()} learned heat time across every DiffusionNet block/channel at the {stage} state; larger is not universally better. Final state is the restored checkpoint except for pruned Runs.",
                             "model/diffusion", priority=25, unit="angstrom^2", role="diagnostic")
                members[name] = {"stage": stage, "statistic": statistic}
        self.families["diffusion_time_summary"] = {
            "label": "Learned surface diffusion times",
            "dimensions": {
                "stage": {"kind": "ordered", "values": ["initial", "final"]},
                "statistic": {"kind": "categorical", "values": ["minimum", "mean", "median", "maximum"]},
            }, "members": members,
        }

    def _questions(self) -> None:
        """Focus framework exploration on observational WISDOM questions, not all pairs.

        Questions are optional because short curves, sealed test, inactive attention, unsupported
        regional interventions, or absent physical views must remain unavailable, not failures.
        Category summaries organize domain groups; parameter screens inherit LambdaForge's
        active authored ParameterSpace. They never independently evaluate ``when`` conditions.
        No special top-region restriction or support threshold is invented: those knobs are
        absent from the current public profile schema. Native question priorities remain at
        their default unless explicitly overridden; they do not change scientific selection.
        """
        core = ["val_wisdom_hpo_global", "val_wisdom_hpo_surface", "val_wisdom_hpo_coupling",
                "val_wisdom_hpo_score", "val_global_surface_spearman", "val_surface_selection_regret"]
        self.questions.extend([
            {"id": "global_surface_tradeoff", "kind": "tradeoff", "metrics": core[:2], "optional": True},
            {"id": "global_surface_alignment", "kind": "relationship", "x": core[0], "y": core[1], "expected": "positive", "optional": True},
            {"id": "coupling_health", "kind": "relationship", "x": core[3], "y": core[4], "expected": "positive", "optional": True},
            {"id": "surface_regret_health", "kind": "relationship", "x": core[3], "y": core[5], "expected": "negative", "optional": True},
            {"id": "hpo_decomposition", "kind": "category_summary", "category": "validation/selection", "optional": True},
            {"id": "hpo_parameters_core_quality", "kind": "parameter_screen", "metrics": core, "optional": True},
            {"id": "quality_resource_tradeoff", "kind": "tradeoff", "metrics": [core[3], "parameter_count", "train_epoch_seconds", "train_cuda_peak_gib"], "optional": True},
            {"id": "quality_whole_run_cost", "kind": "tradeoff", "metrics": [core[3], "resource.gpu_seconds", "resource.duration_seconds", "resource.peak_vram"], "optional": True},
            {"id": "surface_quality_hardware", "kind": "tradeoff", "metrics": ["val_surface_positive_macro_auprc", "train_cuda_peak_gib", "train_train_proteins_per_second"], "optional": True},
            {"id": "localization_negative_control", "kind": "tradeoff", "metrics": ["val_surface_positive_macro_auprc", "val_surface_negative_positive_mass", "val_surface_negative_peak"], "optional": True},
            {"id": "attention_local_evidence", "kind": "relationship", "x": "val_surface_attention_positive_macro_auprc", "y": "val_surface_positive_macro_auprc", "expected": "positive", "optional": True},
            {"id": "surface_view_robustness", "kind": "relationship", "x": "val_view_surface_consistency", "y": "val_surface_positive_macro_auprc", "optional": True},
            {"id": "subgroup_robustness", "kind": "category_summary", "category": "validation/subgroups", "optional": True},
        ])
        for intervention in ("deletion", "insertion"):
            self.questions.append({"id": f"surface_faithfulness_{intervention}", "kind": "metric_family",
                                   "family": f"val_faithfulness_{intervention}", "optional": True})
            self.questions.append({"id": f"localization_faithfulness_{intervention}", "kind": "relationship",
                                   "x": "val_surface_positive_macro_auprc",
                                   "y": f"val_faithfulness_{intervention}_top_minus_random_10", "optional": True})
            for fraction in (1, 2, 5, 10, 20):
                self.questions.append({"id": f"faithfulness_{intervention}_control_{fraction}", "kind": "relationship",
                                       "x": f"val_faithfulness_{intervention}_top_{fraction}",
                                       "y": f"val_faithfulness_{intervention}_random_{fraction}", "optional": True})
        for axis in ("atom_count", "surface_count", "surface_prevalence", "tier"):
            self.questions.append({"id": f"subgroup_{axis}", "kind": "metric_family",
                                   "family": f"val_subgroup_{axis}", "optional": True})
        for category in ("global", "surface", "coupling", "faithfulness", "view-consistency"):
            self.questions.append({"id": f"{category}_quality", "kind": "category_summary",
                                   "category": f"validation/{category}", "optional": True})
        for name in ("pooling_type", "topk_fraction", "attention_hidden_dim", "attention_variant",
                     "regional_diffusion_scale", "regional_diffusion_scale_init", "log_sum_exp_beta",
                     "log_sum_exp_beta_init", "pooling_area_mode", "pooling_curriculum_end_beta",
                     "autopool_alpha", "autopool_alpha_init", "gem_power", "gem_power_init",
                     "max_mean_lambda", "max_mean_lambda_init"):
            self.questions.append({"id": f"pooling_{name}", "kind": "parameter_screen",
                                   "parameters": [name], "metrics": [*core,
                                       "val_surface_attention_positive_macro_auprc",
                                       "val_surface_attention_logit_spearman",
                                       "val_surface_attention_entropy", "train_cuda_peak_gib",
                                       "train_train_proteins_per_second"], "optional": True})
