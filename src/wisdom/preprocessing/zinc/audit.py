"""Describe Zn diversity and shortcut risks without changing labels or independent groups."""

import csv
import json
import numpy as np
import matplotlib.pyplot as plt

from typing import Any
from pathlib import Path
from collections.abc import Mapping, Sequence
from collections import Counter, defaultdict


def write_diversity_report(
    root: Path, raw: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write interpretable class/source/chemistry coverage and covariate diagnostics.

    Args:
        root: Managed Selection output directory receiving JSON, SVG and Markdown.
        raw: Full similarity population, including scientific exclusions.
        selected: Canonical records with fixed splits and physical descriptors; imbalance is valid.

    Returns:
        JSON-compatible coverage, supported standardized mean differences and warnings.
        No matching, relabelling, resplitting, significance testing or model fitting occurs.

    Raises:
        OSError: Unwritable report destination.
        ValueError: Malformed scientific scalar or missing mandatory row field.
    """
    groups = Counter(str(row["leakage_group"]) for row in raw)
    retained = Counter(str(row["leakage_group"]) for row in selected)
    coverage: dict[str, Any] = {}
    warnings: list[str] = []

    # Sequence/structure groups measure independence, not biological families. External family
    # annotations are counted only when evidence supplies them; unknown is not a new family.

    for field in ("origin", "method", "global_phenotype", "local_phenotype",
                  "positive_confidence", "negative_confidence", "family", "pfam", "interpro",
                  "cath", "ec"):
        distribution: dict[str, Any] = {}
        for label in (0, 1):
            values = [row.get(field) for row in selected if int(row["label"]) == label]
            counts = Counter(token for value in values for token in _tokens(value))
            distribution[str(label)] = {"counts": dict(sorted(counts.items())),
                                        "missing": sum(not _tokens(value) for value in values)}
        coverage[field] = distribution
    if all(coverage["family"][str(label)]["missing"] == sum(
        int(row["label"]) == label for row in selected) for label in (0, 1)):
        warnings.append("No functional-family annotations: leakage groups are not families.")

    covariates: dict[str, Any] = {}
    for field in ("sequence_length", "resolution", "radius_of_gyration_normalized",
                  "aspect_ratio", "packing_density", "charge_density"):
        arrays = [np.asarray([float(row[field]) for row in selected
                              if int(row["label"]) == label and row.get(field) is not None
                              and np.isfinite(float(row[field]))]) for label in (0, 1)]
        entry: dict[str, Any] = {"support": [len(values) for values in arrays]}
        entry["class_summaries"] = [{"mean": float(values.mean()),
                                     "median": float(np.median(values)),
                                     "min": float(values.min()), "max": float(values.max())}
                                    if len(values) else None for values in arrays]
        smd = None
        if all(len(values) >= 2 for values in arrays):
            deviation = float(np.sqrt(sum(values.var(ddof=1) for values in arrays) / 2))
            if deviation > 0:
                smd = float((arrays[1].mean() - arrays[0].mean()) / deviation)
        entry["smd"] = smd
        covariates[field] = entry
        if smd is not None and abs(smd) >= 0.5:
            warnings.append(f"{field}: |SMD|={abs(smd):.2f}; potential class shortcut, not proof.")

    chemistry = Counter(("/".join(f"{name}{count}" for name, count in
                        sorted(site["selected_residue_counts"].items()))
                        if "selected_residue_counts" in site else
                        "types_only:" + "/".join(site["selected_residue_names"]))
                        for row in selected if int(row["label"]) == 1
                        for site in row["sites"] if site["accepted"])
    rare = sorted(name for name, count in chemistry.items() if count < 5)
    if rare:
        warnings.append(f"{len(rare)} coordination residue classes have fewer than five sites.")
    result = {"raw_leakage_groups": len(groups), "selected_leakage_groups": len(retained),
              "raw_group_sizes": dict(sorted(groups.items())),
              "selected_group_sizes": dict(sorted(retained.items())),
              "selected_singleton_groups": sum(count == 1 for count in retained.values()),
              "coverage": coverage, "covariates": covariates,
              "site_residue_classes": dict(sorted(chemistry.items())), "warnings": warnings}
    result["site_topology"] = dict(Counter(
        "interchain" if site["interchain"] else "intrachain"
        for row in selected for site in row["sites"] if site["accepted"]))
    result["surface_evaluability"] = dict(Counter(
        str(row.get("surface_evaluability", "historical_unavailable")) for row in selected))
    (root / "diversity.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    # One small figure makes imbalance visible. It is not a clustering embedding or evidence
    # of independence: the transitive grouping and split audit establish those constraints.

    figure, axes = plt.subplots(1, 2, figsize=(10, 4))
    splits = ("train", "validation", "test")
    positions = np.arange(3)
    for label, offset in ((0, -0.18), (1, 0.18)):
        split_counts = [sum(row["split"] == split and int(row["label"]) == label
                      for row in selected) for split in splits]
        axes[0].bar(positions + offset, split_counts, width=0.36,
                    label="Explicit negative" if label == 0 else "Verified coordination")
    axes[0].set_xticks(positions, splits)
    axes[0].set_ylabel("Proteins")
    axes[0].legend(fontsize=8)
    axes[1].plot(np.arange(1, len(groups) + 1), sorted(groups.values(), reverse=True))
    axes[1].set_xlabel("RAW group rank, largest first")
    axes[1].set_ylabel("Proteins per leakage group")
    figure.tight_layout()
    figure.savefig(root / "diversity.svg")
    plt.close(figure)
    (root / "diversity.md").write_text(
        "# Zn diversity and shortcut audit\n\n"
        "![Split counts and RAW group tail](diversity.svg)\n\n"
        "The left panel compares protein counts, not independent sample counts. "
        "Large imbalance merits review but groups must never be broken to hide it. "
        "The right panel ranks full-RAW leakage groups: a long high tail indicates repeated "
        "sequence/structure evidence. These groups are not functional families.\n\n"
        "## Covariates\n\nSMD=(positive mean-negative mean)/sqrt((positive variance+negative "
        "variance)/2), using sample variances. Zero means similar means, not equal distributions. "
        "Positive values mean larger positives; |SMD|>=0.5 is a review flag, not a significance "
        "test or a biological rule. Fewer than two values per class or zero pooled variance "
        "makes SMD unavailable; it is never replaced by zero. Missing support remains explicit.\n\n"
        "|Variable|Negative/positive support|SMD|\n|---|---|---:|\n"
        + "".join(f"|{name}|{entry['support']}|"
                  + (f"{entry['smd']:.3f}" if entry["smd"] is not None else "unavailable")
                  + "|\n" for name, entry in covariates.items())
        + "\n## Coverage and warnings\n\nThe JSON records sources, experimental methods, "
        "physical phenotypes, confidence, optional functional annotations, and residue classes "
        "of accepted sites. Rare chemistry is reported, never removed because it is rare. "
        "Source-confounded labels remain a limitation even when class counts match. "
        "No hypothesis test treating homologous proteins as independent is reported.\n\n"
        + "\n".join(f"- {warning}" for warning in warnings) + "\n", encoding="utf-8",
    )
    return result


def _tokens(value: Any) -> list[str]:
    """Normalize scalar or multi-domain descriptive annotations without inventing a family.

    Args:
        value: Optional scalar annotation or a list of independently assigned domain IDs.

    Returns:
        Distinct nonmissing strings; a multidomain protein contributes once to each domain.
    """
    values = value if isinstance(value, (tuple, list)) else [value]
    return sorted({str(item) for item in values if item not in (None, "", "unknown")})


def write_benchmark_report(
    root: Path, raw: Sequence[Mapping[str, Any]], selected: Sequence[Mapping[str, Any]],
    audit: dict[str, Any], acquisition: Mapping[str, Any], negatives: Mapping[str, Any],
) -> None:
    """Reconcile observable acquisition, quality eligibility, diversity and fixed partitions.

    Args:
        root: Managed design directory receiving report.md, coverage.csv and benchmark.json.
        raw: Complete identity-deduplicated RAW population after transitive leakage grouping.
        selected: Canonical members with fixed train/validation/test assignment.
        audit: Scientific audit, updated with this report's machine-readable benchmark summary.
        acquisition: Frozen Discovery entry/chain/copy counts; absent for historical designs.
        negatives: Frozen experimental-denial acquisition stages and exclusion reasons.

    Returns:
        None; writes explanatory Markdown, exact counts and linked figures. No label, split,
        phenotype or model selection is changed. Retention is computed only within equal units.

    Raises:
        OSError: The report cannot be written.
        KeyError: Mandatory scientific evidence/group fields are missing.
    """
    eligible = [row for row in raw if row["quality_eligible"]]
    funnels: dict[str, list[dict[str, Any]]] = {}
    exclusions: dict[str, Any] = {}
    for label, name in ((1, "positive"), (0, "negative")):
        population = [row for row in raw if int(row["label"]) == label]
        accepted = [row for row in eligible if int(row["label"]) == label]
        retained = [row for row in selected if int(row["label"]) == label]
        steps: list[tuple[str, int | None, str]] = []
        if label:
            steps.extend([(key, acquisition.get(key), unit) for key, unit in (
                ("rcsb_entries_containing_Zn", "entries"), ("inspected_entries", "entries"),
                ("assembly_copies", "chain_copies"), ("candidate_chains", "chains"))])
        else:
            steps.extend([(key, negatives.get(key), unit) for key, unit in (
                ("queried_annotations", "annotations"), ("accepted_annotations", "annotations"),
                ("experimental_subjects", "subjects"))])
            steps.extend((key, value, "subjects") for key, value in
                         negatives.get("subject_funnel", {}).items()
                         if key != "canonical_experimental_subjects")
        steps.append(("deduplicated_RAW", len(population), "chains"))
        if label:
            steps.append(("verified_coordination_chains", sum(bool(row["zn_site_count"])
                          for row in population), "chains"))
        steps.extend([("quality_eligible", len(accepted), "chains"),
                      ("unique_eligible_sequences", len({row["sequence_sha256"]
                                                       for row in accepted}), "sequences"),
                      ("eligible_leakage_groups", len({row["leakage_group"]
                                                      for row in accepted}), "groups"),
                      ("selected_members", len(retained), "chains")])
        records: list[dict[str, Any]] = []
        previous_count, previous_unit = None, None
        for stage, count, unit in steps:
            records.append({"stage": stage, "count": count, "unit": unit,
                "retained_percent": (100 * count / previous_count if count is not None
                    and previous_count and previous_unit == unit else None)})
            previous_count, previous_unit = count, unit
        funnels[name] = records
        primary = Counter(row["quality_exclusion_reason"].split(";")[0] for row in population
                          if not row["quality_eligible"])
        reasons = Counter(reason for row in population for reason in
                          row["quality_exclusion_reason"].split(";") if reason)
        exclusions[name] = {"raw": len(population), "eligible": len(accepted),
            "excluded": len(population) - len(accepted), "primary_reasons": dict(primary),
            "all_reasons_nonadditive": dict(reasons),
            "quota_omitted": len(accepted) - len(retained)}

    # Functional categories may overlap. Count each real annotation independently; absence is
    # measured as missing support, never as a synthetic family or an edge between proteins.

    distributions: dict[str, Any] = {}
    split_distributions: dict[str, Any] = {}
    families: dict[str, Any] = {}
    for field in ("origin", "method", "release_year", "global_phenotype", "local_phenotype",
                  "coordination_signature", "pfam", "interpro", "cath", "ec", "taxonomy",
                  "negative_evidence_tier", "surface_evaluability"):
        phases = {}
        for phase, rows in (("eligible", eligible), ("selected", selected)):
            phases[phase] = {str(label): {"count": sum(int(row["label"]) == label for row in rows),
                "missing": sum(not _tokens(row.get(field)) for row in rows
                               if int(row["label"]) == label),
                "categories": dict(Counter(token for row in rows if int(row["label"]) == label
                                            for token in _tokens(row.get(field))))}
                            for label in (0, 1)}
        distributions[field] = phases
        split_distributions[field] = {}
        for split in ("train", "validation", "test"):
            split_distributions[field][split] = {str(label): {
                "members": sum(row["split"] == split and int(row["label"]) == label
                               for row in selected),
                "missing": sum(not _tokens(row.get(field)) for row in selected
                               if row["split"] == split and int(row["label"]) == label),
                "categories": dict(Counter(token for row in selected
                    if row["split"] == split and int(row["label"]) == label
                    for token in _tokens(row.get(field))))} for label in (0, 1)}
        if field not in ("pfam", "interpro", "cath", "ec", "taxonomy"):
            continue

        # Index overlapping domains once instead of scanning every protein for every category.
        # Protein counts and independent leakage-group counts remain separate in each split.

        eligible_members: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        selected_members: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
        for rows, index in ((eligible, eligible_members), (selected, selected_members)):
            for row in rows:
                for token in _tokens(row.get(field)):
                    index[token].append(row)
        entries = {}
        for token, candidates in sorted(eligible_members.items()):
            members = selected_members[token]
            groups = {row["leakage_group"] for row in members}
            eligible_groups = {row["leakage_group"] for row in candidates}
            splits = Counter(row["split"] for row in members)
            entries[token] = {"eligible_groups": len(eligible_groups),
                "selected_groups": len(groups), "selected_members": len(members),
                "groups_by_split": {split: len({row["leakage_group"] for row in members
                    if row["split"] == split}) for split in ("train", "validation", "test")},
                "splits": dict(splits), "single_independent_group": len(eligible_groups) == 1,
                "train_only_observed": bool(members) and set(splits) == {"train"},
                "heldout_only_observed": bool(members) and "train" not in splits,
                "train_and_heldout": "train" in splits and bool(set(splits) - {"train"})}
        families[field] = entries

    # Sensitivity uses the same RAW assemblies and occupancy policy. Site IDs are not aggregated
    # destructively and alternative thresholds never generate labels or another split assignment.

    sensitivity: dict[tuple[float, int, int], dict[str, Any]] = {}
    for row in raw:
        if not int(row["label"]):
            continue
        baseline = {site["site_id"] for site in row["sites"] if site["accepted"]}
        for policy in row.get("threshold_sensitivity", []):
            key = (policy["cutoff"], policy["minimum_donors"], policy["minimum_residues"])
            record = sensitivity.setdefault(key, {"cutoff_angstrom": key[0],
                "minimum_donors": key[1], "minimum_residues": key[2],
                "surviving_sites": 0, "surviving_chains": 0, "lost_baseline_sites": 0,
                "new_sites": 0})
            surviving_ids = set(policy["site_ids"])
            record["surviving_sites"] += len(surviving_ids)
            record["surviving_chains"] += bool(surviving_ids)
            record["lost_baseline_sites"] += len(baseline - surviving_ids)
            record["new_sites"] += len(surviving_ids - baseline)
    sites = [site for row in selected for site in row["sites"] if site["accepted"]]
    chemistry_support = Counter((
        "/".join(f"{residue}{count}" for residue, count in
                 sorted(site["selected_residue_counts"].items())),
        site.get("surface_support", {}).get("classification", "unavailable"),
        "interchain" if site["interchain"] else "intrachain",
    ) for site in sites)
    summary = {"funnels": funnels, "exclusions": exclusions,
        "negative_annotation_exclusions": negatives.get("exclusion_counts", {}),
        "distributions": distributions, "split_distributions": split_distributions,
        "family_independence": families,
        "threshold_sensitivity": list(sensitivity.values()),
        "positive_site_confidence": dict(Counter(
            site.get("confidence_tier", "historical_unavailable") for site in sites)),
        "site_support": dict(Counter(site.get("surface_support", {}).get("classification",
                                     "unavailable") for site in sites)),
        "site_donor_counts": dict(Counter(site["selected_donor_count"] for site in sites)),
        "site_residue_counts": dict(Counter(len(site["selected_residues"]) for site in sites)),
        "site_chemistry_support": [{"residue_signature": key[0], "surface_support": key[1],
            "topology": key[2], "sites": count}
            for key, count in sorted(chemistry_support.items())],
        "acquisition_bias": {"descriptive_only": True,
            "fields": ["origin", "method", "release_year", "resolution"],
            "interpretation": "Source/class associations are acquisition effects, "
                              "not biological specificity."},
        "physiological_specificity_claimed": False}
    audit["benchmark"] = summary
    (root / "benchmark.json").write_text(json.dumps(summary, indent=2))
    with (root / "coverage.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("class", "stage", "count", "unit", "retained_percent"))
        for name, stages in funnels.items():
            for stage_record in stages:
                writer.writerow((name, *stage_record.values()))

    lines = ["# Zn benchmark construction report", "",
        "## 1. What labels mean", "",
        "Positive means occupied Zn coordinated by the selected deposited protein copy under "
        "the stated distance/donor/residue policy. This structural observation is a proxy; it "
        "does not prove physiological binding capability or Zn specificity. Negatives require "
        "referenced experimental non-binding evidence, never absence of a metal or annotation. "
        "The public target remains `zinc_binding` for compatibility.", "",
        "Acceptance policy: `" + json.dumps(selected[0].get("coordination_parameters", {}),
                                            sort_keys=True) + "`. Cutoff is in Å; occupancy is "
        "the deposited fraction and donor/residue minima apply to the selected copy.", "",
        "## 2. Observable universe and coverage", "",
        "Discovery freezes public responses and coordinate bytes; Selection is offline. Counts "
        "refer to this observed source snapshot, not all real Zn-binding proteins. An entry is a "
        "deposition; a chain is a polymer identity; an assembly copy is a transformed chain. "
        "Annotations, subjects, sequences and leakage groups are different units. A retention "
        "percentage is unavailable across units, rather than a misleading ratio.", "",
        f"Snapshot: `{audit['discovery']['snapshot_id']}`; pilot: "
        f"`{audit['discovery']['pilot']}`. A pilot is capped, not full source coverage.", ""]
    for name, stages in funnels.items():
        lines += [f"### 2.{'1' if name == 'positive' else '2'}. {name.title()} funnel", "",
                  "|Stage|Count|Unit|Retained from preceding equal-unit stage|",
                  "|---|---:|---|---:|"]
        for stage_record in stages:
            percent = stage_record["retained_percent"]
            lines.append(f"|{stage_record['stage']}|{stage_record['count']}|{stage_record['unit']}|"
                         + (f"{percent:.1f}%" if percent is not None else "not comparable") + "|")
        excluded = exclusions[name]
        lines += ["", f"Reconciliation: {excluded['raw']} RAW chains = "
            f"{excluded['eligible']} eligible + {excluded['excluded']} excluded. "
            f"Quota omitted {excluded['quota_omitted']} eligible chains. "
            "Each excluded chain has one primary reason; additional reasons can overlap.", "",
            f"Primary exclusions: `{json.dumps(excluded['primary_reasons'], sort_keys=True)}`.", ""]
    lines += ["## 3. Evidence quality, site chemistry and threshold sensitivity", "",
        "NOT zinc binding directly addresses Zn; NOT metal/cation/ion binding denies a broader "
        "activity and is reported separately. Only experimental, referenced, whole-product "
        "canonical annotations and exact full-sequence structural mappings are used. Tags, "
        "truncations, unsupported isoforms, contextual annotations and contradictory evidence "
        "are excluded. Scarcity is a result, not permission to invent negatives.", "",
        "`structural_basic` satisfies the configured acceptance policy. `structural_strong` "
        "additionally requires at least four selected donors, three selected residues, all local "
        "donor/metal occupancies >=0.8 and maximum local distance <=2.8 Å. These are transparent "
        "review tiers, not calibrated confidence probabilities. `externally_supported` requires "
        "a reviewed exact coordinate/site mapping; missing support does not disqualify a site.", "",
        f"Site tiers: `{summary['positive_site_confidence']}`. "
        f"Local support: `{summary['site_support']}`. Individual sites, donor elements/positions, "
        "residue multiplicities, occupancies, angles and interchain partners remain in "
        "[zinc-sites.jsonl](zinc-sites.jsonl). Residue signatures are observed chemistry, not "
        "biological classes. Buried positives stay positive; their whole group is train-only.", "",
        "|Residue signature|Surface support|Chain topology|Sites|",
        "|---|---|---|---:|"]
    lines += [f"|{key[0]}|{key[1]}|{key[2]}|{count}|"
              for key, count in sorted(chemistry_support.items())]
    lines += ["", "Counts are individual accepted sites, not independent proteins. Interchain "
        "sites involve another chain copy; a single-chain model does not observe every partner. "
        "This table reveals chemistry disproportionately unavailable to surface evaluation.", "",
        "|Cutoff Å|Donor minimum|Residue minimum|Sites|Chains|Lost baseline sites|New sites|",
        "|---:|---:|---:|---:|---:|---:|---:|"]
    lines += ["|" + "|".join(str(record[key]) for key in (
        "cutoff_angstrom", "minimum_donors", "minimum_residues", "surviving_sites",
        "surviving_chains", "lost_baseline_sites", "new_sites")) + "|"
        for record in sensitivity.values()]
    lines += ["", "Large changes across cutoffs indicate policy dependence, not automatically "
        "bad chemistry. These counts cover RAW positive hypotheses in fixed assemblies, including "
        "quality rejects; they never create alternate labels, splits or datasets.", "",
        "## 4. Independence, representation and selection", "",
        "Full RAW sequence/MMseqs2/Foldseek and identity edges are joined transitively before "
        "quality selection: A-B and B-C keep A,B,C together even if B is excluded. Entire "
        "leakage groups occupy one split. They measure dependency, not function. Physical "
        "phenotypes describe shapes: global morphology for both classes, local coordination "
        "geometry for individual positive sites. They never create labels or leakage edges.", "",
        "Median/IQR scaling divides centered features by their middle-half spread. HDBSCAN "
        "finds density-connected shapes and marks uncertain assignments as noise. ARI compares "
        "cluster assignments across the parameter grid (1 identical, near 0 chance agreement); "
        "all-noise agreement is not evidence of clusters. See the actual stability/noise counts "
        "in audit.json; an unsupported multi-cluster solution remains unsupported.", "",
        f"Phenotype results: `{json.dumps(audit['phenotypes'], sort_keys=True)}`.", "",
        "All eligible positives are retained by default. Optional diversity quotas compare "
        "eligible and selected annotations in benchmark.json. An omitted category is a visible "
        "coverage loss, not proof it is biologically redundant. Missing Pfam/InterPro/CATH/EC "
        "assignments are missing; multidomain counts overlap. Source annotations do not enter "
        "the model. Rare-family independence is counted in leakage groups, not protein totals.", "",
        "|Classification|Eligible categories|Lost categories|One eligible group|"
        "Train-only|Heldout-only|",
        "|---|---:|---:|---:|---:|---:|"]
    lines += [f"|{field}|{len(entries)}|"
        f"{sum(not item['selected_members'] for item in entries.values())}|"
        f"{sum(item['single_independent_group'] for item in entries.values())}|"
        f"{sum(item['train_only_observed'] for item in entries.values())}|"
        f"{sum(item['heldout_only_observed'] for item in entries.values())}|"
        for field, entries in families.items()]
    lines += ["", "A category supported by one group cannot be represented independently "
        "in all splits. Heldout-only categories stress extrapolation, while train-and-heldout "
        "categories test within-category generalization, not guaranteed functional validation.", "",
        "## 5. Fixed partitions and imbalance", "",
        "![Protein class counts and RAW leakage-group sizes](diversity.svg)", "",
        "|Split|Positive chains|Negative chains|Leakage groups|", "|---|---:|---:|---:|"]
    lines += [f"|{split}|{counts['positive']}|{counts['negative']}|{counts['leakage_groups']}|"
              for split, counts in audit["splits"].items()]
    lines += ["", "The left plot shows members, not independent experiments. The right plot ranks "
        "RAW group sizes; a large tail means repeated related evidence. Zero crossing groups is "
        "required, not proof all dependencies are known. Imbalance is allowed; 1:1 does not make "
        "labels more trustworthy. Dilutions reduce train only; validation/test stay fixed.", "",
        "benchmark.json records per-class, per-split source/method/year, phenotype, chemistry, "
        "functional annotation and surface-support distributions, with missing counts. Family "
        "tables distinguish members from independent groups within each split. These describe "
        "the fixed benchmark; they never require splitting a group to fill a missing category.", "",
        "## 6. Acquisition bias versus structural shortcuts", "",
        "Origin, method, deposition year and resolution differences describe acquisition bias. "
        "They are not included in the structural baseline. That small logistic classifier uses "
        "sequence length, normalized radius of gyration, aspect ratio and packing density, fits "
        "only train, and reports validation AUROC/AUPRC/balanced accuracy; it never opens test. "
        "High discrimination flags a plausible shape shortcut, not proof. AUROC near 0.5 means "
        "chance ranking; AUPRC must be compared with the validation positive fraction, not 0.5 "
        "unless balanced. Balanced accuracy averages recall of both classes. Unsupported metrics "
        "remain unavailable with their support counts.", "",
        "Structural baseline: `" + json.dumps(audit.get("structural_shortcut_baseline",
            {"available": False, "reason": "disabled"})) + "`.", "",
        "[diversity.md](diversity.md) explains observed class-size/shape differences using "
        "standardized mean differences (SMD), with support and missing values. benchmark.json "
        "separates acquisition distributions from this structural-only validation control.", "",
        "## 7. Limits and next boundary", "",
        "Sources preferentially cover crystallizable/deposited/annotated proteins. Experimental "
        "NOT evidence is conditional on the assay and curation; exact mapping and group checks may "
        "leave too few negatives for a valid split, in which case construction stops. No claim "
        "of universal population coverage, physiological specificity, unbiased acquisition or "
        "perfect family coverage is made. MetalPDB/ZincBind are not automatically treated as exact "
        "site corroboration: their site/assembly/revision mapping requires verification. Reviewed "
        "frozen exact mappings can enrich audits without altering labels.", "",
        "Preprocessing consumes this fixed design, generates protein-only universal geometry and "
        "point-aligned Zn sidecars, and checks them before publication. The separate Validation "
        "Work then independently audits every published member. Its PASS establishes the stated "
        "contracts, not biological truth or neural-model performance.", ""]
    (root / "report.md").write_text("\n".join(lines), encoding="utf-8")
