# WISDOM Research Analysis integration

## 1. Ownership and API

The integration uses the local LambdaForge source's **0.16** public contracts:
`Work.analysis_profile`, `lambdaforge.analysis.AnalysisProfile.resolve`, and
`lambdaforge.analysis.MetricCatalog.resolve`. Numerical exploration is delegated to
`lambdaforge.analysis.ResearchAnalysis.ResearchAnalysis`; CLI/HTML/result persistence remains
framework-owned. LambdaForge was inspected read-only, including `docs/RESEARCH_ANALYSIS.md` and
the catalog, profile, Work configuration, runtime, research, and evidence implementations.

One substantive class, `wisdom.analysis.WisdomAnalysisProfile`, describes existing scientific
measurements. `Training.analysis_profile` receives its declaration at class-definition time.
No data, model, targets, or optimizer are constructed to obtain this profile. No generic
profiling, clustering of metrics, relationship fitting, hypothesis testing, finding ranking,
inbox, search, family plotting, saved views, or HPO controller is implemented in WISDOM.

The framework validates the class declaration plus native YAML overrides before execution and
persists the resolved immutable semantics with a separate semantic identity. Descriptions carry
WISDOM's catalog revision; the framework owns schema versioning and fingerprints. Never inject
an unsupported `profile`, `version`, or custom Study field to mimic those mechanisms.

## 2. Audit of actual producers

The audit follows `Training.metrics.log` calls and the return keys of `BinaryMetricSuite`,
`SurfaceMetricSuite`, `SubgroupMetricSuite`, `SurfaceFaithfulnessAudit`,
`ViewConsistencyMetricSuite`, and `OptimizationDiagnostics`. A call with `split="val"` emits
`val_<name>`, not an unprefixed metric. This matters for already-double-prefixed timing names.

| Source | Actual keys or native bounded patterns | Interpretation |
|---|---|---|
| Binary evaluator | `{val,test}_{accuracy,balanced_accuracy,precision,recall,specificity,f1,mcc,cohen_kappa,auroc,auprc}` | Selected-checkpoint protein measurements; undefined values remain absent, not fabricated scores. |
| Calibration/global heads | `{val,test}_{loss,surface_derived_loss,head_disagreement}`, `{val,test}_surface_derived_<binary metric>` | Protein BCE and disagreement; a surface-derived *protein* score is not local GT accuracy. |
| Local evaluator | `{val,test}_surface_{micro,positive_macro}_{auprc,auroc,balanced_accuracy,f1}` | Pooled points versus equal-weight positive-protein localization. |
| Local ranking | `{val,test}_surface_positive_macro_{normalized_auprc,top_<5,10,25>_<recall,enrichment>}` | Prevalence correction and hotspot ranking; normalized AUPRC has no universal finite lower bound. |
| Negative control | `{val,test}_surface_negative_{positive_mass,peak}` | Mean area-weighted false activation and mean maximum false hotspot. |
| Attention/uncertainty | `{val,test}_surface_attention_*`, `surface_confidence_quality_spearman`, `surface_<map_entropy,head_disagreement>_<error_spearman,auprc_at_coverage_<70,80,90>>` | Pooling attention, confidence/local-quality association, selective-retention diagnostics. |
| Curves | `val_wisdom_hpo_{global,surface,coupling,score}`, `val_surface_selection_regret`, `val_global_surface_spearman[_all]`, `val_surface_{best_score,regret_component}` | Existing G/S/C/W and trajectory summaries, not last-epoch substitutes. |
| Raw duplicate keys | `val_protein_global_score`, `val_surface_selected_score`, `val_surface_coupling_score`, `train_total_loss` | Identity lineage to canonical measurements; retained raw but excluded from automatic discovery. |
| Faithfulness | `{val,test}_faithfulness_<deletion,insertion>_<top,random,bottom,top_minus_random>_<1,2,5,10,20>` | Equal-area pooling-boundary interventions without local targets. |
| Strata | `{val,test}_subgroup_<atom_count,surface_count,surface_prevalence>_<q1,q2,q3,q4>_*`; tiers core/challenge; dynamic global/interface phenotypes | Support, class balance and group quality. Phenotypes are not leakage groups. Curve summaries select within each subgroup, not at the full-population optimum. |
| Optional views | `{val,test}_view_*` | Existing physically aligned-view evaluator; Training does not schedule these views. |
| Optimization | `train_weak_surface_*`, `train_gradient_*`, `train_activation_*`, `train_surface_logit_*`, `train_sampled_gate_*` | Epoch diagnostics, not validation outcomes. |
| Effective controls | `learning_rate`, `effective_gate_lambda`, `pooling_*`, `final_pooling_*`, `initial_local_head_bias` | Scheduled/current controls and selected-checkpoint pooling scalars. Not scientific discoveries. |
| Capacity/representation | `parameter_count`, other parameter counts, `train_maximum_*`, `train_mean_*`, `train_active_*` | Capacity or first-epoch representation profile; topology echoes are excluded from ordinary discovery. |
| Diffusion | `{initial,final}_diffusion_time_{minimum,mean,median,maximum}` | Heat time in square ångströms; no universally favorable direction. |
| Timing | `train_train_seconds`, `train_train_proteins_per_second`, `train_epoch_seconds`, `train_proteins_per_second`, `train_data_wait_seconds`, `val_validation_seconds`, `{val,test}_forward_*` | Pure training, end-to-end epoch, data wait, whole validation and forward-only inference are not aliases. |
| Memory | `train_cuda_{allocated_gib,reserved_gib,peak_gib}` | PyTorch epoch samples/peaks, not all-process physical memory. Terminal value is the last logged epoch. |
| Framework costs | `resource.{duration_seconds,gpu_seconds,cpu_seconds,peak_vram,peak_ram}` | Median per comparable screening Run. Existing framework summaries are reused, not recomputed. |
| Support/integrity | Surface point/protein counts, prediction/gallery counts, `val_mcc_defined`, patience, `val_surface_metrics_computed`, `val_averaged_validation_weights` | Counts/status, hidden and excluded from ordinary discovery. |

Unknown future measurements receive an explicit unclassified description and are excluded from
automatic exploration until their producer is understood. No guessed direction, range, family
coordinate or biological meaning is inferred from a filename or a vague substring.

## 3. Categories, priorities, and dependencies

The catalog hierarchy distinguishes validation/test global quality, thresholded decisions,
localization, ranking, negative controls, attention, calibration, coupling, selection,
faithfulness, subgroups, and view consistency. Training diagnostics, model capacity/gating/
diffusion, runtime costs, and support/integrity remain separate. Explicit `split` metadata—not
category text—enforces test safety.

G/S/C/W have highest priority, followed by surface macro quality, regret/correlation, protein
ranking/MCC, and secondary diagnostics. Priorities focus a bounded discovery budget, not an
objective or a statistical confidence level. Hidden status/count/parameter echoes remain
inspectable through the native metric browser. Resource evidence can support quality/cost
tradeoffs without being ranked as biological accuracy.

Human aliases support search without changing logged keys. Already-recorded duplicate names
instead use `derived_from`, `transformation="identity"`, and `discovery=false`; declaring them
as aliases would collide with canonical keys in the public schema. W's G/S/C dependency, G's
binary components, C's regret/rank components, top-minus-random controls, and L1/total-variation
equivalence are explicit. These relationships must not be advertised as independent findings.

## 4. Families and configured questions

Explicit numeric/categorical coordinates describe top-fraction recall/enrichment, intervention
strategy/fraction, retained coverage/uncertainty proxy, size/prevalence quartiles, difficulty
tiers, view components, and initial/final diffusion summaries. No regex parser or alternative
family registry is maintained. Dataset-dependent phenotype labels receive specific pattern
metadata; `build(global_phenotypes=..., interface_phenotypes=...)` can describe known sanitized
labels, or the author can supply a native YAML family. The default class profile does not invent
an arbitrary range of cluster IDs.

Configured questions cover global/local alignment, absolute-quality tradeoff, regret/coupling,
selection decomposition, negative controls, local faithfulness, attention/evidence alignment,
subgroups, optional views, model size, epoch costs and framework whole-Run costs. Parameter
screens use the real authored ParameterSpace; pooling-specific screens include fractions,
attention capacity/variant, physical regional lengths, LogSumExp beta, area mode, curriculum
endpoint, AutoPool alpha, GeM power and MAX/MEAN mixing. The generic core screen includes other
actual search dimensions without manually enumerating architecture configurations.

Relationship expectations express desired agreement, not a theorem: G and S should correlate
positively across comparable candidates, while high W should not routinely coexist with high
regret. An expectation violation is an inspection lead, not a model-selection rule. Families
and category summaries are descriptive. Their standard deviation across candidates is not a
confidence interval across seeds. Missing attention, unsupported regional point-removal audits,
sealed test and absent views do not produce synthetic observations.

## 5. Terminal evidence and immutable studies

WISDOM now republishes **every available** best-validation binary/head/calibration metric,
not just AUPRC, AUROC, balanced accuracy and MCC. In particular terminal loss, F1, precision,
recall, accuracy and specificity no longer accidentally describe a plateau after checkpoint
selection. Selection utility is republished from the same G-selected epoch. Surface evaluation
already restores these weights before its final pass; the existing curve summaries remain
distinct. Censored/pruned observations are not exact completed candidate evidence.

Declaring `aggregation="selected_epoch"` describes the computation; it does not instruct
LambdaForge to invent missing optima. Likewise epoch peak memory is not relabeled a whole-Run
maximum. Framework resource summaries already cover whole-Run cost, so no duplicate WISDOM
cost aggregator was added.

Test and transitive test-derived metrics are rejected from HPO objectives and constraints by
the native schema. Provisional discovery excludes them. Final test evidence is post-hoc only;
this integration neither opens sealed test nor changes the scientific weak-supervision protocol.

## 6. Unsupported public-framework cases

The following requested mechanisms have **not** been recreated in WISDOM:

1. Standalone metric-group objects: use native hierarchical categories; no group schema exists.
2. Independent analysis-rule objects, question-level descriptions, top-region-only
   filters, quantitative support gates, conditional `enabled_if`, explicit reference-candidate
   anomaly rules: unsupported by the current question schema. Native questions, metric
   priorities, directions and existing finding diagnostics cover only the supported behavior.
3. Dynamic family coordinates inferred from observed dataset labels: the native family schema
   requires a finite known dimension domain. Unknown phenotype values retain pattern metadata.
4. Terminal missing observations: `MetricCollection.log` accepts finite numbers only. An
   undefined selected-checkpoint metric cannot erase an earlier finite value in `_latest`.
   JSON remains authoritative for that absence; MCC additionally has `val_mcc_defined`.
   Do not fabricate zero, NaN or a private-state mutation. Generic terminal analysis must gain
   native missing-observation support before it can reliably consume this transition.
5. Optional-question availability: the backend may call a configured question “available”
   because its metrics are declared, even when finite candidate support is zero. Inspect metric
   profiles/family support instead; WISDOM does not postprocess the framework's status flags.

Suggested LambdaForge follow-up: expose an explicit missing terminal observation that removes
a stale finite terminal value while preserving its trajectory, and determine optional-question
availability from comparable finite support. If stricter domain questions are desired, extend
the public profile with validated support/reference/condition semantics rather than asking a
consumer to implement parallel discovery. These changes require framework-owner authorization.

## 7. Configuration, inspection, and verification

Every Training experiment inherits the profile. V5a/V5b each declare one conditional Study with
small metric-priority overrides on its single executable Work. A composed workflow has `analysis` on
each Work step, never on the workflow root. Class metric fields merge with YAML fields;
families replace by name, defaults/questions replace whole lists. Analysis policy is not HPO
policy. The unified V5 Studies retain all reviewed cells, seeds, objectives and losses, request
two GPUs, and distinguish their seven-day dispatch budget from scheduler wall-time. V5b changes
only the curriculum hold from 0.30 to zero to avoid an unchanged beta through patience 30.

Use `lf results analyze EXECUTION --recompute --json` for machine-readable research,
`lf results report EXECUTION --output research.html` for an offline interactive report, and
`lf export STUDY --output ./exports` to bring remote evidence to the local workspace. Install
the `analysis-report` extra for HTML. Historical studies lacking frozen WISDOM semantics remain
historical: generic inference is not a retroactive catalog migration.

Focused tests resolve the actual native API; audit actual evaluator return keys; verify alias,
lineage and family coordinates; demonstrate inverted G/S evidence with native discovery;
check test objective/constraint exclusion, absent view/intervention support, class-profile
resolution, native pre-run persistence/HTML generation, and selected-checkpoint logging.
No production study is executed or fake dataset
published by these tests. Numerical tests establish integration, not scientific superiority.

The inspected local editable source reports 0.16.0, but its installed distribution metadata
still reports 0.15.0. Tests exercise the new source API, not an old wheel. Reinstall that editable
dependency through the normal environment installation before rebuilding project environments
with the new `>=0.16.0` requirement. WISDOM did not modify external package metadata or reinstall
LambdaForge during this task.

## 8. Conditional pooling Study verification (2026-10-03)

The installed source is commit `d97ebab`, which implements native finite membership conditions.
The pre-refactor candidate union was captured with the public WorkConfig planner before editing:
V5a has exactly the same 56 effective configurations; V5b retains all 79 configurations, with
one intentional change on its ten curriculum cells (`pooling_curriculum_hold_fraction: 0.30`
becomes `0.0`). Irrelevant attention/default-hold fields are omitted, not experimental dimensions.
No model operator, optimizer, loss, dataset, seed or objective changed. Both profiles still inherit
Training's shared metric semantics and questions. The six V5a and eight V5b family steps and their
YAML anchors have been removed; no project-owned conditional generator was introduced.

Native validation and explanation confirm one Study per file, branch counts 1/2/10/16/9/18 and
8/18/4/2/16/18/12/1, four seeds `[4, 7, 32, 54]`, 224/316 required Runs, two requested GPUs,
and 604800-second dispatch and scheduler ceilings. The unique MAX reference belongs to V5a only.
These are complete-design obligations, not proof that a seven-day budget completes every Run.

Remote dry-runs of both files stopped before submission: LambdaForge requires an exact native
package solve that has not been prepared. Read-only `lf doctor --on citius-ctgpgpu12 --json`
confirmed two H100 GPUs and CUDA availability, but reported unsafe temporary mmseqs/Foldseek
paths in the active environment receipt. Hardware capacity is compatible; remote execution
readiness is not yet verified. Follow the framework's bootstrap diagnosis rather than editing
its receipts or source. No bootstrap, training, or remote scientific publication was performed.

The curriculum begins changing at epoch 2 instead of 152, retaining patience 30. At epoch 30
the scheduled beta is `1 + 29/500 * (end_beta - 1)`, so the target choices already differ, but
early stopping can still prevent reaching them. Recorded beta trajectories and the restored
checkpoint must be interpreted as realized behavior, not as the configured endpoint.
