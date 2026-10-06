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
policy. V5a now screens all eleven implemented pooling families with 206 candidates (824 paired
Runs), including both attention variants and fixed/curriculum LSE. V5b isolates learned-scalar
initialization with 75 candidates (300 Runs). Both preserve the reviewed backbone, seeds,
objectives and losses. Their current dispatch and scheduler budgets are 1000 hours for V5a and
168 hours for V5b. Researcher-selected allocations are three GPUs/42 CPUs for V5a and two
GPUs/36 CPUs for V5b. Curriculum hold is zero in V5a so changing schedules do not
remain frozen through patience 30; endpoint 1 is an intentional constant-beta control.

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

This section records the historical conditional-YAML migration, not the current candidate grids.
The subsequent range/family expansion is described in section 9 below.

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

## 9. Full-family pooling coverage and extreme controls (2026-10-04)

V5a now includes all eleven implemented families, both simple/gated attention scorers, and fixed
or curriculum LSE. Its native branch counts are 1/2/32/26/16/54/2/26/24/22/1, totaling 206
candidates and 824 Runs with the same four seeds. V5b contains learned LSE/regional/AutoPool/GeM/
MAX–MEAN scalar initialization only: counts 16/9/16/16/18, totaling 75 candidates and 300 Runs.
No model operator, learned-scalar bound, encoder, dataset, objective or loss was changed.

Top-K includes 0.6, 0.8 and 1.0; attention spans widths 4–512, regional diffusion lengths 0–64
angstroms, fixed LSE beta 0.01–1280, AutoPool alpha 0–1000, GeM power 1–1024, and MAX–MEAN
weight the complete [0,1] interval. The existing multiscale regional bank remains [0,1.5,3,6,12]
angstroms. Learned initializations approach both existing bounds rather than redefining them.
Inactive dimensions remain absent through native conditions. Only V5a contains the exact MAX
reference; endpoint-equivalent cells are deliberate limiting-behavior controls.

The finite screen cannot guarantee that a response curve has reached its optimum or plateau.
An improving edge remains unresolved and requires a separately planned extension based on paired
uncertainty, surface evidence and cost, with held-out test still sealed. Small Top-K fractions
can discretize to identical support counts; very sharp pooling can concentrate gradients or
saturate probabilities. Positive regional lengths operate on a truncated spectrum and retain
component-specific constant modes when available, unlike the exact zero-length identity.

Focused numerical checks construct every native V5a/V5b candidate and verify finite forward and
backward results, including the extreme fixed values and curriculum endpoints. They also verify
Top-K/full-mean and MAX–MEAN endpoint equivalence and high-sharpness convergence toward MAX.
These establish numerical executability on the test fixture, not scientific improvement or
training stability on the actual dataset. The inherited analysis profile and native result
questions remain unchanged; no WISDOM ranking controller was introduced.

Verification for this expansion: Ruff and mypy pass; the complete suite reports 257 passed and
four skipped tests requiring absent production data/local placement. Both changed YAMLs pass
native `lf validate` and `lf explain`. Remote dry-runs for `citius-ctgpgpu12` pass with two exposed
GPUs; the local workstation cannot admit their unchanged two-GPU request. No training was started.
The bilingual README heading hierarchy remains equivalent. Production performance and complete
824/300-Run coverage remain untested; the scheduler still caps each allocation at 168 hours.

## 10. Project protein inspection in native HTML sections (2026-10-05)

The current uncommitted LambdaForge public API adds `outputs.html_section(name, section, title)`.
WISDOM declares `protein-report` under the `wisdom-proteins` section with title `WISDOM proteins`.
LF owns artifact finalization, fingerprint verification, collection, Trial/seed selectors,
isolated iframe presentation and report/export lifecycle. No external framework source or
metadata was patched. Every recorded seed remains available; the viewer does not select a winner.

`ProteinReportPage` packages the shared `ProteinVisualizer` in a searchable split/label gallery
with a return button and a single compressed Plotly library per document. Inspectors open lazily
without nested frames, neighboring files or network requests. Switching proteins waits for queued
updates, removes the previous resize listener and releases its WebGL scene. GT hard/soft,
prediction probability and adjustable hard prediction, structure channels and mesh interpretation
stay identical to the standalone preprocessing inspector.

Training's new `report` presentation mode emits embedded inspectors and small audit/index files
instead of per-protein HTML/PLY/NPZ. Existing `viewer`/`full` modes are preserved; `none` does not
generate maps. `surface_report=true` declares the tab, even if it must explain a disabled, empty-ID
or competitively pruned Run. False omits the tab and cannot be combined with report-only mode.
Exact IDs can override the automatic balanced sample, but only evaluated splits are eligible.
Generation is final, using the restored protein-validation-selected checkpoint; numerical
surface-metric intervals, training losses, objectives and held-out-test policy do not change.

The per-Run budget defaults to 4 MiB, with explicit omitted-viewer IDs. The native limits are
16 MiB/document and 64 MiB/combined report, so large HPO studies must keep sections disabled or
maps off and inspect confirmation Runs. This integration does not bypass the combined limit or
retroactively reconstruct old predictions. Browser compatibility requires DecompressionStream.
Tests cover empty states, byte omission, exact channels, native output metadata and report-only
file policy; optional Chromium exercises the real LF sandbox, prediction threshold, mesh, return
navigation, multiple protein scenes and seed selection with no external requests.

Verification: Ruff and mypy pass; the full suite reports 264 passed and four skips for absent
production data/placement. The seven focused integration tests also pass with Chromium, including
a real one-epoch CPU Training fixture that publishes the native section from its best checkpoint,
without test evaluation or standalone prediction files. This verifies execution, not model quality.
Changed Work configurations validate and explain; preprocessing and V10 dry-runs pass locally.
V1a's authored three-GPU request cannot run on this one-GPU workstation. Its temporary dry-run
uses one GPU and the tiny unregistered synthetic fixture; scientific model parameters and the
repository allocation are unchanged. The production Registry selector must resolve on its cluster.
Both README heading hierarchies and language links match. No remote scientific job was launched.

## 12. Frozen-checkpoint visual review

Training studies no longer compose executable steps. V1b/V1c isolate the two RNG sources;
V6b/V6b2/V6b3 isolate existence, regional existence and ranking; V6c/V6c2/V6c3 isolate
cardinality, total variation and Dirichlet regularization. Their original controls and grids
remain unchanged. The dataset-construction workflow remains a composition.

Separate `VVAL` YAMLs were retired in favor of Training's `visualization` mapping. The existing
`ModelValidation` Work remains available for explicit frozen-export compatibility, not as a
second required campaign. For that optional API, the researcher supplies a
native LF portable Study export and a reviewed mapping of display names to native Trial indices.
An empty mapping publishes inventory only; WISDOM does not rank candidates. Every completed,
unpruned seed remains visible unless explicitly filtered. Native Run/Attempt identities reconnect
the exported `best-model` artifact; its inventory checksum and the source dataset identity are
verified before inference. Saved model/data parameters and gate warm-up override are restored.
Training now persists that override because warm-up state is not part of the weight state_dict.
Older exports reconstruct it from the checkpoint epoch and original warm-up schedule.

Each selected checkpoint evaluates every protein in each requested split without an optimizer.
The existing global/local metric producers are reused. Review metrics are indexed under
`review_<audit-number>_<metric>`; the detailed audit maps each number to Trial, seed and split.
These are descriptive fixed-checkpoint measurements, not new Training objectives or WISDOM
statistical comparisons. The original LF paired-seed analysis remains authoritative.

Presentation selects overlapping categories of binary errors, poor/good positive localization,
global/local mismatch, negative false mass and confident correct cases. Exact-ID and complete
viewer requests are supported. Sampling never changes numerical coverage; undefined ground truth
is unavailable rather than zero. The generated interpretation README explains prevalence-normalized
AP, area-weighted activation mass, peak activation and the descriptive mismatch heuristic.

One shared `ProteinReportPage` embeds the existing `ProteinVisualizer` inspector with model and
seed filters. Full-split CSV/JSON evidence accompanies a 14 MiB default native HTML section;
omitted viewers are explicit. Prediction NPZ export is opt-in and no standalone viewer copies
are generated. Test access requires explicit authorization and never chooses parameters.
Missing checkpoints stop or are audited/skipped, according to `strict_checkpoints`. Exceptional
automatic retraining is not implemented: the current public LF API exposes no nested per-Run
recovery lifecycle, and WISDOM must not reproduce framework orchestration.

Verification: Ruff and mypy pass for 126 source files. The full suite with optional Chromium
reports 266 passed and four skips for unavailable production data/placement. A real one-epoch
CPU fixture exports its native checkpoint, then performs frozen review with optimizer construction
forbidden. It tests warm-up restoration, complete metric coverage, empty inventory mode, byte
corruption, and both missing-checkpoint policies. Chromium verifies model/seed filtering, inspector
navigation, threshold and mesh controls inside LF's sandbox without external requests.
The earlier separated configurations passed validation using temporary bindings; their VVAL
YAMLs have since been retired. Production training and preprocessing configurations validate, and the
preprocessing explain/dry-run passes. Bilingual section/TOC numbering agrees. No remote scientific
job was launched, and no LambdaForge source, installed package or metadata was modified.

### Lightweight reports configured in Training

Use `with.visualization` in the existing training YAML, not a separate visualization Study.
Its modes are none/report/viewer/full; the default content, predictions, includes only probability,
original logits, adjustable hard prediction and available soft/hard GT. Full content restores
the structural inspector. Maximum points defaults to 2000 and affects display only; full-split
scientific metrics and optional numerical exports retain their original coverage and precision.
Report mode writes no standalone protein HTML/PLY/NPZ duplicates. NONE suppresses the report tab
even if the legacy embed flag remains true.

Each eligible Run generates its report once after restoring the best protein-selected checkpoint.
A failed Study can contain successful seeds with reports; a failed Run before publication has no
finished viewer. An omitted gallery indicates a byte budget, not missing model predictions.
In the synthetic regression fixture the compact inspector is about 43 KB versus 762 KB for the
structural inspector, and four embedded compact viewers plus Plotly fit in approximately 2 MiB.
This is a payload measurement, not a scientific-quality or production-speed claim.

The current verified public LF API has no post-Study Work callback for forwarding only a final
best candidate/seed. WISDOM does not implement hidden ranking, scheduling, retraining or generating
all seed pictures to delete them later. Large HPO studies retain mode none until that lifecycle
is available. Per-Run reports never replace repeated-seed evidence.
