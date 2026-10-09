# WISDOM post-hoc model review

## 1. Choose the evidence, not a new training run

These YAMLs evaluate saved models. They do not fit weights, create an optimizer, run backward,
change HPO, select another epoch or publish a StudyDecision. Each report says
**post-hoc inference on frozen checkpoint**. Training YAMLs remain in the parent directory.

| Source | Use when | Selection possibilities | What must be available locally |
|---|---|---|---|
| StudyReview | You want to examine the original replications and trade-offs | Explicit Trials, parameter filters, ranked Trials, Pareto G/S; median/worst/representative seeds | Registered Execution, its saved checkpoints, the exact training dataset |
| ModelSetReview | You have deliberately selected a small set of checkpoints to move | Reviews every promoted model, without re-ranking | Imported ModelSet and the exact training dataset; original Study is unnecessary |

An **Execution** identifies one concrete LF execution, not merely a YAML name. A **Trial** is one
hyperparameter configuration; a **seed** identifies a stochastic replication. A **ModelSet** is
LF's immutable product containing selected weights, their snapshot metrics and scientific input
identities. It owns independent copies, not links into the original Study.

**Required LF capabilities.** The inspected LF main still identifies itself as 0.17.0, while
products/import APIs are marked Unreleased. A version string alone does not guarantee these features.
Use a build with public ResultStore.select/execution_directory/import_export, ProductInput,
SelectionPolicy, native product transport and outputs.html_section. Check before copying large data:

~~~bash
python -c "from lambdaforge.products import ProductInput, SelectionPolicy; from lambdaforge.work import ResultStore; assert callable(ResultStore.execution_directory)"
lf import --help
lf products select --help
~~~

WISDOM's dependency floor is unchanged: no published minimum is invented. Update/bootstrap the
actual LF build on the host running review. LF source remains an external, read-only dependency.

## 2. Workflow A: flexible Study review

Export on a host that can access the Study, or use LF's remote-aware export from the submitting host.
Use the exact Work/Execution ID shown by LF; the output directory is a parent:

~~~bash
lf export SOURCE_EXECUTION_ID --output ./exports
~~~

Transfer the generated package directory to the review machine. Then register it without running it:

~~~bash
lf import ./exports/wisdom_v1a--SOURCE_EXECUTION_ID --apply
lf results list
~~~

Use the **actual package name printed by export**, which may use a configured Study name rather than
the YAML filename. Import preserves original identities and relocates evidence. Imported evidence
is read-only, not resumable training state. Also transfer/register the original DatasetVersion
through LF's dataset facilities: Study export does not guarantee a complete dataset placement.

Edit with.source_execution in visualization/wisdom_v1a.yaml. It must be the registered ID, not a
remote path. Change the seed policy to median for a typical observed replication:

~~~yaml
seed_selection:                         # Replication policy inside the single baseline Trial.
  mode: median                          # Observed central seed; not an averaged model.
  metric: wisdom_score                  # Exact saved-checkpoint validation W.
~~~

~~~bash
lf validate experiments/visualization/wisdom_v1a.yaml
lf explain experiments/visualization/wisdom_v1a.yaml
lf run experiments/visualization/wisdom_v1a.yaml --dry-run
lf run experiments/visualization/wisdom_v1a.yaml
lf results list
lf results report REVIEW_EXECUTION_ID --output review.html
~~~

Open review.html in a modern browser, then **WISDOM proteins**. The report can also be opened from
LF's Research Console. candidate-inventory.json lists eligible native Trial/seed records;
review-audit.json and model-review/*/proteins.csv retain numerical results and display reasons.
A failed source Study can contain successful reviewable Runs. Failed, pruned and reduced-fidelity
Runs are excluded; the latest Attempt is authoritative, not an older successful Attempt.

For best/median/worst together use mode: representative (the V1a template's deliberate choice);
mode: all includes every eligible seed. Duplicates are removed when only one or two seeds exist.
An optional results_root typed file directory selects another **local** ResultStore; it is not an
instruction to chase remote paths.

## 3. Select configurations and seeds explicitly

`wisdom_v5c.yaml` reviews the evidence-refinement sweep without fitting weights. Its default
Trial policy includes all completed settings; narrow it to explicit Trials or a parameter filter
before a large report. Choose explicit protein IDs to compare the same deposited chain across
MAX/LSE/Attention and refiners. The shared viewer shows raw logits/probabilities, refined
predictions/logits, refined-minus-raw logit difference and GT. The chosen checkpoint and learned
refiner parameters are restored exactly. Numerical metrics use all requested proteins/points;
viewer point/protein caps and report budgets affect presentation only.

With multiple Trials, omitting trial_selection fails rather than silently declaring a winner.
With one Trial, omission uses it. Ranked Trial policies aggregate snapshot metrics over eligible
seeds by median by default; aggregation: mean is the explicit alternative.

| Trial mode | Required/optional fields | Meaning |
|---|---|---|
| all | None | Every eligible configuration; can be expensive |
| explicit | trial_indices | Native indices, not display positions |
| parameter_filter | where | Every supplied exact parameter condition must match |
| best_by_metric | metric=wisdom_score, direction=max, aggregation=median | Highest aggregate, explicitly favorable |
| top_k_by_metric | Same, k=3 | First k configurations after ranking |
| pareto | metrics mapping, maximum_trials=8, aggregation=median | Keep configurations not dominated on all requested metrics |

Direction uses a separate key because mode already names the policy. Pareto does not substitute
W for G/S: a configuration dominates another only if it is no worse on every requested score and
strictly better on at least one. If the frontier exceeds maximum_trials, native-index order truncates
it; that cap is not a second quality ranking.

For V5a, replace its active trial_selection with:

~~~yaml
trial_selection:                         # Compare actual families; no automatic best-per-family rule.
  mode: parameter_filter                 # Every condition must match.
  where:                                 # Exact active parameters, not constructor defaults.
    pooling_type: [max, log_sum_exp, attention] # All observed settings in these three families.
seed_selection: {mode: median, metric: wisdom_score} # Typical seed per configuration.
~~~

For exact LSE settings add log_sum_exp_mode: fixed and log_sum_exp_beta: [10.0, 20.0, 40.0, 80.0]
under where. A missing conditional key does not equal an active key with null.
Curriculum LSE in V5a uses pooling_curriculum_end_beta; learned LSE in V5b uses
log_sum_exp_beta_init. A learned initialization is not its final learned value.

Seed modes are all, explicit (seeds list), best, worst, median, top_k, bottom_k and representative.
Ranked modes accept metric (wisdom_score default), direction (max default), and k (3 default).
For an even seed count, median chooses the lower central index of the best-to-worst observed order,
never synthetic averaged weights. A favorable picture does not replace paired-seed statistics.

## 4. Quick inspection of specific proteins

templates/explicit_proteins.yaml demonstrates three IDs. Replace its Execution, Trial, seed, dataset
and IDs with real evidence. For one protein keep only one identifier. Then:

~~~bash
lf validate experiments/visualization/templates/explicit_proteins.yaml
lf run experiments/visualization/templates/explicit_proteins.yaml
lf results report REVIEW_EXECUTION_ID --output explicit-review.html
~~~

evaluation_scope: explicit filters before the DataLoader and neural forward. Only requested proteins
in requested splits are evaluated; metrics are marked **requested subset only**.
evaluation_scope: full evaluates the entire split and can still show just those three IDs.
Unknown IDs fail rather than silently falling back to random pictures.

Automatic selection defaults to diagnostic, two cases per supported category. It contrasts best/worst
positive localization, global false positives/negatives, negative peaks and global/local mismatch.
Unavailable surface GT is not ranked as zero. Strategies best, worst and mixed consider positive
localization; categories lacking support are simply absent. all requires a size warning and remains
subject to the display cap; set maximum_proteins: 0 to remove it deliberately.

## 5. Workflow B: lightweight durable ModelSet review

On the host with the registered Study and exact model bytes:

~~~bash
lf products select SOURCE_EXECUTION_ID \
  --name review-models \
  --contract wisdom/review-models:v1 \
  --policy experiments/visualization/policies/top3-wisdom-score.yaml \
  --apply
lf products export review-models --output ./modelset-export --apply
~~~

Transfer the product bundle (not the whole Study), then on the review host:

~~~bash
lf products import ./modelset-export --apply
lf products verify review-models
lf validate experiments/visualization/templates/modelset_review.yaml
lf run experiments/visualization/templates/modelset_review.yaml
lf results report REVIEW_EXECUTION_ID --output modelset-review.html
~~~

The YAML consumes models: {product: {name: review-models, contract: wisdom/review-models:v1}}.
It uses only ProductInput.payload and ProductInput.artifact. Deleted source Studies cannot prevent
review; missing/corrupt promoted files still fail. Native product selection ranks individual
checkpoints, not median seeds or WISDOM Pareto fronts: use StudyReview for those comparisons.

Policies select top one/three by W, top one/three by G, or top one per exact pooling_type.
best-per-pooling-family.yaml groups **only** pooling_type, so settings/seeds compete within that family.
Do not mistake these observed inspection products for confirmed scientific winners.
No automatic StudyDecision or training products blocks are added, especially not to baseline V1a.

## 6. Snapshot metrics, historical evidence and scientific limits

New Training artifacts attach step/epoch and metadata.metrics to best-model's exact saved weights:

| Name | Meaning |
|---|---|
| global_score, protein_auprc, protein_auroc | Protein validation at the saved epoch; G=0.70 AP+0.30 AUROC |
| surface_score, surface_positive_macro_auprc | Positive-protein macro AP of the restored checkpoint, if available |
| selection_regret | max observed S minus this checkpoint's S, floored at zero; requires at least two paired observations |
| coupling | Run-curve context C anchored to the actual checkpoint's regret |
| wisdom_score | W=0.35G+0.45S+0.20C, only when all components are available |

C uses the existing convention C=0.70(1-min(regret/0.20,1))+0.30(rho+1)/2; rho is late-curve
Spearman rank agreement. Undefined rho contributes the existing neutral zero to this score while
remaining unavailable as a scientific correlation. Sparse observation schedules may miss peaks.
No curve means no fabricated regret/C/W; use global_score ranking or explicit selection instead.
Checkpoint selection remains protein-validation-only. Test metrics never enter artifact ranking.

Historical checkpoints without metadata remain reviewable with all/explicit Trial **and** seed
policies. They are warned about, not automatically called best or median from latest Run metrics.
Prefer lf import before review. ModelValidation remains a compatibility-only direct portable reader,
not the primary interface. No checkpoint exists: recover it through native LF; reviews never retrain.

Default splits=[val], allow_test=false. Test requires both splits=[test] and allow_test=true.
Inspection is descriptive and must not reopen hyperparameter choices. All sources require the
exact training dataset identity; a placement can change, its scientific contents cannot.
A ModelSet combining incompatible datasets needs separate products/reviews.

## 7. Defaults and study-specific examples

| Option | Actual default | Effect |
|---|---|---|
| evaluation_scope | full | Full requested-split numerical coverage |
| protein_selection | automatic/diagnostic/2 | Contrasting diagnostic pictures |
| seed_selection (Study only) | median by wisdom_score | Typical observed replication |
| splits / allow_test | [val] / false | Held-out test remains sealed |
| content | predictions | Only prediction, original logits and available soft/hard GT |
| maximum_surface_points | 2000 | Display-only deterministic point sample |
| maximum_proteins | 12 | Per-model/split display bound; authored examples choose eight |
| prediction_threshold | 0.5 | Initial hard prediction and binary error threshold |
| report_maximum_mib | 14.0 | Document byte ceiling, at most 16 MiB |
| save_predictions | false | No duplicate prediction NPZ by default |
| batch_size / data_workers | 4 / 0 | Inference batching and decoding only |
| strict_checkpoints | true | StudyReview: missing selected model fails; false audits/skips. Native product integrity errors always fail, including missing promoted files |

Scientific metrics use all original points and full precision. Display coordinates/scalars may be
rounded and capped; unavailable metrics remain null with support counts. Report byte omissions are
listed rather than hidden. Embedded HTML has no standalone PLY/NPZ copies unless numerical exports
are explicitly enabled. Change content to full for atoms/mesh/structural channels.
The LF total report limit is separate from WISDOM's document budget.

Examples correspond to every current authoritative training Study:
V1a baseline variance; V1b training RNG; V1c initialization RNG; V2 initialization profiles;
V3 optimizer profiles; V4 core HPO; V5a fixed/curriculum family curves; V5b five learned-scalar adaptations;
V6a negative loss; V6b existence; V6b2 regional existence; V6b3 regional ranking; V6c cardinality;
V6c2 TV; V6c3 Dirichlet; V6d loss interaction HPO; V7 heads; V8 controlled spikes;
V9 local retuning; V10 confirmation. Their files use contrasting actual factors or explicit bounded
Pareto/top-k policies, not twenty hidden identical winner selectors.

wisdom_v5.yaml was the historical pooling screen, now retired; V5a/V5b are authoritative.
The current V1b is already a single Work, with initialization RNG isolated in V1c.
Historical composed Studies must be imported/reviewed by their separately recorded child Execution
IDs, not fabricated as Trials of the present V1b.

## 8. Technical verification

tests/test_review_native.py covers native ResultStore and imported evidence, durable product
export/import after producer deletion, snapshot-only ranking, policies/conditional null, partial
forward coverage, corruption/missing evidence, dataset mismatch, test access and HTML sections.
Optimizer/backward paths are prohibited in review tests. Frozen synthetic weights establish
executability only, not biological accuracy. No real campaign is launched by these checks.

All source-bearing example selectors are placeholders until imported evidence and a dataset placement
exist. Policy YAMLs are SelectionPolicy documents, not executable Works: validate them with that
public schema, not lf validate. Work YAML validation/dry-run uses temporary typed fixtures without
publishing fake production datasets.
