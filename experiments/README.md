# WISDOM experiment index

## Optional task and surface-representation controls

`preprocess/zinc/zinc_preprocess.yaml` runs acquire/select → preprocess → visualize without requiring
a negative-evidence file. QuickGO experimental NOT annotations and exact UniProt/PDB sequence mappings
provide conservative negatives; a reviewed JSONL may override acquisition. Sparse evidence can still
prevent both-class independent splits; no ligand-absence fallback exists. It publishes a frozen
portable design, universal geometry, Zn
evaluation sidecars and an offline gallery. Selection's `skip`/`existing_design` reuses the exact
saved decision without downloads or clustering. `preprocess/zinc/zinc_discovery.yaml` remains an
optional acquisition-only action; its raw evidence can replace fresh acquisition through `raw_path`.
`preprocess/zinc/validate_zinc.yaml`
independently audits the published scientific contract. No Zn DatasetVersion is bundled or claimed
to have been validated experimentally. See both public README versions, section 3.7, for evidence
requirements, thresholds and limitations.

`preprocess/common/surface_features.yaml` adds optional fixed protein-only fields and per-training-view normalization to
an existing DatasetVersion, without replacing geometry or task annotations.
`ablations/surface_representation/surface_representation_ablation.yaml` compares five configurations: learned control, and explicit
or late-hybrid evidence with generic_minimal or generic_basic fields, using four paired seeds.
The logit-only MAX pooler, backbone configuration, protein-only objective and sealed-test policy remain
fixed. Publish the feature-augmented DatasetVersion first; neither missing inputs nor missing
Registry versions are replaced with fake scientific data. See README sections 4.10 and 5.11.
The secondary `ablations/surface_representation/attention.yaml` repeats the alternatives with
Attention, whose scorer parameter count changes with H/X/Z width; it is not the primary control.
The corrected field release is `wisdom-dna-features@2`; older field sidecars retain their historical
meaning. DNA preparation proposes version 7, while existing training ledgers remain on reduced @6.

These actions are independent of the ordered architectural campaign below. They do not renumber
its stages or imply that the historical Zn dataset is a valid transfer benchmark.

Post-hoc inference-only reviews live separately in [reviews/](reviews/README.md).
They consume registered LF Studies or durable ModelSets, not training orchestration steps.
No automatic winner product is declared for baseline diagnostics.

This directory distinguishes two different meanings that must never be conflated:

- a **construction stage** is an experiment used to decide one part of the first frozen WISDOM
  model; and
- a **scientific version** is a later, coherent model generation that answers a new biological or
  methodological question.

Files use short ordered names (`v1/wisdom_v1a.yaml`, `v1/wisdom_v1b.yaml`, ..., `v10/wisdom_v10.yaml`) so the
campaign is easy to scan. Here `v6a`, for example, is only an experiment-order label. The
`model_version` argument is a separate internal code-capability selector, and neither label claims
that a formal scientific WISDOM generation has already been validated.

## Execution order

| Order | File | Scientific question | Prerequisites | What varies | Decision metric | Next experiment |
|---:|---|---|---|---|---|---|
| 1a | `v1/wisdom_v1a.yaml` | How variable and failure-prone is the untouched baseline? | Dataset registered | Complete run seed | Distribution of `val_wisdom_hpo_score` plus collapse diagnostics | 1b |
| 1b | `v1/wisdom_v1b.yaml` | How much variance comes from training randomness with identical initial weights? | Relevant variance in 1a | Training RNG only | Variance decomposition; no winner | 1c |
| 1c | `v1/wisdom_v1c.yaml` | How much variance comes from initial weights with fixed training randomness? | Same baseline as 1b | Initialization RNG only | Complementary variance control; no winner | 2 |
| 2 | `v2/wisdom_v2.yaml` | Which initialization policy is stable on the unchanged backbone? | 1a–1b reviewed | `initialization_profile` | `val_wisdom_hpo_score`, collapse rate, diagnostics | 3 |
| 3 | `v3/wisdom_v3.yaml` | Which optimizer stabilization complements the selected initialization? | Initialization winner | `optimization_profile` | `val_wisdom_hpo_score`, stability, runtime | 4 |
| 4 | `v4/wisdom_v4.yaml` | Which width, depth, topology, and regularization define the strongest fixed-MAX core? | Initialization and optimizer winners copied into `with` | Core model and AdamW hyperparameters | `val_wisdom_hpo_score` | 5 |
| 5a | `v5/wisdom_v5a.yaml` | How do fixed pooling families behave across their parameter curves? | Existing V5 ledger frozen exactly | Family, point/area measure, applicable fixed parameter | G, S, coupling, regret, seed stability, faithfulness, cost | 5b |
| 5b | `v5/wisdom_v5b.yaml` | Does scalar adaptation improve the selected fixed family reference? | Same frozen ledger and four paired seeds as 5a | Five learned scalars, one selected start/measure each | Same metrics plus parameter trajectories | 5c |
| 5c | `v5/wisdom_v5c.yaml` | Does spatial coherence of evidence before pooling improve localization/classification? | Same frozen V5 ledger; pooling choices currently provisional | Eight refiner types crossed with MAX/LSE/Attention; 54 candidates × 4 seeds = 216 Runs | Refined S, G, raw S/gain, negative controls, ranking, coupling and cost; not W alone | Post-hoc review before adapting 6a |
| 6a | `v6/wisdom_v6a.yaml` | Do curated negatives provide useful weak local supervision? | Pooling winner copied into `with` | `negative_surface_lambda` | `val_wisdom_hpo_score` and surface diagnostics | 6b only if useful, otherwise 7 |
| 6b/6b2/6b3 | `v6/wisdom_v6b.yaml`, `v6/wisdom_v6b2.yaml`, `v6/wisdom_v6b3.yaml` | Does existence, regional existence, or ranking add signal over 6a? | Credible 6a signal | One additional loss family per independent Study, respectively | `val_wisdom_hpo_score` | 6c only if a diagnosed failure remains, otherwise 7 |
| 6c/6c2/6c3 | `v6/wisdom_v6c.yaml`, `v6/wisdom_v6c2.yaml`, `v6/wisdom_v6c3.yaml` | Does an identified map pathology justify cardinality, TV, or Dirichlet regularization? | Reviewed 6a/6b winner and explicit failure mode | One regularizer per independent Study, respectively | `val_wisdom_hpo_score` plus pathology-specific diagnostics | 6d if three families are credible; otherwise 7 |
| 6d | `v6/wisdom_v6d.yaml` | Do the selected negative, regional, and regularization terms cooperate? | One credible family from each of 6a–6c | Three contribution weights jointly | `val_wisdom_hpo_score`, then fresh-seed confirmation | 7 |
| 7 | `v7/wisdom_v7.yaml` | Which global/local head relationship is accurate and faithful? | Selected weak-loss policy copied into `with` | `head_type` | `val_wisdom_hpo_score` and faithfulness | 8 |
| 8 | `v8/wisdom_v8.yaml` | Does one isolated architectural spike beat its proper control? | Head winner copied into `with` | `architecture_spike` | `val_wisdom_hpo_score`, faithfulness, compute | 9 |
| 9 | `v9/wisdom_v9.yaml` | What small numerical retune best fits the selected final architecture? | Complete winner ledger from 2–8 | Four sensitive continuous parameters | `val_wisdom_hpo_score` | 10 |
| 10 | `v10/wisdom_v10.yaml` | Does the frozen V1 result replicate on fresh seeds and held-out test data? | Final ledger from 9 | Fresh complete run seed | Uncensored metric distribution and one test evaluation per seed | Freeze V1 or return to the diagnosed experiment |

Run experiments in this order, but do not run a conditional experiment merely because its YAML exists. Each
file begins with `OBJECTIVE`, `PREREQUISITES`, `VARIES`, `FIXED`, and `DECISION`. Values marked
`REPLACE` are an explicit decision ledger: copy the reviewed previous-stage winner before launch.
This manual freeze is preferable to silently selecting a trial or allowing a downstream YAML to
revert to defaults.

Every training study is a single executable Work: none contains `steps`. Only the
dataset-construction YAML composes sequential actions. Splitting the old 1b, 6b and 6c does not
change their grids or controls; it makes each question independently launchable and reviewable.

## Why V1a, V1b and V1c have no `search`

V1a estimates the baseline distribution. V1b keeps epoch-zero weights identical and varies later
training randomness; V1c varies initial weights while fixing later randomness. These are two
separate studies with complementary controls. Adaptive HPO would censor precisely the failures and
tail behavior these diagnostics measure. Their seed lists therefore expand into complete Runs with
no pruning or winner.

V10 follows the same principle for confirmation: all fresh seeds contribute to the final
variance estimate, so it also has no `search` section.

## Diagnostics, complete sweeps, and adaptive HPO

- **Diagnostics:** V1a, V1b and V1c estimate variance and causation; they select nothing.
- **Complete finite sweeps:** V2, V3, V5a/V5b, V6a–V6c, V7, and V8 use LambdaForge's explicit
  `sweep.space`. V5c also uses a complete conditional sweep. Every authored cell is mandatory.
  LambdaForge adds one complete shared-seed block
  at a time and stops with an anytime-valid paired analysis; HPO pruning and per-cell seed racing
  are disabled by the sweep contract.
- **Adaptive HPO:** V4, V6d, and V9 use `search.goal: optimize` plus `search.space`. Candidate
  generation, seed allocation, curve pruning, convergence, fresh confirmation, and child-Run
  packing remain automatic. The authored `objective.practical_margin` defines which score gaps are
  scientifically negligible.
- **Conditional weak-loss sequence:** V6a is mandatory if weak supervision is considered; V6b
  requires useful V6a signal; V6c requires a diagnosed remaining map pathology; V6d runs only when
  one credible term from each family should be tested jointly.
- **Confirmation:** V10 is not HPO. It freezes the ledger, uses fresh seeds, and opens test.
- **Post-hoc analysis:** `interpretability/interpretability_sparse_concepts.yaml` analyzes one explicit frozen
  checkpoint. It is not a scientific WISDOM version and must not influence V1 selection.

The campaign omits `name`, `execution.max_parallel`, `execution.runs_per_gpu`,
`resources.gpu_memory`, authored search seeds, and controller stopping knobs wherever LambdaForge
0.15's defaults match the intended policy. Filenames supply study names, and
`lf config resolve FILE` exposes every resolved automatic value. `objective.mode: max` remains
explicit because the current execution path still requires it for a mapped custom metric.
V1a/V1b/V1c retain their historical explicit seeds, while V10 retains a deliberately fresh explicit
confirmation set; those identities are scientific inputs rather than scheduler boilerplate.

## V5 family characterization and shortlist

V5c inserts an orthogonal `SurfaceEvidenceRefiner` between raw local logits and the existing
pooler, inside the training forward. It does not change the backbone or add local-GT losses.
The refiner choices are none, fixed/learned heat, geometric/embedding/learned anisotropy,
unrolled graph-TV, and sparse Potts CRF. Only learned heat and learned conductance add parameters.
The unchanged protein BCE differentiates through every operator; embedding guidance detaches
only its guide branch by default. V5c crosses 18 refiner settings with three provisional poolers:
MAX, area-LSE beta=0.25, area-Attention simple/256. This gives 54 candidates × four paired seeds
= 216 Runs. Replace the marked LSE/Attention representatives before a definitive campaign.
No local GT enters training or checkpoint selection; refined metrics, raw diagnostics and unclipped
refinement gain are recorded for development. Existing W remains the objective, not the sole
decision criterion. Vertex-deletion faithfulness is unavailable because it changes refiner operators.
No viewers are generated in the sweep; `visualization/wisdom_v5c.yaml` reviews saved models.
See README section 5.3.1 for all equations, defaults, gradient paths and numerical limitations.

V5a and V5b replace the historical `wisdom_v5.yaml`, now retired from the active campaign.
Their frozen `with` values retain the reviewed V5 ledger, not V4 defaults. V5 never
changes the encoder, data, weak losses, initialization/optimizer policy or head relationship.
Protein BCE plus the existing gate penalty remains the entire training objective; local labels
are development-only and test remains sealed.

Both studies use **exactly [4, 7, 32, 54]** for every candidate. Unlike automatic sweeps in the
other stages, they have a finite paired seed budget with no competitive pruning or seed racing.
Ordinary within-Run validation patience remains enabled. LambdaForge owns scheduling, results,
curves and pairing; no new WISDOM HPO controller is introduced.

| Study / branch | Candidates | Required Runs |
|---|---:|---:|
| V5a MAX / mean / attention / Top-K / regional / LSE | 1 / 2 / 32 / 26 / 16 / 54 | 524 |
| V5a linear softmax / AutoPool / GeM / MAX–MEAN / multiscale regional | 2 / 26 / 24 / 22 / 1 | 300 |
| V5a total: all eleven families | 206 | 824 |
| V5b learned LSE / regional / AutoPool / GeM / MAX–MEAN | 1 each | 4 each |
| V5b total: scalar adaptation from declared V5a references | 5 | 20 |

Each file now defines one Study, with native `when: {parent: {in: [...]}}` membership conditions
for shared area parameters and equality conditions for family-specific/nested scalar settings.
Inactive parameters are absent, not default-valued dimensions. V5a has exactly one MAX reference;
V5b has none because it contains no MAX cell. The reviewed backbone remains unchanged. V5a includes
all implemented families, both attention scorers, fixed scalar grids and prescribed LSE schedules.
V5b isolates learned-scalar adaptation from one declared fixed V5a reference per family. Their scientific
questions remain distinct, with no automatic winner handoff. Fixed sweeps reject
`trials` and `proposal_pool_size`; native planning verifies the complete finite design.

V5a currently requests three GPUs, 42 CPUs, 96 GiB RAM and 1000 hours; V5b requests two GPUs,
36 CPUs, 96 GiB RAM and 168 hours, with native automatic packing. Their dispatch budgets are
1000 and 168 hours respectively. These budgets do not extend the authored scheduler limit or
guarantee complete coverage. No packing/pruning compatibility helpers
are required. Shared research semantics come from Training.analysis_profile, with small YAML
metric-priority overrides applied to the entire Study.

V5a intentionally sets the curriculum hold to zero only on curriculum cells. With epochs=500
and patience=30, the previous 0.30 hold delayed the first beta change to epoch 152. The new schedule
changes at epoch 2, while retaining ordinary patience; endpoint 1 is the deliberate constant-beta
control. Other endpoints are still not guaranteed before early stopping. Inspect the recorded beta
trajectory and restored checkpoint, not only the target.

The fixed screen extends Top-K from 0.0005 through 1.0 (including 0.6 and 0.8), attention widths
from 4 through 512, regional length from 0 through 64 angstroms, LSE beta from 0.01 through 1280,
AutoPool alpha from 0 through 1000, GeM power from 1 through 1024, and MAX–MEAN weight across
the full [0,1] interval. Multiscale regional pooling keeps its existing fixed length bank
[0,1.5,3,6,12] angstroms and learned mixture; no artificial parameter is added to this family.
V5b uses one start per family: beta=0.25 (area), length=16 Å, alpha=500 (point), power=2 (point),
and MAX–MEAN coefficient=0.95 (point). These are learned starts, not fixed reductions.
Its explicit domains are beta [0.005,2560], length [0.05,128] Å, alpha [0,2000],
power [1,2048] and mixture [0,1]. Constructor defaults remain unchanged for historical weights;
new checkpoints retain the exact bounds in model_parameters. The comparison reuses V5a controls
and starts the backbone from paired random seeds, not from V5a weights. It does not prove that
learning from arbitrary initializations removes the need for HPO.

If quality still improves at a finite grid edge, the range is **not closed**: compare neighboring
levels over paired seeds and propose a separate extension, never claim the edge is an optimum.
Small Top-K fractions may select the same number of points; high beta/alpha/power can concentrate
gradients on a few points. Report these controls and their runtime/local-quality tradeoffs. Zero
regional length is exact identity, but positive lengths use the existing truncated spectrum;
even 64 angstroms does not imply whole-protein averaging across disconnected components.

Review each family's fixed/learned representatives with paired seed variation, global score G,
surface score S, coupling, selection regret, subgroup/faithfulness diagnostics and runtime.
Missing regional deletion diagnostics are unavailable, not zero: deleting vertices would change
their diffusion operator. Learned scalar values and multiscale weights have epoch metrics and
`evaluation.json` trajectories; final values refer to the selected global checkpoint, not merely
the last optimizer update. Area-aware pooling is an ablation, not an assumed improvement.

The output is a reviewed shortlist of approximately **2–4 nondominated families**, not a single
`pooling_winner`. Account for uncertainty: four seeds do not prove a ranking. Keep V6a–V6d's
current loss spaces unchanged; revising them to cover the shortlist is a subsequent task, not an
implicit change in this implementation. Their current single-pooling ledger must therefore be
reviewed before launching V6.

## Protein reports

Configure presentation directly in each training YAML's `with.visualization` mapping.
V1a/V1b/V1c and V10 request `mode: report`; other screening studies retain `mode: none` to
avoid exceeding LF's combined 64 MiB report limit. NONE disables the tab completely.

The default `content: predictions` embeds only probabilities, original logits and available
soft/hard GT. `maximum_points: 2000` bounds the displayed cloud; `maximum_proteins: 4`
selects a balanced per-split sample and `report_budget_mib: 4.0` caps the document.
Metrics and optional numerical exports retain all points and full precision.
`content: full` restores the structural inspector. `mode: viewer` adds standalone HTML/PLY;
`mode: full` also writes prediction NPZ. See both READMEs for every option/default.

Reports are generated once after restoring each completed Run's best protein-selected checkpoint,
not every epoch. Numerical surface metric frequency remains independent. Failed Runs cannot
publish unfinished reports; failed Studies may still contain reports from successful seeds.
Historical exported HTML is not rewritten by changing configuration.

Separate VVAL YAMLs have been removed. There is no verified public LF post-Study callback to
forward only a final winning candidate/seed. Do not implement a project-owned ranker, hidden
runner, seed cleanup process or training `steps` to imitate that unsupported lifecycle.

## Formal scientific roadmap after V1

The following are future decision-gated generations, not completed campaign stages:

1. **V2 — multi-view signal audit:** use the existing correspondence and consistency machinery to
   measure oracle headroom, aggregation gain, and disagreement as an error signal. It also requires
   a shortcut/identifiability audit and invariance to surface discretization.
2. **V3 — view-robust training:** sample physically equivalent views during training only if V2
   shows useful signal rather than unstructured variance.
3. **V4 — formal pooling comparison:** repeat pooling on the frozen backbone, including a future
   area-aware probabilistic link candidate.
4. **V5 — surface-encoder comparison:** compare DiffusionNet and alternative encoders while every
   preceding V1 decision remains frozen.
5. **V6 — SSL plus protein-specific test-time training:** proceed only after an SSL loss predicts
   surface quality or cross-view consistency without local targets.
6. **V7 — recurrent TTC:** proceed only after a non-recurrent refinement operation is useful and
   stable, then test shared-weight recurrent refinement.

The implemented `SurfaceViewCorrespondence` and `ViewConsistencyMetricSuite` are infrastructure for
future V2. No current YAML exercises them because the view-generation contract and frozen V1 do not
yet exist. Likewise, implemented alternative surface encoders are capability code, not evidence that
formal V5 has been run.

## Validation

Validate an individual stage before submission:

```bash
lf validate experiments/v4/wisdom_v4.yaml
lf explain experiments/v4/wisdom_v4.yaml
lf run experiments/v4/wisdom_v4.yaml --dry-run
```

The dataset selector must resolve on the target cluster. The interpretability YAML additionally
requires a real reviewed `best-model.pt`; never create a placeholder checkpoint for an actual run.
Use `lf config resolve experiments/v4/wisdom_v4.yaml` to inspect the exact automatic policies before
submission; omitted execution ceilings mean that ARI derives safe concurrency from allocated
CPUs, GPUs, live GPU memory, and measured Run envelopes.
