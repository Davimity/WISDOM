# WISDOM experiment index

This directory distinguishes two different meanings that must never be conflated:

- a **construction stage** is an experiment used to decide one part of the first frozen WISDOM
  model; and
- a **scientific version** is a later, coherent model generation that answers a new biological or
  methodological question.

Files use short ordered names (`wisdom_v1a.yaml`, `wisdom_v1b.yaml`, ..., `wisdom_v10.yaml`) so the
campaign is easy to scan. Here `v6a`, for example, is only an experiment-order label. The
`model_version` argument is a separate internal code-capability selector, and neither label claims
that a formal scientific WISDOM generation has already been validated.

## Execution order

| Order | File | Scientific question | Prerequisites | What varies | Decision metric | Next experiment |
|---:|---|---|---|---|---|---|
| 1a | `wisdom_v1a.yaml` | How variable and failure-prone is the untouched baseline? | Dataset registered | Complete run seed | Distribution of `val_wisdom_hpo_score` plus collapse diagnostics | 1b |
| 1b | `wisdom_v1b.yaml` | Does variance mainly originate in parameter initialization or later training randomness? | Relevant variance in 1a | One RNG source at a time | Variance decomposition; no winner | 2 |
| 2 | `wisdom_v2.yaml` | Which initialization policy is stable on the unchanged backbone? | 1a–1b reviewed | `initialization_profile` | `val_wisdom_hpo_score`, collapse rate, diagnostics | 3 |
| 3 | `wisdom_v3.yaml` | Which optimizer stabilization complements the selected initialization? | Initialization winner | `optimization_profile` | `val_wisdom_hpo_score`, stability, runtime | 4 |
| 4 | `wisdom_v4.yaml` | Which width, depth, topology, and regularization define the strongest fixed-MAX core? | Initialization and optimizer winners copied into `with` | Core model and AdamW hyperparameters | `val_wisdom_hpo_score` | 5 |
| 5a | `wisdom_v5a.yaml` | How do fixed pooling families behave across their parameter curves? | Existing V5 ledger frozen exactly | Family, point/area measure, applicable fixed parameter | G, S, coupling, regret, seed stability, faithfulness, cost | 5b |
| 5b | `wisdom_v5b.yaml` | Do learned parameters or additional families provide competitive representatives? | Same frozen ledger and four paired seeds as 5a | Family-specific adaptation and point/area measure | Same metrics plus parameter trajectories | Shortlist review before adapting 6a |
| 6a | `wisdom_v6a.yaml` | Do curated negatives provide useful weak local supervision? | Pooling winner copied into `with` | `negative_surface_lambda` | `val_wisdom_hpo_score` and surface diagnostics | 6b only if useful, otherwise 7 |
| 6b | `wisdom_v6b.yaml` | Does existence, regional existence, or ranking add signal over 6a? | Credible 6a signal | One additional loss family per independent step | `val_wisdom_hpo_score` | 6c only if a diagnosed failure remains, otherwise 7 |
| 6c | `wisdom_v6c.yaml` | Does an identified map pathology justify cardinality, TV, or Dirichlet regularization? | Reviewed 6a/6b winner and explicit failure mode | One regularizer family per independent step | `val_wisdom_hpo_score` plus pathology-specific diagnostics | 6d if three families are credible; otherwise 7 |
| 6d | `wisdom_v6d.yaml` | Do the selected negative, regional, and regularization terms cooperate? | One credible family from each of 6a–6c | Three contribution weights jointly | `val_wisdom_hpo_score`, then fresh-seed confirmation | 7 |
| 7 | `wisdom_v7.yaml` | Which global/local head relationship is accurate and faithful? | Selected weak-loss policy copied into `with` | `head_type` | `val_wisdom_hpo_score` and faithfulness | 8 |
| 8 | `wisdom_v8.yaml` | Does one isolated architectural spike beat its proper control? | Head winner copied into `with` | `architecture_spike` | `val_wisdom_hpo_score`, faithfulness, compute | 9 |
| 9 | `wisdom_v9.yaml` | What small numerical retune best fits the selected final architecture? | Complete winner ledger from 2–8 | Four sensitive continuous parameters | `val_wisdom_hpo_score` | 10 |
| 10 | `wisdom_v10.yaml` | Does the frozen V1 result replicate on fresh seeds and held-out test data? | Final ledger from 9 | Fresh complete run seed | Uncensored metric distribution and one test evaluation per seed | Freeze V1 or return to the diagnosed experiment |

Run experiments in this order, but do not run a conditional experiment merely because its YAML exists. Each
file begins with `OBJECTIVE`, `PREREQUISITES`, `VARIES`, `FIXED`, and `DECISION`. Values marked
`REPLACE` are an explicit decision ledger: copy the reviewed previous-stage winner before launch.
This manual freeze is preferable to silently selecting a trial or allowing a downstream YAML to
revert to defaults.

## Why V1a and V1b have no `search`

V1a estimates the baseline distribution. V1b contains two complementary controls: its first step
keeps epoch-zero weights identical and varies later training randomness; its second step varies the
initial weights while fixing later randomness. Adaptive HPO would censor precisely the failures and
tail behavior these diagnostics measure. Their seed lists therefore expand into complete Runs with
no pruning or winner.

V10 follows the same principle for confirmation: all fresh seeds contribute to the final
variance estimate, so it also has no `search` section.

## Diagnostics, complete sweeps, and adaptive HPO

- **Diagnostics:** V1a and V1b estimate variance and causation; they select nothing.
- **Complete finite sweeps:** V2, V3, V5a/V5b, V6a–V6c, V7, and V8 use LambdaForge's explicit
  `sweep.space`. Every authored cell is mandatory. LambdaForge adds one complete shared-seed block
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
- **Post-hoc analysis:** `interpretability_sparse_concepts.yaml` analyzes one explicit frozen
  checkpoint. It is not a scientific WISDOM version and must not influence V1 selection.

The campaign omits `name`, `execution.max_parallel`, `execution.runs_per_gpu`,
`resources.gpu_memory`, authored search seeds, and controller stopping knobs wherever LambdaForge
0.15's defaults match the intended policy. Filenames supply study names, and
`lf config resolve FILE` exposes every resolved automatic value. `objective.mode: max` remains
explicit because the current execution path still requires it for a mapped custom metric.
V1a/V1b retain their historical explicit seeds, while V10 retains a deliberately fresh explicit
confirmation set; those identities are scientific inputs rather than scheduler boilerplate.

## V5 family characterization and shortlist

V5a and V5b replace the historical `wisdom_v5.yaml`, retained only as a clearly marked legacy
screen. Their frozen `with` values come from the existing V5 file, not V4 defaults. V5 never
changes the encoder, data, weak losses, initialization/optimizer policy or head relationship.
Protein BCE plus the existing gate penalty remains the entire training objective; local labels
are development-only and test remains sealed.

Both studies use **exactly [4, 7, 32, 54]** for every candidate. Unlike automatic sweeps in the
other stages, they have a finite paired seed budget with no competitive pruning or seed racing.
Ordinary within-Run validation patience remains enabled. LambdaForge owns scheduling, results,
curves and pairing; no new WISDOM HPO controller is introduced.

| Study / branch | Candidates | Required Runs |
|---|---:|---:|
| V5a MAX / mean / attention / Top-K / regional / LSE | 1 / 2 / 10 / 16 / 9 / 18 | 224 total |
| V5b gated attention | 8 | 32 |
| V5b adaptive LSE | 18 | 72 |
| V5b learned regional | 4 | 16 |
| V5b linear softmax | 2 | 8 |
| V5b AutoPool | 16 | 64 |
| V5b GeM | 18 | 72 |
| V5b MAX–MEAN | 12 | 48 |
| V5b multiscale regional | 1 | 4 |
| V5b total | 79 | 316 |

Each file now defines one Study, with native `when: {parent: {in: [...]}}` membership conditions
for shared area parameters and equality conditions for family-specific/nested scalar settings.
Inactive parameters are absent, not default-valued dimensions. V5a has exactly one MAX reference;
V5b has none because it contains no MAX cell. The reviewed backbone and all candidate grids remain
unchanged. V5a studies fixed rules; V5b studies adaptive rules and additional controls. Their
scientific questions remain distinct, with no automatic winner handoff. Fixed sweeps reject
`trials` and `proposal_pool_size`; native planning verifies the complete finite design.

Both request two GPUs, 36 CPUs, 96 GiB RAM, and a 168-hour enclosing scheduler ceiling, with native
automatic packing. The old six/eight sequential allocations summed to 42/56-day ceilings.
`execution.max_time: 168h` separately stops dispatching new Runs after seven days; it does not
extend scheduler time or guarantee complete coverage. No packing/pruning compatibility helpers
are required. Shared research semantics come from Training.analysis_profile, with four YAML
metric-priority overrides applied to the entire Study.

V5b intentionally sets the curriculum hold to zero only on curriculum cells. With epochs=500
and patience=30, the previous 0.30 hold delayed the first beta change to epoch 152. The new schedule
changes at epoch 2, while retaining ordinary patience; its endpoint is still not guaranteed before
early stopping. Inspect the recorded beta trajectory and restored checkpoint, not only the target.

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
lf validate experiments/wisdom_v4.yaml
lf explain experiments/wisdom_v4.yaml
lf run experiments/wisdom_v4.yaml --dry-run
```

The dataset selector must resolve on the target cluster. The interpretability YAML additionally
requires a real reviewed `best-model.pt`; never create a placeholder checkpoint for an actual run.
Use `lf config resolve experiments/wisdom_v4.yaml` to inspect the exact automatic policies before
submission; omitted execution ceilings mean that ARI derives safe concurrency from allocated
CPUs, GPUs, live GPU memory, and measured Run envelopes.
