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
| 5 | `wisdom_v5.yaml` | Is MAX clearly inferior to another available aggregation family? | Core winner copied into `with` | Pooling and family-specific parameters | `val_wisdom_hpo_score` and faithfulness | 6a |
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
- **Complete finite sweeps:** V2, V3, V5, V6a–V6c, V7, and V8 retain LambdaForge's concurrent
  adaptive runner but set `min_seeds` equal to the complete seed list, disable curve pruning, and
  disable winner-only confirmation. Every authored candidate therefore completes four paired seeds
  (five in V8). This avoids the serial execution of `strategy: exhaustive` without censoring data.
- **Adaptive HPO:** V4, V6d, and V9 vary several parameters jointly. They deliberately keep
  probability-based pruning, seed racing, and fresh-seed confirmation.
- **Conditional weak-loss sequence:** V6a is mandatory if weak supervision is considered; V6b
  requires useful V6a signal; V6c requires a diagnosed remaining map pathology; V6d runs only when
  one credible term from each family should be tested jointly.
- **Confirmation:** V10 is not HPO. It freezes the ledger, uses fresh seeds, and opens test.
- **Post-hoc analysis:** `interpretability_sparse_concepts.yaml` analyzes one explicit frozen
  checkpoint. It is not a scientific WISDOM version and must not influence V1 selection.

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
