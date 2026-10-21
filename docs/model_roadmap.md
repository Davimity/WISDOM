# WISDOM scientific model roadmap

This document describes scientific model generations. Campaign filenames such as `wisdom_v6d.yaml`
are compact execution-order labels, not declarations of separate scientific generations. The
authoritative construction order is
[`experiments/README.md`](../experiments/README.md).

`model_version` is an internal code-capability selector used by `Training` to instantiate compatible
Python classes. It is not an experimental stage number and is not evidence that the corresponding
scientific generation has been selected, trained, or validated.

## 1. WISDOM V1 — frozen atom-to-surface baseline

V1 is still being constructed. Its current backbone uses a bounded relation-aware atomic graph,
learned atom-to-surface transfer, existing multiscale surface geometry, and intrinsic DiffusionNet
propagation. The baseline protein score uses MAX multiple-instance pooling over surface logits. It
does **not** use the retired dense surface graph, a surface GCN, or area-weighted mean pooling.

The construction campaign measures seed variance, separates initialization from later randomness,
selects initialization and optimizer policies, restores a broad core HPO, checks pooling, evaluates
weak-loss families separately and then jointly when justified, compares head relationships, tests
isolated architectural spikes, compares atomic encoders in V9a–V9f, and finally confirms
the frozen ledger on fresh seeds.

V1 is frozen only after `wisdom_v10.yaml` completes without unacceptable collapse,
surface failure, faithfulness failure, or seed variance. Code that exposes a candidate is not by
itself evidence that the candidate belongs to V1.

### 1.1. Construction campaigns V8a, V8b and V8c

These labels are construction experiments, not formal model generations. Each independently
crosses its architectures with both reviewed V7 lines: MAX R Single and Attention NRK FiLM.
The line includes its exact pooling, priors and head; no cross-line combinations are permitted.
V8a tests existing point-geometry conditioning (16 Runs). V8b compares one-pass baseline, added
surface depth, parameter-matched null-message update and real surface-to-atom feedback (32 Runs).
V8c tests one scalar→equivariant-vector→invariant-norm atomic augmentation with 8/16 channels
(24 Runs). Each uses paired seeds 151/157/163/167 and contemporary controls, without HPO/pruning.
The complete design is 72 Runs; neither Study nor line waits for another to finish.

The primary feedback contrast is bidirectional vs depth_control, not vs baseline. All two-stage
variants share the second transfer/projection/encoder. The matched update never reads S1 and
matches nominal parameters, not effective input-dependent capacity or complete aggregation cost.
There is no new atomic message-passing round, recurrent TTC, preprocessing or persistent deep
vector encoder. The combined former YAML/reviewer survive as non-executable .yaml.example
snapshots; use the three separate validation-only checkpoint reviewers instead.

The reviewed coefficients now use custom weak losses; no pending-winner placeholder remains.
Native references are unique, but scientific comparisons use the SAME line and SAME seed.
The controlled behavior policy isolates new
module RNG, fixes uncovered feedback atoms and undirected-vector symmetry, and keeps the
baseline gate-cost denominator fixed. Historical checkpoints default to legacy behavior.
See README section 5.5.2 for equations, defaults, limitations and the scientific comparisons.

The V9 construction studies replace the former numerical-retuning template. Six independent fixed
sweeps compare classical propagation, edge-conditioned messages, multi-statistic aggregation,
radial/directional models, persistent equivariant states and sparse/global attention. They use
public LambdaForge operators, one reviewed MAX R Single / Heat6 context and four new paired seeds
per profile (172 Runs including contemporary controls). They add no preprocessing, surface
supervision or implicit V8 winner. Only V9f permits bounded global dense attention through an
explicit researcher-approved opt-in. Generic degree/path structural attention is not claimed to
reproduce Graphormer's original path-edge recipe. See root README §5.5.4 for defaults, guards,
checkpoint compatibility and the scientific follow-up phases that still require real evidence.
The completed V8 audit review retains the one-pass MAX baseline for the initial V9 screen;
attention/FiLM remains a possible small follow-up, not an automatic second policy axis.
All six training Studies request 2 GPUs, 36 CPUs and 96 GiB total host RAM with native automatic
concurrency. See [the evidence and execution decision](v9_execution_review.md).

V8d separately implements the missing post-feedback atomic communication, without adopting V9
operators or changing the V8b evidence. D0–D3 reuse the old routes; D4/D5 add one/two independent
hidden-state graph layers; D6 is the matched null-feedback control; D7 skips the feedback MLP;
D8 adds local capacity without edges. Its 32-Run main plan freezes the exact two-policy V8b
ledger and 4+4 surface budget. Identity-start scaling, compact surface budgets and activation
recomputation are explicit options. One feedback cycle is supported, not recurrent TTC or a
second unverified cycle. Completed four-seed V8d results in the researcher's report improve MAX
after added propagation, but the matched null route improves similarly; attention does not show
the same benefit. This is not sufficient evidence to adopt feedback in V9. Fresh-seed confirmation
remains pending. See README §5.5.3, the V8d implementation audit and the execution review above.

## 2. WISDOM V2 — multi-view signal audit

V2 asks whether physically equivalent surface samplings contain complementary information or only
numerical noise. Before training a view-robust model, the audit must establish:

- point correspondence across views where correspondence is physically meaningful;
- prediction correlation and disagreement after matching points;
- oracle headroom and the gain from aggregating multiple views;
- whether disagreement predicts global or local error;
- invariance to surface discretization density; and
- a shortcut/identifiability audit showing that performance does not depend on view-specific
  artifacts, identifiers, mesh density, or another accidental cue.

`SurfaceViewCorrespondence` and `ViewConsistencyMetricSuite` already implement part of the audit.
They are infrastructure not yet exercised by a public experiment because the frozen V1 checkpoint
and formal multi-view dataset contract do not yet exist.

## 3. WISDOM V3 — view-robust training

V3 samples or pairs physically equivalent views during training. It proceeds only if V2 shows a
measurable, interpretable signal: useful aggregation gain, meaningful oracle headroom, or
disagreement that predicts errors. The primary question is whether training can absorb genuine
discretization invariance without averaging away compact binding regions.

V3 must keep the frozen V1 chemical backbone and evaluation protocol fixed. Any consistency loss is
first validated as a proxy; it does not receive surface ground truth and cannot be justified merely
because it decreases during optimization.

## 4. WISDOM V4 — formal pooling comparison

The construction campaign contains a pooling sanity check because MAX may be an unusable baseline.
Formal V4 repeats the comparison after V1 is frozen and treats pooling as the sole scientific
factor. Candidates include MAX, mean, learned attention, top-k mean, local-mean-max, normalized
LogSumExp, and a future **area-aware probabilistic link** whose independence assumptions and
numerical behavior must be specified before implementation.

Family-specific hyperparameters remain conditional. A top-k fraction must not create duplicate
mean/MAX candidates, and a LogSumExp temperature must not affect other pooling families.

## 5. WISDOM V5 — surface-encoder comparison

V5 changes only surface propagation on the frozen prior backbone. DiffusionNet is compared with
implemented candidate encoders such as dMaSIF-like, DeltaConv, Point Transformer V3, and
PointMamba-style controls under matched input evidence and compute reporting.

These encoder classes already exist, but no current construction-stage YAML claims they have been
scientifically compared. The former pre-freeze encoder experiment was removed because it reverted
pooling, weak loss, stability, and head decisions instead of carrying their winners forward.

## 6. WISDOM V6 — SSL and protein-specific test-time training

V6 introduces self-supervision (SSL) only after an SSL objective is shown to predict surface quality
or cross-view consistency on validation data. Protein-specific test-time training (TTT) then adapts
within calibrated limits while monitoring probability change, uncertainty, and failure controls.

An SSL loss decreasing is insufficient evidence. The stage requires a no-adaptation control,
correlation with the intended biological metrics, bounded parameter change, and an explicit rollback
criterion for harmful adaptation.

## 7. WISDOM V7 — recurrent TTC

V7 tests shared-weight recurrent refinement only after one non-recurrent refinement operation has
already improved the frozen model. It compares trained unroll counts, stability across extra test
iterations, convergence behavior, and compute. Surface-to-atom feedback or another refinement must
not be made recurrent merely because the implementation permits repeated calls.

## 8. Cross-generation requirements

Every formal generation must preserve leakage-safe train/validation/test partitions, use repeated
paired seeds, keep held-out test data outside model selection, report global and surface metrics,
measure global–surface coupling and regret, and record compute. Technical executability is not
scientific validation.

The following capabilities remain post-hoc or conditional rather than numbered generations:

- sparse-concept discovery analyzes one frozen checkpoint through
  `interpretability_sparse_concepts.yaml`;
- faithfulness deletion/insertion audits are evaluation tools, not training objectives;
- alternative surface encoders remain dormant until formal V5;
- multi-view consistency remains dormant until formal V2 data exist; and
- μTransfer is deferred until width scaling is large enough to justify a separate transfer study.
