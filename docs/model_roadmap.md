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
isolated architectural spikes, retunes only sensitive numerical parameters, and finally confirms
the frozen ledger on fresh seeds.

V1 is frozen only after `wisdom_v10.yaml` completes without unacceptable collapse,
surface failure, faithfulness failure, or seed variance. Code that exposes a candidate is not by
itself evidence that the candidate belongs to V1.

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
