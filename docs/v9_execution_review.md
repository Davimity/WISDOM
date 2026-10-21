# V9 execution review after V8

## 1. Evidence and comparison rules

This review uses the completed V8 audits available on 2026-10-21 in the researcher's
`TFM-Results/Memoria/docs/` directory: `auditoria_v8a.md`, `auditoria_v8b.md`,
`auditoria_v8c.md`, and `v8d_resultados.md`, together with their aggregate/contrast tables
under `data/experimental/v8a-final` through `v8d-final`. These external results are not copied
into WISDOM or treated as executable configuration inputs.

Here **S** is the mean surface precision–recall area across eligible positive validation
proteins, evaluated at the epoch selected by protein-level validation utility **G**. S measures
localization, whereas G measures protein classification. A change of +0.01 is an absolute
one-percentage-point change in AUPRC, not a relative 1% improvement. Each contrast pairs the
same four development seeds, 151/157/163/167. The quantities after ± below are sample standard
deviations of the four paired changes, **not confidence intervals**. Small samples and many
comparisons prevent a strong superiority claim from a positive mean alone.

The two policies are MAX R Single and attention/FiLM. They differ in pooling, head and weak
losses, not just pooling. Their responses cannot establish a pooling-only causal effect.
Comparisons below stay within the same policy and campaign; V8d's feedback control is not
silently substituted for the original one-pass V8b baseline.

## 2. What V8 supports

| Mechanism and paired control | Change in S, MAX | Change in S, attention/FiLM | Interpretation |
|---|---:|---:|---|
| V8a geometry-conditioned transfer versus original transfer | −0.0008 ± 0.0206 | +0.0020 ± 0.0091 | No convincing gain; peak allocated VRAM increases by about 2.10/1.84 GiB. |
| V8b real return versus matched null return | +0.0015 ± 0.0309 | +0.0013 ± 0.0142 | Little evidence that the returned information, rather than the extra processing stage, accounts for improvement. |
| V8c eight-channel late vector augmentation versus scalar baseline | −0.0043 ± 0.0226 | +0.0161 ± 0.0218 | A potentially useful attention-specific hypothesis, not a universal improvement. |
| V8d two post-return graph layers versus matched null return with the same layers | +0.0041 ± 0.0169 | −0.0118 ± 0.0291 | No consistent advantage specific to the real return. |

V8b's second surface stage raises measured peak allocated memory from about 16 GiB to about
30 GiB; that is the cost of the two-stage route, not of real rather than null messages alone.
In V8d, MAX improves by +0.0408 when two graph layers are added after real return, but the
matched null route also improves by +0.0366 relative to the same no-propagation control.
The real-versus-null difference is much smaller, and attention does not show the same pattern.
This supports investigating processing depth/capacity; it does not prove return is intrinsically
useless or identify which graph/surface operation is responsible.

V8c is the relatively inexpensive exception worth remembering: eight vector channels add about
0.80 GiB for MAX and 0.53 GiB for attention. Attention's surface improvement occurs in three
of four seeds, while its mean G change is −0.0050. The mean MAX surface change is negative,
and increasing to sixteen channels does not improve either policy monotonically. Without a
parameter-matched scalar control and fresh-seed confirmation, do not adopt this augmentation
as V9's baseline. V9e's persistent vector encoders are a different experiment from V8c's late
augmentation and remain in the planned comparison.

Recorded epoch durations come from shared hardware. They describe those executions, not
isolated intrinsic encoder latency. Memory, support counts, convergence and prediction quality
must remain separate from wall-clock comparisons.

## 3. Configuration decision

All six V9 training YAMLs retain the reviewed MAX R Single / Heat6 ledger and change only the
declared atomic-encoder profile. They explicitly select learned surface input, DiffusionNet,
the baseline architecture, no point-geometry conditioning, one interaction pass, zero V8c
augmentation channels, and no post-feedback propagation. The original dataset, surface budgets,
optimizer, losses, scalar interface and G-based checkpoint rule remain unchanged. V9e's own
vector channels are not disabled by zeroing the separate V8c option.

Keep the initial broad comparison at **43 campaign cells × 4 seeds = 172 required Runs**,
including each campaign's contemporary controls. Automatically crossing both policies would
double this to 344 Runs and change more than the atomic encoder. A later small, explicitly
reviewed attention/FiLM comparison should include its own unchanged baseline and a declared
shortlist of operators. Choose that shortlist using localization, global quality, stability and
feasibility, not only the largest mean score. This screening strategy can miss operators that
help only attention; a universal cross-policy claim would require the full comparison or a
separately justified interaction study. The present conclusions must therefore stay scoped to MAX.

Each Study requests **2 GPUs, 36 CPUs and 96 GiB host RAM in total**. `execution` is omitted,
so native LambdaForge chooses admission and packing rather than serializing Runs through
`max_parallel: 1` and `runs_per_gpu: 1`. RAM/CPU are study-wide resources, not replicated per
child. Removing the limits does not guarantee that every candidate overlaps or fits in VRAM.
The explicit V9f dense-attention opt-in, per-protein ownership and byte/triplet guards remain;
an unsupported size must fail clearly, never shrink the scientific batch or replace an operator.
Concurrent timings must not be presented as isolated architecture benchmarks.

Only the researcher launches Studies. Configuration validation, explanation, dry-run planning
and synthetic tests establish executability, not scientific improvement or measured hardware fit.

## 4. Technical verification

The six production configurations pass `lf validate` and `lf explain`, and the public
`WorkRunner.run(..., dry_run=True)` plans the expected 40/28/36/24/20/24 Runs with their actual
dataset binding. The CLI dry-run directed to `local` stops at capacity admission because this
PC exposes one GPU while these Studies request two. This is not a measured cluster-capacity or
VRAM-fit check. No job was submitted.

The full unit suite reports 904 passed and four skipped: one requires the generated production
benchmark, and three require unavailable Playwright browser tooling. Ruff and mypy pass. Tests
also freeze the six Studies' shared scientific ledger, disabled V8 routes, automatic execution
policy, resources and paired seed counts. Both root README tables of contents match their
44 numbered sections and share the same hierarchy.
