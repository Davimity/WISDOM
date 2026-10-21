# V8d technical audit: atomic propagation after surface feedback

## 1. Scope and evidence inspected

This change implements a finite, one-feedback-cycle architecture study. It does not claim a
predictive improvement, launch a scientific Work, change V8b results, introduce V9 encoders,
or add local supervision, TTC, recurrence, a new pooling or another checkpoint selector.

The audit read the current WISDOM/LambdaForge contributor contracts; the native Work, seed,
checkpoint, evidence and recovery contracts; `ArchitectureSpike`, `InteractionRound`,
`GatedAtomicEncoder`, `SurfaceAtomFeedback`, `SurfaceAtomTransfer`, `WisdomV1`/`WisdomV2`,
`DiffusionSurfaceEncoder`/`DiffusionBlock`, `WisdomCollator`, `Training`, and the existing V8,
transfer and encoder tests. Existing dirty V8/V9 work and active remote allocations were preserved.
No Job was submitted, cancelled, retried or otherwise changed. LambdaForge source remains read-only.

The external thesis sources were read, not modified:

- section 7.8.3: `Memoria/chapters/experiments/architecture_design.typ`, including
  `surface_feedback_method.typ` and `surface_feedback_results.typ`;
- closing appendix B.25: `Memoria/chapters/appendix_resultados.typ`, “Cierre de retorno superficial V8b”;
- exported V8b `execution-29c7fa9079512348`, its submitted configuration, run/result metadata,
  and `Memoria/data/experimental/v8b-final` audit records.

The current V8b YAML's entire `with` mapping equals the exported mapping. Both policies are
preserved: MAX/point/Single/R=.3 and simple Attention-256/area/FiLM/N=R=K=.03, normalized .5/.5
protein BCE, copied local initialization, context width 16, detached context and zero head dropout.
Common values are H=128, element/residue widths 32/64, initial atomic depth 2, projection depth 3,
surface depth 4 per stage, spatial K=8, transfer J=16/radius 4 Å, spectral budget 32, Heat6,
AdamW .001/decay .0005, dropout .2, gate coefficient .04, batch 16, epochs 500, patience 30,
minimum G delta .001, auto precision and seeds 151/157/163/167. The resolved historical constructor
has `curvature_features=15`: **five** stored scales, not the two-scale toy unit-test default.

The thesis closes V8b without a consistent surface benefit over depth/null controls; the small
Attention global signal is not evidence for V8d. It reports about 30 GiB peak training VRAM for
the full real-feedback variants under shared hardware. Those historical observations are neither
new V8d measurements nor a guarantee that V8d fits. Historical Runs must not be counted as
contemporary new replications.

## 2. Architecture and compatibility

The route is GNN1 → transfer/projection/DiffusionNet1 → S1 → existing U(A1,m) → GNN2 →
second transfer/projection([S1,C2])/DiffusionNet2 → existing final heads/refiner/pooling/losses.
U retains the area*exp(-distance/radius) mean, valid-candidate coverage, and exact preservation
of uncovered atoms. GNN2 may then influence an uncovered atom through graph neighbours.

`GatedAtomicEncoder.propagate` exposes the unchanged historical message loop. Normal forward
still applies its original input projection first. The new GNN2 uses `project_inputs=False`,
so its input projection is Identity and it consumes N×H hidden states directly. There is no
duplicate GNN equation, recategorization, topology construction or weight sharing with GNN1.
Pairs belonging to both relations contribute to both. All stages share one semantic gate sample.

`FeedbackPropagationMode` declares none/D0–D8. D0–D3 map to existing interaction routes, with
no new trainable state keys. D4 uses real feedback plus one layer; D5 real plus two; D6 null plus
two; D7 two layers with no U; D8 real plus nodewise H→2H→H SiLU capacity. Null means a real H-zero
message with original coverage, not a detached real summary. S1 remains in every second projection.

The graph transform F already contains LayerNorm/SiLU, so stable integration explicitly uses
`A + gamma*(F(A,E)-A)`. No normalization follows. Gamma is learned, initially .1; zero is exact
identity and initially blocks F's internal gradients while allowing gamma to learn. The D8
control instead uses `A + gamma*MLP(A)`. Separate private construction RNG streams preserve
common GNN1, transfer, surface and head initialization. D5 and D6 have identical nominal parameters.
Only final existing heads are supervised. No local target, intermediate BCE, loss, GT-dependent
feature or new detach enters the path. FiLM's original global-context stop-gradient remains.

`surface_stage_layers=None` preserves the original surface depth in both stages. Explicit
[2,2], [3,1], [1,3] create separately labelled budget comparisons. They are never automatic OOM
responses. [4,4] is the main reference. A second feedback cycle is intentionally deferred:
there is no real one-cycle feasibility evidence yet. No unbounded iteration API exists.

## 3. Resource and lifecycle controls

Optional `checkpoint_surface` and `checkpoint_post_feedback` use PyTorch non-reentrant activation
recomputation with preserved RNG. They keep every gradient; the semantic gates are sampled outside
the recomputed blocks. Numerical tests compare outputs and all parameter gradients under dropout.
The same per-protein physical operator mappings and candidate tensors are reused; only a temporary
forward-scoped container is added for the shared derived sparse gradient. No embeddings survive
an optimizer step as a cache. No extra graph or dense pair-distance matrix is materialized.

The existing Training Work logs post-feedback parameter count and learned gamma, and retains
epoch timings, support counts, original/operational S, global metrics, W, regret, coupling,
enrichment, negative evidence, selected epoch, training memory and optimization diagnostics.
Checkpoints remain G-selected; FiLM G is its direct global head. Test stays sealed.
Native LF seeds/evidence/analysis own paired comparison. Work → Execution → Run → Attempt → Job
identities are retained. Exceptions/OOM propagate unchanged to existing LF failure/recovery;
there is no swallowed failure, replacement dataset/batch, hidden retry or fake completed Run.

The prepared `wisdom_v8d.yaml` defines D3–D6 × two policies × four seeds = 32 mandatory new Runs.
It is not a recovery request against V8b. D0/D1/D2 and D7/D8 are available through the same public
mode field, with commented authoring examples. Start production feasibility with an explicitly
smaller NEW design containing D3/D5/D6 and a common seed; do not edit an already running Study.
The current YAML requests two GPUs and delegates concurrency to native LF admission. This is
not a proof of hardware fit. For an isolated cost comparison, explicitly use one granted GPU
and one concurrent Run in a separately named feasibility design; do not pool shared-resource
timings with isolated measurements.

## 4. Exact nominal parameter counts

For width H, each added graph layer has `2*H² + 2*H + 88` parameters: two bias-free relation
maps, LayerNorm, and two edge conditioners. One scalar gamma is added per complete transform.
D8 has `4*H² + 3*H + 1`. At H=128 it is 304 parameters below two graph layers (about .46%).
Equal nominal counts do not mean equal effective capacity, communication, activation memory or FLOPs.

| Mode | Added transform | Total MAX Single | Total Attention FiLM |
|---|---:|---:|---:|
| D0 | 0 | 469,875 | 526,598 |
| D1 | 0; second surface stage without feedback U | 850,724 | 907,447 |
| D2 | 0 beyond the existing null-message V8b route | 900,388 | 957,111 |
| D3 | 0 beyond V8b | 900,388 | 957,111 |
| D4 | 33,113 | 933,501 | 990,224 |
| D5 | 66,225 | 966,613 | 1,023,336 |
| D6 | 66,225 | 966,613 | 1,023,336 |
| D7 | 66,225, but no feedback U | 916,949 | 973,672 |
| D8 | 65,921 | 966,309 | 1,023,032 |

Totals use the archived five-scale constructor, including inherited frozen state (129 parameters
in FiLM). Actual trainable/total counts remain separately reported by Training. The H=8 unit tests
are for execution/invariance, not parameter-matched capacity claims.

## 5. Observed synthetic cost, not scientific or production feasibility

On 2026-10-20 a local CPU smoke audit used two synthetic graphs with 128/135 atoms, 256/272 points,
32 toy DCT modes, five curvature scales, full H=128 and 4+4 surface blocks. The benchmark uses
one warm-up plus five measured forward/loss/backward/AdamW steps, single-thread numerical libraries,
both exact head/weak-loss policies and no local targets. Timings below are medians in milliseconds.
Step is independently measured, not the sum of the individual medians. CPU load and warm-up noise
remain; no exclusive hardware benchmark, confidence interval or GPU acceleration is claimed.

| Policy/mode | Forward + loss | Backward | Optimizer | Complete step |
|---|---:|---:|---:|---:|
| MAX D0 | 26.87 | 28.14 | 4.17 | 58.79 |
| MAX D3 | 49.10 | 53.53 | 6.52 | 108.57 |
| MAX D5 | 52.44 | 56.91 | 7.12 | 115.84 |
| MAX D6 | 50.93 | 62.70 | 7.63 | 121.92 |
| FiLM D0 | 28.65 | 29.03 | 4.63 | 62.32 |
| FiLM D3 | 51.19 | 58.44 | 7.04 | 116.22 |
| FiLM D5 | 62.96 | 79.27 | 9.25 | 153.16 |
| FiLM D6 | 54.11 | 58.47 | 7.70 | 118.65 |

These toy measurements show that the full new path can execute and be timed. They do not predict
production ratios or justify calling D5 faster/slower scientifically: models were tiny in atoms
and points, resource noise is uncontrolled, and there were no full epochs or repeated-seed fits.
CPU VRAM, production peak memory, median production epoch times and reliable FLOPs are unavailable,
not zero. `scripts/benchmark_v8d.py --device cuda` measures this process's peak allocated VRAM;
`--checkpoint` compares recomputation. Neither option submits a Work or claims a scientific result.

A second CPU recomputation smoke (one warm-up, three measured steps, same shapes) completed for
D3/D5/D6 in both policies. Median complete steps were 134.72/153.58/147.49 ms for MAX and
140.59/163.59/145.46 ms for FiLM. This demonstrates the expected compute trade-off, not a measured
memory saving: no CUDA peak was observed here. The optional CUDA peak includes warm-up and measured
steps; timing medians exclude warm-up. Gradient/output/RNG equivalence is established separately
by the deterministic numerical tests, not inferred from these timing numbers.

## 6. Verification and pending decisions

Synthetic tests cover all D0–D8 for MAX/Single and Attention/FiLM, unequal disjoint proteins,
overlapping relation edges, uncovered atoms/invalid points, no cross-protein contamination,
coherent atom permutation and rigid-motion invariance, complete-path nonzero/finite gradients,
exact D0–D3 state/output equivalence, paired initialization, count formulas, null content,
strict constructor/state restoration, AMP/recomputation equivalence, shared operators/tables,
explicit compact budgets, no local-target model inputs, and uncaught injected OOM.
The central test shows that surface evidence at atom A changes uncovered neighbour B after
graph propagation, but not through V8b's atomwise U or D8's local MLP.

Final technical verification on 2026-10-20: Ruff passed, mypy passed for 176 source files,
and the complete suite passed **866 tests**, with four skips (one unavailable generated benchmark
and three optional browser checks because Playwright is not installed). The focused V8d module
contains 61 passing tests. `lf validate`, `lf explain` and `lf config resolve` accepted V8d;
native local WorkRunner dry-run enumerated 32 Runs without invoking Training. The required DNA
preprocessing/validation and V1a/V2 configuration checks also passed, including local DNA planning.
The new LF metadata explicitly distinguishes added parameter count and unconstrained gamma from
quality metrics or HPO controls. Actual target capacity, CUDA conformance and real biological
training remain untested here. No real Work was launched, resumed or mutated by these checks.

Still pending before scientific interpretation:

1. Real D3/D5/D6 feasibility at representative and upper-tail training sizes, same grant/batch/precision.
2. Production peak VRAM, forward/backward/step and median-epoch measurements; reliable FLOPs if available.
3. Paired D5−D3, D5−D6, D4−D5 within each policy, retaining every required seed and missing-cell reason.
4. Attribution D7/D8 and contemporary D0/D1/D2 where scientifically needed.
5. Separately named compact budgets and initialization comparisons, if justified.
6. Frozen shortlist and independent-seed confirmation, without opening test for development.

Architecture implemented, configurations prepared, synthetic verification and real scientific
results are distinct statuses. V8d currently has **no executed scientific Runs or confirmed gain**.
