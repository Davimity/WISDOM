# WISDOM V8 technical review

## 1. Scope and execution status

V8 is now three independent, fixed-design Studies, not one architectural search. No training was
launched, no held-out test data were inspected, and LambdaForge was not modified. The portable
reviewers perform inference only on explicitly selected saved checkpoints. The previous combined
training and review configurations are retained as `.yaml.example` historical snapshots.

The workspace already contained V7 changes before this review. Those changes, including its YAML,
remain intact. This review does not change any V1–V7 experiment YAML or generated `build/lib` copy.

## 2. Implementation audit

| Finding | Classification | Location and resolution |
| --- | --- | --- |
| Some atoms with no valid feedback neighbors were transformed whenever other atoms had coverage. | Confirmed bug | `SurfaceAtomFeedback.forward`: controlled execution selects the original state exactly for uncovered atoms; identity gradients are tested. |
| Ordered endpoint concatenation changed vector coefficients when an undirected edge was reversed or atoms were renumbered. | Confirmed bug | `VectorAtomicState.forward`: controlled execution averages the same coefficient scorer over both endpoint orders, retaining the signed geometric direction. |
| Allocating extra modules advanced the random stream used to initialize later shared modules. | Confirmed comparability defect | `WisdomV1._initialize_controlled_architecture`: optional module families use reproducible private streams after common initialization. Shared V2 heads/pooling remain paired too. |
| Adding a point-geometry gate changed the denominator of the mean L0 cost. | Methodological confound | `WisdomV1.gate_regularization`: optional `baseline_count` fixes the common-gate denominator. With five curvature scales, 41 becomes 42 gates; historical coefficients would decrease by 1/42, approximately 2.38%. |
| Point-count feedback changes when one surface region is sampled more densely. | Integration-policy choice | `SurfaceAtomFeedback.forward`: optional represented-area weighting gives exact invariance when an identical point is subdivided and its area is partitioned. Approximate remeshing need not give exact invariance. |
| Feedback adds parameters beyond a second surface stage. | Capacity confound | The new `depth_control_parameter_matched` profile has the same update module as bidirectional but passes an explicit null message, never detached S1. |
| Point geometry could disappear as a constant softmax offset. | Checked; existing implementation retained | `SurfaceAtomTransfer.forward`: the existing point, atom and pair projections interact before nonlinear scoring. A gate-off equality test and nonzero branch-gradient tests verify its effect. |
| Tiny denominators, empty neighborhoods and mixed precision could produce invalid values. | Numerical risk | Controlled feedback/vector accumulation uses FP32; coverage-aware normalization handles empty/zero-area neighborhoods. CPU BF16 and CUDA FP16 forward/backward tests are finite. |

The corrections are versioned by `architecture_implementation`, not silently imposed on archived
models. Legacy defaults preserve the historical module layout, RNG ordering and forward semantics.
`SurfaceAtomTransfer` retains its existing scorer and checkpoint keys; an explicit method attaches
only its optional point-geometry layers after shared initialization.

## 3. Designs and controls

Every candidate uses seeds **151, 157, 163, 167**. There is no adaptive HPO, competitive pruning,
seed racing, or Cartesian crossing with pooling/head/loss choices. Ordinary per-Run patience remains.

| Study / YAML | Candidates, each evaluated under all four seeds | Native reference | Runs |
| --- | --- | --- | ---: |
| `wisdom-v8a` / `experiments/v8/wisdom_v8a.yaml` | `baseline`, `point_geometry` | `baseline` | 8 |
| `wisdom-v8b` / `experiments/v8/wisdom_v8b.yaml` | `baseline`, `depth_control`, `depth_control_parameter_matched`, `bidirectional` | `depth_control` | 16 |
| `wisdom-v8c` / `experiments/v8/wisdom_v8c.yaml` | `baseline`, `vector_8`, `vector_16` | `baseline` | 12 |
| Total | Nine campaign-specific candidates; controls are not pooled across Studies | — | 36 |

**V8a** tests only whether the existing gated curvature/shape tensor helps assign relative weights
to candidate atoms. It adds no preprocessing or new source of information. Its descriptors are
rigid-motion invariant. The fixed baseline gate denominator maintains the coefficient paid by each
shared gate; the extra gate still incurs its own cost. Consequently this normalized cost can exceed
one and is not the fraction of active gates. Training logs that fraction separately.

**V8b** keeps the same initialized second transfer, projection and DiffusionNet stage across its
three two-stage variants:

| Variant | Information route | Atom update before stage two |
| --- | --- | --- |
| baseline | A → S | None; only one surface stage |
| depth control | A → S1 → S2 | None; stage two receives the original atom states |
| matched depth control | A → S1 → A′ → S2 | Same update module as feedback, but its surface-message input is zero |
| bidirectional | A → S1 → A′ → S2 | Real differentiable feedback from S1 |

In all two-stage variants S1 also enters the second surface projection. Bidirectional does **not**
add another atom–atom message-passing round. Its primary scientific comparison is against depth
control, with the matched control assessing extra nominal capacity. Beating baseline alone does
not establish a benefit from feedback.

The matched control has exactly the same nominal trainable parameter count and update-network
computation as bidirectional. It is not an exact match in effective capacity or total FLOPs: columns
multiplying the zero message have zero data gradient, and message aggregation is cheaper. Both
retain the same geometric coverage mask and area-weighting policy.

**V8c** tests eight or sixteen intermediate three-dimensional vector channels. Scalar edge
coefficients multiply signed unit directions, vectors are accumulated and degree-normalized, and
their norms update scalar atom embeddings. This is scalar augmentation through equivariant
intermediates, not a deep vector encoder carrying vector states through successive layers.

## 4. Configuration and historical compatibility

| Public option | Historical/default value | V8 selection / accepted alternatives |
| --- | --- | --- |
| `architecture_spike` | `custom` | One named candidate above; profiles reset geometry/round/vector choices before applying their own override |
| `surface_geometry_transfer` | `false` | Available in custom configurations; enabled only by the point-geometry candidate here |
| `interaction_round` | `single` | `single`, `depth_control`, `depth_control_parameter_matched`, `bidirectional` |
| `vector_atomic_channels` | `0` | Nonnegative integer; V8c uses 0, 8, 16 through named profiles |
| `architecture_implementation` | `legacy` | `controlled` in every V8 candidate |
| `feedback_weighting` | `point` | `area` in V8; uses existing represented surface areas in Å² |
| `gate_cost_normalization` | `historical_mean` | `baseline_count` in every V8 candidate |

These are the `Training.run`/model defaults. Standalone `SurfaceAtomFeedback` and `VectorAtomicState`
constructors default to controlled behavior; the historical model path explicitly requests legacy.
Matched feedback and area weighting require controlled execution. No feedback strength, width or
round-count sweep has been added.

New model options are persisted in checkpoint `model_parameters`. Older checkpoints omit them and
therefore reconstruct legacy behavior. Strict state restoration is tested for old profiles and all
new profiles with each of the four V7 heads. Controlled and legacy variants may have compatible
tensor shapes but different semantics: do not manually relabel an old checkpoint as controlled.

## 5. Capacity, cost and metrics

These are **constructor counts**, not measured accuracy or memory improvements. They use the
provisional V7 backbone (H=128, element width 32, residue width 64, atomic depth 2, projection depth 3,
surface depth 4), five curvature scales, MAX pooling and a single head. The actual V7 winner can
change the common counts; Training records both all stored parameters and `trainable_parameter_count`.

| Variant | Trainable parameters |
| --- | ---: |
| baseline | 469,875 |
| point_geometry | 472,308 |
| depth_control | 850,724 |
| depth_control_parameter_matched | 900,388 |
| bidirectional | 900,388 |
| vector_8 | 506,243 |
| vector_16 | 509,331 |

Feedback adds 49,664 nominal parameters over depth control. The intermediate FP32 vector buffer
alone occupies `4 × N_atoms × C_vector × 3` bytes, before messages, autograd and scalar activations.
CUDA synthetic tests verify finite backward passes and nonzero peak allocation, but do not estimate
real-protein peak memory, epoch cost or speed. Existing Training metrics retain epoch seconds,
evaluation forward seconds/latency, throughput and CUDA allocation peaks for real runs.

Checkpoint epochs remain selected only by G, the existing 0.70 protein AUPRC + 0.30 protein AUROC
score. The scientific priority is operational positive-protein macro surface AUPRC S at that
checkpoint. Raw S, global metrics, enrichment/recall at top-k, negative false evidence, temporal
regret/coupling and multi-head diagnostics remain available. Surface GT is not an input to training
or backward. The existing joint utility indexes the complete fixed design; it does not add pruning
or change epoch selection. Paired seed variability must be assessed within each Study after all
mandatory cells complete. Four seeds do not guarantee sufficient power for small effects.

## 6. Pending V7 decision

All three `with` mappings are identical. Their explicit `weak_loss_profile: PENDING_V7` guard stops
scientific execution before data loading or model allocation. MAX, single head and zero weak-loss
coefficients are marked provisional, not selected winners. Native planning is still possible.

After reviewing V7, copy its exact selected pooling and hyperparameters, point/area policy, all weak
coefficients, head/context/loss-balance settings, refiner (currently explicit Heat6), initialization,
optimizer, dataset identity and checkpoint semantics identically into all three files. Only then
replace `PENDING_V7` with the applicable profile. Do not open additional factors or infer the winner
from the old combined V8 snapshot. Verify the immutable Registry identity on the execution target.

Each Study has its own reviewer under `experiments/v8/visualization/`. Set its source execution to
the completed corresponding Study and choose a source seed/configuration explicitly. Review is
validation-only and never fits a missing model or opens test.

## 7. Verification actually performed

- Full suite: **618 passed, 4 skipped**. Skips concern an unavailable generated production fixture
  and three unavailable Playwright browser checks, not V8 architecture failures.
- After the final gate-cost arithmetic/import and portable fixture adjustments: **190 passed** in
  the focused V8, V7 heads, diffusion architecture and analysis-profile suites.
- Ruff: passed. Mypy: passed across 172 source files. `git diff --check`: passed.
- New tests exercise all seven architectural profiles and all four V7 heads, Heat6 and weak losses,
  strict checkpoints, common initialization under three policies, gradients through feedback,
  null-message capacity, area subdivision, no coverage, masks, coincident atoms, empty graphs,
  protein isolation, atom/point permutations, rigid motions, vector equivariance, gate costs,
  geometry gate-off equality, CPU BF16 and CUDA FP16, with no surface-target gradient.
- Native LF validation and explain passed for all six new training/review YAML files. Public native
  WorkRunner dry-run planning produced 8/16/12 training Runs and one Run per reviewer. The automated
  planning test uses a typed local fixture without publishing a fake DatasetVersion.
- Required DNA preprocessing validation/explain/dry-run, DNA validation, V1a validation and V2
  validation passed. No real training, preprocessing, or reviewer forward was executed by these checks.

Tests establish technical behavior, not that any architecture improves localization. Scientific
claims require the frozen V7 ledger, completed repeated runs and their paired results.

## 8. Changed-file map

- Model behavior: `ArchitectureImplementation.py`, `ArchitectureSpike.py`, `InteractionRound.py`,
  `SurfaceAtomFeedback.py`, `VectorAtomicState.py`, `SurfaceAtomTransfer.py`, `WisdomV1.py`,
  `WisdomV2.py`.
- Public forwarding/persistence/metrics: `Training.py`, `WisdomAnalysisProfile.py`.
- Experiments: three training YAMLs and three inference-only reviewers; two historical snapshots.
- Verification: `test_v8_architectures.py`, catalogue expectations in `test_design_and_efficiency.py`
  and `test_review_native.py`.
- Documentation: equivalent root READMEs, `docs/model_roadmap.md`, experiment catalogue and this audit.

The accompanying diff is labelled cumulative because some shared files already had uncommitted
V7 edits when this task began. It must not be interpreted as exclusively this task's changes.
