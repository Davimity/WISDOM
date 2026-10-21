# WISDOM V8 technical review

## 1. Scope and execution status

V8 is now three independent, fixed-design Studies, not one architectural search. The architecture
audit below was implemented on 2026-10-18; the two-line configuration update is dated 2026-10-19.
The configuration update changes no implementation under `src/` and no historical V6/V7 YAML.
No training was
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

Every candidate uses seeds **151, 157, 163, 167**. Each Study contains both complete reviewed V7
lines, not an independently crossed set of pooling/head/loss choices. There is no adaptive HPO,
competitive pruning or seed racing. Ordinary per-Run patience remains.

| Study / YAML | Candidates, each evaluated under all four seeds | Native reference | Runs |
| --- | --- | --- | ---: |
| `wisdom-v8a` / `experiments/v8/wisdom_v8a.yaml` | `baseline`, `point_geometry`, each in A and B | A / `baseline` | 16 |
| `wisdom-v8b` / `experiments/v8/wisdom_v8b.yaml` | `baseline`, `depth_control`, `depth_control_parameter_matched`, `bidirectional`, each in A and B | A / `depth_control` | 32 |
| `wisdom-v8c` / `experiments/v8/wisdom_v8c.yaml` | `baseline`, `vector_8`, `vector_16`, each in A and B | A / `baseline` | 24 |
| Total | Eighteen campaign-specific candidates; controls are not pooled across Studies | — | 72 |

The exact candidate matrix is below. Every row requires all four seeds, not a selected favorable
subset. Native references identify A by MAX + single, not by an unsupported `pipeline_id` argument.

| Study | Frozen line | Architecture |
| --- | --- | --- |
| V8a | A: MAX R Single | baseline |
| V8a | A: MAX R Single | point_geometry |
| V8a | B: Attention NRK FiLM | baseline |
| V8a | B: Attention NRK FiLM | point_geometry |
| V8b | A: MAX R Single | baseline |
| V8b | A: MAX R Single | depth_control |
| V8b | A: MAX R Single | depth_control_parameter_matched |
| V8b | A: MAX R Single | bidirectional |
| V8b | B: Attention NRK FiLM | baseline |
| V8b | B: Attention NRK FiLM | depth_control |
| V8b | B: Attention NRK FiLM | depth_control_parameter_matched |
| V8b | B: Attention NRK FiLM | bidirectional |
| V8c | A: MAX R Single | baseline |
| V8c | A: MAX R Single | vector_8 |
| V8c | A: MAX R Single | vector_16 |
| V8c | B: Attention NRK FiLM | baseline |
| V8c | B: Attention NRK FiLM | vector_8 |
| V8c | B: Attention NRK FiLM | vector_16 |

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
frozen V7 backbone (H=128, element width 32, residue width 64, atomic depth 2, projection depth 3,
surface depth 4), five curvature scales, and line A MAX/single. Line B attention/FiLM changes the
common counts; Training records both all stored parameters and `trainable_parameter_count`.

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

## 6. Frozen V7 lines and independent launch

All three `with` mappings are identical and use `weak_loss_profile: custom`. No pending-winner
placeholder remains. The researcher has retained two lines, bound in explicit conditional
`sweep.candidates` mappings. There are no MAX FiLM or Attention Single candidates, extra prior
combinations, duplicate points, inactive head dimensions, dependencies or sequential steps.

| Setting | A: MAX R Single | B: Attention NRK FiLM |
| --- | --- | --- |
| Pooling / measure | max / point | attention / area |
| Attention scorer | Inactive, absent | simple, width 256 |
| Negative suppression N | 0 | 0.03 |
| Point existence | 0 | 0 |
| Regional existence R | 0.3 | 0.03 |
| Regional ranking K | 0 | 0.03 |
| Cardinality / TV / Dirichlet | 0 / 0 / 0 | 0 / 0 / 0 |
| Head | controlled single | controlled FiLM |
| Protein loss | Original BCE unchanged | 0.5 direct BCE + 0.5 surface-derived BCE, applied once |
| Direct-head dropout / reduction | Inactive, absent | 0 / area_mean |
| Context width / detached | Inactive, absent | 16 / true |
| Local initialization / FiLM scale | Inactive, absent | copy / 1 |

Common settings retain the exact V7 backbone, baseline initialization/optimization, AdamW
learning rate 0.001, weight decay 0.0005, gate coefficient 0.04, Heat6, batch 16, 500 epochs,
patience 30, delta 0.001, automatic precision and per-epoch numerical surface diagnostics.
There are no viewers during training, test access or local-GT loss. Checkpoint selection stays G.
V8 differs from V7 by retaining just these two complete lines, using new paired seeds, introducing
only each Study's existing architectural factor, selecting controlled architecture behavior,
area feedback and baseline-denominator gate cost, and reducing operational allocations.

The Registry and physical dataset manifests agree locally and on both targets on content identity
`sha256:ac1745504422cf60b84e57b90efc99206dea039dc36d0c1359782062cb10e202` for
`wisdom-dna-reduced@6`. This matches V7. No new data or preprocessing is required.

Within each Study, group results by the complete line coordinates, then architecture, then seed.
Compute paired S/G differences against the SAME line's control. The single LF reference does not
replace B's controls. V8b's primary contrast is bidirectional versus depth control in both lines;
the matched-control contrast is complementary. Cross-line differences are not pooling-only effects.

Each Study has its own reviewer under `experiments/v8/visualization/`. Set its source execution to
the completed corresponding Study and choose a source seed/configuration explicitly. Review is
validation-only and never fits a missing model or opens test.

### 6.1. Resource observations and remaining limits

Read-only observations on 2026-10-19 found:

| Target | Host inventory | GPU inventory, not ownership | Requested Study allocation |
| --- | --- | --- | --- |
| ctgpgpu12 / V8a | 48 CPUs, 403,642,208,256 bytes RAM (~376 GiB) | Two physical H100 PCIe, 81,559 MiB each | 1 GPU, 12 CPUs, 48 GiB RAM, 168 h |
| ctgpgpu16 / V8b | 96 CPUs, 810,706,395,136 bytes RAM (~755 GiB) | Four physical H200 NVL, 143,771 MiB each | 1 GPU, 16 CPUs, 64 GiB RAM, 168 h |
| ctgpgpu16 / V8c | Same host | Same inventory | 1 GPU, 16 CPUs, 48 GiB RAM, 168 h |

The user's one-GPU budget on the 12 is respected even though physical inventory exposes two.
On the 16, site status showed two free devices, two owned by another user, and **no grant to this
account**. No reservations were requested during this review. All observed 16 devices have the
same total memory; there is no observed higher-memory device to prefer for V8b. Prefer it if
the actual future grant is heterogeneous. Free VRAM and availability are time-dependent.

LF's resource CLI GPU probe reported `ModuleNotFoundError`; direct read-only `nvidia-smi` and
`free -b` supplied the inventory instead. The 12's native dry-run reported two allocatable GPUs;
the 16's dry-runs passed but marked GPU capacity unreliable/unknown. Neither is a real-model VRAM
fit measurement. CPU/RAM inventory is sufficient for the authored requests, subject to live usage
and native admission; execution is explicitly limited to one Run per Study and one per GPU.
Do not silently reduce batch size, model widths, seed count or data if a candidate does not fit.

### 6.2. Commands for the researcher; not automatic submissions

Validate/explain/resolve all three YAMLs, then perform target-specific dry-runs before submission.
The corresponding commands and ordinary independent launches are in README section 5.5.2.
`resources.gpu: 1` is a Study-wide count, not device affinity or a per-Run repeated reservation.
No arbitrary GPU index belongs in `Training.run`.

If explicit separation is needed on the 16, use site reservation slots and two exported native
catalogues rather than editing WISDOM or modifying a live shared profile between submissions.
The provider's actual help confirms `claim --numgpus`, `exec --gpu-slot` (one-based reservation
order, NOT physical GPU index), and `env --shell`. Obtain a valid two-device grant yourself on
the remote host first, subject to site policy:

```bash
gpu status
gpu claim --numgpus 2
gpu mine
gpu env --shell
```

Map the larger granted device to V8b if capacities differ; inspect reservation order rather than
assuming physical indices or that a second reservation is non-preemptible. Then, on the controller:

```bash
lf clusters export citius-ctgpgpu16 --output /tmp/wisdom-v8b-clusters.yaml
lf clusters export citius-ctgpgpu16 --output /tmp/wisdom-v8c-clusters.yaml
lf clusters --catalog /tmp/wisdom-v8b-clusters.yaml set citius-ctgpgpu16 gpu_access.command_prefix '["/home/davidi.lubian/gpu", "exec", "--gpu-slot", "1"]'
lf clusters --catalog /tmp/wisdom-v8c-clusters.yaml set citius-ctgpgpu16 gpu_access.command_prefix '["/home/davidi.lubian/gpu", "exec", "--gpu-slot", "2"]'
lf clusters --catalog /tmp/wisdom-v8b-clusters.yaml set citius-ctgpgpu16 gpu_access.visibility_command '["/home/davidi.lubian/gpu", "env", "--shell"]'
lf clusters --catalog /tmp/wisdom-v8c-clusters.yaml set citius-ctgpgpu16 gpu_access.visibility_command '["/home/davidi.lubian/gpu", "env", "--shell"]'
lf run experiments/v8/wisdom_v8b.yaml --on citius-ctgpgpu16 --clusters /tmp/wisdom-v8b-clusters.yaml --dry-run
lf run experiments/v8/wisdom_v8c.yaml --on citius-ctgpgpu16 --clusters /tmp/wisdom-v8c-clusters.yaml --dry-run

# Submit only after reviewing each plan and grant; the Studies have no dependency.
lf run experiments/v8/wisdom_v8b.yaml --on citius-ctgpgpu16 --clusters /tmp/wisdom-v8b-clusters.yaml
lf run experiments/v8/wisdom_v8c.yaml --on citius-ctgpgpu16 --clusters /tmp/wisdom-v8c-clusters.yaml
```

These optional catalogue/claim commands were **not executed**. They describe supported options,
not a verified reservation-scoped fit or a guarantee against preemption. The account currently
has no slots to select. Release only your own manual grant after your jobs finish, using the site
procedure; WISDOM neither claims nor releases devices. Keep the exported catalogue snapshots
through submission rather than deleting them while asynchronous preparation is pending.

## 7. Verification actually performed

**2026-10-19 configuration update:** `lf validate`, `lf explain` and `lf config resolve --format
json` passed for the three authored training files, with exactly 4/8/6 distinct candidates and
16/32/24 mandatory Runs. Native remote dry-runs passed for V8a→12 and V8b/V8c→16; their capacity
limitations are stated above. No scientific Run was started. Concurrent initial dry-runs collided
while LF built its wheel (`build/lib/lambdaforge/clustering/backend.py` deletion race); serial
retries succeeded without changing LF. Wheel preparation refreshed three tracked WISDOM
`build/lib` copies; only those generated changes were restored to their original state.

Focused existing suites for V8, V7 heads, diffusion, analysis and configuration produced **213
passed, 1 skipped** before the user's instruction to stop further command execution. The skip is
the unavailable generated production benchmark. The configuration regression was updated for
the new conditional two-line design. Its final signature assertion and subsequent documentation
edits were not rerun after that instruction. No new full-suite/lint/type-check result is claimed
for this configuration update. Runtime Python implementation was not changed.

**Historical 2026-10-18 architecture-audit verification (not rerun for the configuration update):**

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
  planning test used a typed local fixture without publishing a fake DatasetVersion. Its old
  8/16/12 budget was superseded by the two-line 16/32/24 design on 2026-10-19.
- Required DNA preprocessing validation/explain/dry-run, DNA validation, V1a validation and V2
  validation passed. No real training, preprocessing, or reviewer forward was executed by these checks.

Tests establish technical behavior, not that any architecture improves localization. Scientific
claims require completed repeated runs under the two frozen V7 lines and their paired results.

## 8. Changed-file map

- Model behavior: `ArchitectureImplementation.py`, `ArchitectureSpike.py`, `InteractionRound.py`,
  `SurfaceAtomFeedback.py`, `VectorAtomicState.py`, `SurfaceAtomTransfer.py`, `WisdomV1.py`,
  `WisdomV2.py`.
- Public forwarding/persistence/metrics: `Training.py`, `WisdomAnalysisProfile.py`.
- Experiments: three training YAMLs and three inference-only reviewers; two historical snapshots.
- Verification: `test_v8_architectures.py`, catalogue expectations in `test_design_and_efficiency.py`
  and `test_review_native.py`.
- Documentation: equivalent root READMEs, `docs/model_roadmap.md`, experiment catalogue and this audit.

The configuration update changes only the three V8 training YAMLs, documentation and the existing
configuration regression test. It does not change model, training, loss or preprocessing source.
The earlier architecture-audit diff included pre-existing V7 edits and remains a historical record,
not a patch to apply as if it contained only the current configuration update.
