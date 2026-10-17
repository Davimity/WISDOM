[Español](README.es.md)

# WISDOM V6: isolated weak regularizers

- [1. Run and inspect](#1-run-and-inspect)
- [2. Frozen comparison and study design](#2-frozen-comparison-and-study-design)
- [3. Exact losses and interpretation](#3-exact-losses-and-interpretation)
- [4. Exploratory mixtures and checkpoint reviews](#4-exploratory-mixtures-and-checkpoint-reviews)

## 1. Run and inspect

V6a–V6g are independent single-Work isolated screens, not steps or copied winners.
V6a/c/d/f now retain their original grids plus selective upper-range additions; extend completed
Studies through native LF rather than restarting historical controls. V6h is an executable
exploratory mixture screen informed by V6a–V6f; V6g evidence remains pending, not assumed positive.

Use the exact V5c benchmark content:
`sha256:ac1745504422cf60b84e57b90efc99206dea039dc36d0c1359782062cb10e202`.
A same-name version with different content is not an equivalent benchmark. Check the execution
target's Registry: local YAML validation does not establish remote content equivalence.

```bash
lf datasets show wisdom-dna-reduced@6 --on citius-ctgpgpu12
lf validate experiments/v6/wisdom_v6c.yaml
lf explain experiments/v6/wisdom_v6c.yaml
lf run experiments/v6/wisdom_v6c.yaml --on citius-ctgpgpu12 --dry-run
lf run experiments/v6/wisdom_v6c.yaml --on citius-ctgpgpu12
lf run experiments/v6/wisdom_v6d.yaml --on citius-ctgpgpu12
lf run experiments/v6/wisdom_v6e.yaml --on citius-ctgpgpu12
lf run experiments/v6/wisdom_v6f.yaml --on citius-ctgpgpu12
lf run experiments/v6/wisdom_v6g.yaml --on citius-ctgpgpu12
lf validate experiments/v6/wisdom_v6h.yaml
lf explain experiments/v6/wisdom_v6h.yaml
lf run experiments/v6/wisdom_v6h.yaml --on citius-ctgpgpu12 --dry-run
lf results list
lf results analyze wisdom-v6f
lf results report wisdom-v6f --output v6f-report.html
lf run experiments/v6/visualization/wisdom_v6f.yaml
```

Change the cluster as needed. Import remote evidence and checkpoints with native `lf import`
before a local review. A review's `source_execution: wisdom-v6f` must resolve to one Execution.
If the name is ambiguous, use its actual `execution-…` ID. Reviews never select the latest
implicitly or retrain missing models.

## 2. Frozen comparison and study design

The ledger matches current V5c and the isolated V6 scientific parent. No initialization,
optimizer, encoder, pooling hyperparameter or dataset search is reopened.

| Item | Fixed value |
|---|---|
| Dataset / generation / view | `wisdom-dna-reduced@6` / `model_version=2` / `full` |
| Initialization / optimization / spike / head | `baseline` / `baseline` / `baseline` / `single` |
| Hidden / element / residue width | 128 / 32 / 64 |
| Atomic / projection / surface depth | 2 / 3 / 4 |
| Atom neighbors / transfer neighbors / spectral modes | 8 / 16 / 32 |
| Transfer radius / dropout | 4 Å / 0.2 |
| AdamW learning rate / decay / gate coefficient | 0.001 / 0.0005 / 0.04 |
| Batch / epoch ceiling / G patience / minimum delta | 16 / 500 / 30 / 0.001 |
| Precision / data workers | `auto` / 4 |
| Surface evaluation / interval | enabled / every epoch |
| Faithfulness deletion / test evaluation | disabled / disabled |
| MAX | point control; no active LSE/Attention fields |
| LSE | area, fixed beta 0.25 |
| Attention | area, simple embedding-based scorer, width 256 |
| Weak profile | `custom`; one isolated coefficient family per screen |
| Seeds | 4, 7, 32, 54 |
| Objective / mode / range / practical margin | `val_wisdom_hpo_score` / max / [0,1] / 0.015 |
| Allocation / dispatch budget | GPU count is operational and read from each YAML; h requests 3 GPUs; 36 CPUs, 96 GiB, 168 h / 168 h |

Here G measures protein classification and S positive-protein surface localization. Their definitions,
support counts and the development utility W are in [README Section 5.5](../../README.md#55-training-evaluation-and-artifacts).
W is unchanged; **G alone selects each Run's checkpoint**. Local contact ground truth enters
evaluation/development utility, never losses or checkpoint selection. Test remains sealed.

| Study / regularizer | Hypothesis / pathology addressed | Active parameters | Heat interaction | Trials / Runs |
|---|---|---|---|---:|
| V6a negative | Remove spurious positive mass/peaks on negative proteins | original grid plus selective LSE/Attention high-lambda points | fixed 6 Å | 23 / 92 |
| V6b existence (frozen) | Ensure a positive protein has at least one positive peak | lambda 0, .03, .10, .30 | fixed 6 Å | 12 / 48 |
| V6c regional positive | Require a positive region, not only a point peak | original grid plus selective LSE/MAX extensions; length 2.5 Å | extra diffusion after Heat6 | 19 / 76 |
| V6d regional ranking | Separate positive/negative regional maxima | original grid plus selective LSE/Attention extensions; length 2.5 Å; margin .5 | regional scoring after Heat6 | 20 / 80 |
| V6e cardinality | Avoid insufficient positive probability mass; test task-prior dependence | lambda 0, .03, .10, .30; active lower mass .005, .010, .020; no upper bound | fixed 6 Å | 30 / 120 |
| V6f TV | Penalize local jumps without necessarily erasing boundaries | original grid plus LSE/none and MAX/Heat extensions | none / Heat6 | 31 / 124 |
| V6g Dirichlet | Penalize high-frequency probability variation | lambda 0, .001, .010, .050, .100 | none / Heat6 | 30 / 120 |
| V6h exploratory mixtures | Separate interactions, redundancy and all-family stress | 18 explicit combinations per pooling; provisional anchors | fixed 6 Å; TV/Dirichlet exclusive | 54 / 216 |

The current isolated YAMLs declare **165 Trials / 660 Runs across V6a–V6g**. This includes the
30 / 120 selective a/c/d/f additions and V6g's already authored five-value grid (30 / 120).
These are design counts, not evidence of completed Runs. V6h adds 54 / 216 separately.
Cardinality's zero coefficient has no active lower-mass child: one control, not three.
Heat length is absent from identity Trials. Native `when` conditions also omit inactive pooler fields.
Full sweeps disable adaptive pruning and seed racing, but G-based patience still stops a completed
Run's optimization. A 168-hour ceiling can censor coverage; verify all required Runs before
declaring the sweep complete.

## 3. Exact losses and interpretation

A protein is a bag of surface points: training knows its binary label, not which points bind DNA.
Let B+ and B− be the positive and negative proteins in a batch; P_b their point sets;
l_p the **operational logit** at point p, q_p = sigmoid(l_p) its probability, and
a_p its represented area divided by the total represented area of its own protein, so the
a_p sum to one per protein. Define softplus(x) = log(1 + exp(x)) and [x]+ = max(0,x).
The identical operational l enters pooling and every weak prior; raw head logits remain diagnostics.
Each class-specific term is differentiable zero if its required class is missing.

The negative penalty averages area-weighted negative binary cross-entropy over negative proteins.
Existence instead requires one positive peak in each positive protein:

```math
L_{neg}=\frac{1}{|B_-|}\sum_{b\in B_-}\sum_{p\in P_b}a_p\operatorname{softplus}(l_p),
\qquad
L_{exist}=\frac{1}{|B_+|}\sum_{b\in B_+}\operatorname{softplus}(-\max_{p\in P_b}l_p).
```

With MAX pooling, existence duplicates positive global BCE: it changes relative class weight,
not localization constraints. Area-LSE/Attention do not have that identity. Strong negative pressure
can remove real positive evidence indirectly through shared parameters, so inspect both classes.

**Regional terms.** Let Phi_b be stored mass-orthonormal surface modes, M_b the diagonal represented
mass, and Lambda_b their nonnegative inverse-square-Å eigenvalues. Regional diffusion uses the
truncated stored basis, not a new preprocessing representation. At length r = 2.5 Å it produces
u_b = Phi_b exp(−r² Lambda_b) Phi_bᵀ M_b l_b. Let R_b = max(u_b).
The regional-positive loss is existence on these scores; ranking compares **every positive-negative
pair in the batch**, with fixed logit margin delta = .5:

```math
L_{regional+}=\frac{1}{|B_+|}\sum_{b\in B_+}\operatorname{softplus}(-R_b),
\qquad
L_{rank}=\frac{1}{|B_+||B_-|}\sum_{b\in B_+}\sum_{c\in B_-}[\delta-R_b+R_c]_+.
```

Ranking is zero in single-class batches. Regional diffusion occurs after operational Heat6 in c/d;
its marginal contribution may be redundant. The additional spectral filter is not a second
learned encoder and is not a spatially annotated target.

**Cardinality.** The expected positive mass m_b = sum(a_p q_p) is an area-weighted probability,
not the fraction exceeding a hard threshold. With lower bound mu, no upper bound, the current loss is:

```math
L_{card}=\frac{1}{|B_+|}\sum_{b\in B_+}[\mu-m_b]_+^2,
\qquad m_b=\sum_{p\in P_b}a_p q_p.
```

It is zero once the lower bound is satisfied. At the largest .02 lower bound, its unweighted maximum
is only .0004, so at lambda .30 its contribution is at most .00012. These conservative settings may
often be inactive or weak; inspect recorded cardinality loss and mass before interpreting a null
effect as proof that the idea fails. No change of scale or upper bound is hidden in this screen.

**TV is not Heat.** E is the multiset of valid directed entries in the stored bounded surface
neighbour table. TV averages absolute probability differences, without area or distance weighting:

```math
L_{TV}=\frac{1}{|E|}\sum_{(i,j)\in E}|q_i-q_j|.
```

An empty table contributes differentiable zero. This L1 penalty may favor homogeneous regions
while permitting sharper boundaries than quadratic smoothing, but offers no guarantee:
small real hotspots can disappear. Its batch average weights proteins according to their valid
neighbour counts, unlike the equal-protein averages above.

**Dirichlet is not Heat either.** For each protein compute spectral coefficients
c_b = Phi_bᵀ M_b q_b. The current normalized energy, averaged equally over proteins, is:

```math
L_D=\frac{1}{B}\sum_{b=1}^{B}
\frac{\sum_k\lambda_{bk}c_{bk}^2}
     {\max(\sum_k c_{bk}^2,\epsilon)}.
```

B is batch size, k indexes stored modes, and epsilon is FP32 machine epsilon. This is a spectral
Rayleigh quotient in inverse-square Å, not an unnormalized graph energy. Large eigenvalues measure
rapid surface variation. A constant has approximately zero energy if the constant mode is faithfully
represented. Truncation cannot see frequencies above the stored modes; numerical operators can
leave small residuals. Zero probability power is safely bounded by epsilon.

Heat explicitly transforms **logits in the forward pass** using exp(−r² Lambda), so pooling sees
the filtered map. TV and Dirichlet instead penalize **probabilities during training**; they do not
install a final forward filter. In f/g compare, within each pooling and seed: none/lambda0,
Heat6/lambda0, none/active loss, Heat6/active loss. These four states test replacement, redundancy,
complementarity and over-smoothing. Raw S, operational S and their unclipped difference reveal
whether the loss improves evidence before refinement or only its filtered map.

Let L_BCE be global protein BCE, L_gate the existing semantic-gate penalty, and F the seven families
above. Their exact training composition is:

```math
L=L_{BCE}+0.04L_{gate}+\sum_{f\in F}\lambda_f L_f.
```

In a–g only one lambda family varies; the others are zero. TV and Dirichlet may never both be active.
Judge each regularizer using G, operational/raw S, negative mass/peak, top-k recall/enrichment,
selection regret, global-surface coupling, paired-seed dispersion and cost. Undefined metrics retain
support counts instead of becoming zero. A high W alone does not justify adding complexity.

## 4. Exploratory mixtures and checkpoint reviews

V6h is now an **exploratory combination screen**, not an empty template or a confirmed final model.
It declares 18 exact candidates per pooling (54 configurations, 216 Runs), all with Heat 6 Å,
auxiliary regional length 2.5 Å, ranking margin 0.5 and the unchanged V5c backbone. Native
`sweep.candidates` executes only the listed points; the parameter domains do not create a large
Cartesian product. YAML anchors reuse zero-prior controls without adding a scientific parameter.

The coefficients below were authored after reading the TFM report's completed V6a–V6f summaries
and experimental tracking on 16 October 2026. They are starting hypotheses, not jointly optimized
weights. No V6g or newly extended-range result was available in that report.

| Pooling | Negative N | Point existence E | Regional existence P | Ranking R | Minimum mass C | TV T | Dirichlet D |
|---|---:|---:|---:|---:|---:|---:|---:|
| MAX | 0.3 | 0.03 | 0.3 | 0.1 | 0.03 | 0.005 | 0.001 |
| Area LSE, beta 0.25 | 1 | 0.03 | 0.1 | 0.03 | 0.1 | 0.005 | 0.001 |
| Area Attention, width 256 | 0.03 | 0.03 | 0.03 | 0.03 | 0.03 | 0.005 | 0.001 |

Every active C uses a lower expected probability-mass bound of 0.02 and no upper bound. This is
not a measured interface-area target. The coefficients multiply differently scaled loss terms;
equal numerical coefficients do not imply equal gradient strength.

Anchors differ between poolings: compare mixture effects against that pooling's own control.
A difference between active MAX and active LSE does not isolate the pooling mechanism alone,
because the loss coefficients can also differ. The zero-prior controls retain that clean comparison.

Each pooling has its own zero control and seven isolated terms. The N/E/R core contains every
on/off state: control, N, E, R, N+E, N+R, E+R and N+E+R. Additional candidates replace E with P,
add C, T or D individually to the core, and stack N+E+P+R+C with either T or D.
These last two are stress tests of all compatible families, not recommended production models.
TV and Dirichlet are never enabled together. The E-to-P replacement helps identify redundancy
between two existence constraints instead of attributing their sum to a new spatial mechanism.

Let A and B denote two enabled priors at their fixed coefficients. For a chosen score Q and a
fixed pooling/seed, compare the pair using
`interaction = Q(A+B) - Q(A) - Q(B) + Q(control)`. Positive values mean more improvement than
the sum of the individual changes on that score's scale; negative values mean less.
For example, opposite changes in G and S are a trade-off, not universal synergy.
The same-seed single-term controls must come from V6h, not be pooled with historical Runs whose
code or optimization trajectory differs.

Evidence is uneven: MAX existence 0.03 and ranking 0.1 improved local metrics; LSE negative 1
improved global ranking and reduced negative mass; LSE/Heat TV 0.005 improved operational S
without improving mean raw S or cleaning negatives. Attention's negative coefficient is kept at
0.03 because larger observed values damaged global ranking. Other coefficients are conservative
sensitivity probes, not claimed winners. In particular, C was often inactive in V6e and D=0.001
is provisional until V6g is reviewed. Inspect recorded weak-loss contributions before interpreting
a cardinality gain; a zero contribution does not demonstrate an effective cardinality mechanism.
Mixing both smoothness penalties is prohibited by the existing implementation.

Checkpoint selection remains G-only; test stays sealed. Inspect operational/raw S, G, threshold-free
protein AUPRC/AUROC, negative mass/peak, enrichment, regret and time trajectories, not W or
accuracy at 0.5 alone. The historical seeds 4, 7, 32 and 54 support paired development comparisons,
not independent confirmation. An adopted mixture still needs fresh seeds and a separate frozen
confirmation. This YAML neither launches that confirmation nor claims the missing V6g evidence.

The review YAMLs forward saved checkpoints without fitting. They select a recorded source seed
(default authored seed 4), automatic contrasting proteins and optional parameter filters; numerical
metrics retain complete split/point coverage. Configure learned channels with `content: features`.
Automatic cases can differ between configurations: the gallery compares available maps, not an
implicitly matched complete panel. Presentation byte/point limits remain separate from scientific
evaluation; missing checkpoints require native recovery, not hidden retraining. See Section 5.10
of the root README for current review syntax and LF report-asset limitations.
