# Experiment catalogue

Use the [main README](../README.md) for installation, dataset contracts and execution commands.
The [scientific report](../docs/WISDOM-memoria.pdf) explains hypotheses, controls, results and
decision criteria. Exact values and unresolved prerequisites live next to the fields in each YAML.
No catalogue entry proves an improvement, a closed scientific comparison or available checkpoints.

## 1. Dataset preparation

| Directory | Purpose |
|---|---|
| `preprocess/dna` | Select/freeze DNA evidence, build the managed representation, audit and inspect it |
| `preprocess/zinc` | Acquire explicit Zn evidence, select/freeze it, build, validate and visualize |
| `preprocess/zinc/zinc_legacy_preprocess.yaml` | Preserve the TFG's 450 Zn members, labels and splits with the current geometry/ESM-2/SLAE assets; no selection |
| `preprocess/enrich_structures.yaml` | Attach declared assets to a compatible immutable representation |
| `preprocess/enrich_physics.yaml` | Recover/publish historical strict physical `@1`, retaining SLAE/structural assets |
| `preprocess/enrich_physics_bounded.yaml` | Publish bounded-completion physical `@2`, re-admitting unchanged strict fields without solves |

Preprocessing compositions are intentional exceptions to single-Work training studies. Read the
`skip` recipes before launching: the DNA recipe currently reuses a design and keeps train-25.
Enrichment is not equivalent to recomputing geometry with changed settings. A new dataset version
does not silently replace the immutable version named by a historical experiment.

## 2. Training stages

| Stage | Question and controlled change |
|---|---|
| `v1` | Baseline seed variation; `v1a` varies full seeds, `v1b` fixes initialization, `v1c` fixes later randomness |
| `v2` | Initialization and early optimization stability |
| `v3` | Optimization policies and weight averaging |
| `v4` | Core backbone hyperparameter optimization |
| `v5` | Pooling: fixed/curriculum curves (`5a`), learned parameters (`5b`), surface refinement (`5c`) |
| `v6` | Isolated weak surface constraints (`6a`–`6g`) and explicit mixtures (`6h`) |
| `v7` | Protein/local prediction-head comparisons |
| `v8` | Point geometry, real/null feedback, vector controls and post-feedback propagation |
| `v9` | Atomic-encoder comparisons with the reviewed single-pass surface context fixed |
| `v10` | Atom/residue representations, chemical context and declared external-model inputs |
| `v11` | Atom-to-surface summaries, values, heads and candidate support |
| `v12` | Frozen isolated tensor/rebuilt rotations, phase, grid and resolution audits; no fitting |
| `v13` | Spectral budgets, simultaneous scales, local branches and explicit encoder/confirmation YAMLs |
| `v14` | Context width, optional frozen atomic SLAE, physical CA graphs, source removal and bounded communication |
| `v15` | Isolated masked travel depth, ray occlusion, associated empty-space radii and a provisional combination |
| `v16` | Small/current references, bounded numerical HPO and prospectively frozen confirmation sheets |
| `vn1` | Matched C0–C7 matrices with/without ESM, coverage pilot and two secondary controls |
| `vf` | Physical reconstruction controls and from-scratch auxiliary-loss pilot; no external descriptors, no test |

The V14/V15 continuation fixes `transfer_options: {message: low_rank, rank: 16}` from V11b,
with short 16/4 support, one DiffusionNet reading, its gradient descriptor, 32 base modes,
MAX/Heat6 and regional existence 0.3. Every cell preserves that policy, including complete
surface mappings in environment sweeps. This is a provisional reference, not confirmation of
a combined winner. Mean/dispersion, shared value projections, wider support, extra diffusion
readings and gradient removal are not silently combined. The report records their evidence
and remaining confirmation questions. Publish the authored `@9` or SLAE dataset before launch.


From V14b–f through V15, the fixed input policy uses ESM-2 context projection 64 and
SchNet hidden state 128 (output 128), following the V14a 64/128 development selection.
V14a/a2 keep their executed configurations. Swept representation mappings repeat context 64;
the SLAE-only control remains without ESM-2 and its independent atomic projection stays 32.
Do not reinterpret a constructor default of context 32 as this selected override. There is no
automatic V16 winner: its new recipes retain 64/128 and the report's compact 16/64 control.
V16a and V16c vary those widths explicitly; no earlier seeds or scientific designs change.

After V14c, the seven pending V14d–f/V15 YAMLs select the CA-contact residue branch
(`residue_graph: {mode: contact}`) as a development reference, retaining context 64,
atomic state/output 128 and residue state 64 with two layers, radius 10 Å and K=16.
V14d fixes that branch in all thirteen source removals and adds one contemporary
no-branch control (15 configurations × four seeds = 60 Runs). Removing `external_context`
zeros ESM-2 content before both the direct and residue-message routes; closing its gate
alone would not remove both. V14e keeps six communication/control configurations and
uses contacts in each communicating route. V14f confirms no branch, nodewise, contacts
and null neighbor messages on its four fresh seeds (16 Runs). That confirmation did NOT support
the contact branch's operational S improvement. The four V15 YAMLs now use the no-residue-graph
reference, retaining 64/128, rank 16 and the same other controls. SLAE and feedback are not
stacked into the default V16/VN01 reference. Historical V14 designs remain unchanged.

Read `OBJECTIVE`, `PREREQUISITES`, `VARIES`, `FIXED` and `DECISION` at the beginning of each study.
Later-stage shortlists are explicit provisional choices, not automatically imported winners.
Review them against prior evidence before launching. V14/V15 require a newly published @9;
V14b additionally requires the optional SLAE runtime and `enrich_atomic.yaml` publication.
Each V12–V15 study fixes four model seeds across its alternatives; confirmation stages retain
independent seed sets. V12 selects existing checkpoints, while its view seeds describe a separate
geometry factor. The V14a 4x3 width grid now requires 48 Runs, not the initial 24-Run screen.
The optional V13d solver, V14 triple feedback and V15 multi-probe/neck extensions are not silently
enabled. V16b now disables ESM while retaining compact SchNet width 64 and learned identity
tables. Its unchanged adaptive search spans three depths, dropout, learning rate and decay:
up to 128 candidates/300 Runs/168h, adaptive development-seed replication and three finalists
on three separate confirmation seeds. Curve pruning remains active. Start a new Study for this
input-source change; the existing ESM Study and its Results are not edited. No new dataset is
required. Review costs before launch.
V16c/d ship reference values, NOT fictitious selected finalists; copy and document the reviewed
upstream choices before treating those sheets as confirmation/freeze experiments. Test stays sealed.

VN01 (`vn1/wisdom_vn1.yaml`) crosses R+/R− with no mining, top-K, memory and current FGW families:
eight cells × four seeds = 32 full Runs. `wisdom_vn1a.yaml` repeats that matrix with ESM disabled,
retaining learned atom/residue identity tables and every other setting, including the four seeds.
`wisdom_vn1b.yaml` is the renamed ten-epoch coverage/intensity pilot (formerly VN1a);
`wisdom_vn1c.yaml` adds point and distributed controls (formerly VN1b; eight Runs). Both renamed
recipes retain ESM. Historical Results keep their original identities; never recover an old VN1a
pilot using the new 32-Run VN1a ablation. New thresholds/K/lambda
are pilot proposals requiring a common prospective decision. N2 uses LF's `optimal-transport`
extra and requires chemically supported fixed surface attributes. N3 never invents a calibration
partition: an absent independent reserve produces an unavailable audit, not a guarantee or a test
leak. See the two main READMEs §4.6 for configuration, checkpoint and numerical contracts.

`enrich_structures.yaml` publishes @9 from @8 while preserving existing bytes; the full DNA
recipe publishes the same version from the frozen design. Choose only one route per release.
Both retain the reduced training population and do not replace historical study selectors.
The obsolete standalone feature-ablation and sparse-concept experiment recipes were removed;
the independent tested sparse-concept library remains available for deliberate post-hoc work.
Selection/provenance JSON ledgers are retained because they preserve actual scientific choices,
not execution scratch. No `.example` filename is presented as a runnable experiment.

Training files contain one executable Work, never sequential training `steps`. Paired sweeps must
retain their declared candidates and seeds; adaptive HPO is a different design, not an equivalent
way to save time on a fixed comparison. Native LF study extensions preserve the original evidence
when adding compatible candidates; changing scientific parameters is not ordinary failed-Run recovery.

### 2.1. V9f SchNet and V9g attention

V9d compares molecular filters/directional messages; V9e compares vector states. New V9f isolates
SchNet radial representation, internal state width and the existing five-metadata ablation. V9g
is the former V9f sparse/global-attention campaign, with identical algorithms, candidates, seeds
and resources. Historical `wisdom-v9f` results retain that name and their exact identities; use
Execution IDs, not the old name, to review them. `v9/campaign_migration.json` embeds the exact
legacy YAML text and its SHA-256. New SchNet uses the unambiguous Work name `wisdom-v9f-schnet`.

All cells fix 128 output channels, the original chemical inputs/graphs/transfer, four DiffusionNet
blocks, MAX by points, Heat6, BCE+regional existence R=0.3, G-selected checkpoints, and evaluation-only
local GT. All require seeds **173, 179, 181, 191**, without pruning or favorable-seed selection.

| Cell | Block | Encoder/metadata | Radial input | Internal channels |
|---|---|---|---|---|
| 01 | A | Contemporary full baseline | Original baseline | 128 |
| 02–05 | A | SchNet full | Direct, RBF-8, RBF-16, RBF-32 | 128 |
| 06–09 | B | SchNet full | Direct, RBF-8, RBF-16, RBF-32 | 64 |
| 10–11 | C | SchNet restricted | Direct, RBF-16 | 128 |
| 12–13 | C | SchNet restricted | Direct, RBF-16 | 64 |

Cell 04 is the compatible V9d SchNet configuration, not reuse of historical means. The complete
matrix requires **52 Runs**. Gaussian count, filter/input parameter count and atomic state channels
are distinct quantities; direct distance is a category with one radial input, not missing geometry.
Cell numbers identify these exact parameter mappings, not LF Trial ordinals or controller order.
Read [the radial data contract](../README.md#43-schnet-radial-and-width-controls) for equations,
units, gates, cutoff policy, initialization checks and the reduced-width output projection.

Validate/list the complete native design, or author a subset without executing anything:

```bash
lf validate experiments/v9/wisdom_v9f.yaml
lf explain experiments/v9/wisdom_v9f.yaml
lf run experiments/v9/wisdom_v9f.yaml --on YOUR_CLUSTER --dry-run
python scripts/configure_schnet_campaign.py --block B --output .lambdaforge/v9f-block-b.yaml
python scripts/configure_schnet_campaign.py --cells 2 4 --output .lambdaforge/v9f-direct-vs-rbf16.yaml
```

Keep generated YAMLs inside this project so LF resolves its dataset Registry. A block B/C subset
adds cell 01 as its contemporary control; all four seeds remain. A cell-only selection needs
`--cells 4` and still runs all four seeds. These are configuration authoring tools, not extra runners.
Names distinguish subsets; completed coverage of a subset does not establish full-matrix coverage.
Do not silently reuse a historical V9d mean as a new control or combine unrelated seeds.

When ready, launch **one** intended design, then use native LF evidence and reporting:

```bash
lf run experiments/v9/wisdom_v9f.yaml --on YOUR_CLUSTER
lf inspect execution-EXACT --json
lf logs execution-EXACT
lf retry execution-EXACT --dry-run
lf export --help
lf results analyze execution-EXACT --json
lf results report execution-EXACT --output schnet-report.html
lf results table execution-EXACT --metric val_wisdom_hpo_surface --format csv --output schnet-scalars.csv
python scripts/analyze_schnet_campaign.py execution-EXACT --output schnet-audit
```

The native report owns seed aggregation, paired reference differences, coverage, parameter discovery,
curves and Study decisions. Its reference is contemporary baseline cell 01. The optional standalone
SchNet audit reads only native metadata from that exact registered/imported Execution. It implements
the predeclared cell-04 paired contrasts, width comparisons (02–05 versus 06–09), metadata pairs
(02/04/06/08 versus 10/11/12/13), Gaussian-versus-direct width interactions, and relative-cost tables.
It preserves per-seed signs, mean/sample SD and matched support; missing/failed seeds are not zero.
Positive deltas favor larger S/G but not smaller costs or negative-protein activation: interpret
each metric's direction rather than treating every positive sign as a favorable seed.
Its PNG contains radial S/G curves, quality/cost scatter and width/metadata contrasts; JSON/CSV
retain coverage and descriptive two-coordinate dominance only for complete four-seed cells. This
does not fit an explanatory model, search for new effects, choose a winner or change StudyDecision.
For Gaussian-count curves keep direct as a separate category; stratify by internal width. Compare
raw/operational S, top-5% enrichment, negative mass, G, C/W and regret, not only an aggregate mean.
Positive-protein AP SD/quartiles describe heterogeneity between proteins, not seed uncertainty.
Inspect whether the full-attribute Gaussian-count×width contrasts interact before discussing the
sparser metadata contrast. Four seeds support descriptive comparisons, not universal conclusions.

Each Run exports exact common-initialization hashes and capacity under `atomic-encoder-audit/`.
At its best G checkpoint it samples both filter relations/layers with distinct declared metadata
profiles, recording pre/post-support responses, amplitudes, sensitivity and channel correlation in
`radial-filters.json/.npz/.png`. This does not save all edges or use GT, and is not a physical-potential
analysis. Whole-model CUDA peaks and complete-epoch medians are not isolated encoder costs. Optional
isolated timing uses an explicit short synthetic benchmark:

```bash
python scripts/benchmark_schnet.py --mode direct --width 64 --device cpu --output benchmarks/schnet-direct-64.json
```

No real H100 improvement or total-memory saving is asserted before measurement. Concurrent study
timings include contention. No encoder-only production-memory attribution, full-model FLOP estimate
or capacity-matched extra filter-width cell is claimed. The scientific PDF is externally authored;
this code/configuration update does not rewrite it or mark this unexecuted design as a result.

### 2.2. Frozen V10 backbone from V9f

Current V10a–h configurations inherit full-metadata SchNet with Gaussian RBF-16, 64 internal
channels and 128 output channels, frozen from `full_rbf16_64` (V9f cell 08,
`execution-d384d8f6b5064a26`). The source comparison completed 13 cells and four paired seeds per
cell. [backbone_selection.json](v10/backbone_selection.json) records the chosen parameters, exact
external evidence digests, four Run IDs and paired comparisons against the original control and
the strongest surface/compact alternatives. No historical results or registered decisions change.

The selection prioritizes a small surface-quality gap (S 0.2932 versus 0.2966 for restricted direct
128), fewer parameters (87,488 versus 281,728 in the encoder), lower false activation on negative
proteins, and lower observed allocated/reserved peaks. Full edge metadata remain available while
V10 changes atomic input sources. It is not a speed winner: mean complete-epoch medians were 119.5
versus 103.2 seconds on shared H200 NVL hardware, not isolated encoder measurements. Its G 0.7030
also falls below the contemporary original baseline's 0.7134 in all four paired seeds. Treat this
as an explicit provisional development choice needing new-seed confirmation, not non-inferiority.

All V10 controls and interventions use the same frozen SchNet and train from scratch with the
existing V10 seeds; none loads a chosen V9f seed's weights. The empty `atomic_representation: {}`
control now means this SchNet without added input sources. Changing the choice later requires
changing all matched controls together; historical baseline means cannot substitute for them.
The enriched @8 selector remains a publication prerequisite, not a silent switch back to @6 or
a claim that their geometry/GT match. New source projections preserve the actual 64-channel atomic
state before the final 64→128 projection. The explicitly superseded legacy confirmation template
is not modified or treated as a current V10 representation experiment.

### 2.3. V10f Hard-Concrete source selection

V10a–e are frozen. The new [V10f](v10/wisdom_v10f.yaml) tests grouped source selection;
the formerly planned physical-initialization V10f is now [V10g](v10/wisdom_v10g.yaml), the external
representation V10g is [V10h](v10/wisdom_v10h.yaml), and the future confirmation template is
[V10i](v10/wisdom_v10i.yaml), now a compact pooling recheck after contextual-input selection.
Historical execution identities and their files are not renamed.
The frozen V10e comment naming confirmation V10h describes its old numbering, not a new execution.

V10f uses V10e's actual executed fixed configuration, checked against its submitted YAML and
full-source checkpoint (48 gates), not a generic template. The native sweep enumerates scalar
scientific coordinates because LF's explicit candidates reject mappings. It declares exactly
20 profiles and four fresh paired seeds, 269/271/277/281: **80 mandatory Runs**, no pruning or test.
The single native reference is the full-source immediate 0.04/0.04 condition.

| Profiles | Changed scientific controls | Count |
|---|---|---:|
| A1–A9 | S/C coefficients: each pair from {0.04, 0.16, 0.64}; O stays 0.04 | 9 |
| B1–B3 | Initial S/C activity: (0.50,0.95), (0.95,0.50), (0.50,0.50); S/C cost 0.16 | 3 |
| C1–C3 | S/C warm-up/ramp fractions: (0,0.02), (0,0.04), (0.01,0.02); S/C cost 0.16 | 3 |
| G1–G2 | Uniform global coefficients 0.16 or 0.64, including O | 2 |
| All-on / all-off / compact | Seven chemical gates fixed open/closed, or only the two identity sources present | 3 |

At initial analytic activity 0.95, expected path sums are S=19, C=6.3175 and O=18.684125.
Thus their direct penalties are respectively `19*lambda_S/41`, `6.3175*lambda_C/41` and
`18.684125*lambda_O/41`. Uniform 0.04 totals about 0.0429284; uniform 0.16 about 0.1717137;
uniform 0.64 about 0.6868546. These are expected costs, not sampled gate values or prediction
metrics. Global and grouped conditions are not automatically penalty-budget matched. Fixed
chemical gates have zero direct penalty; compact has only two chemical gates and different capacity.
Localized warm-up makes the actual first-epoch S/C costs zero, not their nominal final values.

The [preparation record](v10/gate_study.json) records all 20 profiles, their initial group budgets,
source V10e identity, local seed-conflict check, Git HEAD/dirty qualification and frozen YAML digests.
The `gate-audit` manifest records actual initial budgets per group for every Run. Its epoch JSONL
and separate final/selected records prevent confusing last-epoch gates with the G-selected model.
Analyze feature selection and localization separately: paired S/G changes, negative evidence,
closure/attenuation, seed variability and resource costs. The 3×3 A block supports paired main
effects and interaction contrasts; a statistical interaction is not itself biological causality.
No surface target enters gradients or epoch selection; none of these controls proves universality
beyond this dataset. See both main READMEs §4.4 for the public parameter and inference contract.

### 2.4. Continuation after V10g/V10h

The [selection record](v10/representation_selection.json) fixes habitual (`current`) table
initialization plus frozen ESM-2 35M context projected to 32 channels. Keep name and residue-atom
identity at eight channels each, learned gates, uniform cost 0.04/41 and SchNet 64→128. The residue
table is neither externally initialized nor removed. No cross-extractor table/context mixture
was selected. The same explicit baseline now appears in every V11/V13 training YAML.

Run [V10i](v10/wisdom_v10i.yaml) to compare MAX, fixed LSE beta 0.25/area and simple Attention
256/area under that baseline. It has three settings and four fresh seeds, **12 mandatory Runs**,
with the same Heat6, single head and regional prior 0.3. This is a conditional pooling comparison,
not a new optimization sweep or a confirmation of six external extractors. MAX remains provisional
for downstream designs until these results are reviewed. Cost comparison includes memory,
complete seconds per epoch and completed epochs; S stays the scientific priority.

V12 selects ESM-2 context from the completed imported V10h study and restores its checkpoint
configuration without fitting. Isolated protocols rebuild surfaces from the exact base; only the
legacy published protocol needs a surface-view asset. V11f/V13f confirmation
seeds are disjoint from V10i so its pooling choice is not confirmed on its own selection seeds.
Historical V10g/V10h inputs and results are unchanged. YAML edits never start a computation.

## 3. Reviewing saved models

`v*/visualization/*.yaml` and `reviews/*.yaml` run checkpoint reviews, not hidden training. Declare
the imported execution or typed model set, exact dataset, saved seed and desired parameter filter.
The automatic protein panel includes favorable, unfavorable and inconsistent cases. Metrics use
the complete declared evaluation scope, independently of display sampling.

Ordinary training reports belong in `with.visualization` on the training YAML. They are generated
at that Run's restored best checkpoint; they are not a Study-wide best-seed callback. Missing
checkpoints require LF recovery. No project reviewer silently retrains or revises Study decisions.

## 4. Preflight and execution

From the repository root, substitute the intended file and configured target:

```bash
lf validate experiments/v9/wisdom_v9d.yaml
lf explain experiments/v9/wisdom_v9d.yaml
lf run experiments/v9/wisdom_v9d.yaml --dry-run
lf run experiments/v9/wisdom_v9d.yaml --on YOUR_CLUSTER
```

Dry runs do not train. Inspect the current dataset placements and granted resources before the
last command. Leave held-out test disabled during development unless the protocol explicitly
authorizes final evaluation. Compare global and surface outcomes separately, including missing
support, seed variance and cost; the report, not this catalogue, interprets the resulting evidence.
