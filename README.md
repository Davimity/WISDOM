# WISDOM — protein surface learning

[Español](README.es.md)

WISDOM learns a score at each point of a protein's surface from labels describing the **whole
protein**. The first benchmark asks whether a protein binds DNA. The intended result is not only
a correct yes/no answer, but also a useful map of the likely binding region. These are different
goals: good protein classification does not establish good localization.

This README covers installation, files, execution and code organization. The
[scientific report](docs/WISDOM-memoria.pdf), currently in Spanish, contains the derivations,
experimental reasoning, results and limitations. It distinguishes implemented capabilities from
proposals and validated findings. YAML comments document each experiment's exact choices.

## 0. Contents

- [1. Quick start](#1-quick-start)
- [2. Installation](#2-installation)
  - [2.1. Local and cluster environments](#21-local-and-cluster-environments)
  - [2.2. Updating the scientific report](#22-updating-the-scientific-report)
- [3. Building and inspecting datasets](#3-building-and-inspecting-datasets)
  - [3.1. DNA-binding benchmark](#31-dna-binding-benchmark)
  - [3.2. Reuse, versions and zinc](#32-reuse-versions-and-zinc)
- [4. Technical contracts](#4-technical-contracts)
  - [4.1. Structural arrays and annotations](#41-structural-arrays-and-annotations)
  - [4.2. Models, supervision and metrics](#42-models-supervision-and-metrics)
  - [4.3. SchNet radial and width controls](#43-schnet-radial-and-width-controls)
  - [4.4. Grouped feature selection](#44-grouped-feature-selection)
  - [4.5. Frozen geometry audits and multiscale inputs](#45-frozen-geometry-audits-and-multiscale-inputs)
  - [4.6. Consolidation and negative-region supervision](#46-consolidation-and-negative-region-supervision)
  - [4.7. VF physical targets](#47-vf-physical-targets)
- [5. Experiments and development](#5-experiments-and-development)
  - [5.1. Training and protein viewers](#51-training-and-protein-viewers)
  - [5.2. Source organization and verification](#52-source-organization-and-verification)
- [6. References](#6-references)

## 1. Quick start

From the repository root:

```bash
./install.sh
conda activate wisdom
lf validate experiments/preprocess/dna/dna_preprocess.yaml
lf explain experiments/preprocess/dna/dna_preprocess.yaml
lf run experiments/preprocess/dna/dna_preprocess.yaml --dry-run
```

Validation and the dry run plan the work; they do not build a dataset or train a model. The current
DNA YAML reuses `data/dna/design`, builds the 25% training view and retains the configured evaluation
splits. It stages the optional model bundle from `data/models`. Review those inputs and the target
dataset name/version before launching. See §3 for reuse and §4 for the resulting files.

When ready, the researcher launches explicitly:

```bash
lf run experiments/preprocess/dna/dna_preprocess.yaml --on YOUR_CLUSTER
lf overview
lf logs wisdom-dna-preprocess --follow
```

For training, select an experiment from [the experiment catalogue](experiments/README.md), make
sure its dataset is registered on the execution target, then validate and launch that YAML.
Experiment numbers describe scientific stages, not the Python model generation.

## 2. Installation

### 2.1. Local and cluster environments

Use `./install.sh`, not a standalone Python virtual environment. The script reuses Conda or offers
to install Miniforge, prepares the `wisdom` environment, and installs WISDOM and LambdaForge in
editable mode. It reuses a local LambdaForge checkout when available. `./install.sh --yes` accepts
the default setup. Requirements are recorded in `pyproject.toml` and `environment.yml`; the project
currently requires LambdaForge >=0.19.0 without an upper version bound.

Conda also supplies **MMseqs2** and **Foldseek**, the command-line tools used to identify sequence
and structural similarities during selection. **Gemmi** reads deposited coordinates. PyTorch
executes the trainable models; CUDA enables supported NVIDIA GPUs. A GPU is not required for
configuration checks or the CPU unit tests.

```bash
conda activate wisdom
python -m pip check
command -v mmseqs
command -v foldseek
lf doctor
```

Remote environments belong to the project rather than a hard-coded Conda path. Prepare a configured
cluster with `lf clusters bootstrap YOUR_CLUSTER --project .`; this uses the Conda contract in
`[tool.lambdaforge.environment]`. Run it again after relevant dependency changes. For local updates,
rerun `./install.sh`. External protein checkpoints are separate staged inputs, not downloads hidden
inside training. Their inventory is documented in [data/models/README.md](data/models/README.md).

### 2.2. Updating the scientific report

`docs/WISDOM-memoria.pdf` is a **real copy**, not a link to the author's machine. Its companion
JSON records the filename, SHA-256 content digest and size. The report is maintained externally;
WISDOM does not rewrite its scientific results or automatically know when that source changes.
After compiling an updated report, synchronize and verify it before committing:

```bash
python scripts/sync_report.py /path/to/Memoria/main.pdf
python scripts/sync_report.py /path/to/Memoria/main.pdf --check
```

The check fails if the bundled bytes or metadata are stale. File dates are not used as evidence of
equality. Keep operating instructions here, experiment settings in their YAMLs, and detailed
scientific discussion in the report; do not add a new Markdown diary for each execution failure.
Raw local performance measurements belong in `benchmarks/`, not in the scientific conclusions.

## 3. Building and inspecting datasets

### 3.1. DNA-binding benchmark

A deposited structure in the **Protein Data Bank**, queried through **RCSB**, describes coordinates
observed under particular experimental conditions. Missing DNA in that structure does **not** prove
that the protein cannot bind DNA. WISDOM therefore requires explicit curated negative evidence.
BTD and BTD-Combo are the binding-benchmark sources used by source curation; a benchmark label is
retained with its evidence and limitations, not treated as universal proof of non-binding.

| Stage | Input | Work performed | Result consumed next |
|---|---|---|---|
| Source curation | Benchmark records and deposited structures | Resolve evidence and check whether local reference data can be obtained | Typed `raw.jsonl` evidence |
| Selection | Frozen evidence | Verify DNA contacts, group related proteins, characterize shapes, balance and split | `design/` portable snapshot |
| Preprocessing | Exact selected design | Build structural arrays and separate DNA annotations; validate and publish | Registered immutable dataset |
| Visualization | Registered dataset | Render structural inputs for inspection | HTML gallery |

`scripts/create_fasta.py` prepares source evidence; despite its historical name, the selection
interface is typed JSONL, one JSON record per line. Use `python scripts/create_fasta.py --help`
for its current acquisition options. A positive chain must contact DNA in the selected **biological
assembly**, the molecular complex interpreted as biologically relevant rather than arbitrary crystal
neighbors. Contradictory evidence is quarantined, not silently relabeled.

Selection uses MMseqs2/Foldseek and identity connections to form **leakage groups**: related members
must remain in the same split so evaluation cannot exploit near-duplicates from training. Separately,
HDBSCAN groups physical shapes into **phenotypes**. Positive clusters describe local binding-site
shape; negative clusters describe global morphology. These clusters guide coverage and auditing,
not homology or functional labels. Unclustered members remain explicit.

Only after those groups exist does selection fix train/validation/test and create nested training
**dilutions**, meaning smaller subsets of the original training population. Validation and test are
not diluted. Exact class balance can be limited by indivisible leakage groups; inspect the generated
`REPORT.md` and statistics rather than assuming every split is exactly 50/50.

The portable design includes labelled TXT files, `catalog.csv`, dilution membership, manifests,
the selected coordinate snapshot and its index. A labelled line is `1ABC_AQ<TAB>1`: one chain named
`AQ`. `1ABC_A_Q` means chains `A` and `Q`. Labels are `0` or `1`; splits are explicit. An unlabelled
identifier list alone cannot reconstruct assembly/contact evidence.

### 3.2. Reuse, versions and zinc

The DNA YAML has independently skippable `select`, `preprocess` and `visualize` steps. These are
the supported modes; change the commented recipes in the existing YAML, not the algorithms:

| Intent | Selection | Preprocessing | Visualization |
|---|---|---|---|
| Rebuild from frozen evidence | `skip: false`, set `raw_path` | `skip: false` | Optional |
| Reuse an audited design | `skip: true`, set `existing_design` | `skip: false` | Optional |
| Inspect an existing dataset | Skip, with no design input | `skip: true` | Set a registered dataset selector |

`{from: select.train}` refers to the output named `train` declared by the Selection Work, not a
filename guessed by LambdaForge. The downstream Work receives its staged path. Likewise,
`{from: preprocess.dataset}` resolves the dataset created during this workflow; do not replace it
with a version that does not exist yet. Reusing a design still requires its complete snapshot,
not just `train-labelled.txt`.

Managed datasets are immutable and registered by name/version. The current DNA recipe targets
`wisdom-dna-reduced@8`; older studies can deliberately use `wisdom-dna-reduced@6`. Do not silently
substitute one for another. A dataset built with only train-25 uses `subset: full` in training;
`replicate-00/train-25` is appropriate only when that version publishes it as an additional view.
New structural families are separate, content-bound files; enrichment is not a promise that an
older base equals rebuilding with changed geometry settings.

```bash
lf datasets list
lf datasets show wisdom-dna-reduced@6
lf datasets stats wisdom-dna-reduced@6
lf datasets verify wisdom-dna-reduced@6
lf datasets materialize wisdom-dna-reduced@6 --on YOUR_CLUSTER --strategy replicate --apply
lf validate experiments/preprocess/dna/validate_dna.yaml
```

The last YAML performs a scientific audit, separate from LF's byte-integrity checks. Compatible
retries reuse validated per-protein checkpoints; timestamps alone neither authorize reuse nor
invalidate identical bytes. A changed RCSB revision requires explicit new selection, not replacing
the coordinates inside an audited design.

Structural enrichment saves a validated native file checkpoint for each protein and generated
family before publishing the complete dataset. A few failed members do not delete successful
files, but the destination version is not published until every member succeeds. Recovery must
restore the original Run's checkpoints; an ordinary new submission is not equivalent. Check LF's
recovery preview before applying it. Some LF versions recover Studies but resubmit single Works
without restoring their checkpoints: if the preview lacks an explicit recovery plan, request
native single-Work recovery rather than copying internal directories or rerunning everything.

Free-space features use an allocation ceiling of 18 million voxels by default, at unchanged 1 Å
spacing; large domains need substantial host RAM (the enrichment recipe requests 96 GiB). A voxel
is one cell in the spatial grid. The ceiling rejects an oversized grid; it never truncates it or
silently coarsens its resolution. Raising this ceiling does not change computed features.
Enrichment recognizes the historical two-million-cell checkpoint keys and revalidates those exact
files against all physical settings and parent checksums. The actual original ceiling remains in
their metadata. Other physical changes, such as spacing or probe radius, cannot reuse them.

For zinc, use `experiments/preprocess/zinc/zinc_preprocess.yaml`. It composes discovery, selection,
preprocessing, validation and visualization. `zinc_discovery.yaml` is the standalone acquisition
entry point, useful when freezing evidence separately. Zinc negatives require explicit supported
negative annotations, never absence of a metal atom. Discovery can acquire these automatically;
it stops **before bulk positive downloads** unless at least three distinct negative sequences
remain (`minimum_negative_sequences: 3`). Several structures of the same sequence cannot fill
independent train/validation/test roles. Three sequences are only a necessary lower bound: later
full-RAW grouping must still leave at least three eligible negative leakage groups, and every
split must contain both classes. The `negative-feasibility.json` and `negative-discovery.json`
artifacts explain counts and exclusions. More negatives require additional explicit experimental
evidence, not relabeling unknown proteins. DNA labels never enter the zinc task.

**Reproducing the original TFG zinc benchmark.** To compare the current representation and models
with the undergraduate thesis (TFG), use the separate historical recipe:

```bash
lf clusters bootstrap citius-ctgpgpu16 --project .
lf validate experiments/preprocess/zinc/zinc_legacy_preprocess.yaml
lf run experiments/preprocess/zinc/zinc_legacy_preprocess.yaml --on citius-ctgpgpu16
# After publication, validate on the same target:
lf run experiments/preprocess/zinc/validate_zinc_legacy.yaml --on citius-ctgpgpu16
```

It publishes **`wisdom-zinc-tfg@1`**, independently of modern zinc discovery. The six original
`data/zn_old/*.txt` files remain unchanged: `positives.txt` and `negatives.txt` supply labels;
`train.txt`, `val.txt` and `test.txt` supply partitions; `all.txt` checks coverage. They contain
450 unique proteins: train 148 (74 per class), validation 148 (74 per class), and test 154
(77 per class). No selection, balancing, homology regrouping or dilution runs. DNA's train-25
view is **not** applied. Overlapping splits, conflicting labels, duplicate IDs or missing
structures stop the action rather than silently removing or replacing a protein.

The recipe uses the same universal base settings and structural assets as the current DNA `@9`
recipe: five curvature scales, 12 Å/K96 expanded atom neighborhoods, residue graphs, masked
depth/occlusion/free-space fields, extended spectral modes and an alternate surface. It also enables
the selected ESM-2 residue representations and reviewed SLAE atom/residue representations used by
the enriched workflow. The pinned `data/models` bundle and
`data/external-models/slae/autoencoder.ckpt` must be present on the target as for DNA preprocessing.
Geometry uses bounded CPU workers; frozen encoders use only the granted logical GPU. Zn ions
never enter neural inputs. These are new representations of the historical population, not a
claim to reproduce the TFG's original coordinate files or preprocessing byte for byte.

The typed `legacy_train`, `legacy_validation`, `legacy_test`, `legacy_positives`,
`legacy_negatives` and optional `legacy_all` inputs replace `design`; modern Selection inputs
cannot be combined with them. A portable `legacy-design` artifact preserves the original TXT
bytes, their content digests and every inherited decision. First-observed RCSB coordinates become
durable checkpoints reused on compatible native recovery. The first declared biological assembly
containing the exact historical chain and its first copy are chosen deterministically, never by
label/site agreement. A missing assembly fails explicitly rather than guessing another chain.
Stored coordinates with zero occupancy are retained for this historical representation and its
geometric assembly alignment, but do not become occupied Zn-donor evidence. An entirely
zero-occupancy chain (for example `1C78_B`) is flagged in the validation report and has no usable
local GT. The modern selection path still requires occupied atoms for its alignment/evidence.

Before costly structural enrichment, WISDOM checks every observed-to-deposited residue mapping.
Both sides use Gemmi's same chemical dictionary: modified residues such as MSE, PTR, PCA and LLP
map to their known parent amino acids M, Y, E and K. The original three-letter residue names and
atom chemistry stay unchanged. Unknown monomers remain X, and a real identity or sequence-position
disagreement still fails with the chain, author address and deposited position; no approximate
sequence alignment is used. Sequence preflight failures publish all affected IDs up front.

If enrichment stops later, compatible native recovery revalidates and reuses each completed
geometry/annotation/structural file checkpoint, including completed families of failed proteins.
It does not rebuild the other proteins' surfaces, curvatures or spectral operators. Reuse is not
instantaneous: scientific validation still checks atomic solvent exposure against the same
spherical samples and reproduces alternate-view point sampling. Already buried exposure samples
are discarded from subsequent containment tests without changing the resulting areas. Surface
burial tests run in bounded sparse batches; pairs close to a containment threshold retain the
original scalar calculation, preserving the exact point-selection decisions. Native
Map restoration counts in the log describe this attempt; a saved record's original processing
status is not evidence that it was rebuilt. A **new** submission is not recovery and may
start over. Do not delete the failed Job or run storage cleanup before recovery succeeds. If LF
rejects recovery because it renamed staged TXT files despite identical bytes, that framework
identity problem must be fixed without changing the original scientific inputs or manifests.

Surface normals are checked against all nearby physical spheres, not just the model's limited
nearest-atom table. Alternate views reuse their deterministically reproduced, unrounded points
for the normal check: rounding stored coordinates to 32 bits can otherwise toggle an atom across
the normal blend's hard inclusion cutoff. The reference must match stored positions within four
float32 rounding units, scaled by the larger of 1 Å and the maximum absolute stored coordinate;
one unit is the relative spacing `2^-23`. The cosine threshold remains 0.99. Stored geometry,
operator checks and rejection of inverted normals remain active. No archive is rewritten by this
validation correction. A disputed base-surface normal triggers the same precise check only if
the original sampler reproduces every stored point exactly after float32 rounding.

Labels are marked `legacy_tfg_manifest`, **not experimentally confirmed negatives**. Contacts
recover evaluation-only surface references without changing global labels. A positive without
usable local ground truth (GT), or a negative contradicted by a Zn contact, has unavailable local
GT and an empty evaluation mask, not fabricated local labels. Validation checks exact original
membership/labels/splits, geometry, sidecars and enabled assets. Its report explicitly sets
`homology_independence_verified: false`: repeated sequences are warnings, not repaired splits.
A PASS certifies this reproduction contract, not modern negative curation or independence from
homologues. The modern discovery/selection route retains its strict evidence requirements.

The zinc recipe now publishes **`wisdom-zinc@3`**. Discovery downloads depositions through LF's
shared cache with threads, then inspects files with bounded processes (`structure_executor:
process`; `workers` bounds concurrency). Each biological assembly builds one spatial donor index, a lookup of
nearby protein atoms, shared by all chain-copy summaries. Every candidate, assembly alternative,
acceptance threshold and zero-contact chain is preserved: no RAW leakage bridge is removed.
Selection still records complete coordination partners and uses the same task-independent
sequence/structure similarity, group splitting and nested-dilution algorithms as DNA. Site shapes
and negative evidence remain task-specific.

Descriptive UniProt annotations are transferred only when the returned full sequence exactly
matches the deposited chain. A search hit without a sequence remains listed in
`uniprot_sequence_unavailable`; it contributes no UniProt function or family. This does not alter
the zinc label, which comes from independent coordination or explicit negative evidence.
Concurrent metadata readers share LF-validated **durable source checkpoints**; Discovery copies
each distinct response into the portable source audit once, after the parallel phase finishes.
Mutable inventories, evidence responses, metadata and decoded coordinate bytes are pinned as soon
as acquired, not only after the entire Work succeeds. LF owns locking, downloads, integrity and
atomic checkpoint publication. The first URL and byte digest remain fixed: if a missing source
can only be reacquired with different bytes, recovery stops instead of mixing public revisions.
This adds durable storage alongside the disposable download cache; deleting failed-Run checkpoints
still destroys this recovery boundary. It is not a promise that unrelated fresh Runs share data.

`progress_log_seconds: 120` reports phase liveness; LF shows completed/total map items. Downloads
and inspections have separate native per-entry recovery records. Inspect LF's native recovery
preview: it must preserve the original Execution/Run and durable dependencies, including through
an explicitly accepted code revision. The standalone `zinc_discovery.yaml` remains useful for
freezing acquisition separately. A fresh submission does not inherit an old Run's records.
Legacy cache-only Attempts need verified native migration of their original observations before
reuse; WISDOM refuses a fresh inventory over old inspections. Cached files already deleted cannot
be recovered from counts alone. LF also needs dependency-aware collection and shared HTTP-429
cooldowns; WISDOM does not implement a second garbage collector or HTTP retry scheduler. Until
that framework fix is available, do not assume another retry will retain old cache dependencies.
After success, freeze expensive boundaries: set `discover.skip: true` and uncomment
`existing_raw` to reuse `data/zinc/raw`; or remove discovery, set `select.skip: true`, uncomment
`existing_design` and comment `raw_path` to reuse `data/zinc/design`. Keep preprocessing enabled
to publish a new version with richer assets.

Preprocessing requests the **same universal families and scientific settings as DNA**: atomic,
sequence and covalent context, solvent exposure, 12 Å/K96 transfer candidates, residue contact and
alpha-carbon graphs, free-space properties, surface topology, 256-mode extension, alternate
surface and five curvature scales. Both DNA and zinc recipes now select only **ESM-2** residue
context, the source retained by V14/V15. ESM-C, SaProt, static initialization tables and unused 3Di
are disabled by default, without removing their implementations. Stage the identical pinned
`data/models` directory in the remote project mirror and install/bootstrap first, just as for DNA.
Missing required assets stop preflight or publication
rather than disappearing silently. Zinc also enables the reviewed SLAE base checkpoint at
`data/external-models/slae/autoencoder.ckpt`; mirror these exact weights before submission and
bootstrap the isolated `slae` tool runtime. Its GPU extraction runs one protein at a time after
parallel structural generation; `workers: 12` still applies to CPU geometry, not concurrent SLAE
models. The recipe explicitly requests one GPU and uses the granted logical CUDA device for
the selected residue extractor too. The reader removes Zn and other non-protein atoms **before**
geometry, exposure, graphs, tokens and representations are computed. Exact sources retain Zn
for evidence and evaluation targets only; those source coordinates and sidecars are not model inputs.

`python scripts/benchmark_zinc_discovery.py` reproduced identical summaries and measured about
1.3×/8.8×/44× faster inspection for synthetic assemblies with 1/8/32 copies; raw timings are in
`benchmarks/zinc_discovery.json`. These are not whole-pipeline speedups. At the default ceiling
of two coordinate requests per second, spacing 28,033 uncached requests alone needs about 3.9
hours; transfer, metadata, full-RAW homology and final geometry add cost. No sampling, relaxed
leakage threshold or higher request rate is used. Complete structural assets also cost more than
geometry alone: the YAML requests CPU/RAM/storage/GPU explicitly and uses no ungranted device.

The default Zn publication contains contextual ESM-2 vectors (480 values per physical residue)
and SLAE vectors (128 values per supported observed atom and per supported residue). Explicit
validity masks identify missing or unsupported entities; no missing atom is invented.
Residue vectors can be mapped to atoms by
their stored parent indices without pretending they are independently atom-specific predictions.
These are **not ESM-3 embeddings**: the `esm` package used inside SLAE's isolated environment does
not imply that WISDOM extracts a separate ESM-3 model.

To repeat the earlier model comparison, change `protein_model_kinds: [esm2]` to a list containing
`esmc` and/or `saprot`; the selected kinds alone are checked, loaded and generated. SaProt
automatically requests its official Foldseek 3Di input. Set `protein_static_tables: true` only
when studying static-table initialization. These are recipe overrides: the component's historical
defaults remain `protein_model_kinds: null` (all three) and `protein_static_tables: true` for
existing explicit bundle users. Required asset checks follow the chosen subset, not an all-model
assumption. Apply a changed recipe to an unpublished `dataset_version`; existing immutable
datasets keep their original assets and size. No old dataset, cached file or downloaded checkpoint
is deleted by changing these defaults. DNA retains SLAE as a separate `enrich_atomic.yaml` action.

V15's `surface_environment` resource is generated for Zn with the same protein-only algorithm:
travel depth (free-space path to the exterior), Euclidean depth (straight-line exterior distance),
near/far occlusion (fractions of outward rays blocked within 4/15 Å), and empty radius (size of
associated local free space). Every value has a validity mask; closed cavities, bulk solvent and
association offsets remain explicit. All fields follow the base surface point order. The
`mask_only` and `curvature_control` V15 alternatives are training-time controls over these assets
and existing curvatures, not extra task-specific preprocessing or labels.

## 4. Technical contracts

### 4.1. Structural arrays and annotations

Each protein has a universal **NPZ**, a compressed collection of named numerical arrays. It contains
no Python objects: read with `numpy.load(path, allow_pickle=False)`. Structural metadata records
the exact source, configuration and schema. The dataset's `index.jsonl` joins each member to its
split, global label, archive and optional task-specific files.

| Array family | Meaning and downstream use |
|---|---|
| `atom_positions`, `atomic_numbers`, chemical descriptors | Atom coordinates in Å and fixed identities/properties for atomic encoding |
| `atom_edge_index`, distances, bond fields and spatial ranks | Sparse covalent/spatial graph; each undirected pair is stored once with `src < dst` |
| `surface_positions`, `surface_normals`, `surface_area_weights` | Sampled protein boundary, outward unit directions and represented area in Å² |
| `surface_curvatures` | Mean curvature, Gaussian curvature and curvedness at each configured scale |
| `surface_atom_*` | Bounded nearest-atom candidates, distance/orientation geometry and a validity mask |
| `surface_neighbors`, masks and distances | Fixed local surface neighborhoods for applicable consumers |
| `diffusion_*` | Surface masses, spectral basis and sparse derivatives used by DiffusionNet |

Coordinates are centered consistently, not rescaled arbitrarily. Curvatures describe local bending;
their transforms and runtime Shape Index are implemented in the surface model. Missing neighbor
slots use the accompanying mask, not an invented atom. The historic base transfer table remains
6 Å/K32 in the current recipe; the separately enabled extended table supports 12 Å/K96 without
rewriting that base contract. Additional structural families include sequence/covalent context,
atom exposure, residue/surface graphs, spectral extensions and alternate surface samples.

Optional frozen protein-model representations are declared assets with explicit extractor weights
and revisions. Training can request them, but they are not silently activated by their presence.
These are distinct from generic geometry. Some extractors require large checkpoints and appreciable
CPU/GPU time; enabling everything is not free.

DNA ground truth (GT), the local reference used to judge a prediction, is kept in a separate
fingerprinted file aligned exactly with the base surface's point order. A hard target is binary;
a soft target varies continuously near the contact boundary. Unknown regions are not fabricated
as negatives. Base geometry contains neither DNA atoms nor local targets as neural inputs. Zinc
uses its own annotation contract over the same universal geometry.

### 4.2. Models, supervision and metrics

The common path is **atomic graph → atom-to-surface transfer → surface encoder → point evidence →
protein pooling**. The atomic graph exchanges information between neighboring atoms. Transfer
weights nearby atoms separately for each surface point. DiffusionNet propagates surface information
using precomputed geometric operators. A point head produces logits, real-valued scores converted
to probabilities; pooling combines them into a whole-protein score.

`WisdomV1`, `WisdomV2` and `WisdomV3` expose related experimental capabilities, not the sequence
V1–V13 of scientific studies. Alternative graph encoders, poolings and feedback paths are explicit
parameters. Semantic Hard-Concrete gates learn whether a meaningful source of information is useful;
their regularization discourages unnecessary active sources. This refactor changes none of those
equations, module attributes, checkpoint parameter names, defaults or loss weights.

Training uses global protein labels. Optional weak losses constrain surface behavior using those
same global labels and geometry, not local GT. Validation selects a checkpoint under the experiment's
declared policy; test is evaluated only when enabled and is never a tuning source. Protein ownership
`atom_batch` survives device transfer so directional and bounded global encoders cannot treat a batch
of independent molecules as one large molecule. Per-protein guards are not total VRAM guarantees.

V9g alone explicitly permits dense global atomic attention. Its recipe allows 1 GiB for each
principal path-code or attention-score tensor, rather than the constructor default of 128 MiB.
On the registered @6 dataset the maximum is 942,127,232 bytes (10,852 atoms); 74 of 589 members
exceed the default. Two-head FP32 scores and int64 path codes each require 8 bytes per atom pair.
WISDOM reports ownership, atom count, required bytes and limit before allocating these tensors.
The no-structure control supplies zero structural codes directly, avoiding unused shortest-path
searches. This does not change its equations or parameters. Dense attention still scales
quadratically; layers, saved gradients and concurrent Runs need additional memory. Neither this
bound nor mixed precision guarantees H100 fit. LF currently treats this bound as a scientific
argument: a changed bound can prevent native recovery from reusing completed Runs even though
model equations are unchanged. Do not bypass that identity check or truncate proteins silently.

For `structural_global`, Training prepares exact LF shortest-path/degree codes **before neural
execution**, once per member and relation. LF owns file locking, integrity and atomic publication;
the key binds node order, topology, active K and the encoding implementation, never labels or GT.
Layers, epochs and validation views reuse these files; compatible Attempts can restore them.
LF's current cache scope includes the Run/seed: this is not Study-wide sharing. Initial preparation
can still take time, with member-by-member logs. Code buckets 0–31 use lossless uint8 storage and
host loading, then become int64 once per device batch, shared by all layers. Missing prepared
operators fail explicitly; no slow BFS fallback remains inside `forward`. Frozen checkpoint
review/concept extraction uses the same preparation; learned weights and state-dictionary names
are unchanged. No dataset regeneration or new YAML parameter is needed. Global attention itself
is still quadratic: this removes repeated graph preparation, not its intrinsic GPU cost.

Inspect both global and local metrics. AUPRC summarizes precision versus recall as the decision
threshold changes and depends on positive prevalence. MCC measures binary agreement using all four
confusion-table counts. Surface metrics compare point scores to available GT; area weighting avoids
letting dense sampling dominate. Undefined values retain support counts rather than invented zeros.
Read each study's objective: a diagnostic surface score is not automatically an optimization target.
The report explains normalized local scores, composite utilities and their limitations in detail.

### 4.3. SchNet radial and width controls

V9d leaves open whether SchNet benefits from its distance representation or simply its capacity.
The new **V9f SchNet** study compares direct distance and Gaussian distance features, internal
atomic widths 64/128 and full/restricted edge metadata. V9g now names sparse/global attention;
historical `wisdom-v9f` executions remain attention evidence. The new Work name is
`wisdom-v9f-schnet`, deliberately distinct. The unchanged historical YAML and its SHA-256 mapping
are retained in `experiments/v9/campaign_migration.json`; no stored results are renamed.

A Gaussian radial basis function (RBF) is a smooth response centered at a chosen distance. For an
edge of length $d$ in angstroms, $R$ features place centers $\mu_k$ between 0 and 6 Å. With center
spacing $\Delta=6/(R-1)$ Å, their values are:

$$
b_k(d)=\exp[-\gamma(d-\mu_k)^2],\qquad \gamma=0.5/\Delta^2.
$$

For RBF-16, spacing is 0.4 Å and gamma is 3.125 Å⁻², exactly the V9d LF basis. At $d=2$ Å the
responses centered at 1.6, 2.0 and 2.4 Å are approximately 0.607, 1 and 0.607. Direct mode instead
supplies the single value $d/6=1/3$. Neither adds new geometric information: the experiment tests
whether expansion helps learn filters, not whether more features must be better.

For each relation, the unchanged LF filter network takes radial features `[E,R]` and its existing
gated edge attributes `[E,A]`, producing filters `[E,H]`. Here E counts directed edges, A is 3 for
covalent/4 for spatial relations, and H is the **internal atomic width**. Projected sender states
`[E,H]` multiply those filters channel by channel; messages sum at receivers before LF's residual
update. Gaussian and direct modes use the same two-layer SiLU filter structure. Its parameter count
is `H*(R+A+1) + H*H + H`, so changing R changes capacity as well as representation. The original
scalar edge-distance attribute is retained, including in direct mode; no other source is removed.

`atomic_radial_encoding` defaults to `gaussian`; `direct` uses one radial channel.
`atomic_radial_count=16` is the Gaussian default; V9f tests 8/16/32. `atomic_hidden_dim=null`
inherits `hidden_dim`; 64/96/128 are supported for SchNet. `atomic_output_dim=null` also inherits
`hidden_dim` and must equal it: reduced internal states use a learned 64→128 projection; equal
widths add no layer. Chemical embeddings, transfer, DiffusionNet and heads keep their original widths.
The spatial cosine cutoff stays 6 Å; covalent edges have no added cutoff, even beyond the Gaussian
center interval. For a distance gate g, radial input is `g*b(d)` and support is `1+g*(cutoff(d)-1)`;
g=0 removes distance information, not the relation or its non-distance attributes. `schnet_radial`
keeps distances and chemistry but makes the same five metadata sources unavailable as in V9d.

LF records quality/cost curves and paired-seed coverage. `atomic-encoder-audit/initialization.json`
records exact centers, gamma, widths, filter/encoder counts and a hash of common initial weights.
It also counts dense multiply-accumulates in filter Linear layers per edge and in the optional
output projection per atom; this partial operation count is not whole-model FLOPs or measured time.
After restoring the best **G-selected** checkpoint, `radial-filters.json`, `.npz` and `.png` inspect
121 distances on [0,6] Å for separate relations, layers and declared metadata profiles. They retain
responses before/after support and relation gates, amplitudes, finite-difference sensitivity and
channel correlation. These latent filters are not physical energy potentials. Local GT remains
evaluation-only. CUDA allocated/reserved peaks measure the whole Run, not encoder-only memory;
median complete epoch time excludes epoch one but includes scheduled validation. The isolated
synthetic benchmark `scripts/benchmark_schnet.py` measures encoder forward/backward separately.
Positive-protein AUPRC SD/quartiles expose between-protein spread; SD is unavailable below two
evaluable proteins. Neither statistic measures uncertainty between seeds or changes the S mean.

See [the campaign catalogue](experiments/README.md) for the 13 cells and block/cell commands.
The optional `scripts/analyze_schnet_campaign.py` reads native checkpoint metadata for fixed paired
cell-04, width, metadata and interaction contrasts, plots and relative-cost tables. It preserves
missing support and does not replace native LF decisions or parameter discovery.
Four paired seeds cannot establish universal conclusions; do not infer a full metadata×RBF
interaction from unmeasured cells, or claim an encoder memory saving from parameter counts alone.
No capacity-matched extra filter-width control is included in the principal 13-cell matrix.

V10 now freezes **full-metadata SchNet, Gaussian RBF-16, 64 internal channels and 128 output
channels**, following development cell 08 of V9f (`execution-d384d8f6b5064a26`). Its four-seed S is
0.2932 versus 0.2966 for direct/restricted/128, with 87,488 versus 281,728 trainable encoder
parameters, lower observed CUDA peaks and lower false surface activation on negative proteins.
This is a provisional quality/capacity compromise, not a confirmed winner or demonstrated speedup:
G is 0.7030 versus 0.7134 for the original contemporary baseline, and shared-hardware epoch timings
do not favor the narrow model consistently. All V10 controls and interventions train this same
backbone from scratch; the empty input mapping means SchNet without extra input sources, not the
retired baseline encoder. V10 retains its authored seeds and enriched `wisdom-dna-reduced@8` input;
the @6 results do not certify @8 equivalence. Exact parameters, evidence hashes, paired differences
and limitations are recorded in [the V10 selection record](experiments/v10/backbone_selection.json).

[V10e](experiments/v10/wisdom_v10e.yaml) now declares 16 source configurations and four paired
development seeds, 64 Runs. It recomputes the no-added-source control and isolated anchors under
one code/data identity, compares name-only with name plus residue-atom identity as separate
alternatives, then tests A/B/C interactions, selected SASA combinations and nested exact
identity/motif additions. All source gates remain learnable with the fixed baseline denominator.
Small or inconclusive effects in DNA do not remove a source from the reusable model, and an open
gate does not certify utility in another task. The selection remains developmental, with no test
evaluation or new architecture/optimization sweep.

The source/gate policy after V10f remains recorded in
[source_gate_selection.json](experiments/v10/source_gate_selection.json). The current continuation
policy after V10g/V10h is [representation_selection.json](experiments/v10/representation_selection.json):
`embedding_initialization: current`, full-metadata SchNet with 64 internal/128 output channels,
and `atomic_representation: {sources: [atom_name, residue_atom], identity_dim: 8,
force_sources: learned, external_model: esm2, external_mode: context, context_dim: 32}`.
ESM-2 supplies frozen residue context; it does **not** initialize the residue table in this mode.
The habitual residue table stays trainable. A SaProt table is not combined with ESM-2 context.
All gate groups retain coefficient 0.04, denominator 41, initial activity 0.95 and no schedule.
This is a development choice, not a claim of universal superiority or automatic source removal.

[V10i](experiments/v10/wisdom_v10i.yaml) now rechecks three poolers with that fixed representation:
MAX, area-weighted fixed LSE (`beta=0.25`), and area-weighted simple Attention (width 256).
Four new paired seeds, 307/311/313/317, produce **12 mandatory Runs**. Every family keeps the same
single head, Heat6 refiner and regional prior 0.3. No extra HPO or family-specific V6 losses are
introduced. The example YAML mirrors the executable design. Inspect S, G and negative evidence
separately, then peak allocated memory, complete-epoch seconds and completed epochs. MAX remains
the explicit **provisional** pooling in V11/V13 until V10i supplies evidence for replacing it.

All V11/V13 training designs now inherit the selected context, initializer and SchNet instead of
silently reverting to the old relational encoder. Their transfer/surface factors, dataset,
exploration seeds and resource budgets stay fixed. Closure seeds use 401/409/419/421 for V11f and
503/509/521/523 for V13f, avoiding the V10i selection seeds. V12 performs inference only: its
recipe selects exactly ESM-2 context from completed V10h execution `execution-6c675d5c9f10d971`,
with all four saved seeds, and restores the recorded weights/gates. Register the portable study
with native `lf import`; isolated protocols rebuild compatible surfaces from the exact base. It never
adds training overrides. Historical V10g/V10h YAMLs/results remain unchanged; editing any file
starts no computation or submission.

External-input initialization writes one `atomic-input-audit` artifact directory. Its
`normalization.json` records statistics fitted only on the selected training proteins;
`static-table.json`, when applicable, records how the pretrained table initializes the trainable
residue embedding. Both files may coexist. In `replace` mode, contextual residue features
replace that embedding: its zero-width checkpoint entry remains loadable, but no empty lookup
or backward operation is executed.

### 4.4. Grouped feature selection

[V10f](experiments/v10/wisdom_v10f.yaml) tests whether Hard-Concrete gates can select useful
sources while retaining all seven added chemical inputs. A gate is one learned scalar shared
across proteins, not a separate decision for each atom. Its analytic activity probability is
the chance of a nonzero training sample; its deterministic inference value is a different
quantity. For example, initial activity 0.50 gives an inference value of approximately 0.102,
not 0.50. Exact zero means closure; a smaller positive value means attenuation, not removal.

The five-scale model has 41 historical gates and seven added chemical gates. Direct costs form
three disjoint groups: S contains 20 curvature gates, C contains the seven added chemical gates,
and O contains the remaining 21 historical gates. `surface.atom_context` belongs to O because it
controls atomic context, not surface shape. A child's effective activity remains its own
probability multiplied by its ancestors' probabilities. Parents also have their own direct cost;
each named gate is counted once, without adding separate group-parent gates.

With group coefficients denoted by lambda and effective path probabilities by p, the additional
training penalty is:

$$
L_{gate} = \frac{\lambda_O\sum_{i\in O}p_i
                     +\lambda_S\sum_{i\in S}p_i
                     +\lambda_C\sum_{i\in C}p_i}{41}.
$$

Here i identifies a gate, and the denominator stays 41 even when chemical inputs are added or
omitted. This already-weighted term is added once to protein loss and the existing weak
protein-label priors; it is **not multiplied by `gate_lambda` again**. Surface ground truth never
enters the loss or checkpoint selection. Equal coefficients 0.04 reproduce V10e's reference
penalty and gradients. AdamW applies no weight decay to gate parameters.

Training's optional scalar arguments `gate_lambda_surface`, `gate_lambda_chemical` and
`gate_lambda_other` enable this policy. Unspecified group coefficients then default to 0.04.
`gate_initial_active_surface` and `gate_initial_active_chemical` default to 0.95; O retains
`gate_initial_active`. With no group arguments, historical scalar regularization is unchanged.
`atomic_sources` accepts comma-separated canonical source names, and `atomic_force_sources`
accepts `learned`, `open` or `closed` for added sources only. These scalar coordinates allow
native LF `sweep.candidates` to enumerate complete profiles without a hidden profile selector.
The equivalent `gate_policy` mapping is also accepted, but cannot be mixed with scalar policy
controls. See the YAML for complete constructor defaults and the exact 20-profile design.

`gate_group_warmup_fraction` and `gate_group_ramp_fraction` use nominal progress
`(epoch-1)/epochs`. During warm-up, only S/C are forced open at zero direct penalty; O remains
learned and penalized. During ramp, S/C become stochastic and their coefficients increase
linearly. These options cannot be combined with the old global gate schedule. The selected
checkpoint saves and restores the exact runtime overrides, including unfinished warm-up: it
must not be reported as learned closure. Closing inputs writes zero for both values and validity
masks; it does not physically shrink the network or guarantee faster training.

Each enabled Run publishes a `gate-audit` artifact: `manifest.json` identifies the inventory,
widths, initial probabilities, policy, dataset identity, seeds and code-file digests; `epochs.jsonl`
records per-gate activity, actual sampled clipping and pre-step gradients once per epoch.
`initial.json`, `final.json` and `selected.json` distinguish initialization, last training state
and the checkpoint chosen by validation protein G. Native LF attempt provenance supplies the
code/bundle identity. Git metadata may be unavailable in packaged workers; the preparation HEAD
and dirty status are recorded in `experiments/v10/gate_study.json`, not claimed as an exact clean
execution commit. Runtime code digests bind the inspected files.
Group costs and closure fractions are also LF metrics, separate from prediction quality.
Good localization with open gates is not feature selection; closing gates while destroying
localization is not success either. This DNA benchmark alone cannot establish universal utility.

### 4.5. Frozen geometry audits and multiscale inputs

V12–V15 separate numerical stability, molecular organization and new physical surface
information. None changes the benchmark labels, split membership or weak-supervision rule.
V14/V15 keep SchNet output and surface width at 128, one surface pass, MAX, Heat6 and regional
existence 0.3 unless the YAML explicitly declares the tested change. Protein validation still
selects epochs; local targets never enter gradients. These are implemented comparisons, not
evidence that the additional information improves prediction.

V14/V15 use the V11b rank-16 receiver-conditioned value correction as their provisional
development continuation: `transfer_options: {message: low_rank, rank: 16}`. They retain the
single-readout DiffusionNet with its gradient descriptor, 32 modes, short 16-atom/4 Å support
and no local-message branch. This is one selected change, not an untested stack of V11/V13
candidates or a final confirmed model. Environment sweeps repeat the complete surface mapping;
LambdaForge replaces those mappings rather than implicitly merging them. The authored `@9`
and optional SLAE dataset selectors must exist on the execution target before launch.

From V14b onward, the V14b–f and V15 YAMLs adopt `context_dim: 64` for ESM-2 and
`atomic_hidden_dim: 128`, while keeping atom output and surface width at 128. This is the
V14a 64/128 development choice under the available H200 budget, not confirmed superiority
over the compact 16/64 alternative. V14a/a2 retain their executed designs. Complete swept
representation mappings repeat the new context width; the SLAE-only control still omits
ESM-2, SLAE projection stays at 32 and the V14 residue-graph hidden width stays at its own
64 default when that branch is active. Seeds, source removals, communication contrasts and
geometry controls remain unchanged.

V14c provisionally selected the CA-contact residue branch (`residue_graph: {mode: contact}`)
for the V14d–f comparisons and the first V15 drafts. V14d fixes it in thirteen source
removals and adds a no-branch control (15 configurations × four seeds = 60 Runs).
Removing `external_context` zeros ESM-2 content before both the direct and
residue-message routes; closing only its gate would not remove both. V14e compares six
communication/control routes. V14f repeats no branch, nodewise, contacts and null
neighbor messages with four fresh seeds (16 Runs). Contacts lower operational surface
S against no branch in all four seeds, while global G is nearly unchanged. Therefore
all four V15 YAMLs now omit `residue_graph` in their shared molecular reference and vary
only the declared surface geometry. They retain ESM-2 64, SchNet 128, rank-16 transfer,
MAX, Heat6 and regional existence 0.3. The promising V14e null-atomic-summary return
and SLAE combination remain unconfirmed and are not stacked into V15. Executed
V14a–f designs and results remain unchanged.

Every V12–V15 experiment uses four fixed model seeds, shared by all its alternatives. V12
selects four saved checkpoints rather than training; its separate view seeds control geometry
sampling, not model replication. Confirmation stages retain their fresh, independent seed sets.
V14a now runs the twelve width combinations with four seeds (48 Runs), rather than the report's
initial two-seed screening proposal.

| Phase | Input or intervention | What the comparison isolates |
|---|---|---|
| V12 | Frozen checkpoint; identity, rigid tensors, rebuilt rotations, phase, voxel origin and resolution | Model invariance versus dependence on surface discretization |
| V13 | Spectrum, simultaneous diffusion readings, sparse local messages and existing encoder alternatives | Surface processing, not a change of atomic information |
| V14a/a2 | ESM-2 projection 16/32/64/128 × atomic width 32/64/128; aligned/mean/permuted controls separately | Compression, capacity and local residue alignment |
| V14b | Optional frozen SLAE atom features with/without ESM-2 and a no-additional-atomic-message control | Atomic representation versus the existing residue context |
| V14c/d/e/f | Residue graph, nodewise/null controls, one-source removals, bounded communication and confirmation | Physical communication versus added parameters or compensating input routes |
| V15a/b/c/e | Depth, two-horizon occlusion, associated empty-space radius; masks/capacity controls | Distinct protein-only physical descriptors, then a provisional combination |

**Preparing the data.** The DNA recipe now targets `wisdom-dna-reduced@9`, retaining train-25,
full validation and the existing test-exclusion policy. It adds assets to the same physical
representation; it does not overwrite @8. Alternatively, run the authored
`experiments/preprocess/enrich_structures.yaml` to enrich an existing @8 without recomputing
base surfaces, annotations or external residue embeddings. Choose only one publication route
for @9. Historical experiments remain bound to their old selectors. Validation of @9/@SLAE
training YAMLs cannot resolve the destination until that upstream publication exists.

The `residue_graph` asset keeps heavy-atom contacts and separately adds CA contacts out to
12 Å with 32 directed nearest-neighbor ranks. CA means the backbone alpha carbon, one possible
physical reference for a residue; it is not the closest side-chain atom. Runtime V14 uses
up to 16 contacts within 10 Å and retains verified peptide/covalent relations outside that
radius. Union symmetrization can give a node more than 16 incoming neighbors. Every undirected
pair is stored once. Peptide continuity comes from actual C–N bonds, never consecutive residue
numbers. Missing CA positions and degenerate N–CA–C frames remain masked, not replaced by the
origin or a guessed orientation. All indices are physical residues of specific chain copies.

The residue model uses LF SchNet layers with its own weights, normally two layers and width 64,
then adapts to the external-context width. The original context route remains intact: a
zero-initialized final correction with multiplier one recovers the reference output and can
learn immediately. The pointwise control updates each residue without seeing other residues;
the null control retains the graph/filter capacity but suppresses messages. Descending
communication refines context before the atomic GNN; roundtrip adds one mean-by-residue atomic
summary and return adapter, not another hidden GNN; parallel sends a separate distance-weighted
residue message to the surface before the same projection and single DiffusionNet. Its candidates
come from all nearby heavy atoms deduplicated by residue, not the truncated active atom list.
`drop_sources` performs nominal-width input removals; it is not parameter compaction. Removing
ESM-2 also zeros its input to the residue graph. Removing an input does not erase information
that the pretrained extractor already consumed.

**New physical fields.** `surface_environment` is a parent-hash-bound asset in exact surface
point order. It stores nonnegative distances in Å, occlusion fractions in [0,1], validity masks,
closed-cavity/bulk-solvent flags, actual outward offsets and center-association distances.
Protein atoms and van der Waals radii are the only obstacles; no ligand coordinates, functional
mask or class label selects a pocket. Invalid observations are zero only with a false mask.

Travel depth follows unobstructed segments on a 26-neighbor voxel graph to free cells outside
a finite supporting-plane approximation of the expanded-sphere convex envelope. It is not
distance to the protein center or a line through occupied protein. Inaccessible cavities have
no finite exterior depth. A 1 Å voxel, 96 supporting directions and two million voxels per
protein are explicit defaults; exceeding the guard fails rather than silently coarsening.
The finite stencil is anisotropic and narrow-neck connectivity depends on resolution. The
Euclidean distance to the same discrete exterior is stored as a control, not a substitute.

Occlusion counts outward hemisphere rays hitting expanded spheres before 4 and 15 Å, using
64 deterministic directions. Its complement is not duplicated as a second source. Finite
angular quadrature uses a declared global-axis frame and is not exactly rotation invariant.
The empty-space radius is the largest clearance of a visible, wall-compatible interior grid
maximum within an 8 Å association horizon. It is neither cavity volume nor clearance measured
on the SAS itself (which would be nearly the fixed probe radius). Unbounded exterior solvent
is excluded and absent associations stay masked. V15 compares real values with mask-only and
equal-width old-curvature controls; normalization fits valid selected-training values only.
These algorithms are bounded approximations requiring the geometric pilots described in the
report, not proven functional pocket detectors. Optional neck, multi-probe and adaptive-solver
extensions are not silently substituted for the initial plan.

**Auditing fixed predictions.** `experiments/v12/wisdom_v12.yaml` uses native imported checkpoint
evidence and no optimizer. `view_audit.protocols` declares isolated perturbations, `view_seeds`
their repeatable seeds and `maximum_proteins` the audit count (zero means all). Phase uses a
SHA-256-derived shift per physical atom; grid changes only the relative voxel origin. Resolution
ratios 0.8/1.25 and rotated reconstruction regenerate every point-dependent geometric operator
in native managed files. No new point inherits an old label by row index. The initial audit
requires the original point-input contract; extended candidate tables, extra spectra, local
graphs and new environmental inputs must be regenerated coherently before auditing such models.
Unsupported combinations fail rather than appear stable through stale arrays.

Both directions report comparable area, logit/probability errors, rank correlation and top-5/10%
area overlap, before and after Heat6. Constant correlations stay unavailable. Constant, smooth
and localized fields tied to a physical atom calibrate interpolation error. A1 compares identical
physical rows; rebuilt views use normal-compatible inverse-square interpolation. Stability is
not correctness: a constant incorrect map can be stable. Original-split G/S are evaluated normally;
the new-view audit does not invent local GT by interpolating the original annotation.

**Optional atomic pretraining.** The reviewed base SLAE checkpoint is staged at
`data/external-models/slae/autoencoder.ckpt`, with URL, revision, size and SHA-256 in the adjacent
`checkpoint.json`. This is the base model, not the PDB-finetuned one. Upstream SLAE uses
atom37, a fixed atom-name vocabulary per physical residue. The adapter selects only observed
standard heavy atoms, verifies unique physical residue/name slots, sorts explicitly, then scatters
128-channel scalar outputs back to original atom order. Unsupported atoms/residues remain in
WISDOM with masks; nothing is reconstructed or silently deleted from the benchmark.
Coverage is recorded per physical chain, atom name and residue name. Entirely unsupported
members keep all-false masks; ambiguous physical slots fail instead of inventing a correspondence.

SLAE's compiled PyG dependencies, `numpy<2` constraint and `esm` package differ from the standard
WISDOM environment. `pyproject.toml` now declares the independent
`tool.lambdaforge.tool_runtimes.slae` environment: Python 3.11, Torch 2.6.0/CUDA 12.4,
matching precompiled PyG libraries, ESM 3.2.1 and Transformers 4.48.1. WISDOM keeps its own
Transformers 5 dependency; do not install SLAE into that environment with pip or the retired
`atomic-representations` extra. LambdaForge must expose `self.tools.runtime`; use the checkout
that implements isolated tools, not an older installation with the same release number.

From the repository root, prepare the target once, validate the Work and then launch it yourself:

```bash
lf clusters bootstrap citius-ctgpgpu16 --project . --dry-run
lf clusters bootstrap citius-ctgpgpu16 --project .
lf validate experiments/preprocess/enrich_atomic.yaml
lf run experiments/preprocess/enrich_atomic.yaml --on citius-ctgpgpu16
```

The first bootstrap downloads and installs the isolated packages; subsequent compatible jobs
reuse them. It does not extract proteins or load weights. All declared tool runtimes are prepared,
even if the next Work does not use SLAE. Keep the reviewed checkpoint at
`data/external-models/slae/autoencoder.ckpt` locally and at the corresponding mirrored project
path on the target. The typed file input verifies its bytes; a checkpoint is not an installed
extractor. The parent `wisdom-dna-reduced@9` must be registered and available on the target.

Weight loading constructs only SLAE's public encoder, with its packaged configuration. The
verified base checkpoint contains two retired, non-learned fields,
`allegro.env_sum_normalizations` and `edge_eng_sum.per_edge_scales`. The pinned upstream
implementation recomputes neighbor normalization during each forward pass and no longer stores
those fields. WISDOM excludes only these two names and loads every remaining encoder field with
`strict=True`: missing, unknown or incorrectly shaped weights still fail. It does not rewrite the
checkpoint or construct the unused coordinate decoder. This preserves the pinned encoder's
inference rule rather than silently accepting an arbitrary partial model.

The Work checks the runtime, logical GPU and checkpoint before processing proteins, then calls
the staged standalone `SLAEExtraction.py` through `self.tools.runtime("slae")`. Each invocation
loads frozen weights and exchanges explicit NPZ files; it never imports WISDOM into the isolated
environment. The parent validates finite values, masks and physical atom correspondence before
native file-checkpoint acceptance. Successful protein files remain available for compatible native
recovery; there is no package installation or hidden fallback training inside the Work. The action
publishes `wisdom-dna-reduced-slae@1` while preserving @9's proteins, splits, targets and existing
assets; geometry and Selection are not repeated. Native publication may copy the original assets,
so reserve disk space for another placement, not just the small added representations.

The shipped recipe requires CUDA and respects `CUDA_VISIBLE_DEVICES`. CPU use requires an explicit
review of the tool's Torch/vision/PyG pins and package sources, `torch_cuda="cpu"`, a new bootstrap,
`atomic_device: cpu` and `resources.gpu: 0`; changing only the Work parameter is insufficient.
The corrected loader was checked in CPU execution on the target with the real base checkpoint:
an eight-atom synthetic graph produced finite [8,128] atom and [2,128] residue arrays, exactly
equal to the official loader's outputs. The raw evidence is in
`benchmarks/slae_encoder_loading.json`. This checks loading and a small forward, not a real-protein
coverage study, CUDA throughput or pretrained quality. A full target extraction and benchmark
coverage pilot remain necessary; schema, dry-run and mocked tests do not establish those claims.
Pretraining overlap with benchmark families remains unknown. ESM-AA is not
aliased to ESM-2: its public contact example does not establish observed-atom token alignment,
so it is not enabled as a fictitious ready extractor.

### 4.6. Consolidation and negative-region supervision

V16's executable recipes are in `experiments/v16`. V16a compares the historical ESM-2 context
64 / SchNet state 128 reference with compact 16/64 on four paired seeds. The compact route is
the starting point for consolidation because its mean quality is comparable and it uses less GPU
memory. V16a does **not** silently add SLAE, residue feedback or V15 descriptors. The current V16b
recipe removes ESM-2 (`external_model: null`, `external_mode: none`, `external_context_dim: null`)
but keeps the compact SchNet width 64, learned atom/residue identity tables and the rest of the
backbone. It searches atomic, projection and surface depths 1–6, dropout, learning rate
0.0003–0.003 and weight decay. Native adaptive HPO caps proposals at 128 candidates, 300 total Runs
and 168 hours. Four development seeds form a replication pool, with one required initially and
additional seeds allocated adaptively; three finalists receive three separate confirmation seeds.
Curve pruning and ordinary G-based patience remain active. These are budget/design choices, not
evidence that removing ESM improves localization. The running ESM-enabled Study is unchanged:
launch this revision as a **new Study**, never recover or extend the old one into the new input
representation. The same @9 dataset suffices; unused ESM files are not consumed.
`external_context_dim` normally accepts a positive override only with an active external
projection; it must be null when that projection is disabled.

V16a2 separately checks two unresolved V15 signals on four seeds new to V15. It compares the
64/128 reference with occlusion-validity masks alone and the same branch rereading two existing
gated curvature channels. This tests routing and capacity rather than a new physical descriptor:
V15a and V15c did not replicate V15b's localization gain. A second pair compares the travel-depth
validity mask with the real depth value; V15e improved mean localization with depth, whereas V15a
lost localization in all four paired seeds. The five configurations require twenty full Runs.
V16b retains only the compact atomic width 64 and disables external context. V16c/d still ship
the older 16/64 reference until a reviewed finalist is copied explicitly. No V16a2 geometry branch
is inherited automatically; a favorable
result on its historical 64/128 route must be checked on the selected compact route.

V16c reserves new seeds to confirm one compact 16/64 configuration and V16d is a frozen-system
replication sheet. Their shipped learning rate and surface depth are **reference values, not
already selected HPO finalists**. Before using them
to confirm or freeze a winner, copy the fully reviewed configuration from the preceding stage,
including source mappings and numerical values, and record the decision in the external report.
No automatic winner import or best-seed selection occurs. All five recipes keep test sealed;
writing a final decision does not itself require another training run.

VN01 is a separate negative-supervision line in `experiments/vn1`, using the same no-residue-graph
64/128 reference, rank-16 transfer, MAX and Heat6. `wisdom_vn1b.yaml` is a ten-epoch coverage/intensity
pilot, not a short-training proxy for final model quality. Inspect candidate coverage, active
updates, gradient ratios and time before fixing cuts, K and loss intensity. `wisdom_vn1.yaml`
executes eight cells × four common seeds = **32 Runs**: regional positive existence 0.3 or 0,
crossed with no mining, global top-K, top-K with memory, and current FGW families with memory.
R− removes only that positive prior; BCE, gates, dropout, decay and Heat6 remain fixed.
All VN and VF recipes disable pretrained protein descriptors: `external_model: null`,
`external_mode: none` and `atomic_external_model: null` in `atomic_representation`.
This disables ESM-2, ESM-C and SaProt context and pretrained table initialization; frozen atomic
descriptors are also off. Learned atom-name and residue/atom identity tables remain active.
Existing descriptor assets stay in the dataset but are neither required nor loaded by these
recipes; preprocessing does not need to be repeated. The purpose is to examine each technique
without pretrained protein context dominating the comparison, not to maximize scores immediately.
Changing input sources changes scientific identity: start a new Study rather than recovering
an ESM-enabled Study with the edited YAML.
`wisdom_vn1a.yaml` retains its name but now matches VN1's complete no-LLM matrix. Do not execute
both by default or interpret them as an ESM-versus-no-ESM comparison. `wisdom_vn1c.yaml` adds the
no-LLM point-selection and distributed-softplus controls (eight Runs).
The previous VN1a pilot and VN1b controls are now VN1b and VN1c respectively; historical
Results are not renamed. Do not recover a historical VN1a pilot using the new VN1a matrix.
The proposed cuts 0.25/0.40, K=3 and coefficient 0.1 are **uncalibrated pilot values**. Freeze common
choices prospectively rather than fitting different coefficients after each strategy's results.

**N1 — current negative regions.** A negative protein's global label supplies the supervision;
its local ground truth is never loaded for mining. At each completed epoch, deterministic
evaluation reviews every selected training negative. The existing radius/normal/chord-filtered
surface graph delimits connected components above the lower probability cut. A component is
eligible for activation when its maximum reaches the upper cut. Top-K is global to one Run,
not per protein, batch or GPU. Immutable protein ID and minimum point index break ties.
Epoch one has no selective term; the review after epoch t supplies masks for epoch t+1.

For a protein b, let z_bi be the current operational logit at point i (after Heat6 if enabled),
a_bi its proxy area normalized over the **whole protein**, and u=log(U_s/(1−U_s)) the logit of
the lower probability cut U_s. Let H_b and Q_b be disjoint strong and remembered point supports,
with coefficients λ_h and λ_m. The per-protein excess penalty is

$$
J_b=\lambda_h\sum_{i\in H_b}a_{bi}\max(0,z_{bi}-u)^2
   +\lambda_m\sum_{i\in Q_b}a_{bi}\max(0,z_{bi}-u)^2.
$$

For a uniformly sampled protein batch B, divide the negative-protein sum by |B|π₋, where π₋ is
the negative fraction of the complete selected training population. This estimates the mean over
training negatives rather than changing strength with each batch's class composition. No region
is divided by its own area. FP32 reductions preserve gradients through the current logits; old
scores select supports but never become targets. A positive-only batch has a differentiable zero
term; a training population without negatives is invalid. Heat6 may spread influence outside
the directly penalized support. The point control matches each selected protein's proxy area,
including a fractional final point; it does not claim equal gradient magnitude.

Memory matches **current** components by weighted overlap plus a peak-anchor condition. It
records contraction, splits, merges, correction, lost tracking, expiry and budget eviction. It
never penalizes archived masks. Weak supports default to one quarter of the strong coefficient,
at most 4K supports and five epochs since their last priority; the bounded event history defaults
to 10(K+4K) records. All values are configurable in `negative_mining`. VN01 checkpoints include
model, optimizer, precision scaler, G-selected best state, memory, curves and Python/NumPy/Torch/
sampler RNG states. Shuffle and worker seeds use separate streams, shared by the contemporary
controls; exact historical full-run trajectories are not implied. Native LF recovery must retain
the original Run. A new submission is not checkpoint continuation. Weight averaging is currently
rejected for VN01 rather than reviewing one weight set and training another.

**N2 — diverse priorities, not a new neural head.** Each epoch rebuilds the current candidate/
remembered universe, geodesic representatives, FGW distances and native LF precomputed HDBSCAN
families. FGW compares attributes and internal geometry through one shared transport coupling.
Attributes are existing mean/Gaussian curvatures plus explicit Gaussian-projected chemistry,
using common train-negative median/IQR scales. Scores/logits and learned embeddings are not
chemical attributes. Per-channel masks distinguish missing atom/CA/known-residue support.
Only valid train observations fit scales; missing attributes use centered-zero imputation plus
explicit coverage coordinates, not an unmarked physical zero. A channel with no valid training
observations stops preflight. No predictor preprocessing or forward input is changed.

At most 64 geodesically spread representatives retain proxy mass; shortest paths use the full
component's sparse graph and a common 10 Å scale, not each patch's diameter. Dense matrices are
bounded **region comparison objects**, never full atom/surface distance graphs. Both FGW
directions, self-costs, marginals and convergence are audited; nonconvergence gets one predefined
larger-iteration solve. Numerical/resource failure falls back explicitly to top-K for that epoch.
Missing POT is a preflight error, not silent fallback; `./install.sh` and cluster bootstrap install
the LF `optimal-transport` extra. Families prioritize their hardest eligible members, with one
noise representative in the first pass and difficulty-based filling. All-noise recovers top-K.
Medoids describe families; neither medoids nor transport costs enter gradients.

**N3 — post-training decisions.** `negative_calibration` operates only after restoring the G
checkpoint. It leaves scores and G/S rankings untouched. Optional typed `calibration_dataset`
and `decision_dataset` bindings supply a negative reserve and separate F1 threshold-development
population. Their IDs and leakage groups are audited against the original dataset, including
held-out membership without reading its point labels. One negative per reserve group supplies
one maximum operational logit. Independence and exchangeability still require a sampling/history
audit: different IDs alone do not prove them. Without a reserve, `negative-admission.json` states
unavailable and no formal guarantee is reported. Train, validation or test is never silently reused.

F0 uses logit zero; F1 fits a global threshold by development balanced accuracy, breaking ties by
specificity; F2 uses p=(1+count of calibration maxima ≥ regional maximum)/(n+1). This is a tail
rank, not a binding probability or false-discovery rate. Its minimum is 1/(n+1), so n≥19 is
necessary merely to admit at level 0.05. `formal: true` also requires a nonempty
`history_attestation`; its interpretation remains conditional on exchangeability. The current
integration evaluates decisions on validation for development, not a substitute final test.
It reports false admission, sensitivity, thresholded binary metrics, area and local Dice/IoU/
precision/recall with unavailable denominators retained. Changing thresholds cannot improve
AUPRC/AUROC from unchanged scores.

LF receives `train_negative_mining_*` curves and final `val_negative_admission_*` summaries.
Per-review JSON artifacts preserve supports, lineages and FGW diagnostics; `evaluation.json`
contains the review summaries and admission status. Review cost is logged separately from
optimizer/validation time; `negative_mining_complete_epoch_seconds` includes the extra review
and continuation save. These tests establish implementation contracts, not improved biological
localization. The full prospective hypotheses and their limitations remain in the report.

### 4.7. VF physical targets

VF trains the atom encoder to reconstruct physical fields around a protein. Fields are **targets,
not input descriptors or binding labels**. The optional enrichment publishes
the strict `wisdom-dna-reduced-physics@1` from `wisdom-dna-reduced-slae@1`, then bounded-completion
`wisdom-dna-reduced-physics@2` from that strict parent. Both preserve every member, split,
label, base NPZ and inherited asset, including SLAE. The VF predictor nevertheless disables all
SLAE, ESM and SaProt inputs so they cannot bypass learning from the atomic graph.

| Configuration | Produces | Next decision |
|---|---|---|
| `experiments/preprocess/enrich_physics.yaml` | Historical strict physical volumes and query banks, `@1` | Recover original checkpoints before changing the scientific protocol |
| `experiments/preprocess/enrich_physics_bounded.yaml` | Bounded completion, `@2`, reusing admitted strict parent fields | Review modeled atoms, clashes, field support and convergence |
| `experiments/vf/wisdom_vf1.yaml` | Physical-only reconstruction controls, four paired seeds | Check unseen-protein errors and encoder gradients |
| `experiments/vf/wisdom_vf_fields.yaml` | Single-field electrostatic, LJ1 and LJ2 pretraining with the same four seeds | Compare each field against VF1's already scheduled all-field branch |
| `experiments/vf/wisdom_vf_joint.yaml` | From-scratch joint training, eight auxiliary weights including zero | Compare protein G, localization, gradient conflict and cost |

The VF pilot fixes the replicated V16b no-ESM candidate: one SchNet interaction, one
surface-input projection and six DiffusionNet blocks. Its joint-training YAML also fixes that
candidate's dropout (0.2), learning rate (0.0007542867271657824) and AdamW decay (0.05)
across all eight physics weights. The zero-weight row is the contemporary functional control.
Physical-only pretraining uses the same atomic path and gate policy for checkpoint compatibility;
its surface blocks are not executed and its reconstruction optimizer remains separately fixed.
VN1c does not establish a same-code benefit over no negative mining, so the VF pilot keeps mining
disabled to isolate the physical objective. V16b's 20.1 GiB peak was measured without physical
queries and does not bound VF memory. The VF training YAMLs currently select the strict `@1`
physical dataset; bind all three to reviewed `@2` together if the bounded completion is adopted.
The single-field study uses the same strict `@1` dataset and VF1c configuration; it requires
no new field generation. Its three branches reuse VF1c's all-field baseline instead of
training it again. Compare original-unit error for the supervised field and encoder gradients;
the weighted validation loss changes meaning with the set of fields and cannot rank branches.

Prepare the environment, then submit the stages yourself, in that order:

```bash
lf clusters bootstrap citius-ctgpgpu16 --project .
lf validate experiments/preprocess/enrich_physics.yaml
lf explain experiments/preprocess/enrich_physics.yaml
lf run experiments/preprocess/enrich_physics.yaml --on citius-ctgpgpu16
lf run experiments/preprocess/enrich_physics_bounded.yaml --on citius-ctgpgpu16
lf run experiments/vf/wisdom_vf1.yaml --on citius-ctgpgpu16
lf run experiments/vf/wisdom_vf_joint.yaml --on citius-ctgpgpu16
lf run experiments/vf/wisdom_vf_fields.yaml --on citius-ctgpgpu16
```

Do not start training until the enrichment audit is acceptable. Local setup uses `./install.sh`;
`pip install -e '.[physics]'` supplies Python APIs but not the native APBS executable. Conda supplies
APBS, PDB2PQR/PROPKA and OpenMM. They are checked before protein processing. Selection, geometry and
SLAE extraction are not repeated. Compatible native recovery reuses validated per-protein
reference/query checkpoints; new references also use LF's reconstructible file cache. Diagnostics
are retained at the native `physics/last-report.json` checkpoint even if failed-attempt artifacts
are cleaned. Use the original Work's recovery, not a new submission, to recover existing files:

```bash
lf retry <failed-job-id> --config experiments/preprocess/enrich_physics.yaml --accept-code-change --accept-operational-changes --dry-run
lf retry <failed-job-id> --config experiments/preprocess/enrich_physics.yaml --accept-code-change --accept-operational-changes --apply
```

Inspect the dry-run before applying. Code acceptance is appropriate for this compatible fix, not
permission to change scientific inputs. LF must confirm identity and restore the dependencies;
missing/deleted files cannot be recovered by claiming a cache hit. The YAML's voxel limit is an
operational allocation guard and does not change physical reference identity.
StructuralEnrichment declares `physics_maximum_voxels` through LF's public increase-only allocation
guard. Increases are accepted with separate operational approval;
the recovery preview must enumerate the original Work's retained checkpoints. This does **not**
authorize changing chemistry in a retry. Finish/recover strict `@1` first, then run the bounded
YAML as a new protocol/version. Its typed parent lets it revalidate/re-admit unchanged strict
fields without rerunning PDB2PQR/APBS/LJ. Unavailable references are reconsidered, not blindly
reused, and changed tools/equations require new solves. No private checkpoint copying is used.

The strict recipe permits eight million voxels per APBS grid; the bounded recipe permits
24 million to cover elongated proteins and their finer-grid/box audits within its existing
96 GiB allocation. This is an allocation ceiling, not a request to enlarge every grid or a memory
guarantee. The log names each grid and its voxel count. If the bounded Work hits the old ceiling,
recover **that same bounded Work**, not the strict producer:

```bash
lf retry <bounded-job-id> --config experiments/preprocess/enrich_physics_bounded.yaml --accept-operational-changes --dry-run
lf retry <bounded-job-id> --config experiments/preprocess/enrich_physics_bounded.yaml --accept-operational-changes --apply
```

Add `--accept-code-change` only when also applying a reviewed compatible code fix. Recovery
revalidates stored references and query banks and computes missing results; it does not restart
selection, geometry, SLAE extraction or successful physical solves. Actual resource/tool failures
remain errors, never unavailable chemistry or silently coarser physical fields.

**Reference and units.** PDB2PQR/PROPKA prepares hydrogens and protonation at pH 7. Retained heavy
atoms must match the original identities/elements and coordinates within 0.002 Å of PDB rounding.
Strict `@1` rejects any modeled heavy atom. Bounded `@2` admits limited modeled completion under
the criteria below, using the same physical equations. Unsupported preparation leaves VF
unavailable rather than changing the graph. The protein remains in its original
split, with SLAE intact, an explicit reason and an empty physical query bank; it contributes no
physical loss or zero-error metric. Displaced atoms, unknown tool errors, allocation failures and
failed independent LJ audits still block publication. Prepared hydrogens and admitted modeled atoms
are physical sources, never model nodes. Functional DNA, Zn, water and excluded ligands are not
sources. APBS solves the declared linear Poisson–Boltzmann continuum-solvent pilot: 298.15 K,
interior/exterior permittivities 2/78.5, solvent radius 1.4 Å, and +1/−1 ions at 0.15 mol/L each.
Potential stays in kBT/e, with Boltzmann constant kB, temperature T and elementary charge e; no
actual charged probe is inserted. The two neutral Lennard–Jones probes use real AMBER14 types
`ALA:HB1`/`protein-HC` and `ALA:CB`/`protein-CT`. Diameters and well depths come from the installed
force-field XML; both differ, so this is not a size-only contrast. Distances convert from Å to nm,
energies remain kJ/mol. Arithmetic-mean diameters and geometric-mean well depths define mixing.
Default LJ targets sum the complete finite prepared protein. Each protein must pass an independent
OpenMM protein–probe interaction-group comparison with absolute/relative tolerances of 1e-5.

**Volumes and masks.** Nominal spacing is 1 Å and box margin 12 Å; actual multigrid-compatible
origin, spacings and shape are saved. Refinement at 0.75 times nominal spacing and a larger box
compare identical physical positions. The larger box preserves the original fine-grid spacing.
Numerical agreement does not establish physical-model accuracy. Default `potential_failure_policy:
mask` retains raw values everywhere but excludes unstable/noncommon positions from supervision.
The pilot requires 95% potential coverage overall and 80% in each present region, with tolerance
`absolute_error <= 0.25 + 0.1 * abs(reference)` in kBT/e. `fail` requires agreement everywhere.
These are configurable pilot thresholds, not validated universal tolerances. If coverage falls
below either threshold under `mask`, the **entire potential channel** is unavailable for that
protein; raw values and failed coverage audits remain available, while separately audited LJ
channels remain usable. `fail` still blocks publication. The strict YAML allows eight million
voxels per grid, with four serial solvers within 96 GiB; it never silently coarsens a grid.
Training fits scales on valid training targets only. Validation excludes unavailable proteins
from physical-loss averages and reports their count; a selected field absent from all training
proteins, or no valid validation targets, is an error. Review support by split and field before
interpreting VF: chemistry-dependent missingness can bias the supervised subset.

**Incomplete structures.** Missing coordinates do not automatically make a deposited structure
invalid: flexible atoms or entire residues may be unresolved experimentally. A molecular force
field nevertheless requires a chemically specified structure. PDB2PQR normally reconstructs some
missing heavy atoms as part of that preparation; `--noopt` does not disable heavy-atom repair.
WISDOM's strict observed-atom protocol deliberately does not treat those new coordinates as
observations. Its current three fields share this chemistry preparation, so an unsupported
preparation disables **all three**, not just APBS. A failed APBS coverage comparison, in contrast,
disables potential only after independent LJ audits succeed. Reports distinguish these cases,
retain full available atom-identity differences and show support separately by field and split.

An alternative conformation is a different observed placement of some atoms. Selecting atoms
individually can mix these alternatives in an older parent archive. Before PDB2PQR, WISDOM checks
observed terminal oxygen atoms named `OXT`: a carbonyl carbon must exist, its C--OXT distance must
be 0.9--2.2 Å, and that carbon cannot simultaneously continue a peptide to the next observed
residue at 0.9--1.9 Å. A contradiction leaves all three VF fields unavailable, with the atom
identity and measured distances in the audit. It does not remove the protein, change its base
coordinates or SLAE, or abort other proteins. WISDOM never silently renames split chains, removes
an observed oxygen or rebuilds the parent to force preparation to succeed. Unexplained losses,
movement or element changes of observed atoms remain errors. This targeted check is not a general
guarantee of chemical correctness; changing the parent conformation requires a new dataset.

`heavy_atom_policy: bounded` uses the following deliberately limited **pilot admission criteria**;
they are configurable safeguards, not experimentally validated universal tolerances:

| Configuration and selected value | Meaning |
|---|---|
| `repair_maximum_atoms: 32` | Maximum total newly modeled heavy atoms, including terminal OXT |
| `repair_maximum_fraction: 0.05` | Added heavy atoms divided by observed heavy atoms must also be at most 5% |
| `repair_maximum_per_residue: 5` | No residue receives more than five modeled heavy atoms |
| `repair_maximum_residue_fraction: 0.1` | At most 10% of observed residues need completion |
| `repair_minimum_nonbonded_fraction: 0.65` | Reject severe heavy-atom overlaps below 65% of the mixed force-field LJ diameter |
| `repair_exclusion_radius: 3.0` | Exclude optimizer queries within 3 Å of any modeled heavy atom |

All count limits apply simultaneously. N/CA/C/O backbone atoms must already exist, adjacent
observed peptide C--N distances must lie in 0.9--1.9 Å, no new residue is admitted, and observed
atoms cannot move. Modeled OXT requires the base-bound `sequence_context` to confirm the last
deposited chain token; a last observed residue before an unresolved tail is not enough. OpenMM
must parameterize the complete prepared structure. Modeled-heavy bonds must lie in 0.9--2.2 Å;
the nonbonded clash test excludes bonded and angle-neighbor pairs. PDB2PQR's atom-moving debumping
and side-chain optimization remain disabled so observed geometry is unchanged; bad modeled
placements are rejected, not hidden by moving experimental atoms.

Full raw computed fields remain archived. The query-bank exclusion uses float32 coordinates as
seen by the decoder; it avoids supervising the exact near-field placement of an unobserved atom.
It does not bound distant electrostatic effects or certify a repaired conformer. Counts,
identities, geometry checks, sequence evidence and reuse hashes remain in the reference audit;
support is reported by field/split. Repaired sources never become encoder features or nodes.
Masking is safe storage/optimization, not proof of representative or sufficient VF coverage.
Review results separately for observed-only and repaired proteins before making biological
claims. Raw historical audit evidence is in `benchmarks/physics_preparation_audit.json`.
Full-protein LJ evaluation uses bounded blocks of 32 queries with the same atom order and float64
reductions. A synthetic CPU microbenchmark measured 2.38× speedup with bit-identical values/masks
(`benchmarks/physics_lj_runtime.json`); this is not a whole-pipeline or cluster speed estimate.

Signed distance to heavy-atom van der Waals spheres defines interior and boundary. Connectivity
through solvent-probe-free voxels identifies reachable exterior and closed geometric voids, not
observed water or functional pockets. Reachable channels share exterior strata. Six-neighbor
connectivity is the default; 26 is optional. Stratification proposes 40% boundary, 25% nearby
exterior, 20% closed void and 15% other exterior, redistributing missing strata without duplicates.
Each LJ probe separately masks overlaps and energies above 10000 kJ/mol in absolute value; the
finite repulsive band remains available. Masked stored zeros are not zero-energy observations.

`physical_reference` is a base-hash-bound NPZ containing full fields, masks, numerical errors,
distance/accessibility, prepared atom correspondence, charges/LJ parameters, PQR, prepared PDB,
APBS input/OpenDX, units, versions and audits. `physical_queries` independently stores up to 16384
unique grid positions `[Q,3]`, targets/Boolean masks `[Q,3]`, region/distance/accessibility and grid
addresses. All retained atoms within 12 Å are candidates: `offsets[Q+1]` delimits contiguous
`atom_ids` lists, without a top-K cap. Empty neighborhoods are legitimate.

Only query geometry enters the decoder. It sums messages combining atomic states and distance
bases, then predicts potential/LJ1/LJ2 with three heads. Potential also receives mean atomic context
and query distance to the atomic centroid for approximate long-range information, not an exact
APBS/Coulomb solution. There is no dense query-by-atom matrix or new functional pooling/surface
encoder. Conditioned heads, frozen encoders, zero/permuted states, small decoders and no-message
encoders are explicit controls.

**Optimization.** `physics: null` leaves existing Training unchanged. `mode: pretrain` executes
only atom encoding and physical/control heads; functional labels, local GT and test are unused.
The inherited group-disjoint train/validation partition defines an in-domain physical split, not
an independent external corpus. Physical validation loss selects its epoch. `mode: joint` adds
physical errors to the existing functional loss, but checkpoints still use protein G only.

For field s, loss Ls averages normalized error only over its valid queries. Potential stays signed;
LJ uses `asinh(U/u0)`, where U is energy and u0 is median absolute valid training energy, floored at
1e-6 kJ/mol. Means and standard deviations also use training only. The objective is
`L_total = L_WISDOM + sum_s lambda_s * Ls`. Choose a global `objective.weight` or individual
`objective.weights`, never both. Huber error with beta 1 is default; MSE is optional. Default
training samples 512 queries per protein/update; validation uses 4096 fixed queries and final
evaluation covers the complete stored bank. LF records original-unit MAE/RMSE, spatial
Pearson/Spearman, potential-sign accuracy outside ±0.1 kBT/e, region supports and encoder-gradient
diagnostics. Undefined correlations stay unavailable, not zero. These are reconstruction metrics,
not biological binding accuracy.
`val_physics_unweighted_loss` excludes the auxiliary coefficients; unlike weighted loss, it remains
informative for the zero-pressure row. Unspecified individual field weights are zero.

Joint `precision: auto` retains BF16 dense layers on supported GPUs. Sparse surface derivatives
run in FP32; optimizer backward and the functional/physical gradient comparison run outside
forward autocast (automatic mixed precision). The comparison differentiates the same retained
graph without filling optimizer gradients or recomputing a forward. This avoids unsupported
BF16 sparse CUDA backward operations without disabling gradient diagnostics or changing losses.
The physical-only pilot remains explicitly FP32.

Pretraining publishes `physical-encoder.pt`, normalization and evaluation reports; native
continuation retains optimizer, RNG, epoch and best weights. Bind a reviewed file through
`physics_checkpoint: {file: path/to/physical-encoder.pt}` for compatible atomic transfer. Functional
and surface heads do not transfer. Physical heads transfer with matching train normalization;
`transfer_decoder: false` resets them. Source training groups/hashes are checked against destination
validation. `freeze_encoder: true` also disables atomic dropout/stochastic gates. The optional
`pretraining_task: masked_element` reconstructs hidden atomic categories as a genuine nonphysical
control, using a fixed epoch budget and its final epoch rather than untrained physical-head scores.

The joint YAML is a **from-scratch auxiliary pilot**, not the report's final VF1f conclusion.
Reviewed physical checkpoints and the eventual V16 architecture must be fixed explicitly before
T0–T4 transfer comparisons. No scientific winner or nested training Work is invented automatically.

## 5. Experiments and development

### 5.1. Training and protein viewers

Each training YAML is a single `wisdom.Training.Training` Work. `with` supplies scientific arguments;
`seeds`, `sweep` or `search` describe replications/candidates; `resources` and optional `execution`
govern admission. `objective` defines what LF optimizes. Do not infer required GPU reservations or
the number of concurrent Runs from the model size alone. LF manages execution, caching, results
and recovery; WISDOM owns the scientific data transformations and PyTorch loop.

`data_workers` controls subprocesses that decode protein arrays while the GPU processes the
previous batch; zero decodes in the training process. WISDOM initializes training, validation
and review workers with PyTorch's `file_system` sharing strategy. This transfers CPU tensors
through named shared memory (RAM, not a copy of the dataset on disk), without keeping one open
file descriptor per tensor. A descriptor is an operating-system handle; low per-process limits
can otherwise trigger `Too many open files` for batches with many per-protein operators and
external representations, even when NPZ archives are correctly closed. Training keeps one
prefetched batch per worker. This transport policy changes neither values nor seeds, does not
raise system limits, and leaves allocation/cleanup to PyTorch's `torch_shm_manager`. Shared
memory capacity still needs to accommodate the prefetched tensors; it is not unlimited.

Before the first epoch, WISDOM validates each requested split and its structural sidecars.
Logs report progress every 25 selected proteins and the time spent on each split. Surface
evaluation reuses those validated bindings instead of reconstructing the same validation/test
inputs twice. Atomic-context checks decode only atomic fields, but still verify the complete
base-file checksum and reconstruct all context fields. No validation is disabled or reused
across different Runs. LF records `dataset_setup_seconds`, `representation_fit_seconds` (when
applicable), and `startup_seconds`; the last measures application setup through training readiness,
not queueing, environment installation or remote data staging. A quiet GPU during these CPU/file
checks does not establish a hung process. Conversely, a `running` badge alone does not establish
scientific progress: compare LF's observation freshness with its persisted Run completion counts
and final execution result.

```bash
lf validate experiments/v1/wisdom_v1a.yaml
lf explain experiments/v9/wisdom_v9d.yaml
lf run experiments/v9/wisdom_v9d.yaml --dry-run
lf run experiments/v9/wisdom_v9d.yaml --on YOUR_CLUSTER
```

Configure final viewers in `with.visualization`: `mode` is `none`, `report`, `viewer` or `full`.
`none` creates no viewers or empty report tab; `report` embeds inspectors; `viewer` also writes
HTML/PLY; `full` adds prediction NPZ. `content: predictions` keeps probabilities, logits and GT;
the richer content options include additional channels. `maximum_proteins` and `maximum_points`
affect display only, not scientific metric coverage. Existing YAML comments list actual defaults,
size limits and eligibility options.

Enabled training viewers are produced once from that Run's restored best checkpoint, not every
epoch or after hidden best-seed selection. `surface_metrics_interval: 0` computes local diagnostics
at the final selected checkpoint; positive N also measures them every N validation epochs.
Reports appear as LF artifacts and WISDOM report tabs. Use a frozen reviewer such as
`experiments/v5/visualization/wisdom_v5a.yaml` to render imported checkpoints without retraining.
Select the source execution, saved seed and optional parameter filter in that YAML. A missing
checkpoint requires native recovery, not an automatic replacement training run.

### 5.2. Source organization and verification

Keep one class per Python file, including enums. Group by responsibility, not by experiment number:

| Package | Responsibility |
|---|---|
| `models/atomic`, `residue`, `surface`, `communication` | Atom/residue encoding, surface encoding and cross-representation transfer |
| `models/pooling`, `heads`, `refinement` | Protein aggregation, prediction heads and spatial evidence refinement |
| `models/physics`, `preprocessing/common/structure/assets/physics` | Query decoders, physical objectives/transfer and audited offline references |
| `models/initialization`, `optimization`, `losses`, `gating`, `architecture` | Initialization, optimization policies, objectives and explicit architecture controls |
| `data`, `evaluation`, `analysis` | Dataset ingestion/batching, scientific metrics and LF metric descriptions |
| `preprocessing/dna`, `preprocessing/zinc` | Task evidence, selection, orchestration and annotations |
| `preprocessing/common/independence`, `sampling`, `clustering`, `auditing` | Shared scientific selection stages |
| `preprocessing/common/structure/io`, `geometry`, `storage`, `assets`, `viewer` | Coordinates, geometric construction, validated persistence, optional assets and the shared inspector |
| `utils/structure` | Reused structure domain objects: protein → chain → residue → atom |
| `visualization`, `interpretability` | Checkpoint review/reporting and post-hoc concept analysis |

The public Work paths and `models/WisdomV1.py`, `WisdomV2.py`, `WisdomV3.py` remain stable.
Internal imports follow the new packages; no duplicate implementations or forests of forwarding
files remain. The commented Selection and Preprocessing orchestration stays readable as scientific
pseudocode. Stateless stage modules remain functions rather than artificial service classes.

After changing source code:

```bash
ruff check .
mypy src/wisdom
pytest -q
lf validate experiments/preprocess/dna/dna_preprocess.yaml
lf explain experiments/preprocess/dna/dna_preprocess.yaml
lf run experiments/preprocess/dna/dna_preprocess.yaml --dry-run
lf validate experiments/preprocess/dna/validate_dna.yaml
lf validate experiments/v1/wisdom_v1a.yaml
lf validate experiments/v2/wisdom_v2.yaml
```

Tests check contracts, numerical behavior and packaging. They do not establish scientific gains.
Browser, external-weight and production-data tests may require optional local inputs. Keep
performance measurements in `benchmarks/`, downloadable models in `data/models`, generated results
under LF management, and scientific discussion in the bundled report. Do not edit `build/lib`:
it is generated packaging output, not another source tree.

## 6. References

The report contains the full bibliography and qualifications. Main implementation references:

- [RCSB PDB](https://www.rcsb.org/) and [PDBx/mmCIF](https://mmcif.wwpdb.org/): deposited coordinates and their fields.
- [Gemmi](https://gemmi.readthedocs.io/): structure parsing and assembly handling.
- [MMseqs2](https://doi.org/10.1038/nbt.3988) and [Foldseek](https://doi.org/10.1038/s41587-023-01773-0): sequence and structure search.
- [HDBSCAN](https://doi.org/10.1007/978-3-642-37456-2_14): density-based clustering; noise is a legitimate outcome.
- [DiffusionNet](https://doi.org/10.1145/3507905): geometry-based surface learning; WISDOM uses its documented adaptation, not a claim of identical implementation.
- [Hard-Concrete gates](https://arxiv.org/abs/1712.01312): differentiable approximate L0 regularization.
- [SchNet](https://doi.org/10.48550/arXiv.1706.08566): continuous distance-filter interactions; WISDOM uses LF's public layer with its declared adapters and controls.
- [SLAE pinned public release](https://github.com/ProteinDesignLab/SLAE/tree/078b94e436b8eceb02251a254c3c224358ec6ee2): optional frozen atom encoder, atom37 vocabulary and base checkpoint; benchmark overlap remains unknown.
- [PyTorch multiprocessing](https://docs.pytorch.org/docs/2.11/multiprocessing.html#sharing-strategies): CPU-tensor sharing strategies and native shared-memory cleanup.
- [Fused Gromov–Wasserstein](https://proceedings.mlr.press/v97/titouan19a/titouan19a.pdf): joint comparison of attributes and relations; VN01 uses LF's POT-backed local solver, not a certified global optimum.
- [Conformal outlier p-values](https://doi.org/10.1214/22-AOS2244): the finite-rank calibration principle; VN01's negative-protein interpretation remains conditional on its sampling and independence assumptions.
- [PDB2PQR/PROPKA](https://pdb2pqr.readthedocs.io/en/latest/using/index.html): preparation and declared protonation conditions.
- [RCSB missing coordinates](https://pdb101.rcsb.org/learn/guide-to-understanding-pdb-data/missing-coordinates) and [PDB2PQR heavy-atom repair](https://pdb2pqr.readthedocs.io/en/latest/api/biomolecule.html): unresolved atoms and the distinction between observed geometry and modeled completion.
- [APBS grids](https://apbs.readthedocs.io/en/stable/using/input/old/elec/mg-auto.html) and [potential units](https://apbs.readthedocs.io/en/latest/using/input/old/elec/write.html): focusing, numerical grids and kBT/e output.
- [OpenMM interaction groups](https://docs.openmm.org/latest/api-python/generated/openmm.openmm.CustomNonbondedForce.html): independent protein–probe LJ reference.
