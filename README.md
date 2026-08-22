# WISDOM — protein structure and surface learning

**English** | [Español](README.es.md)

WISDOM builds a defensible protein–DNA-binding benchmark, converts protein structures into universal
geometric representations, projects DNA-interface reference data onto those fixed surfaces, and
trains the first two WISDOM models. “Geometric” means that the model reasons about molecular and
surface graphs instead of treating a protein as a rectangular image. Structural preprocessing is
strictly problem-independent: DNA labels live in separate catalogs and annotation sidecars.

The preprocessor converts PDB or PDBx/mmCIF structures into one deterministic, compact NPZ file per
protein. An NPZ is a compressed container of named numerical arrays. It is kept pickle-free: it does
not embed arbitrary serialized Python objects that could execute code when loaded. Each file combines
normalized atomic data, one spatial/covalent atomic graph, a fixed solvent-accessible surface point
cloud, local surface geometry, a surface graph, and surface-to-atom communication edges. Section 4.1
builds a plain-language picture of all these objects before the mathematical detail begins.

WISDOMv1 performs binary protein classification with weakly supervised local surface logits.
WISDOMv2 keeps that backbone unchanged and compares MIL pooling rules for small localized signals.
Neither version implements the later chemistry, bidirectional, quasi-geodesic, dMaSIF-inspired,
contrastive, or language-model stages in the roadmap.

## 0. Table of contents

- [1. Quick start](#1-quick-start)
- [2. Installation](#2-installation)
  - [2.1. Requirements](#21-requirements)
  - [2.2. Development installation](#22-development-installation)
- [3. DNA-binding benchmark and annotations](#3-dna-binding-benchmark-and-annotations)
  - [3.1. Selection and preprocessing actions](#31-selection-and-preprocessing-actions)
  - [3.2. Candidate evidence, contacts, conflicts, and splits](#32-candidate-evidence-contacts-conflicts-and-splits)
  - [3.3. Surface ground truth and sidecar contract](#33-surface-ground-truth-and-sidecar-contract)
- [4. Structural preprocessing](#4-structural-preprocessing)
  - [4.1. Mental model and complete data journey](#41-mental-model-and-complete-data-journey)
  - [4.2. Preparing, running, and inspecting a dataset](#42-preparing-running-and-inspecting-a-dataset)
  - [4.3. From manifest entry to normalized coordinates](#43-from-manifest-entry-to-normalized-coordinates)
  - [4.4. From normalized atoms to the atomic graph](#44-from-normalized-atoms-to-the-atomic-graph)
  - [4.5. From atomic spheres to surface geometry](#45-from-atomic-spheres-to-surface-geometry)
  - [4.6. From surface points to the final NPZ](#46-from-surface-points-to-the-final-npz)
  - [4.7. Validation, reproducibility, and parallel execution](#47-validation-reproducibility-and-parallel-execution)
  - [4.8. Code architecture and testing](#48-code-architecture-and-testing)
  - [4.9. Scientific limitations](#49-scientific-limitations)
- [5. Trainable WISDOM models](#5-trainable-wisdom-models)
  - [5.1. Dataset index and graph batching](#51-dataset-index-and-graph-batching)
  - [5.2. WISDOMv1 models, equations, and tensor shapes](#52-wisdomv1-models-equations-and-tensor-shapes)
  - [5.3. WISDOMv2 pooling and localization diagnostics](#53-wisdomv2-pooling-and-localization-diagnostics)
  - [5.4. Training, evaluation, and artifacts](#54-training-evaluation-and-artifacts)
- [6. Bibliography](#6-bibliography)

## 1. Quick start

The production path has two deliberately separate executions. First,
[`experiments/dna_select.yaml`](experiments/dna_select.yaml) invokes one resumable Python action that
discovers public candidates, verifies labels and structures, assigns homology-safe balanced splits,
and emits a small portable selection. Second,
[`experiments/dna_preprocess.yaml`](experiments/dna_preprocess.yaml) invokes the complete heavy action, consumes that frozen selection,
builds the expensive structural surfaces once, projects DNA evaluation sidecars, and publishes the
immutable LambdaForge dataset. The committed `data/dna/public-sources.json` contains only pinned
public-source provenance, not protein examples. A real selection is a large public-data operation;
validation and planning do not perform it.

The first two commands create and activate `.venv`, an isolated Python environment that prevents
WISDOM's packages from changing the rest of the system. Replace `/absolute/path/to/LambdaForge` with
the real directory of the local LambdaForge checkout. The two `pip install -e` commands install both
projects in **editable** mode, meaning source-code changes take effect without reinstalling.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e "/absolute/path/to/LambdaForge[adaptive-hpo,parquet]"
python -m pip install -e ".[dev]"

lf validate experiments/dna_select.yaml
lf run experiments/dna_select.yaml --dry-run

# After the real selection Task, fetch its small artifact as explained in Section 3.1.
lf validate experiments/dna_preprocess.yaml
lf inspect experiments/dna_preprocess.yaml --resolved
lf run experiments/dna_preprocess.yaml --dry-run
lf validate experiments/wisdom_v1.yaml
lf run experiments/wisdom_v1.yaml --dry-run
lf validate experiments/wisdom_v2.yaml
```

`validate` checks the compact YAML, function signature, imports, and resolvable arguments.
`inspect --resolved` shows the exact internal Work generated by LambdaForge 0.11. `run` executes the
selected function, which publishes one immutable placement only after every member and the canonical
index validate. `lf` and `lambdaforge` are equivalent command names. A local
publication has this shape:

```text
runs/datasets/published/wisdom-dna/2/<content-id-prefix>/
├── base/<structure-hash>.npz
├── structures/<source-structure-hash>.cif
├── <protein>.dna.npz
├── index.jsonl
├── dataset-artifact.json
└── ...
```

`index.jsonl` is the authoritative streaming index: each line names one protein, its split and
tier, its global DNA target, whether local ground truth is available, and checksummed base/sidecar
assets. Dilution membership is member metadata, so a smaller view reuses the same arrays. The
portable TXT/JSON/CSV selection remains the separate `selection` artifact; it is not duplicated into
the managed dataset. `dataset-artifact.json` stores the content ID and build provenance.
Open a universal representation without pickle as follows; Section 4.2 explains how to discover the
actual Registry placement before Sections 4.4–4.6 define every array in detail.

```python
import json

import numpy as np

with np.load("runs/datasets/published/wisdom-dna/2/<content-id-prefix>/base/<hash>.npz",
             allow_pickle=False) as protein:
    atom_positions    = protein["atom_positions"]
    atom_edges        = protein["atom_edge_index"]
    surface_positions = protein["surface_positions"]
    metadata          = json.loads(str(protein["metadata_json"].item()))
```

## 2. Installation

### 2.1. Requirements

- Python 3.10 or newer;
- LambdaForge `>=0.11.0,<0.12`, normally installed from its local checkout;
- a CPU environment with NumPy, SciPy, and Gemmi;
- Internet access only when remote PDB entries are absent from the raw cache.

WISDOM targets LambdaForge `0.11.0`. LambdaForge is the source of truth for function materialization,
file and dataset argument resolution, preprocessing iteration and resume, canonical indexes,
immutable versions, streaming publication, the placement Registry, logging, resources, and run
management. WISDOM remains responsible for protein interpretation, scientific geometry, exact NPZ
and sidecar validation, and protein-specific visualization.

### 2.2. Development installation

A **checkout** is a local copy of a Git repository. A **commit** is the exact recorded revision of
that copy. After replacing both placeholders below, the commands create the isolated environment
described in Section 1 and verify that installed dependencies are mutually compatible.

```bash
git clone <WISDOM repository URL>
cd WISDOM

python -m venv .venv
source .venv/bin/activate

python -m pip install -e "/absolute/path/to/LambdaForge[adaptive-hpo,parquet]"
python -m pip install -e ".[dev]"
python -m pip check
```

LambdaForge 0.11 turns each short `name/run/with/resources` YAML into a callable Work. Inside the two
preprocessing functions, its public `PreprocessingTask` still executes each record-level
source→transform→sink loop with workers, checkpoints, progress, and resume. The heavy function then
streams validated members through `lf.publish_dataset`. WISDOM does not implement another runner,
Registry, or publisher. Section 4.7 explains this boundary after the scientific flow is established.

## 3. DNA-binding benchmark and annotations

### 3.1. Selection and preprocessing actions

The scientific question is whether a selected protein chain binds DNA. A coordinate archive by
itself cannot answer that question: a deposited structure is one experiment under particular
conditions, and DNA may be absent even when the protein binds it in biology. WISDOM therefore keeps
evidence selection separate from universal geometry and evaluation targets. The separation is also
operational: public discovery can take hours, whereas its useful result is only a small membership
contract that should be reusable independently.

[`experiments/dna_select.yaml`](experiments/dna_select.yaml) therefore calls one resumable action.
It discovers public positives and negatives, verifies evidence and the minimum necessary structure,
quarantines contradictions, assigns complete external sequence clusters to splits, balances every
main split, and emits a compact `selection` artifact. It creates no surface or universal NPZ.

[`experiments/dna_preprocess.yaml`](experiments/dna_preprocess.yaml) calls the second action later.
The function first creates universal, label-free NPZ files from `selection/proteins.txt`; it then
joins the frozen catalog to that exact geometry report, reuses the coordinate cache, and creates
aligned DNA sidecars. Finally, `lf.publish_dataset` streams the validated logical members into
`index.jsonl`. Only this action publishes `wisdom-dna@2`; a failed run publishes no version.

This separates four LambdaForge 0.11 concepts. A **Work** is one configured call to a Python
function. A **run** is one execution with fingerprints and provenance. A **version** is the immutable
logical content addressed by `wisdom-dna@2`. A **placement** is one physical copy of that version,
for example on the local workstation or `citius-ctgpgpu12`. Copying verified identical bytes adds a
placement; it does not create another scientific version. The **DatasetRegistry** is authoritative
for managed placements. A **DataCatalog** is now reserved for aliases, external data, loader
definitions, or explicit institutional overrides; WISDOM no longer duplicates managed paths in
`data/datasets.yaml`.

LambdaForge 0.11 makes `lf run` the canonical entry point. The YAML `resources` block requests 36
CPUs, 128 GiB, and 24 hours for the heavy function; its `workers: 36` argument controls the matching
record process pool. Resource reservation and scientific concurrency are deliberately distinct.

The intended production commands make the artifact boundary explicit:

```bash
# First perform public discovery and selection.
lf validate experiments/dna_select.yaml
lf run experiments/dna_select.yaml --dry-run
lf run experiments/dna_select.yaml --on citius-ctgpgpu12

# Fetch only the compact output from the successful selection job.
lf artifact list JOB_ID
lf artifact fetch JOB_ID selection --output data/dna

# Then inspect and run the expensive immutable dataset build.
lf validate experiments/dna_preprocess.yaml
lf inspect experiments/dna_preprocess.yaml --resolved
lf run experiments/dna_preprocess.yaml --dry-run
lf run experiments/dna_preprocess.yaml --on citius-ctgpgpu12

# In another terminal, inspect all jobs or follow this build's durable log.
lf top --history 300
lf jobs logs latest --follow

# Inspect the immutable version and its selected local placement.
lf datasets show wisdom-dna@2
lf datasets stats wisdom-dna@2
lf datasets members wisdom-dna@2 --partition split=train --limit 20
lf datasets verify wisdom-dna@2
```

`lf artifact fetch` materializes the named directory as `data/dna/selection/`. It contains
`catalog.csv`, `identifiers.json`, `proteins.txt`, main/reserve TXT lists, labels, diluted views, and
self-contained JSON/Markdown/statistical/figure audits for the complete and diluted selections.
Large candidate checkpoints and discovery downloads are run evidence and are intentionally not
fetched. LambdaForge fingerprints the fetched directory argument: changed membership creates a new
run identity, while record checkpoints inside an interrupted matching run allow exact resume. It
never trusts an arbitrary output merely because a directory exists.

`train.txt`, `val.txt`, and `test.txt` are exactly class-balanced logical members;
`proteins.txt` additionally contains local-evaluation reserves. `identifiers.json` joins every ID to
its label, split, 30%-identity cluster, tier, and logical-member flag. `labels.csv` is the compact
spreadsheet view; a separate `splits.csv` would duplicate it and is therefore not emitted. The
selection-level `README.md` explains every file, while `audit.md` explains the observed quality
metrics and warnings. The final stage copies these contracts into the published dataset. Geometry
declares its parallel `raw` coordinate cache as an artifact binding for annotation, so selected
structures are not downloaded twice.

To place the same valid dataset on another cluster, copy the immutable version rather than rerunning
discovery, mapping, geometry, and annotation:

```bash
# Let LambdaForge choose a verified source placement and copy it to the target cluster.
lf datasets materialize wisdom-dna@2 --on OTHER_CLUSTER --strategy replicate --apply

# Or name both ends explicitly.
lf datasets replicate wisdom-dna@2 --from citius-ctgpgpu12 --to OTHER_CLUSTER --apply
```

Both commands verify content identity and register another placement of the same version. To rerun
only heavy preprocessing elsewhere, transfer the complete `selection` artifact rather than an
unlabelled list of IDs. To transfer a finished dataset, replication is the complete operation.

The invalid version 1 should be removed preview-first. `delete` verifies the registered content,
refuses active consumers, displays the exact root and bytes, and touches nothing until `--apply`:

```bash
# First inspect the deletion plan carefully; this command is read-only.
lf datasets delete wisdom-dna@1 --on citius-ctgpgpu12

# Delete only that verified physical placement and unregister the placement.
lf datasets delete wisdom-dna@1 --on citius-ctgpgpu12 --apply

# After every placement has been physically deleted, remove the empty version record.
lf datasets remove wisdom-dna@1
```

Do not run `datasets remove` first: it only forgets Registry metadata and deliberately leaves bytes
on disk, making safe managed cleanup harder. The commands above are instructions; WISDOM never
deletes a dataset automatically during migration.

LambdaForge 0.11 writes preprocessing progress to stderr, which becomes the durable job log. One
start line identifies the shard, worker count and workload; later coordinator-owned checkpoint lines
report aggregate `records`, `ok` and `failed` counts. The default ten-second throttle prevents noisy
output, and only the coordinator writes these summaries, so parallel workers cannot race while
formatting progress. In `lf top`, select the job and press `l` to open its complete log. An old job
cannot acquire messages retroactively; these lines appear on new attempts built with the compatible
LambdaForge version.

`dataset.version` is not a cache counter and is not the NPZ schema. `wisdom-dna@2` promises one
immutable membership and byte identity. This version replaces invalidly imbalanced version 1. Any
later intended content or scientific-contract change requires version 3 or higher; LambdaForge
refuses to overwrite an existing name/version with another content ID.

### 3.2. Candidate evidence, contacts, conflicts, and splits

**Pinned public sources.** [`data/dna/public-sources.json`](data/dna/public-sources.json) is a small
reproducibility input, not a list of examples. LambdaForge fingerprints its exact bytes because its
`PreprocessingTask` contract requires mutable source definitions to participate in task identity.
The standard command therefore needs no private dataset and no manually prepared negative file.
The manifest fixes the following source contracts:

- DyProL DNA v1, Zenodo record 19547616, published 13 April 2026. Its
  `DNA-1022_Train` and `DNA-256_Test` manifests contain a PDB-chain identifier, observed protein
  sequence, and equally long binary binding-residue mask. WISDOM range-downloads only these small
  members of the 14.5 GB ZIP and verifies their SHA-256 values; it does not download the ensembles.
  PDB entry letters are normalized, but mmCIF chain IDs are case-sensitive and therefore preserved:
  for example, `7S01_D` and `7S01_d` identify two different deposited chains. Source keys also
  include database, release, and partition, so the same protein reported by two sources does not
  collide. Exact repeated source rows are collapsed; conflicting rows with one key are rejected
  before any structure mapping begins.
- BTD-Combo at authors' Git commit `714756450e537cebbc3d9814a1fc059758fee58b`, dated
  1 September 2024. WISDOM reads only records labelled `non_DBP` from the published train and test
  FASTA files. The source paper constructs this negative class from Swiss-Prot with functional-term
  exclusion and sequence-redundancy filters. It is high-confidence biological inference, not proof
  that a protein can never contact DNA.
- BioLiP2 annotation snapshot dated 29 March 2026 is a contradiction index, not a label source.
  QuickGO's current REST API supplies Gene Ontology DNA-binding annotations and preserves explicit
  `NOT` qualifiers. A `NOT DNA binding` annotation strengthens provenance but is not mandatory.

**Positive construction.** DyProL's curated DNA-binding assertion supplies the global positive
label, its official train/test membership supplies development/external-test provenance, and its
binary mask supplies residue-level local ground truth. WISDOM downloads the referenced experimental
RCSB mmCIF, selects the declared chain, requires exact mask-to-observed-sequence alignment, and
retains the binding residue indices. If a declared DNA chain is present, Gemmi also verifies the
interface geometrically; however, the reviewed DyProL residue mask remains sufficient when explicit
DNA coordinates are unavailable. The selected structure is protein-only model input later—the DNA
or residue labels are retained exclusively for evaluation annotation.

Let (P) be the set of non-hydrogen atoms in the selected protein chain and (D) the non-hydrogen
atoms in the declared DNA chains. For protein atom (p\in P) and DNA atom (d\in D), let
(x_p,x_d\in\mathbb{R}^3) be their Cartesian centres in ångströms. A direct interface pair satisfies

\[
\lVert x_p-x_d\rVert_2 \leq r_c,
\qquad r_c=4.5\ \text{Å by default}.
\]

The Euclidean norm is the ordinary straight-line distance. The 4.5 Å cutoff captures direct and
close protein–DNA contacts; it does not claim that every such pair is a chemical bond. A KD-tree,
which indexes nearby coordinates spatially, finds radius neighbors without building a dense
(|P|\times|D|) matrix. An accepted positive needs positive evidence, DNA atoms, at least one atom
pair, and by default at least two distinct contacting protein residues. DNA present far away is
rejected as an unverified positive.

**Negative construction and filters.** A BTD-Combo `non_DBP` record first enters as a sequence, not
as a PDB chosen for lacking DNA. The official RCSB sequence API must find an exact experimental
polymer-entity match. Every exact match is inspected: the candidate is rejected if any matching PDB
entry contains a DNA polymer, if its mapped UniProt accession occurs in BioLiP2's DNA interactions,
or if QuickGO reports DNA-binding function. Mapping fails closed when there is no exact structure,
no UniProt identity, insufficient observed coverage, unacceptable resolution, or an ambiguous
single-chain selector. The accepted catalog records each Boolean check, the BTD label and commit,
and `negative_confidence=high`; no arbitrary numeric confidence score is invented.

When several experimental structures remain, deterministic ranking prefers X-ray, then electron
microscopy, then NMR and other experiments; within a method it prefers lower numerical resolution,
then lexical PDB/entity/chain identity. Every selected chain must cover at least 80% of the source
sequence, and X-ray/cryo-EM resolution must be no worse than 4 Å by default. Positives and negatives
therefore both use experimental RCSB structures: model training cannot solve the benchmark merely
by distinguishing PDB from AlphaFold geometry.

Absence of DNA in a PDB structure is never used by itself as evidence that a protein is
non-DNA-binding.

Positive and negative assertions for the same identifier, exact sequence, official 90%-identity
group, or split-defining 30%-identity group become `conflict`. Thus a BTD negative homologous to a
DyProL positive is quarantined rather than resolved silently. Conflicts retain their reason and both
evidence lists in `dataset-report/conflicts.csv`; mapping and quality rejections go to
`exclusions.csv`. A documented scientific rejection is an ordinary result and does not abort
curation, but an unexpected parser/program error does, preventing publication of a silently
incomplete scientific contract.

Cheap profiling records protein/total chain counts, observed and deposited sequence lengths,
missing-residue fraction, residue/atom counts, radius of gyration, coordinate span, principal-axis
extents, aspect ratio, interface residue count/fraction, experimental metadata, taxonomy, source
SHA-256, and evidence URLs. Contacting residues form a sparse graph: consecutive residues or
residues with heavy atoms within 8 Å are connected. Its components estimate the number and size of
separate regions; the interface-atom radius of gyration measures their spatial spread. These are
profiling/QC descriptors, not final surface ground truth or automatic difficulty filters. If (c) is
the mean atom position and there are (N) protein atoms, radius of gyration is

\[
R_g=\sqrt{\frac{1}{N}\sum_{i=1}^{N}\lVert x_i-c\rVert_2^2}.
\]

It describes global compactness cheaply; it is not a learned feature. Highly elongated structures
enter the `challenge` tier rather than disappearing silently. `core` is the ordinary
distribution-matched tier.

Exact sequences, homologous proteins, and chains from the same experimental deposition must not
leak between train, validation, and test. WISDOM uses official RCSB MMseqs2 30%-identity groups and
never runs a project-owned mass aligner. It joins rows into transitive components: two rows are
connected when they share either a sequence cluster or a four-character PDB deposition, and that
connection propagates through intermediate rows. A complete component enters only one split. DyProL
and BTD published test records stay in `test`; if their component touches development, only the
external-test rows remain. Development components are assigned approximately 80/20 to
train/validation by seeded SHA-256 ordering. Positive components are also held in
`validation_reserve` and `test_reserve`; reserves never train.

Source interleaving is not class balance: most BTD negative sequences have no exact acceptable
experimental RCSB chain, whereas most DyProL positives already name one. Version 1 consequently
passed byte/schema verification while containing 1,210 positives and only 62 negatives; verification
proved the published bytes were intact, not that the benchmark design was appropriate. Version 2
therefore removes the negative source cap, evaluates every curated BTD negative, and balances only
after evidence filtering and homology-safe split assignment. For split (s), let (n_{s,0}) and
(n_{s,1}) be its accepted negative and positive counts. WISDOM retains

\[
m_s=\min(n_{s,0},n_{s,1})
\]

members from each class. A seeded SHA-256 order selects the majority-class prefix, making the choice
independent of worker completion order and repeatable across clusters. Thus every main split has
exactly (m_s) negatives and (m_s) positives. If either class is absent in any main split, publication
fails. This guarantees binary class parity; it does not claim demographic, taxonomic, functional,
or structural distributional fairness, which remains visible in the report. In particular, the
current positives and negatives come from disjoint source datasets. That source/label coupling is a
scientific warning because a model may exploit source-specific selection bias. Geometry and
annotation also use `on_error: fail`, so a selected member cannot silently disappear and break
parity downstream.

The canonical `catalog.csv` and typed `catalog.parquet` contain label/split, UniProt/PDB/chain,
source version/URL/checksum/record, sequence length and 30%/90% cluster IDs, experimental
method/resolution/taxonomy, negative-filter results, `local_gt_expected`, reference complex/DNA
chains, and binding-residue indices where applicable. Minimal `train.txt`, `val.txt`, `test.txt`,
reserve TXT files, and `proteins.txt` contain only `PDBID_CHAINS` selectors.
`identifiers.json` provides the same selection with label, split, cluster, tier, and explicit
main-member/reserve status. Each complete or diluted view includes `audit.json`, an explained
`audit.md`, tidy `statistics.csv`, and `distributions.png`. They verify file agreement, exact balance,
identifier/exact-sequence/family/PDB/coordinate leakage, fixed evaluation membership, nested
training views, family breadth, geometric tiers, sources, and structural distributions.

### 3.3. Surface ground truth and sidecar contract

The universal NPZ describes the selected protein alone. It contains no `DNA`, `label`, `target`,
`split`, or benchmark field. Consequently DNA cannot leak into atomic chemistry, surface geometry,
or neural features. The annotator creates a separate sidecar on the **same ordered surface points**,
does not regenerate a surface, and never rewrites the base file. It has two explicit local-target
routes: DNA-envelope distance when reference DNA coordinates exist, and projection of DyProL
binding-residue labels when they do not.

The base surface is centered for numerical stability. If (s'_i) is stored point (i) and (o) is
the stored `coordinate_origin`, its source-frame coordinate is (s_i=s'_i+o). For DNA atom (j),
let (x_j) be its source-frame centre and (r_j) its Gemmi tabulated van der Waals radius. The
physical surface gap is

\[
d_i=\min_j\left(\lVert s_i-x_j\rVert_2-r_j\right).
\]

Subtracting the DNA atom radius changes a centre distance into an approximate distance from the
protein surface point to the DNA van der Waals envelope. With positive gap (a=1.4) Å and negative
gap (b=3.0) Å, the primary arrays are defined as follows:

\[
y_i^{hard}=\mathbb{1}[d_i\leq a],\qquad
m_i=\mathbb{1}[d_i\leq a\ \lor\ d_i\geq b].
\]

Here (mathbb{1}) is one when its condition is true. `surface_target_hard` stores
(y_i^{hard}); `surface_valid_mask` stores (m_i). Points with (a<d_i<b) form an ambiguity band:
they remain available for visualization but are excluded from binary surface metrics. The soft
target changes continuously rather than jumping at one cutoff. With
(t_i=\operatorname{clip}((d_i-a)/(b-a),0,1)),

\[
y_i^{soft}=\frac{1+\cos(\pi t_i)}{2}.
\]

Thus it is exactly one at or inside the confident interface, exactly zero beyond the confident
negative boundary, and smooth between them. The sidecar also stores `surface_distance_to_dna`, a
distance-validity mask, hard targets for configured sensitivity cutoffs, those cutoff values, a
schema/provenance JSON scalar, and the SHA-256 of the exact base NPZ. Curated global negatives have
hard/soft zero at every valid point. Their DNA distance is not computable, so it is NaN only where
`surface_distance_valid` is false; it is never disguised as zero distance.

For residue-mask projection, let (q_i) be surface point (i), (A(i)) the atoms connected to it
by the preprocessed sparse atom–surface graph, and (ho(a)) atom (a)'s zero-based residue index.
The nearest represented atom and target are

\[
a_i^*=\arg\min_{a\in A(i)}\lVert q_i-x_a\rVert_2,
\qquad
y_i^{hard}=\mathbb{1}[\rho(a_i^*)\in B],
\]

where (B) is the set of `1` positions in DyProL's mask. This transfers a curated residue region
to the fixed point discretization without using it as model input. If a point unexpectedly has no
atom–surface edge, the nearest protein atom supplies a deterministic fallback. Distance-threshold
sensitivity is not physically defined for this route, so metadata identifies
`local_gt_method=binding_residue_mask`.

Global and local eligibility are deliberately different. A reliable positive may train from its
protein label even when `local_gt_expected=false`. After annotation, `local_gt_available` is true
for a positive only if at least one surface point is positive. A zero-positive projection keeps
global `label=1`, makes every local validity entry false, records
`zero_positive_surface_points`, and is excluded from localization metrics—never converted into an
all-negative surface. The global `manifest.csv` retains main train/validation/test examples.
Reserves are deliberately absent from this manifest and from the published `index.jsonl`; their files exist
only for audited local-GT replacement. A
separate `local-manifest.csv` contains evaluable validation/test examples and deterministic
same-partition reserve replacements; `annotation-report.json` records every original, reason, and
replacement. Global and local reports state their evaluated protein counts separately.

For learning curves, the selection Task creates every fraction listed in `dilutions`—currently 10%,
25%, 50%, and 75%—under `subsets/<fraction>/`. Validation and test remain identical to the complete
selection in every view. If the complete training split contains (N_{train,c}) examples of class
(c), fraction (f) retains only in training

\[
n_{train,c}(f)=\max\left(1,\left\lfloor fN_{train,c}\right\rfloor\right).
\]

Both training classes use the same count, so every training view remains balanced. Training rows are
ordered breadth-first across external 30%-identity clusters: one member of each cluster is visited
before a cluster contributes a second. Seeded ordering makes the training views deterministic and
nested: 10% is contained in 25%, then 50%, 75%, and the full training selection. Keeping validation
and test fixed ensures that changes in a learning curve measure training-data quantity rather than
changes in the evaluation questions. Original split ownership is never changed, so the cross-split
family/PDB barrier remains intact. The heavy action preprocesses the union once and records each
member's view names in `index.jsonl` metadata. To run one view, set `subset: 25pct` in the model YAML;
all views reference the same NPZ and sidecar assets.

With represented surface area weights (w_i>0), annotation records

\[
A_+=\sum_i w_i y_i^{hard},\qquad
A=\sum_i w_i,\qquad
f_{interface}=A_+/A.
\]

These are positive area, total area, and interface fraction. Connected components of positive
points in the sparse surface graph give `number_of_positive_regions`, enabling site-size and
single-/multi-region analyses without changing weakly supervised training.

A globally positive protein may be usable for weakly supervised training even when it is excluded
from surface-localization evaluation because no reliable local ground truth can be generated.

The sink rejects object arrays, length mismatches, invalid probabilities/masks, finite values where
distance is declared unavailable, and base-fingerprint disagreement before atomic publication. The
sidecar itself stays small. For portability, the annotation dataset also packages a byte-identical
copy of each universal NPZ under `annotations/base/<sha256>.npz`; it never rewrites the source.
The resulting relative-path manifests join `file`, `annotation`, global `label`, explicit `split`,
stable `identifier`, and `tier`, so the logical dataset can move to another mount. `WisdomDataset`
verifies the sidecar fingerprint again at load time.

The final `index.jsonl` expresses the same main-split collection in LambdaForge's canonical
dataset model. Each train/validation/test protein is one member with `split` and `tier` partitions;
`dna_binding` and
`local_ground_truth` targets; scientific surface statistics; and checksummed `universal_npz`,
`dna_annotation`, and selected `source_structure` assets. The compact selection artifact separately
retains `catalog.csv`, manifests, split TXT files, `identifiers.json`, and its audit report.
Consequently the managed content ID depends on scientific
membership and exact bytes, not on their workstation/cluster path. Intermediate selection and
geometry directories may later leave their Task/stage caches without making the published version
incomplete: its catalog-relative `structures/` paths, base NPZ files, and sidecars are all present.

## 4. Structural preprocessing

Preprocessing is the conversion from coordinate files written for structural biology into numeric
arrays that a later machine-learning model can consume. This section follows that conversion in the
same order as the program. Each subsection assumes only the concepts introduced before it.

### 4.1. Mental model and complete data journey

**From a physical protein to a coordinate file.**

A protein is a physical molecule made from atoms. Experiments and structure-prediction programs do
not give WISDOM the physical molecule itself; they provide a **coordinate file**. Such a file is a
structured table describing atom names, chemical elements, and three-dimensional positions. The
position of one atom is a triple `(x, y, z)` measured in ångströms, where one ångström (Å) is
`10^-10 m`.

Structural files organize atoms into a hierarchy:

- an **atom** is one chemical element at one position;
- a **residue** is a chemical building block, usually one amino acid, containing several atoms;
- a **chain** is an ordered sequence of residues;
- a **model** is one complete coordinate interpretation of the structure. A file may contain
  several models, for example alternative experimental conformations.

The two supported format families are PDB and PDBx/mmCIF. They encode broadly the same structural
ideas with different syntax. WISDOM delegates their parsing to Gemmi rather than trying to interpret
their text manually. Compressed `.gz` files contain the same PDB or mmCIF text after gzip
decompression; gzip changes storage size, not molecular meaning.

**Why the dataset starts with a TXT manifest.**

A scientific dataset may mix public Protein Data Bank entries with private or locally generated
coordinate files. Copying all structures into one directory would make the dataset harder to move,
audit, and reproduce. WISDOM therefore starts from a small **manifest**: a TXT file in which every
non-comment line says where one protein comes from.

A line can be a public PDB identifier such as `4hhb_AB`, in which case WISDOM can obtain the
coordinate file from RCSB PDB, or a local path such as `../structures/model.cif.gz`. The manifest is
the ordered definition of the dataset; the coordinate files are its physical inputs. This separation
lets the same manifest work with an existing cache, download missing public entries, and report a
failure for one protein without losing successful results for the others.

**Why WISDOM creates three graphs.**

A later geometric model needs more than an unordered table of atoms. It needs to know which objects
may exchange information. WISDOM represents those possible exchanges with **graphs**. A graph is a
set of nodes plus a set of edges; an edge says that two nodes are related. The edge is not itself a
chemical force or a learned message. It is a fixed structural connection on which a future model may
operate.

WISDOM creates three complementary graphs:

1. The **atomic graph** uses atoms as nodes. It joins atoms that are spatially close, chemically
   bonded, or both. It therefore describes both three-dimensional neighborhoods and molecular
   connectivity.
2. The **surface graph** uses sampled molecular-surface points as nodes. It joins nearby points that
   plausibly belong to the same local sheet of the surface. It gives a future surface model a sparse
   neighborhood without constructing every possible point pair.
3. The **surface-to-atom graph** is bipartite: its left nodes are surface points and its right nodes
   are atoms. It joins a surface point to nearby atoms so future computation can move information
   between the molecular interior and its boundary. “Bipartite” simply means that edges cross
   between two different node types.

The NPZ output contains node measurements and these edge lists, but no neural-network activations,
embeddings, attention values, or predictions.

**The complete data journey.**

With those objects in mind, one manifest line follows this path:

```text
manifest line
    -> existing local file or safely downloaded PDBx/mmCIF file
    -> selected model, chains, residues, and atoms
    -> Protein -> Chain -> Residue -> Atom hierarchy
    -> compact atom arrays and atomic graph
    -> sampled solvent-accessible boundary and its local geometry
    -> surface graph and surface-to-atom graph
    -> validated, compressed NPZ plus provenance metadata
```

Each arrow consumes the result immediately above it. Section 4.2 explains how to run and inspect this
journey. Sections 4.3–4.7 then revisit the same arrows in scientific and mathematical detail.

### 4.2. Preparing, running, and inspecting a dataset

The file bound to the logical input `protein_identifiers` contains one structure per line. A logical
name is the stable name used by code; LambdaForge resolves it to the physical path declared under
top-level `inputs`. Empty lines and lines whose first non-whitespace character is `#` are ignored.
Exact duplicate lines are removed while preserving the first occurrence. The order matters because
the final report is restored to this same order even when proteins finish at different times.

The preprocessor accepts one `protein_identifiers` manifest per run, not a list. In this repository,
`proteins.txt` is the master preprocessing manifest: its 450 unique identifiers are exactly the
disjoint union of 148 training, 148 validation, and 154 test identifiers. Processing the master file
once is therefore preferable to running the same scientific transformation separately per split.

Split membership remains dataset metadata rather than molecular structure. Keep `train.txt`,
`val.txt`, and `test.txt` and let the later training/data-loading stage select the already processed
NPZ files. Do not infer an output filename merely by copying identifier capitalization: use the
`identifier -> output` mapping in `preprocessing-report.json`, whose `identifier` preserves the exact
TXT record. LambdaForge's record manifest instead uses an internal stable key so it can resume and
shard records without treating a display filename as identity. Automated tests verify that the three
split files do not overlap and that their union continues to equal `proteins.txt`.

**Remote entries.**

```text
1abc
4hhb_A
4hhb_AB
```

The four-character code `4hhb` is the public identifier assigned by the Protein Data Bank. An
underscore introduces the optional chain selector described in Section 4.1: `4hhb_AB` means “use
PDB entry `4hhb`, but retain only chains `A` and `B`.” Chain characters are concatenated because each
character is one chain ID; commas and the former `#A,B` form are invalid. A selector on the line is
specific to that protein and therefore takes precedence over the global `config.chains` setting.

**Local structures.**

```text
/data/protein.pdb
/data/protein.pdb.gz
/data/protein.cif
/data/protein.mmcif
/data/protein.cif.gz
../structures/protein.mmcif.gz
```

Relative paths are resolved relative to the TXT file. A local filename is opaque: `_AB` in a local
filename is **not** interpreted as a chain selector. Use the global `chains` configuration for local
files. Every local file, or a parent directory containing it, must also appear in the top-level YAML
`inputs`. Declaring the input tells LambdaForge which external bytes the computation depends on. It
can then include those bytes in the **task fingerprint**, a stable identity for this exact run input
and configuration, instead of silently reusing a result produced from different coordinates.

Only `.pdb`, `.cif`, `.mmcif`, and their gzip-compressed variants are accepted. BinaryCIF, MMTF,
XML, trajectories, and archive containers are outside the current input contract.

**Configuration and execution.**

[`experiments/dna_preprocess.yaml`](experiments/dna_preprocess.yaml) is the human-editable structural
description. Its short LambdaForge 0.11 YAML selects the prior `selection` directory, scientific
parameters, worker count, dataset identity, and resources, then calls one complete Python action.

```bash
lf validate experiments/dna_preprocess.yaml
lf inspect experiments/dna_preprocess.yaml --resolved
lf run experiments/dna_preprocess.yaml --dry-run
lf run experiments/dna_preprocess.yaml
```

`validate` catches malformed arguments, missing staged inputs, and unavailable Python callables.
`inspect --resolved` reveals the internal Work without exposing that implementation in authored
YAML. `--dry-run` submits nothing and transforms no proteins. The final command performs the
complete journey from Section 4.1 and publishes only after the final index validates.

The three preprocessing concepts have narrow roles. `ProteinSource` reads the named
`protein_identifiers` input and gives every unique TXT line a stable key. `PreprocessPipeline` is the
transform: it receives one line and returns one validated in-memory protein representation.
`ProteinSink` is the only publication boundary: it atomically writes the NPZ, decides whether an
existing NPZ is scientifically reusable, and produces the protein-specific report. LambdaForge
surrounds those three classes with iteration, processes, checkpoints, record errors, manifests, and
the final dataset identity.

**Configuration reference.**

| Parameter | Default | Meaning |
|---|---:|---|
| `selection` | no default | Exact directory fetched from the independent selection Work. |
| `dataset_name` | `wisdom-dna` | Stable managed dataset name passed to `lf.publish_dataset`. |
| `dataset_version` | `2` | Immutable release label; intended byte changes require a new value. |
| `workers` | `36` | Spawned record processes, normally one per requested CPU. |
| internal workload | `cpu` | Use processes for geometry/annotation while the parent serializes sink writes. |
| internal failure policy | `fail` | Stop if a balanced selected member cannot produce its representation. |
| internal checkpoint interval | `1` | Persist LambdaForge record progress after every completed protein. |
| internal progress interval | `10 s` | Emit one coordinator-owned aggregate line without worker logging races. |
| internal download policy | `true` | Download from RCSB only when a remote entry is absent from the run cache. |
| `resources` | `36 CPU, 128 GiB, 24 h` | Outer Work allocation shared sequentially by geometry and annotation. |
| internal `model_index` | `0` | Zero-based structural model selected from the coordinate file. |
| `chains` | `[]` | Global chain filter, overridden by a remote line selector. |
| `include_hydrogens` | `false` | Retain explicit hydrogen atoms. No hydrogens are invented. |
| `include_waters` | `false` | Retain crystallographic water residues. |
| `include_nonpolymer` | `false` | Retain non-polymer residues such as ligands. |
| `include_metals` | `false` | Retain metal atoms/residues. |
| `center_coordinates` | `true` | Subtract the filtered atom centroid and store it as provenance. |
| `atom_radius` | `6.0 Å` | Spatial atomic graph cutoff. |
| `surface_resolution` | `1.0 Å` | Main surface sampling length scale, denoted below by `h`. |
| `probe_radius` | `1.4 Å` | Radius added to vdW spheres for the solvent-accessible surface. |
| `atom_surface_radius` | `6.0 Å` | Surface-to-atom communication cutoff. |
| `curvature_scales` | `[2.5, 5.0]` | Positive radius multipliers for independently fitted curvature triplets. |

`surface_resolution` controls candidate density, voxel size, and the surface graph radius. Curvature
scales are independently configurable: a value `s` fits one `[H,K,C]` triplet inside radius `s h`,
where `h=surface_resolution`. Adding or removing scales changes
`surface_curvatures` from `[M,S,3]` to the new number `S`; the WISDOMv1 YAML must then set
`curvature_features=3S` and the surface projection input size to `hidden_dim+3S`.

The execution fields and scientific fields are intentionally separate. Changing `workers`,
`workload`, checkpoint cadence, or requested resources changes how the same records are scheduled;
it must not change their NPZ bytes or dataset identity. Changing a scientific field changes the
geometry and therefore invalidates reuse. WISDOM no longer carries paths, worker counts, resume
flags, or failure policy inside `PreprocessConfig`.

**Inspecting the result.**

Each successful input produces exactly one **NPZ**, a compressed container holding several named
NumPy arrays in one file. Section 4.6 describes every stored array. `preprocessing-report.json` is a
separate text report containing ordered records
with status (`processed`, `skipped`, or `failed`), elapsed time, array bytes, compressed file bytes,
graph sizes, surface size, and warnings. A failed record includes its identifier, exception type,
and message without removing successful records.

`processed` means a new NPZ was built, `skipped` means an existing scientifically compatible NPZ was
accepted by the resume checks in Section 4.7, and `failed` means that only this manifest line could
not complete. The JSON metadata inside each NPZ is **provenance**: an audit record of where the
coordinates came from, which settings transformed them, and which software versions performed the
work. Provenance does not change molecular geometry; it makes that geometry traceable.

The hexadecimal directory names are not arbitrary temporary names. Each is the first part of a
SHA-256 fingerprint for one exact combination of task definition and declared input bytes. Keeping
different fingerprints in separate immutable-looking directories prevents a newer experiment from
silently overwriting older scientific evidence. The following commands provide a readable index and
an HTML page, so finding a run does not require opening every directory by hand:

```bash
lambdaforge results runs/tasks --no-archived
lambdaforge dashboard runs/tasks --output runs/index.html
```

The result list shows each attempt, status, fingerprint, and `result.json` path. The dashboard groups
the same information visually. Once a result is used for training or publication, retain its exact
fingerprint rather than a moving “latest” alias; `config.yaml` and `result.json` inside that directory
explain what produced it.

Useful inspection code:

```python
import json

import numpy as np

with np.load("protein.npz", allow_pickle=False) as archive:
    print(archive.files)
    print("atoms:", archive["atom_positions"].shape[0])
    print("atomic edges:", archive["atom_edge_index"].shape[1])
    print("surface points:", archive["surface_positions"].shape[0])
    print("surface edges:", archive["surface_edge_index"].shape[1])
    print(json.loads(str(archive["metadata_json"].item())))
```

LambdaForge 0.11 can safely inspect, constrain, and render explicit arrays without WISDOM-specific
debug code. For example:

```bash
lambdaforge artifact inspect protein.npz --json
lambdaforge artifact inspect protein.npz --array surface_curvatures --rows 20
lambdaforge artifact validate protein.npz --require-array surface_positions \
  --shape 'surface_positions=*,3' --finite
lambdaforge artifact visualize protein.npz --type point-cloud \
  --positions surface_positions --output surface.html
lambdaforge artifact visualize protein.npz --type graph \
  --nodes surface_positions --edges surface_edge_index --output surface-graph.svg
```

These generic tools answer what an array contains and whether explicit shape/finite constraints hold.
They do not know protein chemistry, the signed surface gap, normal orientation, curvature identities,
or the relationship between the atomic and surface graphs. The independent WISDOM validator and
viewer in Section 4.7 remain necessary for those domain questions.

### 4.3. From manifest entry to normalized coordinates

**Shared mathematical language.**

The remaining sections describe the detailed transformation previewed in Section 4.1. They use a
small amount of notation repeatedly, so it is introduced here before any algorithm depends on it.

A bold lowercase letter such as `x` or `y` represents a three-dimensional point. Its three
components are `x_1`, `x_2`, and `x_3`, corresponding to the Cartesian x, y, and z axes. All
coordinates, distances, and radii use ångströms.

The straight-line or **Euclidean distance** between points `x` and `y` is

```math
d(\mathbf{x},\mathbf{y}) = \lVert \mathbf{x}-\mathbf{y}\rVert_2
                           = \sqrt{\sum_{q=1}^{3}(x_q-y_q)^2}.
```

Here `q` visits the three coordinate axes. For each axis, the difference is squared; the three
squares are added; and the square root converts that sum back to a distance. The double bars with
subscript `2`, `||.||_2`, are a compact name for this operation.

Many later rules ask “which points lie within a radius?” Recomputing the distance between every pair
would require a square table that grows rapidly with molecule size. WISDOM instead uses a
**KD-tree**, a spatial index that partitions three-dimensional space so nearby candidates can be
found without enumerating all pairs. The KD-tree changes search efficiency only: every accepted
edge or neighborhood is still decided by the Euclidean distance above.

**Obtaining coordinate files.**

The first program stage turns every manifest line from Section 4.2 into a readable local file. Local
paths already satisfy that requirement. A public identifier may require a download, so WISDOM keeps
a cache: a directory of previously downloaded files that can be reused by later runs.

Remote identifiers are normalized to lowercase for cache names and resolved as
`downloads/<pdb_id>.cif.gz`, where `downloads` is the physical path of the named output. Missing
entries are downloaded from
`https://files.rcsb.org/download/<PDB_ID>.cif.gz`, the compressed PDBx/mmCIF endpoint documented by
RCSB PDB.

LambdaForge and WISDOM divide the work as follows:

1. `ProteinSource` parses and deduplicates TXT lines and yields stable record keys.
2. LambdaForge schedules each record according to `workers` and `workload`.
3. The record transform resolves its local path or downloads its missing remote entry.
4. WISDOM hashes the ready source with SHA-256 and performs the scientific transformation.
5. LambdaForge's parent process passes each completed record to `ProteinSink` and checkpoints it.

Downloads therefore share the same bounded record concurrency as the transformations; WISDOM does
not maintain a hidden download thread pool. A cached entry returns immediately. Different records
for chains of the same PDB may reach the cache concurrently, so the single-writer protocol below is
still required.

Two runs could request the same missing PDB entry simultaneously. A small `.lock` file acts as a
single-writer sign: exclusive creation with `O_EXCL` lets only one run become the downloader. Other
runs check every `0.1 s` for the finished target and stop waiting after `180 s` rather than hanging
forever.

The downloader does not write directly to the final cache name. It streams HTTP data in `1 MiB`
blocks into a uniquely named temporary file. `flush` and `fsync` ask the operating system to finish
writing those bytes; opening the result through gzip verifies that compressed data expands to a
nonempty payload. Only then does `os.replace` rename the temporary file to the final name in one
atomic filesystem operation. Thus another process sees either no cached structure or a completed
one, never a half-downloaded file.

The format label is inferred from the validated suffix; coordinate interpretation itself is
delegated to Gemmi. Finally, WISDOM computes SHA-256, a content fingerprint that maps the exact file
bytes to a fixed-length hexadecimal value. If even one source byte changes, the digest is expected
to change, allowing Section 4.7 to reject stale outputs. For compressed input, the hash covers the
compressed bytes actually supplied. LambdaForge hashes declared static inputs for the task
fingerprint; WISDOM's per-source hash additionally covers dynamically resolved paths and downloaded
coordinates.

**The Gemmi boundary.**

Gemmi is a structural-biology library that understands the syntax and data dictionaries of PDB and
PDBx/mmCIF. After decompressing gzip when necessary, it exposes elements, models, chains, residues,
atoms, coordinates, charges, and source-declared connections through one programming interface.
This boundary matters because format parsing has many edge cases that are unrelated to WISDOM's
scientific representation.

No Gemmi object is stored after reading. WISDOM copies the selected information into its simpler
`Protein -> Chain -> Residue -> Atom` ownership hierarchy. It places audit information in the
separate `ProteinMetadata` object; “metadata” here means information *about* the representation,
such as source hash and coordinate origin, rather than atoms belonging to the molecule.

**Models, chains, residues, and filtering.**

The reader now narrows the parsed file to the molecule requested by the manifest and configuration.
It first validates `model_index`. The default `0` means the first complete coordinate model; WISDOM
does not average several experimental models. It then enumerates chain names and rejects a requested
chain that does not exist, because silently returning a different chain would corrupt dataset
meaning.

Within selected chains, a **polymer residue** belongs to the linked amino-acid or nucleic-acid chain.
A **non-polymer residue** is a separate component such as a ligand. Water molecules, metal ions,
non-polymer components, and hydrogen atoms can each be retained or removed by configuration. These
filters run before graph and surface construction, so removed matter contributes to neither
geometry nor edges. WISDOM never invents missing atoms or repairs incomplete residues.

Water recognition uses Gemmi. Its `EntityType.Polymer` category supplies polymer identity. Metal
recognition uses Gemmi's periodic-table classification plus the centralized fallback set in
`chemical_data.py`. A residue must have a numeric sequence ID: that number, together with chain and
optional insertion code, is the address later used to reconnect bonds to the correct atoms.

**Alternate atom locations.**

A crystal structure can record more than one observed position for the same named atom. The
**alternate-location code**, usually called altLoc, labels those alternatives. **Occupancy** is the
source's estimated fraction of molecules represented by one alternative. Because downstream arrays
require one position per atom, WISDOM retains at most one candidate for each atom name and ranks it
by:

1. larger occupancy;
2. blank or `A` alternate-location code over other codes;
3. blank before `A`, then alphabetical code order.

The first rule chooses the most prevalent observation. Blank and `A` are conventional primary
locations and win an occupancy tie; the final alphabetical rule makes any remaining tie repeatable.
Atom names are then emitted in sorted order within each residue. Chains and residues preserve source
order. Consequently atom indices are deterministic and equal `0, 1, ..., N-1` in hierarchy order.

**Coordinate centering.**

A coordinate file places the molecule in an arbitrary global frame: translating every atom by the
same vector changes the file coordinates but not molecular shape or internal distances. Centering
removes that irrelevant translation and keeps numeric magnitudes close to the molecule.

Let `N` be the number of retained atoms. Let `x_i` be the three-dimensional source coordinate of
atom `i`, where `i` runs from `1` to `N`. First compute the centroid `o`, the componentwise average
position of all retained atoms. Then subtract that same centroid from every atom:

```math
\mathbf{o} = \frac{1}{N}\sum_{i=1}^{N}\mathbf{x}_i,
\qquad
\mathbf{x}'_i = \mathbf{x}_i - \mathbf{o}.
```

The prime in `x'_i` means “centered coordinate”; it is not a second atom. Because the same `o` was
subtracted everywhere, pairwise distances are unchanged. `coordinate_origin` stores `o` as
provenance. Adding it back, `x_i=x'_i+o`, recovers the source frame. If centering is disabled,
WISDOM leaves coordinates untouched and records `(0,0,0)` to indicate that no translation occurred.

Every coordinate must be finite: NaN and positive/negative infinity cannot define a physical point
or a valid distance. At least one retained atom must have atomic number greater than one, which
ensures the filtered structure is not an isolated collection of hydrogen atoms.

**Connections declared by the source.**

Some coordinate files explicitly state that two atom addresses are connected. WISDOM maps each
address `(chain, residue number, insertion code, atom name)` to the consecutive atom index created
above. A connection whose endpoint was filtered out cannot be represented and is skipped.

The relevant chemical terms are:

- a **covalent bond** means atoms share electron density and form the molecule's chemical skeleton;
- a **disulfide bond** is a covalent S–S link between sulfur atoms of two cysteine residues;
- **metal coordination** associates a metal ion with surrounding donor atoms but is not assigned an
  ordinary integer covalent bond order here;
- a **hydrogen bond** is a directional non-covalent interaction involving a hydrogen donor and an
  acceptor. It is useful source evidence, but it is not part of WISDOM's covalent topology.

`ConnectionType` and `BondType` are enums: closed tables that store these meanings as compact,
validated integer categories rather than error-prone free text. In mmCIF, the source column
`_struct_conn.pdbx_value_order` uses `sing`, `doub`, `trip`, and `arom` to mean single, double,
triple, and aromatic bond order. WISDOM translates each code to its enum value. Hydrogen-bond
records remain in the normalized source provenance but are deliberately excluded when Section 4.4
constructs the covalent edges.

### 4.4. From normalized atoms to the atomic graph

**Turning the hierarchy into atomic arrays.**

The hierarchy built earlier in Section 4.3 is natural for reasoning about molecular ownership, but numerical
libraries work efficiently with rectangular arrays. `AtomicStructureBuilder` therefore traverses
the hierarchy once and writes one row per atom. `N` continues to mean the total retained atom count;
shape `[N,3]` means `N` rows with three coordinate values, while `[N]` means one value per atom.

Each atom already has its unique consecutive index. During traversal, the builder additionally
assigns consecutive residue and chain indices so every row can be traced back to its owner without
duplicating atom objects inside parent classes.

The twenty standard amino acids receive canonical residue IDs `1..20` in the fixed order listed by
`AMINO_ACIDS`; `0` means unknown, ligand, or otherwise non-canonical. These IDs are categories, not
measured quantities. Coarse atom roles are assigned with this precedence:

1. hydrogen;
2. metal;
3. water;
4. non-polymer;
5. backbone atom name (`N`, `CA`, `C`, `O`, `OXT`);
6. side chain.

An atom's **atomic number** is the proton count that identifies its element. **Formal charge** is the
integer bookkeeping charge written by the source. Van der Waals radius approximates the space
occupied in non-bonded contact; covalent radius approximates size when judging bonded separation.
Gemmi supplies both radius tables. They are pragmatic geometric inputs, not learned features and not
a claim that an atom has one exact quantum-mechanical boundary.

The table below uses NumPy storage names. `float32` stores a real-valued measurement in 32 bits;
`int8`, `int16`, and `int32` store signed integers of increasing range; `uint8` stores only
nonnegative integers from 0 to 255; and fixed Unicode stores short text without arbitrary Python
objects. These choices keep one protein compact while retaining the required value ranges.

| Array | Shape | Dtype | Meaning |
|---|---:|---|---|
| `atom_positions` | `[N,3]` | `float32` | Centered or original Cartesian coordinates in Å. |
| `atomic_numbers` | `[N]` | `uint8` | Element atomic numbers. |
| `residue_type_ids` | `[N]` | `uint8` | Canonical residue category; zero is unknown. |
| `atom_role_ids` | `[N]` | `uint8` | Coarse `AtomRole` category. |
| `residue_indices` | `[N]` | `int32` | Global consecutive residue ownership. |
| `chain_indices` | `[N]` | `int16` | Consecutive retained-chain ownership. |
| `formal_charges` | `[N]` | `int8` | Input formal charge. |
| `vdw_radii` | `[N]` | `float32` | Gemmi van der Waals radii in Å. |
| `covalent_radii` | `[N]` | `float32` | Gemmi covalent radii in Å. |
| `atom_names` | `[N]` | fixed Unicode | Audit names without `dtype=object`. |
| `residue_names` | `[N]` | fixed Unicode | Audit residue names without duplication in domain objects. |

**One graph with two relations.**

The atom rows just constructed become graph nodes. Let `E_spatial` be the set of unordered atom pairs
that are close in three-dimensional space, and let `E_covalent` be the set joined by reconstructed
chemical bonds. The persisted edge set `E_atom` contains the union of both sets:

```math
E_{atom} = E_{spatial} \cup E_{covalent}.
```

The union symbol means “present in either set or in both.” The graph is undirected, so `(i,j)` and
`(j,i)` mean the same relationship; WISDOM stores only the version with `i<j`. A covalent pair is not
discarded merely because unusual coordinates place it beyond the spatial cutoff.

One pair can carry both meanings. `atom_edge_relation_mask` is a compact bit mask: `1` means only
spatial, `2` only covalent, and `3=1+2` means both. This avoids duplicating the same pair in two graph
files while preserving why it exists.

**Spatial edges.**

Let `r_a` denote the configured `atom_radius`, and let `x_i` and `x_j` be the coordinates of atoms
`i` and `j`. A spatial edge exists exactly when the two indices are distinct and ordered (`i<j`) and
their Euclidean distance from Section 4.3 does not exceed `r_a`:

```math
(i,j)\in E_{spatial}
\iff i<j \text{ and } \lVert\mathbf{x}_i-\mathbf{x}_j\rVert_2\le r_a.
```

In ordinary language, each atom connects to every other atom inside a sphere of radius `r_a` around
it. The KD-tree enumerates these pairs without an `N x N` distance table. Distances are computed with
`float64` working precision and stored as `float32`, which halves storage relative to 64-bit values
while retaining more precision than the input structures normally justify.

**Covalent edges and evidence precedence.**

Many PDB files do not enumerate every ordinary covalent bond. WISDOM therefore combines direct
source declarations with conservative chemistry rules. A dictionary keyed by the ordered atom pair
prevents duplicates. When several rules suggest the same pair, **precedence** decides which evidence
label and bond type survive:

1. **Explicit records** are connections written in the source file and explained in Section 4.3.
   Covalent, disulfide, and metal-coordinate records receive confidence `1.00` and may replace an
   inferred value because the depositor supplied them directly.
2. **Canonical templates** are fixed lists of expected atom-name pairs in the backbone and side
   chain of each of the twenty standard amino acids; confidence `0.98`.
3. A **peptide bond** connects the carbonyl carbon named `C` of one polymer residue to the nitrogen
   named `N` of the next residue in the same chain. It is accepted only when `d(C,N)<=1.9 Å`, so
   sequence adjacency alone cannot bridge a large coordinate break; confidence `0.99`.
4. A possible **disulfide** joins the sulfur atoms named `SG` in two cysteine (`CYS`) residues when
   their separation is at most `2.3 Å`; confidence `0.95`.
5. **Conservative non-canonical fallback** — only within one residue absent from the standard
   templates. Let `r_i^cov` and `r_j^cov` be the covalent radii of candidate atoms and let `d_ij` be
   their Euclidean distance. A broad KD-tree search first finds pairs within 2.3 times the largest
   covalent radius in that residue. A pair is then accepted only when

   ```math
   0.4\ \text{Å} \le d_{ij} \le 1.15(r_i^{cov}+r_j^{cov}).
   ```

   The lower bound rejects coincident or severely clashing coordinates. The upper bound allows a
   15% tolerance beyond the sum of both covalent radii. These inferred single bonds have confidence
   `0.55` because geometry alone is weaker evidence than a source record or known template.

The confidence values encode deterministic source priority; they are heuristics, not calibrated
experimental probabilities. Numeric bond orders are single `1`, double `2`, triple `3`, aromatic
`1.5`, peptide `1`, disulfide `1`, and zero when an order does not apply (including coordination).

Every atomic edge also records whether both endpoints belong to the same residue, whether they
belong to the same chain, and their residue-index separation. Within one chain, separation zero
means one residue, one means adjacent residues, and so on. For atoms in different chains that number
has no meaningful sequence interpretation, so the largest `int16` value is stored as a **sentinel**,
a reserved value meaning “not applicable.” These fields give future models topology without asking
them to recover ownership from atom names.

### 4.5. From atomic spheres to surface geometry

**The surface represented by WISDOM.**

Atoms alone describe molecular matter, but many interactions occur at the boundary exposed to the
surrounding solvent. WISDOM approximates that boundary with a **point cloud**, a finite set of
positions rather than a triangle mesh.

Imagine moving the center of a spherical water-sized probe around the molecule without allowing the
probe to enter any atom. For atom `i`, let `c_i` be its center, `r_i` its van der Waals radius, and
`r_p` the configured probe radius. The probe center must remain at least `r_i+r_p` from `c_i`, so
define the expanded radius

```math
R_i = r_i + r_p.
```

The solid expanded sphere of atom `i` contains every point no farther than `R_i` from `c_i`. The
symbol `Omega` denotes the union—the region belonging to at least one—of all expanded spheres:

```math
\Omega = \bigcup_i \{\mathbf{x}:\lVert\mathbf{x}-\mathbf{c}_i\rVert_2\le R_i\}.
```

The braces describe one solid sphere; `bigcup_i` joins them for every atom. WISDOM samples the
boundary of `Omega`. This is a discrete approximation to the **solvent-accessible surface (SAS)**,
the path accessible to the *center* of the probe. It differs from the **solvent-excluded surface
(SES)**, the contact and inward-curving boundary touched by the probe itself. WISDOM does not
construct the SES's re-entrant spherical and toroidal patches. Section 4.9 returns to this
scientific limitation.

**Signed sphere gaps.**

To decide whether a point lies outside or inside an expanded sphere, WISDOM measures a signed gap.
For an arbitrary point `x`, `||x-c_i||_2` is its distance to atom center `i`, and `R_i` is the
expanded radius defined just above. Their difference is

```math
g_i(\mathbf{x}) = \lVert\mathbf{x}-\mathbf{c}_i\rVert_2 - R_i,
\qquad
g(\mathbf{x}) = \min_i g_i(\mathbf{x}).
```

For one sphere, `g_i(x)=0` means exactly on its boundary, a positive value is the remaining outside
distance, and a negative value means penetration inside the sphere. Taking `g(x)=min_i g_i(x)` asks
for the smallest gap among all atoms. Therefore `g(x)<=0` precisely when at least one sphere contains
`x`, matching membership of the union `Omega`.

This is an **implicit field** because its zero value defines a boundary without listing triangles.
Outside the union it is the exact distance to the nearest expanded sphere. Inside overlapping
spheres, however, the most negative sphere gap is not always the shortest route back to the union
boundary. WISDOM therefore does not claim that this is an exact signed-distance field (SDF)
everywhere. It evaluates gaps only where needed, stores no three-dimensional SDF grid, and does not
use marching cubes to extract a mesh.

**Fibonacci candidate points.**

The boundary now exists mathematically, but WISDOM still needs finitely many nodes for a surface
graph. It first places candidate directions around every expanded sphere using a spherical Fibonacci
pattern. This deterministic construction spreads points approximately uniformly without random
sampling, so identical input produces identical output.

Let `h` denote `surface_resolution`, the target spacing in ångströms. Let `n_i` be the number of raw
candidates assigned to expanded sphere `i`. Its surface area is `4*pi*R_i^2`; dividing by the target
area budget `0.55*h^2` estimates the required count:

```math
n_i = \max\left(24,
      \left\lceil\frac{4\pi R_i^2}{0.55h^2}\right\rceil\right)
```

The ceiling rounds a fractional count upward, and `max(24,...)` ensures that even a small sphere
receives at least 24 directions. For candidate index `k=0,...,n_i-1`, WISDOM first chooses vertical
coordinate `z_k`. The half step `k+1/2` avoids placing a point exactly at either pole. It then obtains
the horizontal radius `rho_k` needed to remain on the unit sphere:

```math
z_k = 1 - \frac{2(k+1/2)}{n_i},
\qquad
\rho_k = \sqrt{\max(0,1-z_k^2)},
```

Next `gamma=pi(3-sqrt(5))` is the golden angle in radians. Advancing longitude by this irrational
fraction of a full turn avoids repeatedly aligning points on the same meridians. `phi_i` is a fixed
atom-specific rotation, so neighboring atoms do not begin with identical patterns:

```math
\gamma = \pi(3-\sqrt{5}),
\qquad
\phi_i = 2\pi\operatorname{frac}(0.7548776662466927i),
\qquad
\theta_k = k\gamma + \phi_i.
```

Here `frac(a)` means the fractional part of `a`. Finally, `u_k` is the unit direction assembled from
horizontal radius, angle `theta_k`, and height `z_k`. Scaling it by `R_i` and adding center `c_i`
moves it onto atom `i`'s expanded sphere:

```math
\mathbf{u}_k = (\rho_k\cos\theta_k,\rho_k\sin\theta_k,z_k),
\qquad
\mathbf{p}_{ik}=\mathbf{c}_i+R_i\mathbf{u}_k.
```

At this stage candidates still include points hidden inside neighboring expanded spheres. Section
The next two steps remove those points and then reduce sampling density.

**Removing buried candidates.**

Candidate `p_ik` is the `k`th point sampled on atom `i`, so atom `i` is its **owner**. A point on the
owner sphere is part of the union boundary only if no other expanded sphere covers it from the
solvent. For every other atom `j`, WISDOM tests

```math
\exists j\ne i:
\lVert\mathbf{p}_{ik}-\mathbf{c}_j\rVert_2 < R_j-\tau,
\qquad
\tau=\max(10^{-5},0.02h).
```

The left side is the candidate-to-center distance. The right side is sphere `j`'s radius reduced by
the small tolerance `tau`. Thus a candidate is removed only when it lies clearly inside sphere `j`,
not when floating-point roundoff places two boundaries almost together. `tau` is the larger of
`10^-5 Å` and 2% of target spacing `h`. Using the signed gap defined above, the same decision is
`g_j(p_ik)<-tau`.

The KD-tree is only an acceleration structure: it returns centers within the largest possible
expanded radius plus `0.05h`; a center beyond that distance cannot contain this candidate. Exact
Euclidean distances still decide removal. Surviving points are called **exposed candidates** because
the probe center can reach them. If none survive, there is no valid surface to pass to later stages,
so preprocessing fails instead of publishing an empty and misleading representation.

**Deterministic voxel reduction.**

Fibonacci candidates from different atoms can cluster where spheres meet. Keeping all of them would
make surface density depend strongly on overlap and would enlarge the graph. WISDOM therefore lays
an imaginary regular grid over the point cloud and retains at most one candidate per grid cell. A
grid cell is also called a **voxel**, the three-dimensional analogue of a pixel.

Let `o` be a three-component origin formed from the smallest exposed x, y, and z coordinates. It is
not the molecular centroid from Section 4.3; it merely anchors the voxel grid. For exposed point
`p`, subtracting `o`, dividing each component by voxel side `h`, and applying floor produces integer
cell coordinate `q(p)`:

```math
\mathbf{q}(\mathbf{p}) =
\left\lfloor\frac{\mathbf{p}-\mathbf{o}}{h}\right\rfloor.
```

The floor operation rounds each component downward, so every point in the same `h x h x h` cube gets
the same integer triple. Cells are sorted first by x index, then y, then z. Within one occupied cell,
the target center is `o+(q+1/2)h`; WISDOM selects the original candidate closest to that target, with
original candidate order resolving an exact tie. It never moves the selected point, so that point
remains exactly on an expanded sphere.

The result is reproducible and has at most one point per voxel. It is a density-control rule, not a
guarantee that every pair of retained points is at least `h` apart; adjacent voxels may hold points
near their shared face. In particular, it is not an optimal Poisson-disk or blue-noise sample.

**Outward surface normals.**

A **surface normal** is a unit vector perpendicular to the local surface. Its orientation matters:
WISDOM needs it to point from the molecule toward solvent, called the outward direction. At a smooth
point belonging to one sphere, that direction is simply the normalized vector from atom center to
surface point. Near overlapping sphere boundaries, abruptly choosing one owner would make normals
jump, so nearby sphere directions are blended.

For retained point `p`, compute the signed gaps `g_j(p)` defined earlier in this section. Let `g_min` be
the smallest nearby gap. Let `sigma` be a smoothing length: 25% of resolution `h`, but never below
`10^-3 Å` to avoid unstable division by an almost-zero value:

```math
\sigma=\max(0.25h,10^{-3}),
\qquad
g_{min}=\min_j g_j(\mathbf{p}),
```

Only atoms with `g_j<=g_min+2.5*sigma` are active; spheres much farther from the local envelope
cannot influence its orientation. For active atom `j`, `w_j` decreases exponentially as its gap
moves above `g_min`. The gradient `nabla g_j` is its outward radial unit direction:

```math
w_j=\exp\left(-\frac{g_j-g_{min}}{\sigma}\right),
\qquad
\nabla g_j(\mathbf{p})=
\frac{\mathbf{p}-\mathbf{c}_j}{\lVert\mathbf{p}-\mathbf{c}_j\rVert_2}.
```

The weighted directions are added and the sum is divided by its Euclidean length, producing one unit
normal:

```math
\mathbf{n}(\mathbf{p})=
\frac{\sum_j w_j\nabla g_j(\mathbf{p})}
     {\left\lVert\sum_j w_j\nabla g_j(\mathbf{p})\right\rVert_2}.
```

This soft-min-like blend varies more smoothly at sphere intersections than selecting one gradient.
If opposite contributions nearly cancel to a zero vector, normalization would be undefined, so the
owner-sphere radial direction provides a deterministic fallback. A KD-tree considers only atoms
within the largest expanded radius plus `h`; as in buried-candidate removal above, this limits work rather than
changing the displayed weighting rule.

`estimate_normals` is a separate utility for synthetic and test point clouds that have no owner
spheres. It uses neighbors within `3h` or up to eight nearest points, subtracts their mean, and forms
the scatter matrix `X^T X`. Principal-component analysis (PCA) finds directions of local variation;
the eigenvector with the smallest eigenvalue is the direction in which the neighborhood varies
least, so it approximates the perpendicular. Eigenvectors have arbitrary sign, therefore an outward
reference or deterministic sign rule orients them. Molecular `build` uses the signed-gap blend above,
not these PCA normals.

**Curvature at configurable scales.**

The normal says which way the surface faces; **curvature** describes how it bends. A plane has zero
curvature. A small sphere bends more sharply than a large sphere. A saddle bends in opposite
directions along two perpendicular axes. Because a sampled and possibly noisy surface has no single
perfect neighborhood size, WISDOM estimates curvature independently at configured radii. If
`q_j` is entry `j` of `curvature_scales`, its physical neighborhood radius is

```math
r_j=q_jh.
```

Here `h` is the surface resolution introduced above. The default `q=(2.5,5.0)` therefore uses radii
`2.5h` and `5h`: the first captures more local bending and the second averages over a wider region.
The YAML may add, remove, or reorder positive unique multipliers. If a radius contains fewer than
seven points, up to twelve nearest points are used so the six-parameter fit below is not
underdetermined.

At retained point `p` with unit normal `n`, WISDOM constructs two unit vectors `t_1` and `t_2` that
are perpendicular to `n` and to each other. They span the local **tangent plane**, the plane that
best represents a tiny flat patch around `p`. For neighbor `x`, define displacement `delta=x-p`.
Dot products project that displacement onto the two tangent directions and the normal direction:

```math
u'=\frac{\delta\cdot\mathbf{t}_1}{r_j},
\qquad
v'=\frac{\delta\cdot\mathbf{t}_2}{r_j},
\qquad
z'=\frac{\delta\cdot\mathbf{n}}{r_j}.
```

Thus `(u',v')` locates the neighbor across the tangent plane and `z'` measures height above or below
that plane, all without units. Dividing by the fitting radius keeps columns numerically comparable
for every configured scale. WISDOM approximates dimensionless height with a quadratic **Monge
patch**:

```math
z'(u',v') \approx
\frac{1}{2}a{u'}^2+bu'v'+\frac{1}{2}c{v'}^2+du'+ev'+f
```

The coefficients `a`, `b`, and `c` describe second-order bending. Coefficients `d` and `e` allow a
small residual tilt, and `f` allows a vertical offset. There are six unknown coefficients, which are
chosen by weighted least squares: the sum of squared prediction errors is minimized. A neighbor with
displacement `delta` receives Gaussian weight

```math
w(\delta)=\exp\left(-\frac{\lVert\delta\rVert_2^2}{r_j^2}\right).
```

so nearby samples influence the fit more strongly, while a point exactly one radius away receives
weight `exp(-1)`. In matrix notation, `A` contains the six polynomial terms for all neighbors,
`beta=(a,b,c,d,e,f)` contains the unknown coefficients, `z` contains observed heights, and diagonal
matrix `W` contains the weights. Solving `sqrt(W) A beta = sqrt(W) z` is the standard weighted
least-squares transformation. The solver discards singular directions smaller than `0.05` times the
largest singular value. This five-percent cutoff prevents sparse, nearly collinear samples from
amplifying numerical noise into curvature spikes.

For a sufficiently small, nearly tangent patch, the second-order coefficients approximate the local
shape operator. WISDOM uses the outward-oriented 2-by-2 matrix

```math
S \approx -\frac{1}{r_j}\begin{bmatrix}a&b\\b&c\end{bmatrix}.
```

The minus sign follows the chosen height/normal convention and makes an outward-oriented convex
sphere positive. The two eigenvalues `k_1` and `k_2` of `S` are the **principal curvatures**: the
maximum and minimum normal bending along perpendicular tangent directions. WISDOM derives three
summary channels:

```math
H=\frac{k_1+k_2}{2},
\qquad
K=k_1k_2,
\qquad
C=\sqrt{\frac{k_1^2+k_2^2}{2}}.
```

`H` is mean curvature and preserves the overall signed bending orientation. `K` is Gaussian
curvature: it is positive when both principal directions bend with the same sign, negative for a
saddle, and zero when at least one direction is locally flat. `C` is curvedness, a nonnegative
magnitude that is large whenever either principal curvature is large.

Since curvature is inverse length, `H` and `C` have units Å⁻¹; multiplying two curvatures gives `K`
units Å⁻². If `M` is the retained point count and `S` is the configured scale count,
`surface_curvatures` has shape `[M,S,3]`: point, scale, and `(H,K,C)` channel. With defaults, `S=2`
at radii `(2.5h,5h)`. A non-finite numerical fit is
replaced by zero so corrupt numeric values cannot enter storage, but this fallback does not turn a
sparse neighborhood into an exact measurement. The limitations in Section 4.9 should therefore be
considered when interpreting curvature.

**Area weights.**

A uniformly sampled surface gives every point roughly equal represented area, but voxel selection
can leave local density variation. To compensate, WISDOM queries each point plus up to six nearest
neighbors. Let `ell_i` be the median distance from point `i` to those non-self neighbors. Squaring a
length gives area units, so the unnormalized proxy is `ell_i^2`.

Let `epsilon_float32` be the smallest positive safety value used here to prevent a zero weight when
coincident points occur. Define raw proxy `A_i^*` and normalized weight `w_i` as

```math
A_i^*=\max(\ell_i^2,\epsilon_{float32}),
\qquad
w_i=\frac{A_i^*}{\sum_j A_j^*}.
```

The denominator adds the raw proxies of all `M` surface points, so normalized weights are finite,
positive, dimensionless, and sum to one. A sparse region has larger neighbor spacing and therefore a
larger weight. Future weighted pooling can use these values so dense regions do not dominate merely
because they contain more samples. The weights are not exact Voronoi cell areas or physical
solvent-accessible surface area (SASA) in Å²; normalization deliberately removes absolute area. A
one-point surface receives weight one.

### 4.6. From surface points to the final NPZ

**The surface graph.**

The retained points and their geometry are now available, but the surface graph described in Section
3.1.3 still needs edges. Connecting every pair would be dense and would join unrelated sides of the
molecule. WISDOM first asks the KD-tree for pairs whose Euclidean separation `d_ij` is at most
`2.5h`, where `h` is surface resolution. It then applies two orientation filters.

Let `n_i` and `n_j` be the outward normals at candidate points `i` and `j`. Their dot product is `1`
when they point the same way, `0` when perpendicular, and `-1` when opposite. The pair is rejected
when

```math
\mathbf{n}_i\cdot\mathbf{n}_j < -0.25,
```

so strongly opposed surface sheets do not connect merely because they are close across a narrow
gap. For the second filter, let `Delta_ij=p_j-p_i` be the displacement from point `i` to point `j`.
The dot products with each normal measure how much of that displacement travels through the surface
rather than along it. Reject when

```math
\max(|\Delta_{ij}\cdot\mathbf{n}_i|,
     |\Delta_{ij}\cdot\mathbf{n}_j|)
>0.8\lVert\Delta_{ij}\rVert_2,
\qquad
\Delta_{ij}=\mathbf{p}_j-\mathbf{p}_i.
```

The right side is 80% of total displacement length. Therefore an edge survives only when its
normal-direction component is not too dominant from either endpoint. Together the rules favor local
tangent travel and reduce shortcuts between nearby opposite walls. Surviving undirected pairs are
stored once with `src<dst`, as in the atomic graph.

A **connected component** is a maximal set of nodes reachable from one another by following graph
edges. WISDOM computes components on a sparse symmetric adjacency matrix: “symmetric” represents both
directions of each undirected edge, while COO is simply a memory-efficient list of nonzero row and
column coordinates.

Let `M` be the surface-point count. A warning is emitted when the number of components exceeds the
larger of 3 and `floor(M/100)`, or when components with fewer than five points exceed the larger of 2
and half the component count. These warnings highlight unusual fragmentation. Component IDs do not
claim that one component is an exterior surface, pocket, channel, or sealed cavity.

The word “component” describes the **graph**, not whether the corresponding atoms form one protein.
Two points in different components really do have no path made of accepted surface edges between
them. That can be physically correct: two selected chains may not touch, and the wall of a sealed
internal cavity is disconnected from the exterior boundary. It can also be a discretization
artifact: voxel sampling may leave a small gap, or the normal filters may reject the only candidate
edge near a sharp crease. A one-point component is an isolated point and receives no neighbor message
from the surface GCN.

Multiple components are therefore permitted, but they are not declared harmless. The numerical
validator separately proves that every point lies on the molecular envelope and has an outward
normal; it then reports component count, isolated points and largest-component fraction as scientific
warnings. In the currently validated 450-protein corpus, 4,280 of 5,389,038 points are isolated
(`0.079%`) and the worst largest-component fraction is about `0.572`. This is geometrically valid but
must be inspected before drawing localization conclusions. WISDOMv1/v2 cannot propagate a surface
GCN message across components. They can nevertheless classify the whole protein because every point
still receives nearby atomic information and final MIL pooling combines points from all components.
If a task requires long-range communication across disconnected sheets, changing the model or graph
criterion is a new scientific decision, not something validation should silently invent.

**The surface-to-atom graph.**

The surface graph moves information across the boundary, but a future model also needs the third
graph introduced in Section 4.1 to exchange information with atoms. Let `p_s` be surface point `s`, let
`x_i` be atom `i`, and let `r_sa` denote configured `atom_surface_radius`. A bipartite edge exists
when

```math
(s,i)\in E_{surface-atom}
\iff \lVert\mathbf{p}_s-\mathbf{x}_i\rVert_2\le r_{sa},
```

In words, every surface point connects to **all** atoms within `r_sa`; the method is a radius graph,
not a fixed-number nearest-neighbor graph. Neighbors are deterministically sorted. Each stored column
contains `[surface_index, atom_index]` plus their Euclidean distance.

There is intentionally no K-nearest-neighbor (KNN) fallback that would connect a point to a distant
atom just to fill a quota. If a surface point has no atom within the scientific cutoff, its
representation is inconsistent and preprocessing fails. These edges specify only where future
communication is permitted; they contain no message, attention coefficient, or learned weight.

**NPZ output schema.**

All three representations now meet in one NPZ. Arrays are separated by role so consumers can load
only what they need. In the table, `N` is atom count, `M` is surface-point count, and edge arrays use
one column per stored pair. A **dtype** is the numeric storage type: for example, `float32` is a
32-bit real number and `int32` a 32-bit signed integer.

| Group | Arrays | Semantics |
|---|---|---|
| Atoms | `atom_positions`, `atomic_numbers`, `residue_type_ids`, `atom_role_ids`, `residue_indices`, `chain_indices`, `formal_charges`, `vdw_radii`, `covalent_radii` | Compact structural atom features. |
| Audit labels | `atom_names`, `residue_names` | Fixed-width Unicode labels. |
| Atomic topology | `atom_edge_index`, `atom_edge_distance`, `atom_edge_relation_mask` | Spatial/covalent pair union. |
| Bond semantics | `atom_edge_bond_type`, `atom_edge_bond_order`, `atom_edge_bond_source`, `atom_edge_bond_confidence` | Chemical type, numeric order, evidence, and heuristic confidence. |
| Atomic context | `atom_edge_same_residue`, `atom_edge_same_chain`, `atom_edge_residue_separation` | Ownership/topological context. |
| Surface | `surface_positions`, `surface_normals`, `surface_curvatures`, `surface_area_weights`, `surface_component_ids` | Fixed point cloud and local geometry. |
| Surface topology | `surface_edge_index`, `surface_edge_distance` | Filtered local undirected graph. |
| Atom communication | `surface_atom_edge_index`, `surface_atom_distance` | Radius-based bipartite incidence. |
| Provenance | `metadata_json` | Scalar JSON Unicode array, never pickle/object. |

Graph indices are `int32`; categorical IDs and flags use compact integer dtypes; distances and
persisted geometry are `float32`. The NPZ intentionally excludes dense adjacency, one-hot features,
RBF expansions, relative vectors, embeddings, messages, patches, and model-specific labels.

Those exclusions keep preprocessing model-independent. A dense adjacency is a full node-by-node
table including mostly absent edges; one-hot encoding expands a category into many zero/one columns;
an RBF (radial basis function) expansion converts a distance into several smooth response channels;
and embeddings/messages are learned neural-network states. A training pipeline may derive such
values from the fixed structure, but storing them here would bind the dataset to one model design.

`metadata_json` is the provenance record introduced in Section 4.2. It includes source
identifier/path/hash/format, selected chains and model, coordinate origin, representation counts,
schema/project versions, effective scientific configuration and hash, dependency versions, and
warnings. LambdaForge records code identity and the complete execution environment once at run
level, so WISDOM does not duplicate Git commit probing inside every protein NPZ. Keeping provenance
separate from the `Protein -> Chain -> Residue -> Atom` hierarchy prevents audit data from being
mistaken for molecular structure.

### 4.7. Validation, reproducibility, and parallel execution

**Validation and atomic publication.**

Producing arrays is not enough: every stage assumes shapes and indices established by the preceding
stage. `StorageManager` therefore checks the complete representation before any final filename is
published. This turns silent corruption into a per-protein failure with a reportable reason.

Before publication, `StorageManager` verifies:

- nonempty finite `[N,3]` atomic coordinates and matching feature lengths;
- valid atomic numbers and nonnegative residue indices;
- `int32` graph indices, in-range endpoints, `src<dst`, no duplicates, and consistent distances;
- relation masks in `{1,2,3}` and one value per atomic edge for every edge feature;
- nonempty finite `[M,3]` surface positions and unit normals;
- finite `[M,S,3]` curvatures, with `S` equal to the configured number of scales;
- positive finite area weights summing to one;
- nonnegative component IDs;
- valid surface graph distances/endpoints;
- valid bipartite endpoints/distances and at least one atom for every surface point;
- absence of `dtype=object`.

NPZ publication is **transactional**, meaning that the final path changes only after the whole new
file is valid. WISDOM writes a uniquely named temporary file, flushes and synchronizes it, reopens it
with `allow_pickle=False`, and revalidates the exact stored arrays and metadata JSON. Disabling pickle
prevents NPZ loading from executing serialized Python objects. `os.replace` then publishes the file
atomically. A failed worker cannot leave an apparently valid final NPZ.

**Run and record reuse.** A matching interrupted run resumes from LambdaForge record checkpoints,
not from filenames alone. WISDOM additionally revalidates molecular records at sink boundaries.
Inspect the resolved call without starting work with

```bash
lf inspect experiments/dna_preprocess.yaml --resolved
```

Changing declared selection bytes, code identity, or a scientific setting creates a different Work
identity. A successfully published `name@version` remains immutable, so intended new content needs a
new explicit dataset version rather than overwriting an old placement.

**Per-protein resume.** LambdaForge checkpoints the stable key, status, attempts, duration, and error
of each record in `preprocessing-manifest.json`. After interruption, it offers completed keys to the
sink rather than blindly trusting their filenames. `ProteinSink.is_complete` then opens the exact
candidate with `allow_pickle=False`, requires the complete array set, reruns all numerical checks
listed above, recomputes the current coordinate-file hash, and also requires equality of:

```text
source_hash
config_hash
preprocessing_schema_version
```

`source_hash` identifies exact coordinate bytes; `config_hash` identifies settings that can change
scientific arrays; and `preprocessing_schema_version` identifies how arrays are named and interpreted.
The configuration hash includes model/chain/filter/centering and graph/surface settings, but excludes
operational paths, worker counts, download policy, and failure policy because those do not alter
array values. A record is resumed only when both layers agree: LambdaForge has matching progress and
the WISDOM sink proves the current NPZ scientifically reusable. `--force` makes another attempt after
success, `--no-resume` ignores partial progress, and `--restart` starts the attempt from scratch.
Ordinary recovery needs none of these flags. Final reports are restored to manifest order.

**Scientific validation at publication.** LambdaForge's artifact digest proves that current bytes
equal the bytes recorded by the task, while WISDOM's preprocessing sink checks the domain meaning
before publishing every NPZ. It opens the archive without pickle, validates the complete schema,
recomputes counts and distances, verifies metadata/configuration/source hashes, and compares report
values with loaded arrays. Scientific surface-fragmentation warnings remain visible without becoming
schema errors. Section 4.2 shows LambdaForge's generic commands for inspecting arrays and explicit
3D roles after publication; no second preprocessing experiment YAML is required.

Surface validation goes beyond array bookkeeping. For surface point `p`, atom centre `c_i`, van der
Waals radius `r_i`, and probe radius `r_probe`, it recomputes the signed expanded-sphere gap

```math
g(p)=\min_i\left(\lVert p-c_i\rVert-r_i-r_{probe}\right).
```

The norm is ordinary Euclidean distance in ångströms. An exposed sampled boundary should have
`g(p)` close to zero: a substantially negative value places the point inside an expanded atom, while
a positive value above tolerance makes it a detached or “flying” point. The accepted tolerance is
`max(0.0005 Å, 0.025h)`. The validator also reconstructs the outward direction from the smooth
soft-min envelope and requires its cosine with the stored normal to be at least `0.99`. It verifies
the curvature identity `C²=2H²-K`, bounds the dimensionless magnitude `C r` at every fitted radius,
and reports isolated points, connected components, the largest-component fraction, and longest
surface edge. These quantities appear both per protein and as readable dataset extrema.

Numerical tests can reveal errors invisible in a table, while LambdaForge's explicit-role 3D artifact
viewer can reveal spatial patterns hidden by one summary number. It can render atomic positions,
surface points, graph edges, and normal vectors directly from an NPZ as shown in Section 4.2. The
viewer deliberately does not invent a triangle mesh: preprocessing stores a point cloud, and naïvely
joining nearby points could draw false sheets across cavities.

**Parallelism, failures, and managed execution.**

Proteins are independent records, so LambdaForge may transform several at once. `workers: 1` is the
sequential reference behavior. With `workload: cpu`, the configured `workers` count bounds a spawned
process pool that runs only the WISDOM transform; the parent process serializes sink writes,
checkpoints, the manifest, and dataset publication. `workload: io` uses bounded threads,
`workload: auto` chooses conservative threads, and `workload: gpu` permits one worker so GPU
parallelism remains an explicit job/resource decision. These execution fields are excluded from the
dataset fingerprint: sequential and parallel executions must yield identical scientific content.

The selection Task requests 36 CPUs but uses 72 bounded I/O threads: two candidates per CPU can wait
on independent network responses without oversubscribing numerical work. A single thread-safe
limiter caps request starts at four per second across all those threads. RCSB recommends beginning
with only a handful of API requests per second and backing off on HTTP 429, so adding more threads
would not raise the safe request rate. Inspecting tens of thousands of candidates can still require
hours because remote-service latency and rate limits—not CPU computation—are the lower bound.

The heavy recipe instead uses 36 spawned processes, one per requested CPU, for both geometry and
annotation. Geometry downloads different selected PDB entries concurrently and performs the costly
surface calculations; annotation consumes the already downloaded coordinate artifact. Therefore it
does not repeat the public structure downloads.

In LambdaForge 0.11, the compact Work resource block determines the enclosing reservation:

```bash
lf run experiments/dna_preprocess.yaml --on citius-ctgpgpu12
```

The resolved Work reports `cpu: 36`, 128 GiB, and 24 hours. Inside its one coordinator, each
`PreprocessingTask` creates the configured 36-process pool. Geometry and annotation are sequential,
so both reuse the same 36-core reservation instead of requiring 72 cores. Do not use 72 CPU-bound
processes with a 36-CPU allocation; oversubscription normally increases context switching and memory
pressure rather than throughput. The selection Task's 72 workers are threads and are appropriate
only because that separate workload spends most of its time waiting for I/O.

`on_error: skip` would record a failed key and continue; `on_error: fail` records it, cancels pending
work, and stops. Selection, geometry, and annotation deliberately select `fail`. Scientific
candidate rejections are ordinary selection rows and do not crash the Task, but an unexpected error
or failure to produce geometry/annotation for a selected balanced member blocks publication.
Completed records remain checkpointed for resume.

NumPy and SciPy may themselves start native math threads. If every Python process started another
full thread pool, the machine could run far more active threads than allocated CPUs, a condition
called **oversubscription**. WISDOM therefore sets `OMP_NUM_THREADS`, `MKL_NUM_THREADS`,
`OPENBLAS_NUM_THREADS`, and `NUMEXPR_NUM_THREADS` to one before importing numerical work in workers.

For LambdaForge 0.11 Works, the top-level `resources` block requests CPU, RAM, GPU, and time
portably. After a cluster profile such as `atlas` is
registered, the managed runner stages the build and later exposes results and artifacts without a
WISDOM-specific SLURM script:

```bash
lf run experiments/dna_preprocess.yaml --on atlas
lf jobs show latest
lf datasets show wisdom-dna@2
```

With LambdaForge 0.11, each `environment: managed` cluster profile should normally leave PyTorch
selection automatic:

```yaml
pytorch:
  channel: auto
  require_cuda: auto
```

Before creating or reusing the remote environment, LambdaForge inspects the cluster's configured
Python, machine architecture, NVIDIA driver, and visible GPU compute capability. It then selects an
official PyTorch CPU or CUDA wheel that is compatible with those facts and verifies CUDA with a real
tensor operation when a GPU is required. The selected wheel becomes part of the environment
identity, so a compatibility change produces a new managed environment instead of silently reusing
an unsuitable one. This mechanism installs user-space PyTorch wheels; it never changes the NVIDIA
driver or installs the system CUDA toolkit. On a scheduler login node that cannot expose the
compute-node GPU, automatic detection cannot prove compatibility and an administrator-reviewed
explicit policy is required.

Physical dataset roots need not be identical between machines. DatasetRegistry records a separate
placement for each verified copy of `wisdom-dna@2`, while the content ID remains unchanged. Training
uses the versioned logical reference and LambdaForge selects a placement in the execution
environment. DataCatalog is unnecessary for this managed version; it remains available for aliases,
external datasets, loader definitions, and explicit institutional pins.

### 4.8. Code architecture and testing

**Code architecture.**

All runtime code now lives under one `src/wisdom` package. Three short public function modules are
the only actions a user needs to recognize: `Selection.py`, `Preprocessing.py`, and `Training.py`.
Their functions read like orchestration pseudocode; cohesive scientific details remain in the
`preprocessing/dna`, `preprocessing/structure`, `data`, `models`, and `evaluation` subpackages.
`PreprocessPipeline` still reads like the one-protein sequence in Section 4.1:

```text
LambdaForge PreprocessingTask
├── ProteinSource                TXT records and stable keys
├── PreprocessPipeline           one-protein scientific transform
│   ├── StructureCache          path resolution, download locking, hashing
│   ├── ProteinReader           Gemmi normalization and explicit connections
│   ├── AtomicStructureBuilder  compact atom features and union graph
│   └── SurfaceBuilder          surface, normals, curvature, graphs
└── ProteinSink                   scientific resume and atomic publication
    └── StorageManager              exact NPZ schema and numerical validation

Protein
└── Chain
    └── Residue
        └── Atom
```

Provenance, defined in Section 4.2, is carried separately by `ProteinMetadata`; the resolved path,
hash, format, and requested chains are carried by `StructureSource`. Closed categories use enums
(`AtomRole`, `BondType`, `BondSource`, `ConnectionType`, and `Relation`), which constrain values to a
documented set instead of accepting arbitrary strings.

**Testing.**

```bash
ruff check .
mypy src/wisdom
pytest -q
lf validate experiments/dna_select.yaml
lf validate experiments/dna_preprocess.yaml
lf validate experiments/wisdom_v1.yaml
lf validate experiments/wisdom_v2.yaml
```

Tests are offline and cover PDB/mmCIF/gzip, input grammar, model and chain errors, filters, alternate
locations, explicit bond order, templates, peptide/disulfide/aromatic chemistry, relation unions,
covalent edges outside the radius, sphere/plane/cylinder/concave curvature, surface determinism,
area weights, LambdaForge source→transform→sink integration, CPU-process equivalence, partial
failure, scientific resume invalidation, dataset-artifact identity, and bounded debug sampling.

### 4.9. Scientific limitations

These limits define what conclusions may safely be drawn from the output:

- BTD-Combo negatives are carefully filtered benchmark inferences, not experimental proof that a
  protein can never bind DNA. QuickGO and BioLiP2 can be incomplete or delayed relative to biology.
- The RCSB Data/Search APIs and BioLiP2 latest snapshot are live public services. Pinned downloaded
  bytes are reproducible and cached, but a future rebuild deliberately fails if a mutable snapshot
  changes instead of silently claiming the old dataset version.
- RCSB's 30%-identity MMseqs2 groups provide conservative homology isolation, but sequence identity
  is not a complete measure of structural or evolutionary relatedness.
- As explained in Section 4.5, the point cloud approximates SAS with expanded spheres, not an
  analytical SES. It therefore lacks the inward re-entrant patches traced by the probe surface and
  has no triangle mesh or torus construction.
- The signed gap in Section 4.5 provides reliable inside/outside sign and outside distance. It is
  neither stored as a volume nor interpreted as exact shortest interior distance where spheres
  overlap.
- Voxel reduction is deterministic density control. Blue-noise or Poisson-disk sampling additionally
  tries to distribute nearest-neighbor distances evenly; WISDOM does not solve that optimization.
- Curvature is a local small-slope quadratic approximation. Few, noisy, or uneven neighbors can bias
  it, and replacing a non-finite fit by zero is a storage safeguard rather than scientific evidence
  of flatness.
- Area weights express relative local spacing and sum to one. They are not absolute square-ångström
  solvent-accessible areas or exact Voronoi cells (regions closest to each sample).
- Surface edges approximate locality using straight-line distance and normals. They do not compute a
  **geodesic**, the shortest path constrained to remain on the surface, so difficult narrow passages
  may still gain a shortcut or lose a valid connection.
- Bond templates cover the twenty standard amino acids. The distance-based fallback for other
  residues cannot replace a complete Chemical Component Dictionary description of ligands and
  modified residues.
- Charges and explicit connections are limited by source records and Gemmi interpretation. WISDOM
  neither adds missing atoms nor selects protonation/tautomeric states—alternative placements of
  hydrogens and double bonds that depend on chemical conditions—and performs no quantum calculation.
- Bond confidence values only rank the evidence sources in Section 4.4. They are not experimentally
  calibrated probabilities that a bond truly exists.
- Only one selected coordinate model is represented. A multi-model ensemble or time-dependent
  molecular-dynamics trajectory would require an additional dimension and is not supported.

## 5. Trainable WISDOM models

WISDOMv1 answers one deliberately narrow question: can fixed internal atomic structure and fixed
surface geometry predict one binary label for a whole protein while exposing a score at every
surface point? A label `0` or `1` belongs to the protein, not to an atom or point. Consequently,
local scores are **weakly supervised**: they are learned only through the global label and must not
be interpreted as experimentally validated binding sites.

### 5.1. Dataset index and graph batching

Universal geometry itself has no experimental label. The selection/annotation flow adds those
meanings when it publishes the managed dataset. In LambdaForge 0.11, `WisdomDataset` reads the
canonical `index.jsonl`: each member supplies an explicit `split` partition, a binary
`dna_binding` target, `universal_npz` and `dna_annotation` assets, and optional dilution names such
as `25pct`. No filename is interpreted as a label and no random split is invented. The older
`file,label,split` CSV remains readable only for small tests and backwards-compatible local use.

After filtering the requested split/view, `WisdomDataset` opens each NPZ with
`allow_pickle=False`, checks the arrays and graph ranges required by the models, and converts only
those arrays to tensors. It does not move points, recompute edges, or mutate preprocessing output.

Proteins have different atom and surface counts, so a rectangular stack is impossible without
padding. `WisdomCollator` instead constructs a **disjoint union**: it concatenates nodes and shifts
each graph endpoint by the number of earlier nodes. Atomic endpoints receive atomic offsets;
surface endpoints receive surface offsets; and the two rows of the bipartite surface-to-atom graph
receive their respective offsets. `surface_batch[p]` records which protein owns surface point `p`.
The preprocessing stores every undirected edge once with `src<dst`, whereas graph convolution sends
directed messages, so the collator adds both `src→dst` and `dst→src` deterministically. Relation
masks retain their meaning while becoming zero-based R-GCN IDs:

Why is this class necessary? A normal image batch can use a tensor such as `[B,height,width]`
because every image has the same rectangular axes. A protein with 2,000 atoms cannot be stacked
directly with one containing 700 atoms, and their edge lists have different lengths as well. Padding
all atomic and surface graphs to the largest protein would waste memory and create fake nodes that
every graph operation would have to mask. LambdaForge's GNN layers instead accept one sparse graph,
so the collator makes several proteins look like one larger graph while guaranteeing that no edge
crosses from one protein to another.

Consider protein A with three atoms and two surface points, followed by protein B with two atoms and
three surface points. Their local indices both start at zero:

```text
                         protein A       protein B before batching    protein B in batch
atom indices             0, 1, 2         0, 1                         3, 4
surface indices          0, 1            0, 1, 2                      2, 3, 4
atom edge                (0, 2)          (0, 1)                       (3, 4)
surface edge             (0, 1)          (0, 2)                       (2, 4)
surface→atom edge        (1, 2)          (2, 1)                       (4, 4)
```

The atom offset for B is three and its surface offset is two. Notice that the bipartite edge needs
**different offsets for its rows**: surface `2` becomes `4`, while atom `1` becomes `4`. Applying the
same offset to both would silently connect the wrong domains. After concatenation,
`surface_batch=[0,0,1,1,1]` says that the first two surface rows belong to A and the next three to B;
`atom_batch=[0,0,0,1,1]` records the analogous atomic ownership. Targets become `[y_A,y_B]`.
WISDOMv1/v2 use `surface_batch` to reduce local predictions into exactly one protein logit. The
collator checks every shifted endpoint so an offset error fails immediately instead of mixing two
proteins during learning.

“Collation” therefore changes bookkeeping only. It does not create scientific edges, recompute
distances, alter coordinates or allow information leakage. At the end of a batch the node rows are
contiguous for efficiency, but the graphs remain mathematically disjoint.

| Stored relation mask | R-GCN ID | Meaning |
|---:|---:|---|
| `1` | `0` | spatial proximity only |
| `2` | `1` | covalent bond only |
| `3` | `2` | both spatial and covalent |

### 5.2. WISDOMv1 models, equations, and tensor shapes

`WisdomV1` is the only domain-specific neural composition in v1. It does not reimplement graph
learning: its constructor builds LambdaForge `RelationalGCN`, `MLP`, `GCN`, indexed scatter, and
sparse pooling components from independent conceptual parameters. Consequently HPO can change one
embedding width or layer count without leaving a stale nested `in_channels` value in YAML.

| Component | Implementation | Input → output | What it learns |
|---|---|---|---|
| Element embedding | `torch.nn.Embedding` | atomic number `[N]` → `[N,E]` | One learned vector per chemical element ID. |
| Optional residue embedding | `torch.nn.Embedding` | residue ID `[N]` → `[N,E]` | Tests whether amino-acid category adds useful context. |
| Atomic encoder | LambdaForge `RelationalGCN` | features `[N,E]` or `[N,2E]`, edges, relation IDs → `[N,D]` | Different message matrices for spatial-only, covalent-only and combined atomic edges. |
| Atom→surface transfer | LambdaForge `Scatter.mean` | atom embeddings and incidence edges → `[M,D]` | Mean atomic context attached to each point. |
| Surface projection | LambdaForge `MLP` | atom context plus curvature `[M,D+3S]` → `[M,D]` | Fuses the two information sources point by point. |
| Surface encoder | LambdaForge `GCN` | point features `[M,D]` and surface edges → `[M,D]` | Exchanges information with neighboring points using normalized graph convolution. |
| Local head | `torch.nn.Linear(D,1)` | surface embedding `[M,D]` → local logits `[M]` | Converts each learned point description into local class evidence. |
| Global reduction | LambdaForge `SparseMaxPooling` | local logits and `surface_batch` → `[B]` | Implements the fixed existential MAX MIL rule. |

An embedding is a trainable lookup table, not a hand-written chemical descriptor. R-GCN means
**relational graph convolutional network**: a neighbor connected by a covalent bond is transformed
differently from one connected only by spatial proximity. The subsequent MLP is a row-wise
multilayer perceptron; in this configuration it is a single learned projection. The surface GCN then
lets each point combine its state with states arriving through the surface graph. Two components of
that graph never exchange GCN messages, exactly as explained in Section 4.6.

Let `N` be the total atoms, `M` the total surface points, `B` the proteins in one batch, `E` the
embedding width, `D` the hidden width, and `S` the configured curvature scales. The residue table is
omitted entirely for the element-only HPO candidate; otherwise both embeddings are concatenated.
LambdaForge's `RelationalGCN` uses the three edge relations to produce `h_atom[N,D]`.

For surface point `p`, let `A(p)` be the atoms joined to it by the stored bipartite graph. The first
atom-to-surface transfer is intentionally only a mean:

```math
h_{A\to S}(p)=\frac{1}{|A(p)|}\sum_{a\in A(p)}h_{atom}(a).
```

Preprocessing guarantees at least one linked atom per point, and LambdaForge's indexed scatter mean
computes the expression without a dense atom-by-point matrix. Each point also has `S` triplets
`[H,K,C]`; flattening them yields `3S` invariant scalar features. WISDOM concatenates those values
with `h_A→S`, projects the resulting `[M,D+3S]` tensor through a LambdaForge `MLP`, and passes it
through a two-layer LambdaForge `GCN` on the surface graph. Absolute positions and normal-vector
components are deliberately absent, so rotating the entire input cannot change a feature merely
because a Cartesian axis changed.

A single linear layer maps each final surface embedding to one unconstrained local logit `l_p`.
“Logit” means a real number before a sigmoid: positive values favour class `1`, negative values
favour class `0`, and zero corresponds to probability `0.5`. For protein `b`, let `P_b` be its point
set. The v1 protein logit is deliberately MAX:

```math
L_b=\max_{p\in P_b} l_p.
```

This encodes the MIL statement “a protein may be positive when at least one surface point has strong
positive evidence.” It also risks overfitting to one spurious point, which is precisely the factor
isolated in v2. The model returns `logits[B]` and `surface_logits[M]`; only the first receives a true
label, so a local score is evidence rather than experimentally validated site annotation.

For target `y_b∈{0,1}`, LambdaForge's binary cross-entropy with logits minimizes

```math
\mathcal L_b=-y_b\log\sigma(L_b)-(1-y_b)\log(1-\sigma(L_b)),
```

where `σ(z)=1/(1+e^{-z})` converts a logit into a probability. AUROC measures how often a randomly
chosen positive is ranked above a randomly chosen negative across all thresholds. AUPRC summarizes
precision versus recall and is especially informative when positives are rare.

Nothing in these model names implies a three-dimensional coordinate update. Positions generated by
preprocessing determine the sparse graphs, but v1 does not feed Cartesian coordinates or normals to
the neural layers. It uses geometry through invariant curvatures and topology while avoiding
dependence on an arbitrary global rotation.

V1 searches only fundamental backbone choices: element-only versus element-plus-residue features;
`E∈{16,32,64}`; `D∈{64,128,256}`; one to four atomic R-GCN layers; one to three projection layers;
one to four surface GCN layers; shared dropout in `[0,0.5]`; weight decay from `10^-6` to `10^-3`;
and learning rate from `10^-5` to `3×10^-3`. Atom→surface mean, global MAX, preprocessing, graph
construction, and dataset splits remain fixed so the experiment answers one question.

### 5.3. WISDOMv2 pooling and localization diagnostics

WISDOMv2 asks whether a rule other than v1 MAX can preserve classification while reducing dependence
on one accidental extreme point. It starts from the reviewed, explicitly materialized v1 backbone
and changes only the operation that turns local logits into a protein logit. Atomic features,
embeddings, R-GCN, atom-to-surface mean, projection, surface GCN, and local head remain controlled
constants. V2 never searches their widths or depths again.

MAX and attention use LambdaForge's sparse indexed poolers; the area-weighted mean uses its sparse
`Scatter` reduction. Top-k and log-sum-exp compact only scalar logits into `X[B,N_max,1]`, where
`N_max` is the largest point count in that batch, and a Boolean mask excludes padding. Atomic and
surface graphs always stay sparse and no fake edges are created.

Global attention uses LambdaForge `SparseAttentionPooling`. Let `h_p∈R^D` be the learned
representation of point `p`, and let `l_p` be its separate positivity logit. Attention computes

```math
s_p=\mathbf v^\top\tanh(\mathbf V h_p),
\qquad
\alpha_p=\frac{e^{s_p}}{\sum_{q\in P_b}e^{s_q}},
\qquad
L_b=\sum_{p\in P_b}\alpha_p l_p.
```

Matrix `V` projects the representation and vector `v` produces one score. Weights `α_p` are
positive and sum to one within a protein. They
mean “importance for this bag decision”; they are not the same quantity as local positivity `l_p`
and must not be presented automatically as a functional-site explanation.

The controlled v2 interface compares these rules:

| YAML value | Implementation | Protein logit and intended behavior |
|---|---|---|
| `max` | `SparseMaxPooling` | Exact v1 existential control: `L_b=max_p l_p`. |
| `mean` | `Scatter.sum` from LambdaForge | Area-weighted mean: `L_b=sum_p w_p l_p/sum_p w_p`. |
| `attention` | `SparseAttentionPooling` | Learned normalized importance from `h_p`, applied to positivity logits. |
| `topk` | `FractionalTopKMeanPooling` | Mean of the largest `ceil(f|P_b|)` logits, with `f` from 1% to 20%. |
| `local_mean_max` | WISDOM regional consensus plus `SparseMaxPooling` | Area-weighted local mean on the existing surface graph, followed by global MAX. |
| `log_sum_exp` | normalized `LogSumExpPooling` | `L_b=β^-1 log(|P_b|^-1 sum_p exp(βl_p))`, a smooth-max control. |

For the main regional hypothesis, let `N(j)` contain vertices with a directed surface edge into
point `j`; let `w_i>0` be represented area; and start with `r_i^(0)=l_i`. Consensus level `t+1` is

```math
r_j^{(t+1)}=
\frac{w_j r_j^{(t)}+\sum_{i\in N(j)}w_i r_i^{(t)}}
     {w_j+\sum_{i\in N(j)}w_i},
\qquad
L_b=\max_{j\in P_b}r_j^{(T)}.
```

The numerator combines area-weighted evidence from a point and its graph neighbors; the denominator
is their represented area. `T∈{1,2,3}` expands the region by one graph hop per level. An isolated
peak is diluted, while a coherent patch remains positive. No geometry is rebuilt. Mathematical
tests cover isolated versus coherent peaks, unequal area weights, and batch isolation. Local
attention is deliberately deferred: LambdaForge has global set attention, but a new learned
variable-neighborhood operator would confound this first regional-consensus test.

Log-sum-exp subtracts its maximum internally for numerical stability and normalizes by point count.
Fractional top-k always selects at least one point. Noisy-OR remains absent because treating thousands
of points as independent Bernoulli variables makes `1-product(1-p)` saturate near one without a
justified physical independence model.

V2 exposes maps that can be saved in the original NPZ point order:

- `surface_logits[M]` and `surface_probabilities[M]=sigmoid(surface_logits)`;
- `localization_scores[M]`, an area-aware distribution
  `q_p = w_p exp(l_p) / sum_q(w_q exp(l_q))` normalized separately per protein;
- `positive_area_fraction[B]`, the normalized represented area with local probability at least 0.5;
- `maximum_surface_probability[B]`;
- `localization_entropy[B]`, equal to `-sum(q_p log q_p)/log(|P_b|)` for bags with more than one
  point. Values near zero indicate concentration and values near one indicate a diffuse map.

These diagnostics describe the model's own map; they are not local labels and are not added to the
loss. In particular, `localization_scores` are a common comparison scale, not necessarily the exact
internal weight of every pooler. These training-time diagnostics do not consume point labels. The
separate post-run evaluator may compare the map with immutable DNA sidecars after model selection;
that later comparison never changes the loss or HPO objective.

### 5.4. Training, evaluation, and artifacts

LambdaForge 0.11 resolves the immutable dataset, expands HPO values and seeds, reserves the GPU,
captures metrics/artifacts, and ranks Works by the declared validation objective. The single public
`train_wisdom` function owns the transparent PyTorch loop: it creates explicit train/validation/test
loaders, applies `WisdomCollator`, trains with AdamW and binary cross-entropy, and preserves the
checkpoint with the greatest validation AUPRC. Test data are read only after that choice.

| Configuration | Responsibility |
|---|---|
| `wisdom_v1.yaml` | Forty sampled candidates over basic capacity, depth, dropout, learning rate, and weight decay; MAX pooling stays fixed. |
| `wisdom_v2.yaml` | An exhaustive six-way pooling ablation with every backbone/training value fixed. |

V1 optimizes validation AUPRC, never test. Its 40 sampled candidates are each repeated with seeds
`[7,17,27]`; `max_parallel: 1` guarantees one single-GPU training at a time. V2 expands MAX, mean,
attention, top-k mean, local-mean/global-MAX, and normalized log-sum-exp exactly once per seed. The
top-k fraction, attention width, regional depth, and log-sum-exp temperature are fixed controls in
this first pooling comparison rather than additional confounded search dimensions.

The callable receives `{dataset: wisdom-dna@2}`, not a machine-specific absolute path. LambdaForge
resolves the selector to the managed root; `WisdomDataset` reads `index.jsonl`, filters the explicit
`split` partition, label target, and requested dilution metadata, and records the exact content/build
identity plus selected placement in materialized evidence. A local workstation and a cluster may
hold verified copies at different paths without editing model parameters or changing scientific
identity. Build or materialize the immutable version before HPO; missing data is never silently
converted into a random split or synthetic labels.

On a managed cluster, first ensure that cluster has a verified placement, then launch the experiment
on the same cluster. No dataset path is passed to the training command because the logical selector
already lives in the YAML:

```bash
lf datasets materialize wisdom-dna@2 --on citius-ctgpgpu12 --strategy replicate --apply
lf run experiments/wisdom_v1.yaml --on citius-ctgpgpu12
```

Inspect composition and plans without creating study state:

```bash
lf datasets list --all
lf datasets show wisdom-dna@2
lf datasets locations wisdom-dna@2
lf validate experiments/wisdom_v1.yaml
lf inspect experiments/wisdom_v1.yaml --resolved
lf run experiments/wisdom_v1.yaml --dry-run

lf validate experiments/wisdom_v2.yaml
lf inspect experiments/wisdom_v2.yaml --resolved
lf run experiments/wisdom_v2.yaml --dry-run
```

Start v1 with the normal command. Repeating the exact command lets LambdaForge reuse or resume its
own durable Work evidence; never edit framework state or event files manually.

```bash
lf run experiments/wisdom_v1.yaml
lf results audit experiments/wisdom_v1.yaml --no-archived
```

Review seed dispersion, learning curves, suspicious search boundaries, and model simplicity; do not
copy the largest decimal blindly. Then copy the selected v1 backbone/optimizer values into the
clearly marked fixed block of `wisdom_v2.yaml` and run its controlled pooling comparison:

```bash
lf run experiments/wisdom_v2.yaml
lf results audit experiments/wisdom_v2.yaml --no-archived
```

Each Work writes two explicit artifacts beside LambdaForge's normal run evidence:

```text
best-model.pt
evaluation.json
```

`best-model.pt` contains the best-validation weights and exact model parameters. `evaluation.json`
contains split sizes, selected epoch, validation AUPRC, and held-out binary metrics. LambdaForge's
`BinaryMetricSuite` preserves mathematically undefined metrics as `null`; it never replaces them
with zero. Local surface sidecars remain excluded from losses, gradients, HPO, and checkpoint
selection. Generic NPZ/3D inspection remains available as described in Section 4.2, independently
of training.

V1 and v2 both omit atomic edge distances, absolute coordinates, normal vectors as neural features,
residue heads, quasi-geodesic kernels, equivariant coordinate updates, dMaSIF convolutions,
bidirectional atom↔surface rounds, contrastive learning, protein language models, and multi-task
outputs. Versions v3–v7 remain documentation-only in
[`docs/model_roadmap.md`](docs/model_roadmap.md). V2 is technically executable but must not be
described as better until the declared poolings are compared on real labels with paired seeds and
disjoint confirmation.

## 6. Bibliography

1. Berman, H. M. et al. (2000). “The Protein Data Bank.” *Nucleic Acids Research*, 28(1),
   235–242. [doi:10.1093/nar/28.1.235](https://doi.org/10.1093/nar/28.1.235).
2. Bourne, P. E. et al. (1997). “Macromolecular Crystallographic Information File.” *Methods in
   Enzymology*, 277, 571–590.
   [doi:10.1016/S0076-6879(97)77032-0](https://doi.org/10.1016/S0076-6879(97)77032-0).
3. Wojdyr, M. (2022). “GEMMI: A library for structural biology.” *Journal of Open Source
   Software*, 7(73), 4200. [doi:10.21105/joss.04200](https://doi.org/10.21105/joss.04200).
4. Lee, B. & Richards, F. M. (1971). “The interpretation of protein structures: Estimation of
   static accessibility.” *Journal of Molecular Biology*, 55(3), 379–400.
   [doi:10.1016/0022-2836(71)90324-X](https://doi.org/10.1016/0022-2836(71)90324-X).
5. Shrake, A. & Rupley, J. A. (1973). “Environment and exposure to solvent of protein atoms:
   Lysozyme and insulin.” *Journal of Molecular Biology*, 79(2), 351–371.
   [doi:10.1016/0022-2836(73)90011-9](https://doi.org/10.1016/0022-2836(73)90011-9).
6. Bondi, A. (1964). “van der Waals Volumes and Radii.” *Journal of Physical Chemistry*, 68(3),
   441–451. [doi:10.1021/j100785a001](https://doi.org/10.1021/j100785a001).
7. Cordero, B. et al. (2008). “Covalent radii revisited.” *Dalton Transactions*, 21, 2832–2838.
   [doi:10.1039/B801115J](https://doi.org/10.1039/B801115J).
8. Saff, E. B. & Kuijlaars, A. B. J. (1997). “Distributing many points on a sphere.” *The
   Mathematical Intelligencer*, 19, 5–11.
   [doi:10.1007/BF03024331](https://doi.org/10.1007/BF03024331).
9. Cazals, F. & Pouget, M. (2005). “Estimating differential quantities using polynomial fitting of
   osculating jets.” *Computer Aided Geometric Design*, 22(2), 121–146.
   [doi:10.1016/j.cagd.2004.09.004](https://doi.org/10.1016/j.cagd.2004.09.004).
10. Bentley, J. L. (1975). “Multidimensional binary search trees used for associative searching.”
    *Communications of the ACM*, 18(9), 509–517.
    [doi:10.1145/361002.361007](https://doi.org/10.1145/361002.361007).
11. Sverrisson, F., Feydy, J., Correia, B. E. & Bronstein, M. M. (2021). “Fast End-to-End Learning
    on Protein Surfaces.” *CVPR 2021*, 15272–15281.
    [Open-access paper](https://openaccess.thecvf.com/content/CVPR2021/html/Sverrisson_Fast_End-to-End_Learning_on_Protein_Surfaces_CVPR_2021_paper.html).
12. Gainza, P. et al. (2020). “Deciphering interaction fingerprints from protein molecular surfaces
    using geometric deep learning.” *Nature Methods*, 17, 184–192.
    [doi:10.1038/s41592-019-0666-6](https://doi.org/10.1038/s41592-019-0666-6).
13. Kipf, T. N. & Welling, M. (2017). “Semi-Supervised Classification with Graph Convolutional
    Networks.” *ICLR 2017*. [arXiv:1609.02907](https://arxiv.org/abs/1609.02907).
14. Schlichtkrull, M. et al. (2018). “Modeling Relational Data with Graph Convolutional Networks.”
    *ESWC 2018*, 593–607. [doi:10.1007/978-3-319-93417-4_38](https://doi.org/10.1007/978-3-319-93417-4_38).
15. Loshchilov, I. & Hutter, F. (2019). “Decoupled Weight Decay Regularization.” *ICLR 2019*.
    [arXiv:1711.05101](https://arxiv.org/abs/1711.05101).
16. Ilse, M., Tomczak, J. M. & Welling, M. (2018). “Attention-based Deep Multiple Instance
    Learning.” *Proceedings of Machine Learning Research*, 80, 2127–2136.
    [PMLR paper](https://proceedings.mlr.press/v80/ilse18a.html).
17. Burley, S. K. et al. (2023). “RCSB Protein Data Bank (RCSB.org): delivery of experimentally
    determined PDB structures alongside one million computed structure models of proteins from
    artificial intelligence/machine learning.” *Nucleic Acids Research*, 51(D1), D488–D508.
    [doi:10.1093/nar/gkac1077](https://doi.org/10.1093/nar/gkac1077).
18. Steinegger, M. & Söding, J. (2017). “MMseqs2 enables sensitive protein sequence searching for
    the analysis of massive data sets.” *Nature Biotechnology*, 35, 1026–1028.
    [doi:10.1038/nbt.3988](https://doi.org/10.1038/nbt.3988).
19. Binns, D. et al. (2009). “QuickGO: a web-based tool for Gene Ontology searching.”
    *Bioinformatics*, 25(22), 3045–3046.
    [doi:10.1093/bioinformatics/btp536](https://doi.org/10.1093/bioinformatics/btp536).
20. Luscombe, N. M., Laskowski, R. A. & Thornton, J. M. (2001). “Amino acid–base interactions: a
    three-dimensional analysis of protein–DNA interactions at an atomic level.” *Nucleic Acids
    Research*, 29(13), 2860–2874.
    [doi:10.1093/nar/29.13.2860](https://doi.org/10.1093/nar/29.13.2860).
21. Li, P., Liu, Y., Liang, L. & Liu, R. (2026). “Datasets for DyProL: Conformational Ensembles of
    Nucleic Acid-Binding Proteins,” version 1. Zenodo.
    [doi:10.5281/zenodo.19547616](https://doi.org/10.5281/zenodo.19547616).
22. Rahman, C. R. et al. (2025). “Benchmarking recent computational tools for DNA-binding protein
    identification.” *Briefings in Bioinformatics*, 26(1), bbae634.
    [doi:10.1093/bib/bbae634](https://doi.org/10.1093/bib/bbae634).
23. Zhang, C., Zhang, X., Freddolino, L. & Zhang, Y. (2024). “BioLiP2: an updated structure database
    for biologically relevant ligand–protein interactions.” *Nucleic Acids Research*, 52(D1),
    D404–D412. [doi:10.1093/nar/gkad630](https://doi.org/10.1093/nar/gkad630).

WISDOM's surface implementation was written independently. dMaSIF and MaSIF motivate the future use
of learned protein-surface representations, but WISDOM does not copy their code and does not claim
algorithmic identity with either method.
