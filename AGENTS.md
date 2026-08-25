# WISDOM contributor instructions

These instructions apply to every request that changes this repository. They are part of the
project contract, not optional style suggestions.

## Language and user-facing documentation

- Write all source-code identifiers, comments, docstrings, validation messages, and developer
  documentation in English.
- Maintain the public documentation in two equivalent files: `README.md` in English and
  `README.es.md` in Spanish. English is the default language. Each README must link to the other at
  the top.
- Keep both READMEs structurally equivalent: the same table of contents, commands, equations,
  configuration reference, limitations, and bibliography. A change to one normally requires the
  corresponding change to the other.
- Put installation and practical usage before implementation or scientific background.
- Give every README section and subsection an explicit hierarchical number. Use `1`, `2`, `3` for
  top-level sections, `3.1`, `3.2` for their children, and continue recursively (`3.2.1`,
  `3.2.1.1`) whenever another level is useful. The table of contents must reproduce that hierarchy;
  do not present a long flat list of unrelated-looking entries.
- Use the shallowest hierarchy that keeps the narrative clear. In the README, prefer visually
  prominent level-two and level-three headings. Introduce a level-four heading only when its section
  contains enough independent material to justify several substantial paragraphs and when merging
  it would make navigation materially worse.
- Do not create a heading for every pipeline action, definition, equation, or short explanation.
  Group consecutive actions into a larger conceptual chapter and use bold lead-in phrases, ordinary
  paragraphs, lists, or transitions for local steps that do not need independent navigation.
- Keep headings visually distinguishable from body text. Avoid long runs of small, deeply nested
  headings that render almost like ordinary paragraphs and turn the page into a visually uniform
  wall of text.
- Balance section size: avoid both one-paragraph microsections and chapters so broad that unrelated
  topics lose their internal progression. A subsection should normally develop one coherent idea
  through multiple connected paragraphs, equations, examples, or tables.
- Organize long documentation around a few clear top-level sections. Prefer a structure such as
  quick start, installation, preprocessing, and bibliography over many top-level headings of equal
  apparent importance.
- Write technical documentation as a continuous, cumulative explanation. Begin by giving a novice
  reader a mental model of the problem, then explain why each input or representation is needed,
  how it is obtained, what transformation follows, and how its output becomes the next step's
  input. Use explicit transitions between subsections; do not present pipeline stages as isolated
  facts.
- Prefer direct sentences and concrete verbs. Remove ceremonial framing, repeated conclusions, and
  inflated wording that does not add a definition, decision, result, limitation, or usable action.
  Scientific rigor does not require formal-sounding filler.
- Introduce a multi-stage workflow with one clean diagram or compact summary table when it shortens
  the explanation. End each substantial stage with a concise statement of the data or decision now
  available and what the next stage may do with it.
- Assume that readers may know neither software engineering nor structural chemistry. Define a
  specialized term at first use, state what practical role it has in WISDOM, and avoid relying on
  unexplained abbreviations, file-format fields, enum names, or implementation jargon. Refresh a
  definition or link back to its numbered subsection when a later explanation depends on it.
- Introduce representations before their implementation details. For every graph, array, cache,
  manifest, identifier, or metadata record, explain what it represents, why the project needs it,
  what information it contains, and how the next pipeline stage consumes it.
- Describe scientific algorithms precisely. State definitions, equations, units, thresholds,
  numerical approximations, assumptions, and limitations. Do not replace an explanation with a
  source-code reference.
- Introduce every equation in prose before displaying it. Define every symbol, index, set, unit, and
  operator before or immediately after first use, including apparently standard notation when the
  intended audience may not know it. For a dense expression, derive it in smaller steps and explain
  the meaning of each intermediate quantity before presenting the complete formula.
- Equations must support the narrative rather than interrupt it. After each important equation,
  explain in ordinary language what it tests or computes, why that operation is appropriate, and
  what happens at relevant limiting or failure cases.
- Never use an implementation mechanism as if it were an explanation. Terms such as KD-tree,
  provenance, enum, gzip, hash, sparse graph, model, chain, residue, altLoc, SDF, SAS, and SES need a
  plain-language definition and purpose when first introduced in public documentation.
- Cite primary literature or authoritative format/database documentation in the bibliography.
  Never imply that WISDOM implements an algorithm exactly when it only follows a related idea.

## YAML configuration documentation

- Document every project-authored YAML field and list item with a concise English comment on the
  same line whenever YAML syntax permits it. Use an immediately preceding English comment only
  when an inline comment would make a nested value materially harder to read.
- After every major YAML section, add an English comment block that enumerates all parameters the
  selected public WISDOM or LambdaForge component exposes there, their actual constructor/schema
  defaults, accepted values or numerical constraints, and their precise effect.
- Distinguish a component's true default from the value deliberately selected by the experiment.
  Never label a project override as a framework default.
- Keep YAML comments synchronized with public constructor signatures and LambdaForge schemas.
  A stale configuration comment is a correctness defect, and every changed YAML must still pass
  LambdaForge validation, `lf explain`, and dry-run checks.

## Architecture inside `src/`

- Use classes for every reasonable stateful or cohesive source concept. Configuration and immutable
  static data are accepted exceptions.
- Keep at most one substantive class in each Python file and name that file exactly like its class.
  Several tiny related enums may share one clearly named vocabulary module when separate files
  would add navigation without isolating behavior.
- Expose exactly the cohesive LambdaForge 0.12 Work classes required by public YAML actions:
  `DatasetDesign`, `Preprocessing`, `DNAValidation`, and `Training`. Keep other cohesive
  stateful/scientific concepts as classes; private module helpers are allowed only when they isolate
  a substantial algorithm and cannot be expressed more clearly as a method.
- Prefer a small number of meaningful domain and service classes. Do not introduce factories,
  adapters, managers, builders, DTOs, wrappers, or interfaces unless they remove real complexity.
- Preserve the ownership hierarchy `Protein -> Chain -> Residue -> Atom`. Do not duplicate atoms or
  residues in parent-level flat collections. Keep provenance and processing metadata outside these
  domain entities.
- Use enums instead of magic strings or unscoped numeric categories whenever the values form a
  closed semantic set.
- Make every YAML-executable action a class derived directly from LambdaForge 0.12 `Work`, with all
  scientific parameters on its single public `run()` method. Never use function targets,
  constructor injection, method escape hatches, removed `Task`/`TaskContext`/`PreprocessingTask`
  APIs, or project-owned framework compatibility shims.
- Author every user-facing LambdaForge 0.12 YAML only with `name`, `run`, `with`, `resources`,
  `seeds`, `search`, `objective`, and `steps` as applicable. Do not add `output_root`, `kind`,
  `schema_version`, `inputs`, `outputs`, `trials` outside `search`, `max_parallel`, object graphs,
  DatasetRecipe stages, or model/loss/optimizer construction trees.
- Use `self.resume_map` for bounded intra-Work parallelism when record results are reconstructible
  and should be reused with dependency-aware checkpoints; use `self.map` only when WISDOM owns a
  stricter scientifically validated record-resume boundary;
  `self.cache.file/fetch/rate_limit` for reconstructible managed files; `self.checkpoints.file` for
  validated durable phase results; `self.tools.require/run` for external programs and their
  provenance; and `self.progress`, `self.metrics`, and `self.outputs` for their exact public
  purposes. Do not recreate cache locks, retry loops, partial files, atomic replacement, subprocess
  logging, environment thread limits, output fingerprinting, or dependency restoration in WISDOM.
- Build WISDOM-DNA through two public Work classes. `DatasetDesign` consumes immutable typed JSONL
  evidence (with legacy FASTA accepted only for reproducibility) and owns structure/contact
  revalidation, full-raw MMseqs2/Foldseek leakage groups,
  physical phenotypes, canonical balancing, fixed splits, nested train dilutions, statistics, and
  reports in one deliberately cohesive implementation file. `Preprocessing` consumes that exact
  design, generates only universal geometry and DNA sidecars, validates the joined result, and then
  publishes it through LambdaForge 0.12 `self.outputs.dataset(...)`. It must never rediscover,
  rebalance, recluster, repartition, or dilute proteins.
- Keep all code under one top-level `wisdom` package. Put the two action modules under
  `wisdom.preprocessing`, structural internals under `wisdom.preprocessing.structure`, DNA-specific
  internals under `wisdom.preprocessing.dna`, and trainable data/models/evaluation in their named
  sibling packages. Do not restore parallel top-level packages with cross-cutting imports.
- Create dilutions only after final leakage groups and phenotype clusters exist. Reduce training
  only; validation and test remain fixed. Keep fractional views deterministic, nested, as
  class-balanced as indivisible groups permit, and representative of leakage/phenotype groups.
- Treat DatasetRegistry as authoritative for managed immutable versions and their placements. Use
  DataCatalog only for aliases, external datasets, loaders, pins, or institutional overrides; never
  duplicate Registry-managed locations in a project catalog.
- Receive external files and datasets as typed `Path` parameters resolved by LambdaForge. Declare
  final files/directories with `self.outputs`, resumable results with `self.checkpoints`, and
  reconstructible downloads with `self.cache`; use `self.temp_dir` only for disposable specialist
  scratch. Treat raw `self.run_dir`, `cache.path`, and `checkpoints.path` as advanced escape hatches,
  not normal application APIs, and never expose physical framework paths in YAML or persisted data.
- Keep the per-record `PreprocessPipeline` transform readable as commented pseudocode. Scientific,
  parsing, geometry, download, and metadata complexity belongs in cohesive peripheral classes;
  exact NPZ validation, scientific resume, and atomic publication belong at the sink boundary.
- Keep operational execution fields (`workers`, map executor choice, and `resources`) out of
  scientific configuration objects and scientific hashes. Sequential, threaded, and spawned-process
  execution must produce the same scientific content and dataset identity.
- Resolve Registry-managed data with typed `{dataset: name@version}` parameters. Reserve
  `DataCatalog` for aliases, external datasets, loaders, pins, or institutional overrides; never put
  cluster-specific absolute data paths into model parameters.
- Use LambdaForge managed runners and portable resource requests instead of project-owned SSH,
  scheduler, GPU allocation, subprocess, or SLURM wrappers.
- Declare project-native command-line dependencies once in `[tool.lambdaforge.environment]`, backed
  by one repository-contained Conda environment file or explicit lock. Keep executable names bare;
  never install native packages from a Work or encode machine-specific Conda paths in WISDOM.
- Resolve and version-probe every external program at the beginning of the Work that actually uses
  it, before downloads or expensive computation. A failure must identify the tool and tell local
  users to run `./install.sh` and managed-cluster users to run
  `lf clusters bootstrap <cluster> --project .`. Do not impose unrelated tool requirements on a
  Work that never executes those programs.
- Treat LambdaForge as an external, read-only dependency. Never modify its source tree, installed
  package, metadata, tests, or documentation without explicit permission from the user for that
  specific change.
- Integrate only through LambdaForge's installed public API and validated configuration schema. If
  a required lifecycle or execution behavior is missing, stop the affected WISDOM work, report the
  exact unsupported case, and ask the user how to proceed. Do not patch LambdaForge provisionally
  and do not duplicate the missing framework infrastructure inside WISDOM.
- Minimize private methods that merely hide a one-use code fragment. Retain a private method when it
  isolates a genuinely complex algorithm, is reused, or materially improves the readable flow.
- Treat documented researcher-authored inputs and scientific parameters as contracts. Do not add a
  second exhaustive validation layer for fixed formats such as the two-line WISDOM-DNA FASTA or for
  every numeric YAML threshold; let malformed usage fail naturally at the operation that consumes
  it. Retain explicit checks only when they prevent silent scientific corruption, data leakage,
  unsafe publication, or an otherwise plausible but invalid benchmark result.
- Do not add learned features, model-specific preprocessing, neural-network code, or redundant graph
  representations to the structural preprocessing phase.
- Keep trainable WISDOM code small and conceptual: one dataset for validated `index.jsonl`/NPZ
  ingestion, one graph collator for domain-specific disjoint batching, the two explicitly requested
  model generations, and one `Training` Work.
- Let LambdaForge 0.12 resolve datasets, expand seeds/search, reserve resources, capture
  metrics/artifacts, and compare objectives. `Training.run()` owns its readable PyTorch loop and
  uses LambdaForge public graph layers, scatter operations, pooling, and metrics; do not reconstruct
  framework scheduling, Registry, results, or HPO infrastructure.
- Prefer LambdaForge result indexing and plotting for run-level evidence. LambdaForge 0.12 does not
  expose the former generic artifact-inspection command family, so retain WISDOM tools that enforce
  protein topology, signed surface gaps, normal orientation, curvature identities, and visual NPZ
  inspection.
- Keep train/validation/test splits explicit in label manifests. Never invent a random split or
  infer scientific labels from filenames.
- For the DNA-binding benchmark, treat RCSB protein-plus-DNA metadata as candidate discovery only.
  Accept a positive chain only after verifying a real heavy-atom contact to DNA in the selected
  biological assembly. Accept a negative only from explicit curated non-DNA-binding evidence;
  absence of DNA from a deposited structure is unknown, not negative. Quarantine contradictory
  evidence instead of resolving it silently.
- Keep three dataset concepts explicit and separate. Leakage groups contain sequence, Foldseek
  structure, exact-sequence, logical-identity, deposition, or coordinate connections and have no
  functional meaning. Positive phenotype clusters describe local physical DNA-binding-site shape;
  negative phenotype clusters describe global protein morphology. Neither phenotype may define a
  leakage edge.
- Compute sequence pair evidence with a versioned external MMseqs2 installation and structure pair
  evidence with a versioned external Foldseek installation after geometry exists. Retain thresholded
  pair tables, join all similarity/identity edges transitively, and assign each connected leakage
  group wholly to train, validation, or test. Never silently fall back when either tool is absent.
- Fit positive and negative physical phenotypes separately with median/IQR robust scaling and
  LambdaForge's public `lambdaforge.clustering.HDBSCAN`/`stability` services. WISDOM owns the
  scientific variables, scale, parameter grid, stability threshold, and interpretation—not a
  clustering backend. Keep HDBSCAN noise explicit, never cluster a UMAP visualization, and report
  honestly when no robust multi-cluster solution exists.
- Require every positive validation/test member to have usable local ground truth with at least one
  positive surface point. A positive without local ground truth may remain in training/global
  classification only; it may never be converted into an all-negative local target.
- Keep DNA surface ground truth in a fingerprinted sidecar whose point order and length exactly
  match the immutable universal NPZ. Never rewrite the base archive to add a task label. Exclude the
  surface target from losses, gradients, HPO objectives, and checkpoint selection unless a future
  request explicitly changes the weak-supervision protocol.
- Select checkpoints only by explicit validation metrics and evaluate the test partition afterwards
  in the same callable Work. Preserve mathematically undefined metrics as unavailable values with
  counts rather than replacing them with zero; never use test/local targets in HPO or checkpoint
  selection.
- Implement only the requested model version. Roadmap descriptions are not authorization to add
  later architectures, features, heads, or training objectives.
- When a model version is intended to test one scientific factor, keep every other component fixed
  and express the alternatives as LambdaForge ablations. Do not change an encoder and its pooling
  rule in the same comparison unless the request explicitly defines that combined hypothesis.
- Distinguish technical verification from scientific validation. Unit tests, configuration
  validation, and dry runs establish executability; only appropriate real data, metrics, controls,
  repeated seeds, and statistical comparisons support claims of scientific improvement.

## Visual layout of Python code

- Separate assignments into logical blocks. Insert one blank line between groups that serve
  different purposes, even when all assignments are consecutive syntactically.
- Within each short logical block, vertically align assignment operators when this improves visual
  scanning. Example:

  ```python
  atom_count    = len(atoms)
  residue_count = len(residues)

  edge_count = edge_index.shape[1]
  file_size  = output_path.stat().st_size
  ```

- Align type-annotation colons and default-value operators for consecutive related parameters or
  fields. Example:

  ```python
  def build(
      self,
      resolution   : float = 1.0,
      probe_radius : float = 1.4,
  ) -> Result:
  ```

- Apply alignment only inside a genuinely related block. Do not align unrelated statements across a
  long distance, and do not create extreme spacing because one identifier is unusually long.
- Keep nested expressions readable. Prefer a descriptive intermediate variable and a visually
  separated block over dense nesting.
- Do not run an automatic formatter over manually aligned `src/wisdom/preprocessing` code if it would remove
  this required alignment. Continue running Ruff linting. Autoformat tests or other unaffected files
  separately when useful.

## Comments and docstrings

- Every source-code method must have a substantive English docstring. Explain its exact purpose,
  algorithm, returned value, important side effects, and failure conditions instead of using a vague
  one-line restatement of the method name.
- Document every non-`self`/`cls` parameter in the method docstring. Use a consistent `Args:`,
  `Returns:`, and `Raises:` structure where applicable. State units, shapes, coordinate systems, and
  expected dtypes for scientific arrays.
- For mathematical or scientific methods, include the relevant equations in readable plain-text or
  reStructuredText form and explain why the approximation is used. Define every symbol close to the
  equation.
- Add an English comment before every relevant non-trivial block explaining what the block
  accomplishes or why the operation is necessary. Comments may use any natural phrasing; they do not
  need to begin with a fixed formula such as “This block”.
- Do not comment obvious syntax, increments, simple assignments, or facts already stated by a nearby
  precise docstring. Comments must add intent, scientific meaning, an invariant, or a non-obvious
  implementation reason.
- Keep comments synchronized with the code. Incorrect mathematical documentation is a correctness
  bug.

## Scientific and data invariants

- Use Gemmi for PDB/mmCIF parsing. No custom coordinate-file parser may be introduced.
- Input TXT records are either remote identifiers such as `XYZ_AQ` (protein `XYZ`, one chain named
  `AQ`) or `XYZ_A_Q` (two chains named `A` and `Q`), or complete local `.pdb`, `.cif`, `.mmcif`, and
  optionally `.gz` paths. The first underscore separates the PDB ID from a complete chain name;
  every later underscore introduces another complete chain name. Do not infer chains from a local
  filename and do not accept the old `#`/comma syntax.
- Keep atom and surface graph construction sparse. Never materialize dense `N x N` or `M x M`
  distance matrices.
- Persist each undirected graph pair once with `src < dst`. Preserve covalent edges outside spatial
  cutoffs and distinguish relation semantics with enums/bit masks.
- Keep generated surfaces deterministic for identical structure and configuration.
- Persist only fixed structural/geometric information. Do not persist one-hot encodings, RBFs,
  embeddings, messages, attention values, or other learned/model-specific features.
- Keep NPZ files pickle-free and free of `dtype=object`. Validate arrays before atomic publication.
- Resume a protein only when source hash, scientific configuration hash, and schema version match
  and the exact existing archive passes the complete schema and numerical validation again.
- Keep whole-task byte-integrity checks and WISDOM scientific validation conceptually separate.
  LambdaForge artifact hashes establish byte identity; the WISDOM validation task must independently
  audit manifest coverage, exact NPZ schema, numerical invariants, provenance, and report agreement.
- Make dataset-validation results useful to humans and programs: publish a concise plain-text verdict
  and an ordered detailed JSON report with explicit per-protein failures.

## Required verification

- Preserve unrelated user changes and never use destructive Git or filesystem commands to clean the
  workspace.
- For normal source changes, run at least:

  ```bash
  ruff check .
  mypy src/wisdom
  pytest -q
  lf validate experiments/dna_design.yaml
  lf explain experiments/dna_design.yaml
  lf run experiments/dna_design.yaml --dry-run
  lf validate experiments/dna_preprocess.yaml
  lf explain experiments/dna_preprocess.yaml
  lf run experiments/dna_preprocess.yaml --dry-run
  lf validate experiments/validate_dna.yaml
  lf validate experiments/wisdom_v1.yaml
  lf validate experiments/wisdom_v2.yaml
  ```

- Also run relevant focused tests while iterating. Multiprocessing tests may require execution
  outside a restricted sandbox because Python's process server creates local IPC resources.
- Validate every changed LambdaForge YAML. When an authored file deliberately references a not-yet
  published design artifact or DatasetVersion, validate an equivalent temporary fixture binding and
  report that the production selector cannot resolve until its upstream action has run. Never
  publish fake data just to satisfy validation.
- Verify both README language versions whenever documentation or scientific behavior changes.
- A task is complete only when implementation, English code documentation, bilingual public
  documentation, tests, configuration, and stated mathematical behavior agree.
