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
  LambdaForge validation, resolved inspection, and dry-run checks.

## Architecture inside `src/`

- Use classes for every reasonable stateful or cohesive source concept. Configuration and immutable
  static data are accepted exceptions.
- Keep at most one class in each Python file. Name a class file exactly like its class, using the
  existing PascalCase convention even though it differs from standard Python module naming.
- Do not add free module-level functions in `src/`.
- Prefer a small number of meaningful domain and service classes. Do not introduce factories,
  adapters, managers, builders, DTOs, wrappers, or interfaces unless they remove real complexity.
- Preserve the ownership hierarchy `Protein -> Chain -> Residue -> Atom`. Do not duplicate atoms or
  residues in parent-level flat collections. Keep provenance and processing metadata outside these
  domain entities.
- Use enums instead of magic strings or unscoped numeric categories whenever the values form a
  closed semantic set.
- Build dataset preprocessing with LambdaForge's public `PreprocessingTask` source, ordered
  transforms, and sink contracts. LambdaForge must own record iteration, stable keys, sharding,
  worker/process selection, failure policy, checkpoints, preprocessing manifests, partial resume,
  and content-derived dataset artifacts; do not recreate those mechanisms inside WISDOM.
- Build WISDOM-DNA in two explicit executions. First run the ordinary LambdaForge selection Task,
  which publishes only evidence tables, homology-safe balanced split lists, deterministic diluted
  views, reports, and resume evidence. Then fetch that small artifact and let one LambdaForge 0.10
  `kind: dataset` recipe generate universal geometry and DNA annotations. Only the recipe's final
  self-contained root with a checksummed `DatasetIndex` may become a `DatasetVersion`.
- Treat dilution membership as a selection concern, not a geometry or annotation concern. Select
  deterministic nested prefixes independently within each split and class, maximize early coverage
  of external sequence clusters, and preprocess the union only once. Materialize each dilution as
  lightweight TXT/JSON/CSV views and final model manifests pointing to shared heavy arrays.
- Treat DatasetRegistry as authoritative for managed immutable versions and their placements. Use
  DataCatalog only for aliases, external datasets, loaders, pins, or institutional overrides; never
  duplicate Registry-managed locations in a project catalog.
- Resolve fixed task paths through named `context.input()` and `context.output()` bindings. Use a
  declared dynamic-path compatibility check only when a manifest can legitimately point to
  arbitrary local files that cannot have one fixed logical name.
- Keep the per-record `PreprocessPipeline` transform readable as commented pseudocode. Scientific,
  parsing, geometry, download, and metadata complexity belongs in cohesive peripheral classes;
  exact NPZ validation, scientific resume, and atomic publication belong at the sink boundary.
- Keep operational execution fields (`workers`, `workload`, `on_error`, checkpoint cadence, and
  resources) out of scientific configuration objects and scientific hashes. Sequential, threaded,
  and spawned-process execution must produce the same scientific content and dataset identity.
- Use LambdaForge `DataCatalog` logical references when one immutable dataset may have different
  physical mounts locally and on managed clusters. Never put cluster-specific absolute data paths
  into model parameters.
- Use LambdaForge managed runners and portable resource requests instead of project-owned SSH,
  scheduler, GPU allocation, subprocess, or SLURM wrappers.
- Treat LambdaForge as an external, read-only dependency. Never modify its source tree, installed
  package, metadata, tests, or documentation without explicit permission from the user for that
  specific change.
- Integrate only through LambdaForge's installed public API and validated configuration schema. If
  a required lifecycle or execution behavior is missing, stop the affected WISDOM work, report the
  exact unsupported case, and ask the user how to proceed. Do not patch LambdaForge provisionally
  and do not duplicate the missing framework infrastructure inside WISDOM.
- Minimize private methods that merely hide a one-use code fragment. Retain a private method when it
  isolates a genuinely complex algorithm, is reused, or materially improves the readable flow.
- Do not add learned features, model-specific preprocessing, neural-network code, or redundant graph
  representations to the structural preprocessing phase.
- Keep trainable WISDOM code equally small and conceptual: one dataset for validated NPZ ingestion,
  one graph collator for domain-specific disjoint batching, and one model class for the
  atom-to-surface computation unless a later requirement demonstrates another cohesive concept.
- Use the installed local LambdaForge public API for training loops, data modules, graph layers,
  scatter operations, losses, metrics, optimizers, checkpoints, seeds, aggregation, and checkpoint
  loading. Do not copy or wrap framework functionality inside WISDOM.
- Prefer LambdaForge result indexing/plotting and generic safe artifact inspection, validation,
  export, and explicit-role visualization over project-local equivalents. Retain a WISDOM-specific
  tool only when it enforces domain semantics that the generic API cannot express at equal quality,
  such as protein topology, signed surface gaps, normal orientation, or curvature identities.
- Keep train/validation/test splits explicit in label manifests. Never invent a random split or
  infer scientific labels from filenames.
- For the DNA-binding benchmark, treat RCSB protein-plus-DNA metadata as candidate discovery only.
  Accept a positive chain only after verifying a real heavy-atom contact to DNA in the selected
  biological assembly. Accept a negative only from explicit curated non-DNA-binding evidence;
  absence of DNA from a deposited structure is unknown, not negative. Quarantine contradictory
  evidence instead of resolving it silently.
- Split DNA data by externally produced sequence clusters (RCSB/MMseqs2 or another documented
  specialist tool), never by individual structures and never with a project-owned mass aligner.
  Reject exact-sequence and cluster leakage across train, validation, and test.
- Keep DNA surface ground truth in a fingerprinted sidecar whose point order and length exactly
  match the immutable universal NPZ. Never rewrite the base archive to add a task label. Exclude the
  surface target from losses, gradients, HPO objectives, and checkpoint selection unless a future
  request explicitly changes the weak-supervision protocol.
- Run final scientific evaluation through LambdaForge's public `post_run` contract and select the
  checkpoint role explicitly. Respect its actual scope: successful completed and trainer-early-
  stopped runs may be evaluated; paused, pruned, cancelled, failed, and cooperatively interrupted
  training runs are not post-run states. Preserve mathematically undefined metrics as unavailable
  values with counts rather than replacing them with zero.
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
- Do not run an automatic formatter over manually aligned `src/preprocess` code if it would remove
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
- Input TXT records are either remote identifiers such as `XYZ_ABC` (protein `XYZ`, chains A/B/C)
  or complete local `.pdb`, `.cif`, `.mmcif`, and optionally `.gz` paths. Do not infer chains from a
  local filename and do not accept the old `#`/comma syntax.
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
  mypy src/preprocess src/wisdom
  pytest -q
  lambdaforge validate experiments/dna_select.yaml
  lambdaforge validate experiments/dna_preprocess.yaml
  lambdaforge run experiments/dna_select.yaml --dry-run
  lambdaforge datasets plan experiments/dna_preprocess.yaml --verbose
  lambdaforge run experiments/dna_preprocess.yaml --dry-run
  ```

- Also run relevant focused tests while iterating. Multiprocessing tests may require execution
  outside a restricted sandbox because Python's process server creates local IPC resources.
- Validate every changed LambdaForge YAML, not only `experiments/dna_preprocess.yaml`. For preprocessing
  migrations, exercise bounded `lambdaforge debug ... --records N` so source and transforms are
  tested independently of production sink publication.
- Verify both README language versions whenever documentation or scientific behavior changes.
- A task is complete only when implementation, English code documentation, bilingual public
  documentation, tests, configuration, and stated mathematical behavior agree.
