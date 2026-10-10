# Common preprocessing and H/X/Z refactor

This report records implementation and verification against installed LambdaForge 0.17.1,
not new benchmark performance. No production
dataset, remote preprocessing job, training Study or external acquisition was executed. LambdaForge
was inspected and used through its installed public API; its source and installed code were not edited.

## 1. Directory structure

| Location | Responsibility |
| --- | --- |
| `src/wisdom/preprocessing/common` | Task-independent design, independence, sampling and scientific resume |
| `src/wisdom/preprocessing/common/structure` | Universal arrays, atomic topology, surfaces and exact NPZ validation |
| `src/wisdom/preprocessing/dna` | DNA evidence, local contact descriptors, design orchestration and annotation |
| `src/wisdom/preprocessing/zinc` | Explicit Zn evidence, coordination, site phenotypes and annotation |
| `src/wisdom/utils/structure` | Shared deposited-structure and molecule ownership hierarchy |
| `src/wisdom/features` | Optional fixed fields and frozen training transforms |
| `src/wisdom/{data,models,evaluation}` | Generic ingestion, trainable hypotheses and diagnostics |

Common algorithms do not import task evidence or orchestration. The parent preprocessing namespace
no longer eagerly imports DNA Works. Task Works remain short scientific orchestration scripts.

## 2. Moved modules

Similarity, transitive leakage grouping, canonical selection, fixed splitting, nested train dilution
and composition auditing moved from DNA selection into `common`. Geometry execution and progress
heartbeat moved from DNA preprocessing. Snapshot publication and verification now share one common
module. Universal structural algorithms moved from `preprocessing/structure` into `common/structure`.
Their public imports and all project references were updated rather than retained as old-path aliases.

## 3. Common scientific boundaries

`descriptors` contains genuinely shared protein coordinates and whole-protein descriptors;
`phenotypes` exposes robust scaling and native LF HDBSCAN/stability without fixing task meanings.
`functional_metadata` binds frozen external annotations to exact sequences. `shortcut` fits a small
covariate classifier on train and evaluates validation only. Population selection takes a supplied
origin preference, arbitrary class ratio, an all-positive policy and extra diversity strata. Split
assignment accepts training-only members and restricts their entire leakage groups before placing
ordinary groups. Phenotypes never create leakage edges.

## 4. Deleted code and duplication

Moved scientific implementations were removed from their old locations. Shared snapshot, descriptor,
clustering and shortcut logic was consolidated. The 27 tracked `build/lib` files were removed; ignored
local build caches were not recursively deleted. `setup.cfg` retains `force=1`: a regression test
demonstrates that otherwise a stale, newer-dated cached module can enter a wheel. Historical LF
bundles, checkpoints, dataset directories and result evidence were preserved, not migrated in place.

## 5. Zn scientific changes

Coordination schema 2 records distinct residue multiplicities, atom/element counts, donor distances,
donor-pair angles, selected chain copy and interchain participation. Cys2His2 and Cys1His3 remain
different. Accepted sites get separate local descriptor rows; multiple sites retain a composition,
not an averaged phenotype. Global morphology includes both classes. `zinc-sites.jsonl` exposes raw
accepted site evidence and whether its protein was selected.

Early surface-support checks use the universal reader's filtering, centered deposited coordinates,
float32 atoms/radii and exact deterministic boundary sampling. Points are then restored and moved
to the selected assembly; directly resampling a rotated copy would change voxel selection. Curvature,
graphs and spectral operators are not required for this check. A buried coordination positive stays
positive but its whole leakage group is train-only. Final annotation independently verifies local
support. Negative evidence is still explicit and sequence-bound; missing Zn is never a negative.
"Buried" here means unsupported on WISDOM's selected protein-only boundary, not experimentally
established solvent inaccessibility in the complete biological assembly.

`positive_policy` selects a diverse quota or all eligible positives. Selection audits report the actual
ratio and compare split proportions to the selected policy. Optional sequence-bound functional
metadata and coordinate/site-bound `site_metadata` can preserve reviewed external database links,
family labels and release provenance. They cannot replace labels or coordination evidence. No live
ZincBind/MetalPDB adapter or new reliable negative inventory is claimed.

The primary GT remains the signed gap to verified coordinating protein donors. Annotation schema 2
adds separate Zn-center hard/soft/mask/sensitivity diagnostics, never replacing that primary target.
Missing Zn-center distances in negatives remain unavailable. Historical annotation schema 1 remains
readable. Geometry does not prove physiological binding, affinity, protonation or metal specificity.

## 6. Fixed field chemistry

Field schema 2 excludes backbone O/OXT from strict named CHED donor eligibility. Zn donor N/O/S
channels now count strict donors rather than duplicating broad elemental channels. CHED residue
density counts CA contributions; hydropathy and polarity average CA residue contributions instead
of giving large sidechains more votes. No ligand coordinates, task GT or learned representation is
used by these projections. Weighted counts remain neighborhood proxies, not physical volume densities.
Schema-1 sidecars preserve their actual stored values and original meanings; they are not relabelled.

## 7. Normalization and feature audit

The augmentation publishes a schema-2 collection with an independent fit for full train and each
published dilution. Validation and test reuse the chosen train fit without filtering or fitting.
Three explicit measures are supported: pooled points, represented area, or equal total protein
weight. Weighted centered covariance merging avoids retaining all surfaces and reduces numerical
cancellation. Statistics bind member IDs, base/feature digests, point counts and ordered channels.

The JSON and Markdown audit explain range, mean/std, finite support, zero/nonzero fractions,
constant/unsupported channels and pairwise correlations. Undefined correlation remains unavailable;
absolute correlation >=0.999 flags possible redundancy without deleting channels. Full-only historical
statistics are rejected for diluted training rather than leaking excluded train proteins into the fit.

## 8. Final H/X/Z contract

| Output | Meaning |
| --- | --- |
| `surface_learned_embeddings` | Real learned H; absent in explicit mode |
| `surface_explicit_features` | Selected, frozen-normalized X; absent in learned mode |
| `surface_evidence_features` | Actual head input Z: H, X, or concatenated H/X |
| `surface_embeddings` | Deprecated compatibility alias for H only |

Attention, direct heads and refiners name their actual Z input explicitly. Refiner guidance is
`SurfaceEvidenceContext.evidence_features`, never an assumed H. Prediction exports distinguish
H/X/Z; lightweight prediction views do not copy full matrices. Sparse concept discovery explicitly
rejects explicit/hybrid checkpoints because its current scientific protocol assumes learned H.
Renaming runtime tensors and method arguments does not change parameter keys or model equations.

## 9. Experiment organization

All 59 authored YAMLs now live in hierarchical families: `preprocess/{dna,zinc,common}`, `v1` through
`v10`, family-local `visualization`, `ablations/surface_representation`, `interpretability`, and shared
`reviews/{policies,templates}`. There are 54 executable configurations and five selection policies.
Names and filename-derived study identities were preserved; typed paths, commands and test lookups
were rebased. Training Studies remain single Works; only preparation composes steps.

The primary five-way H/X/Z comparison fixes logit-only MAX pooling, avoiding an Attention-width
confound. `attention.yaml` is an explicitly secondary control; its scorer parameter count varies.
The corrected augmentation proposes `wisdom-dna-features@2`. New DNA preparation proposes reduced
version 7; existing experimental ledgers intentionally retain historical reduced version 6.

## 10. Historical compatibility

Readers translate historical `interface_phenotype` to generic `local_phenotype` only when needed.
Published versions are not rewritten. Legacy Zn coordination compares its complete original
vocabulary without requiring newly derived statistics; partner ordering is not evidence. Current
coordination compares all coordination fields, excluding separately derived support/phenotype/external
annotations. Exact source bytes and donor evidence are still required. Old feature statistics are
full-only; old checkpoints retain weight keys and learned-mode numerical behavior. Source import
paths and ambiguous internal tensor keyword names changed intentionally; no obsolete source shims
were added to resurrect old Work modules.

## 11. Tests and CI

New offline tests cover common-package independence, the three weighted moments, exact fit membership,
undefined correlations, strict donor chemistry, CA residue weighting, legacy field vocabulary,
H/X/Z outputs and gradients, sparse-concept rejection, residue multiplicity and angles, historical
coordination ordering, exact pre-rotation boundary sampling, multisite clustering, whole-group
train-only restrictions, all-positive ratios and frozen functional/site metadata.

Existing tests now use moved modules/configurations. A native isolated fixture validates, explains
and dry-runs every authored executable and parses every selection policy, without production data
or training. GitHub CI runs Ruff, mypy and offline CPU fixtures. Optional network, native-tool,
GPU, production-data and browser checks have explicit markers. The GitHub workflow itself has not
been run on GitHub as part of this change.

## 12. Verification results

- `ruff check .`: passed.
- `mypy src/wisdom`: passed, 167 source files.
- `pytest -q`: 393 passed, three skipped (one absent production benchmark; two Playwright tests).
- All 54 current executable YAMLs: native fixture-bound validation, explanation and dry run passed;
  all five current policies parsed successfully. Historical hidden LF bundles are excluded.
- Actual DNA preparation: `lf validate`, `lf explain` and `lf run --dry-run` passed.
- Actual DNA validation, V1a and V2 configurations: `lf validate` passed.
- Production Zn evidence/design, `wisdom-zinc@1`, corrected `wisdom-dna-features@2`, sparse winner
  checkpoint and example `review-models` selectors are not all available locally. Their configuration
  checks used temporary fixture bindings, not counterfeit production registrations.

## 13. Deliberately unchanged

No training loss, pooling algorithm, learned encoder, HPO objective, checkpoint rule or scientific
split was changed in an existing experiment. No observed Zn was admitted to model inputs. No
automatic functional-family classifier, inferred negatives, hidden retraining, clustering backend,
cache locks or LF publication infrastructure was implemented. Native tools remain early Work
requirements. Structural hashes were not weakened to hide coordinate revisions.

## 14. Real follow-ups and limitations

Researchers must still assemble and review reliable Zn negatives; coordination is not a general
physiological-affinity assay. Sparse/local site support and phenotype stability need real populations;
all-noise or unsupported clusters remain explicit. Changed sampling settings require a new design
and final evaluation audit. External site mapping needs exact assembly/copy review and a frozen
release, not approximate PDB-name matching. Cross-protein numerical support, covariance shortcuts,
cost and biological usefulness require real repeated-seed experiments. Corrected fields and
normalization require a new augmentation publication. No production dataset was published, no old Study
was relaunched, and passing fixtures does not establish scientific improvement.
