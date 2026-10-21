# Universal structural enrichment: implementation audit

## 1. Compatibility decision

The inspected writer is `ProteinArchive.SCHEMA_VERSION = "3.0"`. DNA publication's
`structural_schema="2.1"` is an incorrect description, not a second archive format:
the shared geometry generator invokes the schema-3 writer for both tasks. Correct the
publication description; do not rewrite published data or change the schema-3 contract.

Atom order is chain traversal, retained residue traversal, then sorted atom name.
`ProteinReader` selects one model (default zero), resolves complete author chain IDs,
filters entities, selects highest occupancy with a blank/A/alphabetical tie-break, and
optionally centers coordinates. `AtomicStructureBuilder` assigns consecutive residue
and chain indices. Consecutive residue indices alone do **not** establish a peptide bond.

## 2. Pre-change property inventory

Here “loader/collator” refers to the existing trainable path, not availability in a file.
Both DNA and Zn preserve the same universal archive; task annotations are separate assets.
N, R, M and E denote retained atoms, residues, surface points and undirected atomic edges.
Costs below are array-size estimates, not measurements.

| Property | Source → reader/domain → base archive | Existing model / loader / collator | Derivable without source? / ambiguity | Additional cost |
| --- | --- | --- | --- | --- |
| Atom/residue names | PDB/mmCIF → preserved → U8 arrays [N] | Names support deterministic chemistry; not neural inputs | Yes; unknown components possible | Reuse strings; compact IDs 2N bytes each |
| Atom→residue→chain | Preserved hierarchy → int32/int16 [N] | Used for structural batching/context | Yes; author numbering not recoverable | Reuse indices, compact residue table |
| Author residue number/insertion/chain name/polymer flag | Preserved in domain, absent as residue tables in NPZ | Not consumed | Requires exact verified source; repeated chain names/copies need indices | O(R + chains) |
| Positions/elements/radii/formal charge | Preserved/element tables → float32/uint8/int8 | Consumed and batched | Yes; formal charge is not a partial charge or a verified protonation state | No duplicate tables |
| Covalent/spatial topology | Templates + explicit links + conservative geometric reconstruction → sparse [2,E] | Consumed and batched | Yes; bond source/confidence retained; inferred bonds are not observations | Derived heavy degree O(N+E) |
| Approximate hybridization/aromatic/donor/acceptor flags | Derived by AtomicDescriptors → optional schema-3 vectors | Consumed; early archives derive deterministically | Yes; maximum bond order and name templates are approximations | Historical semantics unchanged |
| Hydropathy/polarity | Static residue lookup → vectors [N] | Consumed and batched | Yes; unknown zero is historical, not chemical evidence | No duplication |
| Backbone/side-chain torsions | Coordinates/connectivity available, not previously derived/persisted | Not consumed | Yes where canonical atoms and actual peptide connections exist; otherwise masked | Angles + masks O(R) |
| Selected occupancy/B/altloc | Gemmi selection uses occupancy; domain discards values | Not consumed | Source required; missing source fields must not become defaults | O(N), optional quality family |
| Functional motifs | Partially implicit in names/templates, not explicit | Not consumed | Canonical template assignment, not protonation or charge inference | uint16 bit masks + known mask |
| Surface geometry/curvatures/areas | Derived → float32 arrays [M,3], [M,S,3], [M] | Consumed and batched | Yes | No recomputation |
| Intrinsic operators/eigenvectors | Derived → sparse derivatives and float32 spectral arrays | DiffusionNet inputs | Yes | No recomputation |
| Surface→atom candidates | KD-tree → dense int32/float32/bool [M,32], radius 6 Å | Consumed; prefixes select runtime K | Expanded candidates derive from existing coordinates/normals | Optional CSR O(M+P), explicit truncation |
| Local frames/bond angles | Not persisted | Not consumed | Yes with valid nondegenerate atoms/covalent paths | Runtime only |
| Solvent-accessible area | Not persisted per atom | Not consumed | Requires a separately specified algorithm; not equivalent to surface weights | Deferred, not silently approximated |

`ProteinSink.resume` reopens the exact archive, verifies source/config/schema and full
numerical validation. Worker count is outside scientific configuration. Neither historical
`WisdomDataset` nor `WisdomCollator` needs the optional assets: retaining this path prevents
an enrichment release from silently changing a model's hypothesis or a saved checkpoint.

## 3. Publication and validation boundary

New sidecars must bind the exact base SHA-256, use pickle-free arrays, carry independent
schemas and be fully validated before reuse/publication. Native LambdaForge managed files
own checkpoints and retries. DNA and Zn attach the same optional asset names. Enrichment
of an already published version imports existing assets through LambdaForge and publishes
a new version; it must never regenerate a surface or alter a task annotation.

## 4. Implemented sidecar contracts

Both schemas are **1.0**, with scalar Unicode `metadata_json` and `base_npz_sha256`.
All arrays load with `allow_pickle=False`; object arrays are forbidden. The base is
fully schema/numerically validated before enrichment. Its config hash and historical
geometry, source identity and byte digest remain unchanged. Dependency versions and
algorithm names are recorded; worker count is not a scientific parameter.

**Atomic context.** N/R/E/M are as above; C is the number of retained chain objects,
not the number of distinct chain-name strings. The exact mapping remains in the base's
`residue_indices[N]` and `chain_indices[N]`; these arrays and its strings are not copied.

| Arrays | Shape / dtype | Contract |
| --- | --- | --- |
| `atom_name_id`, `atom_identity_id` | [N] uint16 | Fixed canonical name and `(residue,atom)` categories; zero UNK |
| `atom_position_id` | [N] uint8 | Distinct backbone N/CA/C/O/OXT and beta/gamma/delta/epsilon/zeta/eta categories; not ordinal distances |
| `residue_chain_index` | [R] int16 | Parent chain index from the base |
| `author_number`, `insertion_code` | [R] int32 / U4 | Source author residue ID, preserving insertion distinctions |
| `is_polymer`, `author_valid` | [R] bool | Polymer status and source availability, not inferred from unknown names |
| `chain_author_id` | [C] U32 | Retained author chain names; array indices distinguish repeated names/copies |
| `torsion` | [R,7] float32 | phi, psi, omega, chi1..4, radians [-π,π] |
| `torsion_valid`, `torsion_applicable` | [R,7] bool | A canonical definition exists vs actual connected nondegenerate atoms exist |
| `heavy_degree` | [N] uint16 | Number of represented heavy covalent neighbors, not complete valence |
| `ring_member`, `chemistry_known` | [N] bool | Membership in represented canonical intra-residue cycles; canonical named-atom support |
| `functional_group_bits` | [N] uint16 | Overlapping template roles; ordered bit vocabulary in metadata |
| `occupancy`, `b_iso` (optional) | [N] float32 | Fraction [0,1], isotropic B in Å²; only explicitly populated mmCIF values |
| `occupancy_valid`, `b_iso_valid` (optional) | [N] bool | Distinguish missing/default quality from observed zero |
| `selected_altloc`, `alternate_count` (optional) | [N] U4 / uint16 | Selected alternate and candidate multiplicity after element filtering; source-unavailable count is zero |

Source-only fields require the original exact-byte digest, original model/chain/filter
configuration, and one-to-one equality of names, elements, residue/chain ownership and
stored float32 coordinates. An imported portable uncompressed file is not assumed to
equal a historical gzip container. If that exact original is unavailable when constructing
new context, source-only fields are masked, not populated from a current database revision.
An existing compatible context is revalidated and retained, never downgraded. The persisted
`source_aligned` flag makes this limitation explicit. PDB occupancy/B are deliberately
unavailable: Gemmi supplies missing-value defaults and no second coordinate parser was added.
Isotropic B additionally requires a declared diffraction/electron-microscopy method;
predicted or unknown methods remain masked, never treating confidence as experimental B.
Oversized chain/insertion strings fail rather than truncate silently.

Canonical chemistry uses the existing twenty-residue heavy-atom templates. Exact
identities such as ASP OD1/OD2 or LEU CD1/CD2 remain different. No symmetry quotient is
applied: protonation/tautomer state and terminal symmetry are not resolved here.
Backbone amide/carbonyl, carboxyl, guanidino, amine, hydroxyl, thiol/thioether,
aromatic, imidazole and represented disulfide roles may overlap. They are canonical
template assignments, not proof of charge, protonation or a complete functional group
when other atoms are missing. Ring flags describe the represented covalent graph;
unknown chemical assignments have a false `chemistry_known` mask. Spatial-only and
zero-order coordination edges do not contribute to degree, torsions, cycles or angles.
Historical hybridization/donor/acceptor approximations are not changed or rebranded as exact.

**Extended candidates.** P is the number of retained point–atom pairs. CSR avoids M×N
distance matrices and M×K padding. Each row sorts by stored float32 distance, then atom ID.

| Arrays | Shape / dtype | Contract |
| --- | --- | --- |
| `point_ptr` | [M+1] int64 | Starts at zero; nondecreasing; final value equals P |
| `atom_index` | [P] int32 | Base atom indices, unique within each row |
| `distance`, `normal_offset`, `tangential` | [P] float32 | Å; normalized outward-normal convention |
| `candidate_count`, `discarded_count` | [M] int32 | Actual in-radius count and number omitted by K; row length + discarded = candidate count |

Metadata fixes radius, cap, order, units, algorithm and quantiles. Empty rows are legal;
there is no implicit atom zero. Validation deterministically reconstructs the complete
candidate search, including pre-truncation counts, so it detects a missing nearest atom
as well as incorrect stored geometry. The existing 6 Å/32 table is not replaced.
`ExtendedSurfaceAtom.dense` explicitly pads with ID −1, false mask and zero geometry.
The opt-in `ExtendedSurfaceAtom.batch` concatenates validated lists with independent
atom/point/pair offsets and explicit protein boundaries. It uses **base atom counts**,
not the largest observed candidate ID; atoms absent from every candidate list still
reserve their namespace. Historical `WisdomCollator` does not call this method.
For a smaller requested radius, a future adapter must additionally test exact coordinate
distances, not only rounded stored distances at the cutoff boundary.

## 5. Geometry, conventions and derived-only quantities

For ordered points x0,x1,x2,x3, let u=(x2−x1)/||x2−x1||. Remove the component parallel
to u from b0=x0−x1 and b2=x3−x2 to obtain v and w. The signed torsion is
`atan2((u × v) · w, v · w)` radians. A middle bond or projected arm with norm below
1e−10 Å is degenerate and invalid. All three named consecutive covalent bonds must
exist. Phi is C_previous,N,CA,C; psi N,CA,C,N_next; omega CA,C,N_next,CA_next. Actual
same-chain C–N connectivity determines peptide neighbors; branching is unavailable.
The explicit chi table follows named canonical paths; absent residues/atoms and chain
breaks are masked. Terminal name choices are documented, not symmetry-averaged angles.

Runtime bond angles use `acos(clip((xi−xj)·(xk−xj)/(|xi−xj||xk−xj|),−1,1))` for
covalent triplets i–j–k, i<k, in radians. Degenerate arms are omitted. Runtime N–CA–C
frames use e1=normalize(C−CA), e3=normalize(e1×(N−CA)), e2=e3×e1, as row axes. Both
backbone bonds must exist. The frame is orthonormal and rotates with coordinates;
under reflection the cross-product axis is a pseudovector, not a polar vector.
Neither frames nor triplets are persisted by default.

For surface point x, atom a, outward unit normal n, let v=a−x. Candidate distance
is d=||v||, signed normal offset z=v·n, and tangential distance rho=||v−zn||. The
last expression avoids cancellation in sqrt(d²−z²). KD-tree search uses float64
coordinates, inclusive physical cutoff, then float32 distance/ID sorting. This is
the historical transfer writer's convention and is regression-tested at 6 Å/K32.

Existing tabulated inputs need no new per-atom arrays: Gemmi supplies element radii
(Å) and mass; `PhysicalEmbeddingInitializer` derives Z, sqrt(Z), mass, radii and
fixed residue hydropathy/charge-class/polarity/aromaticity/volume/donor/acceptor
tendencies at runtime. These are tabulated priors, not measured residue charge;
its UNK row is mean-imputed before projection. No partial charges are inferred.

Biopython already supplies maintained Shrake–Rupley SASA (accessible area in Å²).
Integrating it is deferred because its atom-identity conversion, radii, hydrogen,
component and probe policies need separate validation. The existing normalized
surface area weights are not absolute per-atom SASA. No electrostatic potential,
dynamics, force field, target atom, target annotation or neural embedding is produced.

## 6. Measured pilot and initial storage choice

Command (local diagnostic only):

```bash
python scripts/benchmark_structural_sidecars.py --design data/dna/design --output costs.json
```

The [raw measurement JSON](structural_enrichment_benchmark.json) records all six
radii (6/8/10/12/14/16 Å), quantiles, fractions exceeding 32/64/96/128, raw/compressed
bytes, discarded pairs, build/write/read/dense-conversion times and memory observations.
Three chains were selected by size near 300/2000/10000 retained heavy atoms, not
prediction performance or labels. The two smaller audit labels are positive and
the largest negative; labels were recorded **after** geometry, never used in building it.
This is a size pilot, not a representative sample of all phenotypes or acquisition sources.

| Chain | Atoms / points | Radius | Mean candidates / p99 | Points above K96 | CSR compressed | Build / read / dense conversion (s) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1PFI_A | 322 / 4379 | 12 Å | 75.1 / 102 | 6.2% | 3.76 MiB | .55 / .021 / .014 |
| 2NRA_C | 2000 / 14411 | 12 Å | 121.9 / 302.9 | 63.1% | 14.34 MiB | 2.85 / .076 / .051 |
| 3UGJ_A | 9884 / 43739 | 12 Å | 161.9 / 367 | 83.4% | 47.25 MiB | 10.07 / .235 / .124 |

Build measurements include tracemalloc instrumentation. Tracked peak allocations at
12 Å were 11.5/42.4/140.4 MiB; these are **not total process peak RAM**. The JSON also
records Linux process cumulative peak RSS, which includes imports, previous radii
and source-surface construction and cannot be attributed to one radius. This avoids
presenting incomplete allocation tracking as isolated full-RAM measurements.

CSR and padded dense files compressed similarly at K96: 12 Å dense payloads were
3.82/14.39/47.22 MiB. The reason to retain CSR is exact variable-length/empty-row
semantics, not a claim that it always saves compressed bytes or accelerates GPU kernels.
For fully populated rows CSR raw bytes approach 16P+16M, while padded dense geometry,
indices and mask occupy 17MK bytes. Compressed size depends on geometry/density.

The initial **12 Å/K96** setting deliberately trades storage against support. It is
not complete coverage: at 12 Å, even K128 would be exceeded by 0%/40.3%/61.9% of
points. All omitted counts remain visible. Increasing to 14/16 Å mostly adds
discarded atoms at fixed K on dense proteins. This does not establish a universal
interaction cutoff. A future whole-radius method must choose more support explicitly
or another measured strategy instead of treating the capped list as exhaustive.

A full-protein float32 H=128 gathered-embedding tensor alone would occupy about
160/604/1979 MiB at K96; these are **estimates**, not GPU measurements or total attention
costs. Chunking and mixed precision change that budget. No future model should gather
all proteins' P×H activations unconditionally. No model or GPU was benchmarked here.

No production schema-3 NPZ was available locally. Thus this pilot measures the actual
three point geometries but does not claim measured compression ratios against full
production archives or their spectral operators. Existing-archive benchmark mode
does record base NPZ bytes and operator array bytes for that comparison when available.
The full tiny integration fixture separately verifies the real NPZ writer/resume path.
Its six-radius measurements are included under `integration_fixture_measurements`:
17 atoms, 427 points, 225,057 bytes in the complete base NPZ, and 186,204 uncompressed
bytes of diffusion arrays (eight modes). Its 12 Å/K96 candidate payload used 78,351
compressed bytes, about 34.8% of that tiny base's compressed bytes. This is a synthetic
regression structure, **not** evidence about production archive size ratios. Candidate
payload measurements exclude the small scalar sidecar metadata/fingerprint overhead.
The fixture also measures concatenating two identical validated CSR copies into
disjoint CPU namespaces: about 0.113 ms at 12 Å/K96 in this run. This is an actual
tiny-fixture conversion measurement, not a production-batch or host-to-GPU transfer
benchmark. The three real-chain pilot predates that microbenchmark and does not
claim measured batch-conversion times for the larger chains.

## 7. Activation, consumption and future embedding boundary

DNA and Zn Preprocessing expose `atomic_context=false`, `source_quality=true`,
`extended_surface_atom=false`, `extended_radius=12`, `extended_k_max=96`; the YAMLs
explicitly enable both families for new DNA @8 / Zn @2 releases. They retain base
6 Å/K32 and the original atomic graph. Both final validators check optional assets;
absent files remain legal historical input. Sidecar checkpoints include base digest,
schema/algorithm settings and source availability. Restoring a file revalidates
scientific content; LambdaForge owns concurrency, temporary files and retry behavior.
Each family's checkpoint key excludes the other family's parameters. A larger radius
therefore does not force recomputation of identical atomic context.

`StructuralEnrichment` is the public geometry-reuse Work. The example
`experiments/preprocess/enrich_structures.yaml` enriches source @6 into destination
@8 without selection, annotations or surfaces being regenerated. It imports original
assets through native output ownership and preserves member IDs, partitions, targets
and metadata. Native publication may copy bytes into a new placement; zero-copy disk
deduplication is not promised. Never run both publication routes to the same version.

Historical loaders/collators/models are intentionally unchanged. Future opt-in code
can resolve `atomic_context`/`extended_surface_atom` from DatasetIndex, validate
with `AtomicContext.validate`/`ExtendedSurfaceAtom.validate`, then read arrays with
pickle disabled. Per-protein atom IDs must be offset explicitly in a disjoint batch;
point pointers need their own accumulated P offset. No implicit loader activation
or second undocumented graph representation was introduced.

A future neural embedding sidecar would require entity (atom/residue), exact row
mapping, model/checkpoint/tokenizer and extraction versions, dimension/dtype,
normalization, source fingerprint, validity mask and exact base SHA-256. No such
extractor or learned representation is implemented by this task.

## 8. Changed components and technical verification

| Component | Change |
| --- | --- |
| `common/structure/AtomicContext.py`, `context_chemistry.py` | Fixed vocabularies, chemistry, masked torsions, optional source fields, runtime frames/triplets and validator |
| `common/structure/ExtendedSurfaceAtom.py` | Sparse candidate construction, audited coverage, dense-prefix conversion and complete validator |
| `common/structure/StructuralSidecars.py` | Shared native managed-file generation, independent family resume and compatible input reuse |
| `common/StructuralEnrichment.py` | Public native dataset-to-dataset geometry-reuse Work |
| `ProteinReader`, domain `Atom`, `ProteinArchive` | Optional explicit quality preservation and streaming fingerprints; base scientific schema unchanged |
| DNA `Preprocessing`, publication and `DNAValidation` | Common options/assets, independent audit and schema-description correction |
| `ZincPreprocessing`, `ZincValidation` | Same optional assets and validator, no duplicate chemistry or Zn-specific inputs |
| DNA/Zn preprocessing YAMLs, `enrich_structures.yaml` | New releases and explicit 12 Å/K96 optional families; historical training YAMLs untouched |
| `test_structural_enrichment.py`, design-configuration contract | Scientific unit fixtures, managed resume, compatibility, actual DNA/Zn publication orchestration and post-hoc Work tests |
| Benchmark script, raw JSON, bilingual READMEs and this audit | Reproducible cost diagnostic, meanings, assumptions and activation guide |

Focused tests compare all canonical chi templates against Gemmi's dihedral calculation,
check a known signed torsion, rotation/translation and frame-reflection conventions,
missing atoms, interrupted peptides, disulfides versus spatial proximity, repeated chain
names, exact naming, source quality absence/predicted-model B, deterministic CSR ordering,
radius cutoffs, empty rows, clipping, hash mismatches, exact legacy 6 Å/K32 conversion,
pickle safety, disabled-family no-I/O, native-file resume and unmodified base bytes.
Mocked output ownership verifies real DNA/Zn orchestration and post-hoc publication
without registering fake data. A loader regression supplies deliberately unreadable
optional assets and proves that the historical default inputs remain unchanged.

Verification on 2026-10-20: `ruff check .` passed; `mypy src/wisdom` passed for 181
source files; `pytest -q` finished with **903 passed, 4 skipped**. The skips concern
an absent generated production benchmark and three optional Playwright browser tests.
Both README tables of contents match all numbered headings. `git diff --check` passed.

Configuration validation, explanation and local dry-run planning cover all three new/changed
recipes. Existing DNA validation and V1a/V2 training selectors also validate. These checks
establish executability, **not** predictive improvement, cluster capacity or scientifically
complete candidate coverage. No remote Work, training, production preprocessing or Registry
publication was launched; only bounded local fixtures and the three-chain pilot were run.

## 9. References

- [IUPAC–IUB polypeptide conformation nomenclature, 1974](https://publications.iupac.org/pac/40/3/0291/index.html): backbone and side-chain torsion conventions.
- [wwPDB atomic-position fields](https://mmcif.wwpdb.org/docs/tutorials/content/atomic-description.html): atom names, alternate conformations and observed quality fields.
- [Biopython Shrake–Rupley API](https://biopython.org/docs/latest/api/Bio.PDB.SASA.html): maintained SASA implementation investigated, not integrated or claimed as WISDOM output.
