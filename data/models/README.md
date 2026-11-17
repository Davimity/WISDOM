# Staged external protein checkpoints

These are frozen external-model inputs, not WISDOM weights, dataset geometry, or extracted
embeddings. The directories were downloaded explicitly at the researcher's request. Works never
download models automatically. Binary model directories are ignored by Git; the small manifests
remain versionable. Cloning this repository alone therefore does not install the checkpoints.

| Directory | Official model | Immutable upstream revision | Weight bytes |
|---|---|---|---:|
| `esm2-35m/` | [facebook/esm2_t12_35M_UR50D](https://huggingface.co/facebook/esm2_t12_35M_UR50D) | `6fbf070e65b0b7291e7bbcd451118c216cff79d8` | 136,008,798 |
| `saprot-35m/` | [westlake-repl/SaProt_35M_AF2](https://huggingface.co/westlake-repl/SaProt_35M_AF2) | `316cd4017d29f4657b959365f24b57f1ee278912` | 136,862,061 |
| `esmc-300m/` | [biohub/ESMC-300M](https://huggingface.co/biohub/ESMC-300M) | `e4bac860f0c502cd80f3aeac3d0ce6524c6627cb` | 1,332,032,536 |

The total is approximately 1.605 GB (1.495 GiB), plus small config/tokenizer/model-card files.
Only one usable weight format is retained per model: ESM-2/ESM-C safetensors, and SaProt's official
Transformers `pytorch_model.bin`. Duplicate TensorFlow/native `.pt` variants and remote Python
implementations were not downloaded. SaProt is loaded with PyTorch's restricted weights-only
loader, not unrestricted pickle execution; no weight conversion is performed.

`checkpoints.json` records repository identities, exact revisions, files, sizes and SHA-256 values.
The three weight hashes were verified against the upstream Hugging Face LFS SHA-256 values;
configurations also match the requested 480/12-layer or 960/30-layer variants. This verifies
download integrity and declared architecture. A separate real CPU test now verifies loading,
short-sequence residue-vector extraction and static input tables for all three pinned models.
`external-test-config.json` supplies the existing opt-in real-model tests with project-relative
paths. Run tests from the repository root after installing the current project dependencies:

```bash
WISDOM_EXTERNAL_TEST_CONFIG=data/models/external-test-config.json pytest -q tests/test_universal_assets.py -k real_external_model
```

All three models use native Transformers loaders (version ≥5.20.2) and safetensors from the normal
project dependencies. ESM-C uses `EsmcForMaskedLM`/`EsmcTokenizer`; it does not require the separate
`esm` SDK with its different Torch constraints. ESM-2/SaProt use the native masked-language-model
loader's encoder, without creating an untrained pooling head. Loaders are local-only and reject
remote Python code. Update the local installation with `./install.sh`, or bootstrap the intended
cluster with `lf clusters bootstrap <cluster> --project .`; do not mutate a running study's environment.

The old `EvolutionaryScale/esmc-300m-2024-12` URL now redirects to a legacy-format Biohub repository
whose own model card recommends `biohub/ESMC-300M`. The staged model uses that official current
safetensors format, not the incompatible legacy `.pth`/empty-config format. Record the exact
revision above in any experiment rather than treating the old and new checkpoints as identical.

The retained model cards declare MIT for ESM-2 and SaProt, and MIT plus third-party terms for the
selected ESM-C release. Review the pinned card and its linked notices for your intended use;
different revisions can have different terms. ESM-C's card links the
[official third-party notice](https://github.com/Biohub/esm/blob/main/THIRD_PARTY_NOTICE.md).
No gated-access form or licence-acceptance action was required or submitted.

The active DNA recipe at `experiments/preprocess/dna/dna_preprocess.yaml` requests all three models:

```yaml
protein_models: {file: ../../../data/models}  # Declare the verified local/remote-mirrored three-model bundle.
# Preprocessing.run default: protein_models=null. A supplied bundle requires exactly esm2/esmc/saprot,
# uses final-layer FP32 extraction on CPU in disjoint <=1022-residue fragments, and writes both the
# contextual residue vectors and canonical static input tables for every model. This binding is
# mutually exclusive with the legacy single-model external_model/external_checkpoint options.
```

`ProteinModelAssets` checks the complete file inventory/hash, then preflights each tokenizer and
model before geometry. It loads only one model at a time, runs one member at a time and records
native file checkpoints for every model/member/family. Geometry workers remain independent.
Static tables contain actual input weights for possible later initialization, not averages of
contextual protein outputs; no current WISDOM encoder is initialized as a side effect.

The six required names are `external_embeddings_{esm2,esmc,saprot}` and
`static_embedding_table_{esm2,esmc,saprot}`. They join the ten structural families before final
scientific validation and publication. SaProt requires official Foldseek 3Di; absent structural
tokens remain masked, including an entirely unsupported member. They are never fabricated.
The bundle's pinned identities/revisions and every file's hash enter scientific dependencies.

Ignored files are not supplied merely by cloning Git. LF automatically bundles inputs only up to
10 MiB; this 1.5 GiB bundle must be copied once to `data/models/` in the remote project mirror,
including the manifests/configurations. Bootstrap installs dependencies, not these checkpoint
bytes. LF verifies exact mirrored content and fails before computation if it differs. Choose
an unpublished DatasetVersion for changed outputs. The complete dataset/cluster extraction and
publication have not been launched here: only small CPU integration tests ran. All-model CPU
extraction has a real cost and no production-duration estimate has been measured.
