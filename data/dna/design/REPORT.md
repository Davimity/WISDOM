# WISDOM-DNA dataset-design report

## 1. Verdict

**PASS.** Every selected protein has one explicit evidence label, every transitive sequence/structure leakage group stays inside one split, and every training dilution contains complete groups only.

The canonical dataset contains **1814 proteins**: 907 DNA-binding positives and 907 curated negatives. A positive has verified heavy-atom contact with DNA in its declared biological assembly. A negative comes from explicit experimental benchmark evidence; absence of DNA in a PDB entry was never treated as a negative label.

## 2. What the selection did

1. It verified the frozen label, chain, assembly, sequence, and structural coordinates.
2. It compared all RAW candidates with MMseqs2 (sequence) and Foldseek (3D structure).
3. It formed transitive leakage groups: if A resembles B and B resembles C, all three remain together even when A and C are not directly linked.
4. It clustered label-free physical descriptors with HDBSCAN. HDBSCAN may mark unusual proteins as noise instead of forcing them into a misleading phenotype.
5. It balanced the two classes and assigned complete groups to train, validation, and test.
6. It created nested train-only dilutions; validation and test never change.

## 3. Canonical splits

| Split | Total | Positive | Negative | Leakage groups |
|---|---:|---:|---:|---:|
| train | 1269 | 634 | 635 | 1256 |
| validation | 272 | 136 | 136 | 195 |
| test | 273 | 137 | 136 | 196 |

Exact 50/50 balance inside every split can be impossible because a leakage group is indivisible. The important hard result is zero cross-split groups; small count deviations are preferable to homologous leakage.

## 4. Physical phenotypes

Global clustering used 4248 complete records and found 51 non-noise clusters; its noise fraction is 64.4%. Interface clustering used 3245 verified positives and found 2 clusters, with 2.9% noise.

Here, *noise* does not mean corrupt data. It means that HDBSCAN did not find enough nearby proteins to claim a stable density-based family. Phenotypes are used only to preserve physical variety; they do not define DNA-binding labels.

## 5. Training dilutions

- `train-10` contains 127 training proteins (64 positive, 63 negative) in 127 complete leakage groups.
- `train-25` contains 317 training proteins (159 positive, 158 negative) in 317 complete leakage groups.
- `train-50` contains 634 training proteins (317 positive, 317 negative) in 634 complete leakage groups.
- `train-75` contains 952 training proteins (476 positive, 476 negative) in 952 complete leakage groups.
- `train-100` contains 1269 training proteins (634 positive, 635 negative) in 1256 complete leakage groups.

The subsets are nested: a protein present at a smaller fraction remains present at every larger fraction. Their fixed validation and test SHA-256 values are recorded in `dilution-audit.json`.

## 6. Warnings and file guide

- the largest RAW leakage group contains 6.0% of candidates
- global_phenotype:G015 occurs in 3 leakage groups but is absent from test; phenotype matching is a soft split objective
- global_phenotype:G020 occurs in 3 leakage groups but is absent from validation; phenotype matching is a soft split objective
- global_phenotype:G021 occurs in 3 leakage groups but is absent from validation; phenotype matching is a soft split objective
- global_phenotype:G040 occurs in 3 leakage groups but is absent from test; phenotype matching is a soft split objective
- global_phenotype:G042 occurs in 3 leakage groups but is absent from test; phenotype matching is a soft split objective

`catalog.csv` is the authoritative canonical table. The `*-labelled.txt` files are compact `RCSB_CHAIN<TAB>0|1` views. The three files below `preprocessing/` add the exact assembly, copy, contact, leakage, phenotype, and dilution metadata required to reproduce NPZ files without staging the complete design. `catalog-all.csv` preserves excluded RAW evidence. `clusters/` contains raw and thresholded similarity evidence. JSON audit files carry machine-readable versions of the counts summarized here.

Selection retained 1814 proteins. The split method was `greedy_whole_group_stratification`.
