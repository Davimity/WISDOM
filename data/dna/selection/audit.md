# WISDOM-DNA selection audit: complete

**Verdict:** `PASS_WITH_WARNINGS`. **Members:** 2140.

A failed check means this view must not be used for training or evaluation. A warning does not indicate file corruption, but it identifies a scientific bias that must be considered when interpreting model performance.

## Safety checks

| Check | Result | Meaning | Observed |
|---|---:|---|---|
| `unique_identifiers` | **PASS** | A protein identifier occurs at most once in this view. | `{"duplicates": 0}` |
| `proteins_txt_agreement` | **PASS** | proteins.txt is the exact union that this view sends to preprocessing. | `{"expected": 2140, "observed": 2140}` |
| `identifiers_json_agreement` | **PASS** | identifiers.json contains one machine-readable record per labels.csv row. | `{"expected": 2140, "observed": 2140}` |
| `train_txt_agreement` | **PASS** | train.txt agrees exactly with labels.csv membership. | `{"expected": 1282, "observed": 1282}` |
| `val_txt_agreement` | **PASS** | val.txt agrees exactly with labels.csv membership. | `{"expected": 314, "observed": 314}` |
| `test_txt_agreement` | **PASS** | test.txt agrees exactly with labels.csv membership. | `{"expected": 436, "observed": 436}` |
| `validation_reserve_txt_agreement` | **PASS** | validation_reserve.txt agrees exactly with labels.csv membership. | `{"expected": 85, "observed": 85}` |
| `test_reserve_txt_agreement` | **PASS** | test_reserve.txt agrees exactly with labels.csv membership. | `{"expected": 23, "observed": 23}` |
| `train_class_balance` | **PASS** | Exact 1:1 parity prevents the majority class from dominating accuracy or the binary loss in this partition. | `{"negative": 641, "positive": 641, "ratio": 1.0}` |
| `val_class_balance` | **PASS** | Exact 1:1 parity prevents the majority class from dominating accuracy or the binary loss in this partition. | `{"negative": 157, "positive": 157, "ratio": 1.0}` |
| `test_class_balance` | **PASS** | Exact 1:1 parity prevents the majority class from dominating accuracy or the binary loss in this partition. | `{"negative": 218, "positive": 218, "ratio": 1.0}` |
| `no_base_identifier_leakage` | **PASS** | No protein identifier may occur in more than one split; otherwise evaluation can contain information already represented during training or model selection. | `{"examples": [], "leaked_groups": 0}` |
| `no_sequence_sha256_leakage` | **PASS** | No exact amino-acid sequence may occur in more than one split; otherwise evaluation can contain information already represented during training or model selection. | `{"examples": [], "leaked_groups": 0}` |
| `no_sequence_cluster_id_leakage` | **PASS** | No 30%-identity sequence family may occur in more than one split; otherwise evaluation can contain information already represented during training or model selection. | `{"examples": [], "leaked_groups": 0}` |
| `no_pdb_id_leakage` | **PASS** | No PDB deposition may occur in more than one split; otherwise evaluation can contain information already represented during training or model selection. | `{"examples": [], "leaked_groups": 0}` |
| `no_protein_structure_sha256_leakage` | **PASS** | No selected-chain coordinates may occur in more than one split; otherwise evaluation can contain information already represented during training or model selection. | `{"examples": [], "leaked_groups": 0}` |
| `external_test_boundary` | **PASS** | Proteins published as external test by a source never enter train or validation. | `{"violations": 0}` |

## Composition and sequence-family diversity

`cluster coverage ratio = distinct 30%-identity clusters / proteins`. A value near 1 means nearly every protein belongs to a different broad sequence family, which is desirable for breadth and reduces domination by repeated homologues. It does not prove coverage of every biochemical DNA-binding mechanism.

| Split | Negative | Positive | Negative families | Positive families | Tiers |
|---|---:|---:|---:|---:|---|
| train | 641 | 641 | 629/641 | 623/641 | `{"challenge": 83, "core": 1199}` |
| val | 157 | 157 | 154/157 | 155/157 | `{"challenge": 17, "core": 297}` |
| test | 218 | 218 | 218/218 | 217/218 | `{"challenge": 33, "core": 403}` |
| validation_reserve | 0 | 85 | 0/0 | 83/85 | `{"challenge": 6, "core": 79}` |
| test_reserve | 0 | 23 | 0/0 | 23/23 | `{"challenge": 1, "core": 22}` |

The lowest main-split family coverage is 97.2%. This is high: repeated close-family membership is uncommon, and no one broad sequence family dominates a class. Both labels contain core and challenge geometries in every main split. These are useful diversity signals, but they remain weaker than a curated functional ontology.

## Scientific warnings and limitations

- **source_label_confounding:** Positive and negative proteins come from disjoint source datasets. Class balance is exact, but a model could exploit source-specific structural or curation biases instead of DNA-binding biology. This is a scientific limitation, not split leakage.
- **Scope limitation:** A 30%-identity MMseqs2 cluster is a sequence-family proxy, not a standardized molecular-function category. The catalog has no ontology-complete function label, so functional-type coverage cannot be claimed from clustering alone.

## Reading the numeric summaries

`statistics.csv` and `audit.json` report the median (the middle observation), Q1 (25% of values are lower), and Q3 (75% are lower). The Q1-Q3 interval describes the central half without being dominated by a few extreme proteins. Sequence length is in residues; experimental resolution is in ångströms, where a smaller value generally means finer structural detail; aspect ratio measures elongation; and sequence coverage is the observed fraction of the source sequence.

`distributions.png` visualizes class parity, sequence-family breadth, sequence length, and geometric-tier representation. These plots diagnose dataset composition; they do not by themselves demonstrate model quality.
