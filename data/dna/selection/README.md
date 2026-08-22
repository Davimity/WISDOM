# WISDOM-DNA selection

Quality verdict: **PASS_WITH_WARNINGS**. The complete portable selection contains 2140 proteins including local-evaluation reserves.

## Which files are needed?

- `catalog.csv` is the authoritative human-readable scientific table. One row stores the label, split, external sequence family, structure choice, evidence, quality measurements, and provenance for one protein chain.
- `catalog.parquet` contains the same table with typed columns for faster analysis. It is an analytical mirror, not a second source of truth.
- `identifiers.json` is the compact machine-readable membership contract. It joins each identifier to its label, split, geometric tier, sequence family, and whether it is a normal dataset member or a reserve.
- `labels.csv` is a smaller spreadsheet-friendly projection of the same membership. It is convenient for inspection and does not replace `catalog.csv`.
- `train.txt`, `val.txt`, and `test.txt` contain only the identifiers in the three model partitions. Their current sizes are 1282, 314, and 436 respectively.
- `validation_reserve.txt` and `test_reserve.txt` contain positive proteins held outside ordinary training and evaluation. They may replace a same-partition positive whose local surface annotation cannot be evaluated. Reserves are intentionally not class-balanced because they are spare localization examples, not model splits.
- `proteins.txt` is the union of main and reserve identifiers. Structural preprocessing reads this union once so a reserve is ready if annotation needs it.
- `audit.json`, `audit.md`, `statistics.csv`, and `distributions.png` are respectively the machine verdict, explained report, tidy statistics, and diagnostic figure.

## Diluted training views

Each `subsets/<percentage>/` directory is a self-contained membership view with its own filtered `catalog.csv`, TXT, compact CSV, JSON, Markdown, and figure. The percentage applies only to balanced training membership. Validation and test are identical in every view, which makes learning-curve comparisons fair. Training selections are nested and visit distinct 30%-identity sequence families before taking repeated family members.

Sequence clustering is a leakage barrier and a family-diversity mechanism. It does not establish coverage of every biochemical protein function; read `audit.md` for the measured breadth and the remaining source/label confounding warning.
