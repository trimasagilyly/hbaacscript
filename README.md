# HBAAC Round 2 Demand Forecasting

This repository contains the final reproducible code and final submissions for the Kaggle qualifying round.

## Data

Do not commit competition raw data. Place the official files here before running:

```text
data/
  train.csv
  sample_submission.csv
```

## Final submissions

```text
submissions/submission_v19_LINESEARCH_public_private_g138.csv
submissions/submission_v20_FINAL_weighted_selective_g_CHECKED.csv
```

## Reproduce v19

v19 is generated from prior confirmed submissions v2 and v6:

```bash
python scripts/03_reproduce_v19_only.py --input-dir . --output outputs/submission_v19_LINESEARCH_public_private_g138.csv
```

Required input files in the working directory:

```text
sample_submission.csv
submission_baseline_v2_local_public.csv
submission_v6_optimized_global.csv
```

## Generate v20

v20 uses v19-like anti-v6 direction with selective adjustment by high-profit/active SKUs:

```bash
python scripts/04_generate_v20_weighted_selective.py --input-dir . --output outputs/submission_v20_FINAL_weighted_selective_g.csv
```

Required input files:

```text
train.csv
sample_submission.csv
submission_baseline_v2_local_public.csv
submission_v6_optimized_global.csv
```

## Validation checklist

Before submitting, verify:

- 31,944 rows
- 29 columns: id + F1..F28
- no duplicate id
- no missing id
- no negative forecasts
- no NaN values
