# HBAAC Round 2 Demand Forecasting - Final Submissions

This repository contains the final selected submissions and reproducibility code for the HBAAC Round 2 demand forecasting competition.

## Final selected submissions

Select these two submissions for the Private leaderboard:

| Version | File | Public score | Role |
|---|---|---:|---|
| v19 | `submissions/submission_v19_LINESEARCH_public_private_g138.csv` | 0.48881 | Best Public score; primary final submission |
| v16 | `submissions/submission_v16_LINESEARCH_public_private_g092.csv` | 0.48926 | Safer hedge with lower anti-v6 extrapolation |

Do **not** select v6, v8, v13, or v20 for final Private selection.

## Repository structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── run_all_final.py
├── VALIDATION_REPORT_final_v19_v16.md
│
├── scripts/
│   ├── 01_generate_v2_variants.py
│   ├── 02_generate_v6_to_v9.py
│   └── 03_generate_final_v16_v19.py
│
├── reference_submissions/
│   ├── submission_baseline_v2_local_public.csv
│   └── submission_v6_optimized_global.csv
│
└── submissions/
    ├── submission_v19_LINESEARCH_public_private_g138.csv
    └── submission_v16_LINESEARCH_public_private_g092.csv
```

## Reproduce final submissions

Install dependencies:

```bash
pip install -r requirements.txt
```

Run:

```bash
python run_all_final.py
```

Outputs will be written to:

```text
outputs/
  submission_v16_LINESEARCH_public_private_g092.csv
  submission_v19_LINESEARCH_public_private_g138.csv
```

## Final formula

The final files are generated from two reference submissions:

- `v2 = submission_baseline_v2_local_public.csv`
- `v6 = submission_v6_optimized_global.csv`

Formula:

```python
prediction = clip(v2 + gamma * (v2 - v6), lower=0)
```

Final gamma values:

```text
v16: gamma = 0.92
v19: gamma = 1.38
```

Rationale:

- v6 performed worse than v2 on Public leaderboard, so the v6 direction was treated as a bad direction.
- Moving in the opposite direction improved Public score.
- v19 achieved the best Public score.
- v16 is selected as the second Private candidate because it is less aggressive than v19 and therefore provides a hedge.

## Raw data

The raw competition files are not committed to this repository:

```text
train.csv
sample_submission.csv
hbaac-round2.zip
```

If the organizer wants to regenerate earlier reference submissions from raw data, place the competition files in a local `data/` folder and run:

```bash
python scripts/01_generate_v2_variants.py
python scripts/02_generate_v6_to_v9.py
```

Then use the generated v2 and v6 files as references for `scripts/03_generate_final_v16_v19.py`.

## Validation

See:

```text
VALIDATION_REPORT_final_v19_v16.md
```

for row count, column count, duplicate id check, negative forecast check, NaN check, and forecast summary.
