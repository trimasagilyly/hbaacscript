#!/usr/bin/env python3
"""
HBAAC Round 2 - Reproduce v19 submission only.

This script reproduces the submitted v19 file from the two already-generated
reference submissions used in the line-search:

    v2: submission_baseline_v2_local_public.csv
    v6: submission_v6_optimized_global.csv

Formula:
    v19 = clip(v2 + 1.38 * (v2 - v6), lower=0)

Why this is the exact reproducibility script:
- The actual v19 submitted to Kaggle was generated as a line-search extrapolation
  between the known good v2 and known bad v6 directions.
- This script intentionally contains no extra model training, no randomness, and no
  environment-specific dependencies besides pandas/numpy.

Expected inputs in --input-dir:
    sample_submission.csv
    submission_baseline_v2_local_public.csv
    submission_v6_optimized_global.csv

Output:
    submission_v19_LINESEARCH_public_private_g138.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "id"
FORECAST_COLS = [f"F{i}" for i in range(1, 29)]
GAMMA_V19 = 1.38


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    df = pd.read_csv(path)
    expected = [ID_COL] + FORECAST_COLS
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if df[ID_COL].duplicated().any():
        dupes = df.loc[df[ID_COL].duplicated(), ID_COL].head(10).tolist()
        raise ValueError(f"{path.name} contains duplicate id values, examples: {dupes}")
    return df[expected].copy()


def align_to_sample(sample: pd.DataFrame, df: pd.DataFrame, name: str) -> pd.DataFrame:
    aligned = sample[[ID_COL]].merge(df, on=ID_COL, how="left", validate="one_to_one")
    if aligned[FORECAST_COLS].isna().any().any():
        missing_ids = aligned.loc[aligned[FORECAST_COLS].isna().any(axis=1), ID_COL].head(10).tolist()
        raise ValueError(f"{name} is missing ids from sample_submission, examples: {missing_ids}")
    return aligned


def make_v19(input_dir: Path, output_path: Path) -> None:
    sample = read_submission(input_dir / "sample_submission.csv")
    v2 = align_to_sample(sample, read_submission(input_dir / "submission_baseline_v2_local_public.csv"), "v2")
    v6 = align_to_sample(sample, read_submission(input_dir / "submission_v6_optimized_global.csv"), "v6")

    out = sample.copy()
    values = v2[FORECAST_COLS].to_numpy(dtype=float) + GAMMA_V19 * (
        v2[FORECAST_COLS].to_numpy(dtype=float) - v6[FORECAST_COLS].to_numpy(dtype=float)
    )
    values = np.clip(values, 0.0, None)

    out[FORECAST_COLS] = values

    # Final Kaggle-format checks.
    if out[ID_COL].duplicated().any():
        raise ValueError("Output has duplicate id values.")
    if set(out[ID_COL]) != set(sample[ID_COL]):
        raise ValueError("Output id set differs from sample_submission.")
    if (out[FORECAST_COLS] < 0).any().any():
        raise ValueError("Output contains negative forecasts.")
    if out.shape != sample.shape:
        raise ValueError(f"Output shape {out.shape} differs from sample shape {sample.shape}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path, index=False)
    print(f"Saved: {output_path}")
    print(f"Rows: {len(out):,}; forecast sum: {out[FORECAST_COLS].to_numpy().sum():,.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_v19_LINESEARCH_public_private_g138.csv"),
    )
    args = parser.parse_args()
    make_v19(args.input_dir, args.output)


if __name__ == "__main__":
    main()
