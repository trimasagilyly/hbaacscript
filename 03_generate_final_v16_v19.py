#!/usr/bin/env python3
"""
Generate the two final selected Kaggle submissions for HBAAC Round 2.

Final selected submissions:
- v19: gamma = 1.38, Public score 0.48881
- v16: gamma = 0.92, Public score 0.48926

Both are produced from two reference submissions:
- v2: submission_baseline_v2_local_public.csv
- v6: submission_v6_optimized_global.csv

Formula:
    final = clip(v2 + gamma * (v2 - v6), lower=0)

The anti-v6 direction was chosen because v6 performed much worse than v2 on Public
leaderboard, while controlled extrapolation in the opposite direction improved scores.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


ID_COL = "id"
FORECAST_COLS = [f"F{i}" for i in range(1, 29)]

FINAL_CONFIGS = {
    "submission_v16_LINESEARCH_public_private_g092.csv": 0.92,
    "submission_v19_LINESEARCH_public_private_g138.csv": 1.38,
}


def read_submission(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing required file: {path}")
    df = pd.read_csv(path)
    expected = [ID_COL] + FORECAST_COLS
    if list(df.columns) != expected:
        missing = [c for c in expected if c not in df.columns]
        extra = [c for c in df.columns if c not in expected]
        raise ValueError(f"{path.name} has invalid columns. Missing={missing}, extra={extra}")
    if df[ID_COL].duplicated().any():
        duplicates = df.loc[df[ID_COL].duplicated(), ID_COL].head(10).tolist()
        raise ValueError(f"{path.name} contains duplicate id values: {duplicates}")
    values = df[FORECAST_COLS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{path.name} contains NaN or infinite forecast values.")
    if (values < 0).any():
        raise ValueError(f"{path.name} contains negative forecast values.")
    return df


def generate_gamma(v2: pd.DataFrame, v6: pd.DataFrame, gamma: float) -> pd.DataFrame:
    if not v2[ID_COL].equals(v6[ID_COL]):
        raise ValueError("v2 and v6 id order differs. Align them before generating final submissions.")

    out = v2.copy()
    values = v2[FORECAST_COLS].to_numpy(dtype=float) + gamma * (
        v2[FORECAST_COLS].to_numpy(dtype=float) - v6[FORECAST_COLS].to_numpy(dtype=float)
    )
    values = np.clip(values, 0.0, None)
    out[FORECAST_COLS] = values
    return out


def validate_output(df: pd.DataFrame, reference: pd.DataFrame, name: str) -> None:
    if df.shape != reference.shape:
        raise ValueError(f"{name} shape {df.shape} differs from reference {reference.shape}.")
    if not df[ID_COL].equals(reference[ID_COL]):
        raise ValueError(f"{name} id order differs from reference.")
    if df[ID_COL].duplicated().any():
        raise ValueError(f"{name} has duplicate id values.")
    values = df[FORECAST_COLS].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} contains NaN or infinite values.")
    if (values < 0).any():
        raise ValueError(f"{name} contains negative values.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--reference-dir",
        type=Path,
        default=Path("reference_submissions"),
        help="Folder containing v2 and v6 reference submissions.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Folder where final submissions will be written.",
    )
    args = parser.parse_args()

    v2 = read_submission(args.reference_dir / "submission_baseline_v2_local_public.csv")
    v6 = read_submission(args.reference_dir / "submission_v6_optimized_global.csv")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for filename, gamma in FINAL_CONFIGS.items():
        out = generate_gamma(v2, v6, gamma)
        validate_output(out, v2, filename)
        output_path = args.output_dir / filename
        out.to_csv(output_path, index=False, float_format="%.6f")
        arr = out[FORECAST_COLS].to_numpy(dtype=float)
        print(
            f"Saved {output_path} | gamma={gamma} | "
            f"rows={len(out):,} | sum={arr.sum():,.6f} | zero_rate={(arr == 0).mean():.3%}"
        )


if __name__ == "__main__":
    main()
