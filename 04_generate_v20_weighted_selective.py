#!/usr/bin/env python3
"""
HBAAC Round 2 - Final v20 candidate.

Purpose:
- v18 and v19 were almost tied, so the pure global gamma line-search is likely near
  saturation.
- v20 keeps v19 as the stable base, then applies a small, weight-aware nonlinear
  adjustment:
    * high-profit active SKUs get a slightly stronger anti-v6 correction;
    * low-profit / non-positive-profit SKUs are not pushed further;
    * very tiny row-level forecasts are pruned to 0 to reduce sparse-tail false positives.

Inputs expected in --input-dir:
    train.csv
    sample_submission.csv
    submission_baseline_v2_local_public.csv
    submission_v6_optimized_global.csv

Output:
    submission_v20_FINAL_weighted_selective_g.csv

Formula:
    base_direction = v2 - v6
    v19 = clip(v2 + 1.38 * base_direction, 0, inf)

    For each SKU:
        if profit <= 0:
            gamma = 1.38
        elif top 5% positive profit and sold in last 180 days:
            gamma = 1.58
        elif top 20% positive profit and sold in last 180 days:
            gamma = 1.50
        elif sold in last 90 days:
            gamma = 1.43
        else:
            gamma = 1.38

    v20_raw = clip(v2 + gamma * base_direction, 0, inf)

    Sparse pruning:
        If a row total is tiny, set the whole 28-day row to 0.
        This is intentionally conservative and mostly affects low-signal SKU/window rows.

Notes:
- No randomness.
- No model training.
- Exact Kaggle submission format.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ID_COL = "id"
FORECAST_COLS = [f"F{i}" for i in range(1, 29)]
GAMMA_V19 = 1.38


def parse_number_series(s: pd.Series) -> pd.Series:
    """Parse numeric columns that may use Vietnamese decimal comma."""
    if pd.api.types.is_numeric_dtype(s):
        return pd.to_numeric(s, errors="coerce").fillna(0.0)
    return (
        s.astype(str)
        .str.replace(".", "", regex=False)  # safe for possible thousands separator
        .str.replace(",", ".", regex=False)
        .replace({"nan": "0", "None": "0", "": "0"})
        .pipe(pd.to_numeric, errors="coerce")
        .fillna(0.0)
    )


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


def extract_sku(submission_id: pd.Series) -> pd.Series:
    return submission_id.str.replace("_validation", "", regex=False).str.replace("_evaluation", "", regex=False)


def build_sku_profile(train_path: Path) -> pd.DataFrame:
    if not train_path.exists():
        raise FileNotFoundError(f"Missing file: {train_path}")

    train = pd.read_csv(train_path)
    required = ["Date", "ItemCode", "Quantity", "SalesAmount", "Cost Amount"]
    missing = [c for c in required if c not in train.columns]
    if missing:
        raise ValueError(f"train.csv missing required columns: {missing}")

    train["Date"] = pd.to_datetime(train["Date"], errors="coerce")
    train["Quantity"] = pd.to_numeric(train["Quantity"], errors="coerce").fillna(0.0)
    train["SalesAmount"] = parse_number_series(train["SalesAmount"])
    train["Cost Amount"] = parse_number_series(train["Cost Amount"])

    train_end = train["Date"].max()
    train["positive_qty"] = train["Quantity"].clip(lower=0)

    profile = train.groupby("ItemCode", as_index=False).agg(
        profit=("SalesAmount", lambda x: float(x.sum())),
        cost=("Cost Amount", lambda x: float(x.sum())),
        last_sale_date=("Date", lambda x: x[train.loc[x.index, "Quantity"] > 0].max()),
        positive_sale_days=("Date", lambda x: train.loc[x.index].loc[train.loc[x.index, "Quantity"] > 0, "Date"].nunique()),
    )
    # Correct profit = sales - cost after aggregation.
    profile["profit"] = profile["profit"] - profile["cost"]
    profile = profile.drop(columns=["cost"])
    profile["profit_pos"] = profile["profit"].clip(lower=0.0)
    profile["days_since_last_sale"] = (train_end - profile["last_sale_date"]).dt.days
    profile["days_since_last_sale"] = profile["days_since_last_sale"].fillna(9999).astype(int)

    positive = profile.loc[profile["profit_pos"] > 0, "profit_pos"]
    if len(positive) > 0:
        q80 = positive.quantile(0.80)
        q95 = positive.quantile(0.95)
    else:
        q80 = q95 = np.inf
    profile["profit_q80"] = q80
    profile["profit_q95"] = q95
    return profile


def make_v20(input_dir: Path, output_path: Path) -> None:
    sample = read_submission(input_dir / "sample_submission.csv")
    v2 = align_to_sample(sample, read_submission(input_dir / "submission_baseline_v2_local_public.csv"), "v2")
    v6 = align_to_sample(sample, read_submission(input_dir / "submission_v6_optimized_global.csv"), "v6")
    profile = build_sku_profile(input_dir / "train.csv")

    meta = sample[[ID_COL]].copy()
    meta["ItemCode"] = extract_sku(meta[ID_COL])
    meta = meta.merge(profile, on="ItemCode", how="left")
    meta["profit_pos"] = meta["profit_pos"].fillna(0.0)
    meta["days_since_last_sale"] = meta["days_since_last_sale"].fillna(9999).astype(int)
    meta["positive_sale_days"] = meta["positive_sale_days"].fillna(0).astype(int)

    # Start with v19 gamma.
    gamma = np.full(len(sample), GAMMA_V19, dtype=float)

    active_180 = meta["days_since_last_sale"].to_numpy() <= 180
    active_90 = meta["days_since_last_sale"].to_numpy() <= 90
    profit_pos = meta["profit_pos"].to_numpy()
    q80 = float(profile["profit_q80"].iloc[0]) if len(profile) else np.inf
    q95 = float(profile["profit_q95"].iloc[0]) if len(profile) else np.inf

    # Weight-aware selective push.
    gamma[(profit_pos > 0) & active_90] = 1.43
    gamma[(profit_pos >= q80) & active_180] = 1.50
    gamma[(profit_pos >= q95) & active_180] = 1.58

    v2_values = v2[FORECAST_COLS].to_numpy(dtype=float)
    v6_values = v6[FORECAST_COLS].to_numpy(dtype=float)
    raw = v2_values + gamma[:, None] * (v2_values - v6_values)
    raw = np.clip(raw, 0.0, None)

    # Nonlinear sparse-tail pruning. Keep this small to avoid the v13-style global mistake.
    row_sum = raw.sum(axis=1)
    low_profit = profit_pos <= 0
    very_tiny = row_sum < 0.08
    tiny_low_signal = (row_sum < 0.18) & (meta["positive_sale_days"].to_numpy() <= 3)
    raw[very_tiny | (low_profit & tiny_low_signal)] = 0.0

    out = sample.copy()
    out[FORECAST_COLS] = raw

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
    print("Gamma summary:")
    print(pd.Series(gamma).describe().to_string())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("submission_v20_FINAL_weighted_selective_g.csv"),
    )
    args = parser.parse_args()
    make_v20(args.input_dir, args.output)


if __name__ == "__main__":
    main()
