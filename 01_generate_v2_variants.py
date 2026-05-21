# HBAAC Round 2 - Conservative demand forecasting variants
# Generates multiple submission CSV files.
#
# Public score of v1 baseline: 0.49550.
# v2+ are more conservative based on local backtesting:
# - inactive SKU cutoff reduced to 30 days
# - stronger shrink for sparse SKUs
# - Sunday forecast = 0
# - profit <= 0 forecast = 0
# - optional yearly lag blend for private/evaluation window

from pathlib import Path
import zipfile
import numpy as np
import pandas as pd


def find_input_files():
    search_roots = [
        Path("/kaggle/input"),
        Path("/content"),
        Path("."),
        Path("/mnt/data"),
    ]

    for root in search_roots:
        if not root.exists():
            continue
        train_files = list(root.rglob("train.csv"))
        sample_files = list(root.rglob("sample_submission.csv"))
        for train_path in train_files:
            for sample_path in sample_files:
                if train_path.parent == sample_path.parent:
                    return ("csv", train_path, sample_path)

    zip_names = ["hbaac-round2.zip", "hbaac_round2.zip"]
    for root in search_roots:
        if not root.exists():
            continue
        for zip_name in zip_names:
            matches = list(root.rglob(zip_name))
            if matches:
                return ("zip", matches[0], matches[0])

    raise FileNotFoundError(
        "Không tìm thấy train.csv/sample_submission.csv hoặc hbaac-round2.zip. "
        "Hãy upload data vào Kaggle/Colab hoặc đặt cùng thư mục với notebook/script."
    )


def read_inputs():
    kind, train_path, sample_path = find_input_files()
    if kind == "csv":
        print(f"Reading train: {train_path}")
        print(f"Reading sample: {sample_path}")
        train = pd.read_csv(train_path, low_memory=False)
        sample = pd.read_csv(sample_path)
    else:
        print(f"Reading zip: {train_path}")
        with zipfile.ZipFile(train_path) as z:
            train = pd.read_csv(z.open("train.csv"), low_memory=False)
            sample = pd.read_csv(z.open("sample_submission.csv"))
    return train, sample


def parse_decimal_series(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def build_matrices(train: pd.DataFrame, sample: pd.DataFrame):
    train = train.copy()
    train["Date"] = pd.to_datetime(train["Date"])

    sku_order = (
        sample["id"]
        .astype(str)
        .str.replace(r"_(validation|evaluation)$", "", regex=True)
        .drop_duplicates()
        .tolist()
    )

    all_dates = pd.date_range(train["Date"].min(), train["Date"].max(), freq="D")
    sku_to_idx = {sku: i for i, sku in enumerate(sku_order)}
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    train["CostAmount_num"] = parse_decimal_series(train["Cost Amount"])
    train["profit_line"] = train["SalesAmount"].astype(float) - train["CostAmount_num"].astype(float)

    profit = train.groupby("ItemCode")["profit_line"].sum().reindex(sku_order).fillna(0.0)
    profit_pos = profit.clip(lower=0).to_numpy(dtype=np.float64)

    train["sales_qty_line"] = train["Quantity"].clip(lower=0)
    train["return_qty_line"] = -train["Quantity"].clip(upper=0)

    daily = (
        train.groupby(["Date", "ItemCode"], observed=True)
        .agg(
            net_qty=("Quantity", "sum"),
            sales_qty=("sales_qty_line", "sum"),
            return_qty=("return_qty_line", "sum"),
        )
        .reset_index()
    )

    n_dates, n_skus = len(all_dates), len(sku_order)

    def build_mat(col: str) -> np.ndarray:
        mat = np.zeros((n_dates, n_skus), dtype=np.float32)
        row_idx = daily["Date"].map(date_to_idx).to_numpy()
        col_idx = daily["ItemCode"].map(sku_to_idx).to_numpy()
        valid = pd.notna(row_idx) & pd.notna(col_idx)
        mat[row_idx[valid].astype(np.int32), col_idx[valid].astype(np.int32)] = (
            daily.loc[valid, col].to_numpy(dtype=np.float32)
        )
        return mat

    net_mat = build_mat("net_qty")
    return train, sku_order, all_dates, net_mat, profit_pos


def compute_days_since_positive(hist: np.ndarray) -> np.ndarray:
    pos = hist > 0
    any_pos = pos.any(axis=0)
    rev_idx = np.argmax(pos[::-1], axis=0).astype(np.int32)
    days_since = rev_idx
    days_since[~any_pos] = 10**9
    return days_since


def precompute_feature(
    mat: np.ndarray,
    dates: pd.DatetimeIndex,
    windows=(7, 14, 28, 56, 84, 90, 112, 180, 365),
    dow_ns=(4, 8, 12, 16, 24),
    recent_windows=(90, 180, 365),
    lags=(7, 14, 28, 56, 364, 365, 371),
):
    hist = mat
    n_days, n_skus = hist.shape

    means = {
        w: hist[max(0, n_days - w):].mean(axis=0).astype(np.float32)
        for w in windows
    }

    sale_days = {
        w: (hist[max(0, n_days - w):] > 0).sum(axis=0).astype(np.float32)
        for w in recent_windows
    }

    days_since = compute_days_since_positive(hist)

    hist_dow = dates.dayofweek.to_numpy()
    future_dates = pd.date_range(dates[-1] + pd.Timedelta(days=1), periods=56, freq="D")
    future_dow = future_dates.dayofweek.to_numpy()

    dow_arrs = {}
    for dn in dow_ns:
        by_dow = []
        for dow in range(7):
            idx = np.where(hist_dow == dow)[0]
            if len(idx) > dn:
                idx = idx[-dn:]
            arr = hist[idx].mean(axis=0).astype(np.float32) if len(idx) else np.zeros(n_skus, dtype=np.float32)
            by_dow.append(arr)
        dow_arrs[dn] = np.vstack([by_dow[d] for d in future_dow]).astype(np.float32)

    lag_arrs = {}
    for lag in lags:
        rows = []
        for h in range(1, 57):
            # Future row index would be n_days - 1 + h.
            ridx = (n_days - 1 + h) - lag
            if 0 <= ridx < n_days:
                rows.append(hist[ridx])
            else:
                rows.append(np.zeros(n_skus, dtype=np.float32))
        lag_arrs[lag] = np.vstack(rows).astype(np.float32)

    return {
        "future_dates": future_dates,
        "means": means,
        "sale_days": sale_days,
        "days_since": days_since,
        "dow_arrs": dow_arrs,
        "lag_arrs": lag_arrs,
        "zero_sunday_mask": (future_dow == 6)[:, None],
    }


def make_forecast(feature: dict, profit_pos: np.ndarray, params: dict) -> np.ndarray:
    n_skus = len(profit_pos)

    raw = np.zeros(n_skus, dtype=np.float32)
    for w, c in zip(params["windows"], params["coeffs"]):
        raw += np.float32(c) * feature["means"][w]

    pred = np.tile(raw[None, :], (56, 1)).astype(np.float32)

    dow_coeff = params.get("dow_coeff", 0.0)
    if dow_coeff:
        pred += np.float32(dow_coeff) * feature["dow_arrs"][params.get("dow_n", 8)]

    for lag, coeff in params.get("lag_coeffs", {}).items():
        pred += np.float32(coeff) * feature["lag_arrs"][lag]

    if params.get("zero_sunday", True):
        pred[feature["zero_sunday_mask"].ravel()] = 0.0

    # Theo metric mô tả: SKU profit âm được weight 0.
    pred[:, profit_pos <= 0] = 0.0

    inactive_days = params.get("inactive_days", 30)
    pred[:, feature["days_since"] > inactive_days] = 0.0

    sparse_req = params.get("sparse_req", 100)
    recent_window = params.get("recent_window", 180)
    confidence = np.minimum(1.0, feature["sale_days"][recent_window] / float(sparse_req)).astype(np.float32)
    pred *= confidence[None, :]

    scale = params.get("scale", 1.0)
    pred *= np.float32(scale)

    return np.clip(pred, 0, None)


def make_submission(sample: pd.DataFrame, sku_order: list[str], pred56: np.ndarray, output_path: str):
    fcols = [f"F{i}" for i in range(1, 29)]

    sub = sample[["id"]].copy()
    for c in fcols:
        sub[c] = 0.0

    val_pred = pd.DataFrame(pred56[:28].T, index=sku_order, columns=fcols)
    eval_pred = pd.DataFrame(pred56[28:].T, index=sku_order, columns=fcols)

    ids = sub["id"].astype(str)
    sku_from_id = ids.str.replace(r"_(validation|evaluation)$", "", regex=True)
    is_val = ids.str.endswith("_validation")
    is_eval = ids.str.endswith("_evaluation")

    sub.loc[is_val, fcols] = val_pred.loc[sku_from_id[is_val].values].to_numpy()
    sub.loc[is_eval, fcols] = eval_pred.loc[sku_from_id[is_eval].values].to_numpy()

    assert sub.shape == sample.shape
    assert list(sub.columns) == list(sample.columns)
    assert sub["id"].is_unique
    assert set(sub["id"]) == set(sample["id"])

    arr = sub[fcols].to_numpy()
    assert np.isfinite(arr).all()
    assert (arr >= 0).all()

    sub.to_csv(output_path, index=False, float_format="%.6f")
    return sub


PARAMS = {
    # Recommended next upload after v1.
    "submission_baseline_v2_local_public.csv": {
        "windows": (7, 28, 56, 180),
        "coeffs": (0.20, 0.35, 0.20, 0.05),
        "dow_coeff": 0.20,
        "dow_n": 24,
        "inactive_days": 30,
        "sparse_req": 100,
        "recent_window": 180,
        "zero_sunday": True,
        "scale": 1.05,
        "lag_coeffs": {},
    },

    # More private/evaluation-oriented: includes yearly lag.
    "submission_baseline_v3_private_lag.csv": {
        "windows": (7, 28, 56, 180),
        "coeffs": (0.15, 0.30, 0.20, 0.05),
        "dow_coeff": 0.20,
        "dow_n": 12,
        "inactive_days": 30,
        "sparse_req": 100,
        "recent_window": 180,
        "zero_sunday": True,
        "scale": 1.00,
        "lag_coeffs": {364: 0.05, 365: 0.05},
    },

    # Slightly higher version of v3.
    "submission_baseline_v4_lag_scaled.csv": {
        "windows": (7, 28, 56, 180),
        "coeffs": (0.15, 0.30, 0.20, 0.05),
        "dow_coeff": 0.20,
        "dow_n": 12,
        "inactive_days": 30,
        "sparse_req": 100,
        "recent_window": 180,
        "zero_sunday": True,
        "scale": 1.10,
        "lag_coeffs": {364: 0.05, 365: 0.05},
    },

    # Smooth conservative alternative without yearly lag.
    "submission_baseline_v5_smooth_private.csv": {
        "windows": (28, 56, 180),
        "coeffs": (0.25, 0.35, 0.20),
        "dow_coeff": 0.20,
        "dow_n": 24,
        "inactive_days": 30,
        "sparse_req": 100,
        "recent_window": 180,
        "zero_sunday": True,
        "scale": 1.00,
        "lag_coeffs": {},
    },
}


def main():
    train, sample = read_inputs()
    train, sku_order, all_dates, net_mat, profit_pos = build_matrices(train, sample)
    feature = precompute_feature(net_mat, all_dates)

    print(f"Train date range: {all_dates[0].date()} -> {all_dates[-1].date()}")
    print(f"Forecast date range: {feature['future_dates'][0].date()} -> {feature['future_dates'][-1].date()}")
    print(f"SKUs: {len(sku_order):,}")

    for filename, params in PARAMS.items():
        pred = make_forecast(feature, profit_pos, params)
        sub = make_submission(sample, sku_order, pred, filename)
        fcols = [f"F{i}" for i in range(1, 29)]
        arr = sub[fcols].to_numpy()
        print(
            f"Saved {filename} | "
            f"min={arr.min():.6f}, max={arr.max():.6f}, mean={arr.mean():.6f}, zero_rate={(arr == 0).mean():.3%}"
        )


if __name__ == "__main__":
    main()
