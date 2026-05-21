# HBAAC Round 2 - v6/v7/v8/v9 forecasting pipeline
# Creates improved submission variants after baseline v2 public score 0.49334.
#
# Variants:
# - submission_v6_optimized_global.csv: best local global baseline.
# - submission_v7_private_selector.csv: same public window as v6, private/evaluation window uses conservative SKU selector.
# - submission_v8_public_ml_private_safe.csv: small ML blend on public/validation window, private/evaluation same as v7.
# - submission_v9_public_ml_global_private.csv: small ML blend on public/validation window, private/evaluation same as v6.
#
# Required files: train.csv + sample_submission.csv, or hbaac-round2.zip containing both.
# Optional library: lightgbm is used for v8/v9. If unavailable, install it or use v6/v7 only.

from pathlib import Path
import zipfile
import gc
import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
except Exception:
    lgb = None


def find_input_files():
    search_roots = [Path("/kaggle/input"), Path("/content"), Path("."), Path("/mnt/data")]
    for root in search_roots:
        if not root.exists():
            continue
        train_files = list(root.rglob("train.csv"))
        sample_files = list(root.rglob("sample_submission.csv"))
        for train_path in train_files:
            for sample_path in sample_files:
                if train_path.parent == sample_path.parent:
                    return ("csv", train_path, sample_path)

    for root in search_roots:
        if not root.exists():
            continue
        for zip_name in ["hbaac-round2.zip", "hbaac_round2.zip"]:
            matches = list(root.rglob(zip_name))
            if matches:
                return ("zip", matches[0], matches[0])

    raise FileNotFoundError("Không tìm thấy train.csv/sample_submission.csv hoặc hbaac-round2.zip.")


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


def parse_decimal_series(s):
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    return pd.to_numeric(s.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def build_matrices(train, sample):
    train = train.copy()
    train["Date"] = pd.to_datetime(train["Date"])

    sku_order = (
        sample["id"].astype(str)
        .str.replace(r"_(validation|evaluation)$", "", regex=True)
        .drop_duplicates()
        .tolist()
    )

    all_dates = pd.date_range(train["Date"].min(), train["Date"].max(), freq="D")
    sku_to_idx = {sku: i for i, sku in enumerate(sku_order)}
    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    train["CostAmount_num"] = parse_decimal_series(train["Cost Amount"])
    train["profit_line"] = train["SalesAmount"].astype(float) - train["CostAmount_num"].astype(float)
    train["sales_qty_line"] = train["Quantity"].clip(lower=0)
    train["return_qty_line"] = -train["Quantity"].clip(upper=0)

    profit = train.groupby("ItemCode")["profit_line"].sum().reindex(sku_order).fillna(0.0)
    profit_pos = profit.clip(lower=0).to_numpy(dtype=np.float64)

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

    def build_mat(col):
        mat = np.zeros((n_dates, n_skus), dtype=np.float32)
        row_idx = daily["Date"].map(date_to_idx).to_numpy()
        col_idx = daily["ItemCode"].map(sku_to_idx).to_numpy()
        valid = pd.notna(row_idx) & pd.notna(col_idx)
        mat[row_idx[valid].astype(np.int32), col_idx[valid].astype(np.int32)] = (
            daily.loc[valid, col].to_numpy(dtype=np.float32)
        )
        return mat

    net_mat = build_mat("net_qty")
    sales_mat = build_mat("sales_qty")
    return_mat = build_mat("return_qty")
    return sku_order, all_dates, net_mat, sales_mat, return_mat, profit, profit_pos


def compute_scale(mat):
    diff = np.diff(mat.astype(np.float64), axis=0)
    return np.mean(diff * diff, axis=0)


def compute_days_since_positive(hist):
    pos = hist > 0
    any_pos = pos.any(axis=0)
    rev_idx = np.argmax(pos[::-1], axis=0).astype(np.int32)
    days_since = rev_idx
    days_since[~any_pos] = 10**9
    return days_since


def precompute_feature_for_cut(mat, dates, cut_idx):
    windows = (7, 14, 28, 56, 84, 90, 112, 180, 365, 730)
    dow_ns = (4, 8, 12, 16, 24, 52)
    recent_windows = (30, 60, 90, 180, 365)
    lags = (1, 7, 14, 21, 28, 56, 84, 91, 182, 364, 365, 371, 728, 729, 735)

    hist = mat[:cut_idx + 1]
    hist_dates = dates[:cut_idx + 1]
    n_days, n_skus = hist.shape
    future_dates = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=56, freq="D")
    future_dow = future_dates.dayofweek.to_numpy()

    means = {w: hist[max(0, n_days - w):].mean(axis=0).astype(np.float32) for w in windows}
    sale_days = {
        w: (hist[max(0, n_days - w):] > 0).sum(axis=0).astype(np.float32)
        for w in recent_windows
    }
    days_since = compute_days_since_positive(hist)

    hist_dow = hist_dates.dayofweek.to_numpy()
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
            ridx = cut_idx + h - lag
            if 0 <= ridx <= cut_idx:
                rows.append(mat[ridx].astype(np.float32))
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
        "future_dow": future_dow,
    }


def make_baseline_forecast(feature, profit_pos, params):
    n_skus = len(profit_pos)
    raw = np.zeros(n_skus, dtype=np.float32)
    for w, c in zip(params.get("windows", ()), params.get("coeffs", ())):
        raw += np.float32(c) * feature["means"][w]

    pred = np.tile(raw[None, :], (56, 1)).astype(np.float32)

    if params.get("dow_coeff", 0.0):
        pred += np.float32(params["dow_coeff"]) * feature["dow_arrs"][params.get("dow_n", 24)]

    for lag, coeff in params.get("lag_coeffs", {}).items():
        pred += np.float32(coeff) * feature["lag_arrs"][lag]

    if params.get("zero_sunday", True):
        pred[feature["zero_sunday_mask"].ravel()] = 0.0

    if params.get("zero_profit", True):
        pred[:, profit_pos <= 0] = 0.0

    if params.get("inactive_days", None) is not None:
        pred[:, feature["days_since"] > params["inactive_days"]] = 0.0

    if params.get("sparse_req", None) is not None:
        rw = params.get("recent_window", 180)
        conf = np.minimum(1.0, feature["sale_days"][rw] / float(params["sparse_req"])).astype(np.float32)
        conf = conf ** params.get("sparse_power", 1.0)
        pred *= conf[None, :]

    pred *= np.float32(params.get("scale", 1.0))
    return np.clip(pred, 0, None)


def make_submission(sample, sku_order, pred56, output_path):
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
    print(f"Saved {output_path} | sum={arr.sum():.3f}, mean={arr.mean():.6f}, zero_rate={(arr == 0).mean():.3%}")
    return sub


# Fixed candidate parameters from local backtesting.
V1_PARAMS = {
    "windows": (28, 56, 180), "coeffs": (0.45, 0.25, 0.10),
    "dow_coeff": 0.20, "dow_n": 8, "inactive_days": 180,
    "sparse_req": 20, "recent_window": 180, "zero_sunday": True,
    "zero_profit": True, "scale": 1.0, "lag_coeffs": {},
}

V2_PARAMS = {
    "windows": (7, 28, 56, 180), "coeffs": (0.20, 0.35, 0.20, 0.05),
    "dow_coeff": 0.20, "dow_n": 24, "inactive_days": 30,
    "sparse_req": 100, "recent_window": 180, "sparse_power": 1.0,
    "zero_sunday": True, "zero_profit": True, "scale": 1.05, "lag_coeffs": {},
}

OPT1_PARAMS = {
    "windows": (28, 56, 90, 180),
    "coeffs": (0.1427, 0.0240, 0.0506, 0.1767),
    "dow_coeff": 0.35281954708469104,
    "dow_n": 52,
    "inactive_days": 14,
    "sparse_req": 10,
    "recent_window": 30,
    "sparse_power": 2.0,
    "zero_sunday": True,
    "zero_profit": True,
    "scale": 1.220333857104341,
    "lag_coeffs": {},
}

OPT2_PARAMS = {
    "windows": (56, 180, 365),
    "coeffs": (0.1977, 0.2382, 0.2541),
    "dow_coeff": 0.06140007173809174,
    "dow_n": 12,
    "inactive_days": 14,
    "sparse_req": 120,
    "recent_window": 365,
    "sparse_power": 1.0,
    "zero_sunday": True,
    "zero_profit": True,
    "scale": 1.210105301070164,
    "lag_coeffs": {},
}

OPT3_PARAMS = {
    "windows": (14, 28, 56, 180),
    "coeffs": (0.1103, 0.0215, 0.2081, 0.0908),
    "dow_coeff": 0.30172510991427004,
    "dow_n": 52,
    "inactive_days": 21,
    "sparse_req": 40,
    "recent_window": 90,
    "sparse_power": 0.75,
    "zero_sunday": True,
    "zero_profit": True,
    "scale": 1.1965259576056848,
    "lag_coeffs": {},
}

OPT4_PARAMS = {
    "windows": (28, 56, 180, 365),
    "coeffs": (0.2206, 0.0110, 0.3353, 0.2454),
    "dow_coeff": 0.13001248082578756,
    "dow_n": 24,
    "inactive_days": 90,
    "sparse_req": 10,
    "recent_window": 30,
    "sparse_power": 1.0,
    "zero_sunday": True,
    "zero_profit": True,
    "scale": 0.8571406897034782,
    "lag_coeffs": {28: 0.066, 365: 0.0847},
}

SEASONAL_PARAMS = {
    "windows": (28, 56, 180, 365), "coeffs": (0.18, 0.12, 0.12, 0.18),
    "dow_coeff": 0.15, "dow_n": 24, "inactive_days": 90,
    "sparse_req": 20, "recent_window": 90, "sparse_power": 1.0,
    "zero_sunday": True, "zero_profit": True, "scale": 0.95,
    "lag_coeffs": {364: 0.08, 365: 0.08, 371: 0.04, 28: 0.03},
}

RECENT_PARAMS = {
    "windows": (7, 14, 28, 56), "coeffs": (0.20, 0.20, 0.20, 0.10),
    "dow_coeff": 0.30, "dow_n": 8, "inactive_days": 30,
    "sparse_req": 30, "recent_window": 90, "sparse_power": 1.0,
    "zero_sunday": True, "zero_profit": True, "scale": 1.0, "lag_coeffs": {},
}

LONG_PARAMS = {
    "windows": (56, 180, 365), "coeffs": (0.20, 0.35, 0.25),
    "dow_coeff": 0.10, "dow_n": 52, "inactive_days": 365,
    "sparse_req": 20, "recent_window": 365, "sparse_power": 0.75,
    "zero_sunday": True, "zero_profit": True, "scale": 0.95, "lag_coeffs": {},
}


CANDIDATES = {
    "zero": None,
    "v1": V1_PARAMS,
    "v2": V2_PARAMS,
    "opt1": OPT1_PARAMS,
    "opt2_smooth": OPT2_PARAMS,
    "opt3": OPT3_PARAMS,
    "opt4_year": OPT4_PARAMS,
    "seasonal": SEASONAL_PARAMS,
    "recent": RECENT_PARAMS,
    "long": LONG_PARAMS,
}


def get_cut_idx(dates, d):
    loc = np.where(dates == pd.Timestamp(d))[0]
    if len(loc) == 0:
        raise ValueError(f"Date {d} not found in training calendar")
    return int(loc[0])


def build_private_selector(net_mat, dates, profit_pos):
    fold_dates = [
        "2025-07-11", "2025-06-13", "2025-05-16", "2025-04-18",
        "2025-03-21", "2025-02-21", "2025-01-24", "2024-12-27",
    ]
    fold_idxs = [get_cut_idx(dates, d) for d in fold_dates]
    cand_names = list(CANDIDATES.keys())
    n_c, n_f, n_s = len(cand_names), len(fold_idxs), net_mat.shape[1]

    preds = np.empty((n_c, n_f, 56, n_s), dtype=np.float32)
    for ci, name in enumerate(cand_names):
        for fi, cut in enumerate(fold_idxs):
            if CANDIDATES[name] is None:
                preds[ci, fi] = 0
            else:
                feat = precompute_feature_for_cut(net_mat, dates, cut)
                preds[ci, fi] = make_baseline_forecast(feat, profit_pos, CANDIDATES[name])

    actuals = np.stack([net_mat[cut + 1:cut + 57] for cut in fold_idxs], axis=0).astype(np.float32)
    err = preds - actuals[None, :, :, :]
    sse_second = np.sum(err[:, :, 28:, :] ** 2, axis=2)

    # Conservative selection: only switch from opt1 if another method is at least 20% better
    # on the second half of local 56-day folds. This is aimed at Private leaderboard robustness.
    default_idx = cand_names.index("opt1")
    train_sse = sse_second.sum(axis=1)
    base_sse = train_sse[default_idx]
    best_local = train_sse.argmin(axis=0)
    best_sse = train_sse[best_local, np.arange(n_s)]
    use = best_sse < base_sse * 0.80
    selected = np.where(use, best_local, default_idx)

    counts = dict(zip(cand_names, np.bincount(selected, minlength=n_c)))
    print("Private selector counts:", counts)
    return cand_names, selected


def make_selected_prediction(final_preds_cand, cand_names, selected, default_pred):
    pred = default_pred.copy()
    for ci in np.unique(selected):
        mask = selected == ci
        pred[:, mask] = final_preds_cand[ci][:, mask]
    return pred


def get_friday_cutoffs(dates, start_date, end_date, step_weeks=2):
    date_to_idx = {d: i for i, d in enumerate(dates)}
    cutoffs = []
    for d in pd.date_range(start_date, end_date, freq=f"{step_weeks}W-FRI"):
        if d in date_to_idx:
            cutoffs.append(date_to_idx[d])
    end_ts = pd.Timestamp(end_date)
    if end_ts in date_to_idx and date_to_idx[end_ts] not in cutoffs:
        cutoffs.append(date_to_idx[end_ts])
    return sorted(cutoffs)


def days_since_positive_selected(hist):
    pos = hist > 0
    any_pos = pos.any(axis=0)
    rev_idx = np.argmax(pos[::-1], axis=0).astype(np.int32)
    out = rev_idx.astype(np.float32)
    out[~any_pos] = 1e6
    return out


def build_ml_static(net_mat, sales_mat, return_mat, profit_pos, denom_full, selected_idx):
    total_sales = sales_mat[:, selected_idx].sum(axis=0).astype(np.float32)
    return_ratio = (return_mat[:, selected_idx].sum(axis=0) / np.maximum(total_sales, 1)).astype(np.float32)
    profit_per_unit = (profit_pos[selected_idx] / np.maximum(total_sales, 1)).astype(np.float32)
    sale_days_all = (sales_mat[:, selected_idx] > 0).sum(axis=0).astype(np.float32)

    static = {
        "log_profit": np.log1p(profit_pos[selected_idx]).astype(np.float32),
        "log_denom": np.log1p(denom_full[selected_idx]).astype(np.float32),
        "return_ratio": return_ratio,
        "log_ppu": np.log1p(np.clip(profit_per_unit, 0, None)).astype(np.float32),
        "freq_all": (sale_days_all / len(net_mat)).astype(np.float32),
    }

    feature_names = [
        "sku_code", "horizon", "future_dow", "future_month", "is_saturday", "is_sunday",
        "dayofyear_sin", "dayofyear_cos",
        "log_profit", "log_ppu", "return_ratio", "freq_all", "log_denom",
        "days_since",
        "sale_days_30", "sale_days_60", "sale_days_90", "sale_days_180", "sale_days_365",
        "mean_7", "mean_14", "mean_28", "mean_56", "mean_90", "mean_180", "mean_365",
        "dow_mean_8", "dow_mean_24", "dow_mean_52",
        "lag_7", "lag_14", "lag_28", "lag_56", "lag_364", "lag_365", "lag_371",
        "base_v2",
    ]
    return static, feature_names


def build_ml_features_for_cut(cut, selected_idx, mat, dates, profit_pos, static, feature_names):
    n_sku = len(selected_idx)
    n_features = len(feature_names)
    X = np.empty((56 * n_sku, n_features), dtype=np.float32)

    hist = mat[:cut + 1, selected_idx]
    n_days = hist.shape[0]
    hist_dates = dates[:cut + 1]
    future_dates = pd.date_range(hist_dates[-1] + pd.Timedelta(days=1), periods=56, freq="D")
    future_dow = future_dates.dayofweek.to_numpy().astype(np.float32)
    future_month = future_dates.month.to_numpy().astype(np.float32)
    doy = future_dates.dayofyear.to_numpy().astype(np.float32)
    sin_doy = np.sin(2 * np.pi * doy / 366).astype(np.float32)
    cos_doy = np.cos(2 * np.pi * doy / 366).astype(np.float32)

    means = {w: hist[max(0, n_days - w):].mean(axis=0).astype(np.float32) for w in [7, 14, 28, 56, 90, 180, 365]}
    sale_days = {w: (hist[max(0, n_days - w):] > 0).sum(axis=0).astype(np.float32) for w in [30, 60, 90, 180, 365]}
    ds = days_since_positive_selected(hist).astype(np.float32)

    hist_dow = hist_dates.dayofweek.to_numpy()
    dow_means_by_n = {}
    for dn in [8, 24, 52]:
        by_dow = []
        for dow in range(7):
            idx = np.where(hist_dow == dow)[0]
            if len(idx) > dn:
                idx = idx[-dn:]
            arr = hist[idx].mean(axis=0).astype(np.float32) if len(idx) > 0 else np.zeros(n_sku, dtype=np.float32)
            by_dow.append(arr)
        dow_means_by_n[dn] = [by_dow[int(d)] for d in future_dow]

    lag_features = {}
    for lag in [7, 14, 28, 56, 364, 365, 371]:
        rows = []
        for h in range(1, 57):
            ridx = cut + h - lag
            if 0 <= ridx <= cut:
                rows.append(mat[ridx, selected_idx].astype(np.float32))
            else:
                rows.append(np.zeros(n_sku, dtype=np.float32))
        lag_features[lag] = rows

    raw_v2 = 0.20 * means[7] + 0.35 * means[28] + 0.20 * means[56] + 0.05 * means[180]
    conf_v2 = np.minimum(1.0, sale_days[180] / 100.0).astype(np.float32)
    inactive_mask = ds > 30
    base_v2_rows = []
    for hidx in range(56):
        pred = raw_v2 + 0.20 * dow_means_by_n[24][hidx]
        if int(future_dow[hidx]) == 6:
            pred = np.zeros(n_sku, dtype=np.float32)
        pred = (pred * conf_v2 * 1.05).copy()
        pred[inactive_mask] = 0
        pred[profit_pos[selected_idx] <= 0] = 0
        base_v2_rows.append(np.clip(pred, 0, None).astype(np.float32))

    X[:, 0] = np.tile(np.arange(n_sku, dtype=np.float32), 56)
    X[:, 1] = np.repeat(np.arange(1, 57, dtype=np.float32), n_sku)
    X[:, 2] = np.repeat(future_dow, n_sku)
    X[:, 3] = np.repeat(future_month, n_sku)
    X[:, 4] = (X[:, 2] == 5).astype(np.float32)
    X[:, 5] = (X[:, 2] == 6).astype(np.float32)
    X[:, 6] = np.repeat(sin_doy, n_sku)
    X[:, 7] = np.repeat(cos_doy, n_sku)

    X[:, 8] = np.tile(static["log_profit"], 56)
    X[:, 9] = np.tile(static["log_ppu"], 56)
    X[:, 10] = np.tile(static["return_ratio"], 56)
    X[:, 11] = np.tile(static["freq_all"], 56)
    X[:, 12] = np.tile(static["log_denom"], 56)
    X[:, 13] = np.tile(ds, 56)

    col = 14
    for ww in [30, 60, 90, 180, 365]:
        X[:, col] = np.tile(sale_days[ww], 56)
        col += 1
    for ww in [7, 14, 28, 56, 90, 180, 365]:
        X[:, col] = np.tile(means[ww], 56)
        col += 1
    for dn in [8, 24, 52]:
        X[:, col] = np.concatenate(dow_means_by_n[dn])
        col += 1
    for lag in [7, 14, 28, 56, 364, 365, 371]:
        X[:, col] = np.concatenate(lag_features[lag])
        col += 1
    X[:, col] = np.concatenate(base_v2_rows)
    col += 1
    assert col == len(feature_names)
    return X, future_dates


def build_ml_training_dataset(cutoffs, selected_idx, mat, dates, profit_pos, denom_full, static, feature_names):
    n_sku = len(selected_idx)
    rows_per_cut = 56 * n_sku
    X_all = np.empty((len(cutoffs) * rows_per_cut, len(feature_names)), dtype=np.float32)
    y_all = np.empty(len(cutoffs) * rows_per_cut, dtype=np.float32)
    w_all = np.empty(len(cutoffs) * rows_per_cut, dtype=np.float32)

    base_weight = profit_pos[selected_idx] / np.maximum(np.sqrt(denom_full[selected_idx]), 1e-9)
    base_weight = base_weight / np.nanmean(base_weight[base_weight > 0])
    base_weight = np.clip(base_weight, 0, 100).astype(np.float32)
    weight_tile = np.tile(base_weight, 56)

    for i, cut in enumerate(cutoffs):
        X_cut, _ = build_ml_features_for_cut(cut, selected_idx, mat, dates, profit_pos, static, feature_names)
        start = i * rows_per_cut
        end = start + rows_per_cut
        X_all[start:end] = X_cut
        y = np.clip(mat[cut + 1:cut + 57, selected_idx], 0, None).reshape(-1).astype(np.float32)
        y_all[start:end] = y
        w_all[start:end] = weight_tile

    return X_all, y_all, w_all


def build_ml_public_blend(net_mat, sales_mat, return_mat, dates, profit_pos, denom_full, pred_opt1):
    if lgb is None:
        print("LightGBM unavailable. Skipping ML blend; using opt1 validation forecast.")
        return pred_opt1.copy()

    top_n = 1000
    order_by_profit = np.argsort(-profit_pos)
    selected_idx = order_by_profit[:top_n]

    static, feature_names = build_ml_static(net_mat, sales_mat, return_mat, profit_pos, denom_full, selected_idx)
    train_cutoffs = get_friday_cutoffs(dates, "2023-01-06", "2025-07-11", step_weeks=2)

    print(f"Training LightGBM log model on {len(train_cutoffs)} cutoffs × 56 horizons × {top_n} SKUs...")
    X_train, y_train, w_train = build_ml_training_dataset(
        train_cutoffs, selected_idx, net_mat, dates, profit_pos, denom_full, static, feature_names
    )

    params = {
        "objective": "regression",
        "metric": "rmse",
        "learning_rate": 0.06,
        "num_leaves": 96,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "lambda_l2": 2.0,
        "max_bin": 255,
        "num_threads": 8,
        "verbosity": -1,
        "seed": 2026,
    }

    dataset = lgb.Dataset(
        X_train,
        label=np.log1p(y_train),
        weight=w_train,
        feature_name=feature_names,
        categorical_feature=["sku_code", "future_dow", "future_month"],
        free_raw_data=False,
    )
    model = lgb.train(params, dataset, num_boost_round=140, callbacks=[lgb.log_evaluation(0)])

    X_pred, future_dates = build_ml_features_for_cut(len(dates) - 1, selected_idx, net_mat, dates, profit_pos, static, feature_names)
    ml_pred = np.expm1(model.predict(X_pred, num_iteration=model.current_iteration()))
    ml_pred = np.clip(ml_pred.astype(np.float32).reshape(56, top_n), 0, None)

    # Conservative public-window blend only. Evaluation/private window remains controlled by v7/v6 logic.
    alpha = 0.05
    scale_ml = 2.5
    pred_blend = pred_opt1.copy()
    pred_blend[:28, selected_idx] = (1 - alpha) * pred_opt1[:28, selected_idx] + alpha * scale_ml * ml_pred[:28]
    pred_blend[future_dates.dayofweek.to_numpy() == 6, :] = 0.0
    pred_blend = np.clip(pred_blend, 0, None)

    del X_train, y_train, w_train, dataset, model, X_pred, ml_pred
    gc.collect()
    return pred_blend


def main():
    train, sample = read_inputs()
    sku_order, dates, net_mat, sales_mat, return_mat, profit, profit_pos = build_matrices(train, sample)
    denom_full = compute_scale(net_mat)

    print(f"Train dates: {dates[0].date()} -> {dates[-1].date()}")
    print(f"SKUs: {len(sku_order):,}")
    print(f"Positive profit SKUs: {(profit_pos > 0).sum():,}")

    last_idx = len(dates) - 1
    feature_final = precompute_feature_for_cut(net_mat, dates, last_idx)
    pred_opt1 = make_baseline_forecast(feature_final, profit_pos, OPT1_PARAMS)

    # Final candidate predictions for private selector.
    cand_names = list(CANDIDATES.keys())
    final_preds_cand = np.empty((len(cand_names), 56, len(sku_order)), dtype=np.float32)
    for ci, name in enumerate(cand_names):
        if CANDIDATES[name] is None:
            final_preds_cand[ci] = 0
        else:
            final_preds_cand[ci] = make_baseline_forecast(feature_final, profit_pos, CANDIDATES[name])

    selector_names, selected = build_private_selector(net_mat, dates, profit_pos)
    assert selector_names == cand_names
    pred_private_selected = make_selected_prediction(final_preds_cand, cand_names, selected, pred_opt1)

    # v6: global optimized baseline.
    pred_v6 = pred_opt1.copy()

    # v7: public/validation same as v6; private/evaluation uses conservative selector.
    pred_v7 = pred_opt1.copy()
    pred_v7[28:] = pred_private_selected[28:]

    # v8 and v9 use a small ML blend only on validation window to test Public uplift.
    pred_public_ml = build_ml_public_blend(net_mat, sales_mat, return_mat, dates, profit_pos, denom_full, pred_opt1)
    pred_v8 = pred_v7.copy()
    pred_v8[:28] = pred_public_ml[:28]
    pred_v9 = pred_opt1.copy()
    pred_v9[:28] = pred_public_ml[:28]

    make_submission(sample, sku_order, pred_v6, "submission_v6_optimized_global.csv")
    make_submission(sample, sku_order, pred_v7, "submission_v7_private_selector.csv")
    make_submission(sample, sku_order, pred_v8, "submission_v8_public_ml_private_safe.csv")
    make_submission(sample, sku_order, pred_v9, "submission_v9_public_ml_global_private.csv")


if __name__ == "__main__":
    main()
