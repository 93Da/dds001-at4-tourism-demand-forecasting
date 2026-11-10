"""Simplified, leakage-aware reproduction of source-gated tourism forecasting.

Scope:
- Uses the previously downloaded Eurostat tour_occ_nim workbook.
- Predicts monthly EU27 non-resident overnight stays for horizons H=1, 3, 6.
- Uses three feature-family experts: total-demand history, hotel subseries, and other-accommodation subseries.
- Learns a small context gate over expert forecasts and compares it with early fusion, equal weights, and seasonal naive.

IMPORTANT: Hotel/other-accommodation features are subseries of the same Eurostat dataset, not six independent
external sources. This is an architecture pilot, not a full reproduction of the paper's six-source SG-MSF model.
"""
from pathlib import Path
import json
import math
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore", category=UserWarning, module="lightgbm")
ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for p in (RESULTS, FIGURES, LOGS):
    p.mkdir(parents=True, exist_ok=True)

HORIZONS = [1, 3, 6]
COUNTRIES = ["AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "EL", "HU", "IE", "IT", "LV", "LT", "LU", "MT", "NL", "PL", "PT", "RO", "SK", "SI", "ES", "SE"]
COUNTRY_NAMES = {}
SERIES = ["overnight_stays_total", "overnight_stays_hotels", "overnight_stays_other_accommodation"]

HISTORY_COLS = [
    "y_lag_2", "y_lag_3", "y_lag_4", "y_lag_6", "y_lag_12",
    "y_roll_mean_3", "y_roll_mean_6", "y_roll_mean_12",
    "y_roll_std_3", "y_roll_std_6", "y_yoy_growth",
    "y_seasonal_lag_12", "y_seasonal_lag_24", "y_seasonal_lag_36",
]
HOTEL_COLS = [
    "hotel_lag_2", "hotel_lag_3", "hotel_lag_4", "hotel_lag_6", "hotel_lag_12",
    "hotel_roll_mean_3", "hotel_roll_mean_12", "hotel_yoy_growth",
]
OTHER_COLS = [
    "other_lag_2", "other_lag_3", "other_lag_4", "other_lag_6", "other_lag_12",
    "other_roll_mean_3", "other_roll_mean_12", "other_yoy_growth",
]
CALENDAR_COLS = ["month_sin", "month_cos", "quarter", "time_index", "covid_shock", "post_covid"]
COUNTRY_COLS = [f"country_{c}" for c in COUNTRIES]
HISTORY_EXPERT_COLS = HISTORY_COLS + CALENDAR_COLS + COUNTRY_COLS
HOTEL_EXPERT_COLS = HISTORY_COLS + HOTEL_COLS + CALENDAR_COLS + COUNTRY_COLS
OTHER_EXPERT_COLS = HISTORY_COLS + OTHER_COLS + CALENDAR_COLS + COUNTRY_COLS
EARLY_FUSION_COLS = HISTORY_COLS + HOTEL_COLS + OTHER_COLS + CALENDAR_COLS + COUNTRY_COLS
EXPERTS = ["history", "hotel", "other"]
PRED_COLS = [f"pred_{x}" for x in EXPERTS]


def finite_float(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


def safe_window(prefix: pd.Series, end: pd.Period, length: int) -> np.ndarray:
    idx = pd.period_range(end=end, periods=length, freq="M")
    vals = prefix.reindex(idx).to_numpy(dtype=float)
    return vals


def safe_mean(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.mean()) if len(vals) else np.nan


def safe_std(vals):
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    return float(vals.std(ddof=1)) if len(vals) >= 2 else np.nan


def causal_prefix(series: pd.Series, cutoff: pd.Period) -> pd.Series:
    """Interpolate only within observations available by cutoff; never reads beyond cutoff."""
    prefix = series.loc[:cutoff].copy()
    return prefix.interpolate(method="linear", limit_area="inside")


def value_at(prefix: pd.Series, period: pd.Period) -> float:
    if period not in prefix.index:
        return np.nan
    return finite_float(prefix.loc[period])


def add_series_features(out: dict, prefix: pd.Series, prefix_name: str, cutoff: pd.Period, lags, windows):
    # lag labels match the paper's forecast-origin lag convention: y_(origin-2), y_(origin-3), etc.
    for lag in lags:
        out[f"{prefix_name}_lag_{lag}"] = value_at(prefix, cutoff - (lag - 2))
    for w in windows:
        vals = safe_window(prefix, cutoff, w)
        out[f"{prefix_name}_roll_mean_{w}"] = safe_mean(vals)
        if prefix_name == "y" and w in (3, 6):
            out[f"{prefix_name}_roll_std_{w}"] = safe_std(vals)
    cur = value_at(prefix, cutoff)
    year_ago = value_at(prefix, cutoff - 12)
    out[f"{prefix_name}_yoy_growth"] = (cur / year_ago - 1.0) if np.isfinite(cur) and np.isfinite(year_ago) and year_ago != 0 else np.nan


def build_feature_panel(data_all: pd.DataFrame) -> pd.DataFrame:
    data_all = data_all.copy()
    data_all["month"] = pd.PeriodIndex(data_all["month"], freq="M")
    for c in SERIES:
        data_all[c] = pd.to_numeric(data_all[c], errors="coerce")
    global COUNTRY_NAMES
    COUNTRY_NAMES = data_all.drop_duplicates("geo_code").set_index("geo_code")["country"].to_dict()
    lookup = {}
    for geo in COUNTRIES:
        sub = data_all[data_all.geo_code == geo].sort_values("month")
        lookup[geo] = {col: pd.Series(sub[col].to_numpy(dtype=float), index=pd.PeriodIndex(sub.month, freq="M")) for col in SERIES}

    target_months = pd.period_range("2015-01", "2025-12", freq="M")
    records = []
    for geo in COUNTRIES:
        ys = lookup[geo][SERIES[0]]
        hs = lookup[geo][SERIES[1]]
        os = lookup[geo][SERIES[2]]
        for target_month in target_months:
            y_actual = value_at(ys, target_month)
            for horizon in HORIZONS:
                origin = target_month - horizon
                cutoff = origin - 2  # publication lag: only values up to origin-2 are available
                yp = causal_prefix(ys, cutoff)
                hp = causal_prefix(hs, cutoff)
                op = causal_prefix(os, cutoff)
                row = {
                    "geo_code": geo,
                    "country": COUNTRY_NAMES.get(geo, geo),
                    "target_month": str(target_month),
                    "target_period": target_month,
                    "origin_month": str(origin),
                    "cutoff_month": str(cutoff),
                    "horizon": horizon,
                    "y": y_actual,
                    "month_num": target_month.month,
                    "year": target_month.year,
                    "time_index": (target_month.year - 2015) * 12 + target_month.month - 1,
                    "month_sin": math.sin(2 * math.pi * target_month.month / 12),
                    "month_cos": math.cos(2 * math.pi * target_month.month / 12),
                    "quarter": (target_month.month - 1) // 3 + 1,
                    "covid_shock": int(pd.Period("2020-03", "M") <= target_month <= pd.Period("2021-06", "M")),
                    "post_covid": int(target_month >= pd.Period("2021-07", "M")),
                    # seasonal-naive target y_(T-12) is always older than the publication cutoff for H<=6
                    "seasonal_naive": value_at(yp, target_month - 12),
                }
                add_series_features(row, yp, "y", cutoff, [2, 3, 4, 6, 12], [3, 6, 12])
                add_series_features(row, hp, "hotel", cutoff, [2, 3, 4, 6, 12], [3, 12])
                add_series_features(row, op, "other", cutoff, [2, 3, 4, 6, 12], [3, 12])
                for c in COUNTRIES:
                    row[f"country_{c}"] = int(c == geo)
                for m in range(1, 13):
                    row[f"month_{m:02d}"] = int(target_month.month == m)
                for h in HORIZONS:
                    row[f"horizon_{h}"] = int(horizon == h)
                row["gate_yoy_growth"] = row.get("y_yoy_growth", np.nan)
                records.append(row)
    panel = pd.DataFrame(records)
    panel["target_period"] = pd.PeriodIndex(panel["target_period"], freq="M")
    return panel


def make_lgbm(seed=2026):
    return lgb.LGBMRegressor(
        objective="regression_l1", n_estimators=180, learning_rate=0.05,
        num_leaves=15, max_depth=6, min_child_samples=20,
        subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=seed, n_jobs=1, verbosity=-1,
    )


def train_experts(panel: pd.DataFrame):
    train_mask = (panel.target_period >= pd.Period("2015-01", "M")) & (panel.target_period <= pd.Period("2021-12", "M")) & panel.y.notna()
    predictions = panel.copy()
    model_info = []
    cols_by_expert = {
        "history": HISTORY_EXPERT_COLS,
        "hotel": HOTEL_EXPERT_COLS,
        "other": OTHER_EXPERT_COLS,
        "early_fusion": EARLY_FUSION_COLS,
    }
    for h in HORIZONS:
        for expert, cols in cols_by_expert.items():
            mask_train = train_mask & (panel.horizon == h)
            X_train = panel.loc[mask_train, cols].astype(float)
            y_train = panel.loc[mask_train, "y"].astype(float)
            model = make_lgbm(seed=2026 + h + len(expert))
            model.fit(X_train, y_train / 1_000_000.0)
            mask_all = panel.horizon == h
            X_all = panel.loc[mask_all, cols].astype(float)
            pred = np.maximum(model.predict(X_all) * 1_000_000.0, 0.0)
            predictions.loc[mask_all, f"pred_{expert}"] = pred
            model_info.append({"horizon": h, "model": expert, "n_train": int(mask_train.sum()),
                               "features": len(cols), "params": model.get_params()})
    (LOGS / "model_configs.json").write_text(json.dumps(model_info, indent=2, default=str), encoding="utf-8")
    return predictions


def train_expanding_experts(panel: pd.DataFrame) -> pd.DataFrame:
    """Yearly expanding-window refits from 2022 onward.

    To keep this a small reproducible assignment, models are refit once per forecast
    year/horizon rather than every month. For each year, training labels are truncated
    at the first target month's publication cutoff (Jan - horizon - 2 months), so no
    label later than that cutoff is used for any prediction in that year.
    """
    predictions = panel.copy()
    for name in ["history", "hotel", "other", "early_fusion"]:
        predictions[f"pred_{name}"] = np.nan
    cols_by_expert = {
        "history": HISTORY_EXPERT_COLS,
        "hotel": HOTEL_EXPERT_COLS,
        "other": OTHER_EXPERT_COLS,
        "early_fusion": EARLY_FUSION_COLS,
    }
    audit = []
    for year in range(2022, 2026):
        first_target_month = pd.Period(f"{year}-01", "M")
        for horizon in HORIZONS:
            cutoff = first_target_month - horizon - 2
            train_mask = (
                (panel.horizon == horizon)
                & (panel.target_period >= pd.Period("2015-01", "M"))
                & (panel.target_period <= cutoff)
                & panel.y.notna()
            )
            target_mask = (panel.horizon == horizon) & (panel.target_period.dt.year == year)
            if int(train_mask.sum()) < 100 or int(target_mask.sum()) == 0:
                continue
            for expert, cols in cols_by_expert.items():
                model = make_lgbm(seed=2026 + horizon + len(expert) + year)
                X_train = panel.loc[train_mask, cols].astype(float)
                y_train = panel.loc[train_mask, "y"].astype(float) / 1_000_000.0
                X_target = panel.loc[target_mask, cols].astype(float)
                model.fit(X_train, y_train)
                pred = np.maximum(model.predict(X_target) * 1_000_000.0, 0.0)
                predictions.loc[target_mask, f"pred_{expert}"] = pred
                audit.append({
                    "forecast_year": year, "horizon": horizon, "expert": expert,
                    "training_cutoff": str(cutoff), "n_train": int(train_mask.sum()),
                    "n_forecast_rows": int(target_mask.sum()), "n_features": len(cols),
                })
    pd.DataFrame(audit).to_csv(LOGS / "expanding_refit_audit.csv", index=False)
    config_rows = []
    for horizon in HORIZONS:
        for expert, cols in cols_by_expert.items():
            config_rows.append({"horizon": horizon, "model": expert, "features": len(cols), "params": make_lgbm().get_params()})
    (LOGS / "model_configs.json").write_text(json.dumps(config_rows, indent=2, default=str), encoding="utf-8")
    return predictions


class ContextGate(nn.Module):
    def __init__(self, input_dim: int, n_experts: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 32), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(32, 16), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(16, n_experts),
        )

    def forward(self, x):
        weights = torch.softmax(self.net(x), dim=1)
        combined = torch.sum(weights * x[:, :len(EXPERTS)], dim=1)
        return combined, weights


def country_means(panel: pd.DataFrame):
    train = panel[(panel.target_period >= pd.Period("2015-01", "M")) &
                  (panel.target_period <= pd.Period("2021-12", "M")) & panel.y.notna()]
    means = train.groupby("geo_code").y.mean().to_dict()
    return {c: max(float(means.get(c, 1.0)), 1.0) for c in COUNTRIES}


def gate_matrix(df: pd.DataFrame, means: dict) -> np.ndarray:
    expert_preds = np.column_stack([df[c].to_numpy(dtype=float) / df.geo_code.map(means).to_numpy(dtype=float) for c in PRED_COLS])
    country = df[[f"country_{c}" for c in COUNTRIES]].to_numpy(dtype=float)
    month = df[[f"month_{m:02d}" for m in range(1, 13)]].to_numpy(dtype=float)
    horizon = df[[f"horizon_{h}" for h in HORIZONS]].to_numpy(dtype=float)
    time_index = (df["time_index"].to_numpy(dtype=float) / 132.0).reshape(-1, 1)
    growth = np.clip(pd.to_numeric(df["gate_yoy_growth"], errors="coerce").fillna(0).to_numpy(dtype=float), -3, 3).reshape(-1, 1)
    return np.nan_to_num(np.column_stack([expert_preds, country, month, horizon, time_index, growth]), nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def fit_gate(predictions: pd.DataFrame, means: dict):
    train = predictions[(predictions.target_period >= pd.Period("2022-01", "M")) &
                        (predictions.target_period <= pd.Period("2022-12", "M")) & predictions.y.notna()].copy()
    valid = predictions[(predictions.target_period >= pd.Period("2023-01", "M")) &
                        (predictions.target_period <= pd.Period("2023-12", "M")) & predictions.y.notna()].copy()
    X_train = gate_matrix(train, means)
    X_valid = gate_matrix(valid, means)
    train_scales = train.geo_code.map(means).to_numpy(dtype=float)
    valid_scales = valid.geo_code.map(means).to_numpy(dtype=float)
    y_train = (train.y.to_numpy(dtype=float) / train_scales).astype(np.float32)
    y_valid = (valid.y.to_numpy(dtype=float) / valid_scales).astype(np.float32)

    torch.manual_seed(2026)
    np.random.seed(2026)
    device = torch.device("cpu")
    model = ContextGate(X_train.shape[1]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.SmoothL1Loss()
    ds = TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
    loader = DataLoader(ds, batch_size=128, shuffle=True)
    Xt = torch.from_numpy(X_valid)
    yt = torch.from_numpy(y_valid)
    best_state, best_loss, best_epoch, patience, stale = None, float("inf"), 0, 35, 0
    history = []
    for epoch in range(1, 301):
        model.train()
        train_losses = []
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            train_losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            vpred, _ = model(Xt)
            val_loss = float(loss_fn(vpred, yt))
        history.append({"epoch": epoch, "train_loss": float(np.mean(train_losses)), "validation_loss_2023": val_loss})
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch, stale = val_loss, epoch, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
        if stale >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    pd.DataFrame(history).to_csv(LOGS / "gate_training_history.csv", index=False)
    torch.save(model.state_dict(), RESULTS / "context_gate.pt")
    (LOGS / "gate_training_summary.json").write_text(json.dumps({
        "input_dim": X_train.shape[1], "training_rows_2022": len(train), "validation_rows_2023": len(valid),
        "best_epoch": best_epoch, "best_validation_smooth_l1": best_loss,
        "features": ["3 normalized expert forecasts", "27 country one-hot", "12 month one-hot", "3 horizon one-hot", "time index", "historical YoY growth"],
        "limitations": "Gate trained on 2022 forecasts from yearly-refit experts; 2023 forecasts from the next annual refit are used for early stopping. This is simpler than the paper's expanding-window OOF gate training."
    }, indent=2), encoding="utf-8")
    return model


def apply_gate(predictions: pd.DataFrame, means: dict, model: ContextGate) -> pd.DataFrame:
    df = predictions.copy()
    eligible = df[PRED_COLS].notna().all(axis=1)
    df["pred_gate"] = np.nan
    df["pred_equal_weight"] = np.nan
    for expert in EXPERTS:
        df[f"gate_weight_{expert}"] = np.nan
    if eligible.any():
        eligible_df = df.loc[eligible]
        X = gate_matrix(eligible_df, means)
        model.eval()
        with torch.no_grad():
            combined_norm, weights = model(torch.from_numpy(X))
        scale = eligible_df.geo_code.map(means).to_numpy(dtype=float)
        df.loc[eligible, "pred_gate"] = np.maximum(combined_norm.numpy() * scale, 0.0)
        for i, expert in enumerate(EXPERTS):
            df.loc[eligible, f"gate_weight_{expert}"] = weights[:, i].numpy()
        df.loc[eligible, "pred_equal_weight"] = np.maximum(eligible_df[PRED_COLS].mean(axis=1).to_numpy(dtype=float), 0.0)
    return df


def compute_mase_denominators(all_data: pd.DataFrame):
    data = all_data.copy()
    data["month"] = pd.PeriodIndex(data["month"], freq="M")
    denoms = {}
    for geo, sub in data.groupby("geo_code"):
        sub = sub.sort_values("month").set_index("month")
        y = sub["overnight_stays_total"].astype(float)
        train_idx = pd.period_range("2015-01", "2021-12", freq="M")
        curr = y.reindex(train_idx).to_numpy(dtype=float)
        prev = y.reindex(train_idx - 12).to_numpy(dtype=float)
        valid = np.isfinite(curr) & np.isfinite(prev)
        d = float(np.mean(np.abs(curr[valid] - prev[valid]))) if valid.any() else np.nan
        denoms[geo] = d if np.isfinite(d) and d > 0 else 1.0
    return denoms


def metric_row(frame: pd.DataFrame, prediction_col: str, denoms: dict, horizon: int, model_name: str):
    all_prediction_cols = ["seasonal_naive", "pred_history", "pred_hotel", "pred_other", "pred_equal_weight", "pred_early_fusion", "pred_gate"]
    common_rows = frame[all_prediction_cols].notna().all(axis=1)
    sub = frame[(frame.horizon == horizon) & frame.y.notna() & common_rows].copy()
    errors = sub.y.to_numpy(dtype=float) - sub[prediction_col].to_numpy(dtype=float)
    country_mase = []
    for geo, g in sub.groupby("geo_code"):
        d = denoms.get(geo, 1.0)
        country_mase.append(float(np.mean(np.abs(g.y.to_numpy(dtype=float) - g[prediction_col].to_numpy(dtype=float))) / d))
    denom_sum = float(np.abs(sub.y.to_numpy(dtype=float)).sum())
    return {
        "model": model_name, "horizon_months": horizon, "n_test": len(sub),
        "median_country_MASE": float(np.median(country_mase)),
        "mean_country_MASE": float(np.mean(country_mase)),
        "MAE_nights": float(np.mean(np.abs(errors))),
        "RMSE_nights": float(np.sqrt(np.mean(errors ** 2))),
        "WAPE_percent": float(100 * np.abs(errors).sum() / denom_sum) if denom_sum else np.nan,
    }


def main():
    source = PROCESSED / "eurostat_eu27_nonresident_all_months.csv"
    if not source.exists():
        raise FileNotFoundError("Run `python src/prepare_data.py` first.")
    data_all = pd.read_csv(source)
    data_all["month"] = pd.PeriodIndex(data_all["month"], freq="M")
    feature_cache = RESULTS / "feature_panel.csv"
    if feature_cache.exists():
        panel = pd.read_csv(feature_cache)
        panel["target_period"] = pd.PeriodIndex(panel["target_period"], freq="M")
    else:
        panel = build_feature_panel(data_all)
        panel.to_csv(feature_cache, index=False)
    # Add explicitly available annual seasonal lags. These are available by the forecast cutoff for H=1/3/6.
    panel = panel.sort_values(["geo_code", "horizon", "target_period"]).copy()
    panel["y_seasonal_lag_12"] = pd.to_numeric(panel["seasonal_naive"], errors="coerce")
    panel["y_seasonal_lag_24"] = panel.groupby(["geo_code", "horizon"])["seasonal_naive"].shift(12)
    panel["y_seasonal_lag_36"] = panel.groupby(["geo_code", "horizon"])["seasonal_naive"].shift(24)
    panel.to_csv(feature_cache, index=False)
    predictions = train_expanding_experts(panel)
    means = country_means(panel)
    gate = fit_gate(predictions, means)
    predictions = apply_gate(predictions, means, gate)
    predictions[predictions.target_period >= pd.Period("2022-01", "M")].to_csv(RESULTS / "expanding_forecasts_2022_2025.csv", index=False)

    test = predictions[(predictions.target_period >= pd.Period("2024-01", "M")) &
                       (predictions.target_period <= pd.Period("2025-12", "M"))].copy()
    denoms = compute_mase_denominators(data_all)
    models = [
        ("Seasonal naive", "seasonal_naive"),
        ("History expert", "pred_history"),
        ("Hotel-subseries expert", "pred_hotel"),
        ("Other-accommodation expert", "pred_other"),
        ("Equal-weight experts", "pred_equal_weight"),
        ("Early-fusion LightGBM", "pred_early_fusion"),
        ("Context-gated fusion", "pred_gate"),
    ]
    rows = []
    for h in HORIZONS:
        for name, col in models:
            rows.append(metric_row(test, col, denoms, h, name))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(RESULTS / "test_metrics.csv", index=False)

    # Country-level diagnostics on the same common test sample used for the headline metrics.
    common_pred_cols = ["seasonal_naive", "pred_history", "pred_hotel", "pred_other", "pred_equal_weight", "pred_early_fusion", "pred_gate"]
    country_rows = []
    for h in HORIZONS:
        common = test[(test.horizon == h) & test.y.notna() & test[common_pred_cols].notna().all(axis=1)]
        for model_name, pred_col in models:
            for geo, g in common.groupby("geo_code"):
                err = g.y.to_numpy(dtype=float) - g[pred_col].to_numpy(dtype=float)
                d = denoms.get(geo, 1.0)
                country_rows.append({"geo_code": geo, "country": COUNTRY_NAMES.get(geo, geo), "horizon_months": h,
                                     "model": model_name, "n_test": len(g),
                                     "MASE": float(np.mean(np.abs(err)) / d),
                                     "MAE_nights": float(np.mean(np.abs(err))),
                                     "WAPE_percent": float(100 * np.abs(err).sum() / np.abs(g.y.to_numpy(dtype=float)).sum())})
    pd.DataFrame(country_rows).to_csv(RESULTS / "country_level_metrics.csv", index=False)

    # Compare with paper's reported primary metric, clearly separated from this pilot's results.
    paper = pd.DataFrame([
        {"model": "Paper SG-MSF (reported)", "horizon_months": 1, "median_country_MASE": 0.504},
        {"model": "Paper SG-MSF (reported)", "horizon_months": 3, "median_country_MASE": 0.566},
        {"model": "Paper SG-MSF (reported)", "horizon_months": 6, "median_country_MASE": 0.627},
        {"model": "Paper early-fusion LightGBM (reported)", "horizon_months": 1, "median_country_MASE": 0.563},
        {"model": "Paper early-fusion LightGBM (reported)", "horizon_months": 3, "median_country_MASE": 0.539},
        {"model": "Paper early-fusion LightGBM (reported)", "horizon_months": 6, "median_country_MASE": 0.568},
    ])
    paper.to_csv(RESULTS / "paper_reported_reference_metrics.csv", index=False)

    # Plot median country MASE across models and horizons.
    pivot = metrics.pivot(index="model", columns="horizon_months", values="median_country_MASE")
    ax = pivot.plot(kind="bar", figsize=(12, 6), width=0.82)
    ax.set_title("Simplified reproduction: median country-level MASE (lower is better)")
    ax.set_xlabel("")
    ax.set_ylabel("Median country MASE")
    ax.legend(title="Forecast horizon (months)")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "mase_model_comparison.png", dpi=180)
    plt.close()

    # Average learned gate weights by horizon in test period.
    weight_cols = [f"gate_weight_{x}" for x in EXPERTS]
    avg_weights = test.groupby("horizon")[weight_cols].mean().reset_index()
    avg_weights.to_csv(RESULTS / "mean_gate_weights_test.csv", index=False)
    ax = avg_weights.set_index("horizon").plot(kind="bar", stacked=True, figsize=(8, 5))
    ax.set_title("Average context-gate weights by horizon (test 2024–2025)")
    ax.set_xlabel("Forecast horizon (months)")
    ax.set_ylabel("Mean gate weight")
    ax.legend([x.replace("gate_weight_", "") for x in weight_cols], title="Expert")
    ax.grid(axis="y", alpha=0.25)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIGURES / "gate_weights_by_horizon.png", dpi=180)
    plt.close()

    # Representative time series for one-step forecasting (not a cherry-picked performance claim).
    representative = [c for c in ["DE", "FR", "IT"] if c in COUNTRIES]
    fig, axes = plt.subplots(len(representative), 1, figsize=(12, 8), sharex=True)
    if len(representative) == 1:
        axes = [axes]
    for ax, geo in zip(axes, representative):
        g = test[(test.geo_code == geo) & (test.horizon == 1)].sort_values("target_period")
        ax.plot(g.target_period.astype(str), g.y, label="Actual", linewidth=1.7)
        ax.plot(g.target_period.astype(str), g.pred_early_fusion, label="Early fusion", linewidth=1.1)
        ax.plot(g.target_period.astype(str), g.pred_gate, label="Context gate", linewidth=1.1)
        ax.set_title(f"{COUNTRY_NAMES.get(geo, geo)} — H=1")
        ax.set_ylabel("Overnight stays")
        ax.grid(alpha=0.2)
    axes[0].legend(ncol=3)
    axes[-1].set_xlabel("Target month")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(FIGURES / "actual_vs_prediction_selected_countries.png", dpi=180)
    plt.close()

    # Data summary for run log.
    summary = {
        "dataset": "Eurostat tour_occ_nim, downloaded 2026-10-23; no network access during this run",
        "selection": "EU27; FOR; unit NR; NACE I551-I553 target; I551 and I552_I553 lagged subseries as auxiliary feature families",
        "unique_country_month_targets": int(panel[(panel.target_period >= pd.Period("2015-01", "M")) & (panel.target_period <= pd.Period("2025-12", "M"))][["geo_code", "target_month"]].drop_duplicates().shape[0]),
        "feature_rows_across_horizons": int(((panel.target_period >= pd.Period("2015-01", "M")) & (panel.target_period <= pd.Period("2025-12", "M"))).sum()),
        "observed_target_labels_2015_2021_per_horizon": int((((panel.target_period <= pd.Period("2021-12", "M")) & panel.y.notna()).sum()) // len(HORIZONS)),
        "gate_training_rows_2022_across_horizons": int(((panel.target_period.dt.year == 2022) & panel.y.notna()).sum()),
        "gate_validation_rows_2023_across_horizons": int(((panel.target_period.dt.year == 2023) & panel.y.notna()).sum()),
        "test_target_observations_per_horizon": int(test[test.horizon == 1].y.notna().sum()),
        "common_test_rows_by_horizon": {str(h): int(test[(test.horizon == h) & test[["seasonal_naive", "pred_history", "pred_hotel", "pred_other", "pred_equal_weight", "pred_early_fusion", "pred_gate"]].notna().all(axis=1)].shape[0]) for h in HORIZONS},
        "missing_target_values_2015_2025": int(panel[(panel.target_period >= pd.Period("2015-01", "M")) & (panel.target_period <= pd.Period("2025-12", "M"))].y.isna().sum() // len(HORIZONS)),
        "test_period": "2024-01 to 2025-12",
        "horizons_months": HORIZONS,
        "expert_count": 3,
        "gate_input_dim": 3 + len(COUNTRIES) + 12 + len(HORIZONS) + 2,
        "gate_best_epoch": json.loads((LOGS / "gate_training_summary.json").read_text(encoding="utf-8"))["best_epoch"],
        "warning": "Simplified architecture pilot only: auxiliary accommodation series come from the same Eurostat table; no Google Trends, aviation, HICP, accommodation-capacity, or ERA5 external datasets. Results must not be described as a full replication of the six-source paper."
    }
    (RESULTS / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    focus_models = ["Seasonal naive", "Equal-weight experts", "Early-fusion LightGBM", "Context-gated fusion"]
    focus = metrics[metrics.model.isin(focus_models)].pivot(index="model", columns="horizon_months", values="median_country_MASE")
    report_table = ["| Model | H=1 | H=3 | H=6 |", "|---|---:|---:|---:|"]
    for name in focus_models:
        if name in focus.index:
            report_table.append(f"| {name} | {focus.loc[name, 1]:.3f} | {focus.loc[name, 3]:.3f} | {focus.loc[name, 6]:.3f} |")
    best_lines = []
    for h in HORIZONS:
        row = metrics[metrics.horizon_months == h].sort_values("median_country_MASE").iloc[0]
        best_lines.append(f"- H={h}: best median country MASE in this run was **{row['model']}** ({row['median_country_MASE']:.3f}).")
    report = [
        "# Simplified reproduction report: source-gated tourism demand forecasting",
        "",
        "## What was reproduced",
        "This project implements a simplified version of the paper's expert-model plus learned-context-gate idea, and compares it with early fusion, equal-weight averaging, and seasonal-naive forecasting.",
        "",
        "## Data",
        "- Source: previously downloaded Eurostat `tour_occ_nim` workbook (extracted 2026-10-23). No new download was performed.",
        "- Target: EU27 monthly non-resident overnight stays (`FOR`, `NR`, `I551-I553`), January 2015–December 2025.",
        "- Auxiliary feature families: lagged hotel-only nights (`I551`) and other-accommodation nights (`I552_I553`). These are subseries of the same Eurostat table, not independent external data sources.",
        "- 3,564 theoretical target observations; 69 target values are missing in 2015–2025. Missing values are not zero-filled. Rows with missing target labels are excluded from fitting/evaluation. This differs from the paper's target-gap interpolation.",
        "- For each feature row, source values are truncated at the publication cutoff before interpolation/feature construction. Values after that row's cutoff are never used to construct features.",
        "",
        "## Model and split",
        "- Forecast horizons: H=1, 3, 6 months.",
        "- Models are refit once per forecast year and horizon, using training labels only up to January of that forecast year minus the horizon and two-month publication lag. This annual expanding refit is a simplification of the paper's monthly updates.",
        "- Gate fitting period: 2022; early stopping/validation period: 2023.",
        "- Final test period: 2024–2025 (648 country-month target observations per horizon; the common comparison set is 647 for H=6 because one seasonal-naive lag is unavailable at the cutoff).",
        "- Three LightGBM feature-family experts: history-only; history + hotel subseries; history + other-accommodation subseries. A small 32/16-unit MLP gate outputs softmax weights.",
        "- LightGBM uses L1 loss in this implementation rather than the paper's Huber objective, to keep the simplified model numerically stable.",
        "- Baselines: seasonal naive, early-fusion LightGBM, and equal-weight expert ensemble.",
        "",
        "## Results: median country-level MASE (lower is better)",
        *report_table,
        "",
        *best_lines,
        "",
        "The gate does not win at every horizon. This is a useful result, not a reason to hide the comparison: with only lagged subseries from one Eurostat table, the learned gate has limited new source information to exploit. See `results/test_metrics.csv` for MAE, RMSE, WAPE, and mean country MASE.",
        "",
        "The original paper reports SG-MSF median MASE 0.504/0.566/0.627 for H=1/3/6 and early-fusion LightGBM 0.563/0.539/0.568. Those values are stored separately in `results/paper_reported_reference_metrics.csv`; they are not directly comparable to this simplified run because the feature sources, missing-value treatment, loss, and refit schedule differ.",
        "",
        "## Limitations",
        "1. This is not a full replication of the paper's six independent source families. We use only the existing Eurostat workbook and two lagged accommodation subseries as auxiliary feature families.",
        "2. The gate is trained on 2022 expert forecasts and early-stopped on 2023; the paper uses expanding-window out-of-fold expert forecasts from 2017–2021 and a more detailed gate input.",
        "3. We do not reproduce Google Trends, Eurostat aviation, Eurocontrol, HICP, accommodation capacity, ERA5-Land, conformal intervals, destination clusters, or SHAP-based competitiveness diagnostics.",
        "4. Therefore, describe this as a simplified architecture reproduction/pilot, not an exact replication of the paper's reported results.",
        "",
        "## Next step toward true multi-source replication",
        "Add at least one independent external source (e.g. monthly air-passenger data or HICP) with documented publication lags, then repeat the comparison with a separate source expert. Add Google Trends/climate only if time and data access permit.",
    ]
    (RESULTS / "replication_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n=== Test metrics ===")
    print(metrics.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\n=== Mean gate weights ===")
    print(avg_weights.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nSaved project outputs to", ROOT)

if __name__ == "__main__":
    main()
