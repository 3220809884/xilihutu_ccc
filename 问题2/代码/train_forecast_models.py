"""Compare next-day load/PV forecasting models for CUMCM 2026 C, Problem 2.

The experiment uses date-ordered train/validation/test splits and evaluates
the next-day 144-slot profile. It intentionally keeps the forecast module
separate from the MILP module.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

SLOTS = 144


def load_daily(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    dates = sorted(df["date"].unique())
    pivot_load = df.pivot(index="date", columns="slot", values="load_actual_kw").loc[dates]
    pivot_pv = df.pivot(index="date", columns="slot", values="pv_actual_kw").loc[dates]
    pivot_q3 = df.pivot(index="date", columns="slot", values="pv_forecast_q3_latest_kw").loc[dates]
    if pivot_load.shape[1] != SLOTS or pivot_pv.shape[1] != SLOTS:
        raise ValueError("expected exactly 144 slots per day")
    return pivot_load.to_numpy(float), pivot_pv.to_numpy(float), pivot_q3.to_numpy(float), dates


def metrics(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    a = actual.reshape(-1)
    p = pred.reshape(-1)
    mask = np.abs(a) > 1.0
    return {
        "MAE_kW": float(mean_absolute_error(a, p)),
        "RMSE_kW": float(np.sqrt(mean_squared_error(a, p))),
        "MAPE_percent_masked": float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100),
    }


def previous_day(history: np.ndarray, n_test: int) -> np.ndarray:
    out = []
    h = history.copy()
    for _ in range(n_test):
        pred = h[-1].copy()
        out.append(pred)
        # walk-forward update is done by the caller for observed values
        h = np.vstack([h, pred])
    return np.asarray(out)


def rolling_baseline(history: np.ndarray, actual_test: np.ndarray, k: int) -> np.ndarray:
    h = history.copy()
    out = []
    for y in actual_test:
        out.append(h[-k:].mean(axis=0))
        h = np.vstack([h, y])
    return np.asarray(out)


def weighted_lag_1_7(history: np.ndarray, actual_test: np.ndarray, w1: float = 0.7) -> np.ndarray:
    """Use recent-day and same-weekday observations."""
    h = history.copy()
    out = []
    for y in actual_test:
        p = w1 * h[-1] + (1.0 - w1) * h[-7]
        out.append(p)
        h = np.vstack([h, y])
    return np.asarray(out)


def exponential_smoothing(history: np.ndarray, actual_test: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    level = history[-1].copy()
    out = []
    for y in actual_test:
        out.append(level.copy())
        level = alpha * y + (1.0 - alpha) * level
    return np.asarray(out)


def xgb_features(series: np.ndarray, day: int, slot: int) -> np.ndarray:
    def at(d: int, s: int) -> float:
        return float(series[d, s % SLOTS])

    prev = day - 1
    vals = [at(prev, slot), at(day - 2, slot), at(day - 3, slot)]
    vals.extend([at(prev, slot - 1), at(prev, slot + 1)])
    vals.append(float(np.mean(series[day - 3 : day, slot])))
    angle = 2 * np.pi * slot / SLOTS
    vals.extend([np.sin(angle), np.cos(angle)])
    return np.asarray(vals, dtype=float)


def enhanced_features(series: np.ndarray, day: int, slot: int, weekday: int) -> np.ndarray:
    """Lag, seasonal and local-curve features for one target slot."""
    def at(d: int, s: int) -> float:
        return float(series[d, s % SLOTS])

    vals = [at(day - k, slot) for k in (1, 2, 3, 7, 14)]
    vals += [float(np.mean(series[day - 3 : day, slot])), float(np.mean(series[day - 7 : day, slot]))]
    vals += [at(day - 1, slot - 2), at(day - 1, slot - 1), at(day - 1, slot + 1), at(day - 1, slot + 2)]
    angle = 2 * np.pi * slot / SLOTS
    dow_angle = 2 * np.pi * weekday / 7
    vals += [np.sin(angle), np.cos(angle), np.sin(dow_angle), np.cos(dow_angle)]
    return np.asarray(vals, dtype=float)


def enhanced_tree_walk_forward(train: np.ndarray, val: np.ndarray, test: np.ndarray, dates: list[str]) -> np.ndarray:
    history = np.vstack([train, val])
    history_dates = dates[: len(history)]
    models: list[XGBRegressor] = []
    for slot in range(SLOTS):
        X = np.vstack([
            enhanced_features(history, d, slot, pd.Timestamp(history_dates[d]).weekday())
            for d in range(14, len(history))
        ])
        y = history[14:, slot]
        model = XGBRegressor(
            n_estimators=120, max_depth=3, learning_rate=0.04,
            subsample=0.9, colsample_bytree=0.9,
            objective="reg:squarederror", n_jobs=2, random_state=2026,
        )
        model.fit(X, y, verbose=False)
        models.append(model)

    observed = history.copy()
    out = []
    for offset, actual in enumerate(test):
        day = len(observed)
        weekday = pd.Timestamp(dates[day]).weekday()
        row = np.zeros(SLOTS, dtype=float)
        for slot, model in enumerate(models):
            f = enhanced_features(observed, day, slot, weekday).reshape(1, -1)
            row[slot] = max(0.0, float(model.predict(f)[0]))
        out.append(row)
        observed = np.vstack([observed, actual])
    return np.asarray(out)


def ridge_walk_forward(train: np.ndarray, val: np.ndarray, test: np.ndarray, dates: list[str]) -> np.ndarray:
    """Regularized dynamic regression as a transparent time-series model."""
    history = np.vstack([train, val])
    history_dates = dates[: len(history)]
    models: list[Ridge] = []
    for slot in range(SLOTS):
        X = np.vstack([
            enhanced_features(history, d, slot, pd.Timestamp(history_dates[d]).weekday())
            for d in range(14, len(history))
        ])
        y = history[14:, slot]
        model = Ridge(alpha=10.0)
        model.fit(X, y)
        models.append(model)
    observed = history.copy()
    out = []
    for offset, actual in enumerate(test):
        day = len(observed)
        weekday = pd.Timestamp(dates[day]).weekday()
        row = np.zeros(SLOTS, dtype=float)
        for slot, model in enumerate(models):
            f = enhanced_features(observed, day, slot, weekday).reshape(1, -1)
            row[slot] = max(0.0, float(model.predict(f)[0]))
        out.append(row)
        observed = np.vstack([observed, actual])
    return np.asarray(out)


def xgb_walk_forward(train: np.ndarray, val: np.ndarray, test: np.ndarray) -> np.ndarray:
    # Validation is used for model selection only; final models are refit on
    # train+validation and then evaluated by walk-forward on the test dates.
    history = np.vstack([train, val])
    models: list[XGBRegressor] = []
    for slot in range(SLOTS):
        X = np.vstack([xgb_features(history, d, slot) for d in range(3, len(history))])
        y = history[3:, slot]
        model = XGBRegressor(
            n_estimators=80,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            objective="reg:squarederror",
            n_jobs=2,
            random_state=2026,
        )
        model.fit(X, y, verbose=False)
        models.append(model)

    # Features for each test day use observed prior days. This is the proper
    # rolling information set; predictions are not fed back as observations.
    observed = history.copy()
    out = []
    for actual in test:
        day = len(observed)
        row = np.zeros(SLOTS, dtype=float)
        for slot, model in enumerate(models):
            row[slot] = max(0.0, float(model.predict(xgb_features(observed, day, slot).reshape(1, -1))[0]))
        out.append(row)
        # The next day's information set contains today's observed value.
        observed = np.vstack([observed, actual])
    return np.asarray(out)


def sarimax_walk_forward(history: np.ndarray, actual_test: np.ndarray) -> np.ndarray:
    """Fit one small seasonal AR model per intraday slot.

    The seasonal period is seven days because each series contains one value
    per day at a fixed 10-minute slot. This is deliberately conservative.
    """
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    out = np.zeros_like(actual_test, dtype=float)
    for slot in range(SLOTS):
        series = history[:, slot].astype(float)
        model = SARIMAX(
            series,
            order=(1, 0, 0),
            seasonal_order=(1, 0, 0, 7),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        fitted = model.fit(disp=False, maxiter=60)
        out[:, slot] = np.maximum(0.0, fitted.forecast(len(actual_test)))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("公共代码/数据/master_10min.csv"))
    parser.add_argument("--out", type=Path, default=Path("问题2/结果"))
    args = parser.parse_args()

    load, pv, pv_q3, dates = load_daily(args.data)
    n = len(dates)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)
    split1, split2 = n_train, n_train + n_val
    args.out.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    predictions: dict[str, dict[str, np.ndarray]] = {"load": {}, "pv": {}}
    for name, data in [("load", load), ("pv", pv)]:
        train, val, test = data[:split1], data[split1:split2], data[split2:]
        history = np.vstack([train, val])
        preds = {
            "previous_day": rolling_baseline(history, test, 1),
            "mean_3_days": rolling_baseline(history, test, 3),
            "seasonal_7_days": rolling_baseline(history, test, 7),
            "weighted_1_day_7_day": weighted_lag_1_7(history, test),
            "exp_smoothing_alpha_0.4": exponential_smoothing(history, test, 0.4),
            "xgboost": xgb_walk_forward(train, val, test),
            "enhanced_xgboost": enhanced_tree_walk_forward(train, val, test, dates),
            "dynamic_ridge": ridge_walk_forward(train, val, test, dates),
        }
        # SARIMAX is the slowest model, but remains part of this reproducible
        # first trial so the result can be compared with the baselines.
        preds["sarimax"] = sarimax_walk_forward(history, test)
        if name == "pv":
            preds["attachment3_q3_latest"] = pv_q3[split2:]
        predictions[name] = preds
        for method, pred in preds.items():
            item = {"target": name, "method": method, **metrics(test, pred)}
            records.append(item)
        pd.DataFrame({"date": dates[split2:] , **{f"slot_{i:03d}": preds["xgboost"][:, i] for i in range(SLOTS)}}).to_csv(
            args.out / f"forecast_{name}_xgboost_test.csv", index=False
        )
        pd.DataFrame({"date": dates[split2:] , **{f"slot_{i:03d}": preds["enhanced_xgboost"][:, i] for i in range(SLOTS)}}).to_csv(
            args.out / f"forecast_{name}_enhanced_xgboost_test.csv", index=False
        )

    pd.DataFrame(records).sort_values(["target", "RMSE_kW"]).to_csv(args.out / "forecast_model_metrics.csv", index=False)
    (args.out / "forecast_experiment_split.json").write_text(
        json.dumps(
            {"n_days": n, "train_days": n_train, "validation_days": n_val, "test_days": n - split2,
             "train_end": dates[split1 - 1], "validation_end": dates[split2 - 1], "test_start": dates[split2]},
            ensure_ascii=False, indent=2,
        ), encoding="utf-8"
    )
    print(pd.DataFrame(records).sort_values(["target", "RMSE_kW"]).to_string(index=False))


if __name__ == "__main__":
    main()
