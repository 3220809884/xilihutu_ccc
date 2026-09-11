"""Compare calendar-aware next-day forecasting models for Problem 2.

The evaluation is strictly date ordered. Models are fitted on train+validation
dates; each test-day feature vector may use completed earlier test days but
never the target day's actual values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
SLOTS = 144
LAGS = (1, 2, 3, 7, 14)
TYPE_COLUMNS = ("holiday", "makeup_workday", "weekend", "workday")


def metric_row(actual: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    a, p = actual.reshape(-1), np.maximum(pred.reshape(-1), 0)
    mask = np.abs(a) > 1.0
    return {
        "MAE_kW": float(mean_absolute_error(a, p)),
        "RMSE_kW": float(np.sqrt(mean_squared_error(a, p))),
        "MAPE_percent_masked": float(np.mean(np.abs((a[mask] - p[mask]) / a[mask])) * 100),
    }


def feature_row(series: np.ndarray, day: int, slot: int, calendar: pd.DataFrame,
                include_calendar: bool) -> np.ndarray:
    def at(d: int, s: int) -> float:
        return float(series[d, s % SLOTS])

    values = [at(day - lag, slot) for lag in LAGS]
    values += [
        float(np.mean(series[day - 3:day, slot])),
        float(np.mean(series[day - 7:day, slot])),
        float(np.mean(series[day - 14:day, slot])),
        at(day - 1, slot - 2),
        at(day - 1, slot - 1),
        at(day - 1, slot + 1),
        at(day - 1, slot + 2),
    ]
    slot_angle = 2 * np.pi * slot / SLOTS
    values += [np.sin(slot_angle), np.cos(slot_angle)]
    if include_calendar:
        row = calendar.iloc[day]
        weekday_angle = 2 * np.pi * int(row.weekday) / 7
        month_angle = 2 * np.pi * (int(row.month) - 1) / 12
        values += [
            np.sin(weekday_angle), np.cos(weekday_angle),
            np.sin(month_angle), np.cos(month_angle),
            *[float(row.day_type == name) for name in TYPE_COLUMNS],
        ]
    return np.asarray(values, dtype=float)


def sample_matrix(series: np.ndarray, calendar: pd.DataFrame, days: range,
                  include_calendar: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs, ys, groups = [], [], []
    for day in days:
        for slot in range(SLOTS):
            xs.append(feature_row(series, day, slot, calendar, include_calendar))
            ys.append(series[day, slot])
            groups.append(calendar.iloc[day].day_type)
    return np.asarray(xs), np.asarray(ys), np.asarray(groups)


def fit_ridge(X: np.ndarray, y: np.ndarray):
    # StandardScaler is fitted only on the training matrix inside the pipeline.
    model = make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="lsqr"))
    model.fit(X, y)
    return model


def fit_xgb(X: np.ndarray, y: np.ndarray):
    model = XGBRegressor(
        n_estimators=220,
        max_depth=5,
        learning_rate=0.04,
        min_child_weight=4,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        objective="reg:squarederror",
        n_jobs=4,
        random_state=2026,
    )
    model.fit(X, y, verbose=False)
    return model


def predict_days(model, series: np.ndarray, calendar: pd.DataFrame,
                 days: range, include_calendar: bool) -> np.ndarray:
    out = np.zeros((len(days), SLOTS), dtype=float)
    for row, day in enumerate(days):
        X = np.vstack([
            feature_row(series, day, slot, calendar, include_calendar)
            for slot in range(SLOTS)
        ])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            prediction = model.predict(X)
        if not np.isfinite(prediction).all():
            raise ValueError(f"non-finite prediction for day index {day}")
        out[row] = np.maximum(prediction, 0)
    return out


def grouped_prediction(models: dict[str, object], fallback, series: np.ndarray,
                       calendar: pd.DataFrame, days: range) -> np.ndarray:
    out = np.zeros((len(days), SLOTS), dtype=float)
    for row, day in enumerate(days):
        day_type = str(calendar.iloc[day].day_type)
        model = models.get(day_type, fallback)
        X = np.vstack([
            feature_row(series, day, slot, calendar, True) for slot in range(SLOTS)
        ])
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            prediction = model.predict(X)
        if not np.isfinite(prediction).all():
            raise ValueError(f"non-finite grouped prediction for day index {day}")
        out[row] = np.maximum(prediction, 0)
    return out


def walk_baseline(series: np.ndarray, split: int, method: str) -> np.ndarray:
    predictions = []
    for day in range(split, len(series)):
        if method == "previous_day":
            pred = series[day - 1]
        elif method == "mean_3_days":
            pred = series[day - 3:day].mean(axis=0)
        elif method == "same_weekday_4weeks":
            pred = series[[day - 7, day - 14, day - 21, day - 28]].mean(axis=0)
        else:
            raise ValueError(method)
        predictions.append(pred)
    return np.asarray(predictions)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data", type=Path, default=ROOT / "问题2/结果/q2_preprocessed_10min.csv"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "问题2/结果")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.data).sort_values(["date", "slot"]).reset_index(drop=True)
    dates = df["date"].drop_duplicates().tolist()
    if len(df) != len(dates) * SLOTS or df.duplicated(["date", "slot"]).any():
        raise ValueError("preprocessed input must contain a complete 144-slot daily grid")
    calendar = df.drop_duplicates("date")[["date", "weekday", "month", "day_type"]].reset_index(drop=True)
    n = len(dates)
    split1 = int(n * 0.70)
    split2 = split1 + int(n * 0.15)
    train_days = range(14, split2)
    test_days = range(split2, n)
    test_calendar = calendar.iloc[split2:].reset_index(drop=True)

    records: list[dict[str, object]] = []
    best_predictions: dict[str, np.ndarray] = {}
    all_predictions: dict[str, dict[str, np.ndarray]] = {}

    for target, column in (("load", "load_actual_kw"), ("pv", "pv_actual_kw")):
        series = df.pivot(index="date", columns="slot", values=column).loc[dates].to_numpy(float)
        actual = series[split2:]
        target_predictions: dict[str, np.ndarray] = {
            method: walk_baseline(series, split2, method)
            for method in ("previous_day", "mean_3_days", "same_weekday_4weeks")
        }

        X_plain, y, _ = sample_matrix(series, calendar, train_days, False)
        plain_ridge = fit_ridge(X_plain, y)
        target_predictions["standardized_ridge_no_calendar"] = predict_days(
            plain_ridge, series, calendar, test_days, False
        )

        X_calendar, y, groups = sample_matrix(series, calendar, train_days, True)
        calendar_ridge = fit_ridge(X_calendar, y)
        target_predictions["standardized_ridge_calendar"] = predict_days(
            calendar_ridge, series, calendar, test_days, True
        )
        calendar_xgb = fit_xgb(X_calendar, y)
        target_predictions["calendar_xgboost"] = predict_days(
            calendar_xgb, series, calendar, test_days, True
        )

        grouped_models: dict[str, object] = {}
        for day_type in TYPE_COLUMNS:
            mask = groups == day_type
            # At least seven full day-profiles are required for an independent model.
            if int(mask.sum()) >= 7 * SLOTS:
                grouped_models[day_type] = fit_xgb(X_calendar[mask], y[mask])
        target_predictions["grouped_calendar_xgboost"] = grouped_prediction(
            grouped_models, calendar_xgb, series, calendar, test_days
        )

        all_predictions[target] = target_predictions
        for method, pred in target_predictions.items():
            overall = {"target": target, "method": method, "day_type": "all", **metric_row(actual, pred)}
            records.append(overall)
            for day_type in sorted(test_calendar.day_type.unique()):
                mask = test_calendar.day_type.eq(day_type).to_numpy()
                if mask.any():
                    records.append({
                        "target": target,
                        "method": method,
                        "day_type": day_type,
                        **metric_row(actual[mask], pred[mask]),
                    })
        overall_rows = [r for r in records if r["target"] == target and r["day_type"] == "all"]
        best_method = min(overall_rows, key=lambda r: r["RMSE_kW"])["method"]
        best_predictions[target] = target_predictions[str(best_method)]

    metrics = pd.DataFrame(records).sort_values(["target", "day_type", "RMSE_kW"])
    metrics.to_csv(args.out / "forecast_calendar_model_metrics.csv", index=False, encoding="utf-8-sig")
    for target, pred in best_predictions.items():
        pd.DataFrame({
            "date": dates[split2:],
            **{f"slot_{slot:03d}": pred[:, slot] for slot in range(SLOTS)},
        }).to_csv(args.out / f"forecast_{target}_best_calendar_test.csv", index=False, encoding="utf-8-sig")

    summary = {
        "data": str(args.data.relative_to(ROOT)),
        "train_days": split1,
        "validation_days": split2 - split1,
        "test_days": n - split2,
        "train_validation_end": dates[split2 - 1],
        "test_start": dates[split2],
        "test_end": dates[-1],
        "test_day_type_counts": test_calendar.day_type.value_counts().sort_index().to_dict(),
        "best_by_target_rmse": {},
    }
    for target in ("load", "pv"):
        subset = metrics[(metrics.target == target) & (metrics.day_type == "all")]
        best = subset.sort_values("RMSE_kW").iloc[0]
        summary["best_by_target_rmse"][target] = {
            "method": best.method,
            "MAE_kW": float(best.MAE_kW),
            "RMSE_kW": float(best.RMSE_kW),
            "MAPE_percent_masked": float(best.MAPE_percent_masked),
        }
    (args.out / "forecast_calendar_experiment_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(metrics[metrics.day_type == "all"].to_string(index=False))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
