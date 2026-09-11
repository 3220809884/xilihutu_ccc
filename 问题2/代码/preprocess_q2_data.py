"""Preprocess Problem 2 data without modifying the canonical master table.

Outputs a cleaned copy, an anomaly log, a calendar table and a JSON audit.
Raw power remains in kW. Standardized columns are modelling features only and
must never be used directly in the MILP energy or cost equations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SLOTS = 144
TARGETS = ("load_actual_kw", "pv_actual_kw")

# 2025 mainland China statutory holiday periods and official make-up workdays.
# Kept in one explicit configuration block so the team can audit or replace it.
HOLIDAY_RANGES = (
    ("2025-01-01", "2025-01-01", "元旦"),
    ("2025-01-28", "2025-02-04", "春节"),
    ("2025-04-04", "2025-04-06", "清明节"),
    ("2025-05-01", "2025-05-05", "劳动节"),
    ("2025-05-31", "2025-06-02", "端午节"),
    ("2025-10-01", "2025-10-08", "国庆中秋"),
)
MAKEUP_WORKDAYS = {
    "2025-01-26",
    "2025-02-08",
    "2025-04-27",
    "2025-09-28",
    "2025-10-11",
}


def calendar_table(dates: pd.Series) -> pd.DataFrame:
    unique = pd.Series(pd.to_datetime(dates).dt.normalize().unique()).sort_values()
    cal = pd.DataFrame({"date_dt": unique})
    cal["date"] = cal["date_dt"].dt.strftime("%Y-%m-%d")
    cal["weekday"] = cal["date_dt"].dt.weekday
    cal["month"] = cal["date_dt"].dt.month
    cal["holiday_name"] = ""
    for start, end, name in HOLIDAY_RANGES:
        mask = cal["date_dt"].between(pd.Timestamp(start), pd.Timestamp(end))
        cal.loc[mask, "holiday_name"] = name
    cal["is_holiday"] = cal["holiday_name"].ne("")
    cal["is_makeup_workday"] = cal["date"].isin(MAKEUP_WORKDAYS)
    cal["is_weekend"] = cal["weekday"].ge(5)
    cal["is_workday"] = (~cal["is_holiday"] & ~cal["is_weekend"]) | cal["is_makeup_workday"]
    cal["day_type"] = np.select(
        [cal["is_holiday"], cal["is_makeup_workday"], cal["is_weekend"]],
        ["holiday", "makeup_workday", "weekend"],
        default="workday",
    )
    return cal


def complete_grid(raw: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    raw = raw.copy()
    raw["date"] = pd.to_datetime(raw["date"]).dt.strftime("%Y-%m-%d")
    dates = pd.date_range(raw["date"].min(), raw["date"].max(), freq="D")
    grid = pd.MultiIndex.from_product(
        [dates.strftime("%Y-%m-%d"), range(SLOTS)], names=["date", "slot"]
    ).to_frame(index=False)
    before = len(raw)
    out = grid.merge(raw, on=["date", "slot"], how="left", validate="one_to_one")
    return out.sort_values(["date", "slot"]).reset_index(drop=True), len(out) - before


def fill_missing(frame: pd.DataFrame, column: str) -> tuple[pd.Series, pd.Series]:
    """Fill missing historical observations while preserving an audit flag."""
    original = frame[column].copy()
    filled = frame.groupby("date", sort=False)[column].transform(
        lambda x: x.interpolate(method="linear", limit_direction="both")
    )
    typed_slot_median = frame.assign(_value=filled).groupby(
        ["day_type", "slot"]
    )["_value"].transform("median")
    slot_median = frame.assign(_value=filled).groupby("slot")["_value"].transform("median")
    filled = filled.fillna(typed_slot_median).fillna(slot_median)
    if filled.isna().any():
        raise ValueError(f"{column}: missing values remain after imputation")
    return filled, original.isna()


def isolated_outliers(frame: pd.DataFrame, column: str) -> pd.Series:
    """Conservatively flag isolated sensor spikes, not seasonal level changes.

    A point is flagged only when it is far from the mean of both neighbours,
    the neighbours agree with one another, and the deviation exceeds a robust
    slot-specific threshold. Edge slots are never altered by this rule.
    """
    matrix = frame.pivot(index="date", columns="slot", values=column).to_numpy(float)
    prev = np.roll(matrix, 1, axis=1)
    nxt = np.roll(matrix, -1, axis=1)
    local = (prev + nxt) / 2
    residual = matrix - local
    residual[:, (0, -1)] = np.nan
    median = np.zeros(SLOTS, dtype=float)
    mad = np.zeros(SLOTS, dtype=float)
    median[1:-1] = np.nanmedian(residual[:, 1:-1], axis=0)
    mad[1:-1] = np.nanmedian(
        np.abs(residual[:, 1:-1] - median[1:-1]), axis=0
    )
    scale = 1.4826 * mad
    magnitude_floor = np.maximum(50.0, 0.15 * np.nanmedian(np.abs(matrix), axis=0))
    threshold = np.maximum(10.0 * scale, magnitude_floor)
    neighbours_agree = np.abs(prev - nxt) <= np.maximum(50.0, 0.05 * np.abs(local))
    flagged = (np.abs(residual - median) > threshold) & neighbours_agree
    flagged[:, (0, -1)] = False
    return pd.Series(flagged.reshape(-1), index=frame.index)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=Path, default=ROOT / "公共代码/数据/master_10min.csv"
    )
    parser.add_argument("--out", type=Path, default=ROOT / "问题2/结果")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    raw = pd.read_csv(args.input)
    required = {"date", "slot", *TARGETS}
    missing_columns = required.difference(raw.columns)
    if missing_columns:
        raise ValueError(f"missing required columns: {sorted(missing_columns)}")
    if raw.duplicated(["date", "slot"]).any():
        raise ValueError("duplicate date-slot keys in canonical master table")

    data, inserted_rows = complete_grid(raw)
    cal = calendar_table(pd.to_datetime(data["date"]))
    data = data.merge(cal.drop(columns="date_dt"), on="date", how="left", validate="many_to_one")

    events: list[dict[str, object]] = []
    audit: dict[str, object] = {
        "source": str(args.input.relative_to(ROOT)),
        "rows_before": int(len(raw)),
        "rows_after": int(len(data)),
        "inserted_date_slot_rows": int(inserted_rows),
        "date_start": data["date"].min(),
        "date_end": data["date"].max(),
        "day_type_counts": cal["day_type"].value_counts().sort_index().to_dict(),
        "targets": {},
    }

    for column in TARGETS:
        data[f"{column}_original"] = data[column]
        filled, missing_flag = fill_missing(data, column)
        invalid_flag = (~np.isfinite(filled)) | (filled < 0)
        data[column] = filled.clip(lower=0)
        outlier_flag = isolated_outliers(data, column)
        repair_flag = missing_flag | invalid_flag | outlier_flag

        matrix = data.pivot(index="date", columns="slot", values=column).to_numpy(float)
        neighbour_mean = (np.roll(matrix, 1, axis=1) + np.roll(matrix, -1, axis=1)) / 2
        replacement = pd.Series(neighbour_mean.reshape(-1), index=data.index)
        # Missing values have already been interpolated. Only isolated spikes are
        # replaced here; raw values stay available in the *_original column.
        data.loc[outlier_flag, column] = replacement.loc[outlier_flag].clip(lower=0)

        data[f"{column}_was_missing"] = missing_flag
        data[f"{column}_was_outlier"] = outlier_flag
        data[f"{column}_was_repaired"] = repair_flag
        for idx in data.index[repair_flag]:
            reasons = []
            if missing_flag.loc[idx]:
                reasons.append("missing")
            if invalid_flag.loc[idx]:
                reasons.append("invalid")
            if outlier_flag.loc[idx]:
                reasons.append("isolated_outlier")
            events.append(
                {
                    "date": data.at[idx, "date"],
                    "slot": int(data.at[idx, "slot"]),
                    "column": column,
                    "reason": "+".join(reasons),
                    "original_value": data.at[idx, f"{column}_original"],
                    "cleaned_value": data.at[idx, column],
                }
            )
        audit["targets"][column] = {
            "missing_cells": int(missing_flag.sum()),
            "invalid_cells": int(invalid_flag.sum()),
            "isolated_outliers": int(outlier_flag.sum()),
            "repaired_cells": int(repair_flag.sum()),
        }

    # Standardization parameters are estimated from the first 70% of dates only.
    # The original kW columns remain available for optimization and reporting.
    ordered_dates = sorted(data["date"].unique())
    train_end_index = int(len(ordered_dates) * 0.70)
    train_dates = set(ordered_dates[:train_end_index])
    train_mask = data["date"].isin(train_dates)
    scaler: dict[str, object] = {"fit_through": ordered_dates[train_end_index - 1]}
    for column in TARGETS:
        mean = float(data.loc[train_mask, column].mean())
        std = float(data.loc[train_mask, column].std(ddof=0))
        if not np.isfinite(std) or std <= 0:
            raise ValueError(f"{column}: invalid standard deviation {std}")
        data[f"{column}_z"] = (data[column] - mean) / std
        scaler[column] = {"mean": mean, "std": std}

    if len(data) != len(ordered_dates) * SLOTS:
        raise AssertionError("preprocessed table is not a complete 144-slot daily grid")
    if data[list(TARGETS)].isna().any().any():
        raise AssertionError("cleaned target columns contain missing values")

    cleaned_path = args.out / "q2_preprocessed_10min.csv"
    events_path = args.out / "q2_preprocessing_events.csv"
    calendar_path = args.out / "q2_calendar_2025.csv"
    data.to_csv(cleaned_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(events, columns=[
        "date", "slot", "column", "reason", "original_value", "cleaned_value"
    ]).to_csv(events_path, index=False, encoding="utf-8-sig")
    cal.drop(columns="date_dt").to_csv(calendar_path, index=False, encoding="utf-8-sig")
    audit["standardization"] = scaler
    (args.out / "q2_preprocessing_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
