"""Build the canonical 10-minute master table for CUMCM 2026 C.

Canonical convention:
  - one row = one date and one 10-minute slot;
  - time_end is the interval end time: 00:10, ..., 24:00;
  - source load/PV/price values remain in their source units;
  - load_kw and PV_kw are not converted here;
  - model code should call to_energy to multiply kW by 1/6 hour.

The source workbook is never modified. The script writes CSV files under
公共代码/数据 and can be rerun whenever the source files are replaced.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


SLOTS_PER_DAY = 144
DT_HOURS = 1 / 6


def canonical_time_end(slot: int) -> str:
    """Return the end time for a zero-based 10-minute slot."""
    if not 0 <= slot < SLOTS_PER_DAY:
        raise ValueError(f"slot must be in [0, 143], got {slot}")
    minutes = (slot + 1) * 10
    hour, minute = divmod(minutes, 60)
    return f"{hour:02d}:{minute:02d}"


def slot_table() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "slot": np.arange(SLOTS_PER_DAY, dtype=int),
            "time_end": [canonical_time_end(i) for i in range(SLOTS_PER_DAY)],
        }
    )


def _normalise_date(value: object) -> pd.Timestamp:
    parsed = pd.to_datetime(value, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(f"cannot parse date: {value!r}")
    return parsed.normalize()


def _read_wide_daily(path: Path, sheet_name: str) -> pd.DataFrame:
    """Read attachment 2/4 style data into date-slot long format."""
    raw = pd.read_excel(path, sheet_name=sheet_name, header=0)
    if raw.shape[1] != SLOTS_PER_DAY + 1:
        raise ValueError(
            f"{path.name}/{sheet_name}: expected 145 columns, got {raw.shape[1]}"
        )
    raw = raw.rename(columns={raw.columns[0]: "date"})
    raw["date"] = raw["date"].map(_normalise_date)
    values = raw.iloc[:, 1:].to_numpy(dtype=float)
    return pd.DataFrame(
        {
            "date": np.repeat(raw["date"].to_numpy(), SLOTS_PER_DAY),
            "slot": np.tile(np.arange(SLOTS_PER_DAY), len(raw)),
            "value": values.reshape(-1),
        }
    )


def _read_attachment1(path: Path) -> pd.DataFrame:
    raw = pd.read_excel(path, sheet_name=0, header=0)
    if raw.shape[0] != SLOTS_PER_DAY:
        raise ValueError(f"{path.name}: expected 144 rows, got {raw.shape[0]}")
    if raw.shape[1] < 4:
        raise ValueError(f"{path.name}: expected time, price, load and PV columns")
    raw = raw.iloc[:, :4].copy()
    raw.columns = [
        "source_time",
        "price_fixed_yuan_per_kwh",
        "load_kw",
        "pv_forecast_q1_kw",
    ]
    raw.insert(0, "slot", np.arange(SLOTS_PER_DAY, dtype=int))
    raw["time_end"] = raw["slot"].map(canonical_time_end)
    return raw


def _read_attachment3(path: Path) -> pd.DataFrame:
    """Read hourly PV forecasts and preserve issue/target semantics."""
    raw = pd.read_excel(path, sheet_name=0, header=0).copy()
    if raw.shape[1] < 3:
        raise ValueError(f"{path.name}: forecast sheet has too few columns")
    raw.iloc[:, 0] = raw.iloc[:, 0].ffill()
    raw["issue_date"] = raw.iloc[:, 0].map(_normalise_date)
    raw["issue_time"] = raw.iloc[:, 1].astype(str).str.slice(0, 5)
    raw["issue_datetime"] = pd.to_datetime(
        raw["issue_date"].dt.strftime("%Y-%m-%d") + " " + raw["issue_time"],
        errors="coerce",
    )
    forecast_columns = [c for c in raw.columns[2:] if str(c).startswith("预报")]
    records: list[dict[str, object]] = []
    for row_index, row in raw.iterrows():
        issue_dt = row["issue_datetime"]
        if pd.isna(issue_dt):
            raise ValueError(f"{path.name}: invalid issue time in row {row_index}")
        for column in forecast_columns:
            horizon = int(str(column).replace("预报", "").replace("小时", ""))
            records.append(
                {
                    "issue_date": row["issue_date"],
                    "issue_time": row["issue_time"],
                    "issue_datetime": issue_dt,
                    "horizon_hours": horizon,
                    "target_datetime": issue_dt + pd.Timedelta(hours=horizon),
                    "pv_forecast_kw": pd.to_numeric(row[column], errors="coerce"),
                }
            )
    result = pd.DataFrame(records)
    if result.empty:
        raise ValueError(f"{path.name}: no forecast records")
    return result.sort_values(["issue_datetime", "horizon_hours"]).reset_index(drop=True)


def _validate_daily_grid(frame: pd.DataFrame, name: str) -> None:
    counts = frame.groupby("date")["slot"].nunique()
    bad = counts[counts != SLOTS_PER_DAY]
    if not bad.empty:
        raise ValueError(f"{name}: dates without exactly 144 slots: {bad.to_dict()}")
    if frame.duplicated(["date", "slot"]).any():
        raise ValueError(f"{name}: duplicate date-slot rows")


def _interpolate_forecast(
    issue_dt: pd.Timestamp,
    issue_pv_kw: float,
    forecasts: pd.DataFrame,
    targets: Iterable[pd.Timestamp],
) -> dict[pd.Timestamp, float]:
    """Interpolate one issue's hourly forecast to arbitrary target times."""
    points = [(issue_dt, float(issue_pv_kw))]
    for row in forecasts.itertuples(index=False):
        if pd.notna(row.pv_forecast_kw):
            points.append((pd.Timestamp(row.target_datetime), float(row.pv_forecast_kw)))
    points.sort(key=lambda item: item[0])
    x = np.array([item[0].value / 1e9 for item in points], dtype=float)
    y = np.array([max(0.0, item[1]) for item in points], dtype=float)
    unique_x, unique_indices = np.unique(x, return_index=True)
    unique_y = y[unique_indices]
    output: dict[pd.Timestamp, float] = {}
    for target in targets:
        target_x = pd.Timestamp(target).value / 1e9
        if target_x < unique_x[0] or target_x > unique_x[-1]:
            output[pd.Timestamp(target)] = np.nan
        else:
            output[pd.Timestamp(target)] = max(
                0.0, float(np.interp(target_x, unique_x, unique_y))
            )
    return output


def to_energy(power_kw: pd.Series | np.ndarray | float) -> pd.Series | np.ndarray | float:
    """Convert kW to kWh for one 10-minute interval."""
    return power_kw * DT_HOURS


def build_master(source_dir: Path, output_dir: Path) -> tuple[Path, Path]:
    attachment1 = _read_attachment1(source_dir / "附件1.xlsx")
    load = _read_wide_daily(source_dir / "附件2.xlsx", "小区负载")
    pv = _read_wide_daily(source_dir / "附件2.xlsx", "光伏发电实际功率")
    volatile_price = _read_wide_daily(source_dir / "附件4.xlsx", "Sheet1")
    load = load.rename(columns={"value": "load_actual_kw"})
    pv = pv.rename(columns={"value": "pv_actual_kw"})
    volatile_price = volatile_price.rename(columns={"value": "price_volatile_yuan_per_kwh"})
    _validate_daily_grid(load, "附件2/小区负载")
    _validate_daily_grid(pv, "附件2/光伏实际功率")
    _validate_daily_grid(volatile_price, "附件4/波动电价")

    dates = sorted(load["date"].unique())
    if dates != sorted(pv["date"].unique()) or dates != sorted(volatile_price["date"].unique()):
        raise ValueError("附件2 and 附件4 date sets do not match")

    slots = slot_table()
    master = (
        pd.MultiIndex.from_product([dates, slots["slot"]], names=["date", "slot"])
        .to_frame(index=False)
        .merge(load, on=["date", "slot"], how="left")
        .merge(pv, on=["date", "slot"], how="left")
        .merge(volatile_price, on=["date", "slot"], how="left")
    )
    master["time_end"] = master["slot"].map(canonical_time_end)
    fixed = attachment1[["slot", "price_fixed_yuan_per_kwh", "pv_forecast_q1_kw"]]
    master = master.merge(fixed, on="slot", how="left", validate="many_to_one")

    forecast_hourly = _read_attachment3(source_dir / "附件3.xlsx")
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast_path = output_dir / "forecast_attachment3_hourly.csv"
    forecast_hourly.to_csv(forecast_path, index=False, encoding="utf-8-sig")

    master["target_datetime"] = [
        date + pd.Timedelta(minutes=(slot + 1) * 10)
        for date, slot in zip(master["date"], master["slot"])
    ]
    issue_rows: list[dict[str, object]] = []
    issue_groups = {
        pd.Timestamp(issue_dt): group
        for issue_dt, group in forecast_hourly.groupby("issue_datetime")
    }
    for date in dates:
        day_master = master.loc[master["date"] == date]
        day_issues = sorted(
            (issue_dt, group)
            for issue_dt, group in issue_groups.items()
            if issue_dt.normalize() == pd.Timestamp(date).normalize()
        )
        for issue_dt, group in day_issues:
            issue_pv = day_master.loc[
                day_master["target_datetime"] == issue_dt, "pv_actual_kw"
            ]
            issue_pv_value = float(issue_pv.iloc[0]) if len(issue_pv) else 0.0
            values = _interpolate_forecast(
                issue_dt,
                issue_pv_value,
                group,
                day_master["target_datetime"].tolist(),
            )
            for target, value in values.items():
                issue_rows.append(
                    {
                        "date": pd.Timestamp(date),
                        "target_datetime": target,
                        "issue_datetime": issue_dt,
                        "pv_forecast_q3_latest_kw": value,
                    }
                )
    q3 = pd.DataFrame(issue_rows)
    if not q3.empty:
        q3 = q3.sort_values(["date", "target_datetime", "issue_datetime"])
        q3 = q3.dropna(subset=["pv_forecast_q3_latest_kw"])
        q3 = q3.drop_duplicates(["date", "target_datetime"], keep="last")
        master = master.merge(q3, on=["date", "target_datetime"], how="left")
    else:
        master["issue_datetime"] = pd.NaT
        master["pv_forecast_q3_latest_kw"] = np.nan

    master = master.drop(columns=["target_datetime"])
    master = master[
        [
            "date",
            "slot",
            "time_end",
            "price_fixed_yuan_per_kwh",
            "price_volatile_yuan_per_kwh",
            "load_actual_kw",
            "pv_actual_kw",
            "pv_forecast_q1_kw",
            "pv_forecast_q3_latest_kw",
            "issue_datetime",
        ]
    ].sort_values(["date", "slot"])
    master_path = output_dir / "master_10min.csv"
    master.to_csv(master_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")
    return master_path, forecast_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "题目" / "附件",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "数据",
    )
    args = parser.parse_args()
    master_path, forecast_path = build_master(args.source_dir, args.output_dir)
    print(f"wrote {master_path}")
    print(f"wrote {forecast_path}")


if __name__ == "__main__":
    main()
