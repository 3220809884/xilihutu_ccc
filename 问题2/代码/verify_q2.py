"""独立复核问题2的CSV、JSON与官方Excel模板输出。"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题2" / "结果"
SLOTS = 144
DT = 1 / 6
TOL = 1e-5


def close(left, right, tolerance: float = TOL) -> None:
    a = np.asarray(left, dtype=float)
    b = np.asarray(right, dtype=float)
    if not np.allclose(a, b, atol=tolerance, rtol=0):
        raise AssertionError(f"values differ; max abs error={np.max(np.abs(a-b))}")


def main() -> None:
    summary = json.loads((OUT / "problem2_summary.json").read_text(encoding="utf-8"))
    frame = pd.read_csv(OUT / "problem2_dispatch_all_2025.csv")
    delivered = frame[frame["date"] >= "2025-02-01"].copy()
    daily = pd.read_csv(OUT / "problem2_daily_summary.csv")
    events = pd.read_csv(OUT / "problem2_emergency_events.csv")
    audit = pd.read_csv(OUT / "problem2_information_audit.csv")
    clean = pd.read_csv(OUT / "q2_preprocessed_10min.csv", nrows=1)

    assert summary["attachment3_used"] is False
    forbidden = [
        column
        for column in clean.columns
        if "q3" in column.lower() or "volatile" in column.lower() or "issue_datetime" in column
    ]
    assert not forbidden, f"forbidden problem-3/4 fields found: {forbidden}"
    assert len(frame) == 365 * SLOTS and len(delivered) == 334 * SLOTS
    assert len(daily) == 334 and daily["date"].is_unique
    assert (delivered.groupby("date").size() == SLOTS).all()
    assert np.array_equal(delivered["slot"].to_numpy(), np.tile(np.arange(SLOTS), 334))
    assert (delivered["time_interval"].iloc[::SLOTS] == "00:00-00:10").all()
    assert (delivered["time_interval"].iloc[SLOTS - 1 :: SLOTS] == "23:50-24:00").all()
    assert audit["target_day_actual_used_for_forecast"].astype(str).str.lower().eq("false").all()
    for index, row in audit.iloc[1:].iterrows():
        assert row["history_through_date"] < row["date"], index
        if isinstance(row["model_train_end_date"], str) and row["model_train_end_date"]:
            assert row["model_train_end_date"] < row["date"], index
        if isinstance(row["reserve_history_end"], str) and row["reserve_history_end"]:
            assert row["reserve_history_end"] < row["date"], index

    actual_balance = (
        frame["plan_kw"]
        + frame["adjustment_kw"]
        + frame["emergency_kw"]
        + frame["pv_actual_kw"]
        + frame["discharge_kw"]
        - frame["load_actual_kw"]
        - frame["charge_kw"]
        - frame["spill_kw"]
    )
    planned_slack = (
        frame["plan_kw"]
        + frame["planned_discharge_kw"]
        - frame["planned_charge_kw"]
        - frame["net_forecast_kw"]
    )
    soc_residual = (
        frame["soc_end_kwh"]
        - frame["soc_start_kwh"]
        - 0.9 * DT * frame["charge_kw"]
        + DT / 0.9 * frame["discharge_kw"]
    )
    continuity = (
        frame["soc_start_kwh"].to_numpy()[1:]
        - frame["soc_end_kwh"].to_numpy()[:-1]
    )
    assert np.max(np.abs(actual_balance)) < TOL
    assert planned_slack.min() >= -TOL
    assert np.max(np.abs(soc_residual)) < TOL
    assert np.max(np.abs(continuity)) < TOL
    assert frame["adjustment_kwh"].abs().max() < TOL
    assert (frame["reserve_shortfall_kw"] <= TOL).all()
    for stem in ["plan", "adjustment", "emergency", "charge", "discharge"]:
        close(frame[f"{stem}_kwh"], frame[f"{stem}_kw"] * DT)
    close(delivered["plan_cost_yuan"], delivered["price_yuan_per_kwh"] * delivered["plan_kwh"])
    close(
        delivered["emergency_cost_yuan"],
        5 * delivered["price_yuan_per_kwh"] * delivered["emergency_kwh"],
    )
    close(delivered["total_cost_yuan"], delivered["plan_cost_yuan"] + delivered["emergency_cost_yuan"])
    for column in [
        "plan_kwh",
        "adjustment_kwh",
        "emergency_kwh",
        "charge_kwh",
        "discharge_kwh",
        "plan_cost_yuan",
        "adjustment_cost_yuan",
        "emergency_cost_yuan",
        "total_cost_yuan",
    ]:
        close(delivered[column].sum(), summary[column])
        close(delivered.groupby("date")[column].sum().to_numpy(), daily[column])

    expected_events: list[tuple[str, str, float]] = []
    for date, day in delivered.groupby("date", sort=True):
        positive = day["emergency_kwh"].to_numpy() > 1e-8
        edges = np.diff(np.r_[False, positive, False].astype(int))
        for start, end in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            label = f"{start // 6:02d}:{(start % 6) * 10:02d}-{end // 6:02d}:{(end % 6) * 10:02d}"
            expected_events.append(
                (date, label, float(day["emergency_kwh"].iloc[start:end].sum()))
            )
    assert list(zip(events["date"], events["interval"])) == [
        (date, label) for date, label, _ in expected_events
    ]
    close(events["emergency_kwh"], [value for _, _, value in expected_events])

    # Normal mode is intentional: read-only worksheets make repeated random cell
    # access O(n^2) because each lookup restarts the XML stream.
    result_book = load_workbook(OUT / "result2.xlsx", data_only=False, read_only=False)
    template_book = load_workbook(
        ROOT / "题目/附件/附件5/result2.xlsx", data_only=False, read_only=False
    )
    assert result_book.sheetnames == template_book.sheetnames
    plan = result_book["计划购电量"]
    template_plan = template_book["计划购电量"]
    assert [plan.cell(1, column).value for column in range(1, 148)] == [
        template_plan.cell(1, column).value for column in range(1, 148)
    ]
    for day_index, (date, group) in enumerate(delivered.groupby("date", sort=True)):
        row = day_index + 2
        assert pd.Timestamp(plan.cell(row, 1).value).strftime("%Y-%m-%d") == date
        close([plan.cell(row, column).value for column in range(2, 146)], group["plan_kwh"])
        assert plan.cell(row, 146).value == f"=SUM(B{row}:EO{row})"
        assert isinstance(plan.cell(row, 147).value, str) and plan.cell(row, 147).value.startswith("=SUMPRODUCT(")
    assert plan.cell(336, 1).value is None and plan.cell(336, 2).value is None
    # Explicitly verify the template's shifted header did not shift the data mapping.
    first_day = delivered[delivered["date"] == "2025-02-01"].sort_values("slot")
    close(plan.cell(2, 2).value, first_day.iloc[0]["plan_kwh"])
    assert first_day.iloc[0]["time_interval"] == "00:00-00:10"

    storage = result_book["充放电量"]
    for day_index, (date, group) in enumerate(delivered.groupby("date", sort=True)):
        row = 2 + 6 * day_index
        assert pd.Timestamp(storage.cell(row, 1).value).strftime("%Y-%m-%d") == date
        charge = group["charge_kwh"].to_numpy().reshape(6, 24).sum(axis=1)
        discharge = group["discharge_kwh"].to_numpy().reshape(6, 24).sum(axis=1)
        close([storage.cell(row + block, 3).value for block in range(6)], charge)
        close([storage.cell(row + block, 4).value for block in range(6)], discharge)
        close(
            [storage.cell(row, 6).value, storage.cell(row + 1, 6).value],
            [group["soc_start_kwh"].iloc[0], group["soc_end_kwh"].iloc[-1]],
        )
    assert storage.cell(2006, 1).value is None and storage.cell(2006, 2).value is None

    book_events: list[tuple[str, str, float]] = []
    current_date = None
    for date_value, label, energy in result_book["紧急购电量"].iter_rows(
        min_row=2, max_col=3, values_only=True
    ):
        if date_value is not None:
            current_date = pd.Timestamp(date_value).strftime("%Y-%m-%d")
        if label and label != "无":
            book_events.append((current_date, label, float(energy)))
        elif label == "无":
            close(energy, 0)
    assert [(date, label) for date, label, _ in book_events] == [
        (date, label) for date, label, _ in expected_events
    ]
    close([value for _, _, value in book_events], [value for _, _, value in expected_events])

    print(
        json.dumps(
            {
                "verified": True,
                "attachment3_used": False,
                "delivered_days": 334,
                "delivered_slots": 334 * SLOTS,
                "time_mapping": "B column = slot 0 = 00:00-00:10",
                "emergency_events": len(expected_events),
                "total_cost_yuan": summary["total_cost_yuan"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
