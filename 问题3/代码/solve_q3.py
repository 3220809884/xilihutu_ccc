"""Problem 3: causal multi-stage rolling MPC with 0/6/12/18 PV forecasts.

Power variables are kW, stored energy is kWh, and every slot is 1/6 hour.
Only completed load observations and the forecast available at the issue time
are used.  The delivered period is 2025-02-01 through 2025-12-31.
"""
from __future__ import annotations

import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题3" / "结果"
DT = 1 / 6
ETA_CHARGE = ETA_DISCHARGE = 0.9
POWER_MAX_KW = 5000.0
SOC_MIN_KWH, SOC_MAX_KWH = 1200.0, 10800.0
SOC_INITIAL_KWH = SOC_TARGET_KWH = 6000.0
SLOTS = 144
UPDATE_HOURS = (6, 12, 18)
TOL = 1e-6
DELIVERY_START = "2025-02-01"
SPECIFIED_DATES = ("2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21")


def load_forecast(load: np.ndarray, day: int, window: int, quantile: float) -> np.ndarray:
    """Forecast each slot from completed historical days only."""
    if day == 0:
        return np.zeros(SLOTS)
    return np.quantile(load[max(0, day - window):day], quantile, axis=0, method="linear")


def pv_forecast_10min(
    hourly: dict[tuple[str, int], np.ndarray], dates: list[str], pv_actual: np.ndarray,
    day: int, issue_hour: int, start_slot: int,
) -> np.ndarray:
    """Interpolate issue+h hourly forecasts to 10-minute interval-end powers.

    The issue-time anchor is the latest PV measurement already available.  A
    forecast issued at 6:00 first affects the interval 6:00-6:10 (slot 36).
    """
    values = hourly[(dates[day], issue_hour)]
    if values.shape != (24,):
        raise ValueError("Each forecast issue must contain horizons 1..24")
    current = pv_actual[day - 1, -1] if issue_hour == 0 and day else (
        0.0 if issue_hour == 0 else pv_actual[day, issue_hour * 6 - 1]
    )
    anchor_hours = np.arange(issue_hour, issue_hour + 25, dtype=float)
    anchor_values = np.r_[current, values]
    target_hours = (np.arange(start_slot, SLOTS) + 1) * DT
    return np.interp(target_hours, anchor_hours, anchor_values)


def solve_schedule(
    net_forecast_kw: np.ndarray, price: np.ndarray, initial_soc_kwh: float,
    base_plan_kw: np.ndarray | None = None,
) -> dict[str, np.ndarray | float]:
    """Solve a plan MILP or an adjustment MILP for the remaining horizon.

    When base_plan_kw is supplied, the original plan cost is sunk.  The
    adjustment settlement is
      1.5*pi*increase*dt - 0.5*pi*decrease*dt.
    This convex piecewise-linear cost is represented by an epigraph variable.
    """
    n = len(net_forecast_kw)
    if n == 0 or price.shape != (n,):
        raise ValueError("Invalid optimization horizon")
    q, c, d, s, z = 0, n, 2 * n, 3 * n, 4 * n + 1
    adjustment = base_plan_kw is not None
    a = 5 * n + 1
    size = 6 * n + 1 if adjustment else 5 * n + 1
    obj = np.zeros(size)
    if adjustment:
        obj[a:a + n] = 1.0
    else:
        obj[q:q + n] = price * DT
    lb = np.zeros(size)
    ub = np.full(size, np.inf)
    ub[c:c + n] = POWER_MAX_KW
    ub[d:d + n] = POWER_MAX_KW
    lb[s:s + n + 1], ub[s:s + n + 1] = SOC_MIN_KWH, SOC_MAX_KWH
    clipped_initial = float(np.clip(initial_soc_kwh, SOC_MIN_KWH, SOC_MAX_KWH))
    lb[s] = ub[s] = clipped_initial
    lb[s + n] = ub[s + n] = SOC_TARGET_KWH
    ub[z:z + n] = 1.0
    if adjustment:
        lb[a:a + n] = -np.inf
    matrix = lil_matrix((nconstraints := (6 * n if adjustment else 4 * n), size))
    lower = np.full(nconstraints, -np.inf)
    upper = np.full(nconstraints, np.inf)
    integrality = np.zeros(size, dtype=int)
    integrality[z:z + n] = 1
    for t in range(n):
        matrix[t, q + t], matrix[t, c + t], matrix[t, d + t] = 1, -1, 1
        lower[t] = net_forecast_kw[t]
        matrix[n + t, s + t + 1], matrix[n + t, s + t] = 1, -1
        matrix[n + t, c + t] = -ETA_CHARGE * DT
        matrix[n + t, d + t] = DT / ETA_DISCHARGE
        lower[n + t] = upper[n + t] = 0
        matrix[2 * n + t, c + t], matrix[2 * n + t, z + t] = 1, -POWER_MAX_KW
        upper[2 * n + t] = 0
        matrix[3 * n + t, d + t], matrix[3 * n + t, z + t] = 1, POWER_MAX_KW
        upper[3 * n + t] = POWER_MAX_KW
        if adjustment:
            base = float(base_plan_kw[t])
            half = 0.5 * price[t] * DT
            one_half = 1.5 * price[t] * DT
            matrix[4 * n + t, a + t], matrix[4 * n + t, q + t] = 1, -half
            lower[4 * n + t] = -half * base
            matrix[5 * n + t, a + t], matrix[5 * n + t, q + t] = 1, -one_half
            lower[5 * n + t] = -one_half * base
    result = milp(
        obj, integrality=integrality, bounds=Bounds(lb, ub),
        constraints=LinearConstraint(matrix.tocsc(), lower, upper),
        options={"mip_rel_gap": 1e-9, "time_limit": 30.0},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    x = result.x
    purchase = np.maximum(x[q:q + n], 0)
    charge = np.maximum(x[c:c + n], 0)
    discharge = np.maximum(x[d:d + n], 0)
    states = x[s:s + n + 1]
    residual = np.diff(states) - ETA_CHARGE * charge * DT + discharge * DT / ETA_DISCHARGE
    assert np.abs(residual).max() < TOL
    assert (purchase + discharge - charge - net_forecast_kw).min() >= -TOL
    assert not ((charge > TOL) & (discharge > TOL)).any()
    direct_cost = float(price @ purchase * DT)
    if adjustment:
        diff = purchase - base_plan_kw
        direct_objective = float(np.sum(np.where(diff >= 0, 1.5 * price * diff * DT, 0.5 * price * diff * DT)))
        assert abs(direct_objective - result.fun) < TOL
    else:
        direct_objective = direct_cost
        assert abs(direct_cost - result.fun) < TOL
    return {
        "purchase": purchase, "charge": charge, "discharge": discharge,
        "soc": states, "objective": direct_objective, "gap": float(result.mip_gap),
    }


def execute_segment(purchase_kw: np.ndarray, net_actual_kw: np.ndarray, initial_soc: float) -> dict:
    """Causal storage feedback and emergency purchase for executed slots."""
    n = len(purchase_kw)
    charge = np.zeros(n)
    discharge = np.zeros(n)
    emergency = np.zeros(n)
    surplus = np.zeros(n)
    soc = np.zeros(n + 1)
    soc[0] = initial_soc
    for t in range(n):
        balance = purchase_kw[t] - net_actual_kw[t]
        if balance >= 0:
            charge[t] = min(balance, POWER_MAX_KW, max(0, (SOC_MAX_KWH - soc[t]) / (ETA_CHARGE * DT)))
            surplus[t] = balance - charge[t]
        else:
            discharge[t] = min(-balance, POWER_MAX_KW, max(0, (soc[t] - SOC_MIN_KWH) * ETA_DISCHARGE / DT))
            emergency[t] = -balance - discharge[t]
        soc[t + 1] = soc[t] + ETA_CHARGE * charge[t] * DT - discharge[t] * DT / ETA_DISCHARGE
    return {"charge": charge, "discharge": discharge, "emergency": emergency, "surplus": surplus, "soc": soc}


def simulate(
    load: np.ndarray, pv: np.ndarray, price: np.ndarray,
    hourly: dict[tuple[str, int], np.ndarray], dates: list[str],
    start_day: int, end_day: int, initial_soc: float, window: int, quantile: float,
    update_hours: tuple[int, ...], keep_rows: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    rows: list[pd.DataFrame] = []
    daily_rows: list[dict] = []
    soc_initial = initial_soc
    max_gap = 0.0
    for day in range(start_day, end_day):
        date = dates[day]
        load_hat = load_forecast(load, day, window, quantile)
        pv_hat_0 = pv_forecast_10min(hourly, dates, pv, day, 0, 0)
        plan = solve_schedule(load_hat - pv_hat_0, price, soc_initial)
        max_gap = max(max_gap, float(plan["gap"]))
        plan_kw = np.asarray(plan["purchase"])
        adjusted_kw = np.zeros(SLOTS)
        forecast_used = np.zeros(SLOTS)
        issue_used = np.zeros(SLOTS, dtype=int)
        charge = np.zeros(SLOTS)
        discharge = np.zeros(SLOTS)
        emergency = np.zeros(SLOTS)
        surplus = np.zeros(SLOTS)
        soc_start = np.zeros(SLOTS)
        soc_end = np.zeros(SLOTS)
        boundaries = [0] + [h * 6 for h in update_hours] + [SLOTS]
        current_soc = soc_initial
        current_schedule = plan_kw.copy()
        current_pv_forecast = pv_hat_0.copy()
        for segment_index, start in enumerate(boundaries[:-1]):
            issue_hour = start // 6
            if start:
                current_pv_forecast = pv_forecast_10min(hourly, dates, pv, day, issue_hour, start)
                update = solve_schedule(
                    load_hat[start:] - current_pv_forecast,
                    price[start:], current_soc, base_plan_kw=plan_kw[start:],
                )
                current_schedule = np.asarray(update["purchase"])
                max_gap = max(max_gap, float(update["gap"]))
            stop = boundaries[segment_index + 1]
            length = stop - start
            scheduled = current_schedule[:length] if start else current_schedule[start:stop]
            pv_segment = current_pv_forecast[:length] if start else current_pv_forecast[start:stop]
            actual_net = load[day, start:stop] - pv[day, start:stop]
            executed = execute_segment(scheduled, actual_net, current_soc)
            adjusted_kw[start:stop] = scheduled
            forecast_used[start:stop] = pv_segment
            issue_used[start:stop] = issue_hour
            charge[start:stop] = executed["charge"]
            discharge[start:stop] = executed["discharge"]
            emergency[start:stop] = executed["emergency"]
            surplus[start:stop] = executed["surplus"]
            soc_start[start:stop] = executed["soc"][:-1]
            soc_end[start:stop] = executed["soc"][1:]
            current_soc = float(executed["soc"][-1])
        diff = adjusted_kw - plan_kw
        adjustment_cost = np.where(diff >= 0, 1.5 * price * diff * DT, 0.5 * price * diff * DT)
        plan_cost = price * plan_kw * DT
        emergency_cost = 5 * price * emergency * DT
        daily_rows.append({
            "date": date,
            "plan_kwh": float(plan_kw.sum() * DT),
            "adjusted_kwh": float(adjusted_kw.sum() * DT),
            "increase_kwh": float(np.maximum(diff, 0).sum() * DT),
            "decrease_kwh": float(np.maximum(-diff, 0).sum() * DT),
            "emergency_kwh": float(emergency.sum() * DT),
            "charge_kwh": float(charge.sum() * DT),
            "discharge_kwh": float(discharge.sum() * DT),
            "plan_cost_yuan": float(plan_cost.sum()),
            "adjustment_cost_yuan": float(adjustment_cost.sum()),
            "ordinary_settlement_cost_yuan": float(plan_cost.sum() + adjustment_cost.sum()),
            "emergency_cost_yuan": float(emergency_cost.sum()),
            "total_cost_yuan": float(plan_cost.sum() + adjustment_cost.sum() + emergency_cost.sum()),
            "soc_start_kwh": float(soc_initial), "soc_end_kwh": float(current_soc),
        })
        if keep_rows:
            block = pd.DataFrame({
                "date": date, "slot": np.arange(SLOTS),
                "time_end": ["24:00" if t == SLOTS - 1 else f"{(t + 1) // 6:02d}:{((t + 1) % 6) * 10:02d}" for t in range(SLOTS)],
                "price_yuan_per_kwh": price, "load_actual_kw": load[day], "pv_actual_kw": pv[day],
                "load_forecast_kw": load_hat, "pv_forecast_00_kw": pv_hat_0,
                "pv_forecast_used_kw": forecast_used, "forecast_issue_hour": issue_used,
                "plan_kw": plan_kw, "adjusted_kw": adjusted_kw,
                "charge_kw": charge, "discharge_kw": discharge, "emergency_kw": emergency,
                "surplus_kw": surplus, "soc_start_kwh": soc_start, "soc_end_kwh": soc_end,
                "plan_cost_yuan": plan_cost, "adjustment_cost_yuan": adjustment_cost,
                "emergency_cost_yuan": emergency_cost,
            })
            for stem in ("plan", "adjusted", "charge", "discharge", "emergency"):
                block[stem + "_kwh"] = block[stem + "_kw"] * DT
            block["ordinary_settlement_cost_yuan"] = block.plan_cost_yuan + block.adjustment_cost_yuan
            block["total_cost_yuan"] = block.ordinary_settlement_cost_yuan + block.emergency_cost_yuan
            rows.append(block)
        soc_initial = current_soc
    frame = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    return frame, pd.DataFrame(daily_rows), soc_initial, max_gap


def emergency_events(frame: pd.DataFrame) -> pd.DataFrame:
    events = []
    for date, day in frame.groupby("date", sort=True):
        values = day.emergency_kwh.to_numpy()
        positive = values > 1e-8
        edges = np.diff(np.r_[False, positive, False].astype(int))
        for start, stop in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            def label(slot: int) -> str:
                return f"{slot // 6:02d}:{slot % 6 * 10:02d}"
            events.append({
                "date": date, "start_slot": int(start), "end_slot_exclusive": int(stop),
                "interval": f"{label(start)}-{label(stop)}",
                "emergency_kwh": float(values[start:stop].sum()),
                "emergency_cost_yuan": float(day.emergency_cost_yuan.iloc[start:stop].sum()),
            })
    return pd.DataFrame(events, columns=["date", "start_slot", "end_slot_exclusive", "interval", "emergency_kwh", "emergency_cost_yuan"])


def validate(frame: pd.DataFrame) -> dict:
    assert len(frame) == 334 * SLOTS and not frame.duplicated(["date", "slot"]).any()
    for stem in ("plan", "adjusted", "charge", "discharge", "emergency"):
        assert frame[stem + "_kw"].min() >= -TOL
        assert np.abs(frame[stem + "_kwh"] - frame[stem + "_kw"] * DT).max() < TOL
    assert frame[["charge_kw", "discharge_kw"]].max().max() <= POWER_MAX_KW + TOL
    assert not ((frame.charge_kw > TOL) & (frame.discharge_kw > TOL)).any()
    assert frame[["soc_start_kwh", "soc_end_kwh"]].min().min() >= SOC_MIN_KWH - TOL
    assert frame[["soc_start_kwh", "soc_end_kwh"]].max().max() <= SOC_MAX_KWH + TOL
    balance = frame.adjusted_kw + frame.emergency_kw + frame.pv_actual_kw + frame.discharge_kw - frame.load_actual_kw - frame.charge_kw - frame.surplus_kw
    soc = frame.soc_end_kwh - frame.soc_start_kwh - ETA_CHARGE * frame.charge_kw * DT + frame.discharge_kw * DT / ETA_DISCHARGE
    continuity = frame.soc_start_kwh.to_numpy()[1:] - frame.soc_end_kwh.to_numpy()[:-1]
    assert np.abs(balance).max() < TOL and np.abs(soc).max() < TOL and np.abs(continuity).max() < TOL
    diff = frame.adjusted_kw - frame.plan_kw
    settlement = np.where(diff >= 0, 1.5 * frame.price_yuan_per_kwh * diff * DT, 0.5 * frame.price_yuan_per_kwh * diff * DT)
    assert np.abs(settlement - frame.adjustment_cost_yuan).max() < TOL
    assert np.abs(frame.emergency_cost_yuan - 5 * frame.price_yuan_per_kwh * frame.emergency_kwh).max() < TOL
    assert (frame.forecast_issue_hour <= np.floor(frame.slot / 6)).all()
    return {
        "max_balance_residual_kw": float(np.abs(balance).max()),
        "max_soc_transition_residual_kwh": float(np.abs(soc).max()),
        "max_soc_continuity_residual_kwh": float(np.abs(continuity).max()),
        "simultaneous_charge_discharge_slots": 0, "all_assertions_passed": True,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    master_path = ROOT / "公共代码" / "数据" / "master_10min.csv"
    forecast_path = ROOT / "公共代码" / "数据" / "forecast_attachment3_hourly.csv"
    master = pd.read_csv(master_path, usecols=["date", "slot", "price_fixed_yuan_per_kwh", "load_actual_kw", "pv_actual_kw"])
    master = master.sort_values(["date", "slot"]).reset_index(drop=True)
    dates = master.date.drop_duplicates().tolist()
    assert dates == pd.date_range("2025-01-01", "2025-12-31").strftime("%Y-%m-%d").tolist()
    load = master.load_actual_kw.to_numpy().reshape(365, SLOTS)
    pv = master.pv_actual_kw.to_numpy().reshape(365, SLOTS)
    prices = master.price_fixed_yuan_per_kwh.to_numpy().reshape(365, SLOTS)
    assert np.all(prices == prices[0])
    price = prices[0]
    raw_forecasts = pd.read_csv(forecast_path)
    raw_forecasts["issue_hour"] = raw_forecasts.issue_time.str.split(":").str[0].astype(int)
    raw_forecasts = raw_forecasts.sort_values(["issue_date", "issue_hour", "horizon_hours"])
    assert len(raw_forecasts) == 365 * 4 * 24
    hourly = {
        (date, int(hour)): group.pv_forecast_kw.to_numpy(float)
        for (date, hour), group in raw_forecasts.groupby(["issue_date", "issue_hour"], sort=False)
    }
    # Operational January uses a fixed conservative load quantile and all updates.
    jan_frame, jan_daily, feb_initial, _ = simulate(
        load, pv, price, hourly, dates, 0, 31, SOC_INITIAL_KWH, 14, 0.8, UPDATE_HOURS, True,
    )
    validation_start_soc = float(jan_daily.loc[jan_daily.date == "2025-01-15", "soc_start_kwh"].iloc[0])
    candidates = []
    for window in (7, 14, 28):
        for quantile in (0.5, 0.65, 0.8, 0.9, 0.95):
            _, trial, terminal, _ = simulate(
                load, pv, price, hourly, dates, 14, 31, validation_start_soc,
                window, quantile, UPDATE_HOURS, False,
            )
            raw_cost = float(trial.total_cost_yuan.sum())
            inventory_adjustment = (SOC_TARGET_KWH - terminal) * float(price.min()) / ETA_CHARGE
            candidates.append({
                "window_days": window, "quantile": quantile,
                "validation_cost_yuan": raw_cost, "terminal_soc_kwh": terminal,
                "inventory_adjustment_yuan": inventory_adjustment,
                "score_yuan": raw_cost + inventory_adjustment,
                "emergency_kwh": float(trial.emergency_kwh.sum()),
            })
            print(f"validation W={window} alpha={quantile}: {raw_cost + inventory_adjustment:.2f}", flush=True)
    selected = min(candidates, key=lambda x: (x["score_yuan"], x["window_days"], x["quantile"]))
    window, quantile = int(selected["window_days"]), float(selected["quantile"])
    main_frame, main_daily, main_terminal, main_gap = simulate(
        load, pv, price, hourly, dates, 31, 365, feb_initial, window, quantile, UPDATE_HOURS, True,
    )
    policy_results = []
    policy_daily = {}
    policy_terminals = {}
    comparison_gap = main_gap
    for hours in ((), (6,), (6, 12)):
        _, compared_daily, compared_terminal, compared_gap = simulate(
            load, pv, price, hourly, dates, 31, 365, feb_initial, window, quantile, hours, False,
        )
        label = "0" if not hours else "0+" + "+".join(map(str, hours))
        policy_daily[label] = compared_daily
        policy_terminals[label] = compared_terminal
        comparison_gap = max(comparison_gap, compared_gap)
    policy_daily["0+6+12+18"] = main_daily
    policy_terminals["0+6+12+18"] = main_terminal
    for label in ("0", "0+6", "0+6+12", "0+6+12+18"):
        compared = policy_daily[label]
        terminal = policy_terminals[label]
        raw_cost = float(compared.total_cost_yuan.sum())
        adjusted_cost = raw_cost + (SOC_TARGET_KWH - terminal) * float(price.min()) / ETA_CHARGE
        policy_results.append({
            "forecast_times": label, "total_cost_yuan": raw_cost,
            "terminal_soc_kwh": terminal, "inventory_adjusted_cost_yuan": adjusted_cost,
            "emergency_kwh": float(compared.emergency_kwh.sum()),
            "adjustment_cost_yuan": float(compared.adjustment_cost_yuan.sum()),
        })
    policy_comparison = pd.DataFrame(policy_results)
    policy_comparison["saving_vs_0_yuan"] = policy_comparison.total_cost_yuan.iloc[0] - policy_comparison.total_cost_yuan
    policy_comparison["marginal_saving_yuan"] = -policy_comparison.total_cost_yuan.diff().fillna(0)
    baseline_daily = policy_daily["0"]
    baseline_terminal = policy_terminals["0"]
    events = emergency_events(main_frame)
    checks = validate(main_frame)
    sums = [
        "plan_kwh", "adjusted_kwh", "increase_kwh", "decrease_kwh", "emergency_kwh",
        "charge_kwh", "discharge_kwh", "plan_cost_yuan", "adjustment_cost_yuan",
        "ordinary_settlement_cost_yuan", "emergency_cost_yuan", "total_cost_yuan",
    ]
    summary = {
        "method": "Historical load quantile + 0/6/12/18 PV forecast rolling MPC",
        "adjustment_settlement": "1.5*pi*increase - 0.5*pi*decrease, added to original plan cost",
        "start_date": DELIVERY_START, "end_date": "2025-12-31", "days": 334, "slots": 334 * SLOTS,
        "selected_window_days": window, "selected_load_quantile": quantile,
        "validation_start": "2025-01-15", "validation_end": "2025-01-31",
        "parameters_frozen_at": "2025-02-01 00:00", "update_hours": list(UPDATE_HOURS),
        "feb1_initial_soc_kwh": feb_initial, "dec31_terminal_soc_kwh": main_terminal,
        "min_soc_kwh": float(min(main_frame.soc_start_kwh.min(), main_frame.soc_end_kwh.min())),
        "max_soc_kwh": float(max(main_frame.soc_start_kwh.max(), main_frame.soc_end_kwh.max())),
        "emergency_events": len(events), "max_milp_gap": comparison_gap,
        "checks": checks, "python_version": platform.python_version(), "scipy_version": scipy.__version__,
        "master_sha256": hashlib.sha256(master_path.read_bytes()).hexdigest(),
        "forecast_sha256": hashlib.sha256(forecast_path.read_bytes()).hexdigest(),
        **{name: float(main_daily[name].sum()) for name in sums},
    }
    baseline_cost = float(baseline_daily.total_cost_yuan.sum())
    summary.update({
        "no_update_total_cost_yuan": baseline_cost,
        "no_update_dec31_terminal_soc_kwh": baseline_terminal,
        "saving_from_updates_yuan": baseline_cost - summary["total_cost_yuan"],
        "saving_from_updates_ratio": (baseline_cost - summary["total_cost_yuan"]) / baseline_cost,
        "inventory_adjusted_saving_yuan": (
            baseline_cost + (SOC_TARGET_KWH - baseline_terminal) * float(price.min()) / ETA_CHARGE
            - summary["total_cost_yuan"] - (SOC_TARGET_KWH - main_terminal) * float(price.min()) / ETA_CHARGE
        ),
        "update_policy_comparison": policy_comparison.to_dict(orient="records"),
    })
    stage_metrics = main_frame.groupby("forecast_issue_hour").apply(
        lambda g: pd.Series({
            "slots": len(g), "pv_forecast_mae_kw": float(np.abs(g.pv_actual_kw - g.pv_forecast_used_kw).mean()),
            "absolute_adjustment_kwh": float(np.abs(g.adjusted_kw - g.plan_kw).sum() * DT),
            "net_adjustment_kwh": float((g.adjusted_kw - g.plan_kw).sum() * DT),
        }), include_groups=False,
    ).reset_index()
    stage_metrics.to_csv(OUT / "problem3_stage_metrics.csv", index=False, encoding="utf-8-sig")
    policy_comparison.to_csv(OUT / "problem3_update_policy_comparison.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(candidates).to_csv(OUT / "problem3_january_validation.csv", index=False, encoding="utf-8-sig")
    pd.concat([jan_frame, main_frame], ignore_index=True).to_csv(OUT / "problem3_dispatch_all_2025.csv", index=False, encoding="utf-8-sig")
    main_daily.to_csv(OUT / "problem3_daily_summary.csv", index=False, encoding="utf-8-sig")
    baseline_daily.to_csv(OUT / "problem3_no_update_daily_summary.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT / "problem3_emergency_events.csv", index=False, encoding="utf-8-sig")
    (OUT / "problem3_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    payload = {
        "dates": main_daily.date.tolist(),
        "plan_rows": [
            g.plan_kwh.tolist() + [float(g.plan_kwh.sum()), float(g.plan_cost_yuan.sum())]
            for _, g in main_frame.groupby("date")
        ],
        "adjusted_rows": [
            g.adjusted_kwh.tolist() + [float(g.adjusted_kwh.sum()), float(g.ordinary_settlement_cost_yuan.sum())]
            for _, g in main_frame.groupby("date")
        ],
        "storage_rows": [
            {"date": date,
             "charge": g.charge_kwh.to_numpy().reshape(6, 24).sum(axis=1).tolist(),
             "discharge": g.discharge_kwh.to_numpy().reshape(6, 24).sum(axis=1).tolist(),
             "initial": float(g.soc_start_kwh.iloc[0]), "terminal": float(g.soc_end_kwh.iloc[-1])}
            for date, g in main_frame.groupby("date")
        ],
        "events": events.to_dict(orient="records"),
    }
    (OUT / "problem3_excel_data.json").write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
