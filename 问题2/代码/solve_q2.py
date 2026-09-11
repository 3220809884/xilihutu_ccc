"""Causal day-ahead planning and real-time execution for problem 2.

Run from any directory: python 问题2/代码/solve_q2.py
Only canonical master-table columns from attachments 1 and 2 are read.
Power: kW; stored energy: kWh; duration: 1/6 h.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import platform

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题2" / "结果"
DT, ETA, P_MAX = 1 / 6, 0.9, 5000.0
S_MIN, S_MAX, S_INITIAL, S_TARGET = 1200.0, 10800.0, 6000.0, 6000.0
TOL = 1e-6
DATES = ["2025-03-20", "2025-06-21", "2025-09-23", "2025-12-21"]


def forecast(history: np.ndarray, window: int, quantile: float) -> np.ndarray:
    """Caller passes completed days ONLY; no access to a target-day record."""
    if len(history) == 0:
        raise ValueError("Forecast needs at least one completed historical day")
    return np.quantile(history[-window:], quantile, axis=0, method="linear")


def plan_day(net_forecast: np.ndarray, price: np.ndarray, initial: float,
             terminal: float = S_TARGET) -> dict:
    """Solve the deterministic risk-adjusted MILP; this is not a stochastic optimum."""
    n = len(price)
    assert net_forecast.shape == (n,) and np.all(price > 0)
    assert S_MIN - TOL <= initial <= S_MAX + TOL
    # [q(n), c(n), d(n), S(n+1), z(n)]
    q, c, d, s, z = 0, n, 2*n, 3*n, 4*n+1
    size = 5*n+1
    obj = np.zeros(size)
    obj[q:q+n] = price * DT
    lb, ub = np.zeros(size), np.full(size, np.inf)
    ub[c:c+n] = ub[d:d+n] = P_MAX
    lb[s:s+n+1], ub[s:s+n+1] = S_MIN, S_MAX
    lb[s] = ub[s] = float(np.clip(initial, S_MIN, S_MAX))
    lb[s+n] = ub[s+n] = terminal
    ub[z:z+n] = 1
    integer = np.zeros(size, dtype=int)
    integer[z:z+n] = 1
    a = lil_matrix((4*n, size))
    lo, hi = np.full(4*n, -np.inf), np.full(4*n, np.inf)
    for t in range(n):
        a[t, q+t], a[t, c+t], a[t, d+t] = 1, -1, 1
        lo[t] = net_forecast[t]
        a[n+t, s+t+1], a[n+t, s+t] = 1, -1
        a[n+t, c+t], a[n+t, d+t] = -ETA*DT, DT/ETA
        lo[n+t] = hi[n+t] = 0
        a[2*n+t, c+t], a[2*n+t, z+t] = 1, -P_MAX
        hi[2*n+t] = 0
        a[3*n+t, d+t], a[3*n+t, z+t] = 1, P_MAX
        hi[3*n+t] = P_MAX
    result = milp(obj, integrality=integer, bounds=Bounds(lb, ub),
                  constraints=LinearConstraint(a.tocsc(), lo, hi),
                  options={"mip_rel_gap": 1e-9, "time_limit": 30.0})
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    x = result.x
    charge = np.maximum(x[c:c+n], 0)
    discharge = np.maximum(x[d:d+n], 0)
    purchase = np.maximum(x[q:q+n], 0)
    states = x[s:s+n+1]
    assert np.max(np.abs(np.diff(states)-ETA*DT*charge+DT/ETA*discharge)) < TOL
    assert np.min(purchase+discharge-charge-net_forecast) >= -TOL
    assert not np.any((charge > TOL) & (discharge > TOL))
    assert max(charge.max(), discharge.max()) <= P_MAX + TOL
    assert abs(float(price@purchase*DT)-result.fun) < TOL
    return dict(purchase=purchase, charge=charge, discharge=discharge, soc=states,
                cost=float(result.fun), gap=float(result.mip_gap))


def execute_day(purchase: np.ndarray, net_actual: np.ndarray, initial: float,
                idle_storage: bool = False) -> dict:
    """Causal feedback: use only the current slot's realized imbalance and SOC.

    Charge paid/PV surplus; discharge to meet a deficit, then buy emergency power.
    Emergency purchases never charge the battery. No future actuals are consulted.
    """
    n = len(purchase)
    c, d, e, w = (np.zeros(n) for _ in range(4))
    soc = np.zeros(n+1)
    soc[0] = initial
    for t in range(n):
        surplus = purchase[t] - net_actual[t]
        if surplus >= 0:
            c[t] = 0 if idle_storage else min(surplus, P_MAX, max(0, (S_MAX-soc[t])/(ETA*DT)))
            w[t] = surplus-c[t]
        else:
            d[t] = 0 if idle_storage else min(-surplus, P_MAX, max(0, (soc[t]-S_MIN)*ETA/DT))
            e[t] = -surplus-d[t]
        soc[t+1] = soc[t]+ETA*c[t]*DT-d[t]*DT/ETA
    return dict(charge=c, discharge=d, emergency=e, surplus=w, soc=soc)


def time_label(slot: int) -> str:
    minute = slot * 10
    return f"{minute//60:02d}:{minute%60:02d}"


def emergency_events(frame: pd.DataFrame) -> pd.DataFrame:
    events = []
    for date, day in frame.groupby("date", sort=True):
        values = day.emergency_kwh.to_numpy()
        t = 0
        while t < 144:
            if values[t] <= 1e-8:
                t += 1
                continue
            start = t
            while t < 144 and values[t] > 1e-8:
                t += 1
            events.append(dict(date=date, start_slot=start, end_slot_exclusive=t,
                               interval=f"{time_label(start)}-{time_label(t)}",
                               emergency_kwh=float(values[start:t].sum()),
                               emergency_cost_yuan=float(day.emergency_cost_yuan.iloc[start:t].sum())))
    return pd.DataFrame(events, columns=["date", "start_slot", "end_slot_exclusive", "interval", "emergency_kwh", "emergency_cost_yuan"])


def validate(frame: pd.DataFrame) -> dict:
    assert len(frame) == 365*144 and not frame.duplicated(["date", "slot"]).any()
    assert np.isfinite(frame.select_dtypes(include="number").to_numpy()).all()
    assert (frame.groupby("date").size() == 144).all()
    for col in ["plan_kw", "charge_kw", "discharge_kw", "emergency_kw", "surplus_kw"]:
        assert frame[col].min() >= -TOL
    assert frame.soc_start_kwh.min() >= S_MIN-TOL and frame.soc_end_kwh.min() >= S_MIN-TOL
    assert frame.soc_start_kwh.max() <= S_MAX+TOL and frame.soc_end_kwh.max() <= S_MAX+TOL
    assert max(frame.charge_kw.max(), frame.discharge_kw.max()) <= P_MAX+TOL
    assert not ((frame.charge_kw>TOL) & (frame.discharge_kw>TOL)).any()
    residual = frame.plan_kw+frame.emergency_kw+frame.pv_actual_kw+frame.discharge_kw-frame.load_actual_kw-frame.charge_kw-frame.surplus_kw
    soc_residual = frame.soc_end_kwh-frame.soc_start_kwh-ETA*DT*frame.charge_kw+DT/ETA*frame.discharge_kw
    continuity = frame.soc_start_kwh.to_numpy()[1:]-frame.soc_end_kwh.to_numpy()[:-1]
    for values in [residual, soc_residual, continuity]:
        assert np.max(np.abs(values)) < TOL
    assert frame.soc_start_kwh.iloc[0] == S_INITIAL
    for stem in ["plan", "charge", "discharge", "emergency"]:
        assert np.max(np.abs(frame[stem+"_kwh"]-DT*frame[stem+"_kw"])) < TOL
    assert np.max(np.abs(frame.plan_cost_yuan-frame.price_yuan_per_kwh*frame.plan_kwh)) < TOL
    assert np.max(np.abs(frame.emergency_cost_yuan-5*frame.price_yuan_per_kwh*frame.emergency_kwh)) < TOL
    return dict(max_balance_residual_kw=float(np.max(np.abs(residual))),
                max_soc_transition_residual_kwh=float(np.max(np.abs(soc_residual))),
                max_continuity_residual_kwh=float(np.max(np.abs(continuity))),
                simultaneous_charge_discharge_slots=0, all_assertions_passed=True)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "公共代码/数据/master_10min.csv"
    columns = ["date", "slot", "time_end", "price_fixed_yuan_per_kwh", "load_actual_kw", "pv_actual_kw"]
    data = pd.read_csv(source, usecols=columns).sort_values(["date", "slot"]).reset_index(drop=True)
    dates = data.date.drop_duplicates().tolist()
    assert dates == pd.date_range("2025-01-01", "2025-12-31").strftime("%Y-%m-%d").tolist()
    assert len(data) == 365*144 and not data.duplicated(["date", "slot"]).any()
    assert np.array_equal(data.slot.to_numpy(), np.tile(np.arange(144),365))
    assert np.isfinite(data.select_dtypes(include="number").to_numpy()).all()
    load = data.load_actual_kw.to_numpy().reshape(365,144)
    pv = data.pv_actual_kw.to_numpy().reshape(365,144)
    prices = data.price_fixed_yuan_per_kwh.to_numpy().reshape(365,144)
    assert (load >= 0).all() and (pv >= 0).all() and np.all(prices == prices[0])
    price, net = prices[0], load-pv

    # Operational January warm-up is fixed ex ante, never replayed with selected parameters.
    january, initial = [], S_INITIAL
    for day in range(31):
        if day == 0:
            pred = np.zeros(144)
            plan = dict(purchase=np.zeros(144), charge=np.zeros(144), discharge=np.zeros(144),
                        soc=np.full(145, initial), cost=0.0, gap=0.0)
        else:
            pred = forecast(net[:day], 14, 0.8)
            plan = plan_day(pred, price, initial)
        executed = execute_day(plan["purchase"], net[day], initial, idle_storage=day==0)
        january.append((pred, plan, executed))
        initial = executed["soc"][-1]
    feb_initial = initial

    # Validation is completed by Jan 31. Candidates see only prior-day observations.
    validation = []
    validation_start_soc = january[14][2]["soc"][0]
    for window in [7,14,28]:
        for quantile in [0.5,0.65,0.8,0.9,0.95]:
            soc, cost, em = validation_start_soc, 0.0, 0.0
            for day in range(14,31):
                pred = forecast(net[:day], window, quantile)
                plan = plan_day(pred, price, soc)
                ex = execute_day(plan["purchase"], net[day], soc)
                cost += plan["cost"]+5*float(price@ex["emergency"]*DT)
                em += float(ex["emergency"].sum()*DT)
                soc = ex["soc"][-1]
            # Equivalent end inventory at low-price replacement value for fair comparison.
            adjustment = (S_TARGET-soc)*float(price.min())/ETA
            validation.append(dict(window_days=window, quantile=quantile,
                                   validation_cost_yuan=cost, terminal_soc_kwh=soc,
                                   inventory_adjustment_yuan=adjustment,
                                   score_yuan=cost+adjustment, emergency_kwh=em))
            print(f"validation window={window} quantile={quantile}: {cost+adjustment:.2f}", flush=True)
    selected = min(validation, key=lambda v: (v["score_yuan"],v["window_days"],v["quantile"]))
    window, quantile = selected["window_days"], selected["quantile"]

    records, audits = [], []
    initial = S_INITIAL
    max_gap = 0.0
    for day, date in enumerate(dates):
        if day < 31:
            pred, plan, ex = january[day]
        else:
            pred = forecast(net[:day], window, quantile)
            plan = plan_day(pred, price, initial)
            ex = execute_day(plan["purchase"], net[day], initial)
        assert abs(ex["soc"][0]-initial) < TOL
        max_gap = max(max_gap, plan["gap"])
        history = net[max(0, day-window):day]
        median = np.median(history, axis=0) if day else np.zeros(144)
        no_storage_plan = np.maximum(pred, 0)
        no_storage_emergency = np.maximum(net[day]-no_storage_plan, 0)
        block = pd.DataFrame(dict(date=date, slot=np.arange(144),
            time_end=[time_label(t+1) for t in range(144)], price_yuan_per_kwh=price,
            load_actual_kw=load[day], pv_actual_kw=pv[day], net_forecast_kw=pred,
            net_median_forecast_kw=median, plan_kw=plan["purchase"],
            planned_charge_kw=plan["charge"], planned_discharge_kw=plan["discharge"],
            planned_soc_start_kwh=plan["soc"][:-1], planned_soc_end_kwh=plan["soc"][1:],
            charge_kw=ex["charge"], discharge_kw=ex["discharge"],
            emergency_kw=ex["emergency"], surplus_kw=ex["surplus"],
            soc_start_kwh=ex["soc"][:-1], soc_end_kwh=ex["soc"][1:],
            baseline_without_storage_cost_yuan=(no_storage_plan+5*no_storage_emergency)*price*DT))
        for stem in ["plan","charge","discharge","emergency"]:
            block[stem+"_kwh"] = block[stem+"_kw"]*DT
        block["plan_cost_yuan"] = block.plan_kwh*price
        block["emergency_cost_yuan"] = block.emergency_kwh*price*5
        block["total_cost_yuan"] = block.plan_cost_yuan+block.emergency_cost_yuan
        records.append(block)
        audits.append(dict(date=date, issued_at=date+" 00:00", history_through_date=dates[day-1] if day else None,
                           history_days=min(day,14 if day<31 else window),
                           quantile=0.8 if day<31 else quantile,
                           initial_soc_kwh=float(initial), planned_terminal_soc_kwh=float(plan["soc"][-1]),
                           milp_gap=plan["gap"]))
        initial = ex["soc"][-1]
        if day % 30 == 0:
            print(f"executed through {date}, SOC={initial:.3f}", flush=True)
    all_rows = pd.concat(records, ignore_index=True)
    checks = validate(all_rows)
    delivered = all_rows[all_rows.date >= "2025-02-01"].copy()
    assert len(delivered) == 334*144
    events = emergency_events(delivered)
    assert abs(events.emergency_kwh.sum()-delivered.emergency_kwh.sum()) < TOL
    sums = ["plan_kwh","charge_kwh","discharge_kwh","emergency_kwh","plan_cost_yuan",
            "emergency_cost_yuan","total_cost_yuan","baseline_without_storage_cost_yuan"]
    daily = delivered.groupby("date")[sums].sum().reset_index()
    daily["soc_start_kwh"] = delivered.groupby("date").soc_start_kwh.first().to_numpy()
    daily["soc_end_kwh"] = delivered.groupby("date").soc_end_kwh.last().to_numpy()
    err = delivered.load_actual_kw-delivered.pv_actual_kw-delivered.net_median_forecast_kw
    summary = dict(method="Rolling historical net-load quantile + day-ahead MILP + causal imbalance feedback",
                   start_date="2025-02-01", end_date="2025-12-31", days=334, slots=334*144,
                   selected_window_days=window, selected_quantile=quantile,
                   validation_start="2025-01-15", validation_end="2025-01-31",
                   parameters_frozen_at="2025-02-01 00:00", feb1_initial_soc_kwh=float(feb_initial),
                   dec31_terminal_soc_kwh=float(initial),
                   min_soc_kwh=float(delivered.soc_end_kwh.min()), max_soc_kwh=float(delivered.soc_end_kwh.max()),
                   median_net_forecast_mae_kw=float(np.abs(err).mean()),
                   median_net_forecast_rmse_kw=float(np.sqrt((err**2).mean())),
                   risk_forecast_empirical_coverage=float((delivered.load_actual_kw-delivered.pv_actual_kw <= delivered.net_forecast_kw).mean()),
                   max_daily_milp_gap=max_gap, emergency_events=len(events), checks=checks,
                   python_version=platform.python_version(), scipy_version=scipy.__version__,
                   master_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
                   **{name:float(delivered[name].sum()) for name in sums})
    summary["cost_saving_vs_same_forecast_without_storage_yuan"] = summary["baseline_without_storage_cost_yuan"]-summary["total_cost_yuan"]
    all_rows.to_csv(OUT/"problem2_dispatch_all_2025.csv",index=False,encoding="utf-8-sig")
    daily.to_csv(OUT/"problem2_daily_summary.csv",index=False,encoding="utf-8-sig")
    events.to_csv(OUT/"problem2_emergency_events.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(validation).to_csv(OUT/"problem2_january_validation.csv",index=False,encoding="utf-8-sig")
    pd.DataFrame(audits).to_csv(OUT/"problem2_information_audit.csv",index=False,encoding="utf-8-sig")
    (OUT/"problem2_summary.json").write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding="utf-8")
    # Compact, typed interchange consumed by the workbook author, not a second calculation.
    payload = dict(dates=daily.date.tolist(),
        plan_rows=[g.plan_kwh.tolist()+[float(g.plan_kwh.sum()),float(g.plan_cost_yuan.sum())]
                   for _,g in delivered.groupby("date")],
        storage_rows=[dict(date=date,charge=g.charge_kwh.to_numpy().reshape(6,24).sum(axis=1).tolist(),
                          discharge=g.discharge_kwh.to_numpy().reshape(6,24).sum(axis=1).tolist(),
                          initial=float(g.soc_start_kwh.iloc[0]),terminal=float(g.soc_end_kwh.iloc[-1]))
                      for date,g in delivered.groupby("date")],
        events=events.to_dict(orient="records"))
    (OUT/"problem2_excel_data.json").write_text(json.dumps(payload,ensure_ascii=False),encoding="utf-8")
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


if __name__ == "__main__":
    main()
