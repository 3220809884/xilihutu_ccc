"""问题3：因果光伏预报 + 24小时MILP滚动优化 + 实际运行回放。

输入主表和公共整点预报表，不重新解析原附件时间；全部优化变量采用kWh。
运行 python3 问题3/代码/solve_q3_mpc.py。仅写入本题结果目录。
"""
from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题3/结果"
DT = 1 / 6
N = 144
P_MAX = 5000.0
E_MAX = P_MAX * DT
SOC_MIN, SOC_MAX, SOC_INITIAL = 1200.0, 10800.0, 6000.0
ETA = .9
ISSUE_HOURS = (0, 6, 12, 18)
TOL = 2e-5


def label(slot: int) -> str:
    return f"{slot // 6:02d}:{slot % 6 * 10:02d}"


def interval(slot: int) -> str:
    return f"{label(slot)}-{label(slot + 1)}"


def save_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def fingerprint(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


@dataclass
class Inputs:
    dates: list[str]
    load: np.ndarray
    pv: np.ndarray
    load_hat: np.ndarray
    price: np.ndarray
    pv_hat: np.ndarray  # [day, issue=0/6/12/18, lead_slot=1..144]
    net_hat: np.ndarray
    residual: np.ndarray  # diagnostic only; slice mature history before use
    load_audit: pd.DataFrame
    hashes: dict


def read_inputs() -> Inputs:
    master_path = ROOT / "公共代码/数据/master_10min.csv"
    hourly_path = ROOT / "公共代码/数据/forecast_attachment3_hourly.csv"
    q2_path = ROOT / "问题2/结果/problem2_dispatch_all_2025.csv"
    audit_path = ROOT / "问题2/结果/problem2_information_audit.csv"
    # Never consume master.pv_forecast_q3_latest_kw: a single 'latest' field
    # cannot reconstruct an older issue's information set.
    master = pd.read_csv(master_path, usecols=["date", "slot", "time_end", "load_actual_kw", "pv_actual_kw", "price_fixed_yuan_per_kwh"])
    master = master.sort_values(["date", "slot"]).reset_index(drop=True)
    assert len(master) == 365 * N and not master.duplicated(["date", "slot"]).any()
    assert np.array_equal(master.slot, np.tile(np.arange(N), 365))
    assert master.time_end.iloc[0] == "00:10" and master.time_end.iloc[-1] == "24:00"
    dates = master.date.drop_duplicates().tolist()
    load = master.load_actual_kw.to_numpy().reshape(-1, N)
    pv = master.pv_actual_kw.to_numpy().reshape(-1, N)
    price = master.price_fixed_yuan_per_kwh.to_numpy().reshape(-1, N)
    assert np.isfinite(np.r_[load.ravel(), pv.ravel(), price.ravel()]).all()
    assert min(load.min(), pv.min(), price.min()) >= 0
    q2 = pd.read_csv(q2_path, usecols=["date", "slot", "load_forecast_kw"])
    merged = master[["date", "slot"]].merge(q2, on=["date", "slot"], validate="one_to_one")
    assert len(merged) == len(master) and merged.load_forecast_kw.notna().all()
    load_hat = merged.load_forecast_kw.to_numpy().reshape(-1, N)
    audit = pd.read_csv(audit_path, keep_default_na=False).set_index("date").loc[dates].reset_index()
    for row in audit.itertuples():
        assert str(row.target_day_actual_used_for_forecast).lower() == "false"
        assert not row.history_through_date or row.history_through_date < row.date
        assert not row.model_train_end_date or row.model_train_end_date < row.date

    hourly = pd.read_csv(hourly_path, parse_dates=["issue_datetime", "target_datetime"])
    assert len(hourly) == 365 * 4 * 24
    assert not hourly.duplicated(["issue_datetime", "horizon_hours"]).any()
    assert (hourly.target_datetime == hourly.issue_datetime + pd.to_timedelta(hourly.horizon_hours, unit="h")).all()
    assert hourly.pv_forecast_kw.notna().all() and (hourly.pv_forecast_kw >= 0).all()
    groups = dict(tuple(hourly.groupby("issue_datetime")))
    pv_hat = np.empty((365, 4, N))
    net_hat = np.empty_like(pv_hat)
    residual = np.full_like(pv_hat, np.nan)
    flat_pv, flat_load = pv.ravel(), load.ravel()
    for day, date in enumerate(dates):
        for phase, hour in enumerate(ISSUE_HOURS):
            issue = pd.Timestamp(date) + pd.Timedelta(hours=hour)
            row = groups[issue].sort_values("horizon_hours")
            assert row.horizon_hours.tolist() == list(range(1, 25))
            first = day * N + hour * 6
            # At an issue time the interval ending at that time has completed.
            anchor = flat_pv[first - 1] if first else 0.0
            pv_hat[day, phase] = np.interp(np.arange(1, N + 1) / 6, np.arange(25), np.r_[anchor, row.pv_forecast_kw])
            slots = (hour * 6 + np.arange(N)) % N
            # Tomorrow is provisional: repeat today's midnight load profile.
            # Never read tomorrow's Q2 forecast (not available at this issue).
            net_hat[day, phase] = load_hat[day, slots] - pv_hat[day, phase]
            count = min(N, len(flat_load) - first)
            residual[day, phase, :count] = flat_load[first:first + count] - flat_pv[first:first + count] - net_hat[day, phase, :count]
    hashes = {str(p.relative_to(ROOT)): fingerprint(p) for p in [master_path, hourly_path, q2_path, audit_path]}
    return Inputs(dates, load, pv, load_hat, price, pv_hat, net_hat, residual, audit, hashes)


def risk_buffers(data: Inputs, alpha: float) -> tuple[np.ndarray, list[dict]]:
    """28-day same-issue/same-lead residual quantiles, using fully mature windows.

    At D 06:00, D-1 06:00's 24h forecast is fully observed through D 06:00.
    Therefore only issue-days j < D can enter the calibration.
    """
    result = np.zeros_like(data.net_hat)
    audit = []
    for day, date in enumerate(data.dates):
        start = max(1, day - 28)  # exclude untrained Jan 1
        for phase, hour in enumerate(ISSUE_HOURS):
            history = data.residual[start:day, phase]
            if alpha > 0 and len(history):
                assert np.isfinite(history).all()
                result[day, phase] = np.clip(np.quantile(history, alpha, axis=0), 0, P_MAX)
            issue = pd.Timestamp(date) + pd.Timedelta(hours=hour)
            audit.append({"issue_datetime": str(issue), "history_issue_days": len(history), "history_first_issue_date": data.dates[start] if len(history) else "", "history_last_issue_date": data.dates[day - 1] if len(history) else "", "latest_history_target": str(issue) if len(history) else "", "quantile": alpha})
    return result, audit


def optimize(net_kw, buffer_kw, price, soc0, baseline=None, terminal=6000.0):
    """24h energy-variable MILP. baseline is today's midnight plan remaining.

    q, charge, discharge, emergency_proxy, up, down (kWh), z (binary), S(kWh).
    For future-day provisional slots q is valued at ordinary fixed prices.
    """
    n = len(net_kw)
    q, c, d, e, up, down, z = [np.arange(i * n, (i + 1) * n) for i in range(7)]
    s = np.arange(7 * n, 8 * n + 1)
    size = 8 * n + 1
    lo, hi = np.zeros(size), np.full(size, np.inf)
    hi[c] = hi[d] = E_MAX
    hi[z] = 1
    lo[s], hi[s] = SOC_MIN, SOC_MAX
    lo[s[0]] = hi[s[0]] = soc0
    lo[s[-1]] = max(SOC_MIN, terminal)
    integer = np.zeros(size, dtype=int)
    integer[z] = 1
    obj = np.zeros(size)
    obj[q] = price
    obj[e] = 5 * price
    obj[c] = obj[d] = 1e-7  # numerical tie-break, not a billed battery charge
    m = 0 if baseline is None else len(baseline)
    hi[up] = hi[down] = 0
    if m:
        obj[q[:m]] = 0  # original plan payment is sunk, not paid again
        hi[up[:m]] = hi[down[:m]] = np.inf
        obj[up[:m]], obj[down[:m]] = 1.5 * price[:m], .5 * price[:m]
    rows, cols, vals, lower, upper = [], [], [], [], []

    def constraint(entries, lb=-np.inf, ub=np.inf):
        r = len(lower)
        for col, val in entries:
            rows.append(r); cols.append(col); vals.append(val)
        lower.append(lb); upper.append(ub)

    for t in range(n):
        constraint([(q[t], 1), (e[t], 1), (d[t], 1), (c[t], -1)], lb=(net_kw[t] + buffer_kw[t]) * DT)
        constraint([(s[t + 1], 1), (s[t], -1), (c[t], -ETA), (d[t], 1 / ETA)], 0, 0)
        constraint([(c[t], 1), (z[t], -E_MAX)], ub=0)
        constraint([(d[t], 1), (z[t], E_MAX)], ub=E_MAX)
        if t < m:
            constraint([(q[t], 1), (up[t], -1), (down[t], 1)], baseline[t], baseline[t])
    A = coo_matrix((vals, (rows, cols)), shape=(len(lower), size)).tocsc()
    result = milp(obj, integrality=integer, bounds=Bounds(lo, hi), constraints=LinearConstraint(A, lower, upper), options={"mip_rel_gap": 1e-7, "time_limit": 30})
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    x = result.x
    residual = A @ x
    violation = max(float(np.max(np.maximum(np.asarray(lower) - residual, 0))), float(np.max(np.maximum(residual - np.asarray(upper), 0))), float(np.max(np.maximum(lo - x, 0))), float(np.max(np.maximum(x - hi, 0))))
    assert violation < TOL
    assert np.max(np.abs(x[z] - np.rint(x[z]))) < TOL
    return {"q": np.maximum(x[q], 0), "charge": np.maximum(x[c], 0), "discharge": np.maximum(x[d], 0), "emergency_proxy": np.maximum(x[e], 0), "soc": x[s], "objective": float(result.fun), "gap": float(result.mip_gap), "violation": violation}


def execute(q, load_kw, pv_kw, soc0):
    """Causal local feedback: absorb surplus; meet a deficit from SOC then emergency.

    Actual interval-average load/PV is measured for balancing that interval.
    No future actuals enter dispatch. The planned battery trajectory is advisory;
    each 6h solve starts from measured, not predicted, SOC.
    """
    net = (load_kw - pv_kw) * DT
    surplus = q - net
    c = min(surplus, E_MAX, (SOC_MAX - soc0) / ETA) if surplus >= 0 else 0.0
    d = min(-surplus, E_MAX, (soc0 - SOC_MIN) * ETA) if surplus < 0 else 0.0
    emergency = max(0., -surplus - d)
    spill = max(0., surplus - c)
    soc1 = soc0 + ETA * c - d / ETA
    return max(0., c), max(0., d), emergency, spill, soc1


def run(data, buffers, hours=(0, 6, 12, 18), start_day=0, end_day=365, soc0=SOC_INITIAL, terminal=6000., keep_windows=False):
    records, windows, audits = [], [], []
    for day in range(start_day, end_day):
        initial_plan = np.zeros(N)
        current_q = np.zeros(N)
        for hour in hours:
            phase = ISSUE_HOURS.index(hour)
            start = hour * 6
            next_hour = next((h for h in hours if h > hour), 24)
            end = next_hour * 6
            n = min(N, (len(data.dates) - day) * N - start)
            slots = (start + np.arange(n)) % N
            price = data.price[day, slots]
            baseline = None if hour == 0 else initial_plan[start:]
            plan = optimize(data.net_hat[day, phase, :n], buffers[day, phase, :n], price, soc0, baseline, terminal)
            if hour == 0:
                initial_plan = plan["q"][:N].copy()
            current_q[start:] = plan["q"][:N - start]
            issue = pd.Timestamp(data.dates[day]) + pd.Timedelta(hours=hour)
            audits.append({"date": data.dates[day], "issue_hour": hour, "issue_datetime": str(issue), "first_target_end": str(issue + pd.Timedelta(minutes=10)), "last_target_end": str(issue + pd.Timedelta(minutes=10 * n)), "committed_until": str(pd.Timestamp(data.dates[day]) + pd.Timedelta(hours=next_hour)), "soc_measured_kwh": soc0, "planned_terminal_kwh": float(plan["soc"][-1]), "horizon_slots": n, "future_day_provisional_slots": max(0, n - (N - start)), "milp_gap": plan["gap"], "constraint_violation": plan["violation"], "load_history_through_date": data.load_audit.iloc[day].history_through_date, "load_model_train_end_date": data.load_audit.iloc[day].model_train_end_date})
            if keep_windows:
                for j in range(n):
                    target_end = issue + pd.Timedelta(minutes=10 * (j + 1))
                    windows.append({"issue_datetime": str(issue), "target_end": str(target_end), "lead_slot": j + 1, "committed": j < end - start, "provisional_next_day": j >= N - start, "load_forecast_kw": data.load_hat[day, slots[j]], "pv_forecast_kw": data.pv_hat[day, phase, j], "buffer_kw": buffers[day, phase, j], "purchase_kwh": plan["q"][j], "charge_kwh": plan["charge"][j], "discharge_kwh": plan["discharge"][j], "emergency_proxy_kwh": plan["emergency_proxy"][j], "soc_start_kwh": plan["soc"][j], "soc_end_kwh": plan["soc"][j + 1]})
            for t in range(start, end):
                j = t - start
                q = float(current_q[t])
                c, d, emergency, spill, soc1 = execute(q, data.load[day, t], data.pv[day, t], soc0)
                p = data.price[day, t]
                up, down = max(0., q - initial_plan[t]), max(0., initial_plan[t] - q)
                plan_cost, adj_cost, emergency_cost = p * initial_plan[t], p * (1.5 * up + .5 * down), 5 * p * emergency
                records.append({"date": data.dates[day], "slot": t, "time_interval": interval(t), "issue_hour": hour, "issue_datetime": str(issue), "target_end": str(pd.Timestamp(data.dates[day]) + pd.Timedelta(minutes=10 * (t + 1))), "price_yuan_per_kwh": p, "load_actual_kw": data.load[day, t], "pv_actual_kw": data.pv[day, t], "load_forecast_kw": data.load_hat[day, t], "pv_forecast_kw": data.pv_hat[day, phase, j], "buffer_kw": buffers[day, phase, j], "plan_kwh": initial_plan[t], "adjusted_purchase_kwh": q, "adjustment_up_kwh": up, "adjustment_down_kwh": down, "charge_kwh": c, "discharge_kwh": d, "emergency_kwh": emergency, "spill_kwh": spill, "soc_start_kwh": soc0, "soc_end_kwh": soc1, "plan_cost_yuan": plan_cost, "adjustment_cost_yuan": adj_cost, "emergency_cost_yuan": emergency_cost, "total_cost_yuan": plan_cost + adj_cost + emergency_cost})
                soc0 = soc1
        if day % 31 == 0:
            print(f"hours={hours}, date={data.dates[day]}, SOC={soc0:.2f}", flush=True)
    return pd.DataFrame(records), pd.DataFrame(windows), pd.DataFrame(audits)


SUM_COLS = ["plan_kwh", "adjusted_purchase_kwh", "adjustment_up_kwh", "adjustment_down_kwh", "charge_kwh", "discharge_kwh", "emergency_kwh", "spill_kwh", "plan_cost_yuan", "adjustment_cost_yuan", "emergency_cost_yuan", "total_cost_yuan"]


def summarize(frame):
    return {c: float(frame[c].sum()) for c in SUM_COLS}


def events(frame):
    result = []
    for date, g in frame.groupby("date", sort=True):
        energy = g.emergency_kwh.to_numpy()
        edges = np.diff(np.r_[False, energy > 1e-8, False].astype(int))
        for a, b in zip(np.where(edges == 1)[0], np.where(edges == -1)[0]):
            result.append({"date": date, "interval": f"{label(a)}-{label(b)}", "emergency_kwh": float(energy[a:b].sum())})
    return pd.DataFrame(result, columns=["date", "interval", "emergency_kwh"])


def validate(frame):
    balance = frame.adjusted_purchase_kwh + frame.emergency_kwh + frame.pv_actual_kw * DT + frame.discharge_kwh - frame.load_actual_kw * DT - frame.charge_kwh - frame.spill_kwh
    soc = frame.soc_end_kwh - frame.soc_start_kwh - ETA * frame.charge_kwh + frame.discharge_kwh / ETA
    continuity = frame.soc_start_kwh.to_numpy()[1:] - frame.soc_end_kwh.to_numpy()[:-1]
    diagnostics = {"energy_balance_max_abs_kwh": float(balance.abs().max()), "soc_dynamic_max_abs_kwh": float(soc.abs().max()), "soc_continuity_max_abs_kwh": float(np.max(np.abs(continuity))), "soc_min_kwh": float(min(frame.soc_start_kwh.min(), frame.soc_end_kwh.min())), "soc_max_kwh": float(max(frame.soc_start_kwh.max(), frame.soc_end_kwh.max())), "max_charge_kw": float(frame.charge_kwh.max() / DT), "max_discharge_kw": float(frame.discharge_kwh.max() / DT), "simultaneous_charge_discharge_count": int(((frame.charge_kwh > TOL) & (frame.discharge_kwh > TOL)).sum())}
    assert max(diagnostics[k] for k in ("energy_balance_max_abs_kwh", "soc_dynamic_max_abs_kwh", "soc_continuity_max_abs_kwh")) < TOL
    assert diagnostics["soc_min_kwh"] >= SOC_MIN - TOL and diagnostics["soc_max_kwh"] <= SOC_MAX + TOL
    assert diagnostics["max_charge_kw"] <= P_MAX + TOL and diagnostics["max_discharge_kw"] <= P_MAX + TOL
    assert diagnostics["simultaneous_charge_discharge_count"] == 0
    assert (pd.to_datetime(frame.issue_datetime) < pd.to_datetime(frame.target_end)).all()
    assert frame.adjustment_down_kwh.max() < TOL  # dominated under full-plan payment
    return diagnostics


def forecast_analysis(data):
    rows = []
    flat_pv = data.pv.ravel()
    for day in range(31, 365):
        for phase, hour in enumerate(ISSUE_HOURS):
            first = day * N + hour * 6
            n = min(N, len(flat_pv) - first)
            for j in range(n):
                rows.append((data.dates[day], hour, (j + 1) / 6, flat_pv[first + j], data.pv_hat[day, phase, j]))
    f = pd.DataFrame(rows, columns=["issue_date", "issue_hour", "lead_hours", "actual_kw", "forecast_kw"])
    f["error_kw"] = f.actual_kw - f.forecast_kw
    f["lead_band"] = pd.cut(f.lead_hours, bins=[0, 6, 12, 18, 24], labels=["0-6h", "6-12h", "12-18h", "18-24h"])
    metrics = []
    for (hour, band), g in f.groupby(["issue_hour", "lead_band"], observed=True):
        metrics.append({"issue_hour": int(hour), "lead_band": str(band), "n": len(g), "MAE_kw": float(g.error_kw.abs().mean()), "RMSE_kw": float(np.sqrt(np.mean(g.error_kw ** 2))), "bias_actual_minus_forecast_kw": float(g.error_kw.mean())})
    # Common target-day slots: compare each update with midnight on identical targets.
    common = []
    for phase, hour in enumerate(ISSUE_HOURS[1:], start=1):
        count = N - hour * 6
        truth = data.pv[31:, hour * 6:]
        old = data.pv_hat[31:, 0, hour * 6:]
        new = data.pv_hat[31:, phase, :count]
        for name, forecast in [("midnight", old), ("updated", new)]:
            err = truth - forecast
            common.append({"update_hour": hour, "forecast": name, "n_common_targets": err.size, "MAE_kw": float(np.abs(err).mean()), "RMSE_kw": float(np.sqrt(np.mean(err ** 2)))})
    return pd.DataFrame(metrics), pd.DataFrame(common)


def excel_payload(frame, event_frame):
    plans, adjusted, storage = [], [], []
    for date, g in frame.groupby("date", sort=True):
        plans.append(g.plan_kwh.tolist() + [float(g.plan_kwh.sum()), float(g.plan_cost_yuan.sum())])
        # This sheet stores the final purchase quantities, NOT signed deltas.
        # Its cost column is the adjustment-related extra cost only.
        adjusted.append(g.adjusted_purchase_kwh.tolist() + [float(g.adjusted_purchase_kwh.sum()), float(g.adjustment_cost_yuan.sum())])
        storage.append({"date": date, "charge": g.charge_kwh.to_numpy().reshape(6, 24).sum(axis=1).tolist(), "discharge": g.discharge_kwh.to_numpy().reshape(6, 24).sum(axis=1).tolist(), "initial": float(g.soc_start_kwh.iloc[0]), "terminal": float(g.soc_end_kwh.iloc[-1])})
    return {"dates": frame.date.drop_duplicates().tolist(), "prices": frame.price_yuan_per_kwh.iloc[:N].tolist(), "intervals": [interval(t) for t in range(N)], "plan_rows": plans, "adjusted_rows": adjusted, "storage_rows": storage, "events": event_frame.to_dict("records")}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=.8)
    parser.add_argument("--terminal", type=float, default=6000.)
    parser.add_argument("--days", type=int, default=365, help="smoke test can use 3; never delivers official workbook")
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--skip-comparison", action="store_true")
    args = parser.parse_args()
    assert 0 <= args.alpha < 1 and SOC_MIN <= args.terminal <= SOC_MAX
    args.out.mkdir(parents=True, exist_ok=True)
    data = read_inputs()
    buffers, risk_audit = risk_buffers(data, args.alpha)
    metrics, common = forecast_analysis(data)
    metrics.to_csv(args.out / "problem3_forecast_metrics.csv", index=False)
    common.to_csv(args.out / "problem3_forecast_common_targets.csv", index=False)
    frame, windows, audit = run(data, buffers, end_day=args.days, terminal=args.terminal, keep_windows=True)
    checks = validate(frame)
    frame.to_csv(args.out / "problem3_dispatch_all_2025.csv", index=False, float_format="%.10f")
    windows.to_csv(args.out / "problem3_rolling_windows.csv.gz", index=False, float_format="%.10f", compression={"method": "gzip", "mtime": 0})
    audit.to_csv(args.out / "problem3_information_audit.csv", index=False)
    pd.DataFrame(risk_audit).to_csv(args.out / "problem3_risk_calibration_audit.csv", index=False)
    if args.days < 365:
        print(json.dumps(checks, ensure_ascii=False), flush=True)
        return
    delivered = frame[frame.date >= "2025-02-01"].copy()
    daily = delivered.groupby("date", as_index=False)[SUM_COLS].sum()
    daily["initial_soc_kwh"] = delivered.groupby("date").soc_start_kwh.first().to_numpy()
    daily["terminal_soc_kwh"] = delivered.groupby("date").soc_end_kwh.last().to_numpy()
    daily.to_csv(args.out / "problem3_daily_summary.csv", index=False)
    ev = events(delivered)
    ev.to_csv(args.out / "problem3_emergency_events.csv", index=False)
    save_json(args.out / "problem3_excel_data.json", excel_payload(delivered, ev))
    common_soc = float(delivered.soc_start_kwh.iloc[0])
    comparisons = [{"strategy": "0+6+12+18", "alpha": args.alpha, "terminal_target_kwh": args.terminal, "initial_soc_kwh": common_soc, "final_soc_kwh": float(delivered.soc_end_kwh.iloc[-1]), **summarize(delivered)}]
    baseline_daily = None
    if not args.skip_comparison:
        # All schedules start from the SAME Feb 1 SOC and use the SAME forecasts.
        for hours in [(0,), (0, 6), (0, 12), (0, 18), (0, 6, 12), (0, 6, 18), (0, 12, 18)]:
            alt, _, _ = run(data, buffers, hours=hours, start_day=31, soc0=common_soc, terminal=args.terminal)
            validate(alt)
            comparisons.append({"strategy": "+".join(map(str, hours)), "alpha": args.alpha, "terminal_target_kwh": args.terminal, "initial_soc_kwh": common_soc, "final_soc_kwh": float(alt.soc_end_kwh.iloc[-1]), **summarize(alt)})
            if hours == (0,):
                baseline_daily = alt.groupby("date", as_index=False)[SUM_COLS].sum()
                baseline_daily.to_csv(args.out / "problem3_midnight_baseline_daily.csv", index=False)
        # Predeclared parameter sensitivity, not in-sample parameter selection.
        for alpha, terminal in [(0., args.terminal), (.7, args.terminal), (.9, args.terminal), (args.alpha, 3600.), (args.alpha, 8400.)]:
            buf, _ = risk_buffers(data, alpha)
            alt, _, _ = run(data, buf, start_day=31, soc0=common_soc, terminal=terminal)
            validate(alt)
            comparisons.append({"strategy": "0+6+12+18", "alpha": alpha, "terminal_target_kwh": terminal, "initial_soc_kwh": common_soc, "final_soc_kwh": float(alt.soc_end_kwh.iloc[-1]), **summarize(alt)})
    comp = pd.DataFrame(comparisons)
    comp.to_csv(args.out / "problem3_strategy_comparison.csv", index=False)
    summary = {"period": "2025-02-01 to 2025-12-31", "n_days": 334, "alpha": args.alpha, "terminal_target_kwh": args.terminal, "initial_soc_kwh": common_soc, "final_soc_kwh": float(delivered.soc_end_kwh.iloc[-1]), "scipy_version": scipy.__version__, "input_sha256": data.hashes, "checks": checks, "max_milp_gap": float(audit.milp_gap.max()), "rolling_solves": len(audit), **summarize(delivered)}
    if baseline_daily is not None:
        base_cost = float(baseline_daily.total_cost_yuan.sum())
        summary["same_forecast_midnight_baseline_cost_yuan"] = base_cost
        summary["saving_vs_midnight_yuan"] = base_cost - summary["total_cost_yuan"]
        summary["saving_vs_midnight_fraction"] = 1 - summary["total_cost_yuan"] / base_cost
    save_json(args.out / "problem3_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
