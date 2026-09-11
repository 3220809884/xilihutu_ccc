"""Solve problem 1 with a fast deterministic storage dynamic program.

All dispatch variables are represented internally as power (kW).  The SOC is
energy (kWh), so its transition multiplies power by the 10-minute duration.
The dynamic program enforces charge/discharge mutual exclusion by allowing
each transition to be exactly one of charging, discharging, or idling.
"""

from __future__ import annotations

import json
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd


DT_HOURS = 1 / 6
ETA_CHARGE = 0.90
ETA_DISCHARGE = 0.90
SOC_MIN_KWH = 1200.0
SOC_MAX_KWH = 10800.0
SOC_INITIAL_KWH = 6000.0
POWER_MAX_KW = 5000.0
SOC_STEP_KWH = 1.0
SLOTS_PER_DAY = 144
TOL = 1e-7


def _window_min(
    values: np.ndarray, lows: np.ndarray, highs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return minimum values and indices over monotone inclusive windows."""
    n = len(values)
    out = np.full(n, np.inf)
    arg = np.full(n, -1, dtype=np.int32)
    candidates: deque[int] = deque()
    added_through = -1

    for i in range(n):
        lo = max(0, int(lows[i]))
        hi = min(n - 1, int(highs[i]))
        if hi < lo:
            continue
        while added_through < hi:
            added_through += 1
            value = values[added_through]
            while candidates and values[candidates[-1]] > value:
                candidates.pop()
            candidates.append(added_through)
        while candidates and candidates[0] < lo:
            candidates.popleft()
        if candidates and candidates[0] <= hi:
            arg[i] = candidates[0]
            out[i] = values[candidates[0]]
    return out, arg


def solve_dispatch(
    price: np.ndarray, load_kw: np.ndarray, pv_kw: np.ndarray
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    """Solve the 144-slot deterministic dispatch on a 1 kWh SOC lattice."""
    states = np.arange(SOC_MIN_KWH, SOC_MAX_KWH + SOC_STEP_KWH, SOC_STEP_KWH)
    n_states = len(states)
    initial_index = int(round((SOC_INITIAL_KWH - SOC_MIN_KWH) / SOC_STEP_KWH))

    dp = np.full(n_states, np.inf)
    dp[initial_index] = 0.0
    predecessors = np.full((SLOTS_PER_DAY, n_states), -1, dtype=np.int32)
    state_indices = np.arange(n_states)

    max_soc_increase = ETA_CHARGE * POWER_MAX_KW * DT_HOURS
    max_soc_decrease = POWER_MAX_KW * DT_HOURS / ETA_DISCHARGE
    max_charge_steps = int(np.floor(max_soc_increase / SOC_STEP_KWH + TOL))
    max_discharge_steps = int(np.floor(max_soc_decrease / SOC_STEP_KWH + TOL))

    for t in range(SLOTS_PER_DAY):
        net_load_kw = float(load_kw[t] - pv_kw[t])
        unit_price = float(price[t])

        # Charging transition: delta_soc >= 0.
        free_charge_soc = max(0.0, -net_load_kw) * ETA_CHARGE * DT_HOURS
        free_steps = free_charge_soc / SOC_STEP_KWH
        charge_slope = unit_price / ETA_CHARGE
        charge_base = max(0.0, net_load_kw) * unit_price * DT_HOURS

        free_lo = np.ceil(state_indices - free_steps - TOL).astype(int)
        free_hi = state_indices
        free_val, free_arg = _window_min(dp, free_lo, free_hi)
        free_val = free_val + charge_base

        paid_lo = state_indices - max_charge_steps
        paid_hi = np.floor(state_indices - free_steps + TOL).astype(int)
        paid_metric = dp - charge_slope * states
        paid_val, paid_arg = _window_min(paid_metric, paid_lo, paid_hi)
        paid_val = (
            paid_val
            + charge_base
            + charge_slope * (states - free_charge_soc)
        )

        # Discharging transition: delta_soc <= 0.
        needed_discharge_soc = max(0.0, net_load_kw) * DT_HOURS / ETA_DISCHARGE
        needed_steps = needed_discharge_soc / SOC_STEP_KWH
        discharge_slope = unit_price * ETA_DISCHARGE
        discharge_base = max(0.0, net_load_kw) * unit_price * DT_HOURS

        linear_lo = state_indices
        linear_hi = np.floor(state_indices + needed_steps + TOL).astype(int)
        linear_metric = dp - discharge_slope * states
        linear_val, linear_arg = _window_min(linear_metric, linear_lo, linear_hi)
        linear_val = linear_val + discharge_base + discharge_slope * states

        zero_lo = np.ceil(state_indices + needed_steps - TOL).astype(int)
        zero_hi = state_indices + max_discharge_steps
        zero_val, zero_arg = _window_min(dp, zero_lo, zero_hi)

        candidates = np.vstack([free_val, paid_val, linear_val, zero_val])
        args = np.vstack([free_arg, paid_arg, linear_arg, zero_arg])
        choices = np.argmin(candidates, axis=0)
        dp = candidates[choices, state_indices]
        predecessors[t] = args[choices, state_indices]

    if not np.isfinite(dp[initial_index]):
        raise RuntimeError("No feasible terminal state found")

    path = np.empty(SLOTS_PER_DAY + 1, dtype=np.int32)
    path[-1] = initial_index
    for t in range(SLOTS_PER_DAY - 1, -1, -1):
        path[t] = predecessors[t, path[t + 1]]
        if path[t] < 0:
            raise RuntimeError(f"Broken predecessor path at slot {t}")

    soc = states[path]
    delta_soc = np.diff(soc)
    charge_kw = np.where(
        delta_soc > 0, delta_soc / (ETA_CHARGE * DT_HOURS), 0.0
    )
    discharge_kw = np.where(
        delta_soc < 0, -delta_soc * ETA_DISCHARGE / DT_HOURS, 0.0
    )
    purchase_kw = np.maximum(load_kw + charge_kw - pv_kw - discharge_kw, 0.0)
    curtailed_kw = purchase_kw + pv_kw + discharge_kw - load_kw - charge_kw
    slot_cost = price * purchase_kw * DT_HOURS

    solution = pd.DataFrame(
        {
            "slot": np.arange(SLOTS_PER_DAY),
            "price_yuan_per_kwh": price,
            "load_kw": load_kw,
            "pv_forecast_kw": pv_kw,
            "purchase_kw": purchase_kw,
            "charge_kw": charge_kw,
            "discharge_kw": discharge_kw,
            "curtailed_kw": curtailed_kw,
            "soc_start_kwh": soc[:-1],
            "soc_end_kwh": soc[1:],
            "purchase_kwh": purchase_kw * DT_HOURS,
            "charge_kwh": charge_kw * DT_HOURS,
            "discharge_kwh": discharge_kw * DT_HOURS,
            "slot_cost_yuan": slot_cost,
        }
    )

    balance_residual = (
        purchase_kw
        + pv_kw
        + discharge_kw
        - load_kw
        - charge_kw
        - curtailed_kw
    )
    simultaneous = (charge_kw > TOL) & (discharge_kw > TOL)
    if abs(soc[0] - SOC_INITIAL_KWH) > TOL or abs(soc[-1] - SOC_INITIAL_KWH) > TOL:
        raise AssertionError("Initial/terminal SOC condition failed")
    if soc.min() < SOC_MIN_KWH - TOL or soc.max() > SOC_MAX_KWH + TOL:
        raise AssertionError("SOC bound failed")
    if charge_kw.max() > POWER_MAX_KW + TOL or discharge_kw.max() > POWER_MAX_KW + TOL:
        raise AssertionError("Charge/discharge power bound failed")
    if simultaneous.any():
        raise AssertionError("Charge/discharge mutual exclusion failed")
    if np.abs(balance_residual).max() > TOL:
        raise AssertionError("Power balance failed")

    baseline_purchase_kw = np.maximum(load_kw - pv_kw, 0.0)
    baseline_cost = float(np.sum(price * baseline_purchase_kw * DT_HOURS))
    summary: dict[str, float | int | str] = {
        "method": "SOC dynamic programming on a 1 kWh lattice",
        "slots": SLOTS_PER_DAY,
        "total_purchase_kwh": float(solution["purchase_kwh"].sum()),
        "total_cost_yuan": float(slot_cost.sum()),
        "baseline_cost_without_storage_yuan": baseline_cost,
        "storage_cost_saving_yuan": baseline_cost - float(slot_cost.sum()),
        "total_load_kwh": float(np.sum(load_kw) * DT_HOURS),
        "total_pv_forecast_kwh": float(np.sum(pv_kw) * DT_HOURS),
        "total_charge_kwh": float(np.sum(charge_kw) * DT_HOURS),
        "total_discharge_kwh": float(np.sum(discharge_kw) * DT_HOURS),
        "total_curtailed_kwh": float(np.sum(curtailed_kw) * DT_HOURS),
        "initial_soc_kwh": float(soc[0]),
        "terminal_soc_kwh": float(soc[-1]),
        "min_soc_kwh": float(soc.min()),
        "max_soc_kwh": float(soc.max()),
        "max_charge_kw": float(charge_kw.max()),
        "max_discharge_kw": float(discharge_kw.max()),
        "simultaneous_charge_discharge_slots": int(simultaneous.sum()),
        "max_balance_residual_kw": float(np.abs(balance_residual).max()),
        "objective_recalculation_error_yuan": float(abs(slot_cost.sum() - dp[initial_index])),
    }
    return solution, summary


def main() -> None:
    project_root = Path(__file__).resolve().parents[2]
    input_path = project_root / "公共代码" / "数据" / "master_10min.csv"
    output_dir = project_root / "问题1" / "结果"
    output_dir.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(input_path)
    day = master.sort_values("slot").drop_duplicates("slot", keep="first")
    if len(day) != SLOTS_PER_DAY:
        raise ValueError(f"Expected 144 slots, got {len(day)}")
    required = ["price_fixed_yuan_per_kwh", "load_q1_kw", "pv_forecast_q1_kw"]
    if day[required].isna().any().any():
        raise ValueError("Problem 1 input contains missing values")

    solution, summary = solve_dispatch(
        day["price_fixed_yuan_per_kwh"].to_numpy(float),
        day["load_q1_kw"].to_numpy(float),
        day["pv_forecast_q1_kw"].to_numpy(float),
    )
    solution.insert(1, "time_end", day["time_end"].to_numpy())
    solution.to_csv(output_dir / "problem1_dispatch.csv", index=False, encoding="utf-8-sig")
    (output_dir / "problem1_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
