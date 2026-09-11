"""求解 C 题问题 1 的单日确定性 MILP，并输出可复核的中间结果。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix, vstack


ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = ROOT / "题目" / "附件" / "附件1.xlsx"
TEMPLATE_PATH = ROOT / "题目" / "附件" / "附件5" / "result1.xlsx"
RESULT_DIR = ROOT / "问题1" / "结果"

N = 144
DT = 1 / 6
ETA = 0.9
E_MIN = 1200.0
E_MAX = 10800.0
E_INITIAL = 6000.0
POWER_MAX = 5000.0
Q_MAX = POWER_MAX * DT


def variable_indices() -> dict[str, np.ndarray]:
    return {
        "grid": np.arange(0, N),
        "charge": np.arange(N, 2 * N),
        "discharge": np.arange(2 * N, 3 * N),
        "energy": np.arange(3 * N, 4 * N + 1),
        "binary": np.arange(4 * N + 1, 5 * N + 1),
    }


def solve() -> tuple[pd.DataFrame, dict[str, object]]:
    source = pd.read_excel(INPUT_PATH, sheet_name=0)
    if source.shape[0] != N or source.shape[1] < 4:
        raise ValueError("附件1应包含144行以及时间、电价、负荷和光伏预测四列")

    source = source.iloc[:, :4].copy()
    source.columns = ["source_time", "price", "load_kw", "pv_forecast_kw"]
    for column in ["price", "load_kw", "pv_forecast_kw"]:
        source[column] = pd.to_numeric(source[column], errors="raise")
    if source[["price", "load_kw", "pv_forecast_kw"]].isna().any().any():
        raise ValueError("附件1存在缺失数值")

    load_kwh = source["load_kw"].to_numpy(float) * DT
    pv_kwh = source["pv_forecast_kw"].to_numpy(float) * DT
    price = source["price"].to_numpy(float)

    idx = variable_indices()
    n_vars = 5 * N + 1
    objective = np.zeros(n_vars)
    objective[idx["grid"]] = price

    lower = np.zeros(n_vars)
    upper = np.full(n_vars, np.inf)
    upper[idx["charge"]] = Q_MAX
    upper[idx["discharge"]] = Q_MAX
    lower[idx["energy"]] = E_MIN
    upper[idx["energy"]] = E_MAX
    lower[idx["energy"][0]] = E_INITIAL
    upper[idx["energy"][0]] = E_INITIAL
    lower[idx["energy"][-1]] = E_INITIAL
    upper[idx["energy"][-1]] = E_INITIAL
    upper[idx["binary"]] = 1.0

    integrality = np.zeros(n_vars, dtype=int)
    integrality[idx["binary"]] = 1

    balance = lil_matrix((N, n_vars))
    soc = lil_matrix((N, n_vars))
    charge_mutex = lil_matrix((N, n_vars))
    discharge_mutex = lil_matrix((N, n_vars))

    for t in range(N):
        balance[t, idx["grid"][t]] = 1.0
        balance[t, idx["charge"][t]] = -1.0
        balance[t, idx["discharge"][t]] = 1.0

        soc[t, idx["energy"][t + 1]] = 1.0
        soc[t, idx["energy"][t]] = -1.0
        soc[t, idx["charge"][t]] = -ETA
        soc[t, idx["discharge"][t]] = 1.0 / ETA

        charge_mutex[t, idx["charge"][t]] = 1.0
        charge_mutex[t, idx["binary"][t]] = -Q_MAX

        discharge_mutex[t, idx["discharge"][t]] = 1.0
        discharge_mutex[t, idx["binary"][t]] = Q_MAX

    matrix = vstack([balance, soc, charge_mutex, discharge_mutex]).tocsr()
    constraint_lower = np.concatenate(
        [load_kwh - pv_kwh, np.zeros(N), np.full(N, -np.inf), np.full(N, -np.inf)]
    )
    constraint_upper = np.concatenate(
        [np.full(N, np.inf), np.zeros(N), np.zeros(N), np.full(N, Q_MAX)]
    )

    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix, constraint_lower, constraint_upper),
        options={"disp": False},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP求解失败：{result.message}")

    tolerance = 1e-7
    grid = result.x[idx["grid"]]
    charge = result.x[idx["charge"]]
    discharge = result.x[idx["discharge"]]
    energy = result.x[idx["energy"]]
    binary = result.x[idx["binary"]]
    for values in [grid, charge, discharge]:
        values[np.abs(values) < tolerance] = 0.0

    template = pd.read_excel(TEMPLATE_PATH, sheet_name="计划购电量")
    if len(template) != N:
        raise ValueError("result1.xlsx的计划购电量工作表应有144个数据位置")

    schedule = pd.DataFrame(
        {
            "slot": np.arange(N),
            "source_time": source["source_time"].astype(str),
            "result_interval": template.iloc[:, 0].astype(str),
            "price_yuan_per_kwh": price,
            "load_kw": source["load_kw"].to_numpy(float),
            "pv_forecast_kw": source["pv_forecast_kw"].to_numpy(float),
            "load_kwh": load_kwh,
            "pv_forecast_kwh": pv_kwh,
            "grid_purchase_kwh": grid,
            "charge_kwh": charge,
            "discharge_kwh": discharge,
            "energy_start_kwh": energy[:-1],
            "energy_end_kwh": energy[1:],
            "charge_state": np.rint(binary).astype(int),
            "period_cost_yuan": price * grid,
        }
    )

    supply_margin = grid + pv_kwh + discharge - load_kwh - charge
    soc_residual = energy[1:] - energy[:-1] - ETA * charge + discharge / ETA
    simultaneous = np.minimum(charge, discharge)
    selected_intervals = [
        "10:00-10:10",
        "12:00-12:10",
        "14:00-14:10",
        "16:00-16:10",
        "18:00-18:10",
        "20:00-20:10",
    ]
    selected = {
        interval: float(
            schedule.loc[schedule["result_interval"] == interval, "grid_purchase_kwh"].iloc[0]
        )
        for interval in selected_intervals
    }
    groups = []
    group_labels = [
        "0:00-4:00",
        "4:00-8:00",
        "8:00-12:00",
        "12:00-16:00",
        "16:00-20:00",
        "20:00-24:00",
    ]
    for group, label in enumerate(group_labels):
        start = group * 24
        stop = start + 24
        groups.append(
            {
                "interval": label,
                "charge_kwh": float(charge[start:stop].sum()),
                "discharge_kwh": float(discharge[start:stop].sum()),
            }
        )

    summary: dict[str, object] = {
        "solver_status": str(result.message),
        "objective_yuan": float(np.dot(price, grid)),
        "total_grid_purchase_kwh": float(grid.sum()),
        "total_charge_kwh": float(charge.sum()),
        "total_discharge_kwh": float(discharge.sum()),
        "energy_initial_kwh": float(energy[0]),
        "energy_final_kwh": float(energy[-1]),
        "energy_min_kwh": float(energy.min()),
        "energy_max_kwh": float(energy.max()),
        "minimum_supply_margin_kwh": float(supply_margin.min()),
        "maximum_soc_residual_kwh": float(np.abs(soc_residual).max()),
        "maximum_simultaneous_charge_discharge_kwh": float(simultaneous.max()),
        "selected_grid_purchase_kwh": selected,
        "four_hour_totals": groups,
    }
    return schedule, summary


def write_outputs(schedule: pd.DataFrame, summary: dict[str, object]) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    schedule.to_csv(RESULT_DIR / "问题1完整调度.csv", index=False, encoding="utf-8-sig")
    (RESULT_DIR / "调度数据.json").write_text(
        json.dumps(schedule.to_dict(orient="records"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (RESULT_DIR / "求解摘要.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

if __name__ == "__main__":
    solved_schedule, solved_summary = solve()
    write_outputs(solved_schedule, solved_summary)
    print(json.dumps(solved_summary, ensure_ascii=False, indent=2))
