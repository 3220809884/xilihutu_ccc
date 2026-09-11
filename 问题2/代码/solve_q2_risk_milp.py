"""问题2：因果日前预测、风险备用 MILP 与逐时实际执行。

只读取问题2允许的信息：附件1固定电价、附件2历史负载和实际光伏。
功率单位为 kW，电量和 SOC 单位为 kWh，时间步长为 1/6 h。
slot=0 表示 00:00--00:10，slot=143 表示 23:50--24:00。
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
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "问题2" / "结果"
INPUT = OUT / "q2_preprocessed_10min.csv"
SLOTS = 144
DT = 1 / 6
ETA_CHARGE = 0.9
ETA_DISCHARGE = 0.9
P_MAX = 5000.0
S_MIN = 1200.0
S_MAX = 10800.0
S_INITIAL = 6000.0
S_TARGET = 6000.0
RESERVE_ALPHA = 0.80
RESERVE_WINDOW_DAYS = 56
MIN_GROUP_DAYS = 7
MODEL_START_DAY = 21
LAGS = (1, 2, 3, 7, 14)
TYPE_COLUMNS = ("holiday", "makeup_workday", "weekend", "workday")
TOL = 1e-6


def time_label(boundary_slot: int) -> str:
    minute = boundary_slot * 10
    return f"{minute // 60:02d}:{minute % 60:02d}"


def interval_label(slot: int) -> str:
    return f"{time_label(slot)}-{time_label(slot + 1)}"


def feature_row(
    series: np.ndarray,
    day: int,
    slot: int,
    calendar: pd.DataFrame,
    include_calendar: bool,
) -> np.ndarray:
    """构造仅依赖目标日前完整历史的特征。"""

    def at(d: int, s: int) -> float:
        return float(series[d, s % SLOTS])

    values = [at(day - lag, slot) for lag in LAGS]
    values += [
        float(np.mean(series[day - 3 : day, slot])),
        float(np.mean(series[day - 7 : day, slot])),
        float(np.mean(series[day - 14 : day, slot])),
        at(day - 1, slot - 2),
        at(day - 1, slot - 1),
        at(day - 1, slot + 1),
        at(day - 1, slot + 2),
    ]
    angle = 2 * np.pi * slot / SLOTS
    values += [np.sin(angle), np.cos(angle)]
    if include_calendar:
        row = calendar.iloc[day]
        weekday_angle = 2 * np.pi * int(row["weekday"]) / 7
        month_angle = 2 * np.pi * (int(row["month"]) - 1) / 12
        values += [
            np.sin(weekday_angle),
            np.cos(weekday_angle),
            np.sin(month_angle),
            np.cos(month_angle),
            *[float(row["day_type"] == name) for name in TYPE_COLUMNS],
        ]
    return np.asarray(values, dtype=float)


def sample_matrix(
    series: np.ndarray,
    calendar: pd.DataFrame,
    days: range,
    include_calendar: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[float] = []
    groups: list[str] = []
    for day in days:
        for slot in range(SLOTS):
            xs.append(feature_row(series, day, slot, calendar, include_calendar))
            ys.append(float(series[day, slot]))
            groups.append(str(calendar.iloc[day]["day_type"]))
    return np.asarray(xs), np.asarray(ys), np.asarray(groups)


def fit_standardized_xgb(X: np.ndarray, y: np.ndarray):
    """标准化只作用于预测特征，输出仍还原为原始 kW。"""
    return make_pipeline(
        StandardScaler(),
        XGBRegressor(
            n_estimators=140,
            max_depth=5,
            learning_rate=0.045,
            min_child_weight=4,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            objective="reg:squarederror",
            n_jobs=4,
            random_state=2026,
        ),
    ).fit(X, y)


def fit_standardized_ridge(X: np.ndarray, y: np.ndarray):
    return make_pipeline(StandardScaler(), Ridge(alpha=10.0, solver="lsqr")).fit(X, y)


def causal_day_ahead_forecasts(
    load: np.ndarray,
    pv: np.ndarray,
    calendar: pd.DataFrame,
    dates: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, object]]]:
    """生成全年因果预测；正式交付从2月1日起，1月用于冷启动。"""
    n_days = len(dates)
    load_pred = np.zeros_like(load)
    pv_pred = np.zeros_like(pv)
    pv_mean3_pred = np.zeros_like(pv)
    audits: list[dict[str, object]] = []
    load_global = None
    load_grouped: dict[str, object] = {}
    pv_ridge = None
    refit_date = None
    train_end_date = None

    for day in range(n_days):
        if day == 0:
            load_pred[day] = np.zeros(SLOTS)
            pv_pred[day] = np.zeros(SLOTS)
            pv_mean3_pred[day] = np.zeros(SLOTS)
            load_method = pv_method = "cold_start_zero"
        elif day < 14:
            load_pred[day] = load[day - 1]
            pv_pred[day] = pv[day - 1]
            pv_mean3_pred[day] = np.mean(pv[max(0, day - 3) : day], axis=0)
            load_method = pv_method = "previous_completed_day"
        else:
            pv_mean3_pred[day] = np.mean(pv[day - 3 : day], axis=0)
            should_refit = day == MODEL_START_DAY or (
                day >= MODEL_START_DAY and pd.Timestamp(dates[day]).day == 1
            )
            if should_refit:
                training_days = range(14, day)
                X_load, y_load, groups = sample_matrix(
                    load, calendar, training_days, include_calendar=True
                )
                load_global = fit_standardized_xgb(X_load, y_load)
                load_grouped = {}
                for day_type in TYPE_COLUMNS:
                    mask = groups == day_type
                    if int(mask.sum()) >= MIN_GROUP_DAYS * SLOTS:
                        load_grouped[day_type] = fit_standardized_xgb(
                            X_load[mask], y_load[mask]
                        )
                X_pv, y_pv, _ = sample_matrix(
                    pv, calendar, training_days, include_calendar=False
                )
                pv_ridge = fit_standardized_ridge(X_pv, y_pv)
                refit_date = dates[day]
                train_end_date = dates[day - 1]

            if load_global is None or pv_ridge is None:
                load_pred[day] = np.mean(load[day - 3 : day], axis=0)
                pv_pred[day] = pv_mean3_pred[day]
                load_method = pv_method = "three_completed_day_mean"
            else:
                X_load_day = np.vstack(
                    [feature_row(load, day, slot, calendar, True) for slot in range(SLOTS)]
                )
                day_type = str(calendar.iloc[day]["day_type"])
                load_model = load_grouped.get(day_type, load_global)
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    load_value = load_model.predict(X_load_day)
                if not np.isfinite(load_value).all():
                    raise ValueError(f"non-finite load forecast on {dates[day]}")
                load_pred[day] = np.maximum(load_value, 0)
                X_pv_day = np.vstack(
                    [feature_row(pv, day, slot, calendar, False) for slot in range(SLOTS)]
                )
                with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
                    pv_value = pv_ridge.predict(X_pv_day)
                if not np.isfinite(pv_value).all():
                    raise ValueError(f"non-finite PV forecast on {dates[day]}")
                pv_pred[day] = np.maximum(pv_value, 0)
                # 夜间零出力由此前14天同一时段的历史观测判定，不看目标日。
                historical_sun = np.max(pv[day - 14 : day], axis=0) > 1.0
                pv_pred[day, ~historical_sun] = 0.0
                load_method = (
                    "standardized_grouped_calendar_xgboost"
                    if day_type in load_grouped
                    else "standardized_calendar_xgboost_fallback"
                )
                pv_method = "standardized_historical_ridge"

        load_pred[day] = np.maximum(load_pred[day], 0)
        pv_pred[day] = np.maximum(pv_pred[day], 0)
        pv_mean3_pred[day] = np.maximum(pv_mean3_pred[day], 0)
        audits.append(
            {
                "date": dates[day],
                "issued_at": dates[day] + " 00:00",
                "history_through_date": dates[day - 1] if day else "",
                "load_model": load_method,
                "pv_model": pv_method,
                "model_refit_date": refit_date or "",
                "model_train_end_date": train_end_date or "",
                "target_day_actual_used_for_forecast": False,
            }
        )
    return load_pred, pv_pred, pv_mean3_pred, audits


def causal_reserve(
    actual_net: np.ndarray,
    point_forecast: np.ndarray,
    calendar: pd.DataFrame,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    """按目标日类型和时段估计单侧正向预测误差分位数。"""
    n_days = len(actual_net)
    reserve = np.zeros_like(actual_net)
    audit: list[dict[str, object]] = []
    residual = actual_net - point_forecast
    for day in range(n_days):
        current_type = str(calendar.iloc[day]["day_type"])
        start = max(1, day - RESERVE_WINDOW_DAYS)
        same_type = [
            j for j in range(start, day) if str(calendar.iloc[j]["day_type"]) == current_type
        ]
        if len(same_type) >= MIN_GROUP_DAYS:
            indices = same_type
            source = "same_day_type"
        else:
            indices = list(range(max(1, day - 28), day))
            source = "all_day_types_fallback" if indices else "no_history"
        if indices:
            reserve[day] = np.clip(
                np.quantile(residual[indices], RESERVE_ALPHA, axis=0, method="linear"),
                0,
                P_MAX,
            )
        audit.append(
            {
                "date": calendar.iloc[day]["date"],
                "reserve_source": source,
                "reserve_history_count": len(indices),
                "reserve_history_start": calendar.iloc[indices[0]]["date"] if indices else "",
                "reserve_history_end": calendar.iloc[indices[-1]]["date"] if indices else "",
                "reserve_quantile": RESERVE_ALPHA,
            }
        )
    return reserve, audit


def plan_day(
    net_forecast: np.ndarray,
    reserve_requirement: np.ndarray,
    price: np.ndarray,
    initial_soc: float,
) -> dict[str, np.ndarray | float]:
    """求解单日含软备用缺口的 MILP。

    计划功率平衡使用 >=。备用缺口变量只用于保证任意日初 SOC 下可行，
    并按五倍电价惩罚；财务结算只统计计划购电量，不统计该惩罚。
    """
    n = SLOTS
    assert net_forecast.shape == reserve_requirement.shape == price.shape == (n,)
    assert np.all(price > 0) and np.all(reserve_requirement >= 0)
    assert S_MIN - TOL <= initial_soc <= S_MAX + TOL

    # x = [q(n), c(n), d(n), S(n+1), z(n), u(n)]
    q, c, d, s, z, u = 0, n, 2 * n, 3 * n, 4 * n + 1, 5 * n + 1
    size = 6 * n + 1
    objective = np.zeros(size)
    objective[q : q + n] = price * DT
    objective[u : u + n] = 5 * price * DT
    lower = np.zeros(size)
    upper = np.full(size, np.inf)
    upper[c : c + n] = P_MAX
    upper[d : d + n] = P_MAX
    lower[s : s + n + 1] = S_MIN
    upper[s : s + n + 1] = S_MAX
    lower[s] = upper[s] = float(np.clip(initial_soc, S_MIN, S_MAX))
    lower[s + n] = upper[s + n] = S_TARGET
    upper[z : z + n] = 1
    upper[u : u + n] = reserve_requirement
    integrality = np.zeros(size, dtype=int)
    integrality[z : z + n] = 1

    matrix = lil_matrix((6 * n, size))
    lo = np.full(6 * n, -np.inf)
    hi = np.full(6 * n, np.inf)
    for t in range(n):
        # 预测功率平衡：q + d - c >= L_hat - PV_hat。
        matrix[t, q + t], matrix[t, c + t], matrix[t, d + t] = 1, -1, 1
        lo[t] = net_forecast[t]
        # SOC 动态。
        matrix[n + t, s + t + 1], matrix[n + t, s + t] = 1, -1
        matrix[n + t, c + t] = -ETA_CHARGE * DT
        matrix[n + t, d + t] = DT / ETA_DISCHARGE
        lo[n + t] = hi[n + t] = 0
        # 充放电互斥。
        matrix[2 * n + t, c + t], matrix[2 * n + t, z + t] = 1, -P_MAX
        hi[2 * n + t] = 0
        matrix[3 * n + t, d + t], matrix[3 * n + t, z + t] = 1, P_MAX
        hi[3 * n + t] = P_MAX
        # 备用功率：d + (R-u) <= P_max。
        matrix[4 * n + t, d + t], matrix[4 * n + t, u + t] = 1, -1
        hi[4 * n + t] = P_MAX - reserve_requirement[t]
        # 备用能量：S_t >= S_min + (d + R-u) * DT / eta_d。
        matrix[5 * n + t, s + t] = 1
        matrix[5 * n + t, d + t] = -DT / ETA_DISCHARGE
        matrix[5 * n + t, u + t] = DT / ETA_DISCHARGE
        lo[5 * n + t] = S_MIN + reserve_requirement[t] * DT / ETA_DISCHARGE

    result = milp(
        objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(matrix.tocsc(), lo, hi),
        options={"mip_rel_gap": 1e-8, "time_limit": 30.0, "presolve": True},
    )
    if not result.success or result.x is None:
        raise RuntimeError(f"MILP failed: {result.message}")
    x = result.x
    purchase = np.maximum(x[q : q + n], 0)
    charge = np.maximum(x[c : c + n], 0)
    discharge = np.maximum(x[d : d + n], 0)
    states = x[s : s + n + 1]
    reserve_shortfall = np.clip(x[u : u + n], 0, reserve_requirement)
    reserve_achieved = reserve_requirement - reserve_shortfall
    plan_cost = float(price @ purchase * DT)
    risk_penalty = float(5 * price @ reserve_shortfall * DT)

    assert np.min(purchase + discharge - charge - net_forecast) >= -TOL
    assert np.max(
        np.abs(
            np.diff(states)
            - ETA_CHARGE * DT * charge
            + DT / ETA_DISCHARGE * discharge
        )
    ) < TOL
    assert np.max(discharge + reserve_achieved - P_MAX) < TOL
    assert np.min(
        states[:-1]
        - S_MIN
        - (discharge + reserve_achieved) * DT / ETA_DISCHARGE
    ) > -TOL
    assert not np.any((charge > TOL) & (discharge > TOL))
    assert abs(plan_cost + risk_penalty - float(result.fun)) < 1e-4
    return {
        "purchase": purchase,
        "charge": charge,
        "discharge": discharge,
        "soc": states,
        "reserve_achieved": reserve_achieved,
        "reserve_shortfall": reserve_shortfall,
        "plan_cost": plan_cost,
        "risk_penalty": risk_penalty,
        "objective": float(result.fun),
        "mip_gap": float(result.mip_gap or 0.0),
    }


def execute_day(
    purchase: np.ndarray, actual_net: np.ndarray, initial_soc: float
) -> dict[str, np.ndarray]:
    """实际执行只使用当前时段已实现的净负荷和当前 SOC。"""
    charge = np.zeros(SLOTS)
    discharge = np.zeros(SLOTS)
    emergency = np.zeros(SLOTS)
    spill = np.zeros(SLOTS)
    soc = np.zeros(SLOTS + 1)
    soc[0] = initial_soc
    for t in range(SLOTS):
        surplus = purchase[t] - actual_net[t]
        if surplus >= 0:
            charge[t] = min(
                surplus,
                P_MAX,
                max(0.0, (S_MAX - soc[t]) / (ETA_CHARGE * DT)),
            )
            spill[t] = surplus - charge[t]
        else:
            deficit = -surplus
            discharge[t] = min(
                deficit,
                P_MAX,
                max(0.0, (soc[t] - S_MIN) * ETA_DISCHARGE / DT),
            )
            emergency[t] = deficit - discharge[t]
        soc[t + 1] = (
            soc[t]
            + ETA_CHARGE * charge[t] * DT
            - discharge[t] * DT / ETA_DISCHARGE
        )
    return {
        "charge": charge,
        "discharge": discharge,
        "emergency": emergency,
        "spill": spill,
        "soc": soc,
    }


def emergency_events(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for date, day in frame.groupby("date", sort=True):
        energy = day["emergency_kwh"].to_numpy(float)
        t = 0
        while t < SLOTS:
            if energy[t] <= 1e-8:
                t += 1
                continue
            start = t
            while t < SLOTS and energy[t] > 1e-8:
                t += 1
            rows.append(
                {
                    "date": date,
                    "start_slot": start,
                    "end_slot_exclusive": t,
                    "interval": f"{time_label(start)}-{time_label(t)}",
                    "emergency_kwh": float(energy[start:t].sum()),
                    "emergency_cost_yuan": float(
                        day["emergency_cost_yuan"].iloc[start:t].sum()
                    ),
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "date",
            "start_slot",
            "end_slot_exclusive",
            "interval",
            "emergency_kwh",
            "emergency_cost_yuan",
        ],
    )


def forecast_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for day_type in ["all", *TYPE_COLUMNS]:
        part = frame if day_type == "all" else frame[frame["day_type"] == day_type]
        if part.empty:
            continue
        for target, actual_col, forecast_col in [
            ("load", "load_actual_kw", "load_forecast_kw"),
            ("pv", "pv_actual_kw", "pv_forecast_kw"),
            ("net_load", "net_actual_kw", "net_forecast_kw"),
            ("pv_mean3_alternative", "pv_actual_kw", "pv_mean3_forecast_kw"),
        ]:
            error = part[actual_col].to_numpy() - part[forecast_col].to_numpy()
            rows.append(
                {
                    "day_type": day_type,
                    "target": target,
                    "observations": len(part),
                    "MAE_kW": float(np.mean(np.abs(error))),
                    "RMSE_kW": float(np.sqrt(np.mean(error**2))),
                    "mean_error_kW": float(np.mean(error)),
                }
            )
    return pd.DataFrame(rows)


def validate(frame: pd.DataFrame) -> dict[str, float | int | bool]:
    assert len(frame) == 365 * SLOTS
    assert not frame.duplicated(["date", "slot"]).any()
    assert (frame.groupby("date").size() == SLOTS).all()
    assert np.array_equal(frame["slot"].to_numpy(), np.tile(np.arange(SLOTS), 365))
    numeric = frame.select_dtypes(include="number").to_numpy()
    assert np.isfinite(numeric).all()
    for column in [
        "plan_kw",
        "adjustment_kw",
        "emergency_kw",
        "charge_kw",
        "discharge_kw",
        "spill_kw",
        "reserve_requirement_kw",
        "reserve_achieved_kw",
        "reserve_shortfall_kw",
    ]:
        assert frame[column].min() >= -TOL
    assert frame["adjustment_kw"].max() <= TOL
    assert frame["soc_start_kwh"].between(S_MIN - TOL, S_MAX + TOL).all()
    assert frame["soc_end_kwh"].between(S_MIN - TOL, S_MAX + TOL).all()
    assert not ((frame["charge_kw"] > TOL) & (frame["discharge_kw"] > TOL)).any()
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
    plan_balance = (
        frame["plan_kw"]
        + frame["planned_discharge_kw"]
        - frame["planned_charge_kw"]
        - frame["net_forecast_kw"]
    )
    actual_soc = (
        frame["soc_end_kwh"]
        - frame["soc_start_kwh"]
        - ETA_CHARGE * DT * frame["charge_kw"]
        + DT / ETA_DISCHARGE * frame["discharge_kw"]
    )
    planned_soc = (
        frame["planned_soc_end_kwh"]
        - frame["planned_soc_start_kwh"]
        - ETA_CHARGE * DT * frame["planned_charge_kw"]
        + DT / ETA_DISCHARGE * frame["planned_discharge_kw"]
    )
    continuity = (
        frame["soc_start_kwh"].to_numpy()[1:]
        - frame["soc_end_kwh"].to_numpy()[:-1]
    )
    reserve_power = (
        frame["planned_discharge_kw"] + frame["reserve_achieved_kw"] - P_MAX
    )
    reserve_energy = (
        S_MIN
        + (frame["planned_discharge_kw"] + frame["reserve_achieved_kw"])
        * DT
        / ETA_DISCHARGE
        - frame["planned_soc_start_kwh"]
    )
    assert np.max(np.abs(actual_balance)) < TOL
    assert np.min(plan_balance) >= -TOL
    assert np.max(np.abs(actual_soc)) < TOL
    assert np.max(np.abs(planned_soc)) < TOL
    assert np.max(np.abs(continuity)) < TOL
    assert reserve_power.max() < TOL and reserve_energy.max() < TOL
    for stem in ["plan", "adjustment", "emergency", "charge", "discharge"]:
        assert np.max(np.abs(frame[f"{stem}_kwh"] - frame[f"{stem}_kw"] * DT)) < TOL
    assert np.max(
        np.abs(
            frame["plan_cost_yuan"]
            - frame["price_yuan_per_kwh"] * frame["plan_kwh"]
        )
    ) < TOL
    assert np.max(
        np.abs(
            frame["emergency_cost_yuan"]
            - 5 * frame["price_yuan_per_kwh"] * frame["emergency_kwh"]
        )
    ) < TOL
    return {
        "max_actual_balance_residual_kw": float(np.max(np.abs(actual_balance))),
        "min_planned_balance_slack_kw": float(np.min(plan_balance)),
        "max_actual_soc_residual_kwh": float(np.max(np.abs(actual_soc))),
        "max_planned_soc_residual_kwh": float(np.max(np.abs(planned_soc))),
        "max_cross_slot_and_day_soc_residual_kwh": float(
            np.max(np.abs(continuity))
        ),
        "max_reserve_power_violation_kw": float(max(0.0, reserve_power.max())),
        "max_reserve_energy_violation_kwh": float(max(0.0, reserve_energy.max())),
        "simultaneous_actual_charge_discharge_slots": int(
            ((frame["charge_kw"] > TOL) & (frame["discharge_kw"] > TOL)).sum()
        ),
        "all_assertions_passed": True,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    required = [
        "date",
        "slot",
        "time_end",
        "price_fixed_yuan_per_kwh",
        "load_actual_kw",
        "pv_actual_kw",
        "weekday",
        "month",
        "day_type",
    ]
    data = pd.read_csv(INPUT, usecols=required).sort_values(["date", "slot"])
    data = data.reset_index(drop=True)
    dates = data["date"].drop_duplicates().tolist()
    assert dates == pd.date_range("2025-01-01", "2025-12-31").strftime("%Y-%m-%d").tolist()
    assert len(data) == 365 * SLOTS and not data.duplicated(["date", "slot"]).any()
    assert np.array_equal(data["slot"].to_numpy(), np.tile(np.arange(SLOTS), 365))
    expected_end = [time_label(i) for i in range(1, SLOTS)] + ["24:00"]
    assert data.groupby("date")["time_end"].apply(list).iloc[0] == expected_end
    assert not any("forecast" in column or "volatile" in column or "issue" in column for column in data.columns)

    calendar = data.drop_duplicates("date")[
        ["date", "weekday", "month", "day_type"]
    ].reset_index(drop=True)
    load = data["load_actual_kw"].to_numpy(float).reshape(365, SLOTS)
    pv = data["pv_actual_kw"].to_numpy(float).reshape(365, SLOTS)
    prices = data["price_fixed_yuan_per_kwh"].to_numpy(float).reshape(365, SLOTS)
    assert np.all(prices == prices[0])
    price = prices[0]
    actual_net = load - pv

    load_pred, pv_pred, pv_mean3_pred, forecast_audit = causal_day_ahead_forecasts(
        load, pv, calendar, dates
    )
    point_net = load_pred - pv_pred
    reserve, reserve_audit = causal_reserve(actual_net, point_net, calendar)
    for left, right in zip(forecast_audit, reserve_audit):
        left.update(right)

    records: list[pd.DataFrame] = []
    solver_audit: list[dict[str, object]] = []
    initial_soc = S_INITIAL
    max_gap = 0.0
    for day, date in enumerate(dates):
        plan = plan_day(point_net[day], reserve[day], price, initial_soc)
        actual = execute_day(plan["purchase"], actual_net[day], initial_soc)
        achieved = np.asarray(plan["reserve_achieved"])
        shortfall = np.asarray(plan["reserve_shortfall"])
        block = pd.DataFrame(
            {
                "date": date,
                "slot": np.arange(SLOTS),
                "time_interval": [interval_label(t) for t in range(SLOTS)],
                "time_end": [time_label(t + 1) for t in range(SLOTS)],
                "day_type": calendar.iloc[day]["day_type"],
                "price_yuan_per_kwh": price,
                "load_actual_kw": load[day],
                "pv_actual_kw": pv[day],
                "net_actual_kw": actual_net[day],
                "load_forecast_kw": load_pred[day],
                "pv_forecast_kw": pv_pred[day],
                "pv_mean3_forecast_kw": pv_mean3_pred[day],
                "net_forecast_kw": point_net[day],
                "forecast_error_kw": actual_net[day] - point_net[day],
                "reserve_requirement_kw": reserve[day],
                "reserve_achieved_kw": achieved,
                "reserve_shortfall_kw": shortfall,
                "plan_kw": np.asarray(plan["purchase"]),
                "adjustment_kw": np.zeros(SLOTS),
                "planned_charge_kw": np.asarray(plan["charge"]),
                "planned_discharge_kw": np.asarray(plan["discharge"]),
                "planned_soc_start_kwh": np.asarray(plan["soc"])[:-1],
                "planned_soc_end_kwh": np.asarray(plan["soc"])[1:],
                "charge_kw": actual["charge"],
                "discharge_kw": actual["discharge"],
                "emergency_kw": actual["emergency"],
                "spill_kw": actual["spill"],
                "soc_start_kwh": actual["soc"][:-1],
                "soc_end_kwh": actual["soc"][1:],
            }
        )
        for stem in ["plan", "adjustment", "emergency", "charge", "discharge"]:
            block[f"{stem}_kwh"] = block[f"{stem}_kw"] * DT
        block["plan_cost_yuan"] = block["price_yuan_per_kwh"] * block["plan_kwh"]
        block["adjustment_cost_yuan"] = 0.0
        block["emergency_cost_yuan"] = (
            5 * block["price_yuan_per_kwh"] * block["emergency_kwh"]
        )
        block["total_cost_yuan"] = (
            block["plan_cost_yuan"] + block["emergency_cost_yuan"]
        )
        risk_no_storage_plan = np.maximum(point_net[day] + reserve[day], 0)
        risk_no_storage_emergency = np.maximum(
            actual_net[day] - risk_no_storage_plan, 0
        )
        block["baseline_risk_no_storage_cost_yuan"] = (
            price * risk_no_storage_plan * DT
            + 5 * price * risk_no_storage_emergency * DT
        )
        point_no_storage_plan = np.maximum(point_net[day], 0)
        point_no_storage_emergency = np.maximum(
            actual_net[day] - point_no_storage_plan, 0
        )
        block["baseline_point_no_storage_cost_yuan"] = (
            price * point_no_storage_plan * DT
            + 5 * price * point_no_storage_emergency * DT
        )
        records.append(block)
        max_gap = max(max_gap, float(plan["mip_gap"]))
        forecast_audit[day].update(
            {
                "initial_soc_kwh": float(initial_soc),
                "planned_terminal_soc_kwh": float(np.asarray(plan["soc"])[-1]),
                "actual_terminal_soc_kwh": float(actual["soc"][-1]),
                "reserve_requirement_kwh": float(reserve[day].sum() * DT),
                "reserve_shortfall_kwh": float(shortfall.sum() * DT),
                "milp_financial_plan_cost_yuan": float(plan["plan_cost"]),
                "milp_risk_penalty_yuan": float(plan["risk_penalty"]),
                "milp_gap": float(plan["mip_gap"]),
            }
        )
        solver_audit.append(forecast_audit[day])
        initial_soc = float(actual["soc"][-1])
        if day % 30 == 0:
            print(
                f"executed through {date}; SOC={initial_soc:.3f}; "
                f"reserve shortfall={shortfall.sum() * DT:.3f} kWh",
                flush=True,
            )

    all_rows = pd.concat(records, ignore_index=True)
    checks = validate(all_rows)
    delivered = all_rows[all_rows["date"] >= "2025-02-01"].copy()
    assert len(delivered) == 334 * SLOTS
    events = emergency_events(delivered)
    assert abs(events["emergency_kwh"].sum() - delivered["emergency_kwh"].sum()) < TOL

    sum_columns = [
        "plan_kwh",
        "adjustment_kwh",
        "emergency_kwh",
        "charge_kwh",
        "discharge_kwh",
        "plan_cost_yuan",
        "adjustment_cost_yuan",
        "emergency_cost_yuan",
        "total_cost_yuan",
        "baseline_risk_no_storage_cost_yuan",
        "baseline_point_no_storage_cost_yuan",
    ]
    daily = delivered.groupby("date")[sum_columns].sum().reset_index()
    daily["day_type"] = delivered.groupby("date")["day_type"].first().to_numpy()
    daily["soc_start_kwh"] = delivered.groupby("date")["soc_start_kwh"].first().to_numpy()
    daily["soc_end_kwh"] = delivered.groupby("date")["soc_end_kwh"].last().to_numpy()
    daily["reserve_requirement_kwh"] = (
        delivered.groupby("date")["reserve_requirement_kw"].sum().to_numpy() * DT
    )
    daily["reserve_shortfall_kwh"] = (
        delivered.groupby("date")["reserve_shortfall_kw"].sum().to_numpy() * DT
    )
    metrics = forecast_metrics(delivered)
    overall = metrics[metrics["day_type"] == "all"].set_index("target")

    sums = {name: float(delivered[name].sum()) for name in sum_columns}
    summary: dict[str, object] = {
        "method": "causal grouped XGBoost load + historical Ridge PV + day-type error reserve + daily MILP + causal execution",
        "allowed_sources": ["附件1固定电价", "附件2历史负载和实际光伏"],
        "attachment3_used": False,
        "start_date": "2025-02-01",
        "end_date": "2025-12-31",
        "days": 334,
        "slots": 334 * SLOTS,
        "time_step_hours": DT,
        "slot_definition": "slot=0 is 00:00-00:10; slot=143 is 23:50-24:00",
        "power_unit": "kW",
        "energy_unit": "kWh",
        "planning_balance_relation": ">=",
        "plan_charge_basis": "planned quantity, whether used or not",
        "emergency_price_multiplier": 5,
        "load_model": "standardized day-type grouped calendar XGBoost with global fallback",
        "pv_model": "standardized Ridge using Attachment-2 PV history only",
        "reserve_method": "causal positive empirical quantile by day type and slot",
        "reserve_alpha": RESERVE_ALPHA,
        "reserve_window_days": RESERVE_WINDOW_DAYS,
        "reserve_alpha_basis": "newsvendor cost ratio: 1-1/5=0.80",
        "feb1_initial_soc_kwh": float(
            all_rows.loc[all_rows["date"] == "2025-02-01", "soc_start_kwh"].iloc[0]
        ),
        "dec31_terminal_soc_kwh": float(initial_soc),
        "min_soc_kwh": float(delivered["soc_end_kwh"].min()),
        "max_soc_kwh": float(delivered["soc_end_kwh"].max()),
        "load_forecast_MAE_kW": float(overall.loc["load", "MAE_kW"]),
        "load_forecast_RMSE_kW": float(overall.loc["load", "RMSE_kW"]),
        "pv_forecast_MAE_kW": float(overall.loc["pv", "MAE_kW"]),
        "pv_forecast_RMSE_kW": float(overall.loc["pv", "RMSE_kW"]),
        "net_forecast_MAE_kW": float(overall.loc["net_load", "MAE_kW"]),
        "net_forecast_RMSE_kW": float(overall.loc["net_load", "RMSE_kW"]),
        "point_forecast_coverage": float(
            (delivered["net_actual_kw"] <= delivered["net_forecast_kw"]).mean()
        ),
        "risk_buffer_coverage": float(
            (
                delivered["net_actual_kw"]
                <= delivered["net_forecast_kw"] + delivered["reserve_requirement_kw"]
            ).mean()
        ),
        "reserve_achievement_ratio": float(
            delivered["reserve_achieved_kw"].sum()
            / max(delivered["reserve_requirement_kw"].sum(), TOL)
        ),
        "reserve_shortfall_kwh": float(delivered["reserve_shortfall_kw"].sum() * DT),
        "emergency_events": len(events),
        "max_daily_milp_gap": max_gap,
        "checks": checks,
        "python_version": platform.python_version(),
        "scipy_version": scipy.__version__,
        "preprocessed_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        **sums,
    }
    summary["cost_saving_vs_risk_no_storage_yuan"] = (
        summary["baseline_risk_no_storage_cost_yuan"] - summary["total_cost_yuan"]
    )
    summary["cost_saving_vs_point_no_storage_yuan"] = (
        summary["baseline_point_no_storage_cost_yuan"] - summary["total_cost_yuan"]
    )

    all_rows.to_csv(
        OUT / "problem2_dispatch_all_2025.csv", index=False, encoding="utf-8-sig"
    )
    delivered[
        [
            "date",
            "slot",
            "time_interval",
            "day_type",
            "load_forecast_kw",
            "pv_forecast_kw",
            "pv_mean3_forecast_kw",
            "net_forecast_kw",
            "reserve_requirement_kw",
        ]
    ].to_csv(OUT / "problem2_day_ahead_forecasts.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "problem2_daily_summary.csv", index=False, encoding="utf-8-sig")
    events.to_csv(OUT / "problem2_emergency_events.csv", index=False, encoding="utf-8-sig")
    pd.DataFrame(solver_audit).to_csv(
        OUT / "problem2_information_audit.csv", index=False, encoding="utf-8-sig"
    )
    metrics.to_csv(
        OUT / "problem2_forecast_metrics_by_day_type.csv",
        index=False,
        encoding="utf-8-sig",
    )
    (OUT / "problem2_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    payload = {
        "dates": daily["date"].tolist(),
        "plan_rows": [
            group["plan_kwh"].tolist()
            + [float(group["plan_kwh"].sum()), float(group["plan_cost_yuan"].sum())]
            for _, group in delivered.groupby("date", sort=True)
        ],
        "storage_rows": [
            {
                "date": date,
                "charge": group["charge_kwh"].to_numpy().reshape(6, 24).sum(axis=1).tolist(),
                "discharge": group["discharge_kwh"].to_numpy().reshape(6, 24).sum(axis=1).tolist(),
                "initial": float(group["soc_start_kwh"].iloc[0]),
                "terminal": float(group["soc_end_kwh"].iloc[-1]),
            }
            for date, group in delivered.groupby("date", sort=True)
        ],
        "events": events.to_dict(orient="records"),
    }
    (OUT / "problem2_excel_data.json").write_text(
        json.dumps(payload, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
