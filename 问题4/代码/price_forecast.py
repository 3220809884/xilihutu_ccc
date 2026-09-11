"""Causal price paths, empirical scenarios, and interval uncertainty for Q4.

No weather/market covariates are invented. Model selection is January-only.
The January warm-up itself uses a predeclared lag-7 rule, not its future winner.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

N = 144
HOURS = (0, 6, 12, 18)
FLOOR = 1e-6
COLD_START_PRICE = .6  # explicit neutral prior, NOT a value from attachment 4
CANDIDATES = ("lag1", "lag7", "mean7", "weekly2", "blend_weekly")


def profile(history: np.ndarray, targets: np.ndarray, method: str) -> np.ndarray:
    """history contains ONLY intervals completed by the decision timestamp.

    A lagged target beyond the observed prefix is skipped, including targets
    after midnight. Fallback uses the most recent completed same-clock slot.
    """
    lags = {"lag1": (1,), "lag7": (7,), "mean7": tuple(range(1, 8)), "weekly2": (7, 14), "blend_weekly": (7, 1)}[method]
    weights = (.8, .2) if method == "blend_weekly" else np.ones(len(lags))
    result = np.full(len(targets), COLD_START_PRICE)
    for t, target in enumerate(targets):
        valid = [(target - lag * N, weight) for lag, weight in zip(lags, weights) if 0 <= target - lag * N < len(history)]
        if valid:
            result[t] = sum(history[j] * w for j, w in valid) / sum(w for _, w in valid)
        else:
            j = int(target)
            while j >= len(history):
                j -= N
            if j >= 0:
                result[t] = history[j]
    return result


def january_selection(price: np.ndarray) -> tuple[str, pd.DataFrame]:
    """Use Jan 15--31 as a rolling validation set, freeze choice on Feb 1."""
    jan = price[:31].ravel()
    rows = []
    for method in CANDIDATES:
        errors = []
        for day in range(14, 31):
            start = day * N
            pred = profile(jan[:start], start + np.arange(N), method)
            errors.extend(jan[start:start + N] - pred)
        errors = np.asarray(errors)
        rows.append({"model": method, "validation_start": "2025-01-15", "validation_end": "2025-01-31", "MAE_yuan_per_kwh": float(np.abs(errors).mean()), "RMSE_yuan_per_kwh": float(np.sqrt(np.mean(errors ** 2)))})
    table = pd.DataFrame(rows).sort_values(["MAE_yuan_per_kwh", "model"]).reset_index(drop=True)
    return str(table.iloc[0].model), table


@dataclass
class PricePaths:
    actual: np.ndarray
    raw: np.ndarray
    mean: np.ndarray
    radius: np.ndarray
    residual: np.ndarray
    audit: pd.DataFrame
    selection: pd.DataFrame
    selected_model: str


def build_prices(price: np.ndarray, dates: list[str]) -> PricePaths:
    assert price.shape == (365, N) and np.isfinite(price).all() and (price > 0).all()
    selected, selection = january_selection(price)
    raw = np.zeros((365, 4, N))
    residual = np.full_like(raw, np.nan)
    mean, radius = np.zeros_like(raw), np.zeros_like(raw)
    flat = price.ravel()
    rows = []
    for day, date in enumerate(dates):
        method = "lag7" if day < 31 else selected
        midnight = day * N
        midnight_pred = profile(flat[:midnight], midnight + np.arange(N), method)
        for phase, hour in enumerate(HOURS):
            issue_index = midnight + hour * 6
            history = flat[:issue_index]
            targets = issue_index + np.arange(N)
            base = profile(history, targets, method)
            # Observed recent price bias; only completed intervals of today.
            if hour:
                start = max(0, hour * 6 - 36)
                observed_bias = float(np.mean(price[day, start:hour * 6] - midnight_pred[start:hour * 6]))
                base += .5 * observed_bias * np.exp(-np.arange(1, N + 1) / 36)
            else:
                observed_bias = 0.
            raw[day, phase] = np.maximum(base, FLOOR)
            # Only issue-day j < day windows are completely observed at this hour.
            first_day = max(7, day - 28)
            history_errors = residual[first_day:day, phase]
            if len(history_errors):
                assert np.isfinite(history_errors).all()
                scenarios = np.maximum(raw[day, phase] + history_errors, FLOOR)
                mean[day, phase] = scenarios.mean(axis=0)
                radius[day, phase] = np.quantile(np.abs(scenarios - mean[day, phase]), .9, axis=0)
            else:
                mean[day, phase] = raw[day, phase]
                # Only used during cold start before historical error paths exist.
                radius[day, phase] = 0.
            count = min(N, len(flat) - issue_index)
            residual[day, phase, :count] = flat[issue_index:issue_index + count] - raw[day, phase, :count]
            issue = pd.Timestamp(date) + pd.Timedelta(hours=hour)
            rows.append({"issue_datetime": str(issue), "model": method, "model_selection_known_at": "2025-02-01 00:00:00" if day >= 31 else "predeclared_lag7", "observations_end": str(issue) if issue_index else "", "scenario_count": len(history_errors), "scenario_issue_first_date": dates[first_day] if len(history_errors) else "", "scenario_issue_last_date": dates[day - 1] if len(history_errors) else "", "latest_scenario_target_end": str(issue) if len(history_errors) else "", "scenario_probability": 1 / len(history_errors) if len(history_errors) else 0., "observed_intraday_bias": observed_bias, "interval_quantile": .9})
    return PricePaths(price, raw, mean, radius, residual, pd.DataFrame(rows), selection, selected)


def issued_price(paths: PricePaths, day: int, phase: int, count: int, scale: float=1., mode="forecast", fixed=None) -> np.ndarray:
    if mode == "perfect_price_benchmark":
        first = day * N + HOURS[phase] * 6
        return paths.actual.ravel()[first:first + count].copy()
    if mode == "fixed_price_planning":
        slots = (HOURS[phase] * 6 + np.arange(count)) % N
        return fixed[slots].copy()
    return paths.mean[day, phase, :count] + scale * paths.radius[day, phase, :count]


def diagnostics(paths: PricePaths) -> pd.DataFrame:
    rows = []
    flat = paths.actual.ravel()
    for phase, hour in enumerate(HOURS):
        for lo, hi in [(0, 36), (36, 72), (72, 108), (108, 144)]:
            errors, hits, widths, biases = [], [], [], []
            for day in range(31, 365):
                first = day * N + hour * 6
                stop = min(hi, len(flat) - first)
                if stop <= lo:
                    continue
                actual = flat[first + lo:first + stop]
                center = paths.mean[day, phase, lo:stop]
                rad = paths.radius[day, phase, lo:stop]
                err = actual - center
                errors.extend(err); hits.extend(np.abs(err) <= rad); widths.extend(2 * rad); biases.extend(actual - paths.raw[day, phase, lo:stop])
            e = np.asarray(errors)
            rows.append({"issue_hour": hour, "lead_band": f"{lo//6}-{hi//6}h", "n": len(e), "MAE_yuan_per_kwh": float(np.abs(e).mean()), "RMSE_yuan_per_kwh": float(np.sqrt(np.mean(e**2))), "bias_actual_minus_mean": float(e.mean()), "interval_coverage": float(np.mean(hits)), "interval_average_width": float(np.mean(widths))})
    return pd.DataFrame(rows)
