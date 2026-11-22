"""Lightweight statistical forecast benchmark. Not a neural model and not calibrated.

Used when Chronos or Kronos weights are not installed. This does not download
models and is not a substitute for a validated predictive system.
"""
from __future__ import annotations

import math
from typing import Any


HORIZON_HOURS = 24
LOOKBACK_HOURS = 168  # one week of hourly closes; frozen, not optimized


def neural_forecast_status() -> dict[str, Any]:
    """Report whether optional local forecast libraries import, without loading weights."""
    status = {
        "kind": "optional_neural_forecast_status",
        "chronos": "unavailable",
        "kronos": "unavailable",
        "reasons": {},
        "weights_loaded": False,
        "order_permissions": False,
        "note": "Libraries alone are not a runnable forecast. Weights are not downloaded here.",
    }
    try:
        import chronos  # noqa: F401
        status["chronos"] = "library_present_weights_not_loaded"
    except Exception as exc:  # ImportError or environment issues
        status["reasons"]["chronos"] = type(exc).__name__
    try:
        import torch  # noqa: F401
        import model as kronos_model  # type: ignore  # noqa: F401
        status["kronos"] = "library_present_weights_not_loaded"
    except Exception as exc:
        status["reasons"]["kronos"] = type(exc).__name__
    return status


def random_walk_drift_forecast(closes: list[float], *, horizon: int = HORIZON_HOURS,
                               lookback: int = LOOKBACK_HOURS) -> dict[str, Any]:
    """Point and interval path from recent log-return mean and residual scale.

    Intervals assume independent Gaussian log returns and are not empirically
    calibrated. Returns unavailable when history is insufficient or invalid.
    """
    base = {
        "kind": "statistical_benchmark_not_neural_model",
        "horizon_hours": horizon,
        "lookback_hours": lookback,
        "calibration_status": "not_calibrated",
        "order_permissions": False,
        "available": False,
    }
    if (type(horizon) is not int or horizon < 1 or type(lookback) is not int or lookback < 2):
        return {**base, "reason": "invalid_horizon_or_lookback"}
    if not isinstance(closes, list) or len(closes) < lookback + 1:
        return {**base, "reason": "insufficient_closed_history"}
    window = closes[-(lookback + 1):]
    if any(isinstance(x, bool) for x in window):
        return {**base, "reason": "non_numeric_close"}
    try:
        values = [float(x) for x in window]
    except (TypeError, ValueError, OverflowError):
        return {**base, "reason": "non_numeric_close"}
    if any(not math.isfinite(x) or x <= 0 for x in values):
        return {**base, "reason": "non_positive_or_non_finite_close"}
    try:
        returns = [math.log(values[i]) - math.log(values[i - 1]) for i in range(1, len(values))]
        n = len(returns)
        drift = sum(returns) / n
        variance = sum((r - drift) ** 2 for r in returns) / (n - 1)
        sigma = math.sqrt(variance) if variance > 0 else 0.0
        last = values[-1]
        point = last * math.exp(drift * horizon)
        half = 1.96 * sigma * math.sqrt(horizon)
        low, high = last * math.exp(drift * horizon - half), last * math.exp(drift * horizon + half)
        if not all(math.isfinite(x) and x > 0 for x in (point, low, high)):
            raise ValueError("nonfinite forecast")
    except (OverflowError, ValueError):
        return {**base, "reason": "forecast_numerically_unavailable"}
    return {
        **base,
        "available": True,
        "last_close": last,
        "drift_per_hour": drift,
        "residual_sigma": sigma,
        "point_horizon": point,
        "interval_low": low,
        "interval_high": high,
        "interval_note": "Approximate 95 percent band under an uncalibrated log-return assumption.",
        "limitations": [
            "This is a random-walk-with-drift benchmark, not Chronos or Kronos.",
            "Intervals are not a calibrated prediction interval or trading signal.",
        ],
    }
