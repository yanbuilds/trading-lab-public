"""Deterministic closed-bar market-state labels, not latent regimes or agent opinions.

Parameters are frozen in config/piia_regime_mean_reversion_v1.yaml and are not
optimized here. A label is an indicator state, not a forecast or order.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
import hashlib
import json

import numpy as np
import pandas as pd
import talib
import yaml

MANIFEST_PATH = Path(__file__).resolve().parents[3] / "config" / "piia_regime_mean_reversion_v1.yaml"

# Defaults match the frozen manifest and exist only so tests can construct labels
# without rereading the file mid-assertion. Do not change these independently.
ADX_PERIOD = 14
ADX_RANGE_BELOW = 20
ADX_TREND_AT_OR_ABOVE = 25
ATR_PERIOD = 14
ATR_MEAN_PERIOD = 50
EXTREME_VOL_ATR_MULTIPLE = 2.0
HTF_FAST = 20
HTF_SLOW = 50
H1_FAST = 20
H1_SLOW = 50

NO_TRADE = frozenset({"insufficient_data", "extreme_vol", "uncertain"})
FROZEN_MANIFEST_SHA256 = "08ef68082d1351041e76bebe00d75a7adc11e40d4b2050d4414ed29908900908"


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    data = yaml.safe_load((path or MANIFEST_PATH).read_text())
    if not isinstance(data, dict):
        raise ValueError("invalid frozen manifest")
    required = {
        "hypothesis_id": "range_bollinger_mean_reversion_v1", "hypothesis_optimized": False,
        "held_out": False, "already_seen_windows": True, "order_permissions": False,
        "deterministic_experiment": True, "not_a_piia_agent": True,
        "capital_primary_usdt": 100, "capital_optional_feasibility_usdt": 30,
        "never_mix_capital_books": True, "leverage": 2, "risk_fraction": 0.01,
        "cash_floor": 0.20, "max_margin_fraction": 0.50, "max_open_positions": 1,
        "slippage_each_side": 0.0005, "stop_atr_multiple": 2.0, "time_stop_hours": 24,
        "no_martingale": True, "no_grid": True, "no_dca": True, "no_pyramiding": True,
    }
    regime = {"adx_period": ADX_PERIOD, "adx_range_below": ADX_RANGE_BELOW,
              "adx_trend_at_or_above": ADX_TREND_AT_OR_ABOVE, "atr_period": ATR_PERIOD,
              "atr_mean_period": ATR_MEAN_PERIOD, "extreme_vol_atr_multiple": EXTREME_VOL_ATR_MULTIPLE,
              "htf": "completed_4h_ema20_vs_ema50", "partial_4h_excluded": True,
              "no_trade_regimes": ["insufficient_data", "extreme_vol", "uncertain"]}
    reversion = {"bollinger_period": 20, "bollinger_std": 2.0, "eligible_regime": "range"}
    for actual, expected in ((data, required), (data.get("regime"), regime), (data.get("mean_reversion"), reversion)):
        if not isinstance(actual, dict):
            raise ValueError("missing frozen manifest section")
        for key, value in expected.items():
            observed = actual.get(key)
            if observed != value or (isinstance(value, bool) != isinstance(observed, bool)):
                raise ValueError(f"frozen manifest mismatch: {key}")
    canonical = json.dumps(data, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    if hashlib.sha256(canonical).hexdigest() != FROZEN_MANIFEST_SHA256:
        raise ValueError("frozen manifest mismatch: canonical content changed")
    return data


def completed_4h_emas(h1: pd.DataFrame) -> pd.DataFrame:
    """Map the last fully completed 4h EMA20/50 onto each 1h bar.

    A 4h bin is included only when it contains four hourly closes (no partial
    current bar). Hourly bars are indexed by open time, as in the existing
    Bitget cache. The 4h close of [t, t+4h) is known at t+4h, which coincides
    with the decision time of the last hourly bar in that bin.
    """
    if h1.empty:
        out = h1.copy()
        out["trend_4h_fast"] = pd.Series(dtype=float)
        out["trend_4h_slow"] = pd.Series(dtype=float)
        return out
    grouped = h1["close"].resample("4h", label="right", closed="left", origin="epoch")
    close = grouped.last().where(grouped.count() == 4)
    fast = pd.Series(talib.EMA(close.to_numpy(dtype=float), timeperiod=HTF_FAST), index=close.index)
    slow = pd.Series(talib.EMA(close.to_numpy(dtype=float), timeperiod=HTF_SLOW), index=close.index)
    close_times = h1.index + pd.Timedelta(hours=1)
    out = h1.copy()
    out["trend_4h_fast"] = fast.reindex(close_times, method="ffill").to_numpy()
    out["trend_4h_slow"] = slow.reindex(close_times, method="ffill").to_numpy()
    return out


def _finite(value) -> bool:
    return value is not None and pd.notna(value) and np.isfinite(value)


def classify_bar(*, adx, atr, atr_mean, ema_fast, ema_slow, htf_fast, htf_slow) -> tuple[str, str]:
    """Single-bar label using the frozen order in the manifest."""
    needed = (adx, atr, atr_mean, ema_fast, ema_slow, htf_fast, htf_slow)
    if not all(_finite(x) for x in needed):
        return "insufficient_data", "required_indicator_undefined"
    if atr_mean <= 0:
        return "insufficient_data", "non_positive_atr_mean"
    if atr >= EXTREME_VOL_ATR_MULTIPLE * atr_mean:
        return "extreme_vol", "atr_at_least_twice_its_50_period_average"
    if adx < ADX_RANGE_BELOW:
        return "range", "adx_below_20"
    htf_dir = 1 if htf_fast > htf_slow else -1 if htf_fast < htf_slow else 0
    h1_dir = 1 if ema_fast > ema_slow else -1 if ema_fast < ema_slow else 0
    if adx >= ADX_TREND_AT_OR_ABOVE and htf_dir != 0 and htf_dir == h1_dir:
        return "trend", "adx_at_least_25_and_1h_agrees_with_completed_4h"
    if ADX_RANGE_BELOW <= adx < ADX_TREND_AT_OR_ABOVE:
        return "uncertain", "adx_gray_zone_20_to_25"
    return "uncertain", "higher_timeframe_or_hourly_trend_conflict"


def classify_regime(h1: pd.DataFrame) -> pd.DataFrame:
    """Attach point-in-time regime labels using only completed bars."""
    out = completed_4h_emas(h1)
    high, low, close = (out[c].to_numpy(dtype=float) for c in ("high", "low", "close"))
    out["adx14"] = talib.ADX(high, low, close, timeperiod=ADX_PERIOD)
    out["atr14"] = talib.ATR(high, low, close, timeperiod=ATR_PERIOD)
    out["ema20"] = talib.EMA(close, timeperiod=H1_FAST)
    out["ema50"] = talib.EMA(close, timeperiod=H1_SLOW)
    out["atr14_mean50"] = out["atr14"].rolling(ATR_MEAN_PERIOD, min_periods=ATR_MEAN_PERIOD).mean()
    labels, reasons = [], []
    for row in out.itertuples():
        label, reason = classify_bar(
            adx=row.adx14, atr=row.atr14, atr_mean=row.atr14_mean50,
            ema_fast=row.ema20, ema_slow=row.ema50,
            htf_fast=row.trend_4h_fast, htf_slow=row.trend_4h_slow,
        )
        labels.append(label)
        reasons.append(reason)
    out["regime"] = labels
    out["regime_reason"] = reasons
    out["regime_is_tradable"] = out["regime"].eq("range") | out["regime"].eq("trend")
    out["kind"] = "deterministic_closed_bar_classifier_not_agent"
    return out


def last_regime_evidence(h1: pd.DataFrame) -> dict[str, Any]:
    """Last closed-bar label for optional research metadata. Not an agent opinion."""
    classified = classify_regime(h1)
    if classified.empty:
        return {"kind": "deterministic_closed_bar_classifier_not_agent",
                "regime": "insufficient_data", "order_permissions": False}
    last = classified.iloc[-1]
    return {
        "kind": "deterministic_closed_bar_classifier_not_agent",
        "regime": str(last["regime"]),
        "reason": str(last["regime_reason"]),
        "adx14": None if not _finite(last["adx14"]) else float(last["adx14"]),
        "atr14_over_mean50": (
            None if not (_finite(last["atr14"]) and _finite(last["atr14_mean50"]) and last["atr14_mean50"] > 0)
            else float(last["atr14"] / last["atr14_mean50"])
        ),
        "completed_4h_ema20_vs_ema50": (
            None if not (_finite(last["trend_4h_fast"]) and _finite(last["trend_4h_slow"]))
            else "up" if last["trend_4h_fast"] > last["trend_4h_slow"]
            else "down" if last["trend_4h_fast"] < last["trend_4h_slow"] else "equal"
        ),
        "bar_open_utc": str(classified.index[-1]),
        "order_permissions": False,
        "limitations": [
            "Indicator state only; not a proven latent regime or forecast.",
            "Parameters are frozen and not sample-optimized.",
        ],
    }
