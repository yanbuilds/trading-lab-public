"""One frozen range-gated Bollinger mean-reversion candidate. Not a proven edge.

Uses the same next-open fill and 2*ATR14 stop budget as the existing trend book.
Trades only when the deterministic classifier labels the closed bar as range,
and never fades a completed 4h EMA20/50 slope. No martingale, grid, or DCA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from trading_lab.perps.regime import classify_regime

# Frozen to match config/piia_regime_mean_reversion_v1.yaml. Do not retune here.
BOLLINGER_PERIOD = 20
BOLLINGER_STD = 2.0


def attach_range_bollinger(h1: pd.DataFrame) -> pd.DataFrame:
    """Prepare signals for simulate_book. Signal at bar t is known at t+1h."""
    out = classify_regime(h1)
    close = out["close"].to_numpy(dtype=float)
    mid, upper, lower = talib.BBANDS(
        close, timeperiod=BOLLINGER_PERIOD, nbdevup=BOLLINGER_STD,
        nbdevdn=BOLLINGER_STD, matype=0,
    )
    out["bb_mid"] = mid
    out["bb_upper"] = upper
    out["bb_lower"] = lower
    out["atr"] = out["atr14"]
    out["fast"] = out["ema20"]
    out["slow"] = out["ema50"]
    htf_up = out["trend_4h_fast"] > out["trend_4h_slow"]
    htf_down = out["trend_4h_fast"] < out["trend_4h_slow"]
    in_range = out["regime"].eq("range")
    # Trend guard: do not buy into a completed 4h down slope, or sell into an up slope.
    long_entry = in_range & (out["close"] < out["bb_lower"]) & htf_up
    short_entry = in_range & (out["close"] > out["bb_upper"]) & htf_down
    out["signal"] = np.where(long_entry, 1, np.where(short_entry, -1, 0))
    mid_ok = out["bb_mid"].notna()
    out["exit_long"] = mid_ok & (out["close"] >= out["bb_mid"])
    out["exit_short"] = mid_ok & (out["close"] <= out["bb_mid"])
    out["trail_long"] = float("nan")
    out["trail_short"] = float("nan")
    return out
