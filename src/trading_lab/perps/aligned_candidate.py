"""Experimental 1h EMA cross gated by the last completed 4h EMA trend.

Fixed illustrative rules, not optimized or validated for trading. No live orders.
The 4h close is available only after its four constituent hourly bars close.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import talib

from trading_lab.perps.compare import attach_ema


def attach_aligned_ema(h1: pd.DataFrame) -> pd.DataFrame:
    out = attach_ema(h1)
    grouped = h1["close"].resample("4h", label="right", closed="left", origin="epoch")
    close = grouped.last().where(grouped.count() == 4)
    fast = pd.Series(talib.EMA(close.to_numpy(dtype=float), timeperiod=20), index=close.index)
    slow = pd.Series(talib.EMA(close.to_numpy(dtype=float), timeperiod=50), index=close.index)
    # Hourly signal bar at t is known at t+1h. A 4h bar ending exactly then
    # is also completed and can be used for the next-hour open decision.
    close_times = out.index + pd.Timedelta(hours=1)
    f = fast.reindex(close_times, method="ffill").to_numpy()
    s = slow.reindex(close_times, method="ffill").to_numpy()
    out["trend_4h_fast"] = f
    out["trend_4h_slow"] = s
    aligned = ((out["signal"] == 1) & (f > s)) | ((out["signal"] == -1) & (f < s))
    out["signal"] = np.where(aligned, out["signal"], 0)
    return out
