from __future__ import annotations

import pandas as pd


def sma(close: pd.Series, window: int) -> pd.Series:
    if window < 1:
        raise ValueError("sma_window 必须 >= 1")
    return close.astype(float).rolling(window=window, min_periods=window).mean()


def long_cash_signal(close: pd.Series, window: int) -> pd.Series:
    """1 = 希望下一时段持有多头，0 = 希望下一时段持有现金。

    只用已经完成的收盘价。信号出现在 bar T，执行必须等到该标的的下一根已完成开盘。
    SMA 在 T 日只用到 close[T]，不用未来数据。
    """
    ma = sma(close, window)
    raw = (close > ma).astype("int64")
    raw = raw.where(ma.notna(), other=pd.NA)
    return raw
