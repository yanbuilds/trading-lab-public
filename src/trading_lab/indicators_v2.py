"""Pure-NumPy/Pandas causal technical indicators with optional TA-Lib fast path.

Provides deterministic closed-bar indicators (EMA, SMA, Wilder ATR, Wilder RSI,
Wilder ADX/+DI/-DI, Bollinger Bands, Donchian Channels, and Chandelier Trailing
Stops) without requiring compiled C extensions (`TA-Lib`), while matching
standard Wilder / TA-Lib conventions closely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float)


def sma(values, period: int) -> np.ndarray:
    """Simple moving average with NaN warmup."""
    arr = _as_float_array(values)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if period <= 0 or n < period:
        return out
    s = pd.Series(arr, dtype=float).rolling(period, min_periods=period).mean()
    return s.to_numpy(dtype=float)


def ema(values, period: int) -> np.ndarray:
    """SMA-seeded exponential moving average (matches TA-Lib default EMA)."""
    arr = _as_float_array(values)
    n = len(arr)
    out = np.full(n, np.nan, dtype=float)
    if period <= 0 or n < period:
        return out
    valid_idx = np.flatnonzero(np.isfinite(arr))
    if len(valid_idx) < period:
        return out
    start = int(valid_idx[0])
    if n - start < period:
        return out
    window = arr[start : start + period]
    if not np.isfinite(window).all():
        return out
    seed_pos = start + period - 1
    out[seed_pos] = float(np.mean(window))
    alpha = 2.0 / (period + 1.0)
    for i in range(seed_pos + 1, n):
        v = arr[i]
        if not np.isfinite(v) or not np.isfinite(out[i - 1]):
            out[i] = np.nan
        else:
            out[i] = alpha * float(v) + (1.0 - alpha) * out[i - 1]
    return out


def true_range(high, low, close) -> np.ndarray:
    """True Range series."""
    h = _as_float_array(high)
    l = _as_float_array(low)
    c = _as_float_array(close)
    n = len(c)
    tr = np.full(n, np.nan, dtype=float)
    if n == 0:
        return tr
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        tr[i] = max(
            h[i] - l[i],
            abs(h[i] - c[i - 1]),
            abs(l[i] - c[i - 1]),
        )
    return tr


def atr(high, low, close, period: int = 14) -> np.ndarray:
    """Wilder RMA Average True Range (matches TA-Lib ATR: first value at index `period`)."""
    h = _as_float_array(high)
    l = _as_float_array(low)
    c = _as_float_array(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=float)
    if period <= 0 or n <= period:
        return out
    tr = true_range(h, l, c)
    seed_slice = tr[1 : period + 1]
    if not np.isfinite(seed_slice).all():
        return out
    out[period] = float(np.mean(seed_slice))
    for i in range(period + 1, n):
        out[i] = (out[i - 1] * (period - 1) + tr[i]) / period
    return out


def rsi(close, period: int = 14) -> np.ndarray:
    """Wilder Relative Strength Index in [0, 100]."""
    c = _as_float_array(close)
    n = len(c)
    out = np.full(n, np.nan, dtype=float)
    if period <= 0 or n <= period:
        return out
    deltas = np.diff(c)
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)
    avg_gain = float(np.mean(gains[:period]))
    avg_loss = float(np.mean(losses[:period]))
    if avg_gain + avg_loss > 0:
        out[period] = 100.0 * avg_gain / (avg_gain + avg_loss)
    else:
        out[period] = 50.0
    for i in range(period, len(deltas)):
        avg_gain = (avg_gain * (period - 1) + float(gains[i])) / period
        avg_loss = (avg_loss * (period - 1) + float(losses[i])) / period
        if avg_gain + avg_loss > 0:
            out[i + 1] = 100.0 * avg_gain / (avg_gain + avg_loss)
        else:
            out[i + 1] = 50.0
    return out


def adx_di(high, low, close, period: int = 14) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Wilder ADX, +DI, and -DI (matches Wilder 1978 / TA-Lib conventions)."""
    h = _as_float_array(high)
    l = _as_float_array(low)
    c = _as_float_array(close)
    n = len(c)
    adx_out = np.full(n, np.nan, dtype=float)
    plus_di = np.full(n, np.nan, dtype=float)
    minus_di = np.full(n, np.nan, dtype=float)
    if period <= 0 or n < 2 * period:
        return adx_out, plus_di, minus_di

    tr = np.zeros(n, dtype=float)
    pdm = np.zeros(n, dtype=float)
    mdm = np.zeros(n, dtype=float)
    for i in range(1, n):
        up_move = h[i] - h[i - 1]
        down_move = l[i - 1] - l[i]
        pdm[i] = up_move if (up_move > down_move and up_move > 0.0) else 0.0
        mdm[i] = down_move if (down_move > up_move and down_move > 0.0) else 0.0
        tr[i] = max(h[i] - l[i], abs(h[i] - c[i - 1]), abs(l[i] - c[i - 1]))

    tr_s = float(np.sum(tr[1 : period + 1]))
    pdm_s = float(np.sum(pdm[1 : period + 1]))
    mdm_s = float(np.sum(mdm[1 : period + 1]))

    dx = np.full(n, np.nan, dtype=float)
    if tr_s > 0:
        pdi = 100.0 * pdm_s / tr_s
        mdi = 100.0 * mdm_s / tr_s
        plus_di[period] = pdi
        minus_di[period] = mdi
        denom = pdi + mdi
        dx[period] = 100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0

    for i in range(period + 1, n):
        tr_s = tr_s - (tr_s / period) + tr[i]
        pdm_s = pdm_s - (pdm_s / period) + pdm[i]
        mdm_s = mdm_s - (mdm_s / period) + mdm[i]
        if tr_s > 0:
            pdi = 100.0 * pdm_s / tr_s
            mdi = 100.0 * mdm_s / tr_s
            plus_di[i] = pdi
            minus_di[i] = mdi
            denom = pdi + mdi
            dx[i] = 100.0 * abs(pdi - mdi) / denom if denom > 0 else 0.0
        else:
            plus_di[i] = 0.0
            minus_di[i] = 0.0
            dx[i] = 0.0

    adx_start = 2 * period - 1
    dx_seed = dx[period : adx_start + 1]
    if len(dx_seed) == period and np.isfinite(dx_seed).all():
        adx_out[adx_start] = float(np.mean(dx_seed))
        for i in range(adx_start + 1, n):
            adx_out[i] = (adx_out[i - 1] * (period - 1) + dx[i]) / period
    return adx_out, plus_di, minus_di


def bollinger_bands(
    close, period: int = 20, std_mult: float = 2.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Bollinger Bands (mid, upper, lower, bandwidth, pct_b) with population std (ddof=0)."""
    s = pd.Series(_as_float_array(close), dtype=float)
    mid = s.rolling(period, min_periods=period).mean()
    std = s.rolling(period, min_periods=period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    span = upper - lower
    bandwidth = np.where((mid > 0) & np.isfinite(span), span / mid, np.nan)
    pct_b = np.where((span > 0) & np.isfinite(span), (s - lower) / span, np.nan)
    return (
        mid.to_numpy(dtype=float),
        upper.to_numpy(dtype=float),
        lower.to_numpy(dtype=float),
        bandwidth,
        pct_b,
    )


def donchian_channels(
    high, low, period: int = 20, *, shift_bars: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Causal Donchian upper/lower channels shifted by `shift_bars` (default 1 = prior bars only)."""
    h = pd.Series(_as_float_array(high), dtype=float)
    l = pd.Series(_as_float_array(low), dtype=float)
    if shift_bars > 0:
        h = h.shift(shift_bars)
        l = l.shift(shift_bars)
    up = h.rolling(period, min_periods=period).max().to_numpy(dtype=float)
    dn = l.rolling(period, min_periods=period).min().to_numpy(dtype=float)
    return up, dn


def chandelier_stops(
    high, low, atr_values, period: int = 22, mult: float = 2.5, *, shift_bars: int = 1
) -> tuple[np.ndarray, np.ndarray]:
    """Causal Chandelier trailing stop levels for longs and shorts."""
    h = pd.Series(_as_float_array(high), dtype=float)
    l = pd.Series(_as_float_array(low), dtype=float)
    a = pd.Series(_as_float_array(atr_values), dtype=float)
    if shift_bars > 0:
        h = h.shift(shift_bars)
        l = l.shift(shift_bars)
        a = a.shift(shift_bars)
    hh = h.rolling(period, min_periods=period).max()
    ll = l.rolling(period, min_periods=period).min()
    long_stop = (hh - mult * a).to_numpy(dtype=float)
    short_stop = (ll + mult * a).to_numpy(dtype=float)
    return long_stop, short_stop
