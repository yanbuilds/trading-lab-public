"""Multi-Regime Adaptive Strategy v2 (`regime_adaptive_v2`) — Pure Python, Causal.

Addresses the structural weakness of v1 (`range_bollinger_mean_reversion_v1`),
which sat flat during directional trends and faded range edges without momentum
exhaustion confirmation:

1. `extreme_vol` (`ATR14 >= 2.2 * SMA50(ATR14)`): Stand aside (capital protection).
2. `trend_up` / `trend_down` (`ADX14 >= 22`, 1h EMA20/50 aligned with completed
   4h EMA20/50, and Wilder +DI/-DI aligned):
   - Enters on 48h causal Donchian breakout OR strong-trend (`ADX >= 25`) 1h EMA20
     pullback resumption.
   - Initial stop `2.0 * ATR14`, Chandelier trailing stop (`22h, 2.5 * ATR14`),
     plus breakeven stop lock once price reaches `+1.5R`.
3. `range` (`ADX14 < 20` and non-elevated volatility):
   - Requires Bollinger Band excursion (`close < lower` / `close > upper`) AND
     Wilder RSI14 oversold/overbought (`< 35` / `> 65`) AND completed 4h EMA slope
     guard (never buy against 4h downtrend or short against 4h uptrend).
   - Exits at `SMA20` middle band or `18h` time stop.
4. Multi-asset contention uses deterministic signal strength (`ADX` / band-RSI Z)
   rather than arbitrary alphabetical ticker ordering.
"""
from __future__ import annotations

from collections import Counter
import math
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.indicators_v2 import (
    adx_di,
    atr,
    bollinger_bands,
    chandelier_stops,
    donchian_channels,
    ema,
    rsi,
    sma,
)
from trading_lab.perps.risk import exit_pnl, size_order, stop_fill
from trading_lab.perps.risk_v2 import load_policy_v2, resolve_spec


def _completed_4h_emas_pure(h1: pd.DataFrame, fast_period: int = 20, slow_period: int = 50) -> tuple[np.ndarray, np.ndarray]:
    """Causal completed 4h EMA20/50 mapped onto 1h bars without TA-Lib."""
    if h1.empty:
        empty = np.array([], dtype=float)
        return empty, empty
    grouped = h1["close"].resample("4h", label="right", closed="left", origin="epoch")
    c4 = grouped.last().where(grouped.count() == 4)
    vals = c4.to_numpy(dtype=float)
    # Compute EMA over non-NaN completed 4h bars, preserving alignment
    valid_mask = np.isfinite(vals)
    fast_arr = np.full(len(vals), np.nan, dtype=float)
    slow_arr = np.full(len(vals), np.nan, dtype=float)
    if valid_mask.sum() >= slow_period:
        compact = vals[valid_mask]
        fast_compact = ema(compact, fast_period)
        slow_compact = ema(compact, slow_period)
        fast_arr[valid_mask] = fast_compact
        slow_arr[valid_mask] = slow_compact
    fast_s = pd.Series(fast_arr, index=c4.index)
    slow_s = pd.Series(slow_arr, index=c4.index)
    close_times = h1.index + pd.Timedelta(hours=1)
    return (
        fast_s.reindex(close_times, method="ffill").to_numpy(dtype=float),
        slow_s.reindex(close_times, method="ffill").to_numpy(dtype=float),
    )


def attach_ema_v2(h1: pd.DataFrame) -> pd.DataFrame:
    """Pure-Python 1h EMA20/50 baseline (compatible with `simulate_book` without TA-Lib)."""
    out = h1.copy()
    close = out["close"].to_numpy(dtype=float)
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    out["fast"] = ema(close, 20)
    out["slow"] = ema(close, 50)
    out["atr"] = atr(high, low, close, 14)
    sig = np.zeros(len(out), dtype=int)
    prev_f = pd.Series(out["fast"], index=out.index).shift(1)
    prev_s = pd.Series(out["slow"], index=out.index).shift(1)
    sig[(out["fast"] > out["slow"]) & (prev_f <= prev_s)] = 1
    sig[(out["fast"] < out["slow"]) & (prev_f >= prev_s)] = -1
    out["signal"] = sig
    out["exit_long"] = out["fast"] < out["slow"]
    out["exit_short"] = out["fast"] > out["slow"]
    out["trail_long"] = np.nan
    out["trail_short"] = np.nan
    out["signal_strength"] = np.where(sig != 0, 1.0, 0.0)
    out["leg"] = np.where(sig != 0, "ema_cross", "none")
    return out


def attach_adaptive_v2(h1: pd.DataFrame, policy: dict[str, Any] | None = None) -> pd.DataFrame:
    """Attach causal multi-regime v2 indicators, signals, trailing stops & priority scores."""
    cfg = policy or load_policy_v2()
    reg_cfg = cfg["regime"]
    tr_cfg = cfg["trend_leg"]
    rg_cfg = cfg["range_leg"]

    out = h1.copy()
    high = out["high"].to_numpy(dtype=float)
    low = out["low"].to_numpy(dtype=float)
    close = out["close"].to_numpy(dtype=float)
    open_arr = out["open"].to_numpy(dtype=float)

    htf_fast, htf_slow = _completed_4h_emas_pure(
        out, int(reg_cfg["htf_fast_ema"]), int(reg_cfg["htf_slow_ema"])
    )
    out["trend_4h_fast"] = htf_fast
    out["trend_4h_slow"] = htf_slow

    out["ema20"] = ema(close, int(reg_cfg["h1_fast_ema"]))
    out["ema50"] = ema(close, int(reg_cfg["h1_slow_ema"]))
    out["atr14"] = atr(high, low, close, int(reg_cfg["atr_period"]))
    out["atr14_mean50"] = sma(out["atr14"].to_numpy(dtype=float), int(reg_cfg["atr_mean_period"]))
    adx_vals, pdi_vals, mdi_vals = adx_di(high, low, close, int(reg_cfg["adx_period"]))
    out["adx14"] = adx_vals
    out["plus_di14"] = pdi_vals
    out["minus_di14"] = mdi_vals
    out["rsi14"] = rsi(close, int(rg_cfg["rsi_period"]))

    bb_mid, bb_upper, bb_lower, bb_bw, bb_pct = bollinger_bands(
        close, int(rg_cfg["bollinger_period"]), float(rg_cfg["bollinger_std"])
    )
    out["bb_mid"] = bb_mid
    out["bb_upper"] = bb_upper
    out["bb_lower"] = bb_lower
    out["bb_bandwidth"] = bb_bw
    out["bb_pct_b"] = bb_pct

    don_hi, don_lo = donchian_channels(high, low, int(tr_cfg["donchian_entry_bars"]), shift_bars=1)
    out["donchian_high"] = don_hi
    out["donchian_low"] = don_lo

    ch_long, ch_short = chandelier_stops(
        high,
        low,
        out["atr14"].to_numpy(dtype=float),
        int(tr_cfg["chandelier_period"]),
        float(tr_cfg["chandelier_atr_multiple"]),
        shift_bars=1,
    )
    out["chandelier_long"] = ch_long
    out["chandelier_short"] = ch_short

    # Deterministic regime classification per closed 1h bar
    adx_range = float(reg_cfg["adx_range_below"])
    adx_trend = float(reg_cfg["adx_trend_at_or_above"])
    ext_mult = float(reg_cfg["extreme_vol_atr_multiple"])

    regimes: list[str] = []
    reasons: list[str] = []
    for i in range(len(out)):
        needed = (
            adx_vals[i],
            out["atr14"].iat[i],
            out["atr14_mean50"].iat[i],
            out["ema20"].iat[i],
            out["ema50"].iat[i],
            htf_fast[i],
            htf_slow[i],
        )
        if not all(np.isfinite(x) for x in needed) or out["atr14_mean50"].iat[i] <= 0:
            regimes.append("insufficient_data")
            reasons.append("warmup_incomplete")
            continue
        atr_ratio = out["atr14"].iat[i] / out["atr14_mean50"].iat[i]
        if atr_ratio >= ext_mult:
            regimes.append("extreme_vol")
            reasons.append("atr_spike_stand_aside")
            continue
        htf_dir = 1 if htf_fast[i] > htf_slow[i] else (-1 if htf_fast[i] < htf_slow[i] else 0)
        h1_dir = 1 if out["ema20"].iat[i] > out["ema50"].iat[i] else (-1 if out["ema20"].iat[i] < out["ema50"].iat[i] else 0)
        if adx_vals[i] >= adx_trend and htf_dir == 1 and h1_dir == 1 and pdi_vals[i] > mdi_vals[i]:
            regimes.append("trend_up")
            reasons.append("aligned_1h_4h_uptrend_with_adx_di")
        elif adx_vals[i] >= adx_trend and htf_dir == -1 and h1_dir == -1 and mdi_vals[i] > pdi_vals[i]:
            regimes.append("trend_down")
            reasons.append("aligned_1h_4h_downtrend_with_adx_di")
        elif adx_vals[i] < adx_range:
            regimes.append("range")
            reasons.append("low_adx_range")
        else:
            regimes.append("uncertain")
            reasons.append("transition_or_timeframe_conflict")

    out["regime"] = regimes
    out["regime_reason"] = reasons

    # Build Trend & Range entry signals
    pullback_adx = float(tr_cfg["pullback_adx_min"])
    pb_rsi_long_max = float(tr_cfg["pullback_rsi_long_max"])
    pb_rsi_short_min = float(tr_cfg["pullback_rsi_short_min"])
    rg_rsi_os = float(rg_cfg["rsi_oversold"])
    rg_rsi_ob = float(rg_cfg["rsi_overbought"])
    rg_max_atr_ratio = float(rg_cfg["max_atr_over_mean"])

    signals = np.zeros(len(out), dtype=int)
    legs = ["none"] * len(out)
    strengths = np.zeros(len(out), dtype=float)
    stop_mults = np.full(len(out), float(tr_cfg["stop_atr_multiple"]), dtype=float)

    for i in range(1, len(out)):
        reg = regimes[i]
        a14 = out["atr14"].iat[i]
        a_mean = out["atr14_mean50"].iat[i]
        if not (np.isfinite(a14) and a14 > 0 and np.isfinite(a_mean) and a_mean > 0):
            continue

        if reg == "trend_up":
            # (A) Breakout above prior 48h high, or (B) Pullback touch of EMA20 with bullish close
            brk = np.isfinite(don_hi[i]) and close[i] > don_hi[i] and close[i - 1] <= don_hi[i]
            pb = (
                adx_vals[i] >= pullback_adx
                and low[i] <= out["ema20"].iat[i]
                and close[i] > out["ema20"].iat[i]
                and close[i] > open_arr[i]
                and np.isfinite(out["rsi14"].iat[i])
                and out["rsi14"].iat[i] <= pb_rsi_long_max
            )
            if brk or pb:
                signals[i] = 1
                legs[i] = "trend_breakout" if brk else "trend_pullback"
                strengths[i] = float(adx_vals[i]) / 25.0 + (0.3 if brk else 0.1)
                stop_mults[i] = float(tr_cfg["stop_atr_multiple"])

        elif reg == "trend_down":
            brk = np.isfinite(don_lo[i]) and close[i] < don_lo[i] and close[i - 1] >= don_lo[i]
            pb = (
                adx_vals[i] >= pullback_adx
                and high[i] >= out["ema20"].iat[i]
                and close[i] < out["ema20"].iat[i]
                and close[i] < open_arr[i]
                and np.isfinite(out["rsi14"].iat[i])
                and out["rsi14"].iat[i] >= pb_rsi_short_min
            )
            if brk or pb:
                signals[i] = -1
                legs[i] = "trend_breakout" if brk else "trend_pullback"
                strengths[i] = float(adx_vals[i]) / 25.0 + (0.3 if brk else 0.1)
                stop_mults[i] = float(tr_cfg["stop_atr_multiple"])

        elif reg == "range" and (a14 / a_mean) <= rg_max_atr_ratio:
            r_val = out["rsi14"].iat[i]
            if not np.isfinite(r_val) or not np.isfinite(bb_lower[i]) or not np.isfinite(bb_upper[i]):
                continue
            if close[i] < bb_lower[i] and r_val < rg_rsi_os and htf_fast[i] > htf_slow[i]:
                signals[i] = 1
                legs[i] = "range_reversion"
                strengths[i] = (rg_rsi_os - float(r_val)) / 15.0 + 0.5
                stop_mults[i] = float(rg_cfg["stop_atr_multiple"])
            elif close[i] > bb_upper[i] and r_val > rg_rsi_ob and htf_fast[i] < htf_slow[i]:
                signals[i] = -1
                legs[i] = "range_reversion"
                strengths[i] = (float(r_val) - rg_rsi_ob) / 15.0 + 0.5
                stop_mults[i] = float(rg_cfg["stop_atr_multiple"])

    out["signal"] = signals
    out["leg"] = legs
    out["signal_strength"] = strengths
    out["stop_atr_mult"] = stop_mults

    # Compatibility columns for `simulate_book` as well as `simulate_adaptive_v2`
    out["atr"] = out["atr14"]
    out["fast"] = out["ema20"]
    out["slow"] = out["ema50"]
    # Exit flags when HTF flips against trend or extreme_vol hits
    out["exit_long"] = (out["ema20"] < out["ema50"]) & (out["trend_4h_fast"] < out["trend_4h_slow"])
    out["exit_short"] = (out["ema20"] > out["ema50"]) & (out["trend_4h_fast"] > out["trend_4h_slow"])
    out["trail_long"] = out["chandelier_long"]
    out["trail_short"] = out["chandelier_short"]
    return out


def simulate_adaptive_v2(
    frames: dict[str, pd.DataFrame],
    specs: dict[str, Any] | None = None,
    funding: dict[str, list[dict[str, Any]]] | None = None,
    *,
    capital: float | None = None,
    leverage: float | None = None,
    risk_fraction: float | None = None,
    slippage: float | None = None,
    start: pd.Timestamp | str | None = None,
    end: pd.Timestamp | str | None = None,
    policy: dict[str, Any] | None = None,
    prepared: bool = False,
) -> dict[str, Any]:
    """Simulate multi-regime Adaptive v2 on 1..N contracts with strength priority & dynamic stops."""
    cfg = policy or load_policy_v2()
    rcfg = cfg["risk"]
    tr_cfg = cfg["trend_leg"]
    rg_cfg = cfg["range_leg"]

    cap = float(rcfg["default_equity_usdt"] if capital is None else capital)
    lev = float(rcfg["default_leverage"] if leverage is None else leverage)
    r_frac = float(rcfg["risk_fraction_per_trade"] if risk_fraction is None else risk_fraction)
    slip = float(rcfg["slippage_each_side"] if slippage is None else slippage)
    be_r = float(rcfg.get("breakeven_trigger_r", 1.5))
    daily_halt = float(rcfg.get("daily_loss_halt_pct", 0.03))
    total_halt = float(rcfg.get("total_loss_halt_pct", 0.12))

    data = {k: (v.copy() if prepared else attach_adaptive_v2(v, policy=cfg)) for k, v in frames.items()}
    v1_specs = {}
    for k in data:
        if specs and k in specs:
            s = specs[k]
            v1_specs[k] = s.to_v1_spec() if hasattr(s, "to_v1_spec") else s
        else:
            v1_specs[k] = resolve_spec(k).to_v1_spec()

    common = sorted(set.intersection(*(set(d.index) for d in data.values())))
    if len(common) < 70:
        raise ValueError("Insufficient common 1h bars for Adaptive v2 simulation")
    start_ts = pd.Timestamp(start).tz_convert("UTC") if start is not None and pd.Timestamp(start).tzinfo else (
        pd.Timestamp(start).tz_localize("UTC") if start is not None else common[65]
    )
    end_ts = pd.Timestamp(end).tz_convert("UTC") if end is not None and pd.Timestamp(end).tzinfo else (
        pd.Timestamp(end).tz_localize("UTC") if end is not None else common[-1]
    )

    fmaps = {
        k: {pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC"): float(r["rate"]) for r in rs}
        for k, rs in (funding or {}).items()
    }

    cash = cap
    pos: dict[str, Any] | None = None
    closed: list[dict[str, Any]] = []
    nav: list[tuple[pd.Timestamp, float]] = []
    rejections: Counter[str] = Counter()
    fees_total = 0.0
    funding_total = 0.0
    kill = False
    day = None
    day_start = cash

    def _close_pos(raw_px: float, ts: pd.Timestamp, reason: str) -> None:
        nonlocal cash, pos, fees_total
        assert pos is not None
        sp = v1_specs[pos["symbol"]]
        side = pos["side"]
        qty = pos["qty"]
        fill_px = raw_px * (1.0 - side * slip)
        exit_fee = qty * sp.contract_size * fill_px * sp.taker
        gross = exit_pnl(side, qty, pos["entry"], fill_px, sp)
        cash += gross - exit_fee
        fees_total += exit_fee
        closed.append({
            "symbol": pos["symbol"],
            "side": side,
            "leg": pos["leg"],
            "entry_time": str(pos["time"]),
            "exit_time": str(ts),
            "entry": pos["entry"],
            "exit": fill_px,
            "qty": qty,
            "net_pnl": gross - pos["entry_fee"] - exit_fee - pos["funding"],
            "exit_reason": reason,
            "breakeven_locked": pos["breakeven_locked"],
        })
        pos = None

    for ts in common:
        if ts < start_ts:
            continue
        if ts > end_ts:
            break
        if ts.date() != day:
            day = ts.date()
            day_start = cash + (
                exit_pnl(pos["side"], pos["qty"], pos["entry"], float(data[pos["symbol"]].loc[ts, "open"]), v1_specs[pos["symbol"]])
                if pos else 0.0
            )
        rows = {k: d.loc[ts] for k, d in data.items()}

        if pos is not None:
            k = pos["symbol"]
            sp = v1_specs[k]
            row = rows[k]
            prev = data[k].loc[ts - pd.Timedelta(hours=1)]

            rate = fmaps.get(k, {}).get(ts, 0.0)
            if rate:
                f_amt = pos["side"] * pos["qty"] * sp.contract_size * float(row["open"]) * rate
                cash -= f_amt
                funding_total += f_amt
                pos["funding"] += f_amt

            # Update trailing stop from prior completed bar (strictly known before bar `ts` opens)
            eff_stop = pos["stop"]
            if pos["leg"].startswith("trend"):
                if pos["side"] == 1 and math.isfinite(float(row["chandelier_long"])):
                    eff_stop = max(eff_stop, float(row["chandelier_long"]))
                elif pos["side"] == -1 and math.isfinite(float(row["chandelier_short"])):
                    eff_stop = min(eff_stop, float(row["chandelier_short"]))
                pos["stop"] = eff_stop

            # Check gap stop at open first
            gap = (pos["side"] == 1 and float(row["open"]) <= eff_stop) or (pos["side"] == -1 and float(row["open"]) >= eff_stop)
            if gap:
                _close_pos(float(row["open"]), ts, "gap_stop")
            else:
                # Leg-specific open exits using completed `prev` bar
                if pos["leg"] == "range_reversion":
                    mid_hit = (
                        math.isfinite(float(prev["bb_mid"]))
                        and ((pos["side"] == 1 and float(prev["close"]) >= float(prev["bb_mid"]))
                             or (pos["side"] == -1 and float(prev["close"]) <= float(prev["bb_mid"])))
                    )
                    max_h = int(rg_cfg["max_hold_hours"])
                    if mid_hit:
                        _close_pos(float(row["open"]), ts, "range_mid_target")
                    elif ts - pos["time"] >= pd.Timedelta(hours=max_h):
                        _close_pos(float(row["open"]), ts, "range_time_stop")
                else:
                    rev = (pos["side"] == 1 and bool(prev["exit_long"])) or (pos["side"] == -1 and bool(prev["exit_short"]))
                    max_h = int(tr_cfg["max_hold_hours"])
                    if rev:
                        _close_pos(float(row["open"]), ts, "trend_reversal")
                    elif ts - pos["time"] >= pd.Timedelta(hours=max_h):
                        _close_pos(float(row["open"]), ts, "trend_time_stop")

        eq_open = cash + (
            exit_pnl(pos["side"], pos["qty"], pos["entry"], float(rows[pos["symbol"]]["open"]), v1_specs[pos["symbol"]])
            if pos else 0.0
        )
        if eq_open <= day_start * (1.0 - daily_halt) or eq_open <= cap * (1.0 - total_halt):
            kill = True

        # Entry evaluation when flat: rank candidates by `signal_strength` (tie-break alphabetical)
        if pos is None and not kill:
            candidates = []
            for k in sorted(data):
                prev = data[k].loc[ts - pd.Timedelta(hours=1)]
                side = int(prev["signal"])
                if side != 0 and math.isfinite(float(prev["atr14"])) and float(prev["atr14"]) > 0:
                    candidates.append((float(prev["signal_strength"]), k, prev))
            candidates.sort(key=lambda item: (-item[0], item[1]))
            for _score, k, prev in candidates:
                side = int(prev["signal"])
                sp = v1_specs[k]
                fill_px = float(rows[k]["open"]) * (1.0 + side * slip)
                stop_dist = float(prev["stop_atr_mult"]) * float(prev["atr14"])
                # Use flexible size_order if leverage<=3 & r_frac<=0.02, else v2 sizing formula
                if 1.0 <= lev <= 3.0 and r_frac <= 0.02:
                    qty, reason = size_order(
                        cash, cash, fill_px, stop_dist, sp,
                        leverage=lev, risk_fraction=r_frac,
                        cash_floor=float(rcfg["cash_floor"]),
                        max_margin_fraction=float(rcfg["max_margin_fraction"]),
                        slippage=slip,
                    )
                else:
                    loss_per_c = sp.contract_size * (stop_dist + sp.taker * (2.0 * fill_px + stop_dist))
                    avail = min(cash * float(rcfg["max_margin_fraction"]), max(0.0, cash * (1.0 - float(rcfg["cash_floor"]))))
                    raw_q = min((cash * r_frac) / loss_per_c, avail / (fill_px * sp.contract_size * (1.0 / lev + sp.taker)))
                    from trading_lab.perps.risk import floor_step
                    qty = floor_step(max(0.0, raw_q), sp.qty_step)
                    reason = "ok" if (qty >= sp.min_qty and qty * fill_px * sp.contract_size >= sp.min_notional) else "minimum_order_exceeds_budget"
                    if reason != "ok":
                        qty = 0.0
                if qty <= 0:
                    rejections[reason] += 1
                    continue
                fee = qty * sp.contract_size * fill_px * sp.taker
                margin = qty * sp.contract_size * fill_px / lev
                if margin + fee > cash:
                    rejections["insufficient_cash"] += 1
                    continue
                cash -= fee
                fees_total += fee
                pos = {
                    "symbol": k,
                    "side": side,
                    "leg": str(prev["leg"]),
                    "qty": qty,
                    "entry": fill_px,
                    "initial_risk_dist": stop_dist,
                    "stop": fill_px - side * stop_dist,
                    "margin": margin,
                    "entry_fee": fee,
                    "funding": 0.0,
                    "time": ts,
                    "breakeven_locked": False,
                }
                break

        # Intrabar stop check, then end-of-bar breakeven ratchet for NEXT bar (no intrabar ordering lookahead!)
        if pos is not None:
            k = pos["symbol"]
            r = rows[k]
            hit_px = stop_fill(pos["side"], pos["stop"], float(r["open"]), float(r["low"]), float(r["high"]))
            if hit_px is not None:
                _close_pos(hit_px, ts + pd.Timedelta(hours=1), "stop_loss")
            elif not pos["breakeven_locked"]:
                sp = v1_specs[k]
                favorable = float(r["high"]) - pos["entry"] if pos["side"] == 1 else pos["entry"] - float(r["low"])
                if favorable >= be_r * pos["initial_risk_dist"]:
                    be_px = pos["entry"] * (1.0 + pos["side"] * (2.0 * sp.taker + slip))
                    if pos["side"] == 1 and be_px > pos["stop"]:
                        pos["stop"] = be_px
                        pos["breakeven_locked"] = True
                    elif pos["side"] == -1 and be_px < pos["stop"]:
                        pos["stop"] = be_px
                        pos["breakeven_locked"] = True

        eq_close = cash + (
            exit_pnl(pos["side"], pos["qty"], pos["entry"], float(rows[pos["symbol"]]["close"]), v1_specs[pos["symbol"]])
            if pos else 0.0
        )
        nav.append((ts + pd.Timedelta(hours=1), eq_close))
        if eq_close <= day_start * (1.0 - daily_halt) or eq_close <= cap * (1.0 - total_halt):
            kill = True
        if kill and pos is not None:
            _close_pos(float(rows[pos["symbol"]]["close"]), ts + pd.Timedelta(hours=1), "loss_halt")
            nav[-1] = (nav[-1][0], cash)

    eq = pd.Series([cap] + [v for _, v in nav], dtype=float)
    pnls = [t["net_pnl"] for t in closed]
    wins = [p for p in pnls if p > 0.01]
    losses = [p for p in pnls if p < -0.01]
    gross_profit = sum(p for p in pnls if p > 0)
    gross_loss = abs(sum(p for p in pnls if p < 0))
    profit_factor = (gross_profit / gross_loss) if gross_loss > 1e-9 else (float("inf") if gross_profit > 0 else None)

    return {
        "policy_id": "regime_adaptive_v2",
        "initial": cap,
        "final": float(eq.iloc[-1]),
        "return_pct": float((eq.iloc[-1] / cap - 1.0) * 100.0),
        "max_drawdown_pct": float((1.0 - eq / eq.cummax()).max() * 100.0) if len(eq) else 0.0,
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "winrate_pct": (100.0 * len(wins) / len(pnls)) if pnls else None,
        "profit_factor": round(profit_factor, 3) if (profit_factor is not None and math.isfinite(profit_factor)) else None,
        "fees": round(fees_total, 4),
        "funding_net_paid": round(funding_total, 4),
        "open_positions": int(pos is not None),
        "loss_halt": bool(kill),
        "rejections": dict(rejections),
        "trades_by_leg": dict(Counter(t["leg"] for t in closed)),
        "exit_reasons": dict(Counter(t["exit_reason"] for t in closed)),
        "trades": closed,
        "nav": [(str(t), float(v)) for t, v in nav],
    }


def latest_bar_signal_summary(
    h1: pd.DataFrame,
    daily: pd.DataFrame | None = None,
    *,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize the latest closed 1h bar's v2 regime, active setup, and watchlist triggers."""
    cfg = policy or load_policy_v2()
    if h1.empty or len(h1) < 65:
        return {
            "ok": False,
            "regime": "insufficient_data",
            "signal": "WAIT",
            "signal_int": 0,
            "reason": "insufficient_closed_hourly_bars",
        }
    prep = attach_adaptive_v2(h1, policy=cfg)
    last = prep.iloc[-1]
    close_px = float(last["close"])
    atr_val = float(last["atr14"]) if np.isfinite(last["atr14"]) else None
    sig_i = int(last["signal"])
    sig_label = "LONG" if sig_i == 1 else ("SHORT" if sig_i == -1 else "WAIT")
    stop_mult = float(last["stop_atr_mult"])
    stop_dist = (stop_mult * atr_val) if atr_val else None

    htf_rel = (
        "up" if (np.isfinite(last["trend_4h_fast"]) and np.isfinite(last["trend_4h_slow"]) and last["trend_4h_fast"] > last["trend_4h_slow"])
        else "down" if (np.isfinite(last["trend_4h_fast"]) and np.isfinite(last["trend_4h_slow"]) and last["trend_4h_fast"] < last["trend_4h_slow"])
        else "flat_or_unavailable"
    )
    h1_rel = (
        "up" if (np.isfinite(last["ema20"]) and np.isfinite(last["ema50"]) and last["ema20"] > last["ema50"])
        else "down" if (np.isfinite(last["ema20"]) and np.isfinite(last["ema50"]) and last["ema20"] < last["ema50"])
        else "flat_or_unavailable"
    )

    d1_rel = "unavailable"
    sma200_val = None
    d20_high = None
    d20_low = None
    if daily is not None and len(daily) >= 50:
        d_close = daily["close"].to_numpy(dtype=float)
        d_e20 = ema(d_close, 20)[-1]
        d_e50 = ema(d_close, 50)[-1]
        if np.isfinite(d_e20) and np.isfinite(d_e50):
            d1_rel = "up" if d_e20 > d_e50 else ("down" if d_e20 < d_e50 else "equal")
        if len(d_close) >= 200:
            sma200_val = float(np.mean(d_close[-200:]))
        if len(daily) >= 20:
            d20_high = float(daily["high"].iloc[-20:].max())
            d20_low = float(daily["low"].iloc[-20:].min())

    ret_24h = float((close_px / float(prep["close"].iloc[-25]) - 1.0) * 100.0) if len(prep) >= 25 else None

    # Actionable explanation & next trigger levels when currently WAIT
    regime = str(last["regime"])
    don_hi = float(last["donchian_high"]) if np.isfinite(last["donchian_high"]) else None
    don_lo = float(last["donchian_low"]) if np.isfinite(last["donchian_low"]) else None
    bb_up = float(last["bb_upper"]) if np.isfinite(last["bb_upper"]) else None
    bb_lo = float(last["bb_lower"]) if np.isfinite(last["bb_lower"]) else None
    bb_mid = float(last["bb_mid"]) if np.isfinite(last["bb_mid"]) else None

    if sig_i == 1:
        summary_cn = f"触发多头信号（{last['leg']}）：环境={regime}，建议下一开盘/回踩轻仓试多，初始止损距离 {stop_dist:.2f}。"
    elif sig_i == -1:
        summary_cn = f"触发空头信号（{last['leg']}）：环境={regime}，建议下一开盘/反弹轻仓试空，初始止损距离 {stop_dist:.2f}。"
    elif regime == "extreme_vol":
        summary_cn = "当前处于极端波动保护状态（ATR 突增），策略强制观望，严禁追涨杀跌。"
    elif regime == "trend_up":
        summary_cn = (
            f"多头趋势环境（1h/4h 共振向上，ADX={float(last['adx14']):.1f}），等待突破前高 "
            f"{don_hi:.2f} 或回踩 1h EMA20（{float(last['ema20']):.2f}）企稳。"
            if don_hi else "多头趋势环境，等待突破或回踩确认。"
        )
    elif regime == "trend_down":
        summary_cn = (
            f"空头趋势环境（1h/4h 共振向下，ADX={float(last['adx14']):.1f}），等待跌破前低 "
            f"{don_lo:.2f} 或反抽 1h EMA20（{float(last['ema20']):.2f}）受阻。"
            if don_lo else "空头趋势环境，等待破位或反抽受阻。"
        )
    elif regime == "range":
        summary_cn = (
            f"低波动震荡环境（ADX={float(last['adx14']):.1f} < 20），仅在顺 4h 方向触及布林下轨 "
            f"{bb_lo:.2f}（RSI<35）或上轨 {bb_up:.2f}（RSI>65）时低吸高抛。"
            if (bb_lo and bb_up) else "震荡环境，等待轨道极值与 RSI 共振。"
        )
    else:
        summary_cn = "多周期方向分歧或处于趋势过渡区（ADX 灰区），当前无高胜率信号，建议耐心观望。"

    return {
        "ok": True,
        "bar_open_utc": str(prep.index[-1]),
        "last_close": close_px,
        "return_24h_pct": round(ret_24h, 3) if ret_24h is not None else None,
        "regime": regime,
        "regime_reason": str(last["regime_reason"]),
        "signal": sig_label,
        "signal_int": sig_i,
        "leg": str(last["leg"]),
        "signal_strength": round(float(last["signal_strength"]), 3),
        "summary_cn": summary_cn,
        "timeframes": {"1h": h1_rel, "4h": htf_rel, "1d": d1_rel},
        "indicators": {
            "adx14": round(float(last["adx14"]), 2) if np.isfinite(last["adx14"]) else None,
            "plus_di14": round(float(last["plus_di14"]), 2) if np.isfinite(last["plus_di14"]) else None,
            "minus_di14": round(float(last["minus_di14"]), 2) if np.isfinite(last["minus_di14"]) else None,
            "rsi14": round(float(last["rsi14"]), 2) if np.isfinite(last["rsi14"]) else None,
            "atr14": round(atr_val, 4) if atr_val is not None else None,
            "atr14_pct": round(atr_val / close_px * 100.0, 3) if (atr_val and close_px > 0) else None,
            "atr_over_mean50": round(float(last["atr14"] / last["atr14_mean50"]), 3)
            if (np.isfinite(last["atr14"]) and np.isfinite(last["atr14_mean50"]) and last["atr14_mean50"] > 0)
            else None,
            "ema20_1h": round(float(last["ema20"]), 4) if np.isfinite(last["ema20"]) else None,
            "ema50_1h": round(float(last["ema50"]), 4) if np.isfinite(last["ema50"]) else None,
            "sma200_1d": round(sma200_val, 4) if sma200_val is not None else None,
        },
        "key_levels": {
            "donchian_48h_high": round(don_hi, 4) if don_hi is not None else None,
            "donchian_48h_low": round(don_lo, 4) if don_lo is not None else None,
            "bb_upper": round(bb_up, 4) if bb_up is not None else None,
            "bb_mid": round(bb_mid, 4) if bb_mid is not None else None,
            "bb_lower": round(bb_lo, 4) if bb_lo is not None else None,
            "chandelier_long_stop": round(float(last["chandelier_long"]), 4) if np.isfinite(last["chandelier_long"]) else None,
            "chandelier_short_stop": round(float(last["chandelier_short"]), 4) if np.isfinite(last["chandelier_short"]) else None,
            "daily_20d_high": round(d20_high, 4) if d20_high is not None else None,
            "daily_20d_low": round(d20_low, 4) if d20_low is not None else None,
        },
        "suggested_stop_distance": round(stop_dist, 6) if stop_dist is not None else None,
        "suggested_stop_price": (
            round(close_px - sig_i * stop_dist, 6) if (sig_i != 0 and stop_dist is not None) else None
        ),
        "suggested_tp1_price": (
            round(close_px + sig_i * 1.5 * stop_dist, 6) if (sig_i != 0 and stop_dist is not None) else None
        ),
        "suggested_tp2_price": (
            round(close_px + sig_i * 2.5 * stop_dist, 6) if (sig_i != 0 and stop_dist is not None) else None
        ),
    }
