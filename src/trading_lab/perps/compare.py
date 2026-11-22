"""Separate experiment: compare fixed, cited strategies on Bitget perps.

This is not the production default, not a live engine, and not a historical
multi-agent or LLM prediction test. Official model APIs are not used.

Precommitted (locked before looking at strategy returns):
- Baseline ema_cross_1h: existing 1h EMA20/50 cross, 2*ATR14 stop, next-open,
  reverse when the lagged EMAs have flipped. Unchanged rules.
- turtle_s1_donchian: Turtle System 1 as documented in Faith (2007), Way of the
  Turtle: 20-day Donchian breakout entry, 10-day Donchian exit, 2N stop with
  N=ATR(20) daily. No pyramiding (this book allows only one global position).
  A 100-day / 10-month SMA trend filter was considered and dropped ex ante
  because NDX100 and XAU public listings are too short for a 100-day filter
  plus a meaningful earlier window.
- tsmom_3m: Moskowitz, Ooi, Pedersen (2012), Journal of Financial Economics,
  "Time Series Momentum", who report lookbacks of 1, 3, 6, 9 and 12 months.
  The paper's headline 12-month lookback is not identifiable on the common
  BTC/XAU/NDX100 history (NDX100 public 1h starts around 2026-06). Precommit
  the longest paper lookback that fits: 3-month sign of close-to-close return,
  monthly rebalance after the last completed UTC month, plus a 2*ATR(20) daily
  stop used only as a 30U risk overlay (not in the original paper). 60-day
  realized vol is recorded but size still goes through the same 1% risk /
  size_order cap as the other candidates so exposure rules stay comparable.

Windows (precommitted):
- REUSED_EXPLORATORY: 2026-09-17 06:00 UTC through last locked bar
  2026-11-16 05:00 UTC. This cache was already used; not a holdout.
- PRECOMMIT_EARLIER: common 1h bars strictly before 2026-09-17 06:00 UTC,
  starting at the first hour when all three candidate signals can be formed
  (warmup uses only earlier public bars). One-shot; no parameter change after
  seeing results.

Same book for all: 30 USDT, 2x, 1% risk, 20% cash floor, 50% max margin,
one global position, taker 0.06%, 5 bps slip each side, historical funding
times open as mark proxy, 3% daily / 10% total loss halt, min-order fail-closed,
next-open entries, no interpolation of gaps, liquidation not venue-verified.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from trading_lab.perps.research import PAIRS, indicators, validate_bars
from trading_lab.perps.risk import Spec, exit_pnl, size_order, stop_fill

ROOT = Path(__file__).resolve().parents[3]
LOCKED_CACHE = ROOT / "data" / "perps"
EXT_CACHE = ROOT / "data" / "perps-extended"
ART = ROOT / "artifacts"

REUSED_START = pd.Timestamp("2026-09-17 06:00:00+00:00")
REUSED_LAST_BAR = pd.Timestamp("2026-11-16 05:00:00+00:00")
EARLIER_END = REUSED_START  # exclusive
FETCH_SINCE = pd.Timestamp("2025-12-01 00:00:00+00:00")
CAPITAL = 30.0
LEVERAGE = 2.0
SLIPPAGE = 0.0005
WIN_EPS = 0.01

CITATIONS = {
    "ema_cross_1h": "Existing trading-lab baseline; 1h EMA20/50, 2*ATR14. Not a published edge.",
    "turtle_s1_donchian": (
        "Faith, C. (2007). Way of the Turtle. McGraw-Hill. System 1: 20-day breakout, "
        "10-day exit, 2N stop, N=20-day ATR. Public restatement of the 1983 Turtle rules."
    ),
    "tsmom_3m": (
        "Moskowitz, T., Ooi, Y. H., Pedersen, L. H. (2012). Time Series Momentum. "
        "Journal of Financial Economics 104(2). Lookbacks 1/3/6/9/12 months; 12-month "
        "not identifiable here, so 3-month is precommitted. 2*ATR(20) stop is an added overlay."
    ),
}


def _utc(ts) -> pd.Timestamp:
    t = pd.Timestamp(ts)
    return t if t.tzinfo is not None else t.tz_localize("UTC")


def daily_bars(h1: pd.DataFrame) -> pd.DataFrame:
    ohlc = h1[["open", "high", "low", "close", "volume"]]
    d = ohlc.resample("1D", label="left", closed="left").agg({
        "open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum",
    })
    return d.dropna(subset=["open", "high", "low", "close"])


def attach_turtle_s1(h1: pd.DataFrame) -> pd.DataFrame:
    """Prefix-only 20/10 Donchian and ATR(20) mapped onto 1h bars."""
    out = h1.copy()
    daily = daily_bars(h1)
    prev_high_20 = daily["high"].shift(1).rolling(20, min_periods=20).max()
    prev_low_20 = daily["low"].shift(1).rolling(20, min_periods=20).min()
    prev_high_10 = daily["high"].shift(1).rolling(10, min_periods=10).max()
    prev_low_10 = daily["low"].shift(1).rolling(10, min_periods=10).min()
    # Wilder / TA-Lib ATR on daily, prefix only via shift(1) for the day's known N
    import talib
    atr20 = pd.Series(
        talib.ATR(daily.high.to_numpy(dtype=float), daily.low.to_numpy(dtype=float),
                  daily.close.to_numpy(dtype=float), timeperiod=20),
        index=daily.index,
    )
    known_atr = atr20.shift(1)
    day_index = out.index.normalize()
    out["donchian_high_20"] = day_index.map(prev_high_20)
    out["donchian_low_20"] = day_index.map(prev_low_20)
    out["donchian_high_10"] = day_index.map(prev_high_10)
    out["donchian_low_10"] = day_index.map(prev_low_10)
    out["atr"] = day_index.map(known_atr)
    out["signal"] = 0
    long_brk = out["high"] >= out["donchian_high_20"]
    short_brk = out["low"] <= out["donchian_low_20"]
    out.loc[long_brk & out["atr"].notna(), "signal"] = 1
    out.loc[short_brk & out["atr"].notna() & ~long_brk, "signal"] = -1
    out["exit_long"] = out["low"] <= out["donchian_low_10"]
    out["exit_short"] = out["high"] >= out["donchian_high_10"]
    out["trail_long"] = out["donchian_low_10"]
    out["trail_short"] = out["donchian_high_10"]
    out["fast"] = out["donchian_high_20"]
    out["slow"] = out["donchian_low_20"]
    return out


def attach_tsmom_3m(h1: pd.DataFrame) -> pd.DataFrame:
    """3-month sign at month-end, held the following month; daily ATR for stops."""
    out = h1.copy()
    daily = daily_bars(h1)
    import talib
    atr20 = pd.Series(
        talib.ATR(daily.high.to_numpy(dtype=float), daily.low.to_numpy(dtype=float),
                  daily.close.to_numpy(dtype=float), timeperiod=20),
        index=daily.index,
    )
    # 3 calendar months of daily close-to-close; require at least 60 daily observations
    ret_3m = daily["close"] / daily["close"].shift(90) - 1.0
    valid = daily["close"].shift(90).notna()
    # A truncated sample's last day is not necessarily a month end, and a
    # partially observed final day cannot supply a month-end closing price.
    # Require the actual UTC calendar month end and its completed 23:00 bar.
    last_hour = out.groupby(out.index.normalize()).tail(1).index
    last_hour = last_hour[last_hour.hour == 23]
    complete_days = set(last_hour.normalize())
    month_end = daily.index[[day.is_month_end and day in complete_days for day in daily.index]]
    monthly_signal = pd.Series(0, index=daily.index, dtype=int)
    for end in month_end:
        if end not in ret_3m.index or not valid.loc[end] or not math.isfinite(ret_3m.loc[end]):
            continue
        monthly_signal.loc[end] = 1 if ret_3m.loc[end] > 0 else -1
    # hold through the next month: map month M signal (set at M month-end) onto all days of month M+1
    hold = pd.Series(0, index=daily.index, dtype=int)
    sig_by_month = {pd.Timestamp(end).to_period("M"): int(monthly_signal.loc[end]) for end in month_end}
    for ts in daily.index:
        prev_m = (pd.Timestamp(ts).to_period("M") - 1)
        hold.loc[ts] = int(sig_by_month.get(prev_m, 0))
    # At any hour of a day only the preceding daily close is available.
    vol60 = daily["close"].pct_change().rolling(60, min_periods=60).std().shift(1)
    day_index = out.index.normalize()
    out["tsmom_hold"] = day_index.map(hold).fillna(0).astype(int)
    out["vol60"] = day_index.map(vol60)
    out["atr"] = day_index.map(atr20.shift(1))
    # Entry signal only on the first 1h bar of a new holding month if the held sign is non-zero
    # (the previous 1h bar is the month-end close bar when simulate looks at prev.signal).
    out["signal"] = 0
    # Put the signal on the last 1h bar of the signal month so next-open is the first hour of the hold month.
    for end in month_end:
        day = pd.Timestamp(end).normalize()
        candidates = last_hour[last_hour.normalize() == day]
        if len(candidates) == 0:
            continue
        sig = int(monthly_signal.loc[end])
        if sig and math.isfinite(out.at[candidates[-1], "atr"] if candidates[-1] in out.index else float("nan")):
            out.at[candidates[-1], "signal"] = sig
    # Also allow re-entry later in the hold month if a stop exited early: use hold sign as a standing signal
    # on the last bar of each hold day except we already place month-end signals. Standing signal on every
    # day's last hour would spam signal attempts; simulate only enters when flat, and seen ids include side+ts.
    # Standing: set signal=hold on each day's last 1h bar so a stopped-out book can re-enter next day open.
    for ts in last_hour:
        if out.at[ts, "signal"] == 0:
            held = int(out.at[ts, "tsmom_hold"])
            if held and math.isfinite(out.at[ts, "atr"] or float("nan")):
                out.at[ts, "signal"] = held
    # Month-end bars carry the new month's signal while tsmom_hold is still the month just ending.
    out["exit_long"] = (out["tsmom_hold"] <= 0) | (out["signal"] < 0)
    out["exit_short"] = (out["tsmom_hold"] >= 0) | (out["signal"] > 0)
    # During ordinary hours signal is 0, so exits follow the held sign only.
    out["fast"] = out["tsmom_hold"].astype(float)
    out["slow"] = 0.0
    out["trail_long"] = float("nan")
    out["trail_short"] = float("nan")
    return out


def attach_ema(h1: pd.DataFrame) -> pd.DataFrame:
    d = indicators(h1)
    d["exit_long"] = d["fast"] < d["slow"]
    d["exit_short"] = d["fast"] > d["slow"]
    d["trail_long"] = float("nan")
    d["trail_short"] = float("nan")
    return d


PREPARERS = {
    "ema_cross_1h": attach_ema,
    "turtle_s1_donchian": attach_turtle_s1,
    "tsmom_3m": attach_tsmom_3m,
}


def simulate_book(frames, specs, funding=None, *, capital=CAPITAL, leverage=LEVERAGE,
                  start=None, end=None, slippage=SLIPPAGE, prepared=False, preparer=None, halt_mode="equity",
                  funding_intervals=None, entry_priority=None, max_hold_hours=None):
    """Same event order as research.simulate, with explicit exit flags and optional trail.

    Entries at next open; ATR (and trail) stops intra-bar, gaps use the adverse open;
    if ATR and trail both trade in the same bar, the worse fill is used.
    Optional max_hold_hours exits at the later open after a gap-stop check; default
    None preserves the existing book with no time stop.

    halt_mode="equity" uses mark-to-market equity for the 3%/10% halt.
    halt_mode="legacy_cash" copies research.simulate, which can latch the daily
    halt after an unrealized gain because cash excludes MTM while day_start includes it.
    """
    if not 1 <= leverage <= 3:
        raise ValueError("Example simulation leverage restricted to 1..3")
    if not math.isfinite(float(capital)) or float(capital) <= 0:
        raise ValueError("capital must be finite and positive")
    if max_hold_hours is not None:
        if (isinstance(max_hold_hours, bool) or not isinstance(max_hold_hours, (int, float))
                or not math.isfinite(max_hold_hours) or int(max_hold_hours) != max_hold_hours
                or int(max_hold_hours) < 1):
            raise ValueError("max_hold_hours must be a positive integer number of hours")
        max_hold_hours = int(max_hold_hours)
    if entry_priority is None:
        entry_priority = tuple(sorted(frames))
    elif (not isinstance(entry_priority, (list, tuple)) or len(entry_priority) != len(frames)
          or len(set(entry_priority)) != len(entry_priority) or set(entry_priority) != set(frames)):
        raise ValueError("entry priority must be a permutation of all symbols")
    for d in frames.values():
        validate_bars(d)
    if prepared:
        data = {k: v.copy() for k, v in frames.items()}
    else:
        fn = preparer or attach_ema
        data = {k: fn(v) for k, v in frames.items()}
    common = sorted(set.intersection(*(set(d.index) for d in data.values())))
    if not common:
        raise ValueError("No common timestamps")
    start = _utc(start) if start is not None else common[60]
    end = _utc(end) if end is not None else common[-1] + pd.Timedelta(hours=1)
    if funding_intervals is not None:
        if set(funding_intervals) != set(data) or funding is None or set(funding) != set(data):
            raise ValueError("complete funding and intervals required for all symbols")
        if not funding_complete_between(funding, funding_intervals, start, min(end, common[-1]))["complete"]:
            raise ValueError("incomplete or invalid funding under supplied interval assumptions")
    cash = float(capital)
    pos = None
    closed = []
    nav = []
    reject = Counter()
    fees = 0.0
    fundcost = 0.0
    kill = False
    day = None
    day_start = cash
    last_event = None
    seen = set()
    stress_breaches = 0
    bars_with_position = 0
    end_bar_notional_ratios = []
    fmaps = {
        k: {pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC"): float(r["rate"]) for r in rs}
        for k, rs in (funding or {}).items()
    }

    def close(px, ts, reason):
        nonlocal cash, pos, fees, stress_breaches
        spec = specs[pos["symbol"]]
        qty = pos["qty"]
        side = pos["side"]
        px = px * (1 - side * slippage)
        exit_fee = qty * spec.contract_size * px * spec.taker
        gross = exit_pnl(side, qty, pos["entry"], px, spec)
        if -gross + exit_fee > pos["margin"]:
            stress_breaches += 1
        cash += gross - exit_fee
        fees += exit_fee
        if cash < 0:
            raise ValueError("Insolvent stress path: exact isolated liquidation model required; do not publish returns")
        closed.append(dict(
            symbol=pos["symbol"], side=side, entry_time=str(pos["time"]), exit_time=str(ts),
            net_pnl=gross - pos["entry_fee"] - exit_fee - pos["funding"], exit_reason=reason,
        ))
        pos = None

    def effective_stop(pos, row):
        stop = pos["stop"]
        if pos["side"] == 1 and math.isfinite(row.get("trail_long", float("nan")) if isinstance(row, dict) else row["trail_long"] if "trail_long" in row.index else float("nan")):
            trail = float(row["trail_long"])
            if math.isfinite(trail):
                stop = max(stop, trail)
        if pos["side"] == -1 and "trail_short" in row.index and math.isfinite(float(row["trail_short"])):
            trail = float(row["trail_short"])
            if math.isfinite(trail):
                stop = min(stop, trail)
        return stop

    for ts in common:
        if ts < start:
            continue
        if ts > end or ts > common[-1]:
            break
        position_at_open = pos is not None
        if last_event is not None and ts - last_event != pd.Timedelta(hours=1):
            raise ValueError("Stale shared timeline")
        last_event = ts
        if ts.date() != day:
            day = ts.date()
            day_start = cash + (exit_pnl(pos["side"], pos["qty"], pos["entry"], data[pos["symbol"]].loc[ts, "open"], specs[pos["symbol"]]) if pos else 0.0)
        rows = {k: d.loc[ts] for k, d in data.items()}
        if pos:
            k = pos["symbol"]
            rate = fmaps.get(k, {}).get(ts, 0.0)
            amount = pos["side"] * pos["qty"] * specs[k].contract_size * rows[k]["open"] * rate
            cash -= amount
            fundcost += amount
            pos["funding"] += amount
            if cash < 0:
                raise ValueError("Funding insolvency; exact liquidation model required")
            prev = data[k].loc[ts - pd.Timedelta(hours=1)]
            estop = effective_stop(pos, rows[k])  # trail uses completed prior days only, known at this open
            gap = (pos["side"] == 1 and rows[k]["open"] <= estop) or (pos["side"] == -1 and rows[k]["open"] >= estop)
            if gap:
                close(rows[k]["open"], ts, "gap_stop")
            else:
                rev = (pos["side"] == 1 and bool(prev["exit_long"])) or (pos["side"] == -1 and bool(prev["exit_short"]))
                if rev:
                    close(rows[k]["open"], ts, "reverse_signal")
                elif (max_hold_hours is not None
                      and ts - pos["time"] >= pd.Timedelta(hours=max_hold_hours)):
                    close(rows[k]["open"], ts, "time_stop")
        if pos:
            kpos = pos["symbol"]
            eq_open = cash + exit_pnl(pos["side"], pos["qty"], pos["entry"], rows[kpos]["open"], specs[kpos])
        else:
            eq_open = cash
        if halt_mode == "legacy_cash":
            if cash <= day_start * 0.97:
                kill = True
            if cash <= capital * 0.90:
                kill = True
        else:
            if eq_open <= day_start * 0.97 or eq_open <= capital * 0.90:
                kill = True
        if not pos and not kill:
            for k in entry_priority:
                prev = data[k].loc[ts - pd.Timedelta(hours=1)]
                side = int(prev["signal"])
                if not side or not math.isfinite(prev["atr"]):
                    continue
                oid = (k, ts, side)
                if oid in seen:
                    raise ValueError("Duplicate signal/order identifier")
                seen.add(oid)
                px = rows[k]["open"] * (1 + side * slippage)
                distance = 2 * float(prev["atr"])
                spec = specs[k]
                qty, reason = size_order(cash, cash, px, distance, spec, leverage=leverage, slippage=slippage)
                if not qty:
                    reject[reason] += 1
                    continue
                margin = qty * spec.contract_size * px / leverage
                fee = qty * spec.contract_size * px * spec.taker
                if margin + fee > cash:
                    raise AssertionError("Margin double spend")
                cash -= fee
                fees += fee
                pos = dict(symbol=k, side=side, qty=qty, entry=px, stop=px - side * distance,
                           margin=margin, entry_fee=fee, funding=0.0, time=ts)
                break
        if position_at_open or pos is not None:
            bars_with_position += 1
        if pos:
            k = pos["symbol"]
            r = rows[k]
            estop = effective_stop(pos, r)
            px = stop_fill(pos["side"], estop, r.open, r.low, r.high)
            if px is not None:
                close(px, ts + pd.Timedelta(hours=1), "atr_stop")
        equity = cash
        if pos:
            k = pos["symbol"]
            equity += exit_pnl(pos["side"], pos["qty"], pos["entry"], rows[k]["close"], specs[k])
        if equity < 0:
            raise ValueError("Negative equity stress path")
        nav.append((ts + pd.Timedelta(hours=1), equity))
        if equity <= day_start * 0.97 or equity <= capital * 0.9:
            kill = True
        if kill and pos:
            close(rows[pos["symbol"]]["close"], ts + pd.Timedelta(hours=1), "loss_halt")
            nav[-1] = (nav[-1][0], cash)
        if pos:
            symbol = pos["symbol"]
            notional = pos["qty"] * specs[symbol].contract_size * rows[symbol]["close"]
            end_bar_notional_ratios.append(notional / capital)
        else:
            end_bar_notional_ratios.append(0.0)

    eq = pd.Series([capital] + [v for _, v in nav], dtype=float)
    pnls = [x["net_pnl"] for x in closed]
    wins = [p for p in pnls if p > WIN_EPS]
    losses = [p for p in pnls if p < -WIN_EPS]
    open_pnl = 0.0
    if pos:
        open_pnl = nav[-1][1] - cash - pos["entry_fee"] - pos["funding"]
    recon = abs(sum(pnls) + open_pnl - (eq.iloc[-1] - capital)) if len(eq) else 0.0
    if recon >= 1e-7:
        raise AssertionError(f"PnL reconciliation failed: {recon}")
    return dict(
        initial=capital, final=float(eq.iloc[-1]) if len(eq) else capital,
        return_pct=float((eq.iloc[-1] / capital - 1) * 100) if len(eq) else 0.0,
        max_drawdown_pct=float((1 - eq / eq.cummax()).max() * 100) if len(eq) else 0.0,
        closed_trades=len(closed), wins=len(wins), losses=len(losses),
        breakeven=int(sum(abs(p) <= WIN_EPS for p in pnls)),
        winrate_pct=(100 * len(wins) / len(pnls)) if pnls else None,
        fees=fees, funding_net_paid=fundcost if funding_intervals is not None else None,
        funding_applied_unverified=fundcost if funding_intervals is None else None,
        open_positions=int(pos is not None),
        funding_coverage_checked=funding_intervals is not None,
        funding_coverage_note=("Complete under supplied interval assumptions; historical schedule unverified."
                               if funding_intervals is not None else
                               "Funding coverage not checked; returns are not validated net performance."),
        rejections=dict(reject), loss_halt=bool(kill),
        stress_margin_breaches=stress_breaches,
        stress_screen_passed=stress_breaches == 0,
        reconciliation_error=recon,
        bars_with_position=bars_with_position,
        bars_processed=len(nav),
        exposure_bar_fraction=(bars_with_position / len(nav)) if nav else None,
        exposure_note="Upper bound: counts an entire hourly bar if a position exists at its open or is opened within it; not notional-weighted.",
        mean_end_bar_notional_over_initial=(sum(end_bar_notional_ratios) / len(end_bar_notional_ratios)) if end_bar_notional_ratios else None,
        peak_end_bar_notional_over_initial=max(end_bar_notional_ratios) if end_bar_notional_ratios else None,
        notional_exposure_note="Hourly close snapshots divided by initial capital; excludes intrabar-only positions and is not exchange margin.",
        start=str(nav[0][0] - pd.Timedelta(hours=1)) if nav else None,
        end=str(nav[-1][0]) if nav else None,
        trades=closed, nav=[(str(t), v) for t, v in nav],
        open_unrealized=open_pnl,
    )


def load_cache(cache: Path):
    prov_path = cache / "provenance.json"
    if not prov_path.exists():
        raise FileNotFoundError(f"no provenance in {cache}")
    p = json.loads(prov_path.read_text())
    frames, specs, funding = {}, {}, {}
    for k, v in p["symbols"].items():
        sym = k.split("/")[0] + "USDT"
        d = pd.read_csv(cache / f"{sym}.csv", parse_dates=["date"]).set_index("date")
        d.index = pd.to_datetime(d.index, utc=True)
        frames[k] = d
        specs[k] = Spec(**v["spec"])
        fp = cache / f"{sym}-funding.csv"
        funding[k] = pd.read_csv(fp).to_dict("records") if fp.exists() else []
    return frames, specs, funding, p


def funding_complete_between(funding: dict, intervals: dict, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """end is last bar open; expected funding times in [start, end]."""
    start, end = _utc(start), _utc(end)
    if start > end or set(funding) != set(intervals) or not funding:
        raise ValueError("ordered window and matching funding/interval symbols required")
    out = {"complete": True, "missing": {}, "invalid": {}, "coverage_start": None, "coverage_end": None,
           "observed_bounds_by_symbol": {},
           "coverage_note": "Bounds are the intersection of observed timestamp extents, not proof of continuous coverage or historical settlement schedules."}
    cov_starts, cov_ends = [], []
    for k, rows in funding.items():
        interval = intervals[k]
        if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or int(interval) != interval or interval <= 0 or 24 % interval:
            raise ValueError(f"invalid funding interval for {k}")
        interval = int(interval)
        ts = []
        problems = []
        seen = set()
        for r in rows:
            try:
                timestamp = r["timestamp"]
                if isinstance(timestamp, bool) or not math.isfinite(float(timestamp)) or float(timestamp) != int(timestamp):
                    raise ValueError("invalid timestamp")
                t = pd.Timestamp(int(timestamp), unit="ms", tz="UTC")
                rate = r["rate"]
                if isinstance(rate, bool) or not math.isfinite(float(rate)):
                    raise ValueError("invalid rate")
            except (KeyError, TypeError, ValueError, OverflowError):
                problems.append("malformed_timestamp_or_rate")
                continue
            if start <= t <= end:
                if t in seen:
                    problems.append("duplicate_funding_timestamp")
                if t != t.floor(f"{interval}h"):
                    problems.append("unexpected_funding_timestamp")
                seen.add(t)
            ts.append(t)
        if ts:
            cov_starts.append(min(ts)); cov_ends.append(max(ts))
            out["observed_bounds_by_symbol"][k] = {"first": str(min(ts)), "last": str(max(ts))}
        else:
            out["observed_bounds_by_symbol"][k] = {"first": None, "last": None}
        expected = pd.date_range(start.ceil(f"{interval}h"), end, freq=f"{interval}h")
        available = set(ts)
        missing = [str(x) for x in expected if x not in available]
        if missing:
            out["complete"] = False
            out["missing"][k] = {"count": len(missing), "first": missing[0], "last": missing[-1]}
        if problems:
            out["complete"] = False
            out["invalid"][k] = sorted(set(problems))
    if len(cov_starts) == len(funding) and max(cov_starts) <= min(cov_ends):
        out["coverage_start"] = str(max(cov_starts))
        out["coverage_end"] = str(min(cov_ends))
    return out


def first_signal_time(frames, preparer) -> pd.Timestamp | None:
    times = []
    for d in frames.values():
        pre = preparer(d)
        sig = pre.index[pre["signal"] != 0]
        if len(sig):
            times.append(sig[0])
    return max(times) if times else None


def first_common_defined(frames, preparer) -> pd.Timestamp | None:
    """Earliest common 1h timestamp at which every symbol has a defined ATR (warmup done)."""
    ready = None
    for d in frames.values():
        pre = preparer(d)
        ok = pre.index[pre["atr"].notna() & pre["atr"].apply(lambda x: math.isfinite(x) if pd.notna(x) else False)]
        if len(ok) == 0:
            return None
        ready = ok[0] if ready is None else max(ready, ok[0])
    return ready


def fetch_extended(since: pd.Timestamp = FETCH_SINCE, until: pd.Timestamp = REUSED_LAST_BAR, cache: Path = EXT_CACHE) -> dict:
    """Public Bitget 1h + funding into a separate directory. Does not rewrite the locked 60d cache."""
    import time
    import ccxt
    e = ccxt.bitget({"enableRateLimit": True, "timeout": 30000})
    markets = e.load_markets()
    end = _utc(until) + pd.Timedelta(hours=1)  # exclusive, include last locked bar
    start = _utc(since)
    cache.mkdir(parents=True, exist_ok=True)
    provenance = {
        "fetched_at_taipei": str(pd.Timestamp.now(tz="UTC").tz_convert("Asia/Taipei")),
        "ccxt": ccxt.__version__,
        "venue": "bitget",
        "kind": "public real 1h perpetual OHLCV for strategy comparison; not the locked 60d cache",
        "fetch_since": str(start),
        "fetch_until_exclusive": str(end),
        "symbols": {},
    }
    for pair in PAIRS:
        m = markets[pair]
        if not (m["active"] and m["swap"] and m["linear"]):
            raise ValueError("Unsupported market " + pair)
        rows = []
        cursor = int(start.timestamp() * 1000)
        empty_streak = 0
        while cursor < int(end.timestamp() * 1000):
            time.sleep(0.8)
            page = e.fetch_ohlcv(pair, "1h", since=cursor, limit=200)
            if not page:
                empty_streak += 1
                if empty_streak >= 3:
                    # listing may be later than `since`; jump forward a week
                    cursor += 7 * 24 * 3600 * 1000
                    empty_streak = 0
                    if cursor >= int(end.timestamp() * 1000):
                        break
                continue
            empty_streak = 0
            rows.extend(page)
            nxt = max(x[0] for x in page)
            if nxt <= cursor:
                break
            cursor = nxt
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"]).drop_duplicates("timestamp").sort_values("timestamp")
        if df.empty:
            raise ValueError(f"no public 1h bars for {pair} in requested window")
        df.index = pd.to_datetime(df.pop("timestamp"), unit="ms", utc=True)
        df = df.loc[(df.index >= start) & (df.index < end)]
        raw_count = len(df)
        gaps = df.index.to_series().diff() > pd.Timedelta(hours=1)
        gap_list = [str(t) for t in df.index[gaps]]
        if gap_list:
            # keep last contiguous suffix so validate_bars can run; record discarded prefix
            df = df.loc[gaps[gaps].index[-1]:]
        validate_bars(df)
        sym = m["id"]
        df.to_csv(cache / f"{sym}.csv", index_label="date")
        funding = []
        funding_error = None
        cursor = int(df.index[0].timestamp() * 1000)
        try:
            for page_no in range(1, 41):
                time.sleep(0.8)
                page = e.fetch_funding_rate_history(pair, limit=100, params={"pageNo": page_no})
                page = [x for x in page if x.get("timestamp") is not None]
                if not page:
                    break
                funding.extend({"timestamp": x["timestamp"], "rate": float(x["fundingRate"])} for x in page)
                if min(x["timestamp"] for x in page) <= int(df.index[0].timestamp() * 1000):
                    break
        except Exception as exc:
            funding_error = f"{type(exc).__name__}: {exc}"
        fd = pd.DataFrame(funding, columns=["timestamp", "rate"]).drop_duplicates("timestamp").sort_values("timestamp")
        if not fd.empty:
            fd = fd.loc[(fd.timestamp >= int(df.index[0].timestamp() * 1000)) & (fd.timestamp < int(end.timestamp() * 1000))]
        fd.to_csv(cache / f"{sym}-funding.csv", index=False)
        spec = Spec(pair, m["limits"]["amount"]["min"], m["precision"]["amount"], m["limits"]["cost"]["min"],
                    float(m.get("taker") or 0.0006), m.get("contractSize") or 1.0)
        interval = int(m["info"].get("fundInterval", 8))
        expected = pd.date_range(df.index[0].ceil(f"{interval}h"), end - pd.Timedelta(milliseconds=1), freq=f"{interval}h")
        available = set(fd.timestamp.tolist()) if not fd.empty else set()
        complete = all(int(x.timestamp() * 1000) in available for x in expected)
        provenance["symbols"][pair] = dict(
            spec=asdict(spec), bars=len(df), first=str(df.index[0]), last=str(df.index[-1]),
            funding_count=len(fd), funding_complete=complete, funding_error=funding_error,
            funding_interval_hours=interval, raw_bars=raw_count, gap_after_dates=gap_list,
            discarded_prefix_bars=raw_count - len(df),
        )
    (cache / "provenance.json").write_text(json.dumps(provenance, indent=2))
    return provenance


def summarize(result: dict) -> dict:
    keys = [
        "initial", "final", "return_pct", "max_drawdown_pct", "closed_trades", "wins", "losses",
        "breakeven", "winrate_pct", "fees", "funding_net_paid", "funding_applied_unverified", "funding_coverage_checked",
        "funding_coverage_note", "open_positions", "open_unrealized",
        "rejections", "loss_halt", "stress_margin_breaches", "stress_screen_passed",
        "reconciliation_error", "start", "end", "bars_with_position", "bars_processed",
        "exposure_bar_fraction", "exposure_note",
        "mean_end_bar_notional_over_initial", "peak_end_bar_notional_over_initial", "notional_exposure_note",
    ]
    out = {k: result.get(k) for k in keys}
    reasons = Counter(t["exit_reason"] for t in result.get("trades") or [])
    symbols = Counter(t["symbol"] for t in result.get("trades") or [])
    out["exit_reasons"] = dict(reasons)
    out["trades_by_symbol"] = dict(symbols)
    return out


def run_comparison(data_dir: Path, locked_dir: Path = LOCKED_CACHE) -> dict:
    frames, specs, funding, prov = load_cache(data_dir)
    common = sorted(set.intersection(*(set(d.index) for d in frames.values())))
    if not common:
        raise ValueError("no common 1h timestamps")
    # Prefix frames may start before REUSED_START
    reused_end = min(REUSED_LAST_BAR, common[-1])
    earlier_bars = [t for t in common if t < EARLIER_END]
    run_recorded_at = datetime.now(timezone.utc)

    # Common start for all three strategies on earlier data: first hour when every
    # preparer has ATR defined on all symbols, plus one hour (so prev bar exists).
    defined_at = []
    for name, fn in PREPARERS.items():
        t = first_common_defined(frames, fn)
        defined_at.append((name, None if t is None else str(t)))
    starts = [first_common_defined(frames, fn) for fn in PREPARERS.values()]
    first_signals = {name: first_signal_time(frames, fn) for name, fn in PREPARERS.items()}
    if any(t is None for t in first_signals.values()):
        common_strategy_start = None
    else:
        # first hour at which every strategy has already been able to emit a signal
        common_strategy_start = max(first_signals.values()) + pd.Timedelta(hours=1)

    windows = {
        "reused_exploratory": {
            "label": "REUSED_EXPLORATORY_NOT_HOLDOUT",
            "start": max(REUSED_START, common[0] + pd.Timedelta(hours=1)),
            "end": reused_end,
        }
    }
    if earlier_bars and common_strategy_start is not None and common_strategy_start < EARLIER_END:
        windows["precommit_earlier"] = {
            "label": "PRECOMMIT_EARLIER_ONE_SHOT",
            "start": max(common_strategy_start, earlier_bars[0] + pd.Timedelta(hours=1)),
            "end": earlier_bars[-1],
        }

    # Replication of the published locked-cache EMA path (internal 60-bar warmup only)
    replication = None
    try:
        locked_frames, locked_specs, locked_funding, locked_prov = load_cache(locked_dir)
        locked_common = sorted(set.intersection(*(set(d.index) for d in locked_frames.values())))
        if len(locked_common) <= 60:
            raise ValueError("insufficient common locked-cache bars")
        locked_intervals = {k: v["funding_interval_hours"] for k, v in locked_prov["symbols"].items()}
        locked_cov = funding_complete_between(locked_funding, locked_intervals, locked_common[60], locked_common[-1])
        if not locked_cov["complete"]:
            raise ValueError("locked-cache funding incomplete or invalid; replication withheld")
        from trading_lab.perps.research import simulate as legacy_simulate
        legacy = legacy_simulate(locked_frames, locked_specs, locked_funding)
        prepared = {k: attach_ema(v) for k, v in locked_frames.items()}
        book = simulate_book(prepared, locked_specs, locked_funding, prepared=True, halt_mode="legacy_cash",
                             funding_intervals=locked_intervals)
        replication = {
            "legacy_return_pct": legacy["return_pct"],
            "book_return_pct": book["return_pct"],
            "legacy_closed_trades": legacy["closed_trades"],
            "book_closed_trades": book["closed_trades"],
            "returns_match": abs(legacy["return_pct"] - book["return_pct"]) < 1e-8,
            "trades_match": legacy["closed_trades"] == book["closed_trades"],
        }
    except Exception as exc:
        replication = {"error": f"{type(exc).__name__}: {exc}"}

    results = {}
    for window_name, window in windows.items():
        intervals = {k: v["funding_interval_hours"] for k, v in prov["symbols"].items()}
        fund_cov = funding_complete_between(funding, intervals, window["start"], window["end"])
        results[window_name] = {
            "window": {**window, "start": str(window["start"]), "end": str(window["end"])},
            "funding_coverage": fund_cov,
            "candidates": {},
        }
        if not fund_cov["complete"]:
            results[window_name]["status"] = "funding_incomplete_or_invalid_returns_withheld"
            continue
        results[window_name]["status"] = "exploratory_complete_funding_under_current_schedule_assumption"
        for name, fn in PREPARERS.items():
            prepared = {k: fn(v) for k, v in frames.items()}
            raw = simulate_book(
                prepared, specs, funding, prepared=True,
                start=window["start"], end=window["end"], funding_intervals=intervals,
            )
            trades = raw.pop("trades")
            raw.pop("nav", None)
            results[window_name]["candidates"][name] = {
                **summarize({**raw, "trades": trades}),
                "citation": CITATIONS[name],
            }

    report = {
        "experiment": "fixed_ex_ante_strategy_comparison",
        "not_a_multi_agent_or_llm_historical_test": True,
        "independent_llm_review": "NOT_WIRED",
        "held_out": False,
        "already_seen_reused_window": True,
        "run_recorded_at": run_recorded_at.isoformat(),
        "citations": CITATIONS,
        "assumptions": {
            "virtual_usdt": CAPITAL, "leverage_example": LEVERAGE, "risk_budget": 0.01,
            "max_margin_fraction": 0.5, "cash_floor": 0.2, "max_positions": 1,
            "slippage_each_side": SLIPPAGE, "taker": 0.0006,
            "funding": "historical rate * contemporaneous open as mark proxy",
            "daily_loss_halt": 0.03, "total_loss_halt": 0.10,
            "liquidation": "approximate_stress_not_venue_verified",
        },
        "data": {
            "extended_cache": str(data_dir),
            "provenance": prov,
            "first_common": str(common[0]),
            "last_common": str(common[-1]),
            "warmup_defined_at": defined_at,
            "common_strategy_start": None if common_strategy_start is None else str(common_strategy_start),
            "first_signals": {k: None if v is None else str(v) for k, v in first_signals.items()},
        },
        "multi_agent_status": {
            "langgraph_roles": "researcher -> critic -> synthesizer exist as on-demand roles",
            "official_api": "not configured; model mode returns needs_model_configuration",
            "chat_assisted_path": "manual / chat-assisted only; cannot be replayed historically without hindsight",
            "fixture_mode": "deterministic demo labels, not market research",
            "why_not_evaluated_historically": (
                "No contemporaneous agent outputs were saved on these past bars. Using a current "
                "chat model now to label past hours would use later knowledge and is not a test of "
                "predictive multi-agent research."
            ),
        },
        "ema_locked_cache_replication": replication,
        "windows": results,
        "limitations": [
            "Reused 60-day window is exploratory and already seen; earlier window is short and one-shot, not a certified holdout.",
            "12-month TSMOM and a 100-day SMA filter are not identifiable on common BTC/XAU/NDX100 public history.",
            "Trade counts are small; win rate is a closed-fill count, not a calibrated predictive accuracy.",
            "More complex or cited rules are not guaranteed to beat the EMA baseline.",
            "Min-order, loss-halt, and unverified liquidation can dominate 30U results, especially in XAU.",
            "Daily Turtle/TSMOM mapped onto 1h fills are adaptations, not a tick-for-tick recreation of the original systems.",
        ],
    }
    return report


def write_report(report: dict, out_json: Path) -> Path:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2, default=str))
    lines = [
        "FIXED STRATEGY COMPARISON — sandbox only, not live, not a multi-agent historical test",
        f"run_recorded_at={report['run_recorded_at']}  held_out={report['held_out']}",
        f"independent_llm_review={report['independent_llm_review']}",
        "",
    ]
    for wname, w in report["windows"].items():
        win = w["window"]
        lines.append(f"== {wname}  {win['label']}  {win['start']} -> {win['end']}  funding_complete={w.get('funding_coverage',{}).get('complete')} ==")
        lines.append(f"status={w.get('status', 'unknown')}")
        for name, c in w["candidates"].items():
            wr = c["winrate_pct"]
            wr_s = "n/a" if wr is None else f"{wr:.1f}% ({c['wins']}/{c['closed_trades']})"
            lines.append(
                f"{name}: {c['initial']:.2f}U -> {c['final']:.2f}U  return {c['return_pct']:.2f}%  "
                f"maxDD {c['max_drawdown_pct']:.2f}%  winrate {wr_s}  "
                f"fees {c['fees']:.4f}  funding {c['funding_net_paid']:.4f}  "
                f"rejections {c['rejections']}  loss_halt={c['loss_halt']}  open={c['open_positions']}"
            )
        lines.append("")
    lines.append("EMA locked-cache replication: " + json.dumps(report.get("ema_locked_cache_replication"), default=str))
    lines.append("12-month TSMOM and 100-day trend filter were not used: common listing history is too short.")
    lines.append("Multi-agent research was not historically evaluated: " + report["multi_agent_status"]["why_not_evaluated_historically"])
    out_txt = out_json.with_suffix(".txt")
    out_txt.write_text("\n".join(lines) + "\n")
    return out_txt


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action", choices=["fetch", "run"], nargs="?", default="run")
    ap.add_argument("--data-dir", default=str(EXT_CACHE))
    ap.add_argument("--out", default=str(ART / "strategy_compare.json"))
    args = ap.parse_args(argv)
    data_dir = Path(args.data_dir)
    if args.action == "fetch":
        prov = fetch_extended(cache=data_dir)
        print(json.dumps(prov, indent=2, default=str))
        return 0
    report = run_comparison(data_dir)
    txt = write_report(report, Path(args.out))
    print(txt.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
