"""Causal Rolling Walk-Forward Evaluator for Perpetual Contract Strategies (v2).

Runs strictly chronological out-of-window evaluation folds with causal indicator
warmup from preceding bars only (never truncating warmup, never peeking into
future folds), plus cost-stress comparison (1x vs 2x slippage). Works without
requiring C-extension `TA-Lib`.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from trading_lab.perps.adaptive_v2 import (
    attach_adaptive_v2,
    attach_ema_v2,
    simulate_adaptive_v2,
)
from trading_lab.perps.risk import exit_pnl, size_order, stop_fill
from trading_lab.perps.risk_v2 import load_policy_v2, resolve_spec


def _simulate_simple_signal_book(
    frames: dict[str, pd.DataFrame],
    *,
    capital: float,
    leverage: float,
    risk_fraction: float,
    slippage: float,
    start: pd.Timestamp,
    end: pd.Timestamp,
    funding: dict[str, list[dict[str, Any]]] | None = None,
) -> dict[str, Any]:
    """Pure-Python baseline simulator for `attach_ema_v2` frames."""
    common = sorted(set.intersection(*(set(d.index) for d in frames.values())))
    specs = {k: resolve_spec(k).to_v1_spec() for k in frames}
    fmaps = {
        k: {pd.Timestamp(int(r["timestamp"]), unit="ms", tz="UTC"): float(r["rate"]) for r in rs}
        for k, rs in (funding or {}).items()
    }
    cash = float(capital)
    pos = None
    closed = []
    nav = []
    fees = 0.0
    fundcost = 0.0
    kill = False
    day = None
    day_start = cash

    def _close(px: float, ts: pd.Timestamp, reason: str) -> None:
        nonlocal cash, pos, fees
        assert pos is not None
        sp = specs[pos["symbol"]]
        side = pos["side"]
        qty = pos["qty"]
        fill = px * (1.0 - side * slippage)
        efee = qty * sp.contract_size * fill * sp.taker
        gross = exit_pnl(side, qty, pos["entry"], fill, sp)
        cash += gross - efee
        fees += efee
        closed.append({"symbol": pos["symbol"], "side": side, "net_pnl": gross - pos["entry_fee"] - efee - pos["funding"], "exit_reason": reason})
        pos = None

    for ts in common:
        if ts < start:
            continue
        if ts > end:
            break
        if ts.date() != day:
            day = ts.date()
            day_start = cash + (
                exit_pnl(pos["side"], pos["qty"], pos["entry"], float(frames[pos["symbol"]].loc[ts, "open"]), specs[pos["symbol"]])
                if pos else 0.0
            )
        rows = {k: d.loc[ts] for k, d in frames.items()}
        if pos is not None:
            k = pos["symbol"]
            sp = specs[k]
            rate = fmaps.get(k, {}).get(ts, 0.0)
            if rate:
                amt = pos["side"] * pos["qty"] * sp.contract_size * float(rows[k]["open"]) * rate
                cash -= amt
                fundcost += amt
                pos["funding"] += amt
            prev = frames[k].loc[ts - pd.Timedelta(hours=1)]
            gap = (pos["side"] == 1 and float(rows[k]["open"]) <= pos["stop"]) or (pos["side"] == -1 and float(rows[k]["open"]) >= pos["stop"])
            if gap:
                _close(float(rows[k]["open"]), ts, "gap_stop")
            elif (pos["side"] == 1 and bool(prev["exit_long"])) or (pos["side"] == -1 and bool(prev["exit_short"])):
                _close(float(rows[k]["open"]), ts, "reverse_signal")
        eq_open = cash + (
            exit_pnl(pos["side"], pos["qty"], pos["entry"], float(rows[pos["symbol"]]["open"]), specs[pos["symbol"]])
            if pos else 0.0
        )
        if eq_open <= day_start * 0.97 or eq_open <= capital * 0.88:
            kill = True
        if pos is None and not kill:
            for k in sorted(frames):
                prev = frames[k].loc[ts - pd.Timedelta(hours=1)]
                side = int(prev["signal"])
                if not side or not math.isfinite(float(prev["atr"])):
                    continue
                fill = float(rows[k]["open"]) * (1.0 + side * slippage)
                dist = 2.0 * float(prev["atr"])
                sp = specs[k]
                qty, _ = size_order(cash, cash, fill, dist, sp, leverage=min(3.0, max(1.0, leverage)), risk_fraction=min(0.02, risk_fraction), slippage=slippage)
                if qty <= 0:
                    continue
                fee = qty * sp.contract_size * fill * sp.taker
                margin = qty * sp.contract_size * fill / leverage
                if margin + fee > cash:
                    continue
                cash -= fee
                fees += fee
                pos = {"symbol": k, "side": side, "qty": qty, "entry": fill, "stop": fill - side * dist, "entry_fee": fee, "funding": 0.0}
                break
        if pos is not None:
            k = pos["symbol"]
            r = rows[k]
            hit = stop_fill(pos["side"], pos["stop"], float(r["open"]), float(r["low"]), float(r["high"]))
            if hit is not None:
                _close(hit, ts + pd.Timedelta(hours=1), "atr_stop")
        eq_close = cash + (
            exit_pnl(pos["side"], pos["qty"], pos["entry"], float(rows[pos["symbol"]]["close"]), specs[pos["symbol"]])
            if pos else 0.0
        )
        nav.append((ts + pd.Timedelta(hours=1), eq_close))
        if eq_close <= day_start * 0.97 or eq_close <= capital * 0.88:
            kill = True
        if kill and pos is not None:
            _close(float(rows[pos["symbol"]]["close"]), ts + pd.Timedelta(hours=1), "loss_halt")
            nav[-1] = (nav[-1][0], cash)

    eq = pd.Series([capital] + [v for _, v in nav], dtype=float)
    pnls = [t["net_pnl"] for t in closed]
    wins = [p for p in pnls if p > 0.01]
    return {
        "initial": capital,
        "final": float(eq.iloc[-1]),
        "return_pct": float((eq.iloc[-1] / capital - 1.0) * 100.0),
        "max_drawdown_pct": float((1.0 - eq / eq.cummax()).max() * 100.0) if len(eq) else 0.0,
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": int(sum(p < -0.01 for p in pnls)),
        "winrate_pct": (100.0 * len(wins) / len(pnls)) if pnls else None,
        "fees": round(fees, 4),
        "funding_net_paid": round(fundcost, 4),
        "loss_halt": bool(kill),
    }


def run_walk_forward(
    frames: dict[str, pd.DataFrame],
    *,
    warmup_bars: int = 240,
    fold_bars: int = 360,
    step_bars: int | None = None,
    capital: float = 500.0,
    leverage: float = 2.0,
    risk_fraction: float = 0.01,
    slippage: float = 0.0005,
    funding: dict[str, list[dict[str, Any]]] | None = None,
    policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run strictly causal walk-forward folds comparing `adaptive_v2` vs `ema_1h_v2`.

    Each fold slices `frames` up to `fold_end` BEFORE computing indicators so future
    bars can never leak into earlier folds.
    """
    if warmup_bars < 70 or fold_bars < 24:
        raise ValueError("warmup_bars must be >= 70 and fold_bars >= 24")
    step = step_bars or fold_bars
    if step < 24:
        raise ValueError("step_bars must be >= 24")

    common = sorted(set.intersection(*(set(d.index) for d in frames.values())))
    if len(common) < warmup_bars + fold_bars:
        raise ValueError(f"Need at least {warmup_bars + fold_bars} common hourly bars, got {len(common)}")

    cfg = policy or load_policy_v2()
    folds: list[dict[str, Any]] = []
    idx = warmup_bars
    fold_num = 1
    while idx + fold_bars <= len(common):
        fold_start = common[idx]
        fold_end = common[idx + fold_bars - 1]
        # Strictly causal prefix slice up to `fold_end` inclusive!
        prefix_frames = {k: d.loc[d.index <= fold_end].copy() for k, d in frames.items()}

        v2_prep = {k: attach_adaptive_v2(df, policy=cfg) for k, df in prefix_frames.items()}
        ema_prep = {k: attach_ema_v2(df) for k, df in prefix_frames.items()}

        v2_res = simulate_adaptive_v2(
            v2_prep,
            funding=funding,
            capital=capital,
            leverage=leverage,
            risk_fraction=risk_fraction,
            slippage=slippage,
            start=fold_start,
            end=fold_end,
            policy=cfg,
            prepared=True,
        )
        v2_stress = simulate_adaptive_v2(
            v2_prep,
            funding=funding,
            capital=capital,
            leverage=leverage,
            risk_fraction=risk_fraction,
            slippage=slippage * 2.0,
            start=fold_start,
            end=fold_end,
            policy=cfg,
            prepared=True,
        )
        ema_res = _simulate_simple_signal_book(
            ema_prep,
            capital=capital,
            leverage=leverage,
            risk_fraction=risk_fraction,
            slippage=slippage,
            start=fold_start,
            end=fold_end,
            funding=funding,
        )
        # Strip full trade/nav arrays in fold summary
        v2_summary = {k: v for k, v in v2_res.items() if k not in {"trades", "nav"}}
        v2_stress_summary = {k: v for k, v in v2_stress.items() if k not in {"trades", "nav"}}

        folds.append({
            "fold": fold_num,
            "start_utc": str(fold_start),
            "end_utc": str(fold_end),
            "bars": fold_bars,
            "adaptive_v2": v2_summary,
            "adaptive_v2_2x_slippage": v2_stress_summary,
            "ema_1h_baseline": ema_res,
            "cash_reference": {"initial": capital, "final": capital, "return_pct": 0.0, "max_drawdown_pct": 0.0},
        })
        idx += step
        fold_num += 1

    v2_rets = [f["adaptive_v2"]["return_pct"] for f in folds]
    ema_rets = [f["ema_1h_baseline"]["return_pct"] for f in folds]
    return {
        "kind": "causal_rolling_walk_forward_v2",
        "folds_count": len(folds),
        "warmup_bars": warmup_bars,
        "fold_bars": fold_bars,
        "step_bars": step,
        "non_overlapping": step >= fold_bars,
        "capital_reset_each_fold": capital,
        "summary": {
            "adaptive_v2_mean_fold_return_pct": round(float(np.mean(v2_rets)), 4) if v2_rets else 0.0,
            "adaptive_v2_median_fold_return_pct": round(float(np.median(v2_rets)), 4) if v2_rets else 0.0,
            "adaptive_v2_positive_folds": int(sum(r > 0 for r in v2_rets)),
            "adaptive_v2_worst_fold_drawdown_pct": round(max((f["adaptive_v2"]["max_drawdown_pct"] for f in folds), default=0.0), 4),
            "ema_1h_mean_fold_return_pct": round(float(np.mean(ema_rets)), 4) if ema_rets else 0.0,
            "ema_1h_positive_folds": int(sum(r > 0 for r in ema_rets)),
        },
        "folds": folds,
        "limitations": [
            "Each fold resets virtual equity to `capital_reset_each_fold`; fold returns are not compounded unless non-overlapping and reinvested.",
            "No hyperparameter optimization is performed across folds; rules come from `trading_policy_v2.yaml`.",
        ],
    }
