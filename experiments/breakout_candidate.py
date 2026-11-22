"""BTC price-only diagnostics for a fixed 4h-trend / daily-breakout candidate.

Funding is unknown and excluded; outputs are not net returns or a trading signal.
Previously viewed 2026 data is exploratory, not a held-out evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from trading_lab.perps.aligned_candidate import attach_aligned_ema
from trading_lab.perps.aligned_compare import input_manifest
from trading_lab.perps.compare import SLIPPAGE, attach_ema, attach_turtle_s1, load_cache, simulate_book, summarize
from trading_lab.perps.risk import floor_step

SYMBOL = "BTC/USDT:USDT"


def long_hold_reference(frame: pd.DataFrame, spec, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    """Illustrative 1x long with entry/exit friction; not matched to strategy risk."""
    entry = float(frame.loc[start, "open"]) * (1 + SLIPPAGE)
    exit_price = float(frame.loc[end - pd.Timedelta(hours=1), "close"]) * (1 - SLIPPAGE)
    qty = floor_step(100 / (entry * spec.contract_size * (1 + spec.taker)), spec.qty_step)
    if qty < spec.min_qty or qty * entry * spec.contract_size < spec.min_notional:
        return {"available": False, "reason": "minimum_order_exceeds_budget"}
    cash = 100 - qty * entry * spec.contract_size * (1 + spec.taker)
    final = cash + qty * spec.contract_size * exit_price * (1 - spec.taker)
    return {"available": True, "initial": 100, "final": final, "return_pct": final - 100,
            "exposure": "one-times long held throughout; full-budget entry, unlike strategy cash and risk limits",
            "funding": "excluded_unknown", "terminal_position": "closed with assumed exit costs",
            "note": "Not risk-matched; isolated liquidation and real fills not modelled."}


def attach_breakout_with_4h_trend(frame: pd.DataFrame) -> pd.DataFrame:
    """20-day breakout only when the last completed 4h EMA20/50 agrees.

    Existing Turtle 10-day exits and 2*ATR(20) stop remain unchanged. Both
    breakout and trend are known only after the 1h signal bar closes.
    """
    result = attach_turtle_s1(frame)
    trend = attach_aligned_ema(frame)
    fast, slow = trend["trend_4h_fast"], trend["trend_4h_slow"]
    agree = ((result["signal"] == 1) & (fast > slow)) | ((result["signal"] == -1) & (fast < slow))
    result["signal"] = result["signal"].where(agree, 0)
    result["trend_4h_fast"], result["trend_4h_slow"] = fast, slow
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    parser.add_argument("--bars", type=Path, default=Path("data/perps-extended/BTCUSDT-year.csv"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive UTC hour")
    parser.add_argument("--out", type=Path, default=Path("artifacts/btc_breakout_price_only.json"))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if (start.tzinfo is None or end.tzinfo is None or start >= end
            or start != start.floor("h") or end != end.floor("h")):
        raise ValueError("ordered timezone-aware hourly boundaries required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    manifest = input_manifest(args.bars, args.data_dir)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    _, specs, _, _ = load_cache(args.data_dir)
    frame = pd.read_csv(args.bars, parse_dates=["date"]).set_index("date")
    frame.index = pd.to_datetime(frame.index, utc=True)
    if frame.index.min() > start - pd.Timedelta(days=25) or frame.index.max() < end - pd.Timedelta(hours=1):
        raise ValueError("insufficient warmup or price coverage")
    frame = frame.loc[frame.index < end]
    report = {"label": "exploratory BTC price-only diagnostic on previously viewed data; funding excluded and unknown",
              "start_utc": str(start), "end_exclusive_utc": str(end), "capital_usdt": 100,
              "model": "offline stress harness, not Freqtrade or exchange matching",
              "input_manifest": manifest, "script_sha256": script_hash, "strategies": {},
              "full_calendar_year": bool(start == pd.Timestamp(year=start.year, month=1, day=1, tz="UTC")
                                         and end == start + pd.DateOffset(years=1)),
              "limitations": ["not net of funding; do not treat as net performance",
                              "no out-of-sample test; candidate and interval previously inspected",
                              "historical contract specs and funding schedule unverified",
                              "end equity includes open positions without hypothetical exit costs",
                              "no exact mark-price liquidation or exchange execution model"]}
    report["long_hold_reference"] = long_hold_reference(frame, specs[SYMBOL], start, end)
    for name, fn in (("ema_1h", attach_ema), ("ema_4h_aligned", attach_aligned_ema),
                     ("turtle_breakout", attach_turtle_s1), ("breakout_4h_aligned", attach_breakout_with_4h_trend)):
        prepared = fn(frame)
        result = simulate_book({SYMBOL: prepared}, {SYMBOL: specs[SYMBOL]}, capital=100,
                               start=start, end=end-pd.Timedelta(hours=1), prepared=True)
        trades = result.pop("trades")
        result.pop("nav", None)
        report["strategies"][name] = summarize({**result, "trades": trades})
    if input_manifest(args.bars, args.data_dir) != manifest or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash:
        raise ValueError("data or code changed during run; report withheld")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
