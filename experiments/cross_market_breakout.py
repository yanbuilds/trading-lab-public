"""Compare fixed strategies in separate accounts and a shared single-position book.

Same interval and starting capital. Funding is excluded and remains unknown.
Results are exploratory research, not net performance or a trading signal.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments.breakout_candidate import attach_breakout_with_4h_trend, long_hold_reference
from experiments.portfolio_cost_stress import portfolio_manifest
from trading_lab.perps.aligned_candidate import attach_aligned_ema
from trading_lab.perps.compare import attach_ema, attach_turtle_s1, load_cache, simulate_book, summarize

SYMBOLS = ("BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT")
STRATEGIES = (("ema_1h", attach_ema), ("ema_4h_aligned", attach_aligned_ema),
              ("turtle_breakout", attach_turtle_s1), ("breakout_4h_aligned", attach_breakout_with_4h_trend))


def evaluate(frames, specs, start: pd.Timestamp, end: pd.Timestamp) -> dict:
    if start.tzinfo is None or end.tzinfo is None or start >= end or start != start.floor("h") or end != end.floor("h"):
        raise ValueError("ordered timezone-aware hourly boundaries required")
    if set(frames) != set(SYMBOLS) or set(specs) != set(SYMBOLS):
        raise ValueError("all three perpetual markets and their own specifications are required")
    report = {"label": "exploratory price-only; funding excluded and unknown",
              "start_utc": str(start), "end_exclusive_utc": str(end), "initial_usdt_per_asset": 100,
              "limitations": ["separate accounts and shared book are distinct; neither reports net returns", "not a full year or held out",
                              "historical specs unverified; exact liquidation and execution not modelled",
                              "long hold reference has different risk and cash constraints",
                              "open strategy positions marked at close without assumed exit costs"],
              "markets": {}, "shared_book": {"capital_usdt": 100, "max_positions": 1,
                "entry_priority": "alphabetical when simultaneous entries compete", "strategies": {}}}
    prepared_all = {name: {} for name, _ in STRATEGIES}
    for symbol in SYMBOLS:
        frame, spec = frames[symbol], specs[symbol]
        if spec.symbol != symbol:
            raise ValueError(f"incorrect contract spec for {symbol}")
        if frame.index.min() > start - pd.Timedelta(days=25) or frame.index.max() < end - pd.Timedelta(hours=1):
            raise ValueError(f"missing warmup or price coverage for {symbol}")
        frame = frame.loc[frame.index < end]
        market = {"hold_reference": long_hold_reference(frame, spec, start, end), "strategies": {}}
        for name, fn in STRATEGIES:
            prepared = fn(frame)
            prepared_all[name][symbol] = prepared
            result = simulate_book({symbol: prepared}, {symbol: spec}, capital=100,
                                   start=start, end=end - pd.Timedelta(hours=1), prepared=True)
            trades = result.pop("trades")
            result.pop("nav", None)
            market["strategies"][name] = summarize({**result, "trades": trades})
        report["markets"][symbol] = market
    for name, _ in STRATEGIES:
        result = simulate_book(prepared_all[name], specs, capital=100,
                               start=start, end=end - pd.Timedelta(hours=1), prepared=True)
        trades = result.pop("trades")
        result.pop("nav", None)
        report["shared_book"]["strategies"][name] = summarize({**result, "trades": trades})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive UTC hour")
    parser.add_argument("--out", type=Path, default=Path("artifacts/cross_market_breakout.json"))
    args = parser.parse_args()
    manifest = portfolio_manifest(args.data_dir)
    sources = (Path(__file__), Path(__file__).with_name("breakout_candidate.py"))
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    frames, specs, _, _ = load_cache(args.data_dir)
    report = evaluate(frames, specs, pd.Timestamp(args.start), pd.Timestamp(args.end))
    if portfolio_manifest(args.data_dir) != manifest or any(hashlib.sha256(p.read_bytes()).hexdigest() != hashes[p.name] for p in sources):
        raise ValueError("input or code changed during run; report withheld")
    report["input_manifest"], report["experiment_sha256"] = manifest, hashes
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
