"""Compare fixed research strategies under two cost assumptions, no orders.

The fee and slippage scenarios are sensitivity checks, not fitted parameters.
The funding schedule is checked against the current interval only; historical
schedule changes, mark-price liquidation, and historical specs are unverified.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pandas as pd

from trading_lab.perps.aligned_candidate import attach_aligned_ema
from trading_lab.perps.aligned_compare import input_manifest
from trading_lab.perps.compare import (
    attach_ema, attach_tsmom_3m, funding_complete_between, load_cache, simulate_book, summarize,
)


def portfolio_manifest(data_dir: Path) -> dict:
    manifest = input_manifest(data_dir / "BTCUSDT.csv", data_dir)
    for symbol in ("XAUUSDT", "NDX100USDT"):
        for suffix in (".csv", "-funding.csv"):
            path = data_dir / (symbol + suffix)
            manifest["files"][symbol + suffix] = ({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
                                                   if path.is_file() else None)
    source = Path(__file__).resolve()
    manifest["files"]["portfolio_cost_stress.py"] = {"path": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    parser.add_argument("--start", required=True, help="inclusive UTC timestamp")
    parser.add_argument("--end", required=True, help="exclusive UTC timestamp")
    parser.add_argument("--out", type=Path, default=Path("artifacts/portfolio_cost_stress.json"))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        parser.error("ordered timezone-aware timestamps required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor("h") or end != end.floor("h"):
        parser.error("timestamps must fall on hour boundaries")
    manifest = portfolio_manifest(args.data_dir)
    frames, specs, funding, provenance = load_cache(args.data_dir)
    expected = {"BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT"}
    if set(frames) != expected or set(funding) != expected:
        raise ValueError("BTC, XAU, NDX100 data and funding are all required")
    for symbol, frame in frames.items():
        if frame.index.min() > start - pd.Timedelta(hours=200) or frame.index.max() < end - pd.Timedelta(hours=1):
            raise ValueError(f"insufficient warmup or coverage for {symbol}")
    intervals = {s: provenance["symbols"][s]["funding_interval_hours"] for s in expected}
    coverage = funding_complete_between(funding, intervals, start, end - pd.Timedelta(hours=1))
    if not coverage["complete"]:
        raise ValueError("incomplete funding under current-interval assumption")

    out = {"label": "exploratory, previously viewed interval; not a held-out or live test",
           "start_utc": str(start), "end_exclusive_utc": str(end), "initial_usdt": 100,
           "risk": "2x isolated example, one global position, 1% stop-risk budget; stop gaps can exceed budget",
           "input_manifest": manifest,
           "funding_coverage_under_current_intervals": coverage,
           "limitations": ["historical funding schedules and contract specs unverified",
                           "funding mark uses contemporaneous open proxy",
                           "historical 1h candles do not verify liquidation or real execution",
                           "base vs stress paths may have different positions and stop times",
                           "cash benchmark has no market exposure; strategies have different realized exposure",
                           "3-month momentum can be unavailable early where the contract history is too short",
                           "single shared position; simultaneous entries are tried in symbol sort order, not allocation-neutral",
                           "open positions are marked at the final close without an assumed closing fee or slippage",
                           "no independent out-of-sample or prospective validation"],
           "scenarios": {}}
    for label, fee_multiplier, slip in (("base", 1.0, .0005), ("double_cost", 2.0, .001)):
        scenario_specs = {s: replace(spec, taker=spec.taker * fee_multiplier) for s, spec in specs.items()}
        scenario = {"fee_multiplier": fee_multiplier, "slippage_each_side": slip, "strategies": {}}
        scenario["cash_benchmark"] = {"initial_usdt": 100.0, "final_usdt": 100.0,
                                      "return_pct": 0.0, "fees": 0.0, "funding_net_paid": 0.0,
                                      "mean_end_bar_notional_over_initial": 0.0,
                                      "peak_end_bar_notional_over_initial": 0.0,
                                      "market_exposure": "none", "trades": 0,
                                      "note": "Same start, end and capital; no position, not exposure matched."}
        for name, prepare in (("baseline", attach_ema), ("aligned_4h_candidate", attach_aligned_ema),
                              ("tsmom_3m_corrected", attach_tsmom_3m)):
            prepared = {s: prepare(frame) for s, frame in frames.items()}
            result = simulate_book(prepared, scenario_specs, funding, capital=100,
                                   start=start, end=end - pd.Timedelta(hours=1),
                                   slippage=slip, prepared=True, funding_intervals=intervals)
            trades = result.pop("trades")
            result.pop("nav", None)
            scenario["strategies"][name] = summarize({**result, "trades": trades})
        out["scenarios"][label] = scenario
    if portfolio_manifest(args.data_dir) != manifest:
        raise ValueError("input or code changed during run; report withheld")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    main()
