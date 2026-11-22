"""Exhaustive entry-order sensitivity on previously seen data, not strategy selection."""
from __future__ import annotations

import argparse
import hashlib
from itertools import permutations
import json
from pathlib import Path

import pandas as pd

from experiments.portfolio_cost_stress import portfolio_manifest
from trading_lab.perps.compare import attach_tsmom_3m, funding_complete_between, load_cache, simulate_book, summarize


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True, help="exclusive UTC hour")
    parser.add_argument("--out", type=Path, default=Path("artifacts/priority_sensitivity.json"))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if (start.tzinfo is None or end.tzinfo is None or start >= end
            or start != start.floor("h") or end != end.floor("h")):
        raise ValueError("ordered timezone-aware hourly boundaries required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    manifest = portfolio_manifest(args.data_dir)
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    frames, specs, funding, provenance = load_cache(args.data_dir)
    expected = {"BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT"}
    if set(frames) != expected or set(specs) != expected or set(funding) != expected:
        raise ValueError("all three markets and funding required")
    for symbol, frame in frames.items():
        if frame.index.min() > start - pd.Timedelta(hours=200) or frame.index.max() < end - pd.Timedelta(hours=1):
            raise ValueError(f"insufficient warmup or coverage for {symbol}")
    intervals = {s: provenance["symbols"][s]["funding_interval_hours"] for s in expected}
    coverage = funding_complete_between(funding, intervals, start, end - pd.Timedelta(hours=1))
    if not coverage["complete"]:
        raise ValueError("incomplete funding under current-interval assumption")
    prepared = {s: attach_tsmom_3m(frame.loc[frame.index < end]) for s, frame in frames.items()}
    cases = []
    for priority in permutations(sorted(expected)):
        result = simulate_book(prepared, specs, funding, capital=100, start=start,
                               end=end - pd.Timedelta(hours=1), prepared=True,
                               funding_intervals=intervals, entry_priority=priority)
        trades = result.pop("trades")
        result.pop("nav", None)
        cases.append({"entry_priority": list(priority), "result": summarize({**result, "trades": trades})})
    if portfolio_manifest(args.data_dir) != manifest or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_sha:
        raise ValueError("input or code changed during run")
    report = {"label": "exploratory priority sensitivity on previously viewed data; not an optimization or holdout",
              "start_utc": str(start), "end_exclusive_utc": str(end), "capital_usdt": 100,
              "input_manifest": manifest, "script_sha256": script_sha, "funding_coverage": coverage,
              "limitations": ["all six fixed orders reported; no winner selected", "same data repeatedly inspected",
                              "one global position with priority applied only to simultaneous eligible entries",
                              "funding schedule and historical specifications unverified; funding uses open as mark proxy",
                              "end equity includes open positions without hypothetical exit costs",
                              "exact liquidation and real execution not modelled", "not a full-year test"],
              "cases": cases}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
