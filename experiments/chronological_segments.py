"""Fixed-rule, reset-capital chronological diagnostics, not walk-forward training.

Runs fixed existing strategies in sequential 30-day windows, plus the final
shorter remainder. No tuning or parameter selection. The previously inspected
data is exploratory, and the reset windows cannot be compounded as a live path.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from trading_lab.perps.aligned_candidate import attach_aligned_ema
from trading_lab.perps.compare import attach_ema, attach_tsmom_3m, funding_complete_between, load_cache, simulate_book, summarize
from experiments.breakout_candidate import attach_breakout_with_4h_trend
from experiments.portfolio_cost_stress import portfolio_manifest


def windows(start: pd.Timestamp, end: pd.Timestamp) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("ordered timezone-aware boundaries required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor("h") or end != end.floor("h"):
        raise ValueError("boundaries must align to the hour")
    result = []
    while start < end:
        next_end = min(start + pd.Timedelta(days=30), end)
        result.append((start, next_end))
        start = next_end
    return result


def audit_segments(frames, specs, funding, intervals, start, end, *, exclude_funding=False):
    expected = {"BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT"}
    if set(frames) != expected or set(specs) != expected or set(funding) != expected or set(intervals) != expected:
        raise ValueError("all three contracts, specs and funding schedules are required")
    for symbol, bars in frames.items():
        if bars.index.min() > start - pd.Timedelta(hours=200) or bars.index.max() < end - pd.Timedelta(hours=1):
            raise ValueError(f"missing warmup or price coverage for {symbol}")
    coverage = None if exclude_funding else funding_complete_between(funding, intervals, start, end - pd.Timedelta(hours=1))
    if coverage is not None and not coverage["complete"]:
        raise ValueError("funding missing under current interval assumption")
    report = {"label": "exploratory repeated data; no training, optimization or out-of-sample claim",
              "capital_usdt_per_segment": 100, "reset_capital_and_flat_start_each_segment": True,
              "funding_excluded_unknown": exclude_funding,
              "funding_coverage_under_current_intervals": coverage,
              "limits": ["end equity includes unrealized positions marked to last close with no hypothetical closing fee",
                         "segment positions are discarded when next segment resets; results cannot be compounded",
                         "historical funding interval and contract specifications are unverified",
                         "3-month momentum may be unavailable where individual contract history is insufficient",
                         "breakout indicators can be unavailable early when daily warmup is insufficient",
                         "simultaneous entries use fixed symbol sort priority, not exposure-matched allocation",
                         "funding excluded and unknown" if exclude_funding else "funding uses hourly open as mark proxy; liquidation model is unverified",
                         "exact liquidation is unverified",
                         "reused data, not independently held out"], "segments": []}
    for left, right in windows(start, end):
        segment = {"start_utc": str(left), "end_exclusive_utc": str(right),
                   "duration_hours": int((right-left)/pd.Timedelta(hours=1)), "strategies": {}}
        for name, prepare in (("baseline", attach_ema), ("aligned_4h_candidate", attach_aligned_ema),
                              ("tsmom_3m_corrected", attach_tsmom_3m),
                              ("breakout_4h_aligned", attach_breakout_with_4h_trend)):
            # Remove all post-window bars before feature construction.
            prepared = {s: prepare(bars.loc[bars.index < right]) for s, bars in frames.items()}
            result = simulate_book(prepared, specs, None if exclude_funding else funding, capital=100, start=left,
                                   end=right-pd.Timedelta(hours=1), prepared=True,
                                   funding_intervals=None if exclude_funding else intervals)
            trades = result.pop("trades")
            result.pop("nav", None)
            segment["strategies"][name] = summarize({**result, "trades": trades})
        report["segments"].append(segment)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    ap.add_argument("--start", required=True)
    ap.add_argument("--end", required=True, help="exclusive UTC timestamp")
    ap.add_argument("--out", type=Path, default=Path("artifacts/chronological_segments.json"))
    ap.add_argument("--exclude-funding", action="store_true", help="explicit price-only diagnostic; funding remains unknown")
    args = ap.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    # Validate before doing any data-dependent work.
    windows(start, end)
    manifest = portfolio_manifest(args.data_dir)
    script_hash = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    candidate_path = Path(__file__).with_name("breakout_candidate.py")
    candidate_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    frames, specs, funding, prov = load_cache(args.data_dir)
    intervals = {s: entry["funding_interval_hours"] for s, entry in prov["symbols"].items()}
    report = audit_segments(frames, specs, funding, intervals, start.tz_convert("UTC"), end.tz_convert("UTC"),
                            exclude_funding=args.exclude_funding)
    if (portfolio_manifest(args.data_dir) != manifest or hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_hash
            or hashlib.sha256(candidate_path.read_bytes()).hexdigest() != candidate_hash):
        raise ValueError("input or code changed during run; report withheld")
    report["input_manifest"] = manifest
    report["script_sha256"] = script_hash
    report["breakout_candidate_sha256"] = candidate_hash
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
