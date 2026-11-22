"""Compare BTC baseline and 4h-aligned candidate using a local public-data cache.

Only exploratory research; this never places orders. Funding must be complete
unless the user explicitly chooses a price-only diagnostic.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

from trading_lab.perps.aligned_candidate import attach_aligned_ema
from trading_lab.perps.compare import (attach_ema, funding_complete_between,
                                       load_cache, simulate_book, summarize)


def input_manifest(bars: Path, data_dir: Path) -> dict:
    """Record input and implementation hashes; hashes do not establish source truth."""
    import talib
    module_dir = Path(__file__).resolve().parent
    paths = {"btc_bars": bars, "market_provenance": data_dir / "provenance.json",
             "btc_funding": data_dir / "BTCUSDT-funding.csv",
             **{name: module_dir / name for name in ("aligned_compare.py", "aligned_candidate.py", "compare.py", "research.py", "risk.py")}}
    files = {}
    for name, path in paths.items():
        files[name] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} if path.is_file() else None
    if files["btc_bars"] is None or files["market_provenance"] is None:
        raise ValueError("required bars or provenance missing")
    return {"files": files, "python": sys.version.split()[0], "pandas": pd.__version__,
            "talib": talib.__version__, "note": "Hashes identify bytes, not data truth, historical availability or exchange accuracy."}


def goal_assessment(result: dict, *, funding_excluded: bool) -> dict:
    """Report a historical target without claiming freedom from overfitting."""
    multiple = result["final"] / result["initial"]
    return {"equity_multiple": multiple, "target_multiple": 10,
            "tenfold_in_simulation": multiple >= 10,
            "funding_included": not funding_excluded,
            "overfitting_status": "unverified",
            "goal_verified": False,
            "reason": "Exploratory historical test; no independent prospective validation, and liquidation is unverified."}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    parser.add_argument("--bars", type=Path, required=True, help="BTC 1h CSV including indicator warmup")
    parser.add_argument("--start", required=True, help="inclusive UTC timestamp")
    parser.add_argument("--end", required=True, help="exclusive UTC timestamp")
    parser.add_argument("--exclude-funding", action="store_true", help="price-only diagnostic, not net performance")
    parser.add_argument("--out", type=Path, default=Path("artifacts/aligned_compare.json"))
    args = parser.parse_args()
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        parser.error("start and end must be ordered timezone-aware timestamps")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    if start != start.floor("h") or end != end.floor("h"):
        parser.error("start and end must fall on an hour boundary")
    manifest = input_manifest(args.bars, args.data_dir)
    _, specs, funding, provenance = load_cache(args.data_dir)
    symbol = "BTC/USDT:USDT"
    bars = pd.read_csv(args.bars, parse_dates=["date"]).set_index("date")
    bars.index = pd.to_datetime(bars.index, utc=True)
    if bars.index.min() > start - pd.Timedelta(hours=200) or bars.index.max() < end - pd.Timedelta(hours=1):
        raise ValueError("missing 200-hour warmup or requested price coverage")
    interval = provenance["symbols"][symbol]["funding_interval_hours"]
    coverage = funding_complete_between({symbol: funding[symbol]}, {symbol: interval}, start, end - pd.Timedelta(hours=1))
    if not args.exclude_funding and not coverage["complete"]:
        raise ValueError("funding coverage incomplete; use --exclude-funding only for a labelled price-only diagnostic")
    report = {"label": "exploratory, not held out, unverified liquidation",
              "start_utc": str(start), "end_exclusive_utc": str(end), "capital_usdt": 100,
              "funding_excluded": args.exclude_funding, "funding_coverage": coverage,
              "input_manifest": manifest, "strategies": {}}
    for name, prepare in (("baseline", attach_ema), ("aligned_4h_candidate", attach_aligned_ema)):
        result = simulate_book({symbol: prepare(bars)}, {symbol: specs[symbol]},
                               {} if args.exclude_funding else {symbol: funding[symbol]},
                               capital=100, start=start, end=end - pd.Timedelta(hours=1), prepared=True,
                               funding_intervals=None if args.exclude_funding else {symbol: interval})
        trades = result.pop("trades")
        result.pop("nav", None)
        report["strategies"][name] = {**summarize({**result, "trades": trades}),
                                      "goal_assessment": goal_assessment(result, funding_excluded=args.exclude_funding)}
    if input_manifest(args.bars, args.data_dir) != manifest:
        raise ValueError("input or code changed during run; report withheld")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
