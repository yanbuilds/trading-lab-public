"""Compare a frozen range mean-reversion candidate with the 1h EMA trend book.

Primary capital is 100 USDT. An optional 30 USDT book is reported separately
and is never mixed with the 100 USDT path. Net results require complete
funding under current interval assumptions. Exploratory, already-seen data
is not a held-out test. No live or demo orders.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from experiments.portfolio_cost_stress import portfolio_manifest
from trading_lab.perps.compare import attach_ema, funding_complete_between, load_cache, simulate_book, summarize
from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.regime import load_manifest

MANIFEST = Path("config/piia_regime_mean_reversion_v1.yaml")
EXPECTED = {"BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT"}
# Already viewed in earlier research; not a holdout.
DEFAULT_START = "2026-08-19T00:00:00Z"
DEFAULT_END = "2026-11-16T06:00:00Z"


def complete_manifest(data_dir: Path) -> dict:
    """Identify each source/input byte; hashes do not prove market data truth."""
    result = portfolio_manifest(data_dir)
    for path in (Path(__file__), MANIFEST, Path("src/trading_lab/perps/regime.py"),
                 Path("src/trading_lab/perps/mean_reversion.py")):
        if not path.is_file():
            raise ValueError(f"required experiment source missing: {path}")
        result["files"][str(path)] = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return result


def run_books(frames, specs, funding, intervals, start, end, capital, *, max_hold_hours=None, preparer=None):
    return simulate_book(
        frames, specs, funding, capital=capital, leverage=2,
        start=start, end=end - pd.Timedelta(hours=1), slippage=0.0005,
        preparer=preparer, halt_mode="equity", funding_intervals=intervals,
        max_hold_hours=max_hold_hours,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data/perps-extended"))
    parser.add_argument("--start", default=DEFAULT_START)
    parser.add_argument("--end", default=DEFAULT_END, help="exclusive UTC hour")
    parser.add_argument("--out", type=Path, default=Path("artifacts/regime_mean_reversion_v1.json"))
    parser.add_argument("--include-optional-30u", action="store_true")
    args = parser.parse_args()
    manifest = load_manifest(MANIFEST)
    if manifest["hypothesis_optimized"] or manifest["held_out"] or manifest["capital_primary_usdt"] != 100:
        raise ValueError("frozen manifest changed unexpectedly")
    start, end = pd.Timestamp(args.start), pd.Timestamp(args.end)
    if start.tzinfo is None or end.tzinfo is None or start >= end:
        raise ValueError("ordered timezone-aware timestamps required")
    start, end = start.tz_convert("UTC"), end.tz_convert("UTC")
    input_hashes = complete_manifest(args.data_dir)
    frames, specs, funding, provenance = load_cache(args.data_dir)
    if set(frames) != EXPECTED or set(funding) != EXPECTED:
        raise ValueError("BTC, XAU and NDX100 bars and funding are required")
    intervals = {s: provenance["symbols"][s]["funding_interval_hours"] for s in EXPECTED}
    coverage = funding_complete_between(funding, intervals, start, end - pd.Timedelta(hours=1))
    file_bytes = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    config_hash = hashlib.sha256(MANIFEST.read_bytes()).hexdigest()
    report = {
        "label": "exploratory already-seen comparison; not held-out; not a live strategy",
        "hypothesis_id": manifest["hypothesis_id"],
        "hypothesis_optimized": False,
        "held_out": False,
        "start_utc": str(start),
        "end_exclusive_utc": str(end),
        "capital_primary_usdt": 100,
        "manifest_sha256": config_hash,
        "script_sha256": file_bytes,
        "input_manifest": input_hashes,
        "funding_coverage_under_current_intervals": coverage,
        "deterministic_experiment": True,
        "not_a_piia_agent": True,
        "limitations": [
            "Parameters were frozen before this run and were not grid-searched here.",
            "Interval was previously inspected and is not out-of-sample.",
            "Open positions are marked at the last close without assumed exit costs.",
            "Historical funding schedules, contract specs and liquidation remain unverified.",
            "Missing or invalid funding forbids net performance figures.",
            "Macro snapshots are not applied to this historical book and cannot support macro-enhanced alpha.",
        ],
    }
    if not coverage["complete"]:
        report["net_performance"] = "unavailable"
        report["reason"] = "incomplete or invalid funding under assumed intervals"
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(json.dumps({"net_performance": "unavailable", "reason": report["reason"]}, indent=2))
        return
    books = {}
    for name, fn, hold in (
        ("ema_cross_1h_trend_baseline", attach_ema, None),
        ("range_bollinger_mean_reversion_v1", attach_range_bollinger, int(manifest["time_stop_hours"])),
    ):
        raw = run_books(frames, specs, funding, intervals, start, end, 100,
                        max_hold_hours=hold, preparer=fn)
        trades = raw.pop("trades")
        raw.pop("nav", None)
        books[name] = {**summarize({**raw, "trades": trades}), "max_hold_hours": hold,
                       "capital_usdt": 100}
    report["primary_100u"] = books
    if args.include_optional_30u:
        optional = {}
        for name, fn, hold in (
            ("ema_cross_1h_trend_baseline", attach_ema, None),
            ("range_bollinger_mean_reversion_v1", attach_range_bollinger, int(manifest["time_stop_hours"])),
        ):
            raw = run_books(frames, specs, funding, intervals, start, end, 30,
                            max_hold_hours=hold, preparer=fn)
            trades = raw.pop("trades")
            raw.pop("nav", None)
            optional[name] = {**summarize({**raw, "trades": trades}), "max_hold_hours": hold,
                              "capital_usdt": 30}
        report["optional_30u_separate_book"] = optional
        report["capital_books_mixed"] = False
    if complete_manifest(args.data_dir) != input_hashes:
        raise ValueError("input, manifest or experiment code changed during the run")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps({k: report[k] for k in ("label", "primary_100u", "optional_30u_separate_book") if k in report},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
