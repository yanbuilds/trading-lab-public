"""Precommitted directional baseline vetoes on every eligible forward segment."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import gc
import json
import math
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from experiments import frozen_policy_capture as capture, segmented_forward_report_v2 as segmented
from experiments import prospective_funding_stress as stress, prospective_paper_report as report
from trading_lab.perps.compare import simulate_book, summarize

ACTIVATION = datetime(2026, 11, 18, 7, tzinfo=timezone.utc)
LOOKBACKS = {"momentum_24h": 24, "drift_168h": 168}
SOURCES = ("FORECAST_BASELINES_V1.md", "experiments/forecast_baseline_report.py")


def verify_precommitted():
    for name in SOURCES:
        disk = (capture.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        tracked = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=capture.ROOT, capture_output=True,
                                 check=True, timeout=15).stdout.replace(b"\r\n", b"\n")
        stamp = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD", "--", name], cwd=capture.ROOT,
                               capture_output=True, check=True, timeout=15).stdout.strip()
        if disk != tracked or int(stamp) >= int(ACTIVATION.timestamp()):
            raise ValueError("baseline rule changed or was committed after activation")
    segmented.verify_precommitted()


def filtered_states(records, candidate, lookback):
    if lookback not in LOOKBACKS.values():
        raise ValueError("unregistered baseline")
    filtered = {s: {} for s in capture.SYMBOLS}
    active = passed = 0
    for record in records:
        cutoff = report._timestamp(record["cutoff_utc"])
        for symbol in capture.SYMBOLS:
            state = candidate[symbol][cutoff]
            signal = state["signal"]
            bars = record["evidence"][symbol]["bars"]
            if len(bars) != 300:
                raise ValueError("incomplete baseline input")
            current, prior = float(bars[-1]["close"]), float(bars[-1-lookback]["close"])
            if not all(math.isfinite(x) and x > 0 for x in (current, prior)):
                raise ValueError("invalid baseline close")
            direction = 1 if current > prior else -1 if current < prior else 0
            active += bool(signal)
            allow = bool(signal and direction == signal)
            passed += allow
            filtered[symbol][cutoff] = {**state, "signal": signal if allow else 0}
    return filtered, {"candidate_signals": active, "direction_agreements": passed}


def add_baselines(records, ledger: Path):
    segments, _ = segmented.partition(records)
    result = segmented.build_report(records, ledger)
    for part, section in zip(segments, result["segments"], strict=True):
        if section["status"] != "price_only_local_diagnostic":
            section["simple_forecast_baselines"] = {"status": "unavailable_for_segment", "results": None}
            continue
        rows = [record for _, record in part]
        try:
            frames, states, first, last = report.prepare_records(rows)
            specs = report.recorded_specs(rows)
            arms = {}
            for name, lookback in LOOKBACKS.items():
                filtered, counts = filtered_states(rows, states["candidate"], lookback)
                if not counts["candidate_signals"]:
                    arms[name] = {"status": "no_candidate_signals", "counts": counts, "result": None}
                    continue
                raw = simulate_book(stress._prepared(frames, filtered), specs, funding=None, capital=100,
                                    start=first + timedelta(hours=1), end=last - timedelta(hours=1),
                                    prepared=True, max_hold_hours=24)
                trades = raw.pop("trades")
                raw.pop("nav", None)
                summary = summarize({**raw, "trades": trades})
                if summary["funding_net_paid"] is not None or summary["funding_coverage_checked"] is not False:
                    raise ValueError("unexpected funding status")
                arms[name] = {"status": "price_only_diagnostic", "counts": counts, "result": summary}
            section["simple_forecast_baselines"] = {"status": "computed_if_candidates", "results": arms,
                "counts_note": "Counts include the final observation, whose next-open fill is not yet available; counts are not trade counts."}
        except (ValueError, TypeError, KeyError, OSError):
            section["simple_forecast_baselines"] = {"status": "unavailable_or_invalid", "results": None}
    result["forecast_baseline_protocol"] = "FORECAST_BASELINES_V1"
    result["limitations"].append("Simple baselines are fixed directional filters, not AI or calibrated forecasts; exposures differ.")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=capture.LEDGER)
    parser.add_argument("--out", type=Path, default=capture.ROOT / "artifacts/forecast_baseline_report.json")
    args = parser.parse_args()
    if datetime.now(timezone.utc) < ACTIVATION:
        raise ValueError("baseline protocol not active")
    verify_precommitted()
    capture.verify_policy()
    report.verify_baseline()
    segmented.chronos2_shadow.source_pins_before(ACTIVATION)
    if not args.ledger.is_file():
        raise FileNotFoundError("ledger missing")
    with tempfile.TemporaryDirectory(prefix="forecast-baselines-") as temp:
        snapshot = Path(temp) / "ledger.sqlite"
        with closing(sqlite3.connect(f"{args.ledger.resolve().as_uri()}?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(snapshot)) as destination:
                source.backup(destination)
        try:
            records = report.read_observations(snapshot)
            now = datetime.now(timezone.utc)
            if any(report._timestamp(r["cutoff_utc"]) > now or report._timestamp(r["created_at_utc"]) > now for r in records):
                raise ValueError("future observation")
            result = add_baselines(records, snapshot)
        finally:
            gc.collect()
    verify_precommitted()
    capture.verify_policy()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
