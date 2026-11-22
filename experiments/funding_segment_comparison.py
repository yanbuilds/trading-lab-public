"""Precommitted, on-demand public-funding estimates for all forward segments."""
from __future__ import annotations

import argparse
from contextlib import closing
from datetime import datetime, timedelta, timezone
import gc
import json
from pathlib import Path
import sqlite3
import subprocess
import tempfile

from experiments import (chronos2_shadow, chronos2_shadow_report, forecast_baseline_report as baseline,
                         frozen_policy_capture as capture, funding_history,
                         prospective_funding_realization as funding,
                         prospective_funding_stress as stress, prospective_paper_report as report,
                         segmented_forward_report_v2 as segmented)
from piia.intelligence import ai_shadow
from piia.intelligence.shadow_claim_audit import audit_claims
from trading_lab.perps.compare import simulate_book, summarize

SOURCES = ("FUNDING_SEGMENT_COMPARISON_V1.md", "experiments/funding_segment_comparison.py")
ACTIVATION = baseline.ACTIVATION


def verify_precommitted():
    for name in SOURCES:
        local = (capture.ROOT / name).read_bytes().replace(b"\r\n", b"\n")
        tracked = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=capture.ROOT,
                                 capture_output=True, check=True, timeout=15).stdout.replace(b"\r\n", b"\n")
        stamp = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD", "--", name], cwd=capture.ROOT,
                               capture_output=True, check=True, timeout=15).stdout.strip()
        if local != tracked or int(stamp) >= int(ACTIVATION.timestamp()):
            raise ValueError("funding comparison not committed before activation")
    baseline.verify_precommitted()
    capture.verify_policy()
    report.verify_baseline()
    ai_shadow.verify_protocol()
    chronos2_shadow.source_pins_before(ACTIVATION)
    funding.verify_committed_before(ACTIVATION)


def evaluate_segment(records, ledger, funding_events, histories):
    """All arms use the same checked public rates; unavailable model arms stay missing."""
    frames, states, first, last = report.prepare_records(records)
    specs = report.recorded_specs(records)
    start, end = first + timedelta(hours=1), last - timedelta(hours=1)
    rates, intervals = funding.verify_schedule(records, funding_events, histories, start, end)
    cutoffs = [r["cutoff_utc"] for r in records]
    candidate = states["candidate"]
    unavailable = {}
    for name, lookback in baseline.LOOKBACKS.items():
        filtered, counts = baseline.filtered_states(records, candidate, lookback)
        if counts["candidate_signals"]:
            states[name] = filtered
        else:
            unavailable[name] = "no_candidate_signals"
    for arm, reader, validator in (
        ("ai_shadow", report.read_shadow_events, report.shadow_states),
        ("chronos_shadow", report.read_chronos_events, report.chronos_shadow_states),
        ("chronos2_shadow", chronos2_shadow_report.read_events, chronos2_shadow_report.validated_states),
    ):
        try:
            if not audit_claims(ledger, arm, cutoffs=cutoffs)["ok"]:
                raise ValueError("claim audit failed")
            filtered, status = validator(records, candidate, reader(ledger))
            if filtered is None:
                unavailable[arm] = status
            else:
                states[arm] = filtered
        except (ValueError, KeyError, TypeError, OSError, sqlite3.Error):
            unavailable[arm] = "missing_or_invalid_shadow"
    accounts = {}
    for name, arm_states in states.items():
        raw = simulate_book(stress._prepared(frames, arm_states), specs, funding=rates, funding_intervals=intervals,
                            capital=100, start=start, end=end, prepared=True,
                            max_hold_hours=None if name == "ema" else 24)
        trades = raw.pop("trades")
        raw.pop("nav", None)
        value = summarize({**raw, "trades": trades})
        value["estimated_final_usdt"] = value.pop("final")
        value["estimated_return_pct"] = value.pop("return_pct")
        value["estimated_funding_paid_usdt"] = value.pop("funding_net_paid")
        value["funding_net_paid"] = None
        value["funding_coverage_checked"] = False
        value["funding_coverage_note"] = "Public schedule and history checked; actual settlement remains unverified."
        accounts[name] = value
    return {"status": "public_funding_estimate_not_verified_net", "accounts": accounts,
            "unavailable_arms": unavailable,
            "cash_reference": {"initial": 100., "estimated_final_usdt": 100., "exposure": "none"},
            "funding_intervals_observed_hours": intervals, "history_provenance": histories}


def build_report(records, ledger, funding_events, fetch_history):
    segments, excluded = segmented.partition(records)
    output = []
    for part in segments:
        rows = [row for _, row in part]
        section = {"first_cutoff_utc": part[0][0].isoformat(), "last_cutoff_utc": part[-1][0].isoformat(),
                   "observations": len(rows), "result": None}
        if len(rows) < 3:
            section["status"] = "insufficient_contiguous_observations"
        else:
            try:
                first, last = part[0][0] + timedelta(hours=1), part[-1][0] - timedelta(hours=1)
                if any(r["cutoff_utc"] not in funding_events or funding_events[r["cutoff_utc"]].get("status") != "complete" for r in rows):
                    raise ValueError("funding observation missing")
                histories = {s: fetch_history(s, first, last) for s in capture.SYMBOLS}
                section["result"] = evaluate_segment(rows, ledger, funding_events, histories)
                section["status"] = "public_funding_estimate_not_verified_net"
            except Exception as exc:
                section.update(status="unavailable", reason=type(exc).__name__)
        output.append(section)
    return {"status": "all_segments_no_aggregate", "order_permissions": False,
            "excluded_pre_activation_observations": excluded, "segments": output, "aggregate_return": None,
            "limitations": ["Public rates and update schedules are not verified account settlement.",
                            "Mark, liquidation, real fills and terminal closing costs remain unverified.",
                            "Segments reset capital and exposures differ; no aggregate or proven edge.",
                            "Local hashes and timestamps are not independent proof."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=capture.LEDGER)
    parser.add_argument("--out", type=Path, default=capture.ROOT / "artifacts/funding_segment_comparison.json")
    args = parser.parse_args()
    if datetime.now(timezone.utc) < ACTIVATION:
        raise ValueError("protocol not active")
    verify_precommitted()
    if not args.ledger.is_file():
        raise FileNotFoundError("ledger unavailable")
    with tempfile.TemporaryDirectory(prefix="funding-segment-") as temp:
        snapshot = Path(temp) / "ledger.sqlite"
        with closing(sqlite3.connect(f"{args.ledger.resolve().as_uri()}?mode=ro", uri=True)) as source:
            with closing(sqlite3.connect(snapshot)) as destination:
                source.backup(destination)
        try:
            records = report.read_observations(snapshot)
            now = datetime.now(timezone.utc)
            if any(report._timestamp(r["cutoff_utc"]) > now or report._timestamp(r["created_at_utc"]) > now for r in records):
                raise ValueError("future observation")
            result = build_report(records, snapshot, funding.read_funding_events(snapshot), funding_history.fetch_history)
        finally:
            gc.collect()
    verify_precommitted()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    main()
