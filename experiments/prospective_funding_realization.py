"""Apply public settled funding rates to prospective paper paths, with explicit uncertainty."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3
import subprocess

import pandas as pd
import httpx

from experiments import funding_history, funding_observer, frozen_policy_capture as capture
from experiments import prospective_funding_stress as stress, prospective_paper_report as report
from piia.intelligence import ai_shadow
from piia.intelligence.paper.specs import CONTRACTS
from trading_lab.perps.compare import funding_complete_between, simulate_book, summarize

ROOT = capture.ROOT


def verify_committed_before(observed):
    stress.verify_committed_before(observed)
    names = ["FUNDING_OBSERVER_V1.md", "experiments/funding_observer.py", "experiments/funding_history.py",
             "experiments/prospective_funding_realization.py"]
    try:
        for name in names:
            tracked = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, timeout=15, check=True).stdout
            local = (ROOT / name).read_bytes()
            if hashlib.sha256(tracked.replace(b"\r\n", b"\n")).digest() != hashlib.sha256(local.replace(b"\r\n", b"\n")).digest():
                raise ValueError("worktree differs from HEAD")
        log = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD", "--", *names], cwd=ROOT,
                             capture_output=True, timeout=15, check=True)
        committed = datetime.fromtimestamp(int(log.stdout.strip()), timezone.utc)
    except (OSError, subprocess.SubprocessError, TypeError, ValueError) as exc:
        raise ValueError("funding observation source commit is not verified") from exc
    if observed.tzinfo is None or observed.utcoffset() is None or committed > observed:
        raise ValueError("funding observation source committed after observation")


def read_funding_events(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("prospective observation ledger not found")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != capture.APP_ID:
            raise ValueError("unrelated database")
        tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()}
        if not {"funding_observations", "funding_observation_claims"}.issubset(tables):
            return {}
        rows = db.execute("SELECT o.cutoff, o.created_at, o.payload_json, a.observation_id, a.created_at, a.payload_json, "
                          "c.started_at, c.observation_sha256 FROM funding_observations a "
                          "JOIN observations o ON a.observation_id=o.id "
                          "JOIN funding_observation_claims c ON c.observation_id=o.id").fetchall()
    events = {}
    for cutoff, observed, source, ident, created, raw, started, claimed_hash in rows:
        event = json.loads(raw)
        if (event.get("observation_id") != ident or event.get("created_at_utc") != created
                or event.get("observation_sha256") != hashlib.sha256(source.encode("utf-8")).hexdigest()
                or claimed_hash != event.get("observation_sha256")
                or not report._timestamp(observed) <= report._timestamp(started) <= report._timestamp(created)
                or report._timestamp(created) >= report._timestamp(cutoff) + timedelta(hours=1)):
            raise ValueError("funding event does not match source observation")
        events[cutoff] = event
    return events


def verify_schedule(records, events, histories, start, end):
    """Check contiguous snapshots against public history; not proof of actual exchange settlement."""
    expected_symbols = set(capture.SYMBOLS)
    if set(histories) != expected_symbols:
        raise ValueError("history required for all symbols")
    script_sha = hashlib.sha256(Path(funding_observer.__file__).read_bytes()).hexdigest()
    protocol_sha = hashlib.sha256((ROOT / "FUNDING_OBSERVER_V1.md").read_bytes()).hexdigest()
    snapshots = {symbol: {} for symbol in capture.SYMBOLS}
    for record in records:
        cutoff = report._timestamp(record["cutoff_utc"])
        event = events.get(record["cutoff_utc"])
        if not event or event.get("status") != "complete" or event.get("errors") or event.get("order_permissions") is not False:
            raise ValueError("missing or failed funding observation")
        created = report._timestamp(event["created_at_utc"])
        if (event.get("observer_script_sha256") != script_sha or event.get("protocol_sha256") != protocol_sha
                or not report._timestamp(record["created_at_utc"]) <= created < cutoff + timedelta(hours=1)
                or set(event.get("snapshots") or {}) != expected_symbols):
            raise ValueError("invalid or late funding observation")
        for symbol in capture.SYMBOLS:
            item = event["snapshots"][symbol]
            received = report._timestamp(item["received_at_utc"])
            if not cutoff <= received <= created:
                raise ValueError("funding snapshot outside observation interval")
            raw = item["raw_json"].encode("utf-8")
            checked = funding_observer.parse_snapshot(json.loads(raw), symbol=symbol, cutoff=cutoff, received=received, raw=raw)
            if checked != item:
                raise ValueError("funding snapshot content mismatch")
            snapshots[symbol][cutoff] = item
    result, intervals = {}, {}
    for symbol in capture.SYMBOLS:
        series = snapshots[symbol]
        hours = sorted(series)
        if len(hours) != len(records) or any(hours[i+1] - hours[i] != timedelta(hours=1) for i in range(len(hours)-1)):
            raise ValueError("funding observations must be contiguous")
        values = {item["reported_interval_hours"] for item in series.values()}
        if len(values) != 1:
            raise ValueError("funding interval changed")
        interval = values.pop()
        for before, after in zip(hours, hours[1:]):
            prev, curr = series[before], series[after]
            previous_update = report._timestamp(prev["next_update_utc"])
            current_update = report._timestamp(curr["next_update_utc"])
            expected = previous_update if report._timestamp(curr["received_at_utc"]) < previous_update else previous_update + timedelta(hours=interval)
            if current_update != expected:
                raise ValueError("funding next-update schedule changed unexpectedly")
        history = histories[symbol]
        rows = funding_history.validate_history(history, symbol, start, end)
        pair = CONTRACTS[symbol]["ccxt_symbol"]
        coverage = funding_complete_between({pair: rows}, {pair: interval}, pd.Timestamp(start), pd.Timestamp(end))
        if not coverage["complete"]:
            raise ValueError("settled funding history missing or invalid under observed interval")
        for at in pd.date_range(pd.Timestamp(start).ceil(f"{interval}h"), pd.Timestamp(end), freq=f"{interval}h"):
            prior = series.get((at - pd.Timedelta(hours=1)).to_pydatetime())
            if not prior or report._timestamp(prior["next_update_utc"]) != at.to_pydatetime():
                raise ValueError("historical funding time not supported by prior update snapshot")
        result[pair], intervals[pair] = rows, interval
    return result, intervals


def evaluate(records, events, histories, specs, shadow=None, chronos_events=None):
    frames, states, first, last = report.prepare_records(records)
    expected_specs = {CONTRACTS[s]["ccxt_symbol"] for s in capture.SYMBOLS}
    if set(specs) != expected_specs or any(specs[k].symbol != k for k in expected_specs):
        raise ValueError("matching specs for all three markets required")
    start, end = first + timedelta(hours=1), last - timedelta(hours=1)
    funding, intervals = verify_schedule(records, events, histories, start, end)
    ai_states, ai_status = report.shadow_states(records, states["candidate"], shadow or {})
    model_states, model_status = report.chronos_shadow_states(records, states["candidate"], chronos_events or {})
    if ai_states is not None: states["ai_shadow"] = ai_states
    if model_states is not None: states["chronos_shadow"] = model_states
    names = ("candidate", "ema") + (("ai_shadow",) if ai_states is not None else ()) + (("chronos_shadow",) if model_states is not None else ())
    accounts = {}
    for name in names:
        data = stress._prepared(frames, states[name])
        raw = simulate_book(data, specs, funding=funding, funding_intervals=intervals, capital=100,
                            start=start, end=end, prepared=True, max_hold_hours=24 if name != "ema" else None)
        trades = raw.pop("trades"); raw.pop("nav", None)
        summary = summarize({**raw, "trades": trades})
        summary["estimated_final_usdt"] = summary.pop("final")
        summary["estimated_return_pct"] = summary.pop("return_pct")
        summary["estimated_funding_paid_usdt"] = summary.pop("funding_net_paid")
        summary["funding_net_paid"] = None
        summary["funding_coverage_checked"] = False
        summary["funding_coverage_note"] = "Public rate and reported schedule checked; actual exchange settlement completeness is unproven."
        accounts[name] = summary
    return {"status": "public_funding_paper_estimate_not_actual_net_performance", "order_permissions": False,
            "first_decision_utc": first.isoformat(), "last_observation_cutoff_utc": last.isoformat(),
            "observations": len(records), "capital_usdt": 100, "accounts": accounts,
            "ai_shadow_status": ai_status, "chronos_shadow_status": model_status,
            "funding_intervals_observed_hours": intervals,
            "cash_reference": {"initial": 100., "final": 100., "exposure": "none"},
            "history_provenance": histories,
            "limitations": ["nextUpdate denotes provider rate update, not confirmed exchange settlement time",
                            "official public records can be revised; schedule completeness is not independently proven",
                            "hourly open proxies mark price; actual account funding and fees are unknown",
                            "liquidation, real execution, terminal close costs and historical specifications remain unverified",
                            "local timestamps and hashes do not independently prove timing or data truth"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=capture.LEDGER)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/prospective_funding_estimate.json")
    args = parser.parse_args()
    capture.verify_policy(); report.verify_baseline(); ai_shadow.verify_protocol()
    records = report.read_observations(args.ledger)
    frames, _states, first, last = report.prepare_records(records)
    del frames
    verify_committed_before(report._timestamp(records[0]["created_at_utc"]))
    events = read_funding_events(args.ledger)
    if (set(events) != {record["cutoff_utc"] for record in records}
            or any(event.get("status") != "complete" for event in events.values())):
        raise ValueError("funding observation coverage incomplete; no history request made")
    start, end = first + timedelta(hours=1), last - timedelta(hours=1)
    with httpx.Client(timeout=15, follow_redirects=False) as client:
        histories = {symbol: funding_history.fetch_history(symbol, start, end, client=client) for symbol in capture.SYMBOLS}
    result = evaluate(records, events, histories, report.recorded_specs(records), report.read_shadow_events(args.ledger),
                      report.read_chronos_events(args.ledger))
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    verify_committed_before(report._timestamp(records[0]["created_at_utc"]))
    capture.verify_policy(); report.verify_baseline(); ai_shadow.verify_protocol()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("status", "observations", "accounts", "limitations")}, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error, httpx.HTTPError) as exc:
        raise SystemExit(f"Funding estimate unavailable: {type(exc).__name__}: {exc}") from exc
