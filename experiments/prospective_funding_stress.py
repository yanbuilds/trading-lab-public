"""Fixed hypothetical funding stress on prospective local observations, never live net P&L."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess

import pandas as pd

from experiments import frozen_policy_capture as capture, prospective_paper_report as report
from piia.intelligence import ai_shadow
from piia.intelligence.paper.specs import CONTRACTS
from trading_lab.perps.compare import simulate_book, summarize

SCENARIOS = (("positive_1bp", 0.0001), ("negative_1bp", -0.0001),
             ("positive_5bp", 0.0005), ("negative_5bp", -0.0005))
INTERVAL_HOURS = 8
ROOT = capture.ROOT


def verify_committed_before(observed_at):
    """Check tracked experiment and simulation sources; Git dates are not trusted timestamps."""
    names = [*capture.PINNED, "AI_SHADOW_V1.md", "src/trading_lab/perps/research.py", "src/piia/intelligence/ai_shadow.py",
             "experiments/prospective_paper_report.py", "experiments/prospective_funding_stress.py", "FUNDING_STRESS_V1.md"]
    try:
        for name in names:
            tracked = subprocess.run(["git", "show", f"HEAD:{name}"], cwd=ROOT, capture_output=True, timeout=15, check=True).stdout
            disk = (ROOT / name).read_bytes()
            if hashlib.sha256(tracked.replace(b"\r\n", b"\n")).digest() != hashlib.sha256(disk.replace(b"\r\n", b"\n")).digest():
                raise ValueError("worktree differs from HEAD")
        log = subprocess.run(["git", "log", "-1", "--format=%ct", "HEAD", "--", *names],
                             cwd=ROOT, capture_output=True, timeout=15, check=True)
        committed = datetime.fromtimestamp(int(log.stdout.strip()), timezone.utc)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError) as exc:
        raise ValueError("funding stress source commit is not verified") from exc
    if observed_at.tzinfo is None or observed_at.utcoffset() is None or committed > observed_at:
        raise ValueError("funding stress source committed after observation")


def _prepared(frames, states):
    output = {}
    for symbol, frame in frames.items():
        out = frame.copy()
        out["signal"], out["atr"] = 0, float("nan")
        out["exit_long"], out["exit_short"] = False, False
        out["trail_long"], out["trail_short"] = float("nan"), float("nan")
        for cutoff, state in states[symbol].items():
            if cutoff in out.index:
                for key, value in state.items():
                    out.at[cutoff, key] = value
        output[CONTRACTS[symbol]["ccxt_symbol"]] = out
    return output


def _funding(symbols, start, end, rate):
    times = pd.date_range(pd.Timestamp(start).ceil("8h"), pd.Timestamp(end), freq="8h")
    return {symbol: [{"timestamp": int(t.timestamp() * 1000), "rate": rate} for t in times] for symbol in symbols}


def evaluate(records, specs, shadow=None, chronos_events=None):
    frames, states, first, last = report.prepare_records(records)
    expected = {CONTRACTS[s]["ccxt_symbol"] for s in capture.SYMBOLS}
    if set(specs) != expected or any(specs[k].symbol != k for k in expected):
        raise ValueError("matching specs for all three markets required")
    ai_states, ai_status = report.shadow_states(records, states["candidate"], shadow or {})
    model_states, model_status = report.chronos_shadow_states(records, states["candidate"], chronos_events or {})
    if ai_states is not None:
        states["ai_shadow"] = ai_states
    if model_states is not None:
        states["chronos_shadow"] = model_states
    names = ("candidate", "ema") + (("ai_shadow",) if ai_states is not None else ()) + (("chronos_shadow",) if model_states is not None else ())
    start, end = first + pd.Timedelta(hours=1), last - pd.Timedelta(hours=1)
    original = report.evaluate(records, specs, shadow, chronos_events)
    prepared = {name: _prepared(frames, states[name]) for name in names}
    price_only = {}
    for name in names:
        raw = simulate_book(prepared[name], specs, funding=None, capital=100, start=start, end=end,
                            prepared=True, max_hold_hours=24 if name != "ema" else None)
        trades = raw.pop("trades"); raw.pop("nav", None)
        price_only[name] = summarize({**raw, "trades": trades})
        reference = original[name] if name in ("candidate", "ema") else original[name]["result"]
        if (price_only[name]["final"] != reference["final"]
                or price_only[name]["closed_trades"] != reference["closed_trades"]):
            raise ValueError("stress harness does not match existing price-only reference")
    scenarios = {}
    for label, rate in SCENARIOS:
        funding = _funding(expected, start, end, rate)
        accounts = {}
        for name in names:
            raw = simulate_book(prepared[name], specs, funding=funding, capital=100, start=start, end=end,
                                prepared=True, max_hold_hours=24 if name != "ema" else None,
                                funding_intervals={symbol: INTERVAL_HOURS for symbol in expected})
            trades = raw.pop("trades"); raw.pop("nav", None)
            summary = summarize({**raw, "trades": trades})
            summary["hypothetical_final_usdt"] = summary.pop("final")
            summary["hypothetical_return_pct"] = summary.pop("return_pct")
            summary["hypothetical_funding_paid_usdt"] = summary.pop("funding_net_paid")
            summary["funding_net_paid"] = None
            summary["funding_coverage_checked"] = False
            summary["funding_coverage_note"] = "Synthetic scenario only; actual funding coverage not checked."
            accounts[name] = summary
        scenarios[label] = {"hypothetical_rate_per_settlement": rate, "settlement_interval_hours": INTERVAL_HOURS,
                            "accounts": accounts}
    return {"status": "hypothetical_funding_stress_not_actual_net_performance", "order_permissions": False,
            "first_decision_utc": first.isoformat(), "last_observation_cutoff_utc": last.isoformat(),
            "observations": len(records), "capital_usdt": 100, "price_only": price_only,
            "ai_shadow_status": ai_status, "chronos_shadow_status": model_status, "scenarios": scenarios,
            "cash_reference": {"initial": 100., "final": 100., "exposure": "none"},
            "limitations": ["fixed synthetic rates and 8-hour schedule are not exchange settlement data",
                            "actual funding, historical schedules and exact liquidation remain unverified",
                            "funding notional uses hourly open as mark proxy",
                            "different positions and halts change account paths and exposure",
                            "terminal open positions exclude closing costs",
                            "local Git and clock are not independent proof of prospective timing"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=capture.LEDGER)
    parser.add_argument("--out", type=Path, default=ROOT / "artifacts/prospective_funding_stress.json")
    args = parser.parse_args()
    capture.verify_policy(); report.verify_baseline(); ai_shadow.verify_protocol()
    records = report.read_observations(args.ledger)
    if not records:
        raise ValueError("no prospective observations")
    verify_committed_before(report._timestamp(records[0]["created_at_utc"]))
    result = evaluate(records, report.recorded_specs(records), report.read_shadow_events(args.ledger),
                      report.read_chronos_events(args.ledger))
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    verify_committed_before(report._timestamp(records[0]["created_at_utc"]))
    capture.verify_policy(); report.verify_baseline(); ai_shadow.verify_protocol()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, default=str, allow_nan=False), encoding="utf-8")
    print(json.dumps(result, indent=2, default=str, allow_nan=False))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError) as exc:
        raise SystemExit(f"Funding stress unavailable: {type(exc).__name__}: {exc}") from exc
