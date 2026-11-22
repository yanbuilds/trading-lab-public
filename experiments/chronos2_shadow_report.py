"""Compare a complete Chronos-2 shadow with frozen price-only accounts."""
from __future__ import annotations
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from experiments import chronos2_shadow as shadow, prospective_paper_report as report
from experiments import prospective_funding_stress as stress
from piia.intelligence import ai_shadow
from trading_lab.perps.compare import simulate_book, summarize


def read_events(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("prospective ledger missing")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != ai_shadow.APP_ID:
            raise ValueError("unrelated ledger")
        tables = {r[0] for r in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if not {"chronos2_shadow_claims", "chronos2_shadow_events"}.issubset(tables):
            return {}
        rows = db.execute("SELECT o.id,o.cutoff,o.created_at,o.payload_json,c.started_at,c.observation_sha256,"
                          "e.created_at,e.payload_json FROM observations o JOIN chronos2_shadow_claims c ON o.id=c.observation_id "
                          "JOIN chronos2_shadow_events e ON o.id=e.observation_id").fetchall()
    output = {}
    for ident, cutoff, observed, raw, started, claim_hash, created, payload in rows:
        event = json.loads(payload)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if (claim_hash != digest or event.get("observation_sha256") != digest or event.get("observation_id") != ident
                or event.get("created_at_utc") != created
                or not report._timestamp(observed) <= report._timestamp(started) <= report._timestamp(created)
                or report._timestamp(created) >= report._timestamp(cutoff) + timedelta(hours=1)):
            raise ValueError("shadow event binding or time invalid")
        output[cutoff] = event
    return output


def validated_states(records, candidate, events):
    if not records:
        raise ValueError("observations missing")
    pins = shadow.source_pins_before(report._timestamp(records[0]["created_at_utc"]))
    result = {s: {} for s in ai_shadow.SYMBOLS}
    active_count = 0
    for record in records:
        cutoff = report._timestamp(record["cutoff_utc"])
        event = events.get(record["cutoff_utc"])
        if not isinstance(event, dict) or event.get("source_pins") != pins or event.get("order_permissions") is not False:
            raise ValueError("missing or incompatible Chronos-2 shadow event")
        created = report._timestamp(event["created_at_utc"])
        if not report._timestamp(record["created_at_utc"]) <= created < cutoff + timedelta(hours=1):
            raise ValueError("late shadow event")
        active = {s for s in ai_shadow.SYMBOLS if candidate[s][cutoff]["signal"]}
        if event.get("status") != ("completed" if active else "no_candidate"):
            raise ValueError("shadow failed or missing")
        if event.get("model_process_attempted") is not bool(active):
            raise ValueError("model invocation status inconsistent")
        output = event.get("model_output")
        if active:
            if event.get("weights_ran_verified") is not True or not shadow.valid_output(output, active):
                raise ValueError("model output invalid")
        elif output is not None or event.get("weights_ran_verified") is not False:
            raise ValueError("unexpected model output")
        decisions = event.get("decisions")
        if not isinstance(decisions, dict) or set(decisions) != set(ai_shadow.SYMBOLS):
            raise ValueError("shadow decisions incomplete")
        for symbol in ai_shadow.SYMBOLS:
            state, decision = candidate[symbol][cutoff], decisions[symbol]
            signal = state["signal"]
            if type(decision.get("model_pass")) is not bool or decision.get("mechanical_signal") != signal:
                raise ValueError("shadow decision signal mismatch")
            if signal:
                median = output["outputs"][symbol]["median"]
                last = float(record["evidence"][symbol]["bars"][-1]["close"])
                direction = 1 if median > last else -1 if median < last else 0
                expected = direction == signal
                active_count += 1
            else:
                expected = False
            if decision["model_pass"] != expected:
                raise ValueError("shadow decision inconsistent with output")
            result[symbol][cutoff] = {**state, "signal": signal if expected else 0}
    if not active_count:
        raise ValueError("no candidate signals; model not evaluated")
    return result, active_count


def evaluate(records, specs, events, tiny_events=None):
    frames, states, first, last = report.prepare_records(records)
    reference_specs = report.recorded_specs(records)
    if specs != reference_specs:
        raise ValueError("specifications do not match observations")
    filtered, active_count = validated_states(records, states["candidate"], events)
    baseline = report.evaluate(records, specs, chronos_events=tiny_events or {})
    data = stress._prepared(frames, filtered)
    raw = simulate_book(data, specs, funding=None, capital=100, start=first + timedelta(hours=1),
                        end=last - timedelta(hours=1), prepared=True, max_hold_hours=24)
    trades = raw.pop("trades")
    raw.pop("nav", None)
    result = {**summarize({**raw, "trades": trades}), "diagnostic_trade_log": trades}
    return {"status": "local_chronos2_shadow_price_only_not_validated_net", "order_permissions": False,
            "first_decision_utc": first.isoformat(), "last_observation_cutoff_utc": last.isoformat(),
            "observations": len(records), "active_model_signals": active_count, "capital_usdt": 100,
            "funding": "unknown_excluded", "candidate": baseline["candidate"], "ema": baseline["ema"],
            "chronos_tiny_shadow": baseline["chronos_shadow"], "chronos2_shadow": result,
            "cash_reference": baseline["cash_reference"],
            "limitations": ["Unknown funding excluded; no verified net performance.",
                            "Exposure and positions differ between arms; cash has no exposure.",
                            "Model uses univariate closes, no external covariates; training overlap unknown.",
                            "Local clock and Git history are not independent timing proof.",
                            "No verified actual fills, liquidation, mark price or terminal close costs."]}


def evaluate_public_funding(records, specs, events, funding_events, histories, tiny_events=None):
    """Re-run all available accounts with checked public funding, still an estimate."""
    from experiments import prospective_funding_realization as realization
    frames, states, first, last = report.prepare_records(records)
    if specs != report.recorded_specs(records):
        raise ValueError("specifications do not match observations")
    filtered, active_count = validated_states(records, states["candidate"], events)
    states["chronos2_shadow"] = filtered
    tiny, tiny_status = report.chronos_shadow_states(records, states["candidate"], tiny_events or {})
    if tiny is not None:
        states["chronos_shadow"] = tiny
    start, end = first + timedelta(hours=1), last - timedelta(hours=1)
    funding, intervals = realization.verify_schedule(records, funding_events, histories, start, end)
    accounts = {}
    names = ("candidate", "ema", "chronos2_shadow") + (("chronos_shadow",) if tiny is not None else ())
    for name in names:
        raw = simulate_book(stress._prepared(frames, states[name]), specs, funding=funding, funding_intervals=intervals,
                            capital=100, start=start, end=end, prepared=True,
                            max_hold_hours=None if name == "ema" else 24)
        trades = raw.pop("trades")
        raw.pop("nav", None)
        result = summarize({**raw, "trades": trades})
        result["estimated_final_usdt"] = result.pop("final")
        result["estimated_return_pct"] = result.pop("return_pct")
        result["estimated_funding_paid_usdt"] = result.pop("funding_net_paid")
        result["funding_net_paid"] = None
        result["funding_coverage_checked"] = False
        result["funding_coverage_note"] = "Public rate and reported schedule checked; actual exchange settlement is not verified."
        accounts[name] = result
    return {"status": "public_funding_paper_estimate_not_actual_net_performance", "order_permissions": False,
            "active_model_signals": active_count, "accounts": accounts, "chronos_tiny_status": tiny_status,
            "funding_intervals_observed_hours": intervals, "history_provenance": histories,
            "limitations": ["Public rates and reported update schedule do not prove account settlement completeness.",
                            "Hourly open is a mark proxy; actual fills, funding, fees and liquidation are unverified.",
                            "Open positions omit terminal close costs; account exposures differ."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=shadow.LEDGER)
    parser.add_argument("--out", type=Path, default=shadow.ROOT / "artifacts/chronos2_shadow_report.json")
    parser.add_argument("--public-funding-estimate", action="store_true", help="fetch public settled-rate history; still an estimate, not verified net P&L")
    args = parser.parse_args()
    shadow.capture.verify_policy()
    report.verify_baseline()
    records = report.read_observations(args.ledger)
    specs = report.recorded_specs(records)
    events = read_events(args.ledger)
    result = evaluate(records, specs, events, report.read_chronos_events(args.ledger))
    if args.public_funding_estimate:
        from experiments import funding_history, prospective_funding_realization as realization
        first = report._timestamp(records[0]["cutoff_utc"]) + timedelta(hours=1)
        last = report._timestamp(records[-1]["cutoff_utc"]) - timedelta(hours=1)
        try:
            histories = {symbol: funding_history.fetch_history(symbol, first, last) for symbol in ai_shadow.SYMBOLS}
            result["public_funding_estimate"] = evaluate_public_funding(
                records, specs, events, realization.read_funding_events(args.ledger), histories,
                report.read_chronos_events(args.ledger))
        except Exception as exc:
            result["public_funding_estimate"] = {"status": "unavailable", "reason": type(exc).__name__,
                                                  "order_permissions": False, "funding_not_replaced_with_zero": True}
    shadow.source_pins_before(report._timestamp(records[0]["created_at_utc"]))
    result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False, default=str), encoding="utf-8")
    print(json.dumps(result, indent=2, allow_nan=False, default=str))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError, sqlite3.Error) as exc:
        raise SystemExit(f"Chronos-2 comparison unavailable: {type(exc).__name__}: {exc}") from exc
