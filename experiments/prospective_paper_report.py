"""Replay contiguous, locally recorded policy observations as paper research.

No historical signal import, model calls, broker or exchange orders. A decision
recorded after hour T is simulated at T+1 open, not the already-passed T open.
Funding is unknown; this is a price-only approximation, not verified net P&L.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3

import pandas as pd

from experiments import frozen_policy_capture as capture
from experiments import chronos_shadow
from trading_lab.perps.compare import attach_ema, simulate_book, summarize
from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.risk import Spec
from piia.intelligence.paper.specs import CONTRACTS
from piia.intelligence import ai_shadow

BASELINE_SHA256 = "91d0adb5cc83d5254370c3e3ed47218240338dab910f3b8ab0d0bafd9b9dc7ea"
AI_SHADOW_CODE_SHA256 = "d722ccbf05aa52e91824f3c3d45e466c67b861dd4c99d3360fea90cb5cacc354"
CHRONOS_SHADOW_CODE_SHA256 = "cece945aed3b57c9443110adca30f0ab8e27f0cb7f108ca0fe2cd9e64fba9182"


def verify_baseline():
    raw = (capture.ROOT / "src/trading_lab/perps/research.py").read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(raw).hexdigest() != BASELINE_SHA256:
        raise ValueError("frozen EMA baseline source changed")


def _timestamp(value):
    stamp = datetime.fromisoformat(value)
    if stamp.tzinfo is None or stamp.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return stamp.astimezone(timezone.utc)


def read_observations(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError("prospective observation ledger not found")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != capture.APP_ID:
            raise ValueError("unrelated database")
        rows = db.execute("SELECT cutoff, created_at, payload_json FROM observations ORDER BY cutoff").fetchall()
    records = []
    for cutoff, created, raw in rows:
        value = json.loads(raw)
        if value.get("cutoff_utc") != cutoff or value.get("created_at_utc") != created:
            raise ValueError("ledger columns disagree with payload")
        records.append(value)
    return records


def read_shadow_events(path: Path) -> dict:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='ai_shadow_events'").fetchone():
            return {}
        rows = db.execute("SELECT o.cutoff, o.payload_json, a.observation_id, a.created_at, a.payload_json "
                          "FROM ai_shadow_events a JOIN observations o ON a.observation_id=o.id").fetchall()
    result = {}
    for cutoff, raw_observation, ident, created, raw in rows:
        event = json.loads(raw)
        if (event.get("observation_id") != ident or event.get("created_at_utc") != created
                or event.get("observation_sha256") != hashlib.sha256(raw_observation.encode("utf-8")).hexdigest()):
            raise ValueError("AI event does not match source observation")
        result[cutoff] = event
    return result


def read_chronos_events(path: Path) -> dict:
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != capture.APP_ID:
            raise ValueError("unrelated database")
        if not db.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='chronos_shadow_events'").fetchone():
            return {}
        rows = db.execute("SELECT o.cutoff, o.payload_json, a.observation_id, a.created_at, a.payload_json "
                          "FROM chronos_shadow_events a JOIN observations o ON a.observation_id=o.id").fetchall()
    result = {}
    for cutoff, raw_observation, ident, created, raw in rows:
        event = json.loads(raw)
        if (event.get("observation_id") != ident or event.get("created_at_utc") != created
                or event.get("observation_sha256") != hashlib.sha256(raw_observation.encode("utf-8")).hexdigest()):
            raise ValueError("Chronos event does not match source observation")
        result[cutoff] = event
    return result


def chronos_shadow_states(records: list[dict], candidate_states: dict, shadow: dict) -> tuple[dict | None, str]:
    if not shadow:
        return None, "no_chronos_shadow_records"
    if not records:
        return None, "no_prospective_observations"
    try:
        chronos_shadow.verify_pins()
        chronos_shadow.verify_committed_before(_timestamp(records[0]["created_at_utc"]))
    except ValueError:
        return None, "chronos_shadow_source_or_commit_unverified"
    if chronos_shadow._sha(chronos_shadow.__file__) != CHRONOS_SHADOW_CODE_SHA256:
        return None, "chronos_shadow_code_changed"
    filtered = {s: {} for s in capture.SYMBOLS}
    active = 0
    for record in records:
        cutoff = _timestamp(record["cutoff_utc"])
        event = shadow.get(record["cutoff_utc"])
        if event is None:
            return None, "missing_chronos_shadow_hour"
        if record.get("capture_script_sha256") != hashlib.sha256((capture.ROOT / "experiments/frozen_policy_capture.py").read_bytes()).hexdigest():
            return None, "chronos_observation_capture_source_mismatch"
        created = _timestamp(event["created_at_utc"])
        if (event.get("status") not in ("completed", "no_candidate") or event.get("order_permissions") is not False
                or event.get("protocol_sha256") != chronos_shadow.PINS["CHRONOS_SHADOW_V1.md"]
                or event.get("source_pins") != chronos_shadow.PINS
                or event.get("shadow_code_sha256") != CHRONOS_SHADOW_CODE_SHA256
                or not _timestamp(record["created_at_utc"]) <= created < cutoff + timedelta(hours=1)):
            return None, "invalid_or_late_chronos_shadow"
        decisions, outputs = event.get("decisions") or {}, event.get("model_outputs") or {}
        active_symbols = {s for s in capture.SYMBOLS if candidate_states[s][cutoff]["signal"]}
        if (set(decisions) != set(capture.SYMBOLS) or set(outputs) != active_symbols
                or event["status"] != ("completed" if active_symbols else "no_candidate")):
            return None, "chronos_output_missing_or_invalid"
        for symbol in capture.SYMBOLS:
            state, decision = candidate_states[symbol][cutoff], decisions[symbol]
            signal = state["signal"]
            if type(decision.get("model_pass")) is not bool or decision.get("mechanical_signal") != signal:
                return None, "chronos_decision_mismatch"
            if signal:
                output = outputs[symbol]
                close = float(record["evidence"][symbol]["bars"][-1]["close"])
                if not chronos_shadow.validate_forecast(output, symbol, record["cutoff_utc"], close):
                    return None, "chronos_output_missing_or_invalid"
                median = output["forecasts"][symbol]["quantile_50"][-1]
                direction = 1 if median > close else -1 if median < close else 0
                if decision["model_pass"] != (direction == signal):
                    return None, "chronos_decision_mismatch"
                active += 1
            elif decision["model_pass"]:
                return None, "chronos_decision_mismatch"
            filtered[symbol][cutoff] = {**state, "signal": signal if decision["model_pass"] else 0}
    if not active:
        return None, "no_model_forecasts_for_candidate_signals"
    return filtered, "complete"


def shadow_states(records: list[dict], candidate_states: dict, shadow: dict) -> tuple[dict | None, str]:
    """Validate all shadow decisions before allowing a separate comparison."""
    if not shadow:
        return None, "no_ai_shadow_records"
    source = Path(ai_shadow.__file__).read_bytes().replace(b"\r\n", b"\n")
    if hashlib.sha256(source).hexdigest() != AI_SHADOW_CODE_SHA256:
        return None, "ai_shadow_code_changed"
    expected_question = hashlib.sha256(ai_shadow.QUESTION.encode("utf-8")).hexdigest()
    models, active = set(), 0
    filtered = {s: {} for s in capture.SYMBOLS}
    for record in records:
        cutoff = _timestamp(record["cutoff_utc"])
        event = shadow.get(record["cutoff_utc"])
        if event is None:
            return None, "missing_ai_shadow_hour"
        created = _timestamp(event["created_at_utc"])
        if (event.get("status") not in ("completed", "no_candidate") or event.get("order_permissions") is not False
                or event.get("protocol_sha256") != ai_shadow.PROTOCOL_SHA256 or event.get("question_sha256") != expected_question
                or event.get("shadow_code_sha256") != AI_SHADOW_CODE_SHA256
                or not _timestamp(record["created_at_utc"]) <= created < cutoff + timedelta(hours=1)):
            return None, "invalid_or_late_ai_shadow"
        decisions, outputs = event.get("decisions") or {}, event.get("model_outputs") or {}
        active_symbols = {s for s in capture.SYMBOLS if candidate_states[s][cutoff]["signal"]}
        if (set(decisions) != set(capture.SYMBOLS) or set(outputs) != active_symbols
                or event["status"] != ("completed" if active_symbols else "no_candidate")):
            return None, "model_output_missing_or_invalid"
        for symbol in capture.SYMBOLS:
            state = candidate_states[symbol][cutoff]
            decision = decisions.get(symbol) or {}
            signal = state["signal"]
            if type(decision.get("ai_pass")) is not bool or decision.get("mechanical_signal") != signal:
                return None, "ai_decision_mismatch"
            if signal:
                output = outputs.get(symbol) or {}
                synth = output.get("synthesizer") or {}
                if (event["status"] != "completed" or output.get("mode") != "model" or output.get("status") != "completed"
                        or synth.get("generation") != "model" or synth.get("direction") not in ("long", "short", "flat")
                        or not all(output.get(k) for k in ("model_provider", "model_name", "prompt_hash"))):
                    return None, "model_output_missing_or_invalid"
                expected_pass = (synth["direction"] == ("long" if signal > 0 else "short")
                                 and synth.get("suggested_action") == "review_paper_plan")
                if decision["ai_pass"] != expected_pass:
                    return None, "ai_decision_mismatch"
                models.add((output["model_provider"], output["model_name"], output["prompt_hash"]))
                active += 1
            elif decision["ai_pass"]:
                return None, "ai_decision_mismatch"
            filtered[symbol][cutoff] = {**state, "signal": signal if decision["ai_pass"] else 0}
    if not active:
        return None, "no_model_opinions_for_candidate_signals"
    if len(models) != 1:
        return None, "model_or_prompt_changed"
    return filtered, "complete"


def prepare_records(records: list[dict]) -> tuple[dict, dict, datetime, datetime]:
    """Recompute recorded signals; reject gaps, revisions or late observations."""
    if len(records) < 3:
        raise ValueError("at least three consecutive prospective observations required")
    frames: dict[str, dict] = {s: {} for s in capture.SYMBOLS}
    states = {"candidate": {s: {} for s in capture.SYMBOLS}, "ema": {s: {} for s in capture.SYMBOLS}}
    previous = None
    for record in records:
        cutoff, created = _timestamp(record["cutoff_utc"]), _timestamp(record["created_at_utc"])
        if (cutoff < capture.START or cutoff != cutoff.replace(minute=0, second=0, microsecond=0)
                or not timedelta(0) <= created - cutoff <= timedelta(minutes=50)
                or record.get("policy_commit") != "69fd853" or record.get("source") != capture.API
                or record.get("order_permissions") is not False or record.get("next_open_fill_available") is not False
                or set(record.get("symbols") or []) != set(capture.SYMBOLS)):
            raise ValueError("invalid or nonprospective observation")
        if previous is not None and cutoff - previous != timedelta(hours=1):
            raise ValueError("missing or duplicate observation hour; cannot infer an unobserved decision")
        previous = cutoff
        for symbol in capture.SYMBOLS:
            item = record["evidence"][symbol]
            received = _timestamp(item["received_at_utc"])
            if received > created:
                raise ValueError("evidence arrived after observation creation")
            bars = item["bars"]
            if not isinstance(bars, list) or len(bars) != 300:
                raise ValueError("incomplete frozen input")
            data = []
            for bar in bars:
                at = _timestamp(bar["open_at_utc"])
                data.append([str(int(at.timestamp() * 1000)), *(bar[k] for k in ("open", "high", "low", "close", "volume"))])
            frame = capture.parse_closed({"code": "00000", "requestTime": item["provider_time_ms"], "data": data}, cutoff, received)
            for timestamp, row in frame.iterrows():
                values = tuple(float(row[k]) for k in ("open", "high", "low", "close", "volume"))
                if timestamp in frames[symbol] and frames[symbol][timestamp] != values:
                    raise ValueError("overlapping input bars were revised or disagree")
                frames[symbol][timestamp] = values
            candidate = attach_range_bollinger(frame).iloc[-1]
            recorded = record["states"][symbol]
            if int(candidate["signal"]) != recorded["candidate_signal"] or str(candidate["regime"]) != recorded["regime"]:
                raise ValueError("recorded candidate state does not match frozen inputs")
            baseline = attach_ema(frame).iloc[-1]
            for name, value in (("candidate", candidate), ("ema", baseline)):
                atr = float(value["atr"])
                signal = int(value["signal"])
                if signal not in (-1, 0, 1) or (signal and (not math.isfinite(atr) or atr <= 0)):
                    raise ValueError("invalid recorded signal or ATR")
                states[name][symbol][cutoff] = {"signal": signal, "atr": atr,
                    "exit_long": bool(value["exit_long"]), "exit_short": bool(value["exit_short"])}
    all_frames = {}
    for symbol, bars in frames.items():
        index = pd.DatetimeIndex(sorted(bars))
        all_frames[symbol] = pd.DataFrame([bars[t] for t in index], index=index,
                                          columns=["open", "high", "low", "close", "volume"])
    return all_frames, states, _timestamp(records[0]["cutoff_utc"]), _timestamp(records[-1]["cutoff_utc"])


def recorded_specs(records: list[dict]) -> dict:
    """Use only contemporaneously captured specs; refuse absent or changed values."""
    if not records:
        raise ValueError("no observations")
    reference = None
    for record in records:
        cutoff, created = _timestamp(record["cutoff_utc"]), _timestamp(record["created_at_utc"])
        snap = record.get("specs_snapshot") or {}
        if (not snap.get("ok") or snap.get("source_url") != capture.CONTRACTS_API
                or set(snap.get("contracts") or {}) != set(capture.SYMBOLS)):
            raise ValueError("contemporaneous contract specifications unavailable")
        received, server = _timestamp(snap["retrieved_at_utc"]), _timestamp(snap["endpoint_request_time_utc"])
        digest = snap.get("raw_sha256")
        if (not cutoff <= received <= created or server < cutoff or abs((received - server).total_seconds()) > 120
                or not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)):
            raise ValueError("invalid contract specification provenance")
        current = {}
        for symbol in capture.SYMBOLS:
            item = snap["contracts"][symbol]
            values = [float(item[k]) for k in ("min_qty", "qty_step", "min_notional", "taker", "contract_size")]
            if (item.get("symbol") != symbol or item.get("status") != "normal" or item.get("instrument_type") != "perpetual"
                    or not all(math.isfinite(v) for v in values) or min(values[0], values[1], values[4]) <= 0
                    or values[2] < 0 or not 0 <= values[3] < 1):
                raise ValueError("invalid recorded contract specification")
            current[symbol] = tuple(values)
        if reference is not None and reference != current:
            raise ValueError("contract specifications changed; continuous result withheld")
        reference = current
    return {CONTRACTS[s]["ccxt_symbol"]: Spec(CONTRACTS[s]["ccxt_symbol"], *reference[s]) for s in capture.SYMBOLS}


def evaluate(records: list[dict], specs: dict, shadow: dict | None = None, chronos_events: dict | None = None) -> dict:
    frames, states, first, end = prepare_records(records)
    keys = {s: CONTRACTS[s]["ccxt_symbol"] for s in capture.SYMBOLS}
    if set(specs) != set(keys.values()) or any(specs[k].symbol != k for k in specs):
        raise ValueError("matching specs for all three markets required")
    results = {}
    filtered, ai_status = shadow_states(records, states["candidate"], shadow or {})
    if filtered is not None:
        states["ai_shadow"] = filtered
    model_states, model_status = chronos_shadow_states(records, states["candidate"], chronos_events or {})
    if model_states is not None:
        states["chronos_shadow"] = model_states
    for name in ("candidate", "ema") + (("ai_shadow",) if filtered is not None else ()) + (("chronos_shadow",) if model_states is not None else ()):
        data = {}
        for symbol, frame in frames.items():
            out = frame.copy()
            out["signal"], out["atr"] = 0, float("nan")
            out["exit_long"], out["exit_short"] = False, False
            out["trail_long"], out["trail_short"] = float("nan"), float("nan")
            for cutoff, state in states[name][symbol].items():
                if cutoff in out.index:
                    for key, val in state.items():
                        out.at[cutoff, key] = val
            data[keys[symbol]] = out
        raw = simulate_book(data, specs, funding=None, capital=100, start=first + timedelta(hours=1),
                            end=end - timedelta(hours=1), prepared=True,
                            max_hold_hours=24 if name in ("candidate", "ai_shadow", "chronos_shadow") else None)
        trades = raw.pop("trades")
        raw.pop("nav", None)
        results[name] = {**summarize({**raw, "trades": trades}), "diagnostic_trade_log": trades}
    return {"status": "local_prospective_paper_approximation_not_independently_verified",
            "first_decision_utc": first.isoformat(), "last_observation_cutoff_utc": end.isoformat(),
            "observations": len(records), "capital_usdt": 100, "funding": "unknown_excluded",
            "order_permissions": False, "candidate": results["candidate"], "ema": results["ema"],
            "ai_shadow": {"status": ai_status, "result": results.get("ai_shadow")},
            "chronos_shadow": {"status": model_status, "result": results.get("chronos_shadow")},
            "cash_reference": {"initial": 100., "final": 100., "return_pct": 0., "exposure": "none"},
            "limitations": ["local timestamps and hashes do not establish independent provenance",
                            "all hourly observations must be contiguous; no missing decisions filled",
                            "entry and signal exits delayed until next full hour; not exact frozen next-open policy",
                            "uses contemporaneously recorded public specs, not account-specific fees or verified actual fills and liquidation",
                            "open positions marked at last close without closing fees; funding unknown",
                            "AI shadow requires a full set of eligible contemporaneous observations; model name may hide version changes",
                            "no statistical significance claim"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=capture.LEDGER)
    parser.add_argument("--out", type=Path, default=capture.ROOT / "artifacts/prospective_paper_report.json")
    args = parser.parse_args()
    capture.verify_policy()
    verify_baseline()
    records = read_observations(args.ledger)
    ai_shadow.verify_protocol()
    shadow = read_shadow_events(args.ledger)
    chronos_events = read_chronos_events(args.ledger)
    specs = recorded_specs(records)
    report = evaluate(records, specs, shadow, chronos_events)
    report["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    capture.verify_policy()
    verify_baseline()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"Report unavailable: {type(exc).__name__}: {exc}") from exc
