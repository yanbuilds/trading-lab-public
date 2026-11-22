"""Manually score matured prospective forecasts; research only, never trades."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
import sqlite3

import httpx

from experiments import chronos_prospective as capture


def score(event: dict, payload: dict, received: datetime) -> dict:
    """Validate the matching completed candle and score against last-close baseline."""
    cutoff = datetime.fromisoformat(event["cutoff_at_utc"])
    target = datetime.fromisoformat(event["target_at_utc"])
    created = datetime.fromisoformat(event["created_at_utc"])
    if any(t.tzinfo is None for t in (cutoff, target, created, received)):
        raise ValueError("timezone-aware timestamps required")
    if (cutoff.minute or cutoff.second or cutoff.microsecond or target != cutoff + timedelta(hours=24)
            or not cutoff <= created < target):
        raise ValueError("invalid forecast timing")
    if received < target + timedelta(minutes=5):
        raise ValueError("outcome not yet eligible")
    if (event.get("symbol") not in capture.SYMBOLS or event.get("model_sha256") != capture.WEIGHTS_HASH
            or event.get("config_sha256") != capture.CONFIG_HASH or event.get("model_revision") != capture.MODEL_REVISION
            or event.get("calibration") != "not_calibrated" or event.get("order_permissions") is not False
            or event.get("source") != capture.API):
        raise ValueError("unsupported forecast")
    source_received = datetime.fromisoformat(event["received_at_utc"])
    source_server = datetime.fromisoformat(event["server_time_utc"])
    if (source_received.tzinfo is None or source_server.tzinfo is None
            or not cutoff <= source_received <= created or source_server < cutoff
            or abs((source_received - source_server).total_seconds()) > 120):
        raise ValueError("invalid forecast source timing")
    evidence = event["evidence"]
    encoded = json.dumps(evidence, sort_keys=True, separators=(",", ":")).encode()
    if len(evidence) != 512 or hashlib.sha256(encoded).hexdigest() != event["evidence_sha256"]:
        raise ValueError("input evidence mismatch")
    for i, item in enumerate(evidence):
        stamp = datetime.fromisoformat(item["open_at_utc"])
        value = item["close"]
        if (stamp.tzinfo is None or stamp != cutoff - timedelta(hours=512 - i)
                or isinstance(value, bool) or not isinstance(value, (int, float))
                or not math.isfinite(value) or value <= 0):
            raise ValueError("invalid or noncontiguous forecast evidence")
    if datetime.fromisoformat(evidence[-1]["open_at_utc"]) != cutoff - timedelta(hours=1):
        raise ValueError("invalid baseline candle")
    baseline = float(evidence[-1]["close"])
    prior = float(evidence[-25]["close"])
    interval = event["forecast_research_only"]
    lower, forecast, upper = (float(interval[k]) for k in ("q10", "median", "q90"))
    if datetime.fromisoformat(evidence[-25]["open_at_utc"]) != cutoff - timedelta(hours=25):
        raise ValueError("invalid momentum candle")
    if (not all(math.isfinite(x) and x > 0 for x in (baseline, prior, lower, forecast, upper))
            or not lower <= forecast <= upper):
        raise ValueError("invalid forecast or baseline")
    if payload.get("code") != "00000" or not isinstance(payload.get("data"), list):
        raise ValueError("provider did not return candles")
    server = datetime.fromtimestamp(int(payload["requestTime"]) / 1000, timezone.utc)
    if abs((received - server).total_seconds()) > 120 or server < target + timedelta(minutes=5):
        raise ValueError("provider clock disagreement or outcome not finalized")
    target_ms = int((target - timedelta(hours=1)).timestamp() * 1000)
    matching = [row for row in payload["data"] if isinstance(row, list) and len(row) >= 6 and int(row[0]) == target_ms]
    if len(matching) != 1:
        raise ValueError("target candle missing or duplicated")
    row = matching[0]
    o, h, low, close, volume = map(float, row[1:6])
    if (not all(math.isfinite(v) for v in (o, h, low, close, volume)) or min(o, h, low, close) <= 0
            or volume < 0 or h < max(o, low, close) or low > min(o, h, close)):
        raise ValueError("invalid target candle")
    return {"observed_at_utc": received.isoformat(), "provider_time_utc": server.isoformat(),
            "target_candle_open_utc": (target - timedelta(hours=1)).isoformat(), "close": close,
            "model_absolute_error": abs(forecast - close), "baseline_absolute_error": abs(baseline - close),
            "model_direction_correct": (forecast > baseline) == (close > baseline) if forecast != baseline and close != baseline else None,
            "momentum_direction_correct": (baseline > prior) == (close > baseline) if baseline != prior and close != baseline else None,
            "research_interval_contains_close": lower <= close <= upper,
            "source": capture.API, "status": "local_research_not_independently_verified"}


def run(path=capture.DEFAULT_LEDGER) -> dict:
    if not path.is_file():
        raise FileNotFoundError("prospective ledger not found")
    with sqlite3.connect(path) as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != capture.APP_ID:
            raise ValueError("unrelated database")
        db.execute("CREATE TABLE IF NOT EXISTS forecast_outcomes (forecast_id INTEGER PRIMARY KEY, payload_json TEXT NOT NULL, "
                   "FOREIGN KEY(forecast_id) REFERENCES forecast_events(id))")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_outcome_{action.lower()} BEFORE {action} ON forecast_outcomes "
                       "BEGIN SELECT RAISE(ABORT, 'append-only outcomes'); END")
        pending = db.execute("SELECT e.id, e.symbol, e.cutoff, e.created_at, e.target_at, e.model_sha256, e.payload_json "
                             "FROM forecast_events e LEFT JOIN forecast_outcomes o ON e.id=o.forecast_id "
                             "WHERE o.forecast_id IS NULL ORDER BY e.id").fetchall()
        ready, waiting, errors = 0, 0, []
        with httpx.Client(timeout=20., follow_redirects=False) as client:
            for event_id, symbol, cutoff, created, target, model, raw in pending:
                event = json.loads(raw)
                if (event.get("symbol"), event.get("cutoff_at_utc"), event.get("created_at_utc"),
                    event.get("target_at_utc"), event.get("model_sha256")) != (symbol, cutoff, created, target, model):
                    raise ValueError("forecast ledger columns disagree with payload")
                target_dt = datetime.fromisoformat(target)
                if target_dt.tzinfo is None:
                    raise ValueError("timezone-aware target required")
                if capture.utc_now() < target_dt + timedelta(minutes=5):
                    waiting += 1
                    continue
                try:
                    response = client.get(capture.API, params={"category": "USDT-FUTURES", "symbol": symbol,
                        "interval": "1H", "type": "market", "endTime": str(int(target_dt.timestamp() * 1000)), "limit": "3"})
                    response.raise_for_status()
                    result = score(event, response.json(), capture.utc_now())
                    db.execute("INSERT INTO forecast_outcomes(forecast_id, payload_json) VALUES (?, ?)",
                               (event_id, json.dumps(result, sort_keys=True, allow_nan=False)))
                    ready += 1
                except (httpx.HTTPError, ValueError, TypeError, KeyError, OverflowError) as exc:
                    errors.append({"forecast_id": event_id, "symbol": symbol, "error": str(exc)})
        grouped = {}
        for symbol, raw in db.execute("SELECT e.symbol, o.payload_json FROM forecast_outcomes o "
                                      "JOIN forecast_events e ON e.id=o.forecast_id ORDER BY e.id"):
            result = json.loads(raw)
            grouped.setdefault(symbol, []).append(result)
        summary = {}
        for symbol, samples in grouped.items():
            directions = [s["model_direction_correct"] for s in samples if s["model_direction_correct"] is not None]
            paired = [s for s in samples if s.get("model_direction_correct") is not None
                      and s.get("momentum_direction_correct") is not None]
            interval = [s["research_interval_contains_close"] for s in samples if "research_interval_contains_close" in s]
            summary[symbol] = {"samples": len(samples),
                               "model_mean_absolute_error": sum(s["model_absolute_error"] for s in samples) / len(samples),
                               "baseline_mean_absolute_error": sum(s["baseline_absolute_error"] for s in samples) / len(samples),
                               "direction_correct_count": sum(directions), "direction_comparable_count": len(directions),
                               "paired_direction_samples": len(paired),
                               "paired_model_correct": sum(s["model_direction_correct"] for s in paired),
                               "paired_momentum_correct": sum(s["momentum_direction_correct"] for s in paired),
                               "interval_observations": len(interval), "interval_contains_count": sum(interval)}
        return {"newly_scored": ready, "not_yet_due": waiting, "errors": errors, "summary_by_symbol": summary,
                "note": "Local research only; errors are not trading returns or calibrated accuracy."}


if __name__ == "__main__":
    print(json.dumps(run(), indent=2))
