"""Manual public funding snapshot attached to a contemporaneous policy observation."""
from __future__ import annotations
from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import sqlite3
import httpx

from experiments import frozen_policy_capture as capture
from piia.intelligence import ai_shadow

URL = "https://api.bitget.com/api/v2/mix/market/current-fund-rate"
LEDGER = capture.LEDGER


def _aware(value):
    dt = datetime.fromisoformat(value)
    if dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timezone-aware timestamp required")
    return dt.astimezone(timezone.utc)


def _integer(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise ValueError("integer field required")
    text = str(value)
    if not text.isdigit():
        raise ValueError("integer field required")
    return int(text)


def parse_snapshot(payload, *, symbol, cutoff, received, raw):
    if cutoff.tzinfo is None or received.tzinfo is None or not cutoff <= received < cutoff + timedelta(hours=1):
        raise ValueError("observation must be in the same UTC hour")
    if payload.get("code") != "00000" or not isinstance(payload.get("data"), list) or len(payload["data"]) != 1:
        raise ValueError("unexpected provider response")
    server = datetime.fromtimestamp(_integer(payload["requestTime"]) / 1000, timezone.utc)
    if server < cutoff or abs((received - server).total_seconds()) > 120:
        raise ValueError("provider clock disagreement")
    row = payload["data"][0]
    if row.get("symbol") != symbol:
        raise ValueError("wrong symbol")
    interval = _integer(row["fundingRateInterval"])
    if interval <= 0 or 24 % interval:
        raise ValueError("unsupported reported interval")
    if isinstance(row["fundingRate"], bool):
        raise ValueError("invalid rate")
    rate = float(row["fundingRate"])
    if not math.isfinite(rate) or abs(rate) > 1:
        raise ValueError("invalid rate")
    next_ms = _integer(row["nextUpdate"])
    next_update = datetime.fromtimestamp(next_ms / 1000, timezone.utc)
    if next_ms % 3_600_000 or not received < next_update <= received + timedelta(hours=interval, minutes=2):
        raise ValueError("invalid or late next update")
    raw_text = raw.decode("utf-8")
    if json.loads(raw_text) != payload:
        raise ValueError("raw response does not match parsed response")
    return {"symbol": symbol, "received_at_utc": received.isoformat(), "server_at_utc": server.isoformat(),
            "reported_interval_hours": interval, "next_update_utc": next_update.isoformat(),
            "current_rate": rate, "source_url": URL, "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "raw_json": raw_text, "next_update_is_verified_settlement": False}


def _save(path, ident, observation, observation_sha, result):
    cutoff = _aware(observation["cutoff_utc"])
    created = _aware(result["created_at_utc"])
    if not _aware(observation["created_at_utc"]) <= created < cutoff + timedelta(hours=1):
        raise ValueError("funding snapshot completed too late")
    if abs((ai_shadow.now_utc() - created).total_seconds()) > 120:
        raise ValueError("funding snapshot is not current")
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA application_id").fetchone()[0] != capture.APP_ID:
            raise ValueError("unrelated database")
        row = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
        claim = db.execute("SELECT observation_sha256 FROM funding_observation_claims WHERE observation_id=?", (ident,)).fetchone()
        if not row or hashlib.sha256(row[0].encode("utf-8")).hexdigest() != observation_sha or not claim or claim[0] != observation_sha:
            raise ValueError("source observation changed")
        db.execute("CREATE TABLE IF NOT EXISTS funding_observations (observation_id INTEGER PRIMARY KEY, "
                   "created_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_funding_observation_{action.lower()} BEFORE {action} ON funding_observations "
                       "BEGIN SELECT RAISE(ABORT, 'append-only funding observations'); END")
        db.execute("INSERT INTO funding_observations VALUES (?, ?, ?)",
                   (ident, created.isoformat(), json.dumps(result, sort_keys=True, allow_nan=False)))


def _claim(path, ident, observation_sha):
    with sqlite3.connect(path) as db:
        db.execute("BEGIN IMMEDIATE")
        if db.execute("PRAGMA application_id").fetchone()[0] != capture.APP_ID:
            raise ValueError("unrelated database")
        row = db.execute("SELECT payload_json FROM observations WHERE id=?", (ident,)).fetchone()
        if not row or hashlib.sha256(row[0].encode("utf-8")).hexdigest() != observation_sha:
            raise ValueError("source observation changed")
        db.execute("CREATE TABLE IF NOT EXISTS funding_observation_claims (observation_id INTEGER PRIMARY KEY, "
                   "started_at TEXT NOT NULL, observation_sha256 TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_funding_claim_{action.lower()} BEFORE {action} ON funding_observation_claims "
                       "BEGIN SELECT RAISE(ABORT, 'append-only funding claim'); END")
        db.execute("INSERT INTO funding_observation_claims VALUES (?, ?, ?)", (ident, ai_shadow.now_utc().isoformat(), observation_sha))


def run(path=LEDGER, *, client=None):
    path = Path(path)
    if client is not None and path.resolve() == LEDGER.resolve():
        raise ValueError("injected clients cannot write the real research ledger")
    capture.verify_policy()
    script_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    protocol_sha = hashlib.sha256((capture.ROOT / "FUNDING_OBSERVER_V1.md").read_bytes()).hexdigest()
    ident, observation, observation_sha = ai_shadow.latest_observation(path, ai_shadow.now_utc())
    if observation.get("capture_script_sha256") != hashlib.sha256(Path(capture.__file__).read_bytes()).hexdigest():
        raise ValueError("observation capture source mismatch")
    _claim(path, ident, observation_sha)
    cutoff = _aware(observation["cutoff_utc"])
    own = client is None
    client = client or httpx.Client(timeout=15, follow_redirects=False)
    snapshots, errors = {}, {}
    try:
        for symbol in capture.SYMBOLS:
            try:
                response = client.get(URL, params={"symbol": symbol, "productType": "USDT-FUTURES"})
                response.raise_for_status()
                received = ai_shadow.now_utc()
                snapshots[symbol] = parse_snapshot(response.json(), symbol=symbol, cutoff=cutoff,
                                                    received=received, raw=response.content)
            except (httpx.HTTPError, ValueError, KeyError, TypeError, OverflowError, OSError, UnicodeError) as exc:
                errors[symbol] = type(exc).__name__
    finally:
        if own:
            client.close()
    created = ai_shadow.now_utc()
    if (hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != script_sha
            or hashlib.sha256((capture.ROOT / "FUNDING_OBSERVER_V1.md").read_bytes()).hexdigest() != protocol_sha):
        raise ValueError("funding observer changed during capture")
    result = {"status": "complete" if not errors and set(snapshots) == set(capture.SYMBOLS) else "incomplete",
              "created_at_utc": created.isoformat(), "observation_id": ident, "observation_sha256": observation_sha,
              "observer_script_sha256": script_sha, "protocol_sha256": protocol_sha,
              "snapshots": snapshots, "errors": errors, "order_permissions": False,
              "actual_funding_paid": None, "next_update_is_verified_settlement": False}
    _save(path, ident, observation, observation_sha, result)
    return {"status": result["status"], "observation_id": ident, "errors": errors, "order_permissions": False}


if __name__ == "__main__":
    try:
        print(json.dumps(run(), indent=2))
    except (ValueError, KeyError, TypeError, OSError, sqlite3.Error) as exc:
        raise SystemExit(f"Funding observation unavailable: {type(exc).__name__}: {exc}") from exc
