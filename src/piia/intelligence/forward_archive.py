"""On-demand, append-only local research snapshots; no decision or order route.

Only the live prepare path calls this module. Local timestamps and SQLite
triggers are not independent proof against an administrator or clock change.
These exploratory observations are not a frozen trading-policy cohort.
"""
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import sqlite3

from piia.intelligence.paths import DATA

LEDGER = DATA / "forward_research_evidence.sqlite"
APP_ID = 0x50494645
MAX_LAG = timedelta(minutes=5)


def now_utc():
    return datetime.now(timezone.utc)


def _aware(value):
    value = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timezone-aware timestamps required")
    return value.astimezone(timezone.utc)


def _canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def append_current(run_id: str, request: dict, forecast: dict, *, path: Path = LEDGER):
    """Archive current evidence and model result. Never import a prior file."""
    created = now_utc()
    cutoff = _aware(request["cutoff_at_utc"])
    if not timedelta(0) <= created - cutoff <= MAX_LAG:
        raise ValueError("research snapshot is stale or from the future")
    if not isinstance(run_id, str) or not run_id or not request.get("request_hash"):
        raise ValueError("run identifier and request hash required")
    if request.get("independent_llm_review") != "NOT_WIRED":
        raise ValueError("invalid research review label")
    envelope = {"kind": "exploratory_current_research_evidence", "run_id": run_id,
                "created_at_utc": created.isoformat(), "cutoff_at_utc": cutoff.isoformat(),
                "historical_point_in_time_proven": False, "is_frozen_policy_observation": False,
                "order_permissions": False, "request": request, "forecast": forecast}
    raw = _canonical(envelope)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    existed = path.exists()
    with sqlite3.connect(path) as db:
        app = db.execute("PRAGMA application_id").fetchone()[0]
        if existed and app != APP_ID:
            raise ValueError("refusing unrelated database")
        if not existed:
            db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("CREATE TABLE IF NOT EXISTS evidence_events (run_id TEXT PRIMARY KEY, "
                   "created_at TEXT NOT NULL, cutoff_at TEXT NOT NULL, sha256 TEXT NOT NULL, payload_json TEXT NOT NULL)")
        for action in ("UPDATE", "DELETE"):
            db.execute(f"CREATE TRIGGER IF NOT EXISTS no_evidence_{action.lower()} BEFORE {action} ON evidence_events "
                       "BEGIN SELECT RAISE(ABORT, 'append-only evidence'); END")
        db.execute("INSERT INTO evidence_events VALUES (?, ?, ?, ?, ?)",
                   (run_id, created.isoformat(), cutoff.isoformat(), digest, raw))
    return {"recorded_at_utc": created.isoformat(), "sha256": digest,
            "kind": envelope["kind"], "historical_point_in_time_proven": False,
            "order_permissions": False}


def read_event(run_id: str, *, path: Path = LEDGER):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError("evidence ledger does not exist")
    with sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True) as db:
        if db.execute("PRAGMA application_id").fetchone()[0] != APP_ID:
            raise ValueError("unrelated database")
        row = db.execute("SELECT created_at, cutoff_at, sha256, payload_json FROM evidence_events WHERE run_id = ?", (run_id,)).fetchone()
    if row is None:
        raise ValueError("unknown run")
    created, cutoff, digest, raw = row
    if hashlib.sha256(raw.encode("utf-8")).hexdigest() != digest:
        raise ValueError("evidence hash mismatch")
    event = json.loads(raw)
    if event.get("run_id") != run_id or event.get("created_at_utc") != created or event.get("cutoff_at_utc") != cutoff:
        raise ValueError("evidence ledger metadata mismatch")
    if not timedelta(0) <= _aware(created) - _aware(cutoff) <= MAX_LAG:
        raise ValueError("invalid evidence timing")
    return event
