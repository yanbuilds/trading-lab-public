"""Synthetic ledger status tests; no market observations or performance claims."""
from datetime import timedelta
import hashlib
import json
import sqlite3

from piia.intelligence.prospective_status import APP_ID, START, collection_status


def test_status_is_read_only_and_marks_pending_claim_and_gaps(tmp_path):
    path = tmp_path / "fixture.sqlite"
    first = START + timedelta(hours=1)
    created = first + timedelta(minutes=2)
    payload = json.dumps({"cutoff_utc": first.isoformat(), "created_at_utc": created.isoformat(), "order_permissions": False})
    digest = hashlib.sha256(payload.encode()).hexdigest()
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)")
        db.execute("INSERT INTO observations VALUES (?,?,?,?)", (1, first.isoformat(), created.isoformat(), payload))
        db.execute("CREATE TABLE chronos2_shadow_claims (observation_id INTEGER PRIMARY KEY, started_at TEXT, observation_sha256 TEXT)")
        db.execute("INSERT INTO chronos2_shadow_claims VALUES (?,?,?)", (1, (created + timedelta(seconds=1)).isoformat(), digest))
    before = path.read_bytes()
    status = collection_status(path=path, now=created + timedelta(minutes=1))
    assert status["status"] == "ok" and status["current_hour_status"] == "recorded"
    assert status["missing_hours_before_first"] == 1 and status["missing_hours_within_recorded_span"] == 0
    assert status["latest_steps"]["chronos2"] == "attempted_without_result"
    assert status["latest_steps"]["funding"] == "not_attempted"
    assert status["latest_steps"]["ai_shadow"] == "not_attempted"
    assert status["minimum_three_contiguous"] is False and status["performance_calculated"] is False
    assert path.read_bytes() == before
    last = first + timedelta(hours=2)
    later = last + timedelta(minutes=2)
    payload2 = json.dumps({"cutoff_utc": last.isoformat(), "created_at_utc": later.isoformat(), "order_permissions": False})
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO observations VALUES (?,?,?,?)", (2, last.isoformat(), later.isoformat(), payload2))
    status = collection_status(path=path, now=later + timedelta(minutes=1))
    assert status["missing_hours_within_recorded_span"] == 1
    assert not status["minimum_three_contiguous"]
    assert [segment["observations"] for segment in status["contiguous_segments"]] == [1, 1]
    assert status["latest_consecutive_hours"] == 1
    last2 = last + timedelta(hours=1)
    created3 = last2 + timedelta(minutes=2)
    payload3 = json.dumps({"cutoff_utc": last2.isoformat(), "created_at_utc": created3.isoformat(), "order_permissions": False})
    with sqlite3.connect(path) as db:
        db.execute("INSERT INTO observations VALUES (?,?,?,?)", (3, last2.isoformat(), created3.isoformat(), payload3))
    status = collection_status(path=path, now=created3 + timedelta(minutes=1))
    assert [segment["observations"] for segment in status["contiguous_segments"]] == [1, 2]
    assert status["latest_consecutive_hours"] == 2
    assert status["performance_calculated"] is False and status["minimum_three_contiguous"] is False


def test_status_refuses_wrong_ledger_and_future_record(tmp_path):
    path = tmp_path / "fixture.sqlite"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE observations (id INTEGER, cutoff TEXT, created_at TEXT, payload_json TEXT)")
    assert collection_status(path=path, now=START)["status"] == "unavailable"
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("INSERT INTO observations VALUES (1,?,?,?)", (START.isoformat(), (START + timedelta(minutes=1)).isoformat(), "{}"))
    assert collection_status(path=path, now=START)["status"] == "unavailable"
    missing = collection_status(path=tmp_path / "missing.sqlite", now=START + timedelta(minutes=51))
    assert missing["status"] == "no_ledger" and missing["current_hour_status"] == "window_closed"
    assert not (tmp_path / "missing.sqlite").exists()


def test_ai_status_distinguishes_valid_event_and_unbound_event(tmp_path):
    path = tmp_path / "fixture.sqlite"
    created = START + timedelta(minutes=1)
    payload = json.dumps({"cutoff_utc": START.isoformat(), "created_at_utc": created.isoformat(), "order_permissions": False})
    digest = hashlib.sha256(payload.encode()).hexdigest()
    event_time = created + timedelta(seconds=2)
    event = {"status": "completed", "observation_id": 1, "observation_sha256": digest,
             "created_at_utc": event_time.isoformat(), "order_permissions": False}
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)")
        db.execute("INSERT INTO observations VALUES (?,?,?,?)", (1, START.isoformat(), created.isoformat(), payload))
        db.execute("CREATE TABLE ai_shadow_claims (observation_id INTEGER PRIMARY KEY, started_at TEXT, observation_sha256 TEXT)")
        db.execute("CREATE TABLE ai_shadow_events (observation_id INTEGER PRIMARY KEY, created_at TEXT, payload_json TEXT)")
        db.execute("INSERT INTO ai_shadow_claims VALUES (?,?,?)", (1, (created + timedelta(seconds=1)).isoformat(), digest))
        db.execute("INSERT INTO ai_shadow_events VALUES (?,?,?)", (1, event_time.isoformat(), json.dumps(event)))
    assert collection_status(path=path, now=event_time)["latest_steps"]["ai_shadow"] == "recorded_completed"
    event["observation_sha256"] = "incorrect"
    with sqlite3.connect(path) as db:
        db.execute("UPDATE ai_shadow_events SET payload_json=?", (json.dumps(event),))
    assert collection_status(path=path, now=event_time)["latest_steps"]["ai_shadow"] == "invalid_binding"
