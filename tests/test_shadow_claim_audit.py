"""Synthetic records test fail-closed model-claim checks."""
from datetime import datetime, timedelta, timezone
import hashlib
import json
import sqlite3

import pytest

from piia.intelligence.shadow_claim_audit import TABLES, audit_claims
from piia.intelligence.report_ledger_check import APP_ID


@pytest.mark.parametrize("arm", list(TABLES))
def test_claim_requires_matching_prior_observation_and_event(tmp_path, arm):
    path = tmp_path / "synthetic.sqlite"
    cutoff = datetime(2026, 11, 18, tzinfo=timezone.utc)
    observed, started, created = [cutoff + timedelta(minutes=n) for n in (1, 2, 3)]
    raw = json.dumps({"cutoff_utc": cutoff.isoformat(), "created_at_utc": observed.isoformat()})
    digest = hashlib.sha256(raw.encode()).hexdigest()
    event = {"created_at_utc": created.isoformat(), "observation_sha256": digest,
             "observation_id": 1, "order_permissions": False}
    claim, events = TABLES[arm]
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)")
        db.execute(f"CREATE TABLE {claim} (observation_id INTEGER PRIMARY KEY, started_at TEXT, observation_sha256 TEXT)")
        db.execute(f"CREATE TABLE {events} (observation_id INTEGER PRIMARY KEY, created_at TEXT, payload_json TEXT)")
        db.execute("INSERT INTO observations VALUES (?,?,?,?)", (1, cutoff.isoformat(), observed.isoformat(), raw))
        db.execute(f"INSERT INTO {events} VALUES (?,?,?)", (1, created.isoformat(), json.dumps(event)))
    assert not audit_claims(path, arm, now=created)["ok"]
    with sqlite3.connect(path) as db:
        db.execute(f"INSERT INTO {claim} VALUES (?,?,?)", (1, started.isoformat(), digest))
    before = path.read_bytes()
    assert audit_claims(path, arm, now=created)["ok"]
    assert audit_claims(path, arm, now=created, cutoffs=[cutoff.isoformat()])["ok"]
    assert not audit_claims(path, arm, now=created, cutoffs=[(cutoff + timedelta(hours=1)).isoformat()])["ok"]
    assert path.read_bytes() == before
    assert not audit_claims(path, arm, now=started)["ok"]
    with sqlite3.connect(path) as db:
        db.execute(f"UPDATE {claim} SET started_at=?", ((created + timedelta(minutes=1)).isoformat(),))
    assert not audit_claims(path, arm, now=created + timedelta(minutes=2))["ok"]
