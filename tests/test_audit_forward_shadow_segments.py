"""Synthetic prospective claims; no financial performance is implied."""
from datetime import timedelta
import hashlib
import json
import sqlite3

from experiments import audit_forward_shadow_segments as audit
from piia.intelligence.shadow_claim_audit import TABLES
from piia.intelligence.report_ledger_check import APP_ID


def test_segment_audit_is_read_only_and_flags_missing_or_future_events(tmp_path, monkeypatch):
    start = audit.segmented_forward_report_v2.ACTIVATION
    cutoffs = [start, start + timedelta(hours=2)]
    records = [{'cutoff_utc': t.isoformat()} for t in cutoffs]
    monkeypatch.setattr(audit.prospective_paper_report, 'read_observations', lambda _: records)
    path = tmp_path / 'synthetic.sqlite'
    with sqlite3.connect(path) as db:
        db.execute(f'PRAGMA application_id={APP_ID}')
        db.execute('CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)')
        for arm in ('chronos_shadow', 'chronos2_shadow'):
            claims, events = TABLES[arm]
            db.execute(f'CREATE TABLE {claims} (observation_id INTEGER PRIMARY KEY, started_at TEXT, observation_sha256 TEXT)')
            db.execute(f'CREATE TABLE {events} (observation_id INTEGER PRIMARY KEY, created_at TEXT, payload_json TEXT)')
        for ident, cutoff in enumerate(cutoffs, 1):
            observed = cutoff + timedelta(minutes=1)
            started = cutoff + timedelta(minutes=2)
            created = cutoff + timedelta(minutes=3)
            raw = json.dumps({'cutoff_utc': cutoff.isoformat(), 'created_at_utc': observed.isoformat()})
            digest = hashlib.sha256(raw.encode()).hexdigest()
            event = json.dumps({'created_at_utc': created.isoformat(), 'observation_sha256': digest,
                                'observation_id': ident, 'order_permissions': False})
            db.execute('INSERT INTO observations VALUES (?,?,?,?)', (ident, cutoff.isoformat(), observed.isoformat(), raw))
            for arm in ('chronos_shadow', 'chronos2_shadow'):
                claims, events = TABLES[arm]
                db.execute(f'INSERT INTO {claims} VALUES (?,?,?)', (ident, started.isoformat(), digest))
                db.execute(f'INSERT INTO {events} VALUES (?,?,?)', (ident, created.isoformat(), event))
    before = path.read_bytes()
    result = audit.audit_snapshot(path, now=cutoffs[1] + timedelta(minutes=2, seconds=30))
    assert len(result['segments']) == 2 and result['order_permissions'] is False
    first, second = result['segments']
    assert first['arms']['chronos2_shadow']['ok'] is True
    assert first['arms']['ai_shadow']['ok'] is False
    assert second['arms']['chronos2_shadow']['ok'] is False  # Event is still in the future.
    assert path.read_bytes() == before
