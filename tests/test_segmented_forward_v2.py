"""Synthetic-only end-to-end replay tests, including Windows temporary cleanup."""
from datetime import timedelta
import json
import sqlite3
import sys
import tempfile

from experiments import frozen_policy_capture as capture, segmented_forward_report_v2 as v2
from tests.test_prospective_paper_report import synthetic_record


def test_real_sqlite_backup_replay_releases_temporary_handles(tmp_path, monkeypatch):
    start = capture.START + timedelta(hours=1)
    rows = [synthetic_record(start + timedelta(hours=i)) for i in range(3)]
    ledger = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(ledger) as db:
        db.execute(f"PRAGMA application_id={capture.APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)")
        for ident, item in enumerate(rows, 1):
            db.execute("INSERT INTO observations VALUES (?,?,?,?)", (ident, item["cutoff_utc"], item["created_at_utc"], json.dumps(item)))
    output = tmp_path / "synthetic-output.json"
    monkeypatch.setattr(v2, "ACTIVATION", start)
    monkeypatch.setattr(v2, "verify_precommitted", lambda: None)
    monkeypatch.setattr(v2.chronos2_shadow, "source_pins_before", lambda _: {})
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    monkeypatch.setattr(sys, "argv", ["fixture", "--ledger", str(ledger), "--out", str(output)])
    v2.main()
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["segments"][0]["status"] == "price_only_local_diagnostic"
    assert result["aggregate_return"] is None and result["order_permissions"] is False
    assert not list(tmp_path.glob("segmented-forward-*"))
