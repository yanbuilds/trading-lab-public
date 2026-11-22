"""Synthetic reports and mock subprocess; no exchange, model, or real returns."""
import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

from piia.intelligence import paper_report_replay as replay


def test_replay_detects_tampered_values_and_never_passes_secret(tmp_path, monkeypatch):
    ledger = tmp_path / "data" / "piia" / "frozen_policy_v1.sqlite"
    ledger.parent.mkdir(parents=True)
    with sqlite3.connect(ledger) as db:
        db.execute(f"PRAGMA application_id={replay.APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, payload_json TEXT)")
        db.execute("INSERT INTO observations VALUES (1, 'synthetic')")
    report_path = tmp_path / "artifacts" / "prospective_paper_report.json"
    report_path.parent.mkdir()
    saved = {"generated_at_utc": "synthetic", "candidate": {"final": 100}, "order_permissions": False}
    report_path.write_text(json.dumps(saved), encoding="utf-8")
    monkeypatch.setattr(replay, "interpreter", lambda: Path("synthetic-python"))
    monkeypatch.setattr(replay, "check_report_ledger", lambda *_args, **_kwargs: {"ok": True})
    monkeypatch.setenv("PIIA_MODEL_API_KEY", "fixture-secret")
    def fake_run(cmd, **kwargs):
        assert "PIIA_MODEL_API_KEY" not in kwargs["env"]
        output = Path(cmd[cmd.index("--out") + 1])
        output.write_text(json.dumps({**saved, "generated_at_utc": "other-time"}), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(replay.subprocess, "run", fake_run)
    assert replay.verify_saved_report(root=tmp_path)["status"] == "matched_local_replay"
    report_path.write_text(json.dumps({**saved, "candidate": {"final": 999}}), encoding="utf-8")
    result = replay.verify_saved_report(root=tmp_path)
    assert result["status"] == "replay_unavailable_or_mismatch"
    assert result["reason"] == "report_values_differ_from_replay"
    assert result["order_permissions"] is False
    report_path.write_text(json.dumps({**saved, "ai_shadow": {"status": "complete", "result": {}}}), encoding="utf-8")
    assert replay.verify_saved_report(root=tmp_path)["reason"] == "shadow_claim_audit_failed"


def test_real_engine_replays_synthetic_ledger(tmp_path, monkeypatch):
    """End-to-end fixture uses future synthetic dates; never writes the real ledger."""
    import os
    import subprocess
    from datetime import timedelta
    import pytest
    pytest.importorskip("talib", reason="full replay fixture needs the isolated Freqtrade environment")
    from experiments import frozen_policy_capture as capture
    from piia.intelligence.report_ledger_check import check_report_ledger
    from tests.test_prospective_paper_report import synthetic_record

    python = replay.interpreter()
    if python is None:
        pytest.skip("isolated replay environment unavailable")
    ledger = tmp_path / "data" / "piia" / "frozen_policy_v1.sqlite"
    ledger.parent.mkdir(parents=True)
    with sqlite3.connect(ledger) as db:
        db.execute(f"PRAGMA application_id={replay.APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT UNIQUE, created_at TEXT, payload_json TEXT)")
        for i in range(3):
            row = synthetic_record(capture.START + timedelta(hours=i))
            db.execute("INSERT INTO observations(cutoff, created_at, payload_json) VALUES (?, ?, ?)",
                       (row["cutoff_utc"], row["created_at_utc"], json.dumps(row)))
    report_path = tmp_path / "artifacts" / "prospective_paper_report.json"
    report_path.parent.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(replay.ROOT / "src")
    subprocess.run([str(python), "-m", "experiments.prospective_paper_report", "--ledger", str(ledger),
                    "--out", str(report_path)], cwd=str(replay.ROOT), env=env, check=True, capture_output=True, timeout=180)
    # Fixture timestamps are deliberately after the real clock. Only this test
    # substitutes a synthetic report generation time for the metadata check.
    fixture_time = capture.START + timedelta(hours=3)
    def synthetic_clock(report, *, ledger):
        return check_report_ledger({**report, "generated_at_utc": fixture_time.isoformat()},
                                   ledger=ledger, now=fixture_time)
    monkeypatch.setattr(replay, "check_report_ledger", synthetic_clock)
    result = replay.verify_saved_report(root=tmp_path)
    assert result["status"] == "matched_local_replay", result
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["candidate"]["final"] += 1
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert replay.verify_saved_report(root=tmp_path)["reason"] == "report_values_differ_from_replay"
