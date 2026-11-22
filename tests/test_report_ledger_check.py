"""Synthetic coverage metadata; does not simulate or prove returns."""
import sqlite3
from datetime import timedelta
from piia.intelligence.report_ledger_check import APP_ID, POLICY_START, check_report_ledger


def test_coverage_gate_refuses_stale_missing_and_future_reports(tmp_path):
    path = tmp_path / "fixture.sqlite"
    t = POLICY_START
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("CREATE TABLE observations (cutoff TEXT, created_at TEXT)")
        for i in range(3):
            cutoff = t + timedelta(hours=i)
            db.execute("INSERT INTO observations VALUES (?, ?)", (cutoff.isoformat(), (cutoff + timedelta(minutes=2)).isoformat()))
    report = {"first_decision_utc": t.isoformat(), "last_observation_cutoff_utc": (t + timedelta(hours=2)).isoformat(),
              "generated_at_utc": (t + timedelta(hours=2, minutes=5)).isoformat(), "observations": 3}
    now = t + timedelta(hours=3)
    assert check_report_ledger(report, ledger=path, now=now)["ok"]
    assert not check_report_ledger(report, ledger=tmp_path / "absent", now=now)["ok"]
    assert "report_generation_time_invalid" in check_report_ledger(report, ledger=path, now=t)["issues"]
    with sqlite3.connect(path) as db:
        cutoff = t + timedelta(hours=3)
        db.execute("INSERT INTO observations VALUES (?, ?)", (cutoff.isoformat(), (cutoff + timedelta(minutes=2)).isoformat()))
    stale = check_report_ledger(report, ledger=path, now=t + timedelta(hours=4))
    assert not stale["ok"] and "observation_count_mismatch" in stale["issues"]
    assert stale["pnl_recomputed"] is False


def test_gap_and_wrong_ledger_are_rejected(tmp_path):
    path = tmp_path / "fixture.sqlite"
    t = POLICY_START
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={APP_ID}")
        db.execute("CREATE TABLE observations (cutoff TEXT, created_at TEXT)")
        for i in (0, 1, 3):
            cutoff = t + timedelta(hours=i)
            db.execute("INSERT INTO observations VALUES (?, ?)", (cutoff.isoformat(), (cutoff + timedelta(minutes=2)).isoformat()))
    assert not check_report_ledger({}, ledger=path, now=t + timedelta(hours=5))["ok"]
    with sqlite3.connect(path) as db:
        db.execute("PRAGMA application_id=0")
    assert not check_report_ledger({}, ledger=path, now=t + timedelta(hours=5))["ok"]
