"""Synthetic evidence only. The archive is not an external timestamp authority."""
from datetime import datetime, timedelta, timezone
import sqlite3
import pytest
from piia.intelligence import forward_archive as archive

NOW = datetime(2026, 11, 17, 8, 0, tzinfo=timezone.utc)


def request(at=NOW):
    return {"cutoff_at_utc": at.isoformat(), "request_hash": "synthetic",
            "independent_llm_review": "NOT_WIRED"}


def test_current_snapshot_is_append_only_and_readable(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "now_utc", lambda: NOW)
    path = tmp_path / "evidence.sqlite"
    receipt = archive.append_current("synthetic-run", request(), {"weights_ran": False}, path=path)
    record = archive.read_event("synthetic-run", path=path)
    assert receipt["historical_point_in_time_proven"] is False
    assert record["is_frozen_policy_observation"] is False and record["order_permissions"] is False
    assert record["forecast"]["weights_ran"] is False
    with sqlite3.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM evidence_events")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE evidence_events SET payload_json = '{}' ")
    with pytest.raises(sqlite3.IntegrityError):
        archive.append_current("synthetic-run", request(), {}, path=path)


def test_future_or_stale_snapshot_never_archived(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "now_utc", lambda: NOW)
    for at in (NOW + timedelta(seconds=1), NOW - timedelta(minutes=5, seconds=1)):
        with pytest.raises(ValueError, match="stale or from the future"):
            archive.append_current("synthetic-run", request(at), {}, path=tmp_path / "new.sqlite")
    assert not (tmp_path / "new.sqlite").exists()


def test_unrelated_database_and_out_of_band_mutation_are_detected(tmp_path, monkeypatch):
    monkeypatch.setattr(archive, "now_utc", lambda: NOW)
    path = tmp_path / "existing.sqlite"
    sqlite3.connect(path).close()
    with pytest.raises(ValueError, match="unrelated"):
        archive.append_current("synthetic", request(), {}, path=path)
    path = tmp_path / "archive.sqlite"
    archive.append_current("synthetic", request(), {}, path=path)
    with sqlite3.connect(path) as db:
        db.execute("DROP TRIGGER no_evidence_update")
        db.execute("UPDATE evidence_events SET payload_json = '{}' ")
    with pytest.raises(ValueError, match="hash mismatch"):
        archive.read_event("synthetic", path=path)
