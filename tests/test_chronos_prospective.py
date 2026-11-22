"""Synthetic fixtures only; no forecast or exchange performance is tested."""
from datetime import datetime, timedelta, timezone
import sqlite3

import pandas as pd
import pytest

from experiments import chronos_prospective as probe

CUTOFF = datetime(2026, 11, 17, 8, 0, tzinfo=timezone.utc)
RECEIVED = CUTOFF + timedelta(minutes=10)


def _payload():
    dates = pd.date_range(pd.Timestamp(CUTOFF) - pd.Timedelta(hours=599), periods=600, freq="h")
    return {"code": "00000", "requestTime": int(RECEIVED.timestamp() * 1000),
            "data": [[str(int(t.timestamp() * 1000)), "100", "101", "99", "100", "1"] for t in dates]}


def test_closed_input_excludes_forming_candle_and_rejects_gaps():
    raw = _payload()
    snap = probe.parse_snapshot(raw, symbol="BTCUSDT", cutoff=CUTOFF, received=RECEIVED)
    assert len(snap["closes"]) == 512
    assert pd.Timestamp(snap["evidence"][-1]["open_at_utc"]) == pd.Timestamp(CUTOFF) - pd.Timedelta(hours=1)
    raw["data"].pop(-2)
    with pytest.raises(ValueError, match="missing closed-bar"):
        probe.parse_snapshot(raw, symbol="BTCUSDT", cutoff=CUTOFF, received=RECEIVED)


def test_snapshot_rejects_clock_drift_and_duplicate():
    raw = _payload()
    raw["requestTime"] -= 300_000
    with pytest.raises(ValueError, match="clocks disagree"):
        probe.parse_snapshot(raw, symbol="BTCUSDT", cutoff=CUTOFF, received=RECEIVED)
    raw = _payload()
    raw["data"].append(raw["data"][-1])
    with pytest.raises(ValueError, match="duplicate"):
        probe.parse_snapshot(raw, symbol="BTCUSDT", cutoff=CUTOFF, received=RECEIVED)


def test_cohort_is_prospective_atomic_and_append_only(tmp_path, monkeypatch):
    created = RECEIVED + timedelta(minutes=1)
    monkeypatch.setattr(probe, "utc_now", lambda: created)
    records = []
    for symbol in probe.SYMBOLS:
        snap = probe.parse_snapshot(_payload(), symbol=symbol, cutoff=CUTOFF, received=RECEIVED)
        snap.pop("closes")
        records.append(snap | {"created_at_utc": created.isoformat(), "target_at_utc": (CUTOFF + timedelta(hours=24)).isoformat(),
            "model_sha256": probe.WEIGHTS_HASH, "config_sha256": probe.CONFIG_HASH, "model_revision": probe.MODEL_REVISION,
            "source": probe.API, "calibration": "not_calibrated", "order_permissions": False,
            "forecast_research_only": {"q10": 98., "median": 100., "q90": 102.}})
    db_path = tmp_path / "synthetic_fixture.sqlite"
    probe.append_cohort(db_path, records)
    with pytest.raises(sqlite3.IntegrityError):
        probe.append_cohort(db_path, records)
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT count(*) FROM forecast_events").fetchone()[0] == 3
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM forecast_events")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE forecast_events SET symbol='BTCUSDT'")
    stale = [r | {"created_at_utc": (created - timedelta(hours=2)).isoformat()} for r in records]
    with pytest.raises(ValueError, match="not prospective"):
        probe.append_cohort(tmp_path / "stale_fixture.sqlite", stale)
    invalid_source = [r | {"received_at_utc": (CUTOFF - timedelta(minutes=1)).isoformat(),
                            "server_time_utc": (CUTOFF - timedelta(minutes=1)).isoformat()} for r in records]
    with pytest.raises(ValueError, match="source timing"):
        probe.append_cohort(tmp_path / "invalid_source.sqlite", invalid_source)
