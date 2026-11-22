"""Synthetic fixtures: no observed performance or live trading."""
from datetime import timedelta
import hashlib
import json
import sqlite3

import pytest

from experiments import chronos_prospective as capture
from experiments import chronos_outcomes as outcomes
from tests.test_chronos_prospective import CUTOFF, RECEIVED, _payload


def event():
    row = capture.parse_snapshot(_payload(), symbol="BTCUSDT", cutoff=CUTOFF, received=RECEIVED)
    row.pop("closes")
    return row | {"created_at_utc": RECEIVED.isoformat(), "target_at_utc": (CUTOFF + timedelta(hours=24)).isoformat(),
                  "model_sha256": capture.WEIGHTS_HASH, "config_sha256": capture.CONFIG_HASH,
                  "model_revision": capture.MODEL_REVISION, "calibration": "not_calibrated", "order_permissions": False,
                  "source": capture.API, "forecast_research_only": {"q10": 98., "median": 102., "q90": 105.}}


def result():
    target = CUTOFF + timedelta(hours=24)
    observed = target + timedelta(minutes=6)
    payload = {"code": "00000", "requestTime": int(observed.timestamp() * 1000), "data": [
        [str(int((target - timedelta(hours=1)).timestamp() * 1000)), "100", "105", "99", "104", "1"]]}
    return payload, observed


def test_scores_model_against_same_start_last_close():
    payload, observed = result()
    scored = outcomes.score(event(), payload, observed)
    assert scored["model_absolute_error"] == 2
    assert scored["baseline_absolute_error"] == 4
    assert scored["model_direction_correct"] is True
    assert scored["momentum_direction_correct"] is None  # flat 24-hour history
    assert scored["research_interval_contains_close"] is True


def test_momentum_uses_known_24_hour_price_and_excludes_ties():
    row = event()
    row["evidence"][-25]["close"] = 101
    encoded = json.dumps(row["evidence"], sort_keys=True, separators=(",", ":")).encode()
    row["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    payload, observed = result()
    scored = outcomes.score(row, payload, observed)
    assert scored["model_direction_correct"] is True
    assert scored["momentum_direction_correct"] is False
    payload["data"][0][4] = "100"
    scored = outcomes.score(row, payload, observed)
    assert scored["model_direction_correct"] is None
    assert scored["momentum_direction_correct"] is None


def test_rejects_immature_missing_duplicate_and_bad_evidence():
    payload, observed = result()
    with pytest.raises(ValueError, match="not yet eligible"):
        outcomes.score(event(), payload, observed - timedelta(minutes=3))
    payload["data"] *= 2
    with pytest.raises(ValueError, match="missing or duplicated"):
        outcomes.score(event(), payload, observed)
    payload["data"] = []
    with pytest.raises(ValueError, match="missing or duplicated"):
        outcomes.score(event(), payload, observed)
    row = event()
    row["evidence"][-1]["close"] = 99
    with pytest.raises(ValueError, match="evidence mismatch"):
        outcomes.score(row, payload, observed)


def test_rejects_rehashed_middle_gap_and_late_source():
    payload, observed = result()
    row = event()
    row["evidence"][100]["open_at_utc"] = row["evidence"][99]["open_at_utc"]
    encoded = json.dumps(row["evidence"], sort_keys=True, separators=(",", ":")).encode()
    row["evidence_sha256"] = hashlib.sha256(encoded).hexdigest()
    with pytest.raises(ValueError, match="noncontiguous"):
        outcomes.score(row, payload, observed)
    row = event()
    row["received_at_utc"] = (RECEIVED + timedelta(seconds=1)).isoformat()
    with pytest.raises(ValueError, match="source timing"):
        outcomes.score(row, payload, observed)


def test_pending_without_network_or_fabricated_outcome(tmp_path, monkeypatch):
    row = event()
    db_path = tmp_path / "synthetic.sqlite"
    monkeypatch.setattr(capture, "utc_now", lambda: RECEIVED)
    records = [row | {"symbol": symbol} for symbol in capture.SYMBOLS]
    capture.append_cohort(db_path, records)
    report = outcomes.run(db_path)
    assert report["not_yet_due"] == 3 and report["newly_scored"] == 0 and not report["errors"]
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT count(*) FROM forecast_outcomes").fetchone()[0] == 0
    payload, observed = result()
    monkeypatch.setattr(capture, "utc_now", lambda: observed)
    class Response:
        def raise_for_status(self):
            pass
        def json(self):
            return payload
    monkeypatch.setattr(outcomes.httpx.Client, "get", lambda *args, **kwargs: Response())
    assert outcomes.run(db_path)["newly_scored"] == 3
    assert outcomes.run(db_path)["newly_scored"] == 0
    with sqlite3.connect(db_path) as db:
        assert db.execute("SELECT count(*) FROM forecast_outcomes").fetchone()[0] == 3
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM forecast_outcomes")
