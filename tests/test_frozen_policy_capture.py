"""Synthetic fixtures only: no actual signal or future performance."""
from datetime import timedelta
import sqlite3

import pytest

from experiments import frozen_policy_capture as capture


def fixture_response(cutoff, received):
    rows = []
    for i in range(300, -1, -1):
        stamp = cutoff - timedelta(hours=i)
        rows.append([str(int(stamp.timestamp() * 1000)), "100", "101", "99", "100", "1"])
    return {"code": "00000", "requestTime": int(received.timestamp() * 1000), "data": rows}


def fixture_specs(cutoff, received):
    return {"code": "00000", "requestTime": int(received.timestamp() * 1000), "data": [
        {"symbol": symbol, "symbolStatus": "normal", "symbolType": "perpetual", "minTradeNum": "0.01",
         "sizeMultiplier": "0.01", "minTradeUSDT": "5", "takerFeeRate": "0.0006"} for symbol in capture.SYMBOLS]}


def test_policy_source_matches_committed_version():
    capture.verify_policy()


def test_closed_hour_only_and_reject_gaps_duplicates_and_bad_clock():
    cutoff = capture.START + timedelta(hours=1)
    received = cutoff + timedelta(minutes=3)
    data = fixture_response(cutoff, received)
    frame = capture.parse_closed(data, cutoff, received)
    assert len(frame) == 300 and frame.index[-1].to_pydatetime() == cutoff - timedelta(hours=1)
    data["data"].pop(-2)
    with pytest.raises(ValueError, match="missing closed"):
        capture.parse_closed(data, cutoff, received)
    data = fixture_response(cutoff, received)
    data["data"].append(data["data"][-1])
    with pytest.raises(ValueError, match="duplicate"):
        capture.parse_closed(data, cutoff, received)
    data = fixture_response(cutoff, received)
    data["requestTime"] -= 300_000
    with pytest.raises(ValueError, match="clock"):
        capture.parse_closed(data, cutoff, received)


def test_public_contract_specifications_are_timed_and_all_required():
    cutoff = capture.START
    received = cutoff + timedelta(minutes=1)
    data = fixture_specs(cutoff, received)
    parsed = capture.parse_current_specs(data, cutoff, received, raw=b"synthetic response")
    assert parsed["ok"] and set(parsed["contracts"]) == set(capture.SYMBOLS)
    data["data"].pop()
    with pytest.raises(ValueError, match="specifications unavailable"):
        capture.parse_current_specs(data, cutoff, received)
    data = fixture_specs(cutoff, received)
    data["data"][0]["minTradeNum"] = "0"
    with pytest.raises(ValueError, match="invalid or unavailable"):
        capture.parse_current_specs(data, cutoff, received)


def test_before_policy_no_capture_and_ledger_is_append_only(tmp_path, monkeypatch):
    cutoff = capture.START
    created = cutoff + timedelta(minutes=1)
    monkeypatch.setattr(capture, "now_utc", lambda: created)
    path = tmp_path / "fixture.sqlite"
    with pytest.raises(ValueError, match="after policy start"):
        capture.save_observation(path, cutoff - timedelta(hours=1), created, {})
    assert not path.exists()
    capture.save_observation(path, cutoff, created, {"fixture_only": True})
    with pytest.raises(sqlite3.IntegrityError):
        capture.save_observation(path, cutoff, created, {"fixture_only": True})
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT count(*) FROM observations").fetchone()[0] == 1
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM observations")
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("UPDATE observations SET payload_json='{}'")
