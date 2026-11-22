"""Synthetic fixtures; these tests never call a model or place orders."""
from datetime import timedelta
import json
import sqlite3

import pytest

from piia.intelligence import ai_shadow


def ledger_fixture(path, signal=1):
    t = ai_shadow.START
    created = t + timedelta(minutes=1)
    bars = [{"open_at_utc": (t - timedelta(hours=i)).isoformat(), "open": 100., "high": 101.,
             "low": 99., "close": 100., "volume": 1.} for i in range(300, 0, -1)]
    event = {"cutoff_utc": t.isoformat(), "created_at_utc": created.isoformat(), "policy_commit": "69fd853",
             "source": ai_shadow.API, "order_permissions": False, "symbols": list(ai_shadow.SYMBOLS),
             "states": {s: {"candidate_signal": signal if s == "BTCUSDT" else 0, "regime": "range"} for s in ai_shadow.SYMBOLS},
             "evidence": {s: {"received_at_utc": created.isoformat(), "bars": bars} for s in ai_shadow.SYMBOLS}}
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={ai_shadow.APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)")
        db.execute("INSERT INTO observations VALUES (1, ?, ?, ?)", (t.isoformat(), created.isoformat(), json.dumps(event, sort_keys=True)))
    return t


def test_unconfigured_model_does_not_claim_or_fake_output(tmp_path, monkeypatch):
    from piia.intelligence import settings
    path = tmp_path / "fixture.sqlite"
    t = ledger_fixture(path)
    monkeypatch.setattr(ai_shadow, "now_utc", lambda: t + timedelta(minutes=2))
    monkeypatch.setattr(settings, "model_configured", lambda: False)
    result = ai_shadow.run(path)
    assert result["status"] == "needs_model_configuration" and not result["model_called"]
    with sqlite3.connect(path) as db:
        assert not db.execute("SELECT name FROM sqlite_master WHERE name='ai_shadow_claims'").fetchone()


def test_completed_model_is_recorded_once_with_no_order_permission(tmp_path, monkeypatch):
    from piia.intelligence import settings
    from piia.intelligence.agents import workflow
    path = tmp_path / "fixture.sqlite"
    t = ledger_fixture(path)
    monkeypatch.setattr(ai_shadow, "now_utc", lambda: t + timedelta(minutes=2))
    monkeypatch.setattr(settings, "model_configured", lambda: True)
    calls = []
    class FakeResponse:
        def model_dump(self):
            return {"mode": "model", "status": "completed", "prompt_hash": "fixture", "model_provider": "test",
                    "model_name": "fixture", "synthesizer": {"generation": "model", "direction": "long", "suggested_action": "review_paper_plan"}}
    def fake_workflow(**kwargs):
        calls.append(kwargs)
        return FakeResponse()
    monkeypatch.setattr(workflow, "run_workflow", fake_workflow)
    assert ai_shadow.run(path)["status"] == "completed"
    assert len(calls) == 1 and calls[0]["payload"]["focus_symbol"] == "BTCUSDT"
    with sqlite3.connect(path) as db:
        event = json.loads(db.execute("SELECT payload_json FROM ai_shadow_events").fetchone()[0])
        assert event["decisions"]["BTCUSDT"]["ai_pass"] is True
        assert event["order_permissions"] is False and event["independent_llm_review"] == "NOT_WIRED"
        with pytest.raises(sqlite3.IntegrityError, match="append-only"):
            db.execute("DELETE FROM ai_shadow_events")
    with pytest.raises(sqlite3.IntegrityError):
        ai_shadow.run(path)
    assert len(calls) == 1


def test_late_observation_is_rejected_before_model_call(tmp_path, monkeypatch):
    path = tmp_path / "fixture.sqlite"
    t = ledger_fixture(path)
    monkeypatch.setattr(ai_shadow, "now_utc", lambda: t + timedelta(minutes=56))
    with pytest.raises(ValueError, match="stale"):
        ai_shadow.run(path)


def test_failed_model_is_recorded_as_failed_and_cannot_be_retried(tmp_path, monkeypatch):
    from piia.intelligence import settings
    from piia.intelligence.agents import workflow
    path = tmp_path / "fixture.sqlite"
    t = ledger_fixture(path)
    monkeypatch.setattr(ai_shadow, "now_utc", lambda: t + timedelta(minutes=2))
    monkeypatch.setattr(settings, "model_configured", lambda: True)
    class FailedResponse:
        def model_dump(self):
            return {"mode": "model", "status": "model_call_failed", "usage_records": [{"input_tokens": 10, "output_tokens": 3}]}
    monkeypatch.setattr(workflow, "run_workflow", lambda **kwargs: FailedResponse())
    assert ai_shadow.run(path)["status"] == "model_failed"
    with sqlite3.connect(path) as db:
        event = json.loads(db.execute("SELECT payload_json FROM ai_shadow_events").fetchone()[0])
        assert event["decisions"]["BTCUSDT"]["reason"] == "model_failed"
        assert event["model_outputs"]["BTCUSDT"]["usage_records"][0]["input_tokens"] == 10
    with pytest.raises(sqlite3.IntegrityError):
        ai_shadow.run(path)
