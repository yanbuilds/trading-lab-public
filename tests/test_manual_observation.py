"""Synthetic subprocess results; no live observations or model calls."""
import json
from pathlib import Path
from types import SimpleNamespace
import subprocess

from piia.intelligence import manual_observation as manual


def test_segmented_report_is_explicit_local_and_does_not_send_model_keys(monkeypatch):
    monkeypatch.setattr(manual, "interpreter", lambda: Path("synthetic-python"))
    monkeypatch.setenv("PIIA_MODEL_API_KEY", "fixture-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert "PIIA_MODEL_API_KEY" not in kwargs["env"] and "OPENAI_API_KEY" not in kwargs["env"]
        assert "experiments.guarded_forecast_report" in cmd
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"status": "all_segments_no_aggregate",
            "aggregate_return": None, "order_permissions": False, "segments": [],
            "forecast_baseline_protocol": "FORECAST_BASELINES_V1",
            "post_hoc_display_guard": {"status": "applied"}}), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(manual.subprocess, "run", fake_run)
    result = manual.run_segmented_report()
    assert len(calls) == 1 and result["status"] == "all_segments_no_aggregate"
    assert result["segments"] == [] and result["order_permissions"] is False


def test_funding_segment_report_uses_only_public_module_and_rejects_unsafe_output(monkeypatch):
    monkeypatch.setattr(manual, "interpreter", lambda: Path("synthetic-python"))
    monkeypatch.setenv("PIIA_MODEL_API_KEY", "fixture-secret")
    monkeypatch.setenv("OPENAI_API_KEY", "fixture-secret")
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd)
        assert "experiments.funding_segment_comparison" in cmd
        assert "PIIA_MODEL_API_KEY" not in kwargs["env"] and "OPENAI_API_KEY" not in kwargs["env"]
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps({"status": "all_segments_no_aggregate",
            "aggregate_return": None, "order_permissions": len(calls) != 1, "segments": []}), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(manual.subprocess, "run", run)
    assert manual.run_funding_segment_report()["status"] == "all_segments_no_aggregate"
    assert manual.run_funding_segment_report()["status"] == "unavailable"


def test_one_manual_capture_runs_only_local_research_modules(monkeypatch):
    calls = []
    monkeypatch.setenv("PIIA_MODEL_API_KEY", "fixture-secret")
    monkeypatch.setattr(manual, "interpreter", lambda: Path("synthetic-python"))
    monkeypatch.setattr(manual, "observation_id_for_cutoff", lambda _: 42)
    def run(cmd, **kwargs):
        calls.append(cmd[-1])
        assert "PIIA_MODEL_API_KEY" not in kwargs["env"]
        status = {"experiments.frozen_policy_capture": "captured_research_only",
                  "experiments.funding_observer": "complete", "experiments.chronos_shadow": "no_candidate"}[cmd[-1]]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": status, "order_permissions": False, "observation_id": 42}))
    monkeypatch.setattr(manual.subprocess, "run", run)
    result = manual.capture_once()
    assert calls == ["experiments.frozen_policy_capture", "experiments.funding_observer", "experiments.chronos_shadow"]
    assert result["status"] == "captured_research_only" and not result["order_permissions"]
    assert result["scheduled"] is False and result["model_api_called"] is False


def test_failed_or_ambiguous_policy_capture_stops_without_retry(monkeypatch):
    monkeypatch.setattr(manual, "interpreter", lambda: Path("synthetic-python"))
    calls = []
    def timeout(cmd, **kwargs):
        calls.append(cmd[-1])
        raise subprocess.TimeoutExpired(cmd, 1)
    monkeypatch.setattr(manual.subprocess, "run", timeout)
    result = manual.capture_once()
    assert calls == ["experiments.frozen_policy_capture"]
    assert result["steps"]["policy"]["status"] == "outcome_unknown"
    assert result["status"] == "unavailable"


def test_partial_funding_error_is_visible_and_not_filled(monkeypatch):
    monkeypatch.setattr(manual, "interpreter", lambda: Path("synthetic-python"))
    monkeypatch.setattr(manual, "observation_id_for_cutoff", lambda _: 42)
    def run(cmd, **kwargs):
        status = {"experiments.frozen_policy_capture": "captured_research_only",
                  "experiments.funding_observer": "incomplete", "experiments.chronos_shadow": "model_failed"}[cmd[-1]]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": status, "order_permissions": False, "observation_id": 42}))
    monkeypatch.setattr(manual.subprocess, "run", run)
    result = manual.capture_once()
    assert result["steps"]["funding"]["status"] == "incomplete"
    assert result["steps"]["chronos"]["status"] == "model_failed"


def test_different_observation_id_stops_remaining_steps(monkeypatch):
    monkeypatch.setattr(manual, "interpreter", lambda: Path("synthetic-python"))
    monkeypatch.setattr(manual, "observation_id_for_cutoff", lambda _: 42)
    calls = []
    def run(cmd, **kwargs):
        calls.append(cmd[-1])
        status = "captured_research_only" if len(calls) == 1 else "complete"
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": status, "order_permissions": False, "observation_id": 43}))
    monkeypatch.setattr(manual.subprocess, "run", run)
    result = manual.capture_once()
    assert calls == ["experiments.frozen_policy_capture", "experiments.funding_observer"]
    assert result["steps"]["funding"]["status"] == "observation_mismatch"
    assert not result["order_permissions"]


def test_resolve_observation_is_read_only_and_checks_database_identity(monkeypatch, tmp_path):
    import sqlite3
    monkeypatch.setattr(manual, "ROOT", tmp_path)
    ledger = tmp_path / "data" / "piia" / "frozen_policy_v1.sqlite"
    ledger.parent.mkdir(parents=True)
    with sqlite3.connect(ledger) as db:
        db.execute("PRAGMA application_id=1179670065")  # 0x46505631
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT)")
        db.execute("INSERT INTO observations VALUES (42, 'fixture-cutoff')")
    assert manual.observation_id_for_cutoff("fixture-cutoff") == 42
    with sqlite3.connect(ledger) as db:
        db.execute("PRAGMA application_id=0")
    import pytest
    with pytest.raises(ValueError):
        manual.observation_id_for_cutoff("fixture-cutoff")


def test_chronos2_action_never_retries_or_sends_model_key(monkeypatch):
    monkeypatch.setattr(manual, "interpreter", lambda: Path("synthetic-python"))
    monkeypatch.setenv("PIIA_MODEL_API_KEY", "fixture-secret")
    calls = []
    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        assert "PIIA_MODEL_API_KEY" not in kwargs["env"]
        return SimpleNamespace(returncode=0, stdout=json.dumps({"status": "no_candidate", "observation_id": 1,
            "model_process_attempted": False, "weights_ran_verified": False, "order_permissions": False}))
    monkeypatch.setattr(manual.subprocess, "run", fake_run)
    result = manual.capture_chronos2_once()
    assert result["status"] == "no_candidate" and not result["weights_ran_verified"]
    assert len(calls) == 1 and calls[0][-1] == "experiments.chronos2_shadow"
