"""Chronos-t5-tiny runner: unit path in piia-env, actual inference only in forecast-env."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("piia")

from piia.intelligence.chronos_forecast import default_forecast_python, default_hf_cache, env_blocker, run_chronos_tiny, MODEL_ID, REVISION


def test_windows_forecast_environment_is_discovered(tmp_path, monkeypatch):
    from piia.intelligence import chronos_forecast as module
    root = tmp_path / "workspace"
    root.mkdir()
    interpreter = tmp_path / "forecast-env" / "Scripts" / "python.exe"
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("synthetic fixture", encoding="utf-8")
    monkeypatch.setattr(module, "ROOT", root)
    monkeypatch.delenv("PIIA_FORECAST_PYTHON", raising=False)
    assert module.default_forecast_python() == interpreter


@pytest.mark.parametrize("mode,reason", [("failure", "chronos_runner_failed"),
                                         ("missing", "chronos_runner_wrote_no_output"),
                                         ("invalid", "chronos_runner_invalid_output"),
                                         ("timeout", "chronos_inference_timeout")])
def test_prior_forecast_is_never_reused_after_runner_failure(tmp_path, monkeypatch, mode, reason):
    from piia.intelligence import chronos_forecast as module
    monkeypatch.setattr(module, "env_blocker", lambda: None)
    monkeypatch.setattr(module, "default_forecast_python", lambda: Path(sys.executable))
    old = {"ok": True, "weights_ran": True, "model_id": MODEL_ID, "revision": REVISION,
           "snapshot": {"safetensors_sha256": module.SAFETENSORS_SHA256}, "forecasts": {"OLD": {"ok": True}}}
    destination = tmp_path / "forecast.json"
    destination.write_text(json.dumps(old), encoding="utf-8")
    def fake_run(cmd, **kwargs):
        result_path = Path(cmd[cmd.index("--out") + 1])
        assert result_path != destination
        if mode == "timeout":
            raise subprocess.TimeoutExpired(cmd, 1)
        if mode == "invalid":
            result_path.write_text("{broken", encoding="utf-8")
        if mode == "failure":
            result_path.write_text(json.dumps(old), encoding="utf-8")
        return SimpleNamespace(returncode=1 if mode == "failure" else 0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = run_chronos_tiny({"NEW": {"closes": [1]}}, destination)
    assert result["weights_ran"] is False and result["reason"] == reason
    assert json.loads(destination.read_text(encoding="utf-8"))["reason"] == reason
    assert "forecasts" not in result


def test_successful_runner_uses_fresh_temporary_output(tmp_path, monkeypatch):
    from piia.intelligence import chronos_forecast as module
    monkeypatch.setattr(module, "env_blocker", lambda: None)
    monkeypatch.setattr(module, "default_forecast_python", lambda: Path(sys.executable))
    def fake_run(cmd, **kwargs):
        path = Path(cmd[cmd.index("--out") + 1])
        path.write_text(json.dumps({"ok": True, "weights_ran": True, "model_id": MODEL_ID, "revision": REVISION,
                                    "snapshot": {"safetensors_sha256": module.SAFETENSORS_SHA256},
                                    "forecasts": {"NEW": {"ok": True}}}), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    out = tmp_path / "forecast.json"
    result = run_chronos_tiny({"NEW": {"closes": [1]}}, out)
    assert result["weights_ran"] is True and "NEW" in result["forecasts"]
    assert json.loads(out.read_text(encoding="utf-8")) == result


def test_seeded_forecast_requires_runner_to_confirm_seed(tmp_path, monkeypatch):
    from piia.intelligence import chronos_forecast as module
    monkeypatch.setattr(module, "env_blocker", lambda: None)
    monkeypatch.setattr(module, "default_forecast_python", lambda: Path(sys.executable))
    def fake_run(cmd, **kwargs):
        request = json.loads(Path(cmd[cmd.index("--series-json") + 1]).read_text(encoding="utf-8"))
        assert request["seed"] == 0
        payload = {"ok": True, "weights_ran": True, "model_id": MODEL_ID, "revision": REVISION,
                   "snapshot": {"safetensors_sha256": module.SAFETENSORS_SHA256}, "forecasts": {}}
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = run_chronos_tiny({}, tmp_path / "out.json", seed=0)
    assert result["weights_ran"] is False and result["reason"] == "unexpected_seed"


@pytest.mark.parametrize("changes,reason", [
    ({"model_id": "other"}, "unexpected_model_id"),
    ({"revision": "wrong"}, "unexpected_revision"),
    ({"snapshot": {"safetensors_sha256": "wrong"}}, "safetensors_sha256_mismatch"),
    ({"ok": False}, "chronos_runner_inconsistent_status"),
])
def test_unexpected_runner_identity_or_status_cannot_claim_inference(tmp_path, monkeypatch, changes, reason):
    from piia.intelligence import chronos_forecast as module
    monkeypatch.setattr(module, "env_blocker", lambda: None)
    monkeypatch.setattr(module, "default_forecast_python", lambda: Path(sys.executable))
    def fake_run(cmd, **kwargs):
        payload = {"ok": True, "weights_ran": True, "model_id": MODEL_ID, "revision": REVISION,
                   "snapshot": {"safetensors_sha256": module.SAFETENSORS_SHA256}} | changes
        Path(cmd[cmd.index("--out") + 1]).write_text(json.dumps(payload), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, "run", fake_run)
    result = run_chronos_tiny({}, tmp_path / "forecast.json")
    assert result["weights_ran"] is False and result["ok"] is False and result["reason"] == reason


def test_piia_env_does_not_need_chronos_import():
    # This test must pass in piia-env, where chronos is intentionally absent.
    import importlib.util
    assert importlib.util.find_spec("chronos") is None
    blocker = env_blocker()
    python = default_forecast_python()
    if python is None:
        assert blocker is not None
        assert blocker["reason"] == "forecast_env_python_not_found"
    else:
        assert blocker is None
        assert python.is_file()


def test_actual_tiny_inference_in_forecast_env(tmp_path):
    python = default_forecast_python()
    cache = default_hf_cache()
    snapshot = cache / "models--amazon--chronos-t5-tiny" / "snapshots" / REVISION / "model.safetensors"
    if python is None:
        pytest.skip("forecast-env python not found; Chronos remains an optional isolated environment")
    if not snapshot.is_file():
        pytest.skip("pinned chronos-t5-tiny cache is not present; will not download in this test")
    series = {
        "FIXTURE_BTC": {
            "closes": [100.0 + i * 0.1 for i in range(64)],
            "last_closed_at_utc": "2026-11-17T01:00:00+00:00",
        }
    }
    out = tmp_path / "chronos_fixture.json"
    payload = run_chronos_tiny(series, out, prediction_length=4, num_samples=4, context_limit=64, allow_download=False, timeout=180)
    assert payload.get("weights_ran") is True, payload
    assert payload.get("ok") is True
    assert payload.get("model_id") == MODEL_ID
    assert payload.get("revision") == REVISION
    assert payload.get("downloaded") is not False or payload.get("downloaded") in {False, None} or payload.get("downloaded") is False
    forecast = (payload.get("forecasts") or {}).get("FIXTURE_BTC") or {}
    assert forecast.get("ok") is True
    assert forecast.get("weights_ran") is True
    assert isinstance(forecast.get("quantile_50"), list) and len(forecast["quantile_50"]) == 4
    assert payload.get("device") == "cpu"
    assert payload.get("gpu_used") is False
    assert payload.get("trust_remote_code") is False
    assert payload.get("calibration_verified") is False
