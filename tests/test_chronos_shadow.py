"""Synthetic fixtures only; these tests do not establish trading performance."""
from datetime import timedelta
import copy
import hashlib
import sqlite3
from types import SimpleNamespace
import pytest
from experiments import chronos_shadow, frozen_policy_capture as capture, prospective_paper_report as report
from tests.test_prospective_paper_report import synthetic_record, specs
from piia.intelligence.chronos_forecast import MODEL_ID, REVISION, SAFETENSORS_SHA256, run_chronos_tiny


def shadow_record(cutoff):
    record = synthetic_record(cutoff)
    record["capture_script_sha256"] = hashlib.sha256((capture.ROOT / "experiments/frozen_policy_capture.py").read_bytes()).hexdigest()
    return record


def fake_forecast(symbol, cutoff, last=100., median=101.):
    return {"ok": True, "weights_ran": True, "model_id": MODEL_ID, "revision": REVISION,
            "snapshot": {"safetensors_sha256": SAFETENSORS_SHA256}, "library_version": "2.3.2",
            "torch_version": "2.10.0", "device": "cpu", "seed": 0, "num_samples": 20,
            "prediction_length": 24, "context_limit": 300,
            "forecasts": {symbol: {"ok": True, "weights_ran": True, "last_close": last,
                "last_close_at_utc": cutoff, "timeframe": "1H_closed", "bars_used": 300,
                "num_samples": 20, "prediction_length": 24, "quantile_10": [99.] * 24,
                "quantile_50": [median] * 24, "quantile_90": [102.] * 24}}}


def test_shadow_decisions_require_contemporaneous_complete_model_outputs(monkeypatch):
    monkeypatch.setattr(chronos_shadow, "verify_committed_before", lambda _: None)  # explicitly synthetic
    t = capture.START
    records = [shadow_record(t + timedelta(hours=i)) for i in range(3)]
    _, states, _, _ = report.prepare_records(records)
    states["candidate"]["BTCUSDT"][t]["signal"] = 1
    events = {}
    for i, record in enumerate(records):
        signal = 1 if i == 0 else 0
        decisions = {s: {"mechanical_signal": signal if s == "BTCUSDT" else 0,
                         "model_pass": bool(signal and s == "BTCUSDT")} for s in capture.SYMBOLS}
        outputs = {"BTCUSDT": fake_forecast("BTCUSDT", record["cutoff_utc"])} if signal else {}
        events[record["cutoff_utc"]] = {"created_at_utc": (t + timedelta(hours=i, minutes=3)).isoformat(),
            "status": "completed" if signal else "no_candidate", "order_permissions": False,
            "protocol_sha256": chronos_shadow.PINS["CHRONOS_SHADOW_V1.md"], "source_pins": chronos_shadow.PINS,
            "shadow_code_sha256": report.CHRONOS_SHADOW_CODE_SHA256, "decisions": decisions, "model_outputs": outputs}
    filtered, status = report.chronos_shadow_states(records, states["candidate"], events)
    assert status == "complete" and filtered["BTCUSDT"][t]["signal"] == 1
    missing = dict(events); missing.pop(records[1]["cutoff_utc"])
    assert report.chronos_shadow_states(records, states["candidate"], missing)[1] == "missing_chronos_shadow_hour"
    late = copy.deepcopy(events); late[records[0]["cutoff_utc"]]["created_at_utc"] = (t + timedelta(hours=1)).isoformat()
    assert report.chronos_shadow_states(records, states["candidate"], late)[1] == "invalid_or_late_chronos_shadow"
    wrong = copy.deepcopy(events); wrong[records[0]["cutoff_utc"]]["decisions"]["BTCUSDT"]["model_pass"] = False
    assert report.chronos_shadow_states(records, states["candidate"], wrong)[1] == "chronos_decision_mismatch"
    wrong = copy.deepcopy(events); wrong[records[0]["cutoff_utc"]]["model_outputs"]["BTCUSDT"]["seed"] = 1
    assert report.chronos_shadow_states(records, states["candidate"], wrong)[1] == "chronos_output_missing_or_invalid"


def test_no_candidate_is_recorded_once_without_model_call(tmp_path, monkeypatch):
    monkeypatch.setattr(chronos_shadow, "verify_committed_before", lambda _: None)  # explicitly synthetic
    t = capture.START
    record = shadow_record(t)
    created = t + timedelta(minutes=2, seconds=1)
    monkeypatch.setattr(capture, "now_utc", lambda: created)
    monkeypatch.setattr(chronos_shadow.ai_shadow, "now_utc", lambda: created)
    path = tmp_path / "synthetic.sqlite"
    capture.save_observation(path, t, created, record)
    monkeypatch.setattr(chronos_shadow, "run_chronos_tiny", lambda *a, **k: pytest.fail("unexpected inference"))
    result = chronos_shadow.run(path)
    assert result["status"] == "no_candidate" and result["model_called"] is False
    with pytest.raises(sqlite3.IntegrityError, match="UNIQUE"):
        chronos_shadow.run(path)
    assert len(report.read_chronos_events(path)) == 1


def test_commit_must_precede_observation_and_match_worktree(monkeypatch):
    t = capture.START
    def fake_run(args, **kwargs):
        if args[1] == "show":
            return SimpleNamespace(stdout=(chronos_shadow.ROOT / args[2].removeprefix("HEAD:")).read_bytes())
        return SimpleNamespace(stdout=str(int((t - timedelta(seconds=1)).timestamp())).encode())
    monkeypatch.setattr(chronos_shadow.subprocess, "run", fake_run)
    chronos_shadow.verify_committed_before(t)
    with pytest.raises(ValueError, match="not committed before"):
        chronos_shadow.verify_committed_before(t - timedelta(seconds=2))
    monkeypatch.setattr(chronos_shadow.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"tampered"))
    with pytest.raises(ValueError, match="cannot be verified"):
        chronos_shadow.verify_committed_before(t)


def test_model_veto_uses_separate_book_without_changing_candidate(monkeypatch):
    monkeypatch.setattr(chronos_shadow, "verify_committed_before", lambda _: None)  # explicitly synthetic
    import pandas as pd
    t = capture.START
    records = [shadow_record(t + timedelta(hours=i)) for i in range(4)]
    idx = pd.date_range(t - timedelta(hours=300), periods=304, freq="h")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "volume": 1.}, index=idx)
    frames = {s: frame.copy() for s in capture.SYMBOLS}
    states = {name: {s: {} for s in capture.SYMBOLS} for name in ("candidate", "ema")}
    events = {}
    for i, record in enumerate(records):
        cutoff = t + timedelta(hours=i)
        decisions, outputs = {}, {}
        for symbol in capture.SYMBOLS:
            signal = int(symbol == "BTCUSDT" and i == 0)
            states["candidate"][symbol][cutoff] = {"signal": signal, "atr": 2., "exit_long": i == 1, "exit_short": False}
            states["ema"][symbol][cutoff] = {"signal": 0, "atr": 2., "exit_long": False, "exit_short": False}
            decisions[symbol] = {"mechanical_signal": signal, "model_pass": False}
            if signal:
                outputs[symbol] = fake_forecast(symbol, record["cutoff_utc"], median=99.5)
        events[record["cutoff_utc"]] = {"created_at_utc": (cutoff + timedelta(minutes=3)).isoformat(),
            "status": "completed" if outputs else "no_candidate", "order_permissions": False,
            "protocol_sha256": chronos_shadow.PINS["CHRONOS_SHADOW_V1.md"], "source_pins": chronos_shadow.PINS,
            "shadow_code_sha256": report.CHRONOS_SHADOW_CODE_SHA256, "decisions": decisions, "model_outputs": outputs}
    monkeypatch.setattr(report, "prepare_records", lambda _: (frames, states, t, t + timedelta(hours=4)))
    out = report.evaluate(records, specs(), chronos_events=events)
    assert out["chronos_shadow"]["status"] == "complete"
    assert len(out["candidate"]["diagnostic_trade_log"]) == 1
    assert len(out["chronos_shadow"]["result"]["diagnostic_trade_log"]) == 0
    assert out["funding"] == "unknown_excluded" and out["order_permissions"] is False


def test_real_fixed_model_runs_on_labeled_synthetic_prices(tmp_path):
    """Actual local inference on artificial data; no market or performance claim."""
    symbol = "SYNTHETIC_BTC"
    cutoff = "2026-11-18T00:00:00+00:00"
    close = [100 + i * .01 for i in range(300)]
    out = run_chronos_tiny({symbol: {"closes": close, "last_close_at_utc": cutoff, "timeframe": "1H_closed"}},
                           tmp_path / "synthetic_forecast.json", prediction_length=24, num_samples=20,
                           context_limit=300, seed=0, allow_download=False)
    if (out.get("reason") == "forecast_env_python_not_found"
            or (out.get("reason") == "chronos_runner_failed" and out.get("runner_error_type") == "FileNotFoundError")):
        pytest.skip("pinned optional model environment or weights unavailable")
    assert chronos_shadow.validate_forecast(out, symbol, cutoff, close[-1])
    repeated = run_chronos_tiny({symbol: {"closes": close, "last_close_at_utc": cutoff, "timeframe": "1H_closed"}},
                                tmp_path / "synthetic_forecast_repeat.json", prediction_length=24, num_samples=20,
                                context_limit=300, seed=0, allow_download=False)
    assert chronos_shadow.validate_forecast(repeated, symbol, cutoff, close[-1])
    assert repeated["forecasts"][symbol]["quantile_50"] == out["forecasts"][symbol]["quantile_50"]
