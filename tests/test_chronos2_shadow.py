"""Synthetic shadow fixtures; no exchange data, real signals, or trade claims."""
from datetime import timedelta
import hashlib
import json
from pathlib import Path
import sqlite3
from types import SimpleNamespace

import pytest
from experiments import chronos2_shadow as shadow, chronos2_shadow_report as comparison
from tests.test_prospective_paper_report import synthetic_record


def fixture_ledger(path, signal):
    cutoff = shadow.capture.START
    record = synthetic_record(cutoff)
    record["capture_script_sha256"] = hashlib.sha256((shadow.ROOT / "experiments/frozen_policy_capture.py").read_bytes()).hexdigest()
    record["states"]["BTCUSDT"]["candidate_signal"] = signal
    raw = json.dumps(record, sort_keys=True)
    with sqlite3.connect(path) as db:
        db.execute(f"PRAGMA application_id={shadow.ai_shadow.APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)")
        db.execute("INSERT INTO observations VALUES (?, ?, ?, ?)", (1, record["cutoff_utc"], record["created_at_utc"], raw))
    return record, hashlib.sha256(raw.encode()).hexdigest()


def fake_output(median=101.):
    return {"model_revision": shadow.REVISION, "weights_sha256": shadow.WEIGHTS, "config_sha256": shadow.CONFIG,
            "chronos_version": "2.3.2", "torch_version": "2.10.0", "device": "cpu",
            "horizon_hours": 24, "context_bars": 300, "seed": 0, "order_permissions": False,
            "outputs": {"BTCUSDT": {"q10": 99., "median": median, "q90": 102.}}}


def setup(monkeypatch, path, signal):
    record, digest = fixture_ledger(path, signal)
    monkeypatch.setattr(shadow.capture, "verify_policy", lambda: None)
    monkeypatch.setattr(shadow, "source_pins_before", lambda _: {"fixture": "pinned"})
    monkeypatch.setattr(shadow.ai_shadow, "now_utc", lambda: shadow.capture.START + timedelta(minutes=3))
    monkeypatch.setattr(shadow.ai_shadow, "latest_observation", lambda *_: (1, record, digest))
    return record


def test_no_candidate_never_calls_model_and_claim_is_unique(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.sqlite"
    setup(monkeypatch, path, 0)
    monkeypatch.setattr(shadow.subprocess, "run", lambda *a, **k: pytest.fail("must not call model"))
    result = shadow.run(path)
    assert result["status"] == "no_candidate" and result["weights_ran_verified"] is False
    with pytest.raises(sqlite3.IntegrityError):
        shadow.run(path)
    with sqlite3.connect(path) as db:
        with pytest.raises(sqlite3.IntegrityError):
            db.execute("DELETE FROM chronos2_shadow_events")


def test_active_shadow_records_model_decision_and_failure_is_not_flat(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.sqlite"
    setup(monkeypatch, path, 1)
    monkeypatch.setattr(shadow, "default_forecast_python", lambda: Path("synthetic-python"))
    def fake_run(cmd, **kwargs):
        assert kwargs["env"]["HF_HUB_OFFLINE"] == "1"
        Path(cmd[cmd.index("--output") + 1]).write_text(json.dumps(fake_output()), encoding="utf-8")
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(shadow.subprocess, "run", fake_run)
    assert shadow.run(path)["weights_ran_verified"] is True
    with sqlite3.connect(path) as db:
        event = json.loads(db.execute("SELECT payload_json FROM chronos2_shadow_events").fetchone()[0])
    assert event["status"] == "completed" and event["decisions"]["BTCUSDT"]["model_pass"] is True
    assert event["order_permissions"] is False
    assert comparison.read_events(path)[shadow.capture.START.isoformat()]["observation_id"] == 1


def test_missing_model_environment_is_recorded_as_failure(tmp_path, monkeypatch):
    path = tmp_path / "synthetic.sqlite"
    setup(monkeypatch, path, 1)
    monkeypatch.setattr(shadow, "default_forecast_python", lambda: None)
    result = shadow.run(path)
    assert result["status"] == "model_failed"
    assert result["model_process_attempted"] is False and result["weights_ran_verified"] is False
    with sqlite3.connect(path) as db:
        event = json.loads(db.execute("SELECT payload_json FROM chronos2_shadow_events").fetchone()[0])
    assert event["status"] == "model_failed" and event["model_output"] is None
    with pytest.raises(sqlite3.IntegrityError):
        shadow.run(path)


def test_shadow_comparison_rejects_missing_and_manipulated_outputs(monkeypatch):
    cutoff = shadow.capture.START
    record = synthetic_record(cutoff)
    record["states"]["BTCUSDT"]["candidate_signal"] = 1
    candidate = {s: {cutoff: {"signal": 1 if s == "BTCUSDT" else 0}} for s in shadow.ai_shadow.SYMBOLS}
    decisions = {s: {"mechanical_signal": 1 if s == "BTCUSDT" else 0, "model_pass": s == "BTCUSDT"} for s in shadow.ai_shadow.SYMBOLS}
    event = {"status": "completed", "created_at_utc": (cutoff + timedelta(minutes=3)).isoformat(),
             "source_pins": {"fixture": "pinned"}, "order_permissions": False, "decisions": decisions,
             "weights_ran_verified": True, "model_process_attempted": True, "model_output": fake_output()}
    monkeypatch.setattr(shadow, "source_pins_before", lambda _: {"fixture": "pinned"})
    states, count = comparison.validated_states([record], candidate, {record["cutoff_utc"]: event})
    assert count == 1 and states["BTCUSDT"][cutoff]["signal"] == 1
    with pytest.raises(ValueError, match="missing or incompatible"):
        comparison.validated_states([record], candidate, {})
    event["decisions"]["BTCUSDT"]["model_pass"] = False
    with pytest.raises(ValueError, match="inconsistent"):
        comparison.validated_states([record], candidate, {record["cutoff_utc"]: event})
    event["status"] = "model_failed"
    with pytest.raises(ValueError, match="failed"):
        comparison.validated_states([record], candidate, {record["cutoff_utc"]: event})


def test_full_synthetic_comparison_uses_same_book_and_withholds_funding(monkeypatch):
    import numpy as np
    import pandas as pd
    from trading_lab.perps.mean_reversion import attach_range_bollinger
    from experiments import prospective_paper_report as paper

    start = shadow.capture.START
    indices = pd.date_range(start - timedelta(hours=300), periods=303, freq="h", tz="UTC")
    i = np.arange(len(indices))
    close = 100 + .01 * i + .3 * np.sin(i * 2 * np.pi / 15)
    close[299] -= .8  # Synthetic preselected signal fixture, not market data.
    opened = np.r_[close[0], close[:-1]]
    frame = pd.DataFrame({"open": opened, "high": np.maximum(opened, close) + .15,
                          "low": np.minimum(opened, close) - .15, "close": close, "volume": 1.}, index=indices)
    records, events = [], {}
    for hour in range(4):
        cutoff = start + timedelta(hours=hour)
        record = synthetic_record(cutoff)
        bars = frame.iloc[hour:hour + 300]
        record["evidence"]["BTCUSDT"]["bars"] = [
            {"open_at_utc": stamp.isoformat(), **{k: float(row[k]) for k in ("open", "high", "low", "close", "volume")}}
            for stamp, row in bars.iterrows()]
        last = attach_range_bollinger(bars).iloc[-1]
        record["states"]["BTCUSDT"] = {"regime": str(last["regime"]), "candidate_signal": int(last["signal"]),
                                          "kind": "research_observation_not_order"}
        active = [s for s in shadow.ai_shadow.SYMBOLS if record["states"][s]["candidate_signal"]]
        outputs, decisions = {}, {}
        for symbol in shadow.ai_shadow.SYMBOLS:
            signal = record["states"][symbol]["candidate_signal"]
            last_price = record["evidence"][symbol]["bars"][-1]["close"]
            if signal:
                median = last_price * (1.01 if signal > 0 else .99)
                outputs[symbol] = {"q10": median * .99, "median": median, "q90": median * 1.01}
            decisions[symbol] = {"mechanical_signal": signal, "model_pass": bool(signal)}
        model = fake_output()
        model["outputs"] = outputs
        events[record["cutoff_utc"]] = {"status": "completed" if active else "no_candidate",
            "created_at_utc": (cutoff + timedelta(minutes=3)).isoformat(), "source_pins": {"fixture": "pinned"},
            "order_permissions": False, "decisions": decisions, "model_process_attempted": bool(active),
            "weights_ran_verified": bool(active), "model_output": model if active else None}
        records.append(record)
    assert records[0]["states"]["BTCUSDT"]["candidate_signal"] == 1
    monkeypatch.setattr(shadow, "source_pins_before", lambda _: {"fixture": "pinned"})
    result = comparison.evaluate(records, paper.recorded_specs(records), events)
    assert result["active_model_signals"] >= 1
    assert result["chronos2_shadow"]["final"] == result["candidate"]["final"]
    assert result["chronos2_shadow"]["fees"] == result["candidate"]["fees"]
    assert result["chronos2_shadow"]["funding_net_paid"] is None
    assert result["funding"] == "unknown_excluded" and result["order_permissions"] is False
    from experiments import prospective_funding_realization as realization
    pairs = ["BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT"]
    funding = {pair: [{"timestamp": int((start + timedelta(hours=hour)).timestamp() * 1000), "rate": .001}
                      for hour in (1, 2)] for pair in pairs}
    monkeypatch.setattr(realization, "verify_schedule", lambda *_: (funding, {pair: 1 for pair in pairs}))
    estimate = comparison.evaluate_public_funding(records, paper.recorded_specs(records), events, {}, {})
    assert estimate["accounts"]["candidate"]["estimated_final_usdt"] == pytest.approx(
        estimate["accounts"]["chronos2_shadow"]["estimated_final_usdt"])
    assert estimate["accounts"]["candidate"]["estimated_funding_paid_usdt"] > 0
    assert estimate["accounts"]["candidate"]["funding_net_paid"] is None
    assert estimate["status"] == "public_funding_paper_estimate_not_actual_net_performance"
    def missing_funding(*_args):
        raise ValueError("missing funding")
    monkeypatch.setattr(realization, "verify_schedule", missing_funding)
    with pytest.raises(ValueError, match="missing funding"):
        comparison.evaluate_public_funding(records, paper.recorded_specs(records), events, {}, {})
