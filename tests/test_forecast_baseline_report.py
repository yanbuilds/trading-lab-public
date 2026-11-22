"""Synthetic direction checks; no market results or claim of model skill."""
from datetime import datetime, timezone
from datetime import timedelta
import json
import sqlite3

import pytest

from experiments import forecast_baseline_report as baseline


def test_baselines_filter_only_existing_candidate_and_never_look_ahead():
    cutoff = datetime(2026, 11, 18, 7, tzinfo=timezone.utc)
    bars = [{"close": 100.} for _ in range(300)]
    bars[-1] = {"close": 110.}
    bars[-25] = {"close": 105.}
    bars[-169] = {"close": 115.}
    record = {"cutoff_utc": cutoff.isoformat(), "evidence": {s: {"bars": bars} for s in baseline.capture.SYMBOLS}}
    states = {s: {cutoff: {"signal": -1 if s == "BTCUSDT" else 0, "atr": 1.}} for s in baseline.capture.SYMBOLS}
    short, first = baseline.filtered_states([record], states, 24)
    long, second = baseline.filtered_states([record], states, 168)
    assert first == {"candidate_signals": 1, "direction_agreements": 0}
    assert second == {"candidate_signals": 1, "direction_agreements": 1}
    assert short["BTCUSDT"][cutoff]["signal"] == 0
    assert long["BTCUSDT"][cutoff]["signal"] == -1
    assert all(long[s][cutoff]["signal"] == 0 for s in ("XAUUSDT", "NDX100USDT"))
    bars[-169] = {"close": 110.}
    flat, counts = baseline.filtered_states([record], states, 168)
    assert flat["BTCUSDT"][cutoff]["signal"] == 0 and counts["direction_agreements"] == 0


def test_unregistered_or_missing_data_cannot_produce_baseline():
    with pytest.raises(ValueError, match="unregistered"):
        baseline.filtered_states([], {}, 25)


def test_real_replay_with_flat_synthetic_prices_has_no_model_or_baseline_trades(tmp_path, monkeypatch):
    from tests.test_prospective_paper_report import synthetic_record
    start = baseline.capture.START + timedelta(hours=1)
    monkeypatch.setattr(baseline.segmented, "ACTIVATION", start)
    records = [synthetic_record(start + timedelta(hours=i)) for i in range(3)]
    ledger = tmp_path / "synthetic.sqlite"
    with sqlite3.connect(ledger) as db:
        db.execute(f"PRAGMA application_id={baseline.capture.APP_ID}")
        db.execute("CREATE TABLE observations (id INTEGER PRIMARY KEY, cutoff TEXT, created_at TEXT, payload_json TEXT)")
        for i, row in enumerate(records, 1):
            db.execute("INSERT INTO observations VALUES (?,?,?,?)", (i, row["cutoff_utc"], row["created_at_utc"], json.dumps(row)))
    result = baseline.add_baselines(records, ledger)
    assert result["aggregate_return"] is None
    arms = result["segments"][0]["simple_forecast_baselines"]["results"]
    assert set(arms) == set(baseline.LOOKBACKS)
    assert all(item["status"] == "no_candidate_signals" and item["result"] is None for item in arms.values())


def test_baseline_trade_path_uses_same_fees_and_fill_assumptions(monkeypatch, tmp_path):
    import pandas as pd
    from trading_lab.perps.risk import Spec
    from piia.intelligence.paper.specs import CONTRACTS
    start = baseline.ACTIVATION
    idx = pd.date_range(start - timedelta(hours=300), start + timedelta(hours=3), freq="h")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "volume": 1.}, index=idx)
    frames = {symbol: frame.copy() for symbol in baseline.capture.SYMBOLS}
    records, states = [], {s: {} for s in baseline.capture.SYMBOLS}
    for hour in range(4):
        cutoff = start + timedelta(hours=hour)
        bars = [{"close": 90.} for _ in range(299)] + [{"close": 100.}]
        records.append({"cutoff_utc": cutoff.isoformat(), "evidence": {s: {"bars": bars} for s in baseline.capture.SYMBOLS}})
        for symbol in baseline.capture.SYMBOLS:
            states[symbol][cutoff] = {"signal": int(symbol == "BTCUSDT" and hour == 0), "atr": 2.,
                                      "exit_long": hour == 1, "exit_short": False}
    specs = {CONTRACTS[s]["ccxt_symbol"]: Spec(CONTRACTS[s]["ccxt_symbol"], .001, .001, 5., .0006, 1.)
             for s in baseline.capture.SYMBOLS}
    monkeypatch.setattr(baseline.segmented, "build_report", lambda *a: {"segments": [{"status": "price_only_local_diagnostic"}],
                                                                          "limitations": [], "aggregate_return": None})
    monkeypatch.setattr(baseline.report, "prepare_records", lambda _: (frames, {"candidate": states, "ema": states}, start, start + timedelta(hours=3)))
    monkeypatch.setattr(baseline.report, "recorded_specs", lambda _: specs)
    result = baseline.add_baselines(records, tmp_path / "unused")
    arms = result["segments"][0]["simple_forecast_baselines"]["results"]
    for arm in arms.values():
        assert arm["counts"] == {"candidate_signals": 1, "direction_agreements": 1}
        assert arm["result"]["closed_trades"] == 1
        assert arm["result"]["fees"] > 0 and arm["result"]["return_pct"] < 0
        assert arm["result"]["funding_net_paid"] is None
    assert arms["momentum_24h"]["result"] == arms["drift_168h"]["result"]
    candidate = baseline.report.evaluate(records, specs)["candidate"]
    candidate.pop("diagnostic_trade_log")
    assert arms["momentum_24h"]["result"] == candidate
