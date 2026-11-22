"""Synthetic fixtures: no real historical observations or trading claims."""
from datetime import timedelta

import pandas as pd
import pytest

from experiments import frozen_policy_capture as capture
from experiments import prospective_paper_report as report
from trading_lab.perps.mean_reversion import attach_range_bollinger
from trading_lab.perps.risk import Spec
from piia.intelligence.paper.specs import CONTRACTS
from tests.test_frozen_policy_capture import fixture_specs


def synthetic_record(cutoff):
    received = cutoff + timedelta(minutes=2)
    evidence, states = {}, {}
    for symbol in capture.SYMBOLS:
        bars = []
        for i in range(300, 0, -1):
            t = cutoff - timedelta(hours=i)
            bars.append({"open_at_utc": t.isoformat(), "open": 100., "high": 101.,
                         "low": 99., "close": 100., "volume": 1.})
        frame = pd.DataFrame(bars).set_index("open_at_utc")
        frame.index = pd.to_datetime(frame.index, utc=True)
        last = attach_range_bollinger(frame).iloc[-1]
        evidence[symbol] = {"received_at_utc": received.isoformat(),
                            "provider_time_ms": int(received.timestamp() * 1000), "bars": bars}
        states[symbol] = {"regime": str(last["regime"]), "candidate_signal": int(last["signal"]),
                          "kind": "research_observation_not_order"}
    snapshot = capture.parse_current_specs(fixture_specs(cutoff, received), cutoff, received, raw=b"synthetic response")
    return {"cutoff_utc": cutoff.isoformat(), "created_at_utc": (received + timedelta(seconds=1)).isoformat(),
            "policy_commit": "69fd853", "source": capture.API, "symbols": list(capture.SYMBOLS),
            "states": states, "evidence": evidence, "specs_snapshot": snapshot, "order_permissions": False,
            "next_open_fill_available": False, "funding": "unknown",
            "status": "local_unverified_research_observation"}


def specs():
    return {CONTRACTS[s]["ccxt_symbol"]: Spec(CONTRACTS[s]["ccxt_symbol"], .001, .001, 5)
            for s in capture.SYMBOLS}


def test_no_data_or_missing_hour_never_produces_performance():
    t = capture.START
    with pytest.raises(ValueError, match="at least three"):
        report.evaluate([synthetic_record(t)], specs())
    with pytest.raises(ValueError, match="missing or duplicate"):
        report.evaluate([synthetic_record(t), synthetic_record(t + timedelta(hours=1)),
                         synthetic_record(t + timedelta(hours=3))], specs())


def test_reads_only_existing_ledger_even_with_unicode_path(tmp_path, monkeypatch):
    t = capture.START
    row = synthetic_record(t)
    created = t + timedelta(minutes=2, seconds=1)
    monkeypatch.setattr(capture, "now_utc", lambda: created)
    path = tmp_path / "带空格 fixture.sqlite"
    with pytest.raises(FileNotFoundError):
        report.read_observations(path)
    capture.save_observation(path, t, created, row)
    assert report.read_observations(path) == [row]


def test_missing_or_changed_contemporaneous_contract_specs_block_report():
    t = capture.START
    rows = [synthetic_record(t + timedelta(hours=i)) for i in range(3)]
    actual = report.recorded_specs(rows)
    assert actual["XAU/USDT:USDT"].min_qty == .01
    rows[1]["specs_snapshot"]["contracts"]["XAUUSDT"]["min_qty"] = "0.02"
    with pytest.raises(ValueError, match="specifications changed"):
        report.recorded_specs(rows)
    rows[1]["specs_snapshot"] = None
    with pytest.raises(ValueError, match="specifications unavailable"):
        report.recorded_specs(rows)


def test_contiguous_observations_report_unknown_funding_and_late_entries():
    t = capture.START
    rows = [synthetic_record(t + timedelta(hours=i)) for i in range(3)]
    result = report.evaluate(rows, specs())
    assert result["observations"] == 3 and result["order_permissions"] is False
    assert result["candidate"]["funding_net_paid"] is None and result["ema"]["funding_net_paid"] is None
    assert result["cash_reference"]["final"] == 100
    rows[1]["created_at_utc"] = (t + timedelta(hours=2, minutes=1)).isoformat()
    with pytest.raises(ValueError, match="nonprospective"):
        report.evaluate(rows, specs())


def test_tampered_frozen_signal_or_revised_bar_fails_closed():
    t = capture.START
    rows = [synthetic_record(t + timedelta(hours=i)) for i in range(3)]
    rows[0]["states"]["BTCUSDT"]["candidate_signal"] = 1
    with pytest.raises(ValueError, match="does not match"):
        report.evaluate(rows, specs())
    rows = [synthetic_record(t + timedelta(hours=i)) for i in range(3)]
    rows[1]["evidence"]["BTCUSDT"]["bars"][-2]["close"] = 100.5
    with pytest.raises(ValueError, match="revised or disagree"):
        report.evaluate(rows, specs())


def test_recorded_signal_cannot_fill_before_following_hour(monkeypatch):
    t = capture.START
    idx = pd.date_range(t - timedelta(hours=300), periods=304, freq="h")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "volume": 1.}, index=idx)
    frames = {s: frame.copy() for s in capture.SYMBOLS}
    states = {name: {s: {} for s in capture.SYMBOLS} for name in ("candidate", "ema")}
    for symbol in capture.SYMBOLS:
        for hour in range(4):
            cutoff = t + timedelta(hours=hour)
            states["candidate"][symbol][cutoff] = {"signal": int(symbol == "BTCUSDT" and hour == 0),
                                                    "atr": 2., "exit_long": hour == 1, "exit_short": False}
            states["ema"][symbol][cutoff] = {"signal": 0, "atr": 2., "exit_long": False, "exit_short": False}
    monkeypatch.setattr(report, "prepare_records", lambda _: (frames, states, t, t + timedelta(hours=4)))
    result = report.evaluate([{}, {}, {}, {}], specs())
    trade = result["candidate"]["diagnostic_trade_log"][0]
    assert pd.Timestamp(trade["entry_time"]) == t + timedelta(hours=1)
    assert pd.Timestamp(trade["exit_time"]) == t + timedelta(hours=2)


def test_ai_shadow_missing_or_late_is_never_a_valid_ablation():
    from piia.intelligence import ai_shadow
    import hashlib
    t = capture.START
    records = [synthetic_record(t + timedelta(hours=i)) for i in range(3)]
    _, states, _, _ = report.prepare_records(records)
    states["candidate"]["BTCUSDT"][t]["signal"] = 1
    def event(i):
        signal = 1 if i == 0 else 0
        decisions = {s: {"mechanical_signal": signal if s == "BTCUSDT" else 0,
                         "ai_pass": bool(signal and s == "BTCUSDT")} for s in capture.SYMBOLS}
        output = {"BTCUSDT": {"mode": "model", "status": "completed", "model_name": "fixture", "model_provider": "fixture",
                               "prompt_hash": "frozen", "synthesizer": {"generation": "model", "direction": "long",
                               "suggested_action": "review_paper_plan"}}} if signal else {}
        return {"created_at_utc": (t + timedelta(hours=i, minutes=3)).isoformat(), "status": "completed" if signal else "no_candidate",
                "order_permissions": False, "protocol_sha256": ai_shadow.PROTOCOL_SHA256,
                "shadow_code_sha256": report.AI_SHADOW_CODE_SHA256,
                "question_sha256": hashlib.sha256(ai_shadow.QUESTION.encode()).hexdigest(), "decisions": decisions, "model_outputs": output}
    shadow = {r["cutoff_utc"]: event(i) for i, r in enumerate(records)}
    filtered, reason = report.shadow_states(records, states["candidate"], shadow)
    assert reason == "complete" and filtered["BTCUSDT"][t]["signal"] == 1
    missing = dict(shadow)
    missing.pop(records[1]["cutoff_utc"])
    assert report.shadow_states(records, states["candidate"], missing)[1] == "missing_ai_shadow_hour"
    shadow[records[0]["cutoff_utc"]]["created_at_utc"] = (t + timedelta(hours=1)).isoformat()
    assert report.shadow_states(records, states["candidate"], shadow)[1] == "invalid_or_late_ai_shadow"


def test_ai_shadow_is_a_separate_book_and_can_veto_only_entry(monkeypatch):
    from piia.intelligence import ai_shadow
    import hashlib
    t = capture.START
    records = [synthetic_record(t + timedelta(hours=i)) for i in range(4)]
    idx = pd.date_range(t - timedelta(hours=300), periods=304, freq="h")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "volume": 1.}, index=idx)
    frames = {s: frame.copy() for s in capture.SYMBOLS}
    states = {name: {s: {} for s in capture.SYMBOLS} for name in ("candidate", "ema")}
    shadow = {}
    for hour, record in enumerate(records):
        cutoff = t + timedelta(hours=hour)
        decisions, outputs = {}, {}
        for symbol in capture.SYMBOLS:
            signal = int(symbol == "BTCUSDT" and hour == 0)
            states["candidate"][symbol][cutoff] = {"signal": signal, "atr": 2., "exit_long": hour == 1, "exit_short": False}
            states["ema"][symbol][cutoff] = {"signal": 0, "atr": 2., "exit_long": False, "exit_short": False}
            decisions[symbol] = {"mechanical_signal": signal, "ai_pass": False}
            if signal:
                outputs[symbol] = {"mode": "model", "status": "completed", "model_name": "fixture", "model_provider": "fixture",
                                   "prompt_hash": "pinned", "synthesizer": {"generation": "model", "direction": "short",
                                   "suggested_action": "review_paper_plan"}}
        shadow[record["cutoff_utc"]] = {"created_at_utc": (cutoff + timedelta(minutes=3)).isoformat(),
            "status": "completed" if outputs else "no_candidate", "order_permissions": False,
            "shadow_code_sha256": report.AI_SHADOW_CODE_SHA256,
            "protocol_sha256": ai_shadow.PROTOCOL_SHA256, "question_sha256": hashlib.sha256(ai_shadow.QUESTION.encode()).hexdigest(),
            "decisions": decisions, "model_outputs": outputs}
    monkeypatch.setattr(report, "prepare_records", lambda _: (frames, states, t, t + timedelta(hours=4)))
    result = report.evaluate(records, specs(), shadow)
    assert result["candidate"]["closed_trades"] == 1
    assert result["ai_shadow"]["status"] == "complete" and result["ai_shadow"]["result"]["closed_trades"] == 0
