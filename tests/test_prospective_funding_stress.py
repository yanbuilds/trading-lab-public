"""Synthetic paths only; no exchange funding or profitability claim."""
from datetime import timedelta
import copy
from types import SimpleNamespace
import pandas as pd
import pytest

from experiments import frozen_policy_capture as capture, prospective_funding_stress as stress
from tests.test_prospective_paper_report import synthetic_record, specs


@pytest.mark.parametrize("side", [1, -1])
@pytest.mark.parametrize("model_veto", [False, True])
def test_hypothetical_funding_sign_and_matched_cash(side, model_veto, monkeypatch):
    t = capture.START
    records = [synthetic_record(t + timedelta(hours=i)) for i in range(12)]
    idx = pd.date_range(t - timedelta(hours=300), periods=312, freq="h")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "volume": 1.}, index=idx)
    frames = {s: frame.copy() for s in capture.SYMBOLS}
    states = {name: {s: {} for s in capture.SYMBOLS} for name in ("candidate", "ema")}
    for i, record in enumerate(records):
        at = t + timedelta(hours=i)
        for symbol in capture.SYMBOLS:
            for name in states:
                states[name][symbol][at] = {"signal": side if symbol == "BTCUSDT" and i == 0 else 0,
                                            "atr": 2., "exit_long": False, "exit_short": False}
    monkeypatch.setattr(stress.report, "prepare_records", lambda _: (frames, states, t, t + timedelta(hours=11)))
    if model_veto:
        filtered = copy.deepcopy(states["candidate"])
        filtered["BTCUSDT"][t]["signal"] = 0
        monkeypatch.setattr(stress.report, "shadow_states", lambda *_: (copy.deepcopy(filtered), "complete"))
        monkeypatch.setattr(stress.report, "chronos_shadow_states", lambda *_: (copy.deepcopy(filtered), "complete"))
    result = stress.evaluate(records, specs())
    positive = result["scenarios"]["positive_5bp"]["accounts"]["candidate"]
    negative = result["scenarios"]["negative_5bp"]["accounts"]["candidate"]
    assert result["status"] == "hypothetical_funding_stress_not_actual_net_performance"
    assert result["price_only"]["candidate"]["initial"] == result["price_only"]["ema"]["initial"] == 100
    assert positive["funding_net_paid"] is None and positive["funding_coverage_checked"] is False
    assert positive["hypothetical_funding_paid_usdt"] * side > 0
    assert negative["hypothetical_funding_paid_usdt"] * side < 0
    assert (positive["hypothetical_final_usdt"] - negative["hypothetical_final_usdt"]) * side < 0
    assert result["cash_reference"]["final"] == 100 and result["order_permissions"] is False
    if model_veto:
        accounts = result["scenarios"]["positive_5bp"]["accounts"]
        assert accounts["candidate"]["open_positions"] == 1
        assert accounts["ai_shadow"]["open_positions"] == accounts["chronos_shadow"]["open_positions"] == 0
        assert accounts["ai_shadow"]["hypothetical_funding_paid_usdt"] == 0


def test_missing_hour_refuses_stress():
    t = capture.START
    records = [synthetic_record(t + timedelta(hours=i)) for i in (0, 1, 3)]
    with pytest.raises(ValueError, match="missing or duplicate"):
        stress.evaluate(records, specs())


def test_stress_source_must_match_git_and_precede_observation(monkeypatch):
    t = capture.START
    def fake_run(args, **kwargs):
        if args[1] == "show":
            return SimpleNamespace(stdout=(stress.ROOT / args[2].removeprefix("HEAD:")).read_bytes())
        return SimpleNamespace(stdout=str(int((t - timedelta(seconds=1)).timestamp())).encode())
    monkeypatch.setattr(stress.subprocess, "run", fake_run)
    stress.verify_committed_before(t)
    with pytest.raises(ValueError, match="committed after"):
        stress.verify_committed_before(t - timedelta(seconds=2))
    monkeypatch.setattr(stress.subprocess, "run", lambda *a, **k: SimpleNamespace(stdout=b"modified"))
    with pytest.raises(ValueError, match="not verified"):
        stress.verify_committed_before(t)
