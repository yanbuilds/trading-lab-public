"""Synthetic data only; no fixture is a market-performance result."""
from datetime import timedelta
import pytest

from experiments import funding_segment_comparison as comparison


def test_missing_funding_withholds_segment_and_never_fetches_history():
    t = comparison.ACTIVATION
    records = [{"cutoff_utc": (t + timedelta(hours=i)).isoformat()} for i in range(3)]
    calls = []
    result = comparison.build_report(records, None, {}, lambda *args: calls.append(args))
    assert result["segments"][0]["status"] == "unavailable"
    assert result["segments"][0]["result"] is None
    assert result["aggregate_return"] is None
    assert not calls


def test_all_available_arms_use_the_same_funding_and_cost_engine(monkeypatch):
    t = comparison.ACTIVATION
    records = [{"cutoff_utc": (t + timedelta(hours=i)).isoformat()} for i in range(3)]
    arms = {"candidate": "candidate", "ema": "ema"}
    monkeypatch.setattr(comparison.report, "prepare_records", lambda _: ("frames", arms, t, t + timedelta(hours=2)))
    monkeypatch.setattr(comparison.report, "recorded_specs", lambda _: "specs")
    monkeypatch.setattr(comparison.funding, "verify_schedule", lambda *args: ("public_rates", "intervals"))
    monkeypatch.setattr(comparison.baseline, "filtered_states", lambda _r, _c, n: (str(n), {"candidate_signals": 1}))
    monkeypatch.setattr(comparison, "audit_claims", lambda *args, **kwargs: {"ok": False})
    monkeypatch.setattr(comparison.stress, "_prepared", lambda frames, state: state)
    calls = []
    def simulate(frame, specs, **kwargs):
        calls.append((frame, specs, kwargs))
        return {"trades": [], "nav": [], "final": 100., "return_pct": 0., "funding_net_paid": 0.}
    monkeypatch.setattr(comparison, "simulate_book", simulate)
    monkeypatch.setattr(comparison, "summarize", lambda raw: dict(raw))
    result = comparison.evaluate_segment(records, None, {}, {})
    assert set(result["accounts"]) == {"candidate", "ema", "momentum_24h", "drift_168h"}
    assert set(result["unavailable_arms"]) == {"ai_shadow", "chronos_shadow", "chronos2_shadow"}
    assert all(c[1] == "specs" and c[2]["funding"] == "public_rates" and
               c[2]["funding_intervals"] == "intervals" and c[2]["capital"] == 100 for c in calls)
    assert all(v["funding_net_paid"] is None and not v["funding_coverage_checked"] for v in result["accounts"].values())


def test_real_engine_replays_flat_synthetic_segment_with_public_funding(monkeypatch, tmp_path):
    from tests.test_prospective_paper_report import synthetic_record
    from piia.intelligence.paper.specs import CONTRACTS
    t = comparison.ACTIVATION
    records = [synthetic_record(t + timedelta(hours=i)) for i in range(3)]
    settled = int((t + timedelta(hours=1)).timestamp() * 1000)
    rates = {CONTRACTS[s]["ccxt_symbol"]: [{"timestamp": settled, "rate": .0001}]
             for s in comparison.capture.SYMBOLS}
    intervals = {pair: 8 for pair in rates}
    monkeypatch.setattr(comparison.funding, "verify_schedule", lambda *args: (rates, intervals))
    monkeypatch.setattr(comparison, "audit_claims", lambda *args, **kwargs: {"ok": False})
    result = comparison.evaluate_segment(records, tmp_path / "synthetic.sqlite", {}, {})
    assert set(result["accounts"]) == {"candidate", "ema"}
    assert all(account["closed_trades"] == 0 and account["estimated_final_usdt"] == 100
               for account in result["accounts"].values())
    assert result["unavailable_arms"]["momentum_24h"] == "no_candidate_signals"
    assert result["unavailable_arms"]["drift_168h"] == "no_candidate_signals"


def test_real_engine_charges_identical_costs_for_identical_baseline_signals(monkeypatch, tmp_path):
    import pandas as pd
    from trading_lab.perps.risk import Spec
    from piia.intelligence.paper.specs import CONTRACTS
    t = comparison.ACTIVATION
    index = pd.date_range(t - timedelta(hours=300), t + timedelta(hours=3), freq="h")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "volume": 1.}, index=index)
    frames = {s: frame.copy() for s in comparison.capture.SYMBOLS}
    records, candidate = [], {s: {} for s in comparison.capture.SYMBOLS}
    for hour in range(4):
        cutoff = t + timedelta(hours=hour)
        bars = [{"close": 90.} for _ in range(299)] + [{"close": 100.}]
        records.append({"cutoff_utc": cutoff.isoformat(), "evidence": {s: {"bars": bars} for s in comparison.capture.SYMBOLS}})
        for symbol in comparison.capture.SYMBOLS:
            candidate[symbol][cutoff] = {"signal": int(symbol == "BTCUSDT" and hour == 0), "atr": 2.,
                                         "exit_long": hour == 1, "exit_short": False}
    specs = {CONTRACTS[s]["ccxt_symbol"]: Spec(CONTRACTS[s]["ccxt_symbol"], .001, .001, 5., .0006, 1.)
             for s in comparison.capture.SYMBOLS}
    settled = int((t + timedelta(hours=1)).timestamp() * 1000)
    rates = {pair: [{"timestamp": settled, "rate": .0001}] for pair in specs}
    monkeypatch.setattr(comparison.report, "prepare_records", lambda _: (frames, {"candidate": candidate, "ema": candidate}, t, t + timedelta(hours=3)))
    monkeypatch.setattr(comparison.report, "recorded_specs", lambda _: specs)
    monkeypatch.setattr(comparison.funding, "verify_schedule", lambda *args: (rates, {pair: 8 for pair in specs}))
    monkeypatch.setattr(comparison, "audit_claims", lambda *args, **kwargs: {"ok": False})
    accounts = comparison.evaluate_segment(records, tmp_path / "synthetic.sqlite", {}, {})["accounts"]
    assert accounts["candidate"]["closed_trades"] == 1
    assert accounts["candidate"]["fees"] > 0
    assert accounts["candidate"] == accounts["momentum_24h"] == accounts["drift_168h"]


@pytest.mark.parametrize("side", [1, -1])
def test_funding_is_charged_during_open_position_for_each_comparable_arm(monkeypatch, tmp_path, side):
    import pandas as pd
    from trading_lab.perps.risk import Spec
    from piia.intelligence.paper.specs import CONTRACTS
    t = comparison.ACTIVATION
    index = pd.date_range(t - timedelta(hours=300), t + timedelta(hours=11), freq="h")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100., "volume": 1.}, index=index)
    frames = {symbol: frame.copy() for symbol in comparison.capture.SYMBOLS}
    candidate = {symbol: {} for symbol in comparison.capture.SYMBOLS}
    records = []
    for hour in range(12):
        cutoff = t + timedelta(hours=hour)
        bars = [{"close": 90. if side == 1 else 110.} for _ in range(299)] + [{"close": 100.}]
        records.append({"cutoff_utc": cutoff.isoformat(), "evidence": {s: {"bars": bars} for s in comparison.capture.SYMBOLS}})
        for symbol in comparison.capture.SYMBOLS:
            candidate[symbol][cutoff] = {"signal": side if symbol == "BTCUSDT" and hour == 0 else 0, "atr": 2.,
                                         "exit_long": hour == 9, "exit_short": hour == 9}
    specs = {CONTRACTS[s]["ccxt_symbol"]: Spec(CONTRACTS[s]["ccxt_symbol"], .001, .001, 5., .0006, 1.)
             for s in comparison.capture.SYMBOLS}
    rates = {pair: [{"timestamp": int((t + timedelta(hours=h)).timestamp() * 1000), "rate": .001}
                    for h in (1, 9)] for pair in specs}
    monkeypatch.setattr(comparison.report, "prepare_records", lambda _: (frames, {"candidate": candidate, "ema": candidate}, t, t + timedelta(hours=11)))
    monkeypatch.setattr(comparison.report, "recorded_specs", lambda _: specs)
    monkeypatch.setattr(comparison.funding, "verify_schedule", lambda *args: (rates, {pair: 8 for pair in specs}))
    monkeypatch.setattr(comparison, "audit_claims", lambda *args, **kwargs: {"ok": False})
    accounts = comparison.evaluate_segment(records, tmp_path / "synthetic.sqlite", {}, {})["accounts"]
    result = accounts["candidate"]
    assert result["closed_trades"] == 1 and side * result["estimated_funding_paid_usdt"] > 0
    assert result["funding_net_paid"] is None and not result["funding_coverage_checked"]
    assert result == accounts["momentum_24h"] == accounts["drift_168h"]
