"""Synthetic wiring tests; no fixture represents market performance."""
import pytest

from experiments import three_contract_rolling_diagnostic as d


def setup_synthetic(monkeypatch):
    monkeypatch.setattr(d.frozen_policy_capture, 'verify_policy', lambda: None)
    monkeypatch.setattr(d.prospective_paper_report, 'verify_baseline', lambda: None)
    monkeypatch.setattr(d, 'fingerprints', lambda _: {'synthetic': True})
    frames = {s: s for s in d.SYMBOLS}
    monkeypatch.setattr(d, 'load_cache', lambda _: (frames, frames, frames,
        {'symbols': {s: {'funding_interval_hours': 8} for s in d.SYMBOLS}}))
    monkeypatch.setattr(d, 'funding_complete_between', lambda *_: {'complete': True})
    monkeypatch.setattr(d.candidate, 'rolling_prepared', lambda f, *_: (f + ':candidate', {'decision_hours': 1}))
    monkeypatch.setattr(d.ema, 'rolling_ema', lambda f, *_: (f + ':ema', 0))
    calls = []
    def simulate(bars, specs, funding, **kwargs):
        calls.append((bars, kwargs))
        return {'trades': [], 'nav': []}
    monkeypatch.setattr(d, 'simulate_book', simulate)
    monkeypatch.setattr(d, 'summarize', lambda _: {'initial': 100})
    return calls


def test_both_books_use_three_symbols_and_same_cost_and_risk_settings(monkeypatch):
    calls = setup_synthetic(monkeypatch)
    result = d.run('synthetic')
    assert result['status'] == 'exploratory_seen_three_contract_rolling_not_held_out_or_live'
    assert result['order_permissions'] is False
    assert len(calls) == 2 and all(set(bars) == d.SYMBOLS for bars, _ in calls)
    a, b = calls[0][1], calls[1][1]
    assert a['max_hold_hours'] == 24 and b['max_hold_hours'] is None
    assert {k: v for k, v in a.items() if k != 'max_hold_hours'} == {k: v for k, v in b.items() if k != 'max_hold_hours'}


def test_missing_funding_stops_both_simulations(monkeypatch):
    calls = setup_synthetic(monkeypatch)
    monkeypatch.setattr(d, 'funding_complete_between', lambda *_: {'complete': False})
    with pytest.raises(ValueError, match='funding coverage incomplete'):
        d.run('synthetic')
    assert calls == []
