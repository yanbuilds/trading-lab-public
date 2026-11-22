"""Synthetic arithmetic fixtures; these are not market results."""
import copy

import pytest

from experiments.audit_btc_diagnostic_ledger import audit


def sample():
    trade = {'symbol': 'BTC/USDT:USDT', 'side': 1, 'entry_time': '2025-01-01T01:00:00+00:00',
             'exit_time': '2025-01-01T02:00:00+00:00', 'net_pnl': -1., 'exit_reason': 'atr_stop'}
    summary = {'initial': 100., 'final': 99., 'closed_trades': 1, 'wins': 0, 'losses': 1,
               'breakeven': 0, 'exit_reasons': {'atr_stop': 1}, 'open_positions': 0,
               'funding_net_paid': None, 'funding_coverage_checked': False}
    account = {'summary': summary, 'diagnostic_trades': [trade]}
    return {'status': 'exploratory_overlapping_history_price_only_not_net_or_held_out',
            'order_permissions': False, 'funding': 'unknown_excluded', 'symbol': 'BTC/USDT:USDT',
            'window': {'start_inclusive_utc': '2025-01-01T00:00:00+00:00',
                       'end_exclusive_utc': '2025-01-02T00:00:00+00:00'},
            'accounts': {'frozen_range_candidate': account, 'ema_baseline': copy.deepcopy(account)}}


def test_arithmetic_and_limitations():
    result = audit(sample())
    assert result['status'] == 'internal_arithmetic_consistent_only'
    assert result['accounts']['frozen_range_candidate']['sum_reported_trade_pnl'] == -1
    assert 'Does not independently recalculate' in result['limitations'][0]


@pytest.mark.parametrize('change', ['pnl', 'counts', 'funding', 'open', 'time', 'nonfinite'])
def test_rejects_inconsistent_or_unverifiable_ledger(change):
    report = sample()
    account = report['accounts']['ema_baseline']
    if change == 'pnl': account['diagnostic_trades'][0]['net_pnl'] = -2.
    if change == 'counts': account['summary']['wins'] = 1
    if change == 'funding': account['summary']['funding_coverage_checked'] = True
    if change == 'open': account['summary']['open_positions'] = 1
    if change == 'time': account['diagnostic_trades'][0]['exit_time'] = '2025-01-03T00:00:00+00:00'
    if change == 'nonfinite': account['diagnostic_trades'][0]['net_pnl'] = float('nan')
    with pytest.raises(ValueError):
        audit(report)
