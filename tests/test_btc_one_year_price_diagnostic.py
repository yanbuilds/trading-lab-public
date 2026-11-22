"""Synthetic data only; no fixture is evidence of trading performance."""
import json
import sys

import pandas as pd

from experiments import btc_one_year_price_diagnostic as diagnostic


def test_diagnostic_replays_fixed_rules_and_ignores_later_candles(tmp_path, monkeypatch, capsys):
    start = pd.Timestamp('2026-01-15T00:00:00Z')
    end = start + pd.Timedelta(hours=4)
    monkeypatch.setattr(diagnostic, 'START', start)
    monkeypatch.setattr(diagnostic, 'END', end)
    index = pd.date_range(start - pd.Timedelta(hours=300), end + pd.Timedelta(hours=1), freq='h')
    frame = pd.DataFrame({'date': index, 'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1.})
    bars = tmp_path / 'synthetic.csv'
    out = tmp_path / 'synthetic_report.json'

    def run():
        frame.to_csv(bars, index=False)
        monkeypatch.setattr(sys, 'argv', ['diagnostic', '--bars', str(bars), '--out', str(out)])
        diagnostic.main()
        capsys.readouterr()
        return json.loads(out.read_text(encoding='utf-8'))

    first = run()
    frame.loc[frame['date'] >= end, ['open', 'high', 'low', 'close']] = [200., 201., 199., 200.]
    second = run()
    assert first['status'] == 'exploratory_overlapping_history_price_only_not_net_or_held_out'
    assert first['order_permissions'] is False and first['funding'] == 'unknown_excluded'
    assert first['input_sha256'] != second['input_sha256']
    for name in ('frozen_range_candidate', 'ema_baseline'):
        a, b = first['accounts'][name]['summary'], second['accounts'][name]['summary']
        assert a == b
        assert a['initial'] == 100 and a['funding_net_paid'] is None
        assert a['funding_coverage_checked'] is False
