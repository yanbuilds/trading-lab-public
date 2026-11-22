"""Previously inspected overlapping history, fixed rules, no orders or funding claims."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pandas as pd
from experiments import frozen_policy_capture as frozen, btc_rolling_window_replay as candidate
from experiments import btc_rolling_ema_comparison as ema, btc_frozen_policy_history_check as base
from piia.intelligence.paper.specs import CONTRACTS
from trading_lab.perps.compare import simulate_book, summarize
from trading_lab.perps.risk import Spec

START = pd.Timestamp('2025-11-16T06:00:00Z')
END = pd.Timestamp('2026-11-16T06:00:00Z')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bars", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=Path("artifacts/btc_one_year_price_diagnostic.json"))
    args = parser.parse_args()
    source, output = args.bars, args.out
    frozen.verify_policy()
    sources = [Path(__file__), Path(candidate.__file__), Path(ema.__file__), Path(base.__file__)]
    before = {'input_sha256': sha(source), 'source_sha256': {str(p): sha(p) for p in sources}}
    frame = pd.read_csv(source, parse_dates=['date']).set_index('date')
    frame.index = pd.to_datetime(frame.index, utc=True)
    terms = CONTRACTS['BTCUSDT']
    spec = Spec(base.SYMBOL, *(terms[k] for k in ('min_qty', 'qty_step', 'min_notional', 'taker', 'contract_size')))
    models = {}
    prepared, changes = candidate.rolling_prepared(frame, START, END)
    models['frozen_range_candidate'] = (prepared, 24)
    baseline, ema_changes = ema.rolling_ema(frame, START, END)
    models['ema_baseline'] = (baseline, None)
    accounts = {}
    for name, (bars, max_hold) in models.items():
        result = simulate_book({base.SYMBOL: bars}, {base.SYMBOL: spec}, capital=100, leverage=2,
                               start=START, end=END - pd.Timedelta(hours=1), slippage=.0005,
                               prepared=True, max_hold_hours=max_hold, funding=None)
        trades = result.pop('trades'); result.pop('nav', None)
        summary = summarize({**result, 'trades': trades})
        if summary['funding_net_paid'] is not None or summary['funding_coverage_checked'] is not False:
            raise ValueError('funding unexpectedly marked available')
        accounts[name] = {'summary': summary, 'diagnostic_trades': trades}
    if before != {'input_sha256': sha(source), 'source_sha256': {str(p): sha(p) for p in sources}}:
        raise ValueError('input or source changed during calculation')
    frozen.verify_policy()
    report = {'status': 'exploratory_overlapping_history_price_only_not_net_or_held_out',
              'created_at_utc': datetime.now(timezone.utc).isoformat(), 'order_permissions': False,
              'window': {'start_inclusive_utc': START.isoformat(), 'end_exclusive_utc': END.isoformat(),
                         'warmup_hours': 300}, **before, 'symbol': base.SYMBOL, 'capital_usdt': 100,
              'specification_assumed': asdict(spec), 'leverage_example': 2, 'slippage_each_side': .0005,
              'funding': 'unknown_excluded', 'candidate_signal_window_differences': changes,
              'ema_signal_window_differences': ema_changes, 'accounts': accounts,
              'cash_reference': {'initial_usdt': 100, 'final_usdt': 100, 'exposure': 'none'},
              'limitations': ['Window overlaps previously viewed data and is not a held-out test.',
                              'Current public historical candles do not prove historical availability or original source.',
                              'Fees, slippage and specs are assumptions; funding, actual fills and exact liquidation unverified.',
                              'Candidate and EMA have different exposure and exit rules; not risk-matched.',
                              'Only BTC, not a 3-contract portfolio; period-end open positions lack terminal close costs.']}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'window': report['window'], 'funding': report['funding'],
                      'accounts': {k: v['summary'] for k, v in accounts.items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
