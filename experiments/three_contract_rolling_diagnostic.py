"""Already-seen 300-bar three-contract replay; not a holdout or real trading."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from experiments import (btc_rolling_ema_comparison as ema, btc_rolling_window_replay as candidate,
                         frozen_policy_capture, portfolio_cost_stress, prospective_paper_report)
from trading_lab.perps.compare import funding_complete_between, load_cache, simulate_book, summarize

START = pd.Timestamp('2026-08-19T00:00:00Z')
END = pd.Timestamp('2026-11-16T06:00:00Z')
SYMBOLS = {'BTC/USDT:USDT', 'XAU/USDT:USDT', 'NDX100/USDT:USDT'}


def fingerprints(data_dir):
    files = [Path(__file__), Path(candidate.__file__), Path(ema.__file__), Path(prospective_paper_report.__file__)]
    return {'inputs_and_engine': portfolio_cost_stress.portfolio_manifest(data_dir),
            'extra_sources': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}}


def run(data_dir):
    frozen_policy_capture.verify_policy()
    prospective_paper_report.verify_baseline()
    before = fingerprints(data_dir)
    frames, specs, funding, provenance = load_cache(data_dir)
    if set(frames) != SYMBOLS or set(funding) != SYMBOLS or set(specs) != SYMBOLS:
        raise ValueError('all three contracts and funding are required')
    intervals = {s: provenance['symbols'][s]['funding_interval_hours'] for s in SYMBOLS}
    coverage = funding_complete_between(funding, intervals, START, END - pd.Timedelta(hours=1))
    if not coverage['complete']:
        raise ValueError('funding coverage incomplete under supplied current-interval assumption')
    books = {'candidate': {}, 'ema': {}}
    differences = {}
    for symbol in sorted(SYMBOLS):
        books['candidate'][symbol], candidate_diff = candidate.rolling_prepared(frames[symbol], START, END)
        books['ema'][symbol], ema_diff = ema.rolling_ema(frames[symbol], START, END)
        differences[symbol] = {'candidate': candidate_diff, 'ema_signal_differences': ema_diff}
    accounts = {}
    for name, bars in books.items():
        raw = simulate_book(bars, specs, funding, capital=100, leverage=2, start=START,
                            end=END - pd.Timedelta(hours=1), slippage=.0005, prepared=True,
                            funding_intervals=intervals, max_hold_hours=24 if name == 'candidate' else None)
        trades = raw.pop('trades')
        raw.pop('nav', None)
        accounts[name] = {'summary': summarize({**raw, 'trades': trades}), 'diagnostic_trades': trades}
    frozen_policy_capture.verify_policy()
    prospective_paper_report.verify_baseline()
    if fingerprints(data_dir) != before:
        raise ValueError('inputs or source changed during replay')
    return {'status': 'exploratory_seen_three_contract_rolling_not_held_out_or_live',
            'order_permissions': False, 'start_utc': START.isoformat(), 'end_exclusive_utc': END.isoformat(),
            'capital_usdt': 100, 'leverage_example': 2, 'window_bars': 300, 'input_manifest': before,
            'funding_coverage_under_current_interval_assumption': coverage, 'signal_differences_vs_full': differences,
            'accounts': accounts, 'cash_reference': {'initial': 100, 'final': 100, 'market_exposure': 'none'},
            'limitations': ['Previously inspected history; not a holdout or pre-registered test.',
                            'Funding schedule, historical specifications, mark settlement and liquidation unverified.',
                            'Next-open fills and slippage are assumptions; not actual execution.',
                            'EMA and candidate exposures and exits differ; not risk matched.',
                            'Terminal open positions lack closing costs; no AI or macro attribution.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    result = run(args.data_dir)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False, default=str), encoding='utf-8')
    print(json.dumps({'status': result['status'], 'accounts': {k: v['summary'] for k, v in result['accounts'].items()}},
                     ensure_ascii=False))


if __name__ == '__main__':
    main()
