"""Check internal arithmetic of a price-only diagnostic, not fill validity."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path


def utc_time(value):
    timestamp = datetime.fromisoformat(value)
    if timestamp.tzinfo is None or timestamp.utcoffset() != timezone.utc.utcoffset(timestamp):
        raise ValueError('timestamps must be UTC-aware')
    return timestamp


def number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError('expected finite numeric value')
    return float(value)


def audit(report):
    if report.get('status') != 'exploratory_overlapping_history_price_only_not_net_or_held_out':
        raise ValueError('unexpected report status')
    if report.get('order_permissions') is not False or report.get('funding') != 'unknown_excluded':
        raise ValueError('unsafe report classification')
    start = utc_time(report['window']['start_inclusive_utc'])
    end = utc_time(report['window']['end_exclusive_utc'])
    if start >= end:
        raise ValueError('invalid window')
    accounts = report['accounts']
    if set(accounts) != {'frozen_range_candidate', 'ema_baseline'}:
        raise ValueError('unexpected accounts')
    result = {}
    for name, account in accounts.items():
        summary, trades = account['summary'], account['diagnostic_trades']
        if summary['funding_net_paid'] is not None or summary['funding_coverage_checked'] is not False:
            raise ValueError('funding incorrectly classified')
        if summary['open_positions'] != 0:
            raise ValueError('cannot reconcile an account with open positions')
        if len(trades) != summary['closed_trades']:
            raise ValueError('trade count mismatch')
        previous_exit, pnls, reasons = None, [], Counter()
        for trade in trades:
            entry, exit_ = utc_time(trade['entry_time']), utc_time(trade['exit_time'])
            if not (start <= entry <= exit_ <= end) or (previous_exit is not None and entry < previous_exit):
                raise ValueError('trade timing or overlap mismatch')
            if trade['symbol'] != report['symbol'] or trade['side'] not in (-1, 1):
                raise ValueError('unexpected instrument or side')
            pnls.append(number(trade['net_pnl']))
            reasons[trade['exit_reason']] += 1
            previous_exit = exit_
        if dict(reasons) != summary['exit_reasons']:
            raise ValueError('exit reason counts mismatch')
        counts = (sum(p > .01 for p in pnls), sum(p < -.01 for p in pnls), sum(abs(p) <= .01 for p in pnls))
        if counts != (summary['wins'], summary['losses'], summary['breakeven']):
            raise ValueError('win/loss counts mismatch')
        initial, final = number(summary['initial']), number(summary['final'])
        if not math.isclose(initial + math.fsum(pnls), final, rel_tol=0, abs_tol=1e-7):
            raise ValueError('account arithmetic mismatch')
        result[name] = {'closed_trades': len(trades), 'sum_reported_trade_pnl': math.fsum(pnls),
                        'last_exit_utc': previous_exit.isoformat() if previous_exit else None}
    return {'status': 'internal_arithmetic_consistent_only', 'accounts': result,
            'limitations': ['Does not independently recalculate fills, individual fees, or funding.',
                            'Does not validate timing availability, historical data, or profitability.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('report', type=Path)
    args = parser.parse_args()
    print(json.dumps(audit(json.loads(args.report.read_text(encoding='utf-8'))), indent=2))


if __name__ == '__main__':
    main()
