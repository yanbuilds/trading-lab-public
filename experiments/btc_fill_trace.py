"""Observe frozen simulated closes without editing the engine; no exchange fills."""
from contextlib import contextmanager
import argparse
import json
import math
from pathlib import Path
import sys

from trading_lab.perps import compare


@contextmanager
def capture_simulated_closes():
    """Capture local values from the pinned engine's nested close function.

    This tracing depends on Python implementation details and must fail closed if
    the engine or trace shape changes. It does not establish market execution.
    """
    if sys.getprofile() is not None:
        raise RuntimeError('a Python profiler is already active')
    fills, active = [], {}

    def profiler(frame, event, arg):
        if frame.f_code.co_name != 'close' or frame.f_code.co_filename != compare.__file__:
            return
        if frame.f_code.co_qualname != 'simulate_book.<locals>.close':
            return
        if event == 'call':
            loc = frame.f_locals
            active[id(frame)] = dict(loc['pos'])
        elif event == 'return':
            pos = active.pop(id(frame))
            loc = frame.f_locals
            spec = loc['spec']
            entry, exit_ = float(pos['entry']), float(loc['px'])
            qty, side = float(loc['qty']), int(loc['side'])
            entry_fee, exit_fee = float(pos['entry_fee']), float(loc['exit_fee'])
            funding = float(pos['funding'])
            computed_gross = side * qty * float(spec.contract_size) * (exit_ - entry)
            computed_entry_fee = qty * float(spec.contract_size) * entry * float(spec.taker)
            computed_exit_fee = qty * float(spec.contract_size) * exit_ * float(spec.taker)
            for actual, calculated in [(loc['gross'], computed_gross), (entry_fee, computed_entry_fee),
                                       (exit_fee, computed_exit_fee)]:
                if not math.isclose(float(actual), calculated, rel_tol=1e-12, abs_tol=1e-10):
                    raise ValueError('independent fill arithmetic mismatch')
            fills.append({'symbol': pos['symbol'], 'side': side, 'entry_time': str(pos['time']),
                          'exit_time': str(loc['ts']), 'exit_reason': loc['reason'],
                          'quantity': qty, 'entry_price_with_slippage': entry,
                          'exit_price_with_slippage': exit_, 'contract_size': float(spec.contract_size),
                          'taker_fee_rate_assumed': float(spec.taker), 'entry_fee': entry_fee,
                          'exit_fee': exit_fee, 'gross_pnl': computed_gross,
                          'funding_model_value_not_verified': funding,
                          'pnl_excluding_unknown_funding': computed_gross - entry_fee - exit_fee})

    sys.setprofile(profiler)
    try:
        yield fills
    finally:
        sys.setprofile(None)


def check_fills(fills, trades):
    if len(fills) != len(trades):
        raise ValueError('incomplete fill trace')
    for fill, trade in zip(fills, trades, strict=True):
        for key in ('symbol', 'side', 'entry_time', 'exit_time', 'exit_reason'):
            if fill[key] != trade[key]:
                raise ValueError(f'fill identity mismatch: {key}')
        qty, size, rate = fill['quantity'], fill['contract_size'], fill['taker_fee_rate_assumed']
        entry, exit_ = fill['entry_price_with_slippage'], fill['exit_price_with_slippage']
        gross = fill['side'] * qty * size * (exit_ - entry)
        entry_fee, exit_fee = qty * size * entry * rate, qty * size * exit_ * rate
        for key, value in [('gross_pnl', gross), ('entry_fee', entry_fee), ('exit_fee', exit_fee),
                           ('pnl_excluding_unknown_funding', gross - entry_fee - exit_fee)]:
            if not math.isclose(fill[key], value, rel_tol=1e-12, abs_tol=1e-10):
                raise ValueError(f'fill arithmetic mismatch: {key}')
        net = gross - entry_fee - exit_fee - fill['funding_model_value_not_verified']
        if not math.isclose(net, float(trade['net_pnl']), rel_tol=1e-12, abs_tol=1e-10):
            raise ValueError('reported trade PnL mismatch')


def main():
    from experiments import btc_one_year_price_diagnostic as diagnostic

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bars', type=Path, required=True)
    parser.add_argument('--reference', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if len({args.bars.resolve(), args.reference.resolve(), args.out.resolve()}) != 3:
        raise ValueError('output paths must be distinct')
    reference = json.loads(args.reference.read_text(encoding='utf-8'))
    if diagnostic.sha(args.bars) != reference['input_sha256']:
        raise ValueError('bar file differs from reference')
    reference_accounts = list(reference['accounts'].values())
    captured = []
    original = diagnostic.simulate_book

    def traced(*positional, **keywords):
        with capture_simulated_closes() as fills:
            result = original(*positional, **keywords)
        check_fills(fills, result['trades'])
        expected = reference_accounts[len(captured)]
        summary = diagnostic.summarize(result)
        if summary != expected['summary'] or result['trades'] != expected['diagnostic_trades']:
            raise ValueError('traced replay differs from reference')
        captured.append(fills)
        return result

    # The existing entry point writes its normal report to a separate local file.
    temp_report = args.out.with_name(args.out.stem + '_replay.json')
    if temp_report.resolve() in {args.bars.resolve(), args.reference.resolve(), args.out.resolve()}:
        raise ValueError('output paths must be distinct')
    original_argv, diagnostic.simulate_book = sys.argv, traced
    try:
        sys.argv = ['diagnostic', '--bars', str(args.bars), '--out', str(temp_report)]
        diagnostic.main()
    finally:
        diagnostic.simulate_book, sys.argv = original, original_argv
    if len(captured) != len(reference_accounts):
        raise ValueError('incomplete account trace')
    report = {'status': 'simulated_fill_arithmetic_only_not_net_or_real_execution',
              'input_sha256': reference['input_sha256'],
              'reference_sha256': diagnostic.sha(args.reference),
              'accounts': dict(zip(reference['accounts'], captured, strict=True)),
              'limitations': ['Engine-observed simulated fills are not exchange execution records.',
                              'Funding is unknown; actual fees, specs, slippage and liquidation unverified.',
                              'Historical window was inspected and is not a held-out test.']}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'fills': {k: len(v) for k, v in report['accounts'].items()}}))


if __name__ == '__main__':
    main()
