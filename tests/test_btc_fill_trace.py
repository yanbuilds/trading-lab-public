"""Synthetic fills only: no fixture is a trading result."""
import pandas as pd
import pytest

from experiments.btc_fill_trace import capture_simulated_closes, check_fills
from trading_lab.perps.compare import simulate_book
from trading_lab.perps.risk import Spec


@pytest.mark.parametrize('side', [1, -1])
def test_trace_preserves_results_and_recomputes_fills(side):
    idx = pd.date_range('2026-01-01', periods=65, freq='h', tz='UTC')
    bars = pd.DataFrame({'open': 100., 'high': 101., 'low': 99., 'close': 100., 'volume': 1.,
                         'atr': 1., 'signal': 0, 'exit_long': False, 'exit_short': False}, index=idx)
    bars.loc[idx[0], 'signal'] = side
    bars.loc[idx[2], ['exit_long', 'exit_short']] = True
    symbol = 'BTC/USDT:USDT'
    args = ({symbol: bars}, {symbol: Spec(symbol, .0001, .0001, 5., .0006, 1.)})
    kwargs = {'capital': 100, 'leverage': 2, 'prepared': True, 'start': idx[1], 'end': idx[-1]}
    plain = simulate_book(*args, **kwargs)
    with capture_simulated_closes() as fills:
        traced = simulate_book(*args, **kwargs)
    assert plain == traced
    assert len(fills) == len(traced['trades']) == 1
    check_fills(fills, traced['trades'])
    fills[0]['exit_price_with_slippage'] += 1
    # The verifier compares the stored PnL with the engine's value; changing
    # individual prices must also invalidate its independent calculation.
    with pytest.raises(ValueError):
        check_fills(fills, traced['trades'])
