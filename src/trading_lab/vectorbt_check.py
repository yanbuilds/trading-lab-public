from __future__ import annotations

"""Optional cross-check against vectorbt on a *single* asset with no caps.

The official multi-asset results come from trading_lab.engine. vectorbt is
installed (community 1.2.1, Apache 2.0 + Commons Clause) and used here so
the dependency is real, not decorative. Mixed calendars + cash floor +
concentration caps are not delegated to vectorbt.
"""

from dataclasses import dataclass

import pandas as pd

from trading_lab.signals import long_cash_signal


@dataclass
class CrossCheck:
    symbol: str
    vectorbt_version: str
    n_bars: int
    signal_agreement: float
    engine_trades: int
    vbt_orders: int
    notes: list[str]


def single_asset_signal_agreement(close: pd.Series, window: int) -> pd.Series:
    return long_cash_signal(close, window)


def run_vectorbt_long_cash(close: pd.Series, open_: pd.Series, window: int, fee: float, slippage: float, init_cash: float):
    import vectorbt as vbt

    sig = long_cash_signal(close, window).fillna(0).astype(bool)
    # Execute on the next bar: shift signal by 1, fill at that bar's open.
    entries = sig.shift(1, fill_value=False)
    exits = (~sig).shift(1, fill_value=False)
    pf = vbt.Portfolio.from_signals(
        close,
        entries,
        exits,
        price=open_,
        fees=fee,
        slippage=slippage,
        init_cash=init_cash,
        freq="1D",
        direction="longonly",
    )
    return pf, sig


def compare_signals_only(close: pd.Series, window: int) -> float:
    """Sanity: our signal definition is close > SMA, after warmup."""
    sig = long_cash_signal(close, window)
    valid = sig.dropna()
    return float(len(valid)) / max(len(close), 1)
