"""Synthetic fixtures for the exploratory per-market comparison."""
import numpy as np
import pandas as pd
import pytest

from experiments.cross_market_breakout import SYMBOLS, evaluate
from trading_lab.perps.risk import Spec


def test_each_market_is_separate_and_funding_stays_unknown():
    idx = pd.date_range("2026-01-01", periods=800, freq="h", tz="UTC")
    price = 100 + np.arange(len(idx)) * .01
    bars = pd.DataFrame({"open": price, "high": price + 1, "low": price - 1,
                         "close": price + .1, "volume": 1.}, index=idx)
    frames = {symbol: bars.copy() for symbol in SYMBOLS}
    specs = {symbol: Spec(symbol, .001, .001, 5) for symbol in SYMBOLS}
    report = evaluate(frames, specs, idx[700], idx[-1] + pd.Timedelta(hours=1))
    assert set(report["markets"]) == set(SYMBOLS)
    for market in report["markets"].values():
        for strategy in market["strategies"].values():
            assert strategy["initial"] == 100 and strategy["funding_net_paid"] is None
    assert report["shared_book"]["capital_usdt"] == 100
    assert report["shared_book"]["max_positions"] == 1
    for strategy in report["shared_book"]["strategies"].values():
        assert strategy["initial"] == 100 and strategy["funding_net_paid"] is None
    specs[SYMBOLS[1]] = specs[SYMBOLS[0]]
    with pytest.raises(ValueError, match="incorrect contract spec"):
        evaluate(frames, specs, idx[700], idx[-1] + pd.Timedelta(hours=1))


def test_missing_market_does_not_fall_back_to_another():
    with pytest.raises(ValueError, match="all three perpetual"):
        evaluate({}, {}, pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-02", tz="UTC"))
