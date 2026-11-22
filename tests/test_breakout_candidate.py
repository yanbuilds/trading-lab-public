"""Synthetic data only; no prediction of trading performance."""
import numpy as np
import pandas as pd

from experiments.breakout_candidate import attach_breakout_with_4h_trend, long_hold_reference
from trading_lab.perps.compare import attach_turtle_s1
from trading_lab.perps.risk import Spec


def test_breakout_filter_is_causal_and_does_not_change_exit_rules():
    idx = pd.date_range("2025-09-01", periods=100 * 24, freq="h", tz="UTC")
    close = 100 + np.arange(len(idx)) * .02 + np.sin(np.arange(len(idx)) / 30)
    bars = pd.DataFrame({"open": close, "high": close + .5, "low": close - .5,
                         "close": close, "volume": 1.}, index=idx)
    original = attach_breakout_with_4h_trend(bars)
    base = attach_turtle_s1(bars)
    assert (original.loc[original.signal != 0, "signal"] == base.loc[original.signal != 0, "signal"]).all()
    assert (original.signal != 0).any()
    for col in ("exit_long", "exit_short", "atr", "trail_long", "trail_short"):
        pd.testing.assert_series_equal(original[col], base[col])
    for index in (1300, 1301, 1302, 1303):
        cutoff = idx[index]
        prefix = attach_breakout_with_4h_trend(bars.loc[:cutoff])
        for col in ("signal", "trend_4h_fast", "trend_4h_slow", "atr", "exit_long", "exit_short"):
            pd.testing.assert_series_equal(prefix[col], original.loc[:cutoff, col])


def test_hold_reference_pays_both_sides_and_never_upsizes():
    idx = pd.date_range("2026-01-01", periods=2, freq="h", tz="UTC")
    bars = pd.DataFrame({"open": [100., 100.], "close": [100., 100.]}, index=idx)
    result = long_hold_reference(bars, Spec("BTC", .001, .001, 5, .0006, 1), idx[0], idx[-1] + pd.Timedelta(hours=1))
    assert result["available"] and -0.3 < result["return_pct"] < 0
    assert result["funding"] == "excluded_unknown"
    assert not long_hold_reference(bars, Spec("BTC", 2, 1, 5), idx[0], idx[-1] + pd.Timedelta(hours=1))["available"]
