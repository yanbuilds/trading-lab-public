"""Synthetic alignment checks, not market or performance evidence."""
import numpy as np
import pandas as pd
import pytest

from experiments.btc_rolling_ema_comparison import rolling_ema
from experiments.btc_rolling_window_replay import FIELDS
from trading_lab.perps.compare import attach_ema


def test_ema_uses_prior_closed_300_bar_window():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    end = start + pd.Timedelta(hours=3)
    idx = pd.date_range(start - pd.Timedelta(hours=350), end - pd.Timedelta(hours=1), freq="h")
    close = 100 + np.sin(np.arange(len(idx)) / 13)
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "volume": 1.}, index=idx)
    actual, _ = rolling_ema(frame, start, end)
    stamp = start - pd.Timedelta(hours=1)
    expected = attach_ema(frame.loc[stamp - pd.Timedelta(hours=299):stamp]).iloc[-1]
    pd.testing.assert_series_equal(actual.loc[stamp, list(FIELDS)], expected.loc[list(FIELDS)], check_names=False)
    changed = frame.copy()
    changed.loc[start:, ["open", "high", "low", "close"]] *= 2
    revised, _ = rolling_ema(changed, start, end)
    pd.testing.assert_series_equal(actual.loc[stamp, list(FIELDS)], revised.loc[stamp, list(FIELDS)], check_names=False)
    with pytest.raises(ValueError, match="Missing"):
        rolling_ema(frame.drop(idx[100]), start, end)
