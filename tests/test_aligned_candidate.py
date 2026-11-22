import numpy as np
import pandas as pd

from trading_lab.perps.aligned_candidate import attach_aligned_ema
from trading_lab.perps.aligned_compare import goal_assessment


def test_aligned_signal_uses_only_completed_bars():
    idx = pd.date_range("2025-01-01", periods=800, freq="h", tz="UTC")
    close = 100 + np.sin(np.arange(len(idx)) / 9) + np.arange(len(idx)) * 0.01
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1.0}, index=idx)
    cut = idx[601]
    original = attach_aligned_ema(bars)
    changed = bars.copy()
    changed.loc[changed.index > cut, ["open", "high", "low", "close"]] *= 3
    revised = attach_aligned_ema(changed)
    cols = ["signal", "trend_4h_fast", "trend_4h_slow"]
    pd.testing.assert_frame_equal(original.loc[:cut, cols], revised.loc[:cut, cols])
    assert original["trend_4h_slow"].notna().any()


def test_tenfold_backtest_alone_does_not_prove_goal():
    assessment = goal_assessment({"initial": 100, "final": 1000}, funding_excluded=True)
    assert assessment["equity_multiple"] == 10
    assert assessment["tenfold_in_simulation"] is True
    assert assessment["goal_verified"] is False
    assert assessment["funding_included"] is False
