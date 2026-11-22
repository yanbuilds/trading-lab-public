"""Synthetic fixture tests; no actual performance claims."""
import pandas as pd
import pytest

from experiments.policy_latency_stress import lag_closed_bar_decisions, run
from trading_lab.perps.compare import simulate_book
from trading_lab.perps.risk import Spec


def test_delay_uses_old_signal_and_its_old_atr_not_current_bar():
    frame = pd.DataFrame({"signal": [1, 0, -1, 0], "atr": [2., 8., 3., 9.],
                          "exit_long": [False, True, False, False],
                          "exit_short": [False, False, True, False]})
    result = lag_closed_bar_decisions(frame)
    assert result["signal"].tolist() == [0, 1, 0, -1]
    assert result.loc[1, "atr"] == 2. and result.loc[3, "atr"] == 3.
    assert result["exit_long"].tolist() == [False, False, True, False]
    assert result["exit_short"].tolist() == [False, False, False, True]
    assert frame.loc[0, "signal"] == 1  # source unchanged


def test_missing_markets_and_naive_times_are_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        run({}, {}, pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02"))
    with pytest.raises(ValueError, match="three contract"):
        run({}, {}, pd.Timestamp("2026-01-01", tz="UTC"), pd.Timestamp("2026-01-02", tz="UTC"))


def test_lag_changes_entry_and_signal_exit_by_one_hour():
    idx = pd.date_range("2026-01-01", periods=80, freq="h", tz="UTC")
    frame = pd.DataFrame({"open": 100., "high": 100.1, "low": 99.9, "close": 100.,
                          "volume": 1., "signal": 0, "atr": 2.,
                          "exit_long": False, "exit_short": False,
                          "trail_long": float("nan"), "trail_short": float("nan")}, index=idx)
    frame.loc[idx[65], "signal"] = 1
    frame.loc[idx[68], "exit_long"] = True
    symbol = "BTC/USDT:USDT"
    specs = {symbol: Spec(symbol, .001, .001, 5)}
    def replay(data):
        return simulate_book({symbol: data}, specs, capital=100, start=idx[64], end=idx[72],
                             prepared=True, max_hold_hours=24)["trades"][0]
    immediate = replay(frame)
    delayed = replay(lag_closed_bar_decisions(frame))
    assert pd.Timestamp(immediate["entry_time"]) == idx[66]
    assert pd.Timestamp(immediate["exit_time"]) == idx[69]
    assert pd.Timestamp(delayed["entry_time"]) == idx[67]
    assert pd.Timestamp(delayed["exit_time"]) == idx[70]
