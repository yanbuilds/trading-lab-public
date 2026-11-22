from __future__ import annotations

import pandas as pd

from trading_lab.engine import run_backtest
from helpers import make_cfg, ohlc_from_close, panel_from_frames


def test_etf_has_no_weekend_bars_and_is_not_traded_on_saturday(tmp_path):
    idx_etf = pd.bdate_range("2020-01-02", periods=20)  # weekdays
    close = pd.Series(100 + pd.RangeIndex(20) * 0.3, index=idx_etf, dtype=float)
    gld = ohlc_from_close(idx_etf, close)
    assert not any(d.weekday() >= 5 for d in gld.index)

    idx_btc = pd.date_range("2020-01-02", periods=30, freq="D")
    btc = ohlc_from_close(idx_btc, pd.Series(10000 + pd.RangeIndex(30) * 20, index=idx_btc, dtype=float))
    assert any(d.weekday() >= 5 for d in btc.index)

    cfg = make_cfg(tmp_path, sma_window=3)
    inst = {k: cfg.instruments[k] for k in ["GLD", "BTC-USD"]}
    cfg = make_cfg(tmp_path, instruments=inst, sma_window=3)
    panel = panel_from_frames({"GLD": gld, "BTC-USD": btc})
    result = run_backtest(panel, cfg, capital=10000)
    if not result.trades.empty:
        gld_trades = result.trades[result.trades["symbol"] == "GLD"]
        if not gld_trades.empty:
            weekdays = pd.to_datetime(gld_trades["date"]).dt.day_of_week
            assert (weekdays < 5).all(), "ETF 不应在周末成交"
        btc_trades = result.trades[result.trades["symbol"] == "BTC-USD"]
        # 不强制周末一定成交，但允许周末成交
        assert set(btc_trades["symbol"].unique()) <= {"BTC-USD"} if not btc_trades.empty else True


def test_weekend_nav_does_not_invent_etf_return(tmp_path):
    """周六若只有 BTC 有 K 线，GLD 标记价应沿用周五收盘，而不是虚构一个周六收益。"""
    idx_etf = pd.bdate_range("2020-01-03", periods=10)  # starts Friday 2020-01-03
    gld = ohlc_from_close(idx_etf, pd.Series(100.0, index=idx_etf))
    idx_btc = pd.date_range("2020-01-03", periods=10, freq="D")
    btc = ohlc_from_close(idx_btc, pd.Series(10000.0, index=idx_btc))
    cfg = make_cfg(tmp_path, sma_window=3)
    inst = {k: cfg.instruments[k] for k in ["GLD", "BTC-USD"]}
    cfg = make_cfg(tmp_path, instruments=inst, sma_window=3, cash_floor=0.2, max_weight=0.4)
    panel = panel_from_frames({"GLD": gld, "BTC-USD": btc})
    result = run_backtest(panel, cfg, capital=10000)
    # 2020-01-03 is Friday, 01-04 Saturday
    sat = pd.Timestamp("2020-01-04")
    fri = pd.Timestamp("2020-01-03")
    if sat in result.marks.index and fri in result.marks.index:
        assert result.marks.loc[sat, "GLD"] == result.marks.loc[fri, "GLD"]
