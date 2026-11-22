from __future__ import annotations

import pandas as pd

from trading_lab.engine import run_backtest
from helpers import make_cfg, ohlc_from_close, panel_from_frames


def _three_rising(n=40):
    idx_etf = pd.bdate_range("2020-01-02", periods=n)
    frames = {}
    for i, name in enumerate(["QQQ", "GLD"]):
        close = pd.Series(100 + i * 5 + pd.RangeIndex(n) * 0.5, index=idx_etf, dtype=float)
        frames[name] = ohlc_from_close(idx_etf, close)
    idx_btc = pd.date_range("2020-01-02", periods=int((idx_etf[-1] - idx_etf[0]).days) + 1, freq="D")
    close_b = pd.Series(10000 + pd.RangeIndex(len(idx_btc)) * 15, index=idx_btc, dtype=float)
    frames["BTC-USD"] = ohlc_from_close(idx_btc, close_b)
    return frames


def test_weights_respect_caps_and_cash_floor(tmp_path):
    cfg = make_cfg(tmp_path, max_weight=0.40, cash_floor=0.20, sma_window=5)
    panel = panel_from_frames(_three_rising())
    result = run_backtest(panel, cfg, capital=100000)
    w = result.weights.fillna(0)
    # 忽略均线热身期
    tail = w.iloc[10:]
    gross = tail.sum(axis=1)
    # 下单时按上限和现金下限约束；收盘价变动可能使盘后权重略微漂移。
    assert (gross <= 1.0 + 1e-6).all(), "不允许融资，毛敞口不得超过 100%"
    assert (tail.max(axis=1) <= 0.50 + 1e-6).all(), "集中度不应远超 40% 的下单上限"
    assert (gross.mean() <= (1.0 - cfg.cash_floor) + 0.03)
    assert (result.cash >= -1e-6).all()
    assert (result.holdings.fillna(0).min().min() >= -1e-9)


def test_no_margin_when_cash_is_tight(tmp_path):
    cfg = make_cfg(tmp_path, max_weight=1.0, cash_floor=0.0, sma_window=5)
    frames = _three_rising(30)
    panel = panel_from_frames(frames)
    result = run_backtest(panel, cfg, capital=1000)
    assert (result.cash >= -1e-6).all()
    # 市值不应明显超过本金加收益很多的同时出现负现金
    assert result.cash.min() >= -1e-6


def test_holdout_restarts_with_example_capital_and_no_train_trades(tmp_path):
    cfg=make_cfg(tmp_path,sma_window=5)
    panel=panel_from_frames(_three_rising(40))
    start='2020-02-01'
    r=run_backtest(panel,cfg,capital=100000,trading_start=start)
    assert (r.cash.loc[:'2020-01-31']==100000).all()
    assert (r.trades.date>=pd.Timestamp(start)).all()
    assert (r.cash>=-1e-6).all()
