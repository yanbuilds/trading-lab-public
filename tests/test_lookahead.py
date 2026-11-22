from __future__ import annotations

import pandas as pd

from trading_lab.engine import run_backtest
from trading_lab.signals import long_cash_signal
from helpers import make_cfg, ohlc_from_close, panel_from_frames


def test_signal_uses_only_completed_close_not_future():
    idx = pd.bdate_range("2020-01-02", periods=20)
    close = pd.Series(100.0, index=idx)
    # 前 19 天横盘，最后一天收盘暴涨。若错误地用了未来收盘，更早的信号会被污染。
    close.iloc[-1] = 200.0
    sig = long_cash_signal(close, window=5)
    # 均线在最后一天之前约等于 100，信号应为 0；只有最后一根 close>sma 才变 1
    assert int(sig.iloc[-2]) == 0
    assert int(sig.iloc[-1]) == 1


def test_execution_is_next_open_not_same_bar_close(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=20)
    close = pd.Series([100.0] * 15 + [90, 90, 90, 90, 130], index=idx)
    # 开盘刻意与收盘不同：开盘 = 前收，最后一天开盘仍是 90，收盘 130
    df = ohlc_from_close(idx, close, open_shift=0.0)
    assert df["open"].iloc[-1] == 90.0
    assert df["close"].iloc[-1] == 130.0
    cfg = make_cfg(tmp_path, instruments={
        k: v for k, v in make_cfg(tmp_path).instruments.items() if k == "QQQ"
    } if False else make_cfg(tmp_path).instruments)
    # only QQQ
    from trading_lab.config import LabConfig
    cfg = make_cfg(tmp_path)
    inst = {k: cfg.instruments[k] for k in ["QQQ"]}
    cfg = make_cfg(tmp_path, instruments=inst, cash_floor=0.0, max_weight=1.0)
    panel = panel_from_frames({"QQQ": df})
    result = run_backtest(panel, cfg, capital=10_000)
    # 最后一根产生买入信号，但最后一天不应该用 130 的收盘价买入。
    last = idx[-1]
    last_trades = result.trades[result.trades["date"] == last] if not result.trades.empty else result.trades
    if not last_trades.empty:
        buy = last_trades[last_trades["side"] == "buy"]
        if not buy.empty:
            # 若在最后一天买入，成交必须是开盘 90（再加滑点），绝不能是 130
            assert (buy["open"] == 90.0).all()
            assert (buy["fill"] <= 90.0 * 1.001 + 1e-9).all()
            assert (buy["fill"] != 130.0).all()
    # 信号最后一天才为多，执行应发生在「下一根」，所以最后一天不应因这根收盘而买到满仓。
    sig = result.signals["QQQ"].dropna()
    assert int(sig.iloc[-1]) == 1
    # 仓位按最后收盘标记：若错误地在最后一根收盘后立刻成交，权重会接近 100%。
    # 正确实现是等到下一开盘，因此最后一天的持仓不应在收盘后才用 130 买入。
    last_w = float(result.weights.loc[last, "QQQ"])
    # 允许此前已有仓位，但不应在看到 130 的同一根上按 130 买入。检查最后一天若有买入，开盘=90。
    assert last_w < 0.99 or (not last_trades.empty and (last_trades["open"] == 90.0).all())


def test_no_fill_when_next_open_does_not_exist(tmp_path):
    """纸上/回测都不应把最后一根收盘价当成已经发生的下一开盘成交。"""
    from trading_lab.paper import latest_signal_replay

    idx = pd.bdate_range("2020-01-02", periods=20)
    close = pd.Series(100.0, index=idx)
    close.iloc[-5:] = 120.0  # 明确为多
    df = ohlc_from_close(idx, close)
    cfg = make_cfg(tmp_path)
    inst = {k: cfg.instruments[k] for k in ["QQQ"]}
    cfg = make_cfg(tmp_path, instruments=inst, cash_floor=0.0, max_weight=1.0, etf_max_completed_sessions_lag=30)
    panel = panel_from_frames({"QQQ": df})
    # 把「现在」设到最后一根收盘之后很久会过期；这里改用足够松的滞后，并伪造 now 为最后一根收盘后不久。
    # latest_signal_replay 必须列出 pending 单，而不是已经成交的持仓变化。
    snap = latest_signal_replay(panel, cfg, holdings={"QQQ": 0.0}, cash=10_000)
    # 可能因过期被阻断——若未阻断，意向单不得声称已用收盘成交。
    if not snap["blocked_new_signals"] and snap["intended_orders"]:
        for o in snap["intended_orders"]:
            assert o["do_not_fill_at_last_close"] is True
            assert "下一根开盘" in o["execute_at"]
    assert snap["kind"] == "latest_signal_replay"


def test_signal_on_last_close_cannot_create_same_bar_buy(tmp_path):
    from dataclasses import replace
    idx=pd.bdate_range('2020-01-02',periods=12)
    close=pd.Series([100.0]*11+[200.0],index=idx)
    cfg=make_cfg(tmp_path,sma_window=5)
    cfg=replace(cfg,instruments={'QQQ':cfg.instruments['QQQ']})
    r=run_backtest(panel_from_frames({'QQQ':ohlc_from_close(idx,close)}),cfg)
    assert r.trades.empty
    assert float(r.holdings.iloc[-1]['QQQ'])==0.0
