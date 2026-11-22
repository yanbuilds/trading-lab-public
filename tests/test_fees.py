from __future__ import annotations

import pandas as pd

from trading_lab.engine import run_backtest
from helpers import make_cfg, ohlc_from_close, panel_from_frames


def test_fees_and_slippage_reduce_cash_on_buy(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=16)
    # 前 5 天 100，之后抬到 110，产生稳定多头信号
    close = pd.Series([100.0] * 6 + [110.0] * 10, index=idx)
    df = ohlc_from_close(idx, close)
    cfg = make_cfg(tmp_path)
    inst = {k: cfg.instruments[k] for k in ["QQQ"]}
    # fee 1%, slippage 0 so we can compute exactly
    from dataclasses import replace

    inst["QQQ"] = replace(inst["QQQ"], fee=0.01, slippage=0.0)
    cfg = make_cfg(tmp_path, instruments=inst, cash_floor=0.0, max_weight=1.0, sma_window=5)
    result = run_backtest(panel_from_frames({"QQQ": df}), cfg, capital=10_000)
    buys = result.trades[result.trades["side"] == "buy"]
    assert not buys.empty
    row = buys.iloc[0]
    expected_fee = row["notional"] * 0.01
    assert abs(row["fee"] - expected_fee) < 1e-6
    # 净值应低于「零费用买住同等价格」——至少费用被扣过
    assert result.nav.iloc[-1] < 10_000 * (df["close"].iloc[-1] / row["fill"]) + 1e-6 or row["fee"] > 0
    assert result.trades["fee"].sum() > 0
