from __future__ import annotations

import pandas as pd

from trading_lab.config import LabConfig
from trading_lab.data import PricePanel
from trading_lab.engine import BacktestResult


def scenario_report(result: BacktestResult, panel: PricePanel, cfg: LabConfig) -> dict:
    """压力情景。日线无法衡量流动性；杠杆在本系统设计中为 0。"""
    keys = [k for k in cfg.instruments if k in panel.frames]
    last = result.nav.index[-1]
    weights = result.weights.loc[last].to_dict() if last in result.weights.index else {}
    holdings = result.holdings.loc[last].to_dict() if last in result.holdings.index else {}
    marks = result.marks.loc[last].to_dict() if last in result.marks.index else {}
    nav = float(result.nav.iloc[-1])

    # Simultaneous historical down days: all available assets have a bar and negative return that day
    rets = {}
    for k in keys:
        rets[k] = panel.frames[k]["close"].pct_change()
    # intersection of ETF dates (common sessions) plus note crypto
    etf = cfg.etf_keys
    if len(etf) >= 2 and all(k in rets for k in etf + [k for k in keys if k not in etf]):
        common = rets[etf[0]].index
        for k in etf[1:]:
            common = common.intersection(rets[k].index)
        both_down = None
        for d in common:
            if all(k in rets and d in rets[k].index and rets[k].loc[d] < 0 for k in keys if d in rets[k].index):
                # require every key that has a bar that day is down, and all ETFs have a bar
                if all(d in rets[k].index and rets[k].loc[d] < 0 for k in etf):
                    both_down = d
                    # keep last such day; continue
        last_all_down = None
        days_all_down = []
        for d in common:
            present = [k for k in keys if d in rets[k].index]
            if not present:
                continue
            if all(rets[k].loc[d] < 0 for k in present) and set(etf).issubset(present):
                days_all_down.append(d)
        last_all_down = days_all_down[-1] if days_all_down else None
    else:
        days_all_down, last_all_down = [], None

    # Hypothetical -10% same-mark shock on last marks, no new trading (cannot assume fills).
    shock = -0.10
    shocked = 0.0
    cash = float(result.cash.iloc[-1])
    shocked += cash
    for k in keys:
        px = marks.get(k)
        q = holdings.get(k, 0.0)
        if px is None or px != px:
            continue
        shocked += q * px * (1 + shock)
    shock_dd = 0.0 if nav <= 0 else (nav - shocked) / nav

    max_w = max(weights.values()) if weights else 0.0
    gross = sum(max(0.0, w) for w in weights.values())
    cash_w = 1.0 - gross

    return {
        "as_of_nav_date": str(last.date()),
        "nav": nav,
        "weights": {k: round(float(v), 4) for k, v in weights.items()},
        "cash_weight": round(float(cash_w), 4),
        "gross_exposure": round(float(gross), 4),
        "leverage": 0.0,
        "leverage_note": "本系统禁止借钱和做空，毛敞口上限 100%，杠杆按 0 记录。",
        "concentration": {
            "max_name_weight": round(float(max_w), 4),
            "cap_example": cfg.max_weight,
            "breach": bool(max_w - cfg.max_weight > 1e-6),
        },
        "liquidity_note": (
            "日线回测不能衡量流动性、冲击成本和能否成交。成交量只作背景，"
            "不能当成可交易深度。纸上意向单也不保证能按开盘价全部成交。"
        ),
        "simultaneous_declines": {
            "common_session_days_all_present_down": len(days_all_down),
            "last_such_day": None if last_all_down is None else str(pd.Timestamp(last_all_down).date()),
            "note": "统计的是各标的在同一自然日都有 K 线且收盘相对前收盘为负的日子；BTC 与 ETF 日历并不总重叠。",
        },
        "hypothetical_same_mark_shock": {
            "shock": shock,
            "nav_after": round(shocked, 2),
            "loss_fraction_vs_last_nav": f"{shock_dd:.2%}",
            "note": "把最近标记价格同时下调 10%，不假设能提前卖出。这是示意，不是预测。",
        },
        "max_drawdown_in_backtest": f"{float(result.drawdown.max()):.2%}" if not result.drawdown.empty else "n/a",
    }
