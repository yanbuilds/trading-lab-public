from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class Performance:
    label: str
    start: str
    end: str
    start_nav: float
    end_nav: float
    total_return: float
    max_drawdown: float
    trading_days: int
    turnover: float
    trade_count: int
    fees_paid: float
    time_in_market: float | None
    notes: list[str]


def max_drawdown(nav: pd.Series) -> float:
    if nav.empty:
        return 0.0
    peak = nav.cummax()
    dd = (peak - nav) / peak.replace(0, pd.NA)
    return float(dd.max() if dd.notna().any() else 0.0)


def total_return(nav: pd.Series) -> float:
    if nav.empty or nav.iloc[0] == 0:
        return 0.0
    return float(nav.iloc[-1] / nav.iloc[0] - 1.0)


def turnover(trades: pd.DataFrame, nav: pd.Series) -> float:
    """成交名义金额之和 / 区间平均净值。没有成交则为 0。"""
    if trades is None or trades.empty or nav.empty:
        return 0.0
    t = trades.copy()
    t["date"] = pd.to_datetime(t["date"])
    t = t[(t["date"] >= nav.index[0]) & (t["date"] <= nav.index[-1])]
    notional = float(t["notional"].sum()) if not t.empty else 0.0
    avg_nav = float(nav.mean())
    if avg_nav <= 0:
        return 0.0
    return notional / avg_nav


def summarize(
    nav: pd.Series,
    trades: pd.DataFrame,
    *,
    label: str,
    holdings: pd.DataFrame | None = None,
    notes: list[str] | None = None,
) -> Performance:
    if nav.empty:
        return Performance(label, "", "", 0, 0, 0, 0, 0, 0, 0, 0, None, ["区间没有净值。"])
    t = trades.copy() if trades is not None and not trades.empty else pd.DataFrame()
    if not t.empty:
        t["date"] = pd.to_datetime(t["date"])
        t = t[(t["date"] >= nav.index[0]) & (t["date"] <= nav.index[-1])]
        trade_count = int(len(t))
        fees = float(t["fee"].sum()) if "fee" in t.columns else 0.0
    else:
        trade_count = 0
        fees = 0.0
    time_in = None
    if holdings is not None and not holdings.empty:
        h = holdings.reindex(nav.index).fillna(0)
        invested = (h.fillna(0).abs().sum(axis=1) > 1e-12)
        time_in = float(invested.mean()) if len(invested) else 0.0
    return Performance(
        label=label,
        start=str(nav.index[0].date()),
        end=str(nav.index[-1].date()),
        start_nav=float(nav.iloc[0]),
        end_nav=float(nav.iloc[-1]),
        total_return=total_return(nav),
        max_drawdown=max_drawdown(nav),
        trading_days=int(len(nav)),
        turnover=turnover(trades, nav),
        trade_count=trade_count,
        fees_paid=fees,
        time_in_market=time_in,
        notes=list(notes or []),
    )


def performance_to_dict(p: Performance) -> dict:
    return {
        "区间": p.label,
        "起始日": p.start,
        "结束日": p.end,
        "期初净值": round(p.start_nav, 2),
        "期末净值": round(p.end_nav, 2),
        "净收益（已扣费用/滑点）": f"{p.total_return:.2%}",
        "最大回撤": f"{p.max_drawdown:.2%}",
        "净值点数": p.trading_days,
        "换手（名义/平均净值）": f"{p.turnover:.2f}",
        "成交笔数": p.trade_count,
        "费用合计": round(p.fees_paid, 2),
        "持仓时间占比": "" if p.time_in_market is None else f"{p.time_in_market:.2%}",
    }
