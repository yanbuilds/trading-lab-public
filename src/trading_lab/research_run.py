"""Reproducible, development-only hold-out comparison; no forecasting claims."""
from __future__ import annotations
import json
import pandas as pd
from trading_lab.config import load_config
from trading_lab.data import load_panel_from_cache
from trading_lab.engine import run_backtest
from trading_lab.baselines import buy_and_hold
from trading_lab.metrics import max_drawdown, turnover
from trading_lab.directions import direction_report
from trading_lab.paths import ARTIFACTS
from trading_lab.timeutil import now_utc, format_shanghai


def metrics(nav, trades, capital):
    first=nav.index[0]; last=nav.index[-1]
    # Add initial capital so opening costs are not silently removed by normalization.
    extended=pd.concat([pd.Series([capital],index=[first-pd.Timedelta(days=1)]),nav])
    total=float(nav.iloc[-1]/capital-1)
    years=max((last-first).days/365.25,1/365.25)
    return dict(start=str(first.date()),end=str(last.date()),initial_capital=capital,
                final_nav=float(nav.iloc[-1]),total_return=total,
                cagr=(1+total)**(1/years)-1,max_drawdown=max_drawdown(extended),
                turnover=turnover(trades,nav),trades=len(trades),
                fees=float(trades.fee.sum()) if not trades.empty else 0)


def main():
    cfg=load_config(); panel=load_panel_from_cache(cfg)
    if len(panel.frames)!=len(cfg.instruments): raise RuntimeError('Missing real market data')
    result=run_backtest(panel,cfg,trading_start=cfg.test_start)
    strategy=result.slice_dates(cfg.test_start)
    # Both start with same capital/time; matched baseline has same initial risk budget, not dynamically equal exposure.
    nav,trades,_=buy_and_hold(panel,cfg,start=cfg.test_start,matched_risk_budget=True)
    result.events.to_csv(ARTIFACTS/'utc_event_ledger.csv',index=False)
    result.trades.to_csv(ARTIFACTS/'utc_trades.csv',index=False)
    stats=dict(strategy=metrics(strategy.nav,strategy.trades,cfg.example_capital_usd),
               buyhold_80pct_initial=metrics(nav,trades,cfg.example_capital_usd))
    report=dict(as_of_shanghai=format_shanghai(now_utc()),status='UTC_EVENT_CORRECTED_RESEARCH_PROTOTYPE',
                source='yfinance auto_adjust=True, local completed-bar cache',
                coverage={k:m.to_dict() for k,m in panel.metas.items()},results=stats,
                assumptions='Same 100000 USD initial capital and test dates. Buyhold uses same initial cash-floor and per-name cap (80% total, 26.67% each). Later weights drift; strategy dynamic exposure differs. Not an exposure-matched causal treatment comparison.',
                audit='Corrected: chronological UTC open/close events, NYSE DST/holiday calendar, prior completed signals only, ETF sales unavailable until next NYSE open (conservative T+1), same-instant buys pro rata. Remaining: zero-latency crypto boundary, full opening fills and adjusted prices are simplifications; no independent audit, walk-forward or forecast calibration. Daily NAV is last observed event on each UTC date, not fabricated ETF weekend returns.',
                directions=direction_report(panel,cfg))
    ARTIFACTS.mkdir(exist_ok=True)
    (ARTIFACTS/'research_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['status'],results=stats,assumptions=report['assumptions'],audit=report['audit']),ensure_ascii=False,indent=2))

if __name__=='__main__': main()
