from __future__ import annotations
from trading_lab.config import LabConfig
from trading_lab.data import PricePanel
from trading_lab.engine import run_backtest
from trading_lab.metrics import summarize


def buy_and_hold(panel: PricePanel,cfg: LabConfig,*,start:str,end:str|None=None,
                 capital:float|None=None,matched_risk_budget:bool=False):
    """Same UTC event clock/fill costs as strategy; fixed first-open allocations."""
    result=run_backtest(panel,cfg,capital=capital,trading_start=start,baseline=True,
                        matched_risk_budget=matched_risk_budget).slice_dates(start,end)
    perf=summarize(result.nav,result.trades,label=f'买住 {start} 起',notes=result.notes)
    return result.nav,result.trades,perf
