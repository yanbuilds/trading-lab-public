"""Conditional trend research, not a calibrated forecast or an order generator."""
from __future__ import annotations
import json
from trading_lab.config import load_config
from trading_lab.data import load_panel_from_cache, stale_report
from trading_lab.signals import sma
from trading_lab.timeutil import now_utc, format_shanghai
from trading_lab.paths import ARTIFACTS


def direction_report(panel, cfg, now=None):
    now = now or now_utc()
    freshness = stale_report(panel, cfg, now)
    rows = []
    for symbol, inst in cfg.instruments.items():
        state = freshness[symbol]
        row = dict(symbol=symbol, proxy=inst.proxy_for,
                   as_of_shanghai=format_shanghai(now),
                   last_completed_bar=state.get('last_complete_bar'),
                   stale=state['stale'], data_note=state['reason'],
                   classification='unavailable', action='unavailable', close=None, sma=None,
                   forecast='无已校准的未来价格预测、概率或价格目标。',
                   invalidation=f'下一根完整收盘跨过 SMA{cfg.sma_window}，重新评估；这是滞后趋势规则，不预测底部。')
        if symbol in panel.frames and not state['stale']:
            close = panel.frames[symbol]['close']
            avg = sma(close, cfg.sma_window).iloc[-1]
            if avg == avg:
                last = float(close.iloc[-1])
                row.update(close=last, sma=float(avg),
                    classification='bullish' if last > avg else 'bearish' if last < avg else 'neutral',
                    action='long' if last > avg else 'cash',
                    conditional_scenario='若完整收盘持续在均线上方则保留多头趋势；若跌回均线下则转现金。反之亦然。无做空策略。')
        rows.append(row)
    return dict(as_of_shanghai=format_shanghai(now), instruments=rows,
                portfolio_new_orders_blocked=any(r['stale'] for r in rows),
                note='单资产趋势分类不等于组合可执行目标，更不是个性化买卖建议。缺失任一数据时组合新意向单暂停；无 LLM 下单。')


def main():
    cfg=load_config(); panel=load_panel_from_cache(cfg)
    report=direction_report(panel,cfg)
    ARTIFACTS.mkdir(exist_ok=True)
    path=ARTIFACTS/'directions.json'
    path.write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
