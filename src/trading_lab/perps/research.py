"""Offline CONTRACT research harness, NOT Freqtrade's matching engine or an exchange demo.

Public CCXT market access only. One wallet, one position, causal 1h EMA20/50 crosses,
ATR14 stop 2*ATR fixed at signal close, next-open fills. No fitted parameters.
"""
from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
from pathlib import Path
import json
import math
import pandas as pd
from .risk import Spec,size_order,exit_pnl,stop_fill,assert_simulation

ROOT=Path(__file__).resolve().parents[3]
CACHE=ROOT/'data'/'perps'
ART=ROOT/'artifacts'
PAIRS=['BTC/USDT:USDT','XAU/USDT:USDT','NDX100/USDT:USDT']

def indicators(df):
    # Native Freqtrade uses TA-Lib; keep one authoritative indicator implementation.
    import talib
    d=df.copy()
    d['fast']=talib.EMA(d.close.to_numpy(dtype=float),timeperiod=20)
    d['slow']=talib.EMA(d.close.to_numpy(dtype=float),timeperiod=50)
    d['atr']=talib.ATR(d.high.to_numpy(dtype=float),d.low.to_numpy(dtype=float),d.close.to_numpy(dtype=float),timeperiod=14)
    d['signal']=0
    d.loc[(d.fast>d.slow)&(d.fast.shift(1)<=d.slow.shift(1)),'signal']=1
    d.loc[(d.fast<d.slow)&(d.fast.shift(1)>=d.slow.shift(1)),'signal']=-1
    return d

def validate_bars(df):
    if len(df)<61 or df.index.has_duplicates or not df.index.is_monotonic_increasing:
        raise ValueError('Insufficient/duplicate/unsorted bars')
    if not (df.index.to_series().diff().dropna()==pd.Timedelta(hours=1)).all():
        raise ValueError('Missing 1h bars: do not interpolate or silently trade gaps')
    if not ((df[['open','high','low','close']]>0).all().all() and (df.high>=df[['open','close','low']].max(axis=1)).all()
            and (df.low<=df[['open','close','high']].min(axis=1)).all()):
        raise ValueError('Invalid OHLC')

def fetch(days=60):
    import ccxt
    e=ccxt.bitget({'enableRateLimit':True,'timeout':20000})
    markets=e.load_markets();now=pd.Timestamp.now(tz='UTC');end=now.floor('h')
    start=end-pd.Timedelta(days=days)
    CACHE.mkdir(parents=True,exist_ok=True)
    provenance={'fetched_at_taipei':str(now.tz_convert('Asia/Taipei')),'ccxt':ccxt.__version__,
                'venue':'bitget','kind':'public real 1h perpetual OHLCV; not ETFs, no synthesized prices',
                'symbols':{}}
    for pair in PAIRS:
        m=markets[pair]
        if not(m['active'] and m['swap'] and m['linear']):raise ValueError('Unsupported market '+pair)
        rows=[];cursor=int(start.timestamp()*1000)
        while cursor<int(end.timestamp()*1000):
            page=e.fetch_ohlcv(pair,'1h',since=cursor,limit=200)
            if not page:break
            rows.extend(page);nxt=max(x[0] for x in page)  # overlap boundary; venue since is exclusive
            if nxt<=cursor:break
            cursor=nxt
        df=pd.DataFrame(rows,columns=['timestamp','open','high','low','close','volume']).drop_duplicates('timestamp').sort_values('timestamp')
        df.index=pd.to_datetime(df.pop('timestamp'),unit='ms',utc=True)
        df=df.loc[(df.index>=start)&(df.index<end)]
        raw_count=len(df)
        gaps=df.index.to_series().diff()>pd.Timedelta(hours=1)
        gap_list=[str(t) for t in df.index[gaps]]
        if gap_list:
            df=df.loc[gaps[gaps].index[-1]:]  # explicit contiguous suffix, never interpolate
        validate_bars(df)
        sym=m['id'];df.to_csv(CACHE/f'{sym}.csv',index_label='date')
        # Native Freqtrade input format, candles only; separate native backtest needs marks/funding.
        native=df.reset_index().rename(columns={df.index.name or 'index':'date'})
        native['date']=pd.to_datetime(native['date'],utc=True)
        native.to_feather(CACHE/f"{pair.replace('/','_').replace(':','_')}-1h-futures.feather")
        funding=[];funding_error=None;cursor=int(df.index[0].timestamp()*1000)
        try:
            for page_no in range(1,17):
                page=e.fetch_funding_rate_history(pair,limit=100,params={"pageNo":page_no})
                page=[x for x in page if x.get('timestamp') is not None]
                if not page:break
                funding.extend({'timestamp':x['timestamp'],'rate':float(x['fundingRate'])} for x in page)
                if min(x['timestamp'] for x in page)<=int(df.index[0].timestamp()*1000):break
        except Exception as exc:
            funding_error=f'{type(exc).__name__}: {exc}'
        fd=pd.DataFrame(funding,columns=['timestamp','rate']).drop_duplicates('timestamp').sort_values('timestamp')
        fd=fd.loc[(fd.timestamp>=int(df.index[0].timestamp()*1000))&(fd.timestamp<int(end.timestamp()*1000))]
        fd.to_csv(CACHE/f'{sym}-funding.csv',index=False)
        spec=Spec(pair,m['limits']['amount']['min'],m['precision']['amount'],m['limits']['cost']['min'],
                  float(m.get('taker') or .0006),m.get('contractSize') or 1.)
        interval=int(m['info'].get('fundInterval',8))
        expected=pd.date_range(df.index[0].ceil(f'{interval}h'),end-pd.Timedelta(milliseconds=1),freq=f'{interval}h')
        available=set(fd.timestamp)
        complete=all(int(x.timestamp()*1000) in available for x in expected)
        provenance['symbols'][pair]=dict(spec=asdict(spec),bars=len(df),first=str(df.index[0]),last=str(df.index[-1]),
            funding_count=len(fd),funding_complete=complete,funding_error=funding_error,funding_interval_hours=interval,raw_bars=raw_count,gap_after_dates=gap_list,discarded_prefix_bars=raw_count-len(df))
    (CACHE/'provenance.json').write_text(json.dumps(provenance,indent=2))
    return provenance

def simulate(frames,specs,funding=None,*,capital=30.,leverage=2.,start=None,slippage=.0005):
    """Stop-based stress research, NOT exact venue mark/liquidation simulation.

    Funding assessed on contemporaneous contract open (mark proxy explicitly), actual
    historic rate. Same-hour entries precede intra-hour stop exits: no future cash use.
    """
    if not 1<=leverage<=3:raise ValueError('Example simulation leverage restricted to 1..3')
    for d in frames.values():validate_bars(d)
    data={k:indicators(v) for k,v in frames.items()}
    common=sorted(set.intersection(*(set(d.index) for d in data.values())))
    if not common:raise ValueError('No common timestamps')
    start=pd.Timestamp(start) if start is not None else common[60]
    cash=float(capital);pos=None;closed=[];nav=[];reject=Counter();fees=0.;fundcost=0.;kill=False
    day=None;day_start=cash;last_event=None;seen=set();stress_breaches=0
    fmaps={k:{pd.Timestamp(int(r['timestamp']),unit='ms',tz='UTC'):float(r['rate']) for r in rs}
           for k,rs in (funding or {}).items()}
    def close(px,ts,reason):
        nonlocal cash,pos,fees,kill,stress_breaches
        spec=specs[pos['symbol']];qty=pos['qty'];side=pos['side']
        px*=1-side*slippage
        exit_fee=qty*spec.contract_size*px*spec.taker
        gross=exit_pnl(side,qty,pos['entry'],px,spec)
        if -gross+exit_fee>pos['margin']:stress_breaches+=1
        cash+=gross-exit_fee;fees+=exit_fee
        if cash<0:raise ValueError('Insolvent stress path: exact isolated liquidation model required; do not publish returns')
        closed.append(dict(symbol=pos['symbol'],side=side,entry_time=str(pos['time']),exit_time=str(ts),
             net_pnl=gross-pos['entry_fee']-exit_fee-pos['funding'],exit_reason=reason))
        pos=None
    for ts in common:
        if ts<start:continue
        if last_event is not None and ts-last_event!=pd.Timedelta(hours=1):raise ValueError('Stale shared timeline')
        last_event=ts
        if ts.date()!=day:
            day=ts.date();day_start=cash+(exit_pnl(pos['side'],pos['qty'],pos['entry'],data[pos['symbol']].loc[ts,'open'],specs[pos['symbol']]) if pos else 0.)
        rows={k:d.loc[ts] for k,d in data.items()}
        if pos:
            k=pos['symbol'];rate=fmaps.get(k,{}).get(ts,0.)
            amount=pos['side']*pos['qty']*specs[k].contract_size*rows[k]['open']*rate
            cash-=amount;fundcost+=amount;pos['funding']+=amount
            if cash<0:raise ValueError('Funding insolvency; exact liquidation model required')
            # At open we know only prior completed indicators.
            prev=data[k].loc[ts-pd.Timedelta(hours=1)]
            gap=(pos['side']==1 and rows[k]['open']<=pos['stop']) or (pos['side']==-1 and rows[k]['open']>=pos['stop'])
            if gap:close(rows[k]['open'],ts,'gap_stop')
            elif (pos['side']==1 and prev.fast<prev.slow) or (pos['side']==-1 and prev.fast>prev.slow):
                close(rows[k]['open'],ts,'reverse_signal')
        if cash<=day_start*.97:kill=True
        if cash<=capital*.90:kill=True  # latched total-loss halt, no reset on flat
        if not pos and not kill:
            for k in sorted(data):
                prev=data[k].loc[ts-pd.Timedelta(hours=1)]
                side=int(prev.signal)
                if not side or not math.isfinite(prev.atr):continue
                oid=(k,ts,side)
                if oid in seen:raise ValueError('Duplicate signal/order identifier')
                seen.add(oid)
                px=rows[k]['open']*(1+side*slippage)
                distance=2*float(prev.atr);spec=specs[k]
                qty,reason=size_order(cash,cash,px,distance,spec,leverage=leverage,slippage=slippage)
                if not qty:reject[reason]+=1;continue
                margin=qty*spec.contract_size*px/leverage
                fee=qty*spec.contract_size*px*spec.taker
                if margin+fee>cash:raise AssertionError('Margin double spend')
                cash-=fee;fees+=fee
                pos=dict(symbol=k,side=side,qty=qty,entry=px,stop=px-side*distance,
                         margin=margin,entry_fee=fee,funding=0.,time=ts)
                break
        if pos:
            k=pos['symbol'];r=rows[k]
            px=stop_fill(pos['side'],pos['stop'],r.open,r.low,r.high)
            if px is not None:close(px,ts+pd.Timedelta(hours=1),'atr_stop')
        equity=cash
        if pos:
            k=pos['symbol'];equity+=exit_pnl(pos['side'],pos['qty'],pos['entry'],rows[k]['close'],specs[k])
        if equity<0:raise ValueError('Negative equity stress path')
        nav.append((ts+pd.Timedelta(hours=1),equity))
        if equity<=day_start*.97 or equity<=capital*.9:kill=True
        if kill and pos: # deterministic kill at observed close, not an earlier price
            close(rows[pos['symbol']]['close'],ts+pd.Timedelta(hours=1),'loss_halt')
            nav[-1]=(nav[-1][0],cash)
    eq=pd.Series([capital]+[v for _,v in nav],dtype=float)
    pnls=[x['net_pnl'] for x in closed];wins=[p for p in pnls if p>.01];losses=[p for p in pnls if p<-.01]
    open_pnl=0.
    if pos:
        open_pnl=nav[-1][1]-cash-pos['entry_fee']-pos['funding']
    assert abs(sum(pnls)+open_pnl-(eq.iloc[-1]-capital))<1e-7
    return dict(initial=capital,final=float(eq.iloc[-1]),return_pct=float((eq.iloc[-1]/capital-1)*100),
       max_drawdown_pct=float((1-eq/eq.cummax()).max()*100),closed_trades=len(closed),wins=len(wins),
       losses=len(losses),breakeven=int(sum(abs(p)<=.01 for p in pnls)),winrate_pct=100*len(wins)/len(pnls) if pnls else None,
       fees=fees,funding_net_paid=fundcost,open_positions=int(pos is not None),rejections=dict(reject),
       loss_halt=bool(kill),stress_margin_breaches=stress_breaches,stress_screen_passed=stress_breaches==0,
       start=str(nav[0][0]-pd.Timedelta(hours=1)) if nav else None,end=str(nav[-1][0]) if nav else None,
       trades=closed,nav=[(str(t),v) for t,v in nav])

def backtest():
    p=json.loads((CACHE/'provenance.json').read_text());frames={};specs={};funding={}
    for k,v in p['symbols'].items():
        sym=k.split('/')[0]+'USDT'
        d=pd.read_csv(CACHE/f'{sym}.csv',parse_dates=['date']).set_index('date');d.index=pd.to_datetime(d.index,utc=True)
        frames[k]=d;specs[k]=Spec(**v['spec'])
        funding[k]=pd.read_csv(CACHE/f'{sym}-funding.csv').to_dict('records')
    common=sorted(set.intersection(*(set(d.index) for d in frames.values())))
    if len(common)<=60:raise ValueError('insufficient shared bars for warmup')
    # Provenance flags may be stale; validate actual records before writing any result.
    from .compare import funding_complete_between
    intervals={k:v['funding_interval_hours'] for k,v in p['symbols'].items()}
    coverage=funding_complete_between(funding,intervals,common[60],common[-1])
    if not coverage['complete']:
        raise ValueError('funding missing or invalid under assumed intervals; returns withheld')
    # Fixed 2/3 chronological development / 1/3 later evaluation. No model fitting or tuning.
    split=common[int(len(common)*2/3)]
    spans={'full':(None,None),'development':(None,split),'later_evaluation':(split,None)}
    results={}
    for name,(start,end) in spans.items():
        sub={k:d.loc[d.index<end] if end is not None else d for k,d in frames.items()}
        r=simulate(sub,specs,funding,start=start)
        pd.DataFrame(r.pop('trades')).to_csv(ART/f'perps-{name}-trades.csv',index=False)
        pd.DataFrame(r.pop('nav'),columns=['date','equity']).to_csv(ART/f'perps-{name}-equity.csv',index=False)
        results[name]=r
    complete=coverage['complete']
    report=dict(engine='offline stress research harness, NOT native Freqtrade fills or exchange demo',
      strategy='1h EMA20/50 crosses long/short; next-open; frozen 2*ATR14 stop; no target or optimization',
      assumptions={'virtual_usdt':30,'leverage_example':2,'risk_budget':.01,'max_margin_fraction':.5,
        'cash_floor':.2,'max_positions':1,'slippage_each_side':.0005,'fee':'actual public market taker metadata',
        'funding':'historical rates * contemporaneous open as mark proxy; not exact historical mark',
        'funding_history_complete_under_assumed_intervals':complete,'funding_coverage':coverage,
        'daily_loss_halt':.03,'total_loss_halt':.1,
        'halt_latched':'halt persists until end; no auto topup/reset','liquidation':'NOT exact venue tier/mark/ADL model'},
      provenance=p,results=results,
      limitations=['No independent unseen holdout or forecasting calibration. Later slice is a chronological evaluation, not certification.',
       'Fees/slippage/mark proxy are assumptions; stops cannot guarantee fills, intrabar mark liquidation not modelled.',
       'No authenticated demo/live order has been submitted; local stop simulation only.',
       'Funding coverage assumes current intervals; historical changes and exact settlement marks are unverified.',
       'Any stress_margin_breaches invalidates strategy performance for real leveraged use.'])
    (ART/'perps-report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps({k:v for k,v in report.items() if k!='provenance'},indent=2))

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('action',choices=['fetch','backtest','check-config'])
    ap.add_argument('--days',type=int,default=60);a=ap.parse_args();ART.mkdir(exist_ok=True)
    if a.action=='fetch': print(json.dumps(fetch(a.days),indent=2))
    elif a.action=='backtest':backtest()
    else:
        assert_simulation(json.loads((ROOT/'freqtrade/config.simulation.json').read_text()));print('SIMULATION ONLY: guard passed')
if __name__=='__main__':main()
