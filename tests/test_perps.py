"""Synthetic fixtures only: independent of provider prices/credentials."""
import json
from pathlib import Path
import pandas as pd
import pytest
from trading_lab.perps.risk import Spec,floor_step,size_order,exit_pnl,stop_fill,assert_simulation

S=Spec('TEST/USDT:USDT',.001,.001,5)
@pytest.mark.parametrize('side,exit,expected',[(1,110,10),(1,90,-10),(-1,90,10),(-1,110,-10)])
def test_long_short_pnl(side,exit,expected):
    assert exit_pnl(side,1,100,exit,S)==expected
@pytest.mark.parametrize('side,op,lo,hi,stop,expected',[(1,100,94,103,95,95),(1,90,88,94,95,90),(-1,100,98,106,105,105),(-1,110,108,112,105,110)])
def test_stop_and_adverse_gap(side,op,lo,hi,stop,expected):
    assert stop_fill(side,stop,op,lo,hi)==expected

def test_sizing_risk_independent_from_leverage():
    a,_=size_order(30,30,100,2,S,leverage=1)
    b,_=size_order(30,30,100,2,S,leverage=2)
    assert a==b
    assert a*(2+2*100*(.0006+.0005))<=.30
    assert floor_step(.000299999,.0001)==.0002


def test_short_stop_budget_includes_higher_exit_notional_and_slippage():
    spec = Spec('TEST', .0001, .0001, .1, taker=.01, contract_size=1.)
    price, distance, slip = 100., 40., .01
    qty, reason = size_order(100., 100., price, distance, spec, leverage=1., slippage=slip)
    assert reason == 'ok'
    short_entry = price * (1 - slip)
    short_exit = (price + distance) * (1 + slip)
    actual_loss = qty * (short_exit - short_entry + spec.taker * (short_entry + short_exit))
    assert actual_loss <= 1.0 + 1e-12
    old_qty = floor_step(1 / (distance + 2 * price * (spec.taker + slip)), spec.qty_step)
    assert old_qty * (short_exit - short_entry + spec.taker * (short_entry + short_exit)) > 1.0


def test_sizing_rejects_invalid_assumptions():
    assert size_order(30,30,100,2,S,slippage=float('nan'))[1]=='invalid_input'
    assert size_order(30,30,100,2,S,slippage=-.01)[1]=='risk_bounds'
    assert size_order(30,30,100,2,Spec('bad', .001, float('nan'), 5))[1]=='invalid_input'
    with pytest.raises(ValueError):
        floor_step(1, float('nan'))

def test_min_order_not_rounded_up():
    q,r=size_order(30,30,4000,100,Spec('XAU',.01,.01,5))
    assert q==0 and r=='minimum_order_exceeds_budget'

def test_cash_floor_and_leverage_bounds():
    assert size_order(30,5,100,2,S)[0]==0
    assert size_order(30,30,100,2,S,leverage=10)[1]=='risk_bounds'
    assert size_order(30,30,100,60,S)[1]=='stop_outside_stress_margin'

def test_simulation_guard():
    cfg={'dry_run':True,'margin_mode':'isolated','max_open_trades':1,'exchange':{}}
    assert_simulation(cfg)
    for bad in [dict(cfg,dry_run=False),dict(cfg,exchange={'key':'fake'}),dict(cfg,max_open_trades=2),dict(cfg,margin_mode='cross')]:
        with pytest.raises(ValueError):assert_simulation(bad)

def bars():
    ix=pd.date_range('2026-01-01',periods=65,freq='h',tz='UTC')
    return pd.DataFrame({'open':100.,'high':100.1,'low':99.9,'close':100.,'volume':1.},index=ix)

def prepared(df,side=1):
    d=df.copy();d['fast']=101. if side==1 else 99.;d['slow']=100.;d['atr']=1.;d['signal']=0
    d.loc[d.index[59],'signal']=side
    return d

@pytest.fixture
def harness(monkeypatch):
    from trading_lab.perps import research as r
    monkeypatch.setattr(r,'indicators',prepared)
    return r

def test_next_open_intrabar_long_and_roundtrip_fee(harness):
    d=bars();d.loc[d.index[59],'close']=100.05
    d.loc[d.index[61],['low','close']]=[96.,97.]
    r=harness.simulate({'A':d},{'A':S},slippage=0.)
    assert r['closed_trades']==1 and r['losses']==1
    assert r['trades'][0]['entry_time']==str(d.index[60])
    assert r['trades'][0]['exit_reason']=='atr_stop'
    assert r['fees']>0 and r['final']<30

def test_short_stop(harness,monkeypatch):
    monkeypatch.setattr(harness,'indicators',lambda df:prepared(df,-1))
    d=bars();d.loc[d.index[61],['high','close']]=[104.,103.]
    r=harness.simulate({'A':d},{'A':S},slippage=0.)
    assert r['closed_trades']==1 and r['losses']==1

def test_global_margin_one_position_no_duplicate(harness):
    d=bars()
    r=harness.simulate({'A':d,'B':d},{'A':S,'B':S})
    assert r['open_positions']==1 and r['final']>0

def test_funding_on_notional_and_short_receipt(harness,monkeypatch):
    d=bars();t=int(d.index[61].timestamp()*1000)
    a=harness.simulate({'A':d},{'A':S},slippage=0.)
    b=harness.simulate({'A':d},{'A':S},{'A':[{'timestamp':t,'rate':.001}]},slippage=0.)
    assert a['final']-b['final']==pytest.approx(b['funding_net_paid'])
    assert b['funding_net_paid']>0
    monkeypatch.setattr(harness,'indicators',lambda df:prepared(df,-1))
    c=harness.simulate({'A':d},{'A':S},{'A':[{'timestamp':t,'rate':.001}]},slippage=0.)
    assert c['funding_net_paid']<0

def test_missing_candle_halts(harness):
    d=bars().drop(bars().index[62])
    with pytest.raises(ValueError,match='Missing'):harness.simulate({'A':d},{'A':S})

def test_future_candle_cannot_change_entry(harness):
    d=bars();a=harness.simulate({'A':d},{'A':S})
    d.loc[d.index[63],['high','low','close']]=[110.,90.,100.]
    b=harness.simulate({'A':d},{'A':S})
    assert b['trades'][0]['entry_time']==str(d.index[60])
    assert a['fees']<b['fees']

def test_native_live_guard_direct():
    # Importing Freqtrade is optional in the original ETF environment.
    pytest.importorskip('freqtrade.strategy')
    import importlib.util
    path=Path(__file__).parents[1]/'freqtrade/strategies/BitgetBaseline.py'
    sp=importlib.util.spec_from_file_location('bitget_strategy',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    cfg={'dry_run':False,'margin_mode':'isolated','max_open_trades':1}
    with pytest.raises(ValueError,match='LIVE DISABLED'):m.BitgetBaseline(cfg)

def test_gap_cannot_hide_negative_cash(harness,monkeypatch):
    monkeypatch.setattr(harness,'indicators',lambda df:prepared(df,-1))
    d=bars();d.loc[d.index[61],['open','high','low','close']]=[10000.,10001.,9999.,10000.]
    with pytest.raises(ValueError,match='Insolvent'):
        harness.simulate({'A':d},{'A':S},slippage=0.)

def test_funding_insolvency_rejected_not_clamped(harness):
    d=bars();t=int(d.index[61].timestamp()*1000)
    with pytest.raises(ValueError,match='Funding insolvency'):
        harness.simulate({'A':d},{'A':S},{'A':[{'timestamp':t,'rate':10.}]},slippage=0.)

def test_native_risk_callback_errors_fail_closed():
    pytest.importorskip('freqtrade.strategy')
    import importlib.util
    path=Path(__file__).parents[1]/'freqtrade/strategies/BitgetBaseline.py'
    sp=importlib.util.spec_from_file_location('safe_strategy',path);m=importlib.util.module_from_spec(sp);sp.loader.exec_module(m)
    obj=m.BitgetBaseline({'dry_run':True,'margin_mode':'isolated','max_open_trades':1})
    assert obj.custom_stake_amount('A',None,100,30,1,30,2,None,'long')==0.
    assert obj.confirm_trade_entry('A','market',1,100,'GTC',None,None,'long') is False


def test_native_signals_do_not_change_when_future_candles_change():
    pytest.importorskip('freqtrade.strategy')
    import importlib.util
    import numpy as np
    path=Path(__file__).parents[1]/'freqtrade/strategies/BitgetBaseline.py'
    sp=importlib.util.spec_from_file_location('native_prefix_test',path)
    module=importlib.util.module_from_spec(sp); sp.loader.exec_module(module)
    obj=module.BitgetBaseline({'dry_run':True,'margin_mode':'isolated','max_open_trades':1})
    n=160
    close=100.+np.sin(np.arange(n)/8.)*8
    frame=pd.DataFrame({'date':pd.date_range('2026-01-01',periods=n,freq='h',tz='UTC'),
                        'open':close, 'high':close+1, 'low':close-1, 'close':close, 'volume':1.})
    cutoff=120
    changed=frame.copy()
    changed.loc[cutoff:,['open','high','low','close']]=[500.,501.,499.,500.]
    def analyze(d):
        d=obj.populate_indicators(d.copy(),{})
        d=obj.populate_entry_trend(d,{})
        return obj.populate_exit_trend(d,{})
    full, modified, prefix=analyze(frame),analyze(changed),analyze(frame.iloc[:cutoff])
    assert full['enter_long'].iloc[:cutoff].fillna(0).sum() + full['enter_short'].iloc[:cutoff].fillna(0).sum() > 0
    for column in ('ema20','ema50','atr','enter_long','enter_short','exit_long','exit_short'):
        pd.testing.assert_series_equal(full[column].iloc[:cutoff], modified[column].iloc[:cutoff])
        pd.testing.assert_series_equal(full[column].iloc[:cutoff], prefix[column])


def test_native_confirmation_uses_closed_atr_contract_size_and_worst_stop_cost():
    pytest.importorskip('freqtrade.strategy')
    import importlib.util
    from types import SimpleNamespace
    from trading_lab.perps.risk import worst_stop_loss_per_contract
    path=Path(__file__).parents[1]/'freqtrade/strategies/BitgetBaseline.py'
    sp=importlib.util.spec_from_file_location('native_cost_test',path)
    module=importlib.util.module_from_spec(sp); sp.loader.exec_module(module)
    obj=module.BitgetBaseline({'dry_run':True,'margin_mode':'isolated','max_open_trades':1})
    now=pd.Timestamp('2026-11-16T08:00:00Z')
    rows=pd.DataFrame({'date':[now-pd.Timedelta(hours=1), now], 'atr':[20., .01]})
    obj.dp=SimpleNamespace(get_analyzed_dataframe=lambda *args:(rows,None),
                           market=lambda pair:{'contractSize':2.})
    obj.wallets=SimpleNamespace(get_total_stake_amount=lambda:100.)
    spec=Spec('A',1.,1.,0.,taker=.0006,contract_size=2.)
    unit_loss=worst_stop_loss_per_contract(100.,40.,spec,slippage=.0005)
    args=('A','market')
    def confirm(amount):
        return obj.confirm_trade_entry(*args,amount,100.,'GTC',now,None,'short')
    assert confirm(.9/unit_loss)
    assert not confirm(1.1/unit_loss)
    assert not confirm(float('nan'))
    assert not confirm(-1.)
    rows['date'] = rows['date'] - pd.Timedelta(hours=5)
    assert not confirm(.9/unit_loss)


def test_native_start_guards_on_real_freqtrade_class():
    pytest.importorskip('freqtrade.strategy')
    import importlib.util
    path=Path(__file__).parents[1]/'freqtrade/strategies/BitgetBaseline.py'
    sp=importlib.util.spec_from_file_location('native_start_test',path)
    module=importlib.util.module_from_spec(sp); sp.loader.exec_module(module)
    cfg={'dry_run':True,'margin_mode':'isolated','max_open_trades':1,'runmode':'backtest'}
    obj=module.BitgetBaseline(cfg)
    obj.bot_start()
    obj.config['runmode']='hyperopt'
    with pytest.raises(ValueError,match='Only dry_run/backtest'):
        obj.bot_start()
    obj.config['runmode']='dry_run'
    obj.config['dry_run']=False
    with pytest.raises(ValueError,match='LIVE DISABLED'):
        obj.bot_start()
