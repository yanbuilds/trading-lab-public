from dataclasses import replace
import pandas as pd
from helpers import make_cfg, panel_from_frames
from trading_lab.engine import run_backtest, market_events
from trading_lab.baselines import buy_and_hold


def bars(dates, closes):
    s=pd.Series(closes,index=pd.DatetimeIndex(dates),dtype=float)
    return pd.DataFrame({'open':s.shift(1).fillna(s.iloc[0]),'high':s+20,'low':s-20,'close':s,'volume':10000})


def cash_case(tmp_path):
    cfg=make_cfg(tmp_path,sma_window=2,cash_floor=0,max_weight=1,max_drawdown_halt=1)
    cfg=replace(cfg,instruments={k:replace(cfg.instruments[k],fee=0,slippage=0) for k in ['BTC-USD','QQQ']})
    q=bars(['2020-01-02','2020-01-03','2020-01-06','2020-01-07','2020-01-08','2020-01-09','2020-01-10'],[100,110,120,80,80,80,80])
    b=bars(pd.date_range('2020-01-02','2020-01-11'),[100,100,100,100,100,200,210,220,230,240])
    return cfg,panel_from_frames({'QQQ':q,'BTC-USD':b})


def test_earlier_btc_cannot_spend_later_etf_sale_or_unsettled_cash(tmp_path):
    cfg,p=cash_case(tmp_path)
    r=run_backtest(p,cfg,capital=10000)
    sales=r.trades[(r.trades.symbol=='QQQ') & (r.trades.side=='sell')]
    assert len(sales)==1
    sale=sales.iloc[0]
    assert sale.timestamp_utc=='2020-01-08T14:30:00+00:00'
    assert sale.cash_available_at_utc=='2020-01-09T14:30:00+00:00'
    buys=r.trades[(r.trades.symbol=='BTC-USD') & (r.trades.side=='buy')]
    assert not buys.empty
    assert (pd.to_datetime(buys.timestamp_utc)>=pd.Timestamp('2020-01-10',tz='UTC')).all()
    assert (r.events.settled_cash>=-1e-7).all()
    pending=r.events[(r.events.timestamp_utc>=sale.timestamp_utc)&(r.events.timestamp_utc<sale.cash_available_at_utc)]
    assert (pending.unsettled_cash>0).all()
    assert (pending.settled_cash.abs()<1e-6).all()


def test_friday_etf_sale_waits_through_us_holiday_before_btc_can_spend(tmp_path):
    cfg=make_cfg(tmp_path,sma_window=2,cash_floor=0,max_weight=1,max_drawdown_halt=1)
    cfg=replace(cfg,instruments={k:replace(cfg.instruments[k],fee=0,slippage=0) for k in ['BTC-USD','QQQ']})
    q=bars(['2020-08-31','2020-09-01','2020-09-02','2020-09-03','2020-09-04','2020-09-08','2020-09-09'],
           [100,110,120,80,80,80,80])
    b=bars(pd.date_range('2020-08-31','2020-09-10'),[100,100,100,100,200,210,220,230,240,250,260])
    result=run_backtest(panel_from_frames({'QQQ':q,'BTC-USD':b}),cfg,capital=10000)
    sale=result.trades[(result.trades.symbol=='QQQ') & (result.trades.side=='sell')].iloc[0]
    assert sale.timestamp_utc=='2020-09-04T13:30:00+00:00'
    assert sale.cash_available_at_utc=='2020-09-08T13:30:00+00:00'
    buys=result.trades[(result.trades.symbol=='BTC-USD') & (result.trades.side=='buy')]
    assert not buys.empty
    assert (pd.to_datetime(buys.timestamp_utc)>=pd.Timestamp('2020-09-09',tz='UTC')).all()
    pending=result.events[(result.events.timestamp_utc>=sale.timestamp_utc)&
                          (result.events.timestamp_utc<sale.cash_available_at_utc)]
    assert (pending.unsettled_cash>0).all() and (pending.settled_cash>=-1e-7).all()


def test_monotonic_events_and_signal_available_before_fills(tmp_path):
    cfg,p=cash_case(tmp_path); r=run_backtest(p,cfg)
    assert pd.to_datetime(r.events.timestamp_utc).is_monotonic_increasing
    for _,t in r.trades.iterrows():
        assert pd.Timestamp(t.signal_at_utc)<=pd.Timestamp(t.timestamp_utc)
    assert (r.trades.qty>=0).all()
    assert (r.holdings>=-1e-9).all().all()


def test_calendar_dst_holiday_and_weekend(tmp_path):
    cfg=make_cfg(tmp_path)
    days=['2019-03-08','2019-03-09','2019-03-11','2019-07-04','2019-07-05']
    p=panel_from_frames({'QQQ':bars(days,[100]*5)})
    opens=[e for e in market_events(p,cfg) if e[1]=='open']
    assert [str(e[0]) for e in opens]==['2019-03-08 14:30:00+00:00','2019-03-11 13:30:00+00:00','2019-07-05 13:30:00+00:00']


def test_baseline_uses_identical_event_timestamps_and_costs(tmp_path):
    cfg,p=cash_case(tmp_path)
    cfg=replace(cfg,instruments={k:replace(v,fee=.01,slippage=.01) for k,v in cfg.instruments.items()})
    nav,t,_=buy_and_hold(p,cfg,start='2020-01-02',matched_risk_budget=True)
    assert t.timestamp_utc.isin([e[0].isoformat() for e in market_events(p,cfg) if e[1]=='open']).all()
    assert ((t.fill-t.open*1.01).abs()<1e-6).all()
    assert ((t.fee-t.notional*.01).abs()<1e-6).all()
    assert len(t)==2
    assert not nav.empty


def test_simultaneous_buy_allocations_independent_of_dict_order(tmp_path):
    cfg=make_cfg(tmp_path,sma_window=2,cash_floor=0,max_weight=1)
    cfg=replace(cfg,instruments={k:replace(cfg.instruments[k],fee=0,slippage=0) for k in ['GLD','QQQ']})
    df=bars(['2020-01-02','2020-01-03','2020-01-06'],[100,110,120])
    p=panel_from_frames({'GLD':df,'QQQ':df})
    a=run_backtest(p,cfg,capital=10000)
    b=run_backtest(p,replace(cfg,instruments=dict(reversed(list(cfg.instruments.items())))),capital=10000)
    pd.testing.assert_frame_equal(a.trades,b.trades)
    first=a.trades[a.trades.side=='buy'].groupby('symbol').first()
    assert abs(first.loc['GLD','notional']-first.loc['QQQ','notional'])<1e-6


def test_later_etf_close_cannot_change_earlier_btc_fill(tmp_path):
    cfg,p=cash_case(tmp_path)
    original=run_backtest(p,cfg)
    p.frames['QQQ'].loc[pd.Timestamp('2020-01-07'),'close']=8000
    modified=run_backtest(p,cfg)
    boundary=pd.Timestamp('2020-01-07T14:30:00Z')
    def earlier(r):
        return r.trades[pd.to_datetime(r.trades.timestamp_utc)<boundary].reset_index(drop=True)
    pd.testing.assert_frame_equal(earlier(original),earlier(modified))


def test_us_early_close_is_actual_close_not_assumed_1600(tmp_path):
    cfg=make_cfg(tmp_path)
    p=panel_from_frames({'QQQ':bars(['2020-11-27'],[100])})
    closes=[e for e in market_events(p,cfg) if e[1]=='close']
    assert closes[0][0]==pd.Timestamp('2020-11-27T18:00:00Z')
