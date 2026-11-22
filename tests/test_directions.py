from datetime import datetime
import pandas as pd
from helpers import make_cfg, ohlc_from_close, panel_from_frames
from trading_lab.directions import direction_report
from trading_lab.timeutil import UTC


def test_stale_direction_is_unavailable_not_short(tmp_path):
    cfg=make_cfg(tmp_path)
    dates=pd.date_range('2020-01-01',periods=30)
    data=ohlc_from_close(dates,pd.Series(range(100,130),index=dates))
    panel=panel_from_frames({'BTC-USD':data})
    report=direction_report(panel,cfg,datetime(2026,11,16,tzinfo=UTC))
    assert report['portfolio_new_orders_blocked']
    assert all(r['action']=='unavailable' for r in report['instruments'])
    assert all(r['close'] is None for r in report['instruments'])


def test_bearish_fresh_signal_is_cash_not_short(tmp_path):
    from dataclasses import replace
    cfg=make_cfg(tmp_path,sma_window=5)
    cfg=replace(cfg,instruments={'BTC-USD':cfg.instruments['BTC-USD']})
    dates=pd.date_range('2020-01-01',periods=30)
    data=ohlc_from_close(dates,pd.Series(range(130,100,-1),index=dates))
    r=direction_report(panel_from_frames({'BTC-USD':data}),cfg,datetime(2020,1,31,1,tzinfo=UTC))
    assert r['instruments'][0]['classification']=='bearish'
    assert r['instruments'][0]['action']=='cash'
