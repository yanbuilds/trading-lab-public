from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from trading_lab.data import stale_report
from trading_lab.paper import latest_signal_replay
from trading_lab.timeutil import UTC
from helpers import make_cfg, ohlc_from_close, panel_from_frames


def test_crypto_stale_when_older_than_threshold(tmp_path):
    now = datetime(2020, 2, 1, 15, 0, tzinfo=UTC)
    # last complete bar 2020-01-28 ends 2020-01-29 00:00 UTC → age >> 36h
    idx = pd.date_range("2020-01-10", "2020-01-28", freq="D")
    df = ohlc_from_close(idx, pd.Series(10000.0, index=idx))
    cfg = make_cfg(tmp_path, btc_max_age_hours=36)
    inst = {k: cfg.instruments[k] for k in ["BTC-USD"]}
    cfg = make_cfg(tmp_path, instruments=inst, btc_max_age_hours=36)
    panel = panel_from_frames({"BTC-USD": df})
    # fix last_complete
    panel.metas["BTC-USD"].last_complete_bar = idx[-1].date()
    panel.metas["BTC-USD"].last_bar = idx[-1].date()
    report = stale_report(panel, cfg, now=now)
    assert report["BTC-USD"]["stale"] is True


def test_crypto_fresh_just_after_bar_close(tmp_path):
    # bar date 2020-01-30 completes at 2020-01-31 00:00 UTC
    now = datetime(2020, 1, 31, 1, 0, tzinfo=UTC)
    idx = pd.date_range("2020-01-10", "2020-01-30", freq="D")
    df = ohlc_from_close(idx, pd.Series(10000.0, index=idx))
    inst_cfg = make_cfg(tmp_path)
    inst = {k: inst_cfg.instruments[k] for k in ["BTC-USD"]}
    cfg = make_cfg(tmp_path, instruments=inst, btc_max_age_hours=36)
    panel = panel_from_frames({"BTC-USD": df})
    panel.metas["BTC-USD"].last_complete_bar = idx[-1].date()
    panel.metas["BTC-USD"].last_bar = idx[-1].date()
    report = stale_report(panel, cfg, now=now)
    assert report["BTC-USD"]["stale"] is False


def test_stale_blocks_new_paper_signals(tmp_path):
    now = datetime(2020, 2, 1, 15, 0, tzinfo=UTC)
    idx = pd.date_range("2020-01-10", "2020-01-28", freq="D")
    df = ohlc_from_close(idx, pd.Series(10000.0, index=idx))
    inst_cfg = make_cfg(tmp_path)
    inst = {k: inst_cfg.instruments[k] for k in ["BTC-USD"]}
    cfg = make_cfg(tmp_path, instruments=inst, btc_max_age_hours=36)
    panel = panel_from_frames({"BTC-USD": df})
    panel.metas["BTC-USD"].last_complete_bar = idx[-1].date()
    panel.metas["BTC-USD"].last_bar = idx[-1].date()
    snap = latest_signal_replay(panel, cfg, now=now)
    assert snap["blocked_new_signals"] is True
    assert snap["intended_orders"] == []
    assert snap["status"] == "blocked_stale"
