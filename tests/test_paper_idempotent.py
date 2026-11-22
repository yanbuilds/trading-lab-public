from __future__ import annotations

from datetime import datetime

from trading_lab.paper import latest_signal_replay, write_snapshot
from trading_lab.timeutil import UTC
from helpers import make_cfg, ohlc_from_close, panel_from_frames
import pandas as pd


def test_snapshot_idempotent(tmp_path):
    idx = pd.bdate_range("2020-01-02", periods=20)
    df = ohlc_from_close(idx, pd.Series(100 + pd.RangeIndex(20) * 0.5, index=idx, dtype=float))
    inst_cfg = make_cfg(tmp_path)
    inst = {k: inst_cfg.instruments[k] for k in ["QQQ"]}
    cfg = make_cfg(tmp_path, instruments=inst, etf_max_completed_sessions_lag=500)
    panel = panel_from_frames({"QQQ": df})
    now = datetime(2020, 2, 3, tzinfo=UTC)
    snap1 = latest_signal_replay(panel, cfg, now=now)
    path = tmp_path / "paper_snapshot.json"
    p1, unchanged1 = write_snapshot(snap1, path)
    assert p1 == path and unchanged1 is False
    snap2 = latest_signal_replay(panel, cfg, now=now)
    _, unchanged2 = write_snapshot(snap2, path)
    assert snap1["fingerprint"] == snap2["fingerprint"]
    assert unchanged2 is True
