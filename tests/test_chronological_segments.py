import numpy as np
import pandas as pd
import pytest

from experiments.chronological_segments import audit_segments, windows
from trading_lab.perps.risk import Spec


def test_windows_are_nonoverlapping_and_keep_short_last_segment():
    start = pd.Timestamp("2026-08-19T00:00:00Z")
    end = pd.Timestamp("2026-11-16T06:00:00Z")
    chunks = windows(start, end)
    assert [int((b-a)/pd.Timedelta(hours=1)) for a, b in chunks] == [720, 720, 702]
    assert chunks[0][0] == start and chunks[-1][1] == end
    assert all(a[1] == b[0] for a, b in zip(chunks, chunks[1:]))
    with pytest.raises(ValueError):
        windows(pd.Timestamp("2026-01-01"), pd.Timestamp("2026-02-01"))


def test_first_segment_does_not_use_future_bars_and_missing_funding_rejects():
    symbols = ("BTC/USDT:USDT", "XAU/USDT:USDT", "NDX100/USDT:USDT")
    idx = pd.date_range("2026-01-01", periods=961, freq="h", tz="UTC")
    close = 100 + np.sin(np.arange(len(idx)) / 12) * 3
    bars = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                         "close": close, "volume": 1.0}, index=idx)
    frames = {s: bars.copy() for s in symbols}
    specs = {s: Spec(s, 0.001, 0.001, 5) for s in symbols}
    funding = {s: [{"timestamp": int(t.timestamp()*1000), "rate": 0.0}
                   for t in idx if t.hour % 8 == 0] for s in symbols}
    intervals = {s: 8 for s in symbols}
    start, end = idx[240], idx[-1] + pd.Timedelta(hours=1)
    before = audit_segments(frames, specs, funding, intervals, start, end)
    assert "tsmom_3m_corrected" in before["segments"][0]["strategies"]
    assert before["segments"][0]["strategies"]["tsmom_3m_corrected"]["closed_trades"] == 0  # synthetic data lacks 90-day warmup
    changed = {s: f.copy() for s, f in frames.items()}
    for frame in changed.values():
        frame.loc[frame.index >= start + pd.Timedelta(days=30), ["open", "high", "low", "close"]] *= 2
    after = audit_segments(changed, specs, funding, intervals, start, end)
    assert before["segments"][0] == after["segments"][0]
    funding[symbols[0]] = []
    with pytest.raises(ValueError, match="funding missing"):
        audit_segments(frames, specs, funding, intervals, start, end)
    price_only = audit_segments(frames, specs, funding, intervals, start, end, exclude_funding=True)
    assert price_only["funding_excluded_unknown"] is True
    assert price_only["funding_coverage_under_current_intervals"] is None
    for segment in price_only["segments"]:
        assert "breakout_4h_aligned" in segment["strategies"]
        assert all(strategy["funding_net_paid"] is None for strategy in segment["strategies"].values())
