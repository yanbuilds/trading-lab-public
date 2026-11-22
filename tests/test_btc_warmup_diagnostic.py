"""Synthetic prices verify sampling and missing-bar rejection, not performance."""
import numpy as np
import pandas as pd
import pytest

from experiments.btc_warmup_diagnostic import compare_window


def test_sampling_and_missing_bar_rejection():
    start = pd.Timestamp("2026-01-01T00:00:00Z")
    end = start + pd.Timedelta(days=22)
    idx = pd.date_range(start - pd.Timedelta(hours=500), end - pd.Timedelta(hours=1), freq="h")
    close = 100 + np.sin(np.arange(len(idx)) / 20)
    frame = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1,
                          "close": close, "volume": 1.}, index=idx)
    result = compare_window(frame, start, end)
    assert result["sample_count"] == 4
    assert 0 <= result["signal_mismatches"] <= 4
    with pytest.raises(ValueError, match="Missing"):
        compare_window(frame.drop(idx[450]), start, end)
