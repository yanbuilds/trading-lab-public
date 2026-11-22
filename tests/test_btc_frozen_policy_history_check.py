"""Synthetic fixture tests; never treat the fixture as market performance."""
import pandas as pd
import pytest

from experiments.btc_frozen_policy_history_check import SYMBOL, evaluate_window
from trading_lab.perps.risk import Spec


def test_fixed_accounts_share_window_and_do_not_claim_funding():
    start = pd.Timestamp("2026-06-01T00:00:00Z")
    end = start + pd.Timedelta(hours=6)
    index = pd.date_range(start - pd.Timedelta(days=26), end - pd.Timedelta(hours=1), freq="h")
    frame = pd.DataFrame({"open": 100., "high": 101., "low": 99., "close": 100., "volume": 1.}, index=index)
    result = evaluate_window(frame, Spec(SYMBOL, .0001, .0001, 5., .0006, 1.), start, end)
    assert result["start_utc"] == start.isoformat() and result["end_exclusive_utc"] == end.isoformat()
    assert set(result["accounts"]) == {"frozen_range_candidate", "ema_baseline"}
    for account in result["accounts"].values():
        assert account["initial"] == 100 and account["funding_net_paid"] is None
        assert account["funding_coverage_checked"] is False
    with pytest.raises(ValueError, match="warmup"):
        evaluate_window(frame.loc[start:], Spec(SYMBOL, .0001, .0001, 5.), start, end)
